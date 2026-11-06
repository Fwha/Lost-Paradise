#include "host/save_menu.h"

#include "host/kernel.h"
#include "host/menu_font.h"

#ifdef LPN_HAVE_SDL2
#include <stb_image.h>
#endif

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>

namespace fs = std::filesystem;

namespace host {

bool windowShowsMenus(); // window.cpp: a window is open and no scripted presses drive the game

namespace {

std::mutex gM;
std::atomic<bool> gOpen{false};
SaveMenuKind gKind = SaveMenuKind::Save;
std::string gChoice;
bool gChoseNew = false; // the "EMPTY" new-save row was picked

struct Row {
    SaveSlot slot;
    bool used = false;
    bool isNew = false; // the game's "new save" item, below the listed saves
    time_t mtime = 0;     // newest file's time, 0 when empty
    uint64_t bytes = 0;   // the save's size on disk
    // What the game set for the system's save list (sce_sys/param.bin, see hle_services.cpp):
    // the game's name, the chapter and place, and more detail. Empty for saves made before the
    // port kept them.
    std::string title, subTitle, detail;
    std::string icon; // sce_sys/icon0.png, "" when the game gave none
};
std::vector<Row> gRows;

// List of slots, then the actions for the highlighted one, then a yes/no for the action.
enum class Screen { List, Actions, Confirm };
enum class Action { Save, Overwrite, Load, Delete, Cancel };
Screen gScreen = Screen::List;
int gCursor = 0, gActionCursor = 0, gConfirmCursor = 1; // confirm starts on "No"
std::vector<Action> gActions;
Action gPending = Action::Cancel;
std::string gMessage; // shown under the list after a delete

std::string cString(const char* p, size_t cap) { return std::string(p, strnlen(p, cap)); }

void refreshRow(Row& r) {
    r.used = false;
    if (r.isNew) return;
    r.mtime = 0;
    r.bytes = 0;
    r.title.clear(), r.subTitle.clear(), r.detail.clear(), r.icon.clear();
    std::error_code ec;
    const fs::path sys = fs::path(r.slot.path) / "sce_sys";
    for (fs::recursive_directory_iterator it(r.slot.path, ec), end; !ec && it != end; it.increment(ec)) {
        struct stat st;
        if (!it->is_regular_file(ec) || ::stat(it->path().c_str(), &st) != 0) continue;
        r.bytes += uint64_t(st.st_size);
        if (it->path().parent_path() == sys) continue; // the list's own data is not the save
        r.used = true;
        r.mtime = std::max(r.mtime, st.st_mtime);
    }
    if (!r.used) return;
    char prm[1328];
    if (FILE* f = std::fopen((sys / "param.bin").c_str(), "rb")) {
        if (std::fread(prm, sizeof prm, 1, f) == 1) {
            r.title = cString(prm, 128);
            r.subTitle = cString(prm + 128, 128);
            r.detail = cString(prm + 256, 1024);
        }
        std::fclose(f);
    }
    if (fs::exists(sys / "icon0.png", ec)) r.icon = (sys / "icon0.png").string();
}

const char* actionName(Action a) {
    switch (a) {
    case Action::Save: return "Save";
    case Action::Overwrite: return "Overwrite";
    case Action::Load: return "Load";
    case Action::Delete: return "Delete";
    case Action::Cancel: return "Cancel";
    }
    return "";
}

std::vector<Action> actionsFor(const Row& r) {
    switch (gKind) {
    case SaveMenuKind::Save:
        if (r.isNew) return {Action::Save, Action::Cancel};
        if (!r.used) return {Action::Save, Action::Cancel};
        return {Action::Overwrite, Action::Delete, Action::Cancel};
    case SaveMenuKind::Load:
        if (!r.used) return {};
        return {Action::Load, Action::Delete, Action::Cancel};
    case SaveMenuKind::Delete:
        if (!r.used) return {};
        return {Action::Delete, Action::Cancel};
    }
    return {};
}

void finish(const std::string& choice, bool chooseNew = false) {
    gChoice = choice;
    gChoseNew = chooseNew;
    gOpen = false;
    std::fprintf(stderr, "[save menu] %s\n", chooseNew ? "chose a new save" : choice.empty() ? "cancelled" : ("chose " + choice).c_str());
}

// Run the confirmed (or confirmation-free) action on the highlighted slot.
void perform(Action a) {
    Row& r = gRows[size_t(gCursor)];
    // In the game's own delete dialog the game deletes the slot it gets back.
    if (a == Action::Delete && gKind != SaveMenuKind::Delete) {
        std::error_code ec;
        fs::remove_all(r.slot.path, ec);
        std::fprintf(stderr, "[save menu] deleted %s%s\n", r.slot.dir.c_str(), ec ? " (failed)" : "");
        gMessage = ec ? "Could not delete save " + r.slot.dir : "Deleted save " + r.slot.dir;
        refreshRow(r);
        // A deleted save leaves the list, as on a PS4; a new save can take its number again.
        if (!r.used) {
            gRows.erase(gRows.begin() + gCursor);
            gCursor = std::clamp(gCursor, 0, std::max(0, int(gRows.size()) - 1));
        }
        gScreen = Screen::List;
        return;
    }
    if (a == Action::Cancel) {
        gScreen = Screen::List;
        return;
    }
    if (r.isNew) finish("", true);
    else finish(r.slot.dir);
}

void select() {
    switch (gScreen) {
    case Screen::List:
        if (gRows.empty()) return;
        gActions = actionsFor(gRows[size_t(gCursor)]);
        if (gActions.empty()) return;
        gActionCursor = 0;
        gMessage.clear();
        gScreen = Screen::Actions;
        break;
    case Screen::Actions: {
        const Action a = gActions[size_t(gActionCursor)];
        if (a == Action::Overwrite || a == Action::Delete) {
            gPending = a;
            gConfirmCursor = 1;
            gScreen = Screen::Confirm;
        } else {
            perform(a);
        }
        break;
    }
    case Screen::Confirm:
        if (gConfirmCursor == 0) perform(gPending);
        else gScreen = Screen::Actions;
        break;
    }
}

void back() {
    switch (gScreen) {
    case Screen::List: finish(""); break;
    case Screen::Actions: gScreen = Screen::List; break;
    case Screen::Confirm: gScreen = Screen::Actions; break;
    }
}

void quickDelete() {
    if (gScreen != Screen::List || gRows.empty() || !gRows[size_t(gCursor)].used) return;
    gActions = actionsFor(gRows[size_t(gCursor)]);
    const auto it = std::find(gActions.begin(), gActions.end(), Action::Delete);
    if (it == gActions.end()) return;
    gActionCursor = int(it - gActions.begin());
    gPending = Action::Delete;
    gConfirmCursor = 1;
    gMessage.clear();
    gScreen = Screen::Confirm;
}

void move(int d) {
    int& c = gScreen == Screen::List ? gCursor : gScreen == Screen::Actions ? gActionCursor : gConfirmCursor;
    const int n = gScreen == Screen::List ? int(gRows.size()) : gScreen == Screen::Actions ? int(gActions.size()) : 2;
    if (n > 0) c = (c + d + n) % n;
}

#ifdef LPN_HAVE_SDL2
// 5x7 glyphs, rows top to bottom; lowercase is drawn as uppercase.
struct Glyph {
    char c;
    const char* rows;
};
constexpr Glyph kFont[] = {
    {'A', ".###.#...##...#######...##...##...#"}, {'B', "####.#...##...#####.#...##...#####."},
    {'C', ".###.#...##....#....#....#...#.###."}, {'D', "####.#...##...##...##...##...#####."},
    {'E', "######....#....####.#....#....#####"}, {'F', "######....#....####.#....#....#...."},
    {'G', ".###.#...##....#.####...##...#.####"}, {'H', "#...##...##...#######...##...##...#"},
    {'I', ".###...#....#....#....#....#...###."}, {'J', "..###...#....#....#....##..#..##.."},
    {'K', "#...##..#.#.#..##...#.#..#..#.#...#"}, {'L', "#....#....#....#....#....#....#####"},
    {'M', "#...###.###.#.##.#.##...##...##...#"}, {'N', "#...##...###..##.#.##..###...##...#"},
    {'O', ".###.#...##...##...##...##...#.###."}, {'P', "####.#...##...#####.#....#....#...."},
    {'Q', ".###.#...##...##...##.#.##..#..##.#"}, {'R', "####.#...##...#####.#.#..#..#.#...#"},
    {'S', ".#####....#.....###.....#....#####."}, {'T', "#####..#....#....#....#....#....#.."},
    {'U', "#...##...##...##...##...##...#.###."}, {'V', "#...##...##...##...##...#.#.#...#.."},
    {'W', "#...##...##...##.#.##.#.##.#.#.#.#."}, {'X', "#...##...#.#.#...#...#.#.#...##...#"},
    {'Y', "#...##...#.#.#...#....#....#....#.."}, {'Z', "#####....#...#...#...#...#....#####"},
    {'0', ".###.#...##..###.#.###..##...#.###."}, {'1', "..#...##....#....#....#....#...###."},
    {'2', ".###.#...#....#...#...#...#...#####"}, {'3', "#####...#...#.....#....##...#.###."},
    {'4', "...#...##..#.#.#..#.#####...#....#."}, {'5', "######....####.....#....##...#.###."},
    {'6', "..##..#...#....####.#...##...#.###."}, {'7', "#####....#...#...#...#....#....#..."},
    {'8', ".###.#...##...#.###.#...##...#.###."}, {'9', ".###.#...##...#.####....#...#..##.."},
    {':', "......##...##.......##...##........"}, {'-', "...............#####..............."},
    {'/', ".........#...#...#...#...#........."}, {'.', ".........................##...##..."},
    {',', "....................##....#...#...."}, {'?', ".###.#...#....#...#...#.........#.."},
    {'(', "...#...#...#....#....#.....#.....#."}, {')', ".#.....#.....#....#....#...#...#..."},
    {'!', "..#....#....#....#....#.........#.."}, {'>', ".#.....#.....#.....#...#...#...#..."},
};

const char* glyphRows(char c) {
    if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
    for (const auto& g : kFont)
        if (g.c == c) return g.rows;
    return nullptr;
}

void drawText(SDL_Renderer* ren, int x, int y, int px, const std::string& s) {
    std::vector<SDL_Rect> rects;
    for (char c : s) {
        if (const char* rows = glyphRows(c))
            for (int i = 0; i < 35; ++i)
                if (rows[i] == '#') rects.push_back({x + (i % 5) * px, y + (i / 5) * px, px, px});
        x += 6 * px;
    }
    if (!rects.empty()) SDL_RenderFillRects(ren, rects.data(), int(rects.size()));
}

int textWidth(const std::string& s, int px) { return int(s.size()) * 6 * px; }

// Menu text in Arial (menu_font.h), or the pixel font when no font file was found. size is the
// line height in pixels.
void text(SDL_Renderer* ren, int x, int y, int size, const std::string& s, bool bold = false) {
    if (menufont::available()) menufont::draw(ren, x, y, size, s, bold);
    else drawText(ren, x, y + size / 8, std::max(1, size / 9), s);
}
int textW(const std::string& s, int size, bool bold = false) {
    return menufont::available() ? menufont::width(s, size, bold) : textWidth(s, std::max(1, size / 9));
}
std::string fitText(const std::string& s, int size, int maxW, bool bold = false) {
    if (menufont::available()) return menufont::fit(s, size, maxW, bold);
    const int n = std::max(0, maxW / (6 * std::max(1, size / 9)));
    return int(s.size()) <= n ? s : s.substr(0, size_t(std::max(0, n - 3))) + "...";
}

std::string sizeText(uint64_t b) {
    char buf[32];
    if (b >= (1u << 20)) std::snprintf(buf, sizeof buf, "%.2f MB", double(b) / (1 << 20));
    else std::snprintf(buf, sizeof buf, "%.0f KB", std::max(1.0, double(b) / 1024));
    return buf;
}

std::string dateText(time_t t) {
    char buf[48];
    struct tm tm;
    localtime_r(&t, &tm);
    std::strftime(buf, sizeof buf, "%a %d %b %Y, %H:%M", &tm);
    return buf;
}

// "Just now", "25 minutes ago", "Yesterday", ...: how old a save is at a glance.
std::string agoText(time_t t) {
    const time_t now = std::time(nullptr);
    const long d = long(std::max<time_t>(0, now - t));
    char buf[48];
    if (d < 60) return "Just now";
    if (d < 3600) std::snprintf(buf, sizeof buf, "%ld minute%s ago", d / 60, d / 60 == 1 ? "" : "s");
    else if (d < 86400) std::snprintf(buf, sizeof buf, "%ld hour%s ago", d / 3600, d / 3600 == 1 ? "" : "s");
    else if (d < 2 * 86400) return "Yesterday";
    else std::snprintf(buf, sizeof buf, "%ld days ago", d / 86400);
    return buf;
}

std::string firstLine(const std::string& s) {
    const size_t n = s.find_first_of("\r\n");
    return n == std::string::npos ? s : s.substr(0, n);
}

// The game's detail line packs everything into one, e.g. "Total Play Time: 3:14:51/Total Cash:
// 11,500/Level: 12/Skills Unlocked - Mind: 10/...". The list shows only the play time and level;
// a detail without either is shown as it is.
std::string detailSummary(const std::string& detail, const std::string& sep) {
    auto field = [&](const std::string& key) {
        const size_t k = detail.find(key);
        if (k == std::string::npos) return std::string();
        size_t b = k + key.size();
        while (b < detail.size() && detail[b] == ' ') ++b;
        return detail.substr(b, detail.find('/', b) - b);
    };
    const std::string time = field("Total Play Time:"), level = field("Level:");
    if (time.empty() && level.empty()) return firstLine(detail);
    std::string out;
    if (!time.empty()) out = "Play time " + time;
    if (!level.empty()) out += (out.empty() ? "" : sep) + "Level " + level;
    return out;
}

// Save icons by file, made into textures on the window thread; reloaded when the file changes.
struct Icon {
    SDL_Texture* tex = nullptr;
    int w = 0, h = 0;
    fs::file_time_type mtime{};
};
std::map<std::string, Icon> gIcons;
SDL_Renderer* gIconRen = nullptr;

const Icon* icon(SDL_Renderer* ren, const std::string& path) {
    if (path.empty()) return nullptr;
    if (ren != gIconRen) {
        for (auto& [k, i] : gIcons) SDL_DestroyTexture(i.tex);
        gIcons.clear();
        gIconRen = ren;
    }
    std::error_code ec;
    const auto mt = fs::last_write_time(path, ec);
    if (ec) return nullptr;
    auto it = gIcons.find(path);
    if (it != gIcons.end() && it->second.mtime == mt) return it->second.tex ? &it->second : nullptr;
    if (it != gIcons.end()) SDL_DestroyTexture(it->second.tex);
    Icon i;
    i.mtime = mt;
    int n = 0;
    if (unsigned char* px = stbi_load(path.c_str(), &i.w, &i.h, &n, 4)) {
        i.tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, i.w, i.h);
        if (i.tex) {
            SDL_UpdateTexture(i.tex, nullptr, px, i.w * 4);
            SDL_SetTextureBlendMode(i.tex, SDL_BLENDMODE_BLEND);
            SDL_SetTextureScaleMode(i.tex, SDL_ScaleModeLinear);
        }
        stbi_image_free(px);
    }
    Icon& slot = gIcons[path] = i;
    return slot.tex ? &slot : nullptr;
}

void fill(SDL_Renderer* ren, int x, int y, int w, int h, Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255) {
    SDL_Rect rc{x, y, w, h};
    SDL_SetRenderDrawColor(ren, r, g, b, a);
    SDL_RenderFillRect(ren, &rc);
}
void outline(SDL_Renderer* ren, int x, int y, int w, int h, int t, Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255) {
    fill(ren, x, y, w, t, r, g, b, a);
    fill(ren, x, y + h - t, w, t, r, g, b, a);
    fill(ren, x, y, t, h, r, g, b, a);
    fill(ren, x + w - t, y, t, h, r, g, b, a);
}
void color(SDL_Renderer* ren, Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255) { SDL_SetRenderDrawColor(ren, r, g, b, a); }
#endif

} // namespace

#ifdef LPN_HAVE_SDL2
void drawMenuText(SDL_Renderer* ren, int x, int y, int px, const std::string& s) { drawText(ren, x, y, px, s); }
int menuTextWidth(const std::string& s, int px) { return textWidth(s, px); }
#endif

bool openSaveMenu(SaveMenuKind kind, std::vector<SaveSlot> slots, int focus, bool newItem) {
    const char* env = std::getenv("LPN_SAVE_MENU");
    newItem = newItem && kind == SaveMenuKind::Save;
    if ((env && !std::strcmp(env, "0")) || (slots.empty() && !newItem) || !windowShowsMenus()) return false;
    std::lock_guard lock(gM);
    gKind = kind;
    gRows.clear();
    for (auto& s : slots) {
        Row r;
        r.slot = std::move(s);
        refreshRow(r);
        gRows.push_back(std::move(r));
    }
    if (newItem) {
        Row r;
        r.isNew = true;
        gRows.push_back(std::move(r));
    }
    gCursor = std::clamp(focus, 0, int(gRows.size()) - 1);
    gScreen = Screen::List;
    gMessage.clear();
    gChoice.clear();
    gChoseNew = false;
    gOpen = true;
    return true;
}

bool saveMenuOpen() { return gOpen; }

std::string saveMenuChoice() {
    std::lock_guard lock(gM);
    return gChoice;
}

bool saveMenuChoseNew() {
    std::lock_guard lock(gM);
    return gChoseNew;
}

void closeSaveMenu() {
    std::lock_guard lock(gM);
    if (gOpen) finish("");
}

#ifdef LPN_HAVE_SDL2
bool saveMenuEvent(const SDL_Event& e) {
    if (!gOpen) return false;
    std::lock_guard lock(gM);
    if (e.type == SDL_KEYDOWN) {
        switch (e.key.keysym.scancode) {
        case SDL_SCANCODE_UP: case SDL_SCANCODE_W: move(-1); break;
        case SDL_SCANCODE_DOWN: case SDL_SCANCODE_S: move(1); break;
        case SDL_SCANCODE_LEFT: case SDL_SCANCODE_A: if (gScreen != Screen::List) move(-1); break;
        case SDL_SCANCODE_RIGHT: case SDL_SCANCODE_D: if (gScreen != Screen::List) move(1); break;
        case SDL_SCANCODE_RETURN: case SDL_SCANCODE_KP_ENTER: case SDL_SCANCODE_SPACE:
            if (!e.key.repeat) select();
            break;
        case SDL_SCANCODE_BACKSPACE: case SDL_SCANCODE_ESCAPE:
            if (!e.key.repeat) back();
            break;
        case SDL_SCANCODE_Z: case SDL_SCANCODE_DELETE:
            if (!e.key.repeat) quickDelete();
            break;
        default: return false;
        }
        return true;
    }
    if (e.type == SDL_CONTROLLERBUTTONDOWN) {
        switch (e.cbutton.button) {
        case SDL_CONTROLLER_BUTTON_DPAD_UP: move(-1); break;
        case SDL_CONTROLLER_BUTTON_DPAD_DOWN: move(1); break;
        case SDL_CONTROLLER_BUTTON_DPAD_LEFT: if (gScreen != Screen::List) move(-1); break;
        case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: if (gScreen != Screen::List) move(1); break;
        case SDL_CONTROLLER_BUTTON_A: select(); break;
        case SDL_CONTROLLER_BUTTON_B: back(); break;
        case SDL_CONTROLLER_BUTTON_X: quickDelete(); break;
        default: return false;
        }
        return true;
    }
    return false;
}

// A list of saves, one row each: the save's icon with its number on it, the chapter and place
// the game gave it, then when it was saved (and how long ago) and its size. The newest save is
// marked. Under the list: the actions or the yes/no question for the highlighted save.
void drawSaveMenu(SDL_Renderer* ren) {
    if (!gOpen) return;
    std::lock_guard lock(gM);
    int w = 0, h = 0;
    SDL_GetRendererOutputSize(ren, &w, &h);
    const float u = std::max(0.5f, h / 720.0f); // 1 at 720p
    auto U = [u](float v) { return std::max(1, int(std::lround(v * u))); };
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    fill(ren, 0, 0, w, h, 0, 0, 0, 180);

    const int headH = U(64), rowH = U(96), gap = U(6), barH = U(56), hintH = U(34), pad = U(20);
    const int maxRows = std::max(1, (int(h * 0.94f) - headH - barH - hintH - 2 * pad) / (rowH + gap));
    const int n = int(gRows.size());
    const int visible = std::min(n, maxRows);
    const int first = std::max(0, std::min(gCursor - visible / 2, n - visible));
    const int panelW = std::min(w - U(32), U(860));
    const int listH = std::max(visible, 1) * (rowH + gap) - gap;
    const int panelH = headH + pad + listH + pad + barH + hintH;
    const int x0 = (w - panelW) / 2, y0 = (h - panelH) / 2;

    // Panel and header.
    fill(ren, x0, y0, panelW, panelH, 16, 16, 22, 242);
    fill(ren, x0, y0, panelW, headH, 26, 24, 30, 255);
    fill(ren, x0, y0 + headH - U(2), panelW, U(2), 200, 170, 90);
    const char* title = gKind == SaveMenuKind::Save ? "Save Game" : gKind == SaveMenuKind::Load ? "Load Game" : "Delete Save";
    color(ren, 245, 240, 228);
    text(ren, x0 + pad, y0 + (headH - U(30)) / 2, U(30), title, true);
    int saves = 0, latest = -1;
    for (int i = 0; i < n; ++i)
        if (gRows[size_t(i)].used) {
            ++saves;
            if (latest < 0 || gRows[size_t(i)].mtime > gRows[size_t(latest)].mtime) latest = i;
        }
    const std::string count = saves == 0 ? "No saves" : std::to_string(saves) + (saves == 1 ? " save" : " saves");
    color(ren, 150, 146, 138);
    text(ren, x0 + panelW - pad - textW(count, U(18)), y0 + (headH - U(18)) / 2, U(18), count);

    // Rows.
    const int listX = x0 + pad, listY = y0 + headH + pad;
    const int rowW = panelW - 2 * pad - (n > visible ? U(10) : 0);
    const int thumbH = rowH - U(16), thumbW = thumbH * 16 / 9;
    const std::string gameIcon = process().gameDir.empty() ? "" : process().gameDir + "/sce_sys/icon0.png";
    for (int i = 0; i < visible; ++i) {
        const int idx = first + i;
        const Row& r = gRows[size_t(idx)];
        const bool sel = idx == gCursor;
        const int ry = listY + i * (rowH + gap);
        if (sel) {
            fill(ren, listX, ry, rowW, rowH, 58, 47, 24);
            outline(ren, listX, ry, rowW, rowH, U(2), 214, 182, 98);
        } else {
            fill(ren, listX, ry, rowW, rowH, 30, 30, 38);
        }

        // Thumbnail: the save's icon (or the game's), fitted into a 16:9 box, number on its corner.
        const int tx = listX + U(8), ty = ry + U(8);
        fill(ren, tx, ty, thumbW, thumbH, 8, 8, 12);
        if (r.isNew) {
            color(ren, 214, 182, 98);
            const int s = U(48);
            text(ren, tx + (thumbW - textW("+", s, true)) / 2, ty + (thumbH - s) / 2, s, "+", true);
        } else if (const Icon* ic = r.used ? icon(ren, r.icon.empty() ? gameIcon : r.icon) : nullptr) {
            const float k = std::min(float(thumbW) / ic->w, float(thumbH) / ic->h);
            const int dw = int(ic->w * k), dh = int(ic->h * k);
            SDL_Rect dst{tx + (thumbW - dw) / 2, ty + (thumbH - dh) / 2, dw, dh};
            SDL_SetTextureColorMod(ic->tex, sel ? 255 : 200, sel ? 255 : 200, sel ? 255 : 200);
            SDL_RenderCopy(ren, ic->tex, nullptr, &dst);
        }
        if (!r.isNew) {
            const int bs = U(17), bw = textW(r.slot.dir, bs, true) + U(12);
            fill(ren, tx, ty, bw, bs + U(6), 0, 0, 0, 200);
            color(ren, 232, 200, 112);
            text(ren, tx + U(6), ty + U(3), bs, r.slot.dir, true);
        }

        // Text: headline, a second line, then the date and size.
        const int cx = tx + thumbW + U(16), cw = listX + rowW - cx - U(12);
        if (r.isNew) {
            color(ren, 245, 240, 228);
            text(ren, cx, ry + (rowH - U(24)) / 2 - U(10), U(24), "New Save", true);
            color(ren, 150, 146, 138);
            text(ren, cx, ry + (rowH - U(24)) / 2 + U(20), U(16), "Make a new save in the next free slot");
            continue;
        }
        if (!r.used) {
            color(ren, 140, 140, 140);
            text(ren, cx, ry + (rowH - U(24)) / 2, U(24), "Empty Slot", true);
            continue;
        }
        // The chapter and place are what tell saves apart, so they lead; the game's name, the same
        // on every save, only stands in when nothing else was given.
        const std::string sep = menufont::available() ? "   \xC2\xB7   " : "   ";
        std::string head = firstLine(r.subTitle), second = detailSummary(r.detail, sep);
        if (head.empty()) head = firstLine(r.title);
        else if (second.empty()) second = firstLine(r.title);
        if (head.empty()) head = "Save " + r.slot.dir;
        if (r.subTitle.empty() && r.detail.empty()) second = "No details: saved before they were recorded";
        std::string tag = idx == latest && saves > 1 ? "LATEST" : "";
        const int tagS = U(13), tagW = tag.empty() ? 0 : textW(tag, tagS, true) + U(14);
        color(ren, 245, 240, 228);
        text(ren, cx, ry + U(12), U(24), fitText(head, U(24), cw - (tagW ? tagW + U(10) : 0), true), true);
        if (!tag.empty()) {
            const int gx = listX + rowW - U(12) - tagW, gy = ry + U(14);
            fill(ren, gx, gy, tagW, tagS + U(8), 200, 170, 90);
            color(ren, 24, 20, 12);
            text(ren, gx + U(7), gy + U(4), tagS, tag, true);
        }
        color(ren, 186, 182, 172);
        text(ren, cx, ry + U(42), U(17), fitText(second, U(17), cw));
        const std::string when = dateText(r.mtime) + sep + sizeText(r.bytes);
        const std::string ago = agoText(r.mtime);
        color(ren, 214, 182, 98);
        text(ren, cx, ry + U(68), U(16), fitText(when, U(16), cw - textW(ago, U(16)) - U(16)));
        color(ren, 150, 146, 138);
        text(ren, listX + rowW - U(12) - textW(ago, U(16)), ry + U(68), U(16), ago);
    }
    if (n == 0) {
        color(ren, 150, 146, 138);
        text(ren, listX, listY + (rowH - U(20)) / 2, U(20), "No saves");
    }
    // Scroll bar when the list runs past the panel.
    if (n > visible) {
        const int sx = x0 + panelW - pad - U(4);
        fill(ren, sx, listY, U(4), listH, 40, 40, 48);
        const int th = std::max(U(20), listH * visible / n), tyy = listY + (listH - th) * first / std::max(1, n - visible);
        fill(ren, sx, tyy, U(4), th, 200, 170, 90);
    }

    // The actions or the yes/no question for the highlighted save.
    const int by = y0 + headH + pad + listH + pad;
    fill(ren, x0, by, panelW, U(1), 50, 48, 56);
    const int bs = U(19), bh = bs + U(14), byy = by + (barH - bh) / 2;
    if (gScreen == Screen::Actions || gScreen == Screen::Confirm) {
        int x = x0 + pad;
        std::vector<std::string> items;
        if (gScreen == Screen::Actions) {
            for (Action a : gActions) items.push_back(actionName(a));
        } else {
            const Row& r = gRows[size_t(gCursor)];
            const std::string q = std::string(actionName(gPending)) + " save " + r.slot.dir + "?";
            color(ren, 245, 240, 228);
            text(ren, x, byy + U(7), bs, q, true);
            x += textW(q, bs, true) + U(20);
            items = {"Yes", "No"};
        }
        const int sel = gScreen == Screen::Actions ? gActionCursor : gConfirmCursor;
        for (size_t i = 0; i < items.size(); ++i) {
            const int iw = std::max(U(90), textW(items[i], bs) + U(28));
            if (int(i) == sel) {
                fill(ren, x, byy, iw, bh, 200, 170, 90);
                color(ren, 24, 20, 12);
            } else {
                outline(ren, x, byy, iw, bh, U(1), 110, 104, 92);
                color(ren, 225, 220, 208);
            }
            text(ren, x + (iw - textW(items[i], bs)) / 2, byy + U(7), bs, items[i]);
            x += iw + U(10);
        }
    } else if (!gMessage.empty()) {
        color(ren, 225, 220, 208);
        text(ren, x0 + pad, byy + U(7), bs, gMessage);
    }

    // Controls.
    fill(ren, x0, y0 + panelH - hintH, panelW, hintH, 10, 10, 14, 255);
    color(ren, 140, 136, 128);
    text(ren, x0 + pad, y0 + panelH - hintH + (hintH - U(15)) / 2, U(15),
         "Enter / A: select      Backspace / B: back      Z / X: delete");
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
}
#endif

} // namespace host
