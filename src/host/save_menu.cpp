#include "host/save_menu.h"

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
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
    std::string when; // newest file's time, "" when empty
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

void refreshRow(Row& r) {
    r.used = false;
    if (r.isNew) return;
    r.when.clear();
    time_t newest = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(r.slot.path, ec), end; !ec && it != end; it.increment(ec)) {
        struct stat st;
        if (!it->is_regular_file(ec) || ::stat(it->path().c_str(), &st) != 0) continue;
        r.used = true;
        newest = std::max(newest, st.st_mtime);
    }
    if (r.used) {
        char buf[32];
        struct tm tm;
        localtime_r(&newest, &tm);
        std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
        r.when = buf;
    }
}

const char* actionName(Action a) {
    switch (a) {
    case Action::Save: return "SAVE";
    case Action::Overwrite: return "OVERWRITE";
    case Action::Load: return "LOAD";
    case Action::Delete: return "DELETE";
    case Action::Cancel: return "CANCEL";
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
        gMessage = ec ? "COULD NOT DELETE " + r.slot.dir : "DELETED " + r.slot.dir;
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

void drawSaveMenu(SDL_Renderer* ren) {
    if (!gOpen) return;
    std::lock_guard lock(gM);
    int w = 0, h = 0;
    SDL_GetRendererOutputSize(ren, &w, &h);
    const int px = std::max(1, h / 240);  // 3 at 720p
    const int line = 12 * px;
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 170);
    SDL_RenderFillRect(ren, nullptr);

    const int visible = std::min<int>(int(gRows.size()), 10);
    const int first = std::max(0, std::min(gCursor - visible / 2, int(gRows.size()) - visible));
    const int panelW = std::min(w - 4 * px, 200 * px), panelH = (visible + 6) * line;
    const int x0 = (w - panelW) / 2, y0 = (h - panelH) / 2;
    SDL_Rect panel{x0, y0, panelW, panelH};
    SDL_SetRenderDrawColor(ren, 20, 20, 28, 235);
    SDL_RenderFillRect(ren, &panel);
    SDL_SetRenderDrawColor(ren, 200, 170, 90, 255);
    SDL_RenderDrawRect(ren, &panel);

    const char* title = gKind == SaveMenuKind::Save ? "SAVE GAME" : gKind == SaveMenuKind::Load ? "LOAD GAME" : "DELETE SAVE";
    drawText(ren, x0 + 6 * px, y0 + 4 * px, px, title);

    for (int i = 0; i < visible; ++i) {
        const int idx = first + i;
        const Row& r = gRows[size_t(idx)];
        const int y = y0 + (i + 2) * line;
        if (idx == gCursor) {
            SDL_Rect hl{x0 + 3 * px, y - 2 * px, panelW - 6 * px, line - px};
            SDL_SetRenderDrawColor(ren, 90, 70, 30, 255);
            SDL_RenderFillRect(ren, &hl);
        }
        SDL_SetRenderDrawColor(ren, r.used ? 235 : 140, r.used ? 235 : 140, r.used ? 225 : 140, 255);
        if (r.isNew) {
            drawText(ren, x0 + 6 * px, y, px, "EMPTY");
            continue;
        }
        drawText(ren, x0 + 6 * px, y, px, "SLOT " + r.slot.dir);
        drawText(ren, x0 + 6 * px + textWidth("SLOT " + r.slot.dir + "  ", px), y, px, r.used ? r.when : "EMPTY");
    }

    // The actions or the yes/no question for the highlighted slot, at the bottom of the panel.
    const int yb = y0 + (visible + 3) * line;
    SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
    if (gScreen == Screen::Actions || gScreen == Screen::Confirm) {
        int x = x0 + 6 * px;
        std::vector<std::string> items;
        if (gScreen == Screen::Actions) {
            for (Action a : gActions) items.push_back(actionName(a));
        } else {
            const std::string q = std::string(actionName(gPending)) + " SLOT " + gRows[size_t(gCursor)].slot.dir + "?";
            drawText(ren, x, yb, px, q);
            x += textWidth(q + "  ", px);
            items = {"YES", "NO"};
        }
        const int sel = gScreen == Screen::Actions ? gActionCursor : gConfirmCursor;
        for (size_t i = 0; i < items.size(); ++i) {
            if (int(i) == sel) {
                SDL_Rect hl{x - 2 * px, yb - 2 * px, textWidth(items[i], px) + 3 * px, line - px};
                SDL_SetRenderDrawColor(ren, 200, 170, 90, 255);
                SDL_RenderFillRect(ren, &hl);
                SDL_SetRenderDrawColor(ren, 20, 20, 28, 255);
            } else {
                SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
            }
            drawText(ren, x, yb, px, items[i]);
            x += textWidth(items[i] + "   ", px);
        }
    } else if (!gMessage.empty()) {
        drawText(ren, x0 + 6 * px, yb, px, gMessage);
    }
    SDL_SetRenderDrawColor(ren, 150, 150, 150, 255);
    const int hpx = std::max(1, px * 2 / 3);
    drawText(ren, x0 + 6 * px, y0 + panelH - line + px, hpx, "ENTER SELECT   BACKSPACE BACK   Z DELETE");
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
}
#endif

} // namespace host
