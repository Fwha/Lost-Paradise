// lpn-launcher: a small window with Play and Settings in front of lpn-host.
//
// It runs in the install folder: the folder the AppImage is in ($APPIMAGE), else $LPN_HOME, else
// the working directory (on Windows: $LPN_HOME, else the folder of the .exe). That folder holds game/CUSA12776 (common/game_dir.h), lpn-data,
// lpn-display.ini and play.log, as for launch.sh. The settings are the in-game PC Settings: both
// read and write lpn-display.ini (common/settings_file.h), so a change in either shows in the other.
// Play replaces the launcher with lpn-host, its log going to play.log. Check for updates: updater.h.
// The gear at the bottom right opens Tools: unpacking a PS4 .pkg into the game folder (pkg/pkg.h),
// picked with the desktop's file chooser (zenity or kdialog) or dropped on the window.
#include "common/game_dir.h"
#include "common/settings_file.h"
#include "common/ui_palette.h"
#include "host/menu_font.h"
#include "host/pad_glyphs.h"
#include "host/rounded.h"
#include "launcher/updater.h"
#include "pkg/install.h"
#include "pkg/pkg.h"

#include <SDL.h>
#include <nanosvg.h>
#include <nanosvgrast.h>
#include <stb_image.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

namespace fs = std::filesystem;
namespace font = host::menufont;
using host::DisplaySettings;

namespace {

constexpr int kW = 1152, kH = 648;

// Paths
fs::path gInstall, gExeDir;
std::string gGameDir, gHost, gNids;
DisplaySettings gS;

fs::path exeDir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, buf, DWORD(std::size(buf)));
    return n ? fs::path(std::wstring(buf, n)).parent_path() : fs::current_path();
#else
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::current_path() : p.parent_path();
#endif
}

fs::path installDir() {
    if (const char* a = std::getenv("APPIMAGE"); a && *a) return fs::absolute(a).parent_path();
    if (const char* h = std::getenv("LPN_HOME"); h && *h) return fs::absolute(h);
#ifdef _WIN32
    return gExeDir; // the unzipped folder, which the game folder goes in
#else
    return fs::current_path();
#endif
}

std::string firstExisting(std::initializer_list<fs::path> paths) {
    std::error_code ec;
    for (const fs::path& p : paths)
        if (fs::is_regular_file(p, ec)) return fs::absolute(p).lexically_normal().string();
    return {};
}

void findFiles() {
    // The game folder is made for the player to put the dump in.
    std::error_code ec;
    fs::create_directories(gInstall / host::kGameFolder, ec);
    gGameDir = host::findGameDir(gInstall.string());
    // In the AppImage: usr/bin/lpn-launcher, lpn-host beside it and usr/share/lost-paradise.
    // In a build tree: build/src/launcher, build/src/host and the repo's aerolib.csv.
#ifdef _WIN32
    gHost = firstExisting({gExeDir / "lpn-host.exe", gExeDir / "../host/lpn-host.exe"});
#else
    gHost = firstExisting({gExeDir / "lpn-host", gExeDir / "../host/lpn-host",
                           gInstall / "lost-paradise/build/src/host/lpn-host"});
#endif
    const char* env = std::getenv("LPN_NIDS");
    gNids = firstExisting({env ? fs::path(env) : fs::path(), gExeDir / "../share/lost-paradise/aerolib.csv",
                           gExeDir / "aerolib.csv", gExeDir / "../../../aerolib.csv",
                           gInstall / "lost-paradise/aerolib.csv"});
}

void save() {
    if (!host::writeSettings(host::kSettingsFile, gS))
        std::fprintf(stderr, "[launcher] cannot write %s/%s\n", gInstall.string().c_str(), host::kSettingsFile);
}

// Settings rows: the in-game PC Settings, in its order (src/host/pc_settings.cpp).
struct Row {
    const char* label;
    const char* description;
};
const Row kRows[] = {
    {"Window Mode", "Play in a window, a borderless window or fullscreen."},
    {"Resolution", "Set the 3D rendering resolution."},
    {"Aspect Ratio", "Match your monitor's shape: the 3D view widens to fit, the HUD stays 16:9 in the middle."},
    {"Frame Rate Cap", "Set the most frames shown per second. Uncapped follows the monitor's refresh rate."},
    {"FPS Counter", "Show the frame rate in the top right corner of the screen."},
    {"DLSS", "NVIDIA DLAA: DLSS anti-aliasing in place of the game's. Needs an RTX card; Off on other GPUs."},
    {"Skip Intro Logos", "Go straight to the title screen when the game starts."},
    {"Button Icons", "Which controller's buttons to show. Auto matches your controller. Takes effect at the next start."},
};
constexpr int kRowCount = int(sizeof kRows / sizeof kRows[0]);

int resIndex() {
    for (int i = 0; i < host::kResCount; ++i)
        if (host::kResolutions[i].w == gS.width && host::kResolutions[i].h == gS.height) return i;
    return -1;
}
int fpsIndex() {
    for (int i = 0; i < host::kFpsCount; ++i)
        if (host::kFpsChoices[i] == gS.fps) return i;
    return -1;
}

std::string valueName(int row) {
    switch (row) {
    case 0: return host::kWindowModeNames[int(gS.window)];
    case 1: return resIndex() >= 0 ? host::kResolutions[resIndex()].label
                                   : std::to_string(gS.width) + "x" + std::to_string(gS.height);
    case 2: {
        std::string name = host::kAspectNames[int(gS.aspect)];
        uint32_t w, h;
        host::renderSize(gS, &w, &h);
        if (gS.aspect != host::AspectRatio::Wide16x9) name += " (" + std::to_string(w) + "x" + std::to_string(h) + ")";
        return name;
    }
    case 3: return gS.fps ? std::to_string(gS.fps) : "Uncapped";
    case 4: return gS.fpsCounter ? "On" : "Off";
    case 5: return gS.dlss != host::DlssMode::Off ? "DLAA" : "Off";
    case 6: return gS.skipIntro ? "On" : "Off";
    default: return host::kButtonIconNames[int(gS.buttonIcons)];
    }
}

// Steps a row's choice by d (+1 or -1), wrapping round.
void stepValue(int row, int d) {
    auto wrap = [](int i, int n) { return ((i % n) + n) % n; };
    switch (row) {
    case 0: gS.window = host::WindowMode(wrap(int(gS.window) + d, 3)); break;
    case 1: {
        // A size from the ini that the list does not hold steps to the list's ends.
        const int i = resIndex();
        const int n = wrap(i < 0 ? (d > 0 ? 0 : -1) : i + d, host::kResCount);
        gS.width = host::kResolutions[n].w, gS.height = host::kResolutions[n].h;
        break;
    }
    case 2: gS.aspect = host::AspectRatio(wrap(int(gS.aspect) + d, host::kAspectChoices)); break;
    case 3: {
        const int i = fpsIndex();
        gS.fps = host::kFpsChoices[wrap(i < 0 ? (d > 0 ? 0 : -1) : i + d, host::kFpsCount)];
        break;
    }
    case 4: gS.fpsCounter = !gS.fpsCounter; break;
    case 5: gS.dlss = gS.dlss == host::DlssMode::Off ? host::DlssMode::Dlaa : host::DlssMode::Off; break;
    case 6: gS.skipIntro = !gS.skipIntro; break;
    default: gS.buttonIcons = host::ButtonIcons(wrap(int(gS.buttonIcons) + d, host::kButtonIconChoices)); break;
    }
    save();
}

// Drawing
SDL_Renderer* gRen = nullptr;
SDL_Texture* gArt = nullptr; // the game's own key art (sce_sys/pic1.png), when there is a game
int gArtW = 0, gArtH = 0;

void color(Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255) { SDL_SetRenderDrawColor(gRen, r, g, b, a); }
void color(ui::Rgb c, Uint8 a = 255) { color(c.r, c.g, c.b, a); }
void fill(int x, int y, int w, int h) {
    SDL_Rect r{x, y, w, h};
    SDL_RenderFillRect(gRen, &r);
}
// Rounded versions of fill and outline (rounded.h) in the current colour; radius r.
SDL_Color drawColor() {
    SDL_Color c;
    SDL_GetRenderDrawColor(gRen, &c.r, &c.g, &c.b, &c.a);
    return c;
}
void fillR(int x, int y, int w, int h, int r, unsigned corners = ui::kAll) {
    ui::fillRounded(gRen, x, y, w, h, r, drawColor(), corners);
}
void outlineR(int x, int y, int w, int h, int r, int t) { ui::outlineRounded(gRen, x, y, w, h, r, t, drawColor()); }
void text(int x, int y, int size, const std::string& s, bool bold = false) { font::draw(gRen, x, y, size, s, bold); }
void textCentered(int cx, int y, int size, const std::string& s, bool bold = false) {
    text(cx - font::width(s, size, bold) / 2, y, size, s, bold);
}

SDL_Surface* loadPng(const fs::path& p) {
    int w, h, n;
    unsigned char* px = stbi_load(p.string().c_str(), &w, &h, &n, 4);
    if (!px) return nullptr;
    SDL_Surface* s = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    if (s)
        for (int y = 0; y < h; ++y) std::memcpy(static_cast<uint8_t*>(s->pixels) + y * s->pitch, px + y * w * 4, w * 4);
    stbi_image_free(px);
    return s;
}

void loadArt(SDL_Window* win) {
    if (gGameDir.empty()) return;
    if (SDL_Surface* s = loadPng(fs::path(gGameDir) / "sce_sys/pic1.png")) {
        gArt = SDL_CreateTextureFromSurface(gRen, s);
        gArtW = s->w, gArtH = s->h;
        SDL_FreeSurface(s);
    }
    if (SDL_Surface* s = loadPng(fs::path(gGameDir) / "sce_sys/icon0.png")) {
        SDL_SetWindowIcon(win, s);
        SDL_FreeSurface(s);
    }
}

void drawBackground(bool dim) {
    color(12, 10, 9);
    SDL_RenderClear(gRen);
    if (gArt) {
        // Cover the window, cropping the art's sides or top and bottom.
        const double scale = std::max(double(kW) / gArtW, double(kH) / gArtH);
        const int w = int(gArtW * scale), h = int(gArtH * scale);
        SDL_Rect dst{(kW - w) / 2, (kH - h) / 2, w, h};
        SDL_RenderCopy(gRen, gArt, nullptr, &dst);
    }
    SDL_SetRenderDrawBlendMode(gRen, SDL_BLENDMODE_BLEND);
    // Darker towards the bottom, where the buttons are.
    for (int i = 0; i < 64; ++i) {
        color(0, 0, 0, Uint8(std::min(230, (dim ? 140 : 0) + i * 3)));
        fill(0, kH - 256 + i * 4, kW, 4);
    }
    if (dim) {
        color(0, 0, 0, 140);
        fill(0, 0, kW, kH - 256);
    }
}

struct Box {
    int x, y, w, h;
    bool in(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; }
};

// Main screen
const Box kPlay{kW - 340, kH - 196, 300, 64};
const Box kSettings{kW - 340, kH - 116, 300, 52};
const Box kSkip{40, kH - 64, 300, 32};
const Box kUpdate{kW - 260, 24, 236, 40};
const Box kGear{kW - 84, kH - 56, 44, 44};
// Settings screen
const Box kPanel{176, 64, kW - 352, 470};
constexpr int kRowY0 = 132, kRowH = 42;
Box rowBox(int r) { return {kPanel.x + 24, kRowY0 + r * kRowH, kPanel.w - 48, kRowH - 6}; }
Box arrowBox(int r, bool right) {
    const Box b = rowBox(r);
    return right ? Box{b.x + b.w - 44, b.y, 44, b.h} : Box{b.x + b.w - 330, b.y, 44, b.h};
}
const Box kBack{kW / 2 + 12, kPanel.y + kPanel.h + 20, 200, 48};
const Box kDefaults{kW / 2 - 212, kPanel.y + kPanel.h + 20, 200, 48};
// Tools screen
const Box kToolsPanel{226, 52, kW - 452, 476};
const Box kToolButtons[3] = {{kToolsPanel.x + 28, kToolsPanel.y + 140, 260, 44},  // unpack a .pkg
                             {kToolsPanel.x + 28, kToolsPanel.y + 266, 260, 44},  // update/DLC from a .pkg
                             {kToolsPanel.x + 308, kToolsPanel.y + 266, 260, 44}}; // update/DLC from a folder
const Box kToolsBack{kW / 2 - 100, kToolsPanel.y + kToolsPanel.h + 20, 200, 48};
// Version screen (Play with more than one game version in game/): rows as in Settings
constexpr int kMaxVersions = 8;
const Box kVersionsBack{kW / 2 - 100, kPanel.y + kPanel.h + 20, 200, 48};

void button(const Box& b, const std::string& label, bool focus, bool primary, bool enabled = true) {
    if (primary && enabled) color(focus ? ui::kAccent.bright : ui::kAccent.line);
    else color(0, 0, 0, focus ? 200 : 150);
    fillR(b.x, b.y, b.w, b.h, 8);
    color(ui::kAccent.line, enabled ? 255 : 90);
    outlineR(b.x, b.y, b.w, b.h, 8, focus ? 3 : 1);
    const int size = b.h * 9 / 20;
    if (primary && enabled) color(ui::kAccent.on);
    else color(245, 240, 228, enabled ? 255 : 110);
    textCentered(b.x + b.w / 2, b.y + (b.h - size) / 2, size, label, true);
}

void checkbox(const Box& b, const std::string& label, bool on, bool focus) {
    const int s = b.h - 6;
    color(0, 0, 0, 170);
    fillR(b.x, b.y + 3, s, s, 6);
    color(ui::kAccent.line);
    outlineR(b.x, b.y + 3, s, s, 6, focus ? 3 : 2);
    if (on) fillR(b.x + 6, b.y + 9, s - 12, s - 12, 3);
    color(245, 240, 228);
    text(b.x + s + 12, b.y + (b.h - 20) / 2, 20, label, focus);
}

// A gear, white on clear, rasterised once at the size it is drawn: eight teeth round a ring.
SDL_Texture* gGear = nullptr;
int gGearSize = 0;

SDL_Texture* gearTexture(int size) {
    if (gGear && gGearSize == size) return gGear;
    if (gGear) SDL_DestroyTexture(gGear), gGear = nullptr;
    constexpr float kPi = 3.14159265f, c = 50, rOut = 47, rBody = 36, rHole = 15;
    std::string d;
    char buf[64];
    for (int i = 0; i < 8; ++i) {
        const float a = i * kPi / 4;
        const float pts[4][2] = {{rBody, a - 0.25f}, {rOut, a - 0.16f}, {rOut, a + 0.16f}, {rBody, a + 0.25f}};
        for (int k = 0; k < 4; ++k) {
            std::snprintf(buf, sizeof buf, "%c%.2f %.2f ", i == 0 && k == 0 ? 'M' : 'L',
                          c + pts[k][0] * std::cos(pts[k][1]), c + pts[k][0] * std::sin(pts[k][1]));
            d += buf;
        }
    }
    std::snprintf(buf, sizeof buf, "Z M%.0f %.0f A%.0f %.0f 0 1 0 %.0f %.0f A%.0f %.0f 0 1 0 %.0f %.0f Z", c + rHole, c,
                  rHole, rHole, c - rHole, c, rHole, rHole, c + rHole, c);
    d += buf;
    std::string svg = "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><path fill='#ffffff' "
                      "fill-rule='evenodd' d='" + d + "'/></svg>";
    NSVGimage* img = nsvgParse(svg.data(), "px", 96.0f);
    if (!img) return nullptr;
    std::vector<uint8_t> px(size_t(size) * size * 4);
    NSVGrasterizer* rast = nsvgCreateRasterizer();
    nsvgRasterize(rast, img, 0, 0, size / 100.0f, px.data(), size, size, size * 4);
    nsvgDeleteRasterizer(rast);
    nsvgDelete(img);
    gGear = SDL_CreateTexture(gRen, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, size, size);
    if (!gGear) return nullptr;
    SDL_UpdateTexture(gGear, nullptr, px.data(), size * 4);
    SDL_SetTextureBlendMode(gGear, SDL_BLENDMODE_BLEND);
    gGearSize = size;
    return gGear;
}

void gearButton(const Box& b, bool focus) {
    color(0, 0, 0, focus ? 200 : 150);
    fillR(b.x, b.y, b.w, b.h, 8);
    color(ui::kAccent.line);
    outlineR(b.x, b.y, b.w, b.h, 8, focus ? 3 : 1);
    const int s = b.h * 6 / 10;
    if (SDL_Texture* t = gearTexture(s)) {
        const ui::Rgb c = focus ? ui::kAccent.bright : ui::Rgb{245, 240, 228};
        SDL_SetTextureColorMod(t, c.r, c.g, c.b);
        SDL_Rect dst{b.x + (b.w - s) / 2, b.y + (b.h - s) / 2, s, s};
        SDL_RenderCopy(gRen, t, nullptr, &dst);
    }
}

// Text broken between words into lines no wider than maxW.
std::vector<std::string> wrap(const std::string& s, int size, int maxW) {
    std::vector<std::string> lines;
    std::string line;
    size_t at = 0;
    while (at < s.size()) {
        size_t sp = s.find(' ', at);
        if (sp == std::string::npos) sp = s.size();
        const std::string word = s.substr(at, sp - at);
        const std::string next = line.empty() ? word : line + " " + word;
        if (!line.empty() && font::width(next, size) > maxW) lines.push_back(line), line = word;
        else line = next;
        at = sp + 1;
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

// With a controller plugged in (or LPN_MENU_BUTTONS=pad), its Select and Back buttons at (x, y),
// as the Button Icons setting draws them.
void padHints(int x, int y, const char* backLabel) {
    if (const char* e = std::getenv("LPN_MENU_BUTTONS"); e && !std::strcmp(e, "keys")) return;
    bool pad = false;
    for (int i = 0; i < SDL_NumJoysticks(); ++i) pad = pad || SDL_IsGameController(i);
    if (const char* e = std::getenv("LPN_MENU_BUTTONS"); e && !std::strcmp(e, "pad")) pad = true;
    if (!pad) return;
    const host::PadStyle style = host::padStyle(gS.buttonIcons);
    const std::pair<SDL_GameControllerButton, const char*> hints[] = {{SDL_CONTROLLER_BUTTON_A, "Select"},
                                                                     {SDL_CONTROLLER_BUTTON_B, backLabel}};
    constexpr int kIcon = 28, kText = 18;
    int w = 12;
    for (const auto& h : hints) w += kIcon + 8 + font::width(h.second, kText) + 20;
    color(0, 0, 0, 160);
    fillR(x, y, w - 8, kIcon + 12, 8);
    x += 12;
    for (const auto& h : hints) {
        host::drawPadGlyph(gRen, x, y + 6, kIcon, style, h.first);
        x += kIcon + 8;
        color(225, 220, 208);
        text(x, y + 6 + (kIcon - kText) / 2, kText, h.second);
        x += font::width(h.second, kText) + 20;
    }
}

std::string gStatus; // why Play cannot start, or a launch error

// Play stays pressable while only the game is missing: pressing it looks again (tryPlay), and
// opens the no-game dialog when there is still none.
bool canPlay() { return !gHost.empty() && !gNids.empty(); }

constexpr const char* kNoGame = "No game in the '/game' folder, please place your game in the '/game' folder";
bool gNoGameDialog = false; // shown on Play with an empty game folder
const Box kDialog{kW / 2 - 280, kH / 2 - 120, 560, 240};
const Box kDialogOk{kW / 2 - 90, kDialog.y + kDialog.h - 72, 180, 48};

// The game folder, for showing where it is.
std::string gameFolderPath() { return (gInstall / host::kGameFolder).string(); }

void refreshStatus() {
    // A missing game has its own notice in the middle of the main screen (drawNoGame).
    if (gHost.empty()) gStatus = "lpn-host not found beside the launcher";
    else if (gNids.empty()) gStatus = "aerolib.csv not found (scripts/fetch-nids.sh)";
    else gStatus.clear();
}

// Looks for the game again (it may have been put in place since); false with the status saying
// what is missing.
bool tryPlay() {
    findFiles();
    refreshStatus();
    return canPlay() && !gGameDir.empty();
}

// Tools jobs (unpacking a .pkg, applying an update or DLC): the file chooser and the work run on
// a thread of their own, sharing one of these with the window.
struct Unpack {
    enum State { Idle, Choosing, Working, Finished };
    std::atomic<int> state{Idle};
    int button = 0; // the Tools button that started it (it shows Cancel)
    pkg::Progress progress;
    std::mutex mutex;
    std::string message; // the result, or why the chooser could not open
    bool ok = false;
};
std::shared_ptr<Unpack> gUnpack = std::make_shared<Unpack>();

bool unpackBusy() {
    const int s = gUnpack->state;
    return s == Unpack::Choosing || s == Unpack::Working;
}

// The desktop's file chooser: zenity (GNOME and most others), else kdialog (KDE); a .pkg file or
// a folder. Empty when cancelled, or with err set when there is no chooser.
#ifdef _WIN32
// Windows' own open dialogs. Empty when cancelled.
std::string choosePath(bool folder, std::string&) {
    char path[MAX_PATH * 4] = {};
    if (folder) {
        BROWSEINFOA bi{};
        bi.lpszTitle = "Choose an unpacked update or DLC folder";
        bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
        LPITEMIDLIST id = SHBrowseForFolderA(&bi);
        if (!id) return {};
        const bool ok = SHGetPathFromIDListA(id, path);
        CoTaskMemFree(id);
        return ok ? path : std::string();
    }
    OPENFILENAMEA o{};
    o.lStructSize = sizeof o;
    o.lpstrFilter = "PS4 packages (*.pkg)\0*.pkg;*.PKG\0All files\0*.*\0";
    o.lpstrFile = path;
    o.nMaxFile = DWORD(sizeof path);
    o.lpstrTitle = "Choose a .pkg file";
    o.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&o) ? path : std::string();
}
#else
std::string choosePath(bool folder, std::string& err) {
    const char* home = std::getenv("HOME");
    const std::string start = home && *home ? std::string(home) + "/" : "./";
    auto quote = [](const std::string& s) {
        std::string q = "'";
        for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
        return q + "'";
    };
    std::string cmd;
    if (std::system("command -v zenity >/dev/null 2>&1") == 0)
        cmd = folder ? "zenity --file-selection --directory --title='Choose an unpacked update or DLC folder' --filename=" +
                           quote(start) + " 2>/dev/null"
                     : "zenity --file-selection --title='Choose a .pkg file' --filename=" + quote(start) +
                           " --file-filter='PS4 packages (*.pkg) | *.pkg *.PKG' --file-filter='All files | *' 2>/dev/null";
    else if (std::system("command -v kdialog >/dev/null 2>&1") == 0)
        cmd = folder ? "kdialog --title 'Choose an unpacked update or DLC folder' --getexistingdirectory " + quote(start) +
                           " 2>/dev/null"
                     : "kdialog --title 'Choose a .pkg file' --getopenfilename " + quote(start) +
                           " 'PS4 packages (*.pkg *.PKG)' 2>/dev/null";
    else {
        err = "No file chooser found (zenity or kdialog): drop the .pkg file or folder on this window instead";
        return {};
    }
    std::FILE* f = popen(cmd.c_str(), "r");
    if (!f) {
        err = "Cannot open the file chooser";
        return {};
    }
    std::string out;
    char buf[4096];
    while (std::fgets(buf, sizeof buf, f)) out += buf;
    pclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}
#endif

// Runs a Tools job on path: a .pkg is unpacked (a game) or applied (an update or DLC), a folder
// is applied as an unpacked update or DLC. An empty path asks the chooser for one (a folder
// when folder is set). button: the Tools button that started it.
void startJob(std::string path, bool folder, int button) {
    if (unpackBusy()) return;
    auto u = std::make_shared<Unpack>();
    u->state = path.empty() ? Unpack::Choosing : Unpack::Working;
    u->button = button;
    gUnpack = u;
    const fs::path games = gInstall / host::kGameFolder;
    std::thread([u, path, folder, games]() mutable {
        auto finish = [&](bool ok, const std::string& message) {
            {
                std::lock_guard lock(u->mutex);
                u->ok = ok, u->message = message;
            }
            u->state = Unpack::Finished;
        };
        std::string err, message;
        if (path.empty()) {
            path = choosePath(folder, err);
            if (path.empty()) {
                if (err.empty()) u->state = Unpack::Idle;
                else finish(false, err);
                return;
            }
            u->state = Unpack::Working;
        }
        std::error_code ec;
        const bool isDir = fs::is_directory(path, ec);
        std::fprintf(stderr, "[launcher] %s %s\n", isDir ? "applying" : "unpacking", path.c_str());
        const bool ok = isDir ? pkg::applyFolder(path, games.string(), false, &u->progress, message, err)
                              : pkg::installPkg(path, games.string(), &u->progress, message, err);
        if (ok) return finish(true, message);
        std::fprintf(stderr, "[launcher] failed: %s\n", err.c_str());
        const std::string name = fs::path(path).filename().string();
        finish(false, err == "cancelled" ? "Cancelled" : "Cannot use " + name + ": " + err);
    }).detach();
}

bool isPkg(const std::string& p) {
    std::string ext = fs::path(p).extension().string();
    for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".pkg";
}

// The main screen's notice while game/ holds no game: there is no art behind it then.
void drawNoGame() {
    const int w = 620, x = (kW - w) / 2, y = 150;
    color(10, 8, 6, 225);
    fillR(x, y, w, 160, 12);
    color(ui::kAccent.line);
    outlineR(x, y, w, 160, 12, 1);
    color(245, 240, 228);
    textCentered(kW / 2, y + 24, 26, "No game found", true);
    int ty = y + 66;
    for (const std::string& l : wrap(kNoGame, 18, w - 56)) textCentered(kW / 2, ty, 18, l), ty += 25;
    color(200, 192, 176);
    textCentered(kW / 2, y + 160 - 34, 15, font::fit(gameFolderPath(), 15, w - 56));
}

// The dialog Play opens with an empty game folder, over whichever screen is drawn.
void drawNoGameDialog() {
    SDL_SetRenderDrawBlendMode(gRen, SDL_BLENDMODE_BLEND);
    color(0, 0, 0, 170);
    fill(0, 0, kW, kH);
    const Box& d = kDialog;
    color(10, 8, 6, 245);
    fillR(d.x, d.y, d.w, d.h, 12);
    color(ui::kAccent.line);
    outlineR(d.x, d.y, d.w, d.h, 12, 1);
    color(245, 240, 228);
    text(d.x + 28, d.y + 22, 24, "No game found", true);
    int y = d.y + 64;
    for (const std::string& l : wrap(kNoGame, 18, d.w - 56)) text(d.x + 28, y, 18, l), y += 25;
    color(200, 192, 176);
    text(d.x + 28, y + 6, 15, font::fit(gameFolderPath(), 15, d.w - 56));
    button(kDialogOk, "OK", true, true);
}

void drawMain(int focus) {
    drawBackground(false);
    button(kPlay, "Play", focus == 0, true, canPlay());
    button(kSettings, "Settings", focus == 1, false);
    checkbox(kSkip, "Skip intro logos", gS.skipIntro, focus == 2);
    // Check for updates, top right, with what it found under it.
    const launcher::UpdateStatus u = launcher::updateStatus();
    const bool busy = u.state == launcher::UpdateState::Checking || u.state == launcher::UpdateState::Downloading;
    const bool canApply = u.state == launcher::UpdateState::Available && launcher::canSelfUpdate();
    button(kUpdate, canApply ? "Update to " + u.latest : "Check for updates", focus == 3, canApply, !busy);
    SDL_SetRenderDrawBlendMode(gRen, SDL_BLENDMODE_BLEND);
    const std::string line = u.message.empty() ? std::string("Version ") + launcher::version() : u.message;
    const std::string fitted = font::fit(line, 16, 520);
    const int lw = font::width(fitted, 16);
    color(0, 0, 0, 160);
    fillR(kUpdate.x + kUpdate.w - lw - 12, kUpdate.y + kUpdate.h + 8, lw + 12, 24, 6);
    color(245, 240, 228, 230);
    text(kUpdate.x + kUpdate.w - lw - 6, kUpdate.y + kUpdate.h + 12, 16, fitted);
    gearButton(kGear, focus == 4);
    padHints(40, 24, "Quit");
    std::string status = gStatus;
    if (gUnpack->state == Unpack::Working) {
        const uint64_t total = gUnpack->progress.total, done = gUnpack->progress.done;
        std::string step;
        {
            std::lock_guard lock(gUnpack->progress.mutex);
            step = gUnpack->progress.step;
        }
        status = (step.empty() ? std::string("Working") : step) + ": " + std::to_string(total ? int(done * 100 / total) : 0) + "%";
    }
    if (gGameDir.empty() && gUnpack->state != Unpack::Working) drawNoGame();
    if (!status.empty()) {
        color(245, 240, 228, 220);
        const std::string s = font::fit(status, 17, kGear.x - 16 - 380);
        text(kGear.x - 16 - font::width(s, 17), kGear.y + (kGear.h - 17) / 2, 17, s);
    }
}

void drawTools(int focus) {
    drawBackground(true);
    const Box& p = kToolsPanel;
    color(10, 8, 6, 225);
    fillR(p.x, p.y, p.w, p.h, 12);
    color(ui::kAccent.line);
    outlineR(p.x, p.y, p.w, p.h, 12, 1);
    color(245, 240, 228);
    text(p.x + 28, p.y + 22, 30, "Tools", true);
    color(ui::kAccent.line);
    fill(p.x + 28, p.y + 62, p.w - 56, 2);

    const std::shared_ptr<Unpack> u = gUnpack;
    const int state = u->state;
    const bool busy = state == Unpack::Choosing || state == Unpack::Working;
    const char* labels[3] = {"Choose .pkg file", "From .pkg file", "From folder"};
    const char* headings[2] = {"Unpack a .pkg", "Apply an update or DLC"};
    const char* lines[2] = {"Extracts .pkg file", "Adds it to the game version it is for"};
    for (int s = 0; s < 2; ++s) {
        const int y = kToolButtons[s == 0 ? 0 : 1].y;
        color(245, 240, 228);
        text(p.x + 28, y - 62, 22, headings[s], true);
        color(200, 192, 176);
        text(p.x + 28, y - 32, 17, lines[s]);
    }
    for (int b = 0; b < 3; ++b) {
        const bool mine = busy && u->button == b;
        const char* label = mine ? (state == Unpack::Working ? "Cancel" : "Choosing...") : labels[b];
        button(kToolButtons[b], label, focus == b, false, !busy || (mine && state == Unpack::Working));
    }
    if (!busy) {
        color(200, 192, 176);
        const Box& b = kToolButtons[0];
        text(b.x + b.w + 20, b.y + (b.h - 17) / 2, 17, "or drop a .pkg or folder on this window");
    }

    const int x = p.x + 28, w = p.w - 56;
    int y = kToolButtons[1].y + kToolButtons[1].h + 26;
    if (state == Unpack::Working) {
        const uint64_t total = u->progress.total, done = u->progress.done;
        const int pct = total ? int(std::min<uint64_t>(100, done * 100 / total)) : 0;
        std::string step, file;
        {
            std::lock_guard lock(u->progress.mutex);
            step = u->progress.step, file = u->progress.file;
        }
        color(245, 240, 228);
        text(x, y, 19, font::fit((step.empty() ? std::string("Working") : step) + ": " + std::to_string(pct) + "%", 19, w, true), true);
        color(0, 0, 0, 170);
        fillR(x, y + 32, w, 14, 7);
        color(ui::kAccent.fill);
        if (pct > 0) fillR(x, y + 32, std::max(14, w * pct / 100), 14, 7);
        color(ui::kAccent.line);
        outlineR(x, y + 32, w, 14, 7, 1);
        color(200, 192, 176);
        text(x, y + 56, 15, font::fit(file, 15, w));
    } else if (state == Unpack::Finished) {
        std::string message;
        bool ok;
        {
            std::lock_guard lock(u->mutex);
            message = u->message, ok = u->ok;
        }
        if (ok) color(ui::kAccent.bright);
        else color(245, 240, 228);
        int n = 0;
        for (const std::string& l : wrap(message, 18, w))
            if (n++ < 4) text(x, y, 18, font::fit(l, 18, w)), y += 25;
    }
    button(kToolsBack, "Back", focus == 3, false);
    padHints(40, kToolsBack.y + (kToolsBack.h - 40) / 2, "Back");
}

// Which game version to play, when game/ holds more than one.
std::vector<host::GameVersion> gVersions;

void drawVersions(int focus) {
    drawBackground(true);
    color(10, 8, 6, 225);
    fillR(kPanel.x, kPanel.y, kPanel.w, kPanel.h, 12);
    color(ui::kAccent.line);
    outlineR(kPanel.x, kPanel.y, kPanel.w, kPanel.h, 12, 1);
    color(245, 240, 228);
    text(kPanel.x + 28, kPanel.y + 22, 30, "Choose a version", true);
    color(ui::kAccent.line);
    fill(kPanel.x + 28, kPanel.y + 62, kPanel.w - 56, 2);
    const int n = std::min<int>(int(gVersions.size()), kMaxVersions);
    for (int r = 0; r < n; ++r) {
        const host::GameVersion& g = gVersions[size_t(r)];
        const Box b = rowBox(r);
        if (focus == r) {
            color(ui::kAccent.line, 60);
            fillR(b.x, b.y, b.w, b.h, 6);
            color(ui::kAccent.line);
            fillR(b.x, b.y, 4, b.h, 2);
        }
        std::string right = g.version.empty() ? std::string() : "Version " + g.version;
        if (g.category.rfind("gp", 0) == 0) right += right.empty() ? "Updated" : " (updated)";
        const int rw = font::width(right, 20);
        color(245, 240, 228);
        text(b.x + 18, b.y + (b.h - 22) / 2, 22, font::fit(g.name, 22, b.w - rw - 60, focus == r), focus == r);
        color(200, 192, 176);
        text(b.x + b.w - 18 - rw, b.y + (b.h - 20) / 2, 20, right);
    }
    if (focus < n && !gVersions[size_t(focus)].title.empty()) {
        color(200, 192, 176);
        text(kPanel.x + 28, kPanel.y + kPanel.h - 46, 17,
             font::fit(gVersions[size_t(focus)].title + "  -  game/" + gVersions[size_t(focus)].name, 17, kPanel.w - 56));
    }
    button(kVersionsBack, "Back", focus == n, false);
    padHints(40, kBack.y + (kBack.h - 40) / 2, "Back");
}

void drawSettings(int focus) {
    drawBackground(true);
    color(10, 8, 6, 225);
    fillR(kPanel.x, kPanel.y, kPanel.w, kPanel.h, 12);
    color(ui::kAccent.line);
    outlineR(kPanel.x, kPanel.y, kPanel.w, kPanel.h, 12, 1);
    color(245, 240, 228);
    text(kPanel.x + 28, kPanel.y + 22, 30, "PC Settings", true);
    color(ui::kAccent.line);
    fill(kPanel.x + 28, kPanel.y + 62, kPanel.w - 56, 2);
    for (int r = 0; r < kRowCount; ++r) {
        const Box b = rowBox(r);
        if (focus == r) {
            color(ui::kAccent.line, 60);
            fillR(b.x, b.y, b.w, b.h, 6);
            color(ui::kAccent.line);
            fillR(b.x, b.y, 4, b.h, 2);
        }
        color(245, 240, 228);
        text(b.x + 18, b.y + (b.h - 22) / 2, 22, kRows[r].label, focus == r);
        const Box l = arrowBox(r, false), rr = arrowBox(r, true);
        color(ui::kAccent.line);
        textCentered(l.x + l.w / 2, l.y + (l.h - 24) / 2, 24, "<", true);
        textCentered(rr.x + rr.w / 2, rr.y + (rr.h - 24) / 2, 24, ">", true);
        color(245, 240, 228);
        const int mid = (l.x + l.w + rr.x) / 2;
        textCentered(mid, b.y + (b.h - 22) / 2, 22, valueName(r), true);
    }
    if (focus < kRowCount) {
        color(200, 192, 176);
        std::string d = kRows[focus].description;
        if (focus == 1 || focus == 2 || focus == 5) d += " Takes effect at the next start.";
        // Up to two lines, broken between words.
        const int maxW = kPanel.w - 56;
        std::string first = d, rest;
        while (font::width(first, 17) > maxW) {
            const size_t sp = first.rfind(' ');
            if (sp == std::string::npos) break;
            rest = first.substr(sp + 1) + (rest.empty() ? "" : " " + rest);
            first.resize(sp);
        }
        const int y = kPanel.y + kPanel.h - (rest.empty() ? 46 : 60);
        text(kPanel.x + 28, y, 17, font::fit(first, 17, maxW));
        if (!rest.empty()) text(kPanel.x + 28, y + 23, 17, font::fit(rest, 17, maxW));
    }
    button(kDefaults, "Defaults", focus == kRowCount, false);
    button(kBack, "Back", focus == kRowCount + 1, false);
    padHints(40, kBack.y + (kBack.h - 40) / 2, "Back");
}

// Play: hand over to lpn-host in the install folder, the log to play.log, as launch.sh does.
void play(SDL_Window* win, int argc, char** argv) {
    std::vector<std::string> args = {gHost, gGameDir, "--nids", gNids, "--data", "lpn-data"};
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
    std::vector<char*> cargs;
    for (auto& a : args) cargs.push_back(a.data());
    cargs.push_back(nullptr);
    SDL_DestroyRenderer(gRen);
    SDL_DestroyWindow(win);
    SDL_Quit();
#ifdef _WIN32
    // No exec on Windows: start lpn-host with its output in play.log, and leave.
    (void)cargs;
    std::string cmd;
    for (const std::string& a : args) {
        std::string q = "\"";
        for (char c : a) q += c == '"' ? std::string("\\\"") : std::string(1, c);
        cmd += (cmd.empty() ? "" : " ") + q + "\"";
    }
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE log = CreateFileA("play.log", GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOA si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = log;
    si.hStdError = log;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(gHost.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        const std::string why = "Cannot start " + gHost + " (error " + std::to_string(GetLastError()) + ")";
        MessageBoxA(nullptr, why.c_str(), "Lost Paradise", MB_ICONERROR);
        std::exit(1);
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    std::exit(0);
#else
    const int log = open("play.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log >= 0) dup2(log, 2), close(log);
    execv(gHost.c_str(), cargs.data());
    std::fprintf(stderr, "[launcher] cannot start %s: %s\n", gHost.c_str(), std::strerror(errno));
    std::exit(1);
#endif
}

// Check for updates, or apply the update a check found.
void updateButton() {
    const launcher::UpdateStatus u = launcher::updateStatus();
    if (u.state == launcher::UpdateState::Available) launcher::applyUpdate();
    else launcher::checkForUpdates(gInstall.string());
}

} // namespace

int main(int argc, char** argv) {
    gExeDir = exeDir();
    gInstall = installDir();
    std::error_code ec;
    fs::current_path(gInstall, ec);
    if (ec) std::fprintf(stderr, "[launcher] cannot enter %s: %s\n", gInstall.string().c_str(), ec.message().c_str());
    launcher::cleanUpAfterUpdate(gInstall.string());
    findFiles();
    host::readSettings(host::kSettingsFile, gS);
    refreshStatus();
    std::fprintf(stderr, "[launcher] install %s, game %s, host %s, nids %s\n", gInstall.string().c_str(),
                 gGameDir.empty() ? "(none)" : gGameDir.c_str(), gHost.empty() ? "(none)" : gHost.c_str(),
                 gNids.empty() ? "(none)" : gNids.c_str());

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) {
        std::fprintf(stderr, "[launcher] SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* win = SDL_CreateWindow("Lost Paradise", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, kW, kH, 0);
    if (!win) {
        std::fprintf(stderr, "[launcher] cannot open a window: %s\n", SDL_GetError());
        return 1;
    }
    gRen = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!gRen) gRen = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    loadArt(win);
    SDL_GameController* pad = nullptr;

    enum Screen { Main, Settings, Tools, Versions } screen = Main;
    // main: 0 play, 1 settings, 2 skip, 3 update, 4 tools; settings: rows, defaults, back;
    // tools: 0 unpack, 1 update from .pkg, 2 update from folder, 3 back; versions: rows, back
    int focus = canPlay() ? 0 : 1;
    // For docs: "main.bmp", "settings:x.bmp", "tools:x.bmp" or "versions:x.bmp"
    // (LPN_LAUNCHER_UNPACK=<file.pkg or folder> to show one being used).
    const char* shot = std::getenv("LPN_LAUNCHER_SCREENSHOT");
    if (shot && !std::strncmp(shot, "settings:", 9)) screen = Settings, focus = 0, shot += 9;
    if (shot && !std::strncmp(shot, "tools:", 6)) screen = Tools, focus = 0, shot += 6;
    if (shot && !std::strncmp(shot, "versions:", 9))
        screen = Versions, focus = 0, shot += 9, gVersions = host::findGames(gInstall.string());
    if (shot && !std::strncmp(shot, "nogame:", 7)) gNoGameDialog = true, focus = 0, shot += 7;
    if (const char* p = std::getenv("LPN_LAUNCHER_UNPACK"); p && *p) screen = Tools, focus = 0, startJob(p, false, 0);
    auto versionCount = [] { return std::min<int>(int(gVersions.size()), kMaxVersions); };
    auto toolButton = [&](int b) {
        if (unpackBusy()) {
            if (gUnpack->button == b && gUnpack->state == Unpack::Working) gUnpack->progress.cancel = true;
        } else {
            startJob({}, b == 2, b);
        }
    };
    // Play: straight in with one game version, else the version screen.
    auto play_ = [&]() -> bool {
        if (!tryPlay()) {
            if (gGameDir.empty() && canPlay()) gNoGameDialog = true;
            return false;
        }
        gVersions = host::findGames(gInstall.string());
        if (gVersions.size() < 2) return true;
        screen = Versions, focus = 0;
        for (int i = 0; i < versionCount(); ++i)
            if (gVersions[size_t(i)].dir == gGameDir) focus = i;
        return false;
    };
    auto pickVersion = [&](int i) -> bool {
        if (i < 0 || i >= versionCount()) return false;
        gGameDir = gVersions[size_t(i)].dir;
        return true;
    };
    auto confirm = [&]() -> bool {
        if (screen == Main) {
            if (focus == 0 && canPlay()) return play_();
            if (focus == 1) screen = Settings, focus = 0;
            if (focus == 2) gS.skipIntro = !gS.skipIntro, save();
            if (focus == 3) updateButton();
            if (focus == 4) screen = Tools, focus = 0;
        } else if (screen == Tools) {
            if (focus < 3) toolButton(focus);
            else screen = Main, focus = 4;
        } else if (screen == Versions) {
            if (focus < versionCount()) return pickVersion(focus);
            screen = Main, focus = 0;
        } else if (focus < kRowCount) stepValue(focus, +1);
        else if (focus == kRowCount) gS = DisplaySettings{}, save();
        else screen = Main, focus = 1;
        return false;
    };
    auto back = [&] {
        if (screen == Settings) screen = Main, focus = 1;
        else if (screen == Tools) screen = Main, focus = 4;
        else if (screen == Versions) screen = Main, focus = 0;
    };
    auto move = [&](int d) {
        const int n = screen == Settings ? kRowCount + 2 : screen == Tools ? 4 : screen == Versions ? versionCount() + 1 : 5;
        focus = ((focus + d) % n + n) % n;
        if (screen == Main && focus == 0 && !canPlay()) focus = (focus + d + n) % n;
    };
    const Unpack* seen = nullptr; // the finished unpack the game folder was looked at again for
    for (bool run = true; run;) {
        SDL_Event e;
        bool start = false;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) run = false;
            else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
                // The game may have changed them since (its PC Settings).
                gS = DisplaySettings{};
                host::readSettings(host::kSettingsFile, gS);
            }
            else if (e.type == SDL_CONTROLLERDEVICEADDED && !pad) pad = SDL_GameControllerOpen(e.cdevice.which);
            else if (gNoGameDialog && e.type != SDL_DROPFILE) {
                // The dialog takes the input until it is closed: OK, Enter, Escape, A or B.
                if (e.type == SDL_MOUSEBUTTONDOWN && (e.button.button == SDL_BUTTON_RIGHT || kDialogOk.in(e.button.x, e.button.y)))
                    gNoGameDialog = false;
                else if (e.type == SDL_KEYDOWN) {
                    const SDL_Keycode k = e.key.keysym.sym;
                    if (k == SDLK_RETURN || k == SDLK_KP_ENTER || k == SDLK_SPACE || k == SDLK_ESCAPE || k == SDLK_BACKSPACE)
                        gNoGameDialog = false;
                } else if (e.type == SDL_CONTROLLERBUTTONDOWN) {
                    const int b = e.cbutton.button;
                    if (b == SDL_CONTROLLER_BUTTON_A || b == SDL_CONTROLLER_BUTTON_B || b == SDL_CONTROLLER_BUTTON_START)
                        gNoGameDialog = false;
                }
            }
            else if (e.type == SDL_DROPFILE) {
                const std::string path = e.drop.file;
                SDL_free(e.drop.file);
                std::error_code ec;
                const bool dir = fs::is_directory(path, ec);
                if ((isPkg(path) || dir) && !unpackBusy()) gNoGameDialog = false, screen = Tools, focus = dir ? 2 : 0, startJob(path, dir, dir ? 2 : 0);
            }
            else if (e.type == SDL_MOUSEMOTION) {
                const int x = e.motion.x, y = e.motion.y;
                if (screen == Main) {
                    if (kPlay.in(x, y) && canPlay()) focus = 0;
                    else if (kSettings.in(x, y)) focus = 1;
                    else if (kSkip.in(x, y)) focus = 2;
                    else if (kUpdate.in(x, y)) focus = 3;
                    else if (kGear.in(x, y)) focus = 4;
                } else if (screen == Tools) {
                    for (int b = 0; b < 3; ++b)
                        if (kToolButtons[b].in(x, y)) focus = b;
                    if (kToolsBack.in(x, y)) focus = 3;
                } else if (screen == Versions) {
                    for (int r = 0; r < versionCount(); ++r)
                        if (rowBox(r).in(x, y)) focus = r;
                    if (kVersionsBack.in(x, y)) focus = versionCount();
                } else {
                    for (int r = 0; r < kRowCount; ++r)
                        if (rowBox(r).in(x, y)) focus = r;
                    if (kDefaults.in(x, y)) focus = kRowCount;
                    if (kBack.in(x, y)) focus = kRowCount + 1;
                }
            } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                const int x = e.button.x, y = e.button.y;
                if (screen == Main) {
                    if (kPlay.in(x, y) && canPlay()) start = play_();
                    else if (kSettings.in(x, y)) screen = Settings, focus = 0;
                    else if (kSkip.in(x, y)) gS.skipIntro = !gS.skipIntro, save();
                    else if (kUpdate.in(x, y)) updateButton();
                    else if (kGear.in(x, y)) screen = Tools, focus = 0;
                } else if (screen == Tools) {
                    for (int b = 0; b < 3; ++b)
                        if (kToolButtons[b].in(x, y)) toolButton(b);
                    if (kToolsBack.in(x, y)) screen = Main, focus = 4;
                } else if (screen == Versions) {
                    for (int r = 0; r < versionCount(); ++r)
                        if (rowBox(r).in(x, y)) start = pickVersion(r);
                    if (kVersionsBack.in(x, y)) screen = Main, focus = 0;
                } else {
                    for (int r = 0; r < kRowCount; ++r)
                        if (rowBox(r).in(x, y)) stepValue(r, arrowBox(r, false).in(x, y) ? -1 : +1);
                    if (kDefaults.in(x, y)) gS = DisplaySettings{}, save();
                    if (kBack.in(x, y)) screen = Main, focus = 1;
                }
            } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT) {
                back();
            } else if (e.type == SDL_KEYDOWN || e.type == SDL_CONTROLLERBUTTONDOWN) {
                int key = 0; // 1 up, 2 down, 3 left, 4 right, 5 confirm, 6 back
                if (e.type == SDL_KEYDOWN) {
                    switch (e.key.keysym.sym) {
                    case SDLK_UP: key = 1; break;
                    case SDLK_DOWN: case SDLK_TAB: key = 2; break;
                    case SDLK_LEFT: key = 3; break;
                    case SDLK_RIGHT: key = 4; break;
                    case SDLK_RETURN: case SDLK_KP_ENTER: case SDLK_SPACE: key = 5; break;
                    case SDLK_ESCAPE: case SDLK_BACKSPACE: key = 6; break;
                    }
                } else {
                    switch (e.cbutton.button) {
                    case SDL_CONTROLLER_BUTTON_DPAD_UP: key = 1; break;
                    case SDL_CONTROLLER_BUTTON_DPAD_DOWN: key = 2; break;
                    case SDL_CONTROLLER_BUTTON_DPAD_LEFT: key = 3; break;
                    case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: key = 4; break;
                    case SDL_CONTROLLER_BUTTON_A: case SDL_CONTROLLER_BUTTON_START: key = 5; break;
                    case SDL_CONTROLLER_BUTTON_B: key = 6; break;
                    }
                }
                if (key == 1) move(-1);
                else if (key == 2) move(+1);
                else if ((key == 3 || key == 4) && screen == Settings && focus < kRowCount)
                    stepValue(focus, key == 3 ? -1 : +1);
                else if (key == 3 || key == 4) move(key == 3 ? -1 : +1);
                else if (key == 5) start = confirm();
                else if (key == 6) {
                    if (screen != Main) back();
                    else run = false;
                }
            }
        }
        if (!run) break;
        // A Tools job just finished: the game, or a new version of it, may be there now.
        if (gUnpack->state == Unpack::Finished && seen != gUnpack.get()) {
            seen = gUnpack.get();
            findFiles();
            refreshStatus();
            if (!gArt) loadArt(win);
            if (screen == Main && canPlay()) focus = 0;
        }
        if (shot && std::getenv("LPN_LAUNCHER_UNPACK")) {
            // Wait for some of it to be done, to show the progress bar.
            while (unpackBusy() && (!gUnpack->progress.total || gUnpack->progress.done * 3 < gUnpack->progress.total))
                SDL_Delay(1);
        }
        if (screen == Settings) drawSettings(focus);
        else if (screen == Tools) drawTools(focus);
        else if (screen == Versions) drawVersions(focus);
        else drawMain(focus);
        if (gNoGameDialog) drawNoGameDialog();
        if (shot) {
            // Read back what was drawn and save it, then quit (for screenshots in docs and tests).
            SDL_Surface* s = SDL_CreateRGBSurfaceWithFormat(0, kW, kH, 32, SDL_PIXELFORMAT_ARGB8888);
            SDL_RenderReadPixels(gRen, nullptr, SDL_PIXELFORMAT_ARGB8888, s->pixels, s->pitch);
            SDL_SaveBMP(s, shot);
            SDL_FreeSurface(s);
            break;
        }
        SDL_RenderPresent(gRen);
        if (start) play(win, argc, argv);
        if (launcher::restartReady()) {
            SDL_DestroyRenderer(gRen);
            SDL_DestroyWindow(win);
            SDL_Quit();
            launcher::restart(argc, argv);
        }
    }
    // Stop an unpack so it does not leave a .part folder behind (the chooser is left to close).
    if (gUnpack->state == Unpack::Working) {
        gUnpack->progress.cancel = true;
        for (int i = 0; i < 1000 && gUnpack->state == Unpack::Working; ++i) SDL_Delay(10);
    }
    if (pad) SDL_GameControllerClose(pad);
    SDL_DestroyRenderer(gRen);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
