// lpn-launcher: a small window with Play and Settings in front of lpn-host.
//
// It runs in the install folder: the folder the AppImage is in ($APPIMAGE), else $LPN_HOME, else
// the working directory. That folder holds game/CUSA12776 (common/game_dir.h), lpn-data,
// lpn-display.ini and play.log, as for launch.sh. The settings are the in-game PC Settings: both
// read and write lpn-display.ini (common/settings_file.h), so a change in either shows in the other.
// Play replaces the launcher with lpn-host, its log going to play.log.
#include "common/game_dir.h"
#include "common/settings_file.h"
#include "host/menu_font.h"

#include <SDL.h>
#include <stb_image.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace font = host::menufont;
using host::DisplaySettings;

namespace {

constexpr int kW = 1152, kH = 648;

// Paths
fs::path gInstall, gExeDir;
std::string gGameDir, gHost, gNids;
bool gLegacyGame = false;
DisplaySettings gS;

fs::path exeDir() {
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::current_path() : p.parent_path();
}

fs::path installDir() {
    if (const char* a = std::getenv("APPIMAGE"); a && *a) return fs::absolute(a).parent_path();
    if (const char* h = std::getenv("LPN_HOME"); h && *h) return fs::absolute(h);
    return fs::current_path();
}

std::string firstExisting(std::initializer_list<fs::path> paths) {
    std::error_code ec;
    for (const fs::path& p : paths)
        if (fs::is_regular_file(p, ec)) return fs::absolute(p).lexically_normal().string();
    return {};
}

void findFiles() {
    gGameDir = host::findGameDir(gInstall.string(), &gLegacyGame);
    // In the AppImage: usr/bin/lpn-launcher, lpn-host beside it and usr/share/lost-paradise.
    // In a build tree: build/src/launcher, build/src/host and the repo's aerolib.csv.
    gHost = firstExisting({gExeDir / "lpn-host", gExeDir / "../host/lpn-host",
                           gInstall / "lost-paradise/build/src/host/lpn-host"});
    const char* env = std::getenv("LPN_NIDS");
    gNids = firstExisting({env ? fs::path(env) : fs::path(), gExeDir / "../share/lost-paradise/aerolib.csv",
                           gExeDir / "aerolib.csv", gExeDir / "../../../aerolib.csv",
                           gInstall / "lost-paradise/aerolib.csv"});
}

void save() {
    if (!host::writeSettings(host::kSettingsFile, gS))
        std::fprintf(stderr, "[launcher] cannot write %s/%s\n", gInstall.c_str(), host::kSettingsFile);
}

// Settings rows: the in-game PC Settings, in its order (src/host/pc_settings.cpp).
struct Row {
    const char* label;
    const char* description;
};
const Row kRows[] = {
    {"Window Mode", "Play in a window, a borderless window or fullscreen."},
    {"Resolution", "Set the 3D rendering resolution."},
    {"Frame Rate Cap", "Set the most frames shown per second. Uncapped follows the monitor's refresh rate."},
    {"FPS Counter", "Show the frame rate in the top right corner of the screen."},
    {"DLSS", "NVIDIA DLAA: DLSS anti-aliasing in place of the game's. Needs an RTX card; Off on other GPUs."},
    {"Skip Intro Logos", "Go straight to the title screen when the game starts."},
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
    case 2: return gS.fps ? std::to_string(gS.fps) : "Uncapped";
    case 3: return gS.fpsCounter ? "On" : "Off";
    case 4: return gS.dlss != host::DlssMode::Off ? "DLAA" : "Off";
    default: return gS.skipIntro ? "On" : "Off";
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
    case 2: {
        const int i = fpsIndex();
        gS.fps = host::kFpsChoices[wrap(i < 0 ? (d > 0 ? 0 : -1) : i + d, host::kFpsCount)];
        break;
    }
    case 3: gS.fpsCounter = !gS.fpsCounter; break;
    case 4: gS.dlss = gS.dlss == host::DlssMode::Off ? host::DlssMode::Dlaa : host::DlssMode::Off; break;
    default: gS.skipIntro = !gS.skipIntro; break;
    }
    save();
}

// Drawing
SDL_Renderer* gRen = nullptr;
SDL_Texture* gArt = nullptr; // the game's own key art (sce_sys/pic1.png), when there is a game
int gArtW = 0, gArtH = 0;

void color(Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255) { SDL_SetRenderDrawColor(gRen, r, g, b, a); }
void fill(int x, int y, int w, int h) {
    SDL_Rect r{x, y, w, h};
    SDL_RenderFillRect(gRen, &r);
}
void outline(int x, int y, int w, int h, int t) {
    fill(x, y, w, t), fill(x, y + h - t, w, t), fill(x, y, t, h), fill(x + w - t, y, t, h);
}
void text(int x, int y, int size, const std::string& s, bool bold = false) { font::draw(gRen, x, y, size, s, bold); }
void textCentered(int cx, int y, int size, const std::string& s, bool bold = false) {
    text(cx - font::width(s, size, bold) / 2, y, size, s, bold);
}

SDL_Surface* loadPng(const fs::path& p) {
    int w, h, n;
    unsigned char* px = stbi_load(p.c_str(), &w, &h, &n, 4);
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
// Settings screen
const Box kPanel{176, 64, kW - 352, 470};
constexpr int kRowY0 = 140, kRowH = 52;
Box rowBox(int r) { return {kPanel.x + 24, kRowY0 + r * kRowH, kPanel.w - 48, kRowH - 6}; }
Box arrowBox(int r, bool right) {
    const Box b = rowBox(r);
    return right ? Box{b.x + b.w - 44, b.y, 44, b.h} : Box{b.x + b.w - 330, b.y, 44, b.h};
}
const Box kBack{kW / 2 + 12, kPanel.y + kPanel.h + 20, 200, 48};
const Box kDefaults{kW / 2 - 212, kPanel.y + kPanel.h + 20, 200, 48};

void button(const Box& b, const std::string& label, bool focus, bool primary, bool enabled = true) {
    if (primary && enabled) color(focus ? 236 : 214, focus ? 204 : 182, focus ? 120 : 98, 255);
    else color(0, 0, 0, focus ? 200 : 150);
    fill(b.x, b.y, b.w, b.h);
    color(214, 182, 98, enabled ? 255 : 90);
    outline(b.x, b.y, b.w, b.h, focus ? 3 : 1);
    const int size = b.h * 9 / 20;
    if (primary && enabled) color(20, 14, 8);
    else color(245, 240, 228, enabled ? 255 : 110);
    textCentered(b.x + b.w / 2, b.y + (b.h - size) / 2, size, label, true);
}

void checkbox(const Box& b, const std::string& label, bool on, bool focus) {
    const int s = b.h - 6;
    color(0, 0, 0, 170);
    fill(b.x, b.y + 3, s, s);
    color(214, 182, 98);
    outline(b.x, b.y + 3, s, s, focus ? 3 : 2);
    if (on) fill(b.x + 6, b.y + 9, s - 12, s - 12);
    color(245, 240, 228);
    text(b.x + s + 12, b.y + (b.h - 20) / 2, 20, label, focus);
}

std::string gStatus; // why Play cannot start, or a launch error

bool canPlay() { return !gGameDir.empty() && !gHost.empty() && !gNids.empty(); }

void updateStatus() {
    if (gGameDir.empty())
        gStatus = "Game not found: put the CUSA12776 folder in " + (gInstall / host::kGameFolder).string();
    else if (gHost.empty()) gStatus = "lpn-host not found beside the launcher";
    else if (gNids.empty()) gStatus = "aerolib.csv not found (scripts/fetch-nids.sh)";
    else if (gLegacyGame) gStatus = "Tip: move CUSA12776 into the game folder";
    else gStatus.clear();
}

void drawMain(int focus) {
    drawBackground(false);
    button(kPlay, "Play", focus == 0, true, canPlay());
    button(kSettings, "Settings", focus == 1, false);
    checkbox(kSkip, "Skip intro logos", gS.skipIntro, focus == 2);
    if (!gStatus.empty()) {
        color(245, 240, 228, 220);
        const std::string s = font::fit(gStatus, 17, kW - 420);
        text(kW - 40 - font::width(s, 17), kH - 46, 17, s);
    }
}

void drawSettings(int focus) {
    drawBackground(true);
    color(10, 8, 6, 225);
    fill(kPanel.x, kPanel.y, kPanel.w, kPanel.h);
    color(214, 182, 98);
    outline(kPanel.x, kPanel.y, kPanel.w, kPanel.h, 1);
    color(245, 240, 228);
    text(kPanel.x + 28, kPanel.y + 22, 30, "PC Settings", true);
    color(214, 182, 98);
    fill(kPanel.x + 28, kPanel.y + 62, kPanel.w - 56, 2);
    for (int r = 0; r < kRowCount; ++r) {
        const Box b = rowBox(r);
        if (focus == r) {
            color(214, 182, 98, 60);
            fill(b.x, b.y, b.w, b.h);
            color(214, 182, 98);
            fill(b.x, b.y, 4, b.h);
        }
        color(245, 240, 228);
        text(b.x + 18, b.y + (b.h - 22) / 2, 22, kRows[r].label, focus == r);
        const Box l = arrowBox(r, false), rr = arrowBox(r, true);
        color(214, 182, 98);
        textCentered(l.x + l.w / 2, l.y + (l.h - 24) / 2, 24, "<", true);
        textCentered(rr.x + rr.w / 2, rr.y + (rr.h - 24) / 2, 24, ">", true);
        color(245, 240, 228);
        const int mid = (l.x + l.w + rr.x) / 2;
        textCentered(mid, b.y + (b.h - 22) / 2, 22, valueName(r), true);
    }
    if (focus < kRowCount) {
        color(200, 192, 176);
        std::string d = kRows[focus].description;
        if (focus == 1 || focus == 4) d += " Takes effect at the next start.";
        text(kPanel.x + 28, kPanel.y + kPanel.h - 46, 17, font::fit(d, 17, kPanel.w - 56));
    }
    button(kDefaults, "Defaults", focus == kRowCount, false);
    button(kBack, "Back", focus == kRowCount + 1, false);
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
    const int log = open("play.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log >= 0) dup2(log, 2), close(log);
    execv(gHost.c_str(), cargs.data());
    std::fprintf(stderr, "[launcher] cannot start %s: %s\n", gHost.c_str(), std::strerror(errno));
    std::exit(1);
}

} // namespace

int main(int argc, char** argv) {
    gExeDir = exeDir();
    gInstall = installDir();
    std::error_code ec;
    fs::current_path(gInstall, ec);
    if (ec) std::fprintf(stderr, "[launcher] cannot enter %s: %s\n", gInstall.c_str(), ec.message().c_str());
    findFiles();
    host::readSettings(host::kSettingsFile, gS);
    updateStatus();
    std::fprintf(stderr, "[launcher] install %s, game %s, host %s, nids %s\n", gInstall.c_str(),
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

    bool settings = false;
    int focus = canPlay() ? 0 : 1; // main: 0 play, 1 settings, 2 skip; settings: rows, defaults, back
    const char* shot = std::getenv("LPN_LAUNCHER_SCREENSHOT"); // for docs: "main.bmp" or "settings:x.bmp"
    if (shot && !std::strncmp(shot, "settings:", 9)) settings = true, focus = 0, shot += 9;
    auto confirm = [&]() -> bool {
        if (!settings) {
            if (focus == 0 && canPlay()) return true;
            if (focus == 1) settings = true, focus = 0;
            if (focus == 2) gS.skipIntro = !gS.skipIntro, save();
        } else if (focus < kRowCount) stepValue(focus, +1);
        else if (focus == kRowCount) gS = DisplaySettings{}, save();
        else settings = false, focus = 1;
        return false;
    };
    auto back = [&] {
        if (settings) settings = false, focus = 1;
    };
    auto move = [&](int d) {
        const int n = settings ? kRowCount + 2 : 3;
        focus = ((focus + d) % n + n) % n;
        if (!settings && focus == 0 && !canPlay()) focus = (focus + d + n) % n;
    };
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
            else if (e.type == SDL_MOUSEMOTION) {
                const int x = e.motion.x, y = e.motion.y;
                if (!settings) {
                    if (kPlay.in(x, y) && canPlay()) focus = 0;
                    else if (kSettings.in(x, y)) focus = 1;
                    else if (kSkip.in(x, y)) focus = 2;
                } else {
                    for (int r = 0; r < kRowCount; ++r)
                        if (rowBox(r).in(x, y)) focus = r;
                    if (kDefaults.in(x, y)) focus = kRowCount;
                    if (kBack.in(x, y)) focus = kRowCount + 1;
                }
            } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                const int x = e.button.x, y = e.button.y;
                if (!settings) {
                    if (kPlay.in(x, y) && canPlay()) start = true;
                    else if (kSettings.in(x, y)) settings = true, focus = 0;
                    else if (kSkip.in(x, y)) gS.skipIntro = !gS.skipIntro, save();
                } else {
                    for (int r = 0; r < kRowCount; ++r)
                        if (rowBox(r).in(x, y)) stepValue(r, arrowBox(r, false).in(x, y) ? -1 : +1);
                    if (kDefaults.in(x, y)) gS = DisplaySettings{}, save();
                    if (kBack.in(x, y)) settings = false, focus = 1;
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
                else if ((key == 3 || key == 4) && settings && focus < kRowCount) stepValue(focus, key == 3 ? -1 : +1);
                else if ((key == 3 || key == 4) && !settings) move(key == 3 ? -1 : +1);
                else if (key == 5) start = confirm();
                else if (key == 6) {
                    if (settings) back();
                    else run = false;
                }
            }
        }
        if (!run) break;
        if (settings) drawSettings(focus);
        else drawMain(focus);
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
    }
    if (pad) SDL_GameControllerClose(pad);
    SDL_DestroyRenderer(gRen);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
