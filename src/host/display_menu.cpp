#include "host/display_menu.h"

#include "host/display_settings.h"
#include "host/gpu/renderer.h"
#include "host/hle.h"
#include "host/save_menu.h"
#include "host/videoout.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace host {

namespace {
std::atomic<bool> gOpen{false};
} // namespace

bool displayMenuOpen() { return gOpen; }

#ifdef LPN_HAVE_SDL2
namespace {

std::mutex gM;
int gRow = 0;         // 0 window mode, 1 resolution, 2 fps, 3 fps counter, 4 exit game
bool gList = false;   // a drop-down is open under gRow
int gItem = 0;        // highlighted drop-down entry

const char* const kModes[] = {"WINDOWED", "BORDERLESS FULLSCREEN", "FULLSCREEN"};
struct Res {
    uint32_t w, h;
    const char* label;
};
const Res kResolutions[] = {
    {1280, 720, "1280X720"},
    {0, 0, "1600X896 (PS4)"},
    {1920, 1080, "1920X1080 (PS4 PRO)"},
    {2560, 1440, "2560X1440"},
};
constexpr int kResCount = int(sizeof kResolutions / sizeof kResolutions[0]);
// The game runs at 60 and drops itself to 30 in cutscenes and 20 on loading screens. Above 60
// (90, or uncapped at the monitor's refresh rate) the frame clock is patched to keep its speed,
// and the pad history keeps one entry per 1/60 s so mashing moves still count (input_history.h).
// Uncapped, the 30 fps scenes (title screen, cutscenes) run at the monitor's rate too.
const int kFps[] = {30, 60, 90, 0};
constexpr int kFpsCount = int(sizeof kFps / sizeof kFps[0]);
constexpr int kRows = 5;

int resIndex(uint32_t w, uint32_t h) {
    for (int i = 0; i < kResCount; ++i)
        if (kResolutions[i].w == w && kResolutions[i].h == h) return i;
    return -1;
}

std::string resLabel(uint32_t w, uint32_t h) {
    const int i = resIndex(w, h);
    return i >= 0 ? kResolutions[i].label : std::to_string(w) + "X" + std::to_string(h);
}

int fpsIndex(int fps) {
    for (int i = 0; i < kFpsCount; ++i)
        if (kFps[i] == fps) return i;
    return -1;
}

std::string fpsLabel(int fps) { return fps ? std::to_string(fps) : "UNCAPPED"; }

int itemCount() { return gRow == 0 ? 3 : gRow == 1 ? kResCount : kFpsCount; }

std::string itemLabel(int i) {
    return gRow == 0 ? kModes[i] : gRow == 1 ? kResolutions[i].label : fpsLabel(kFps[i]);
}

void pick(SDL_Window* win) {
    DisplaySettings& s = displaySettings();
    if (gRow == 0) {
        s.window = WindowMode(gItem);
        applyWindowMode(win);
        applyFrameRate(win);
    } else if (gRow == 2) {
        s.fps = kFps[gItem];
        setFpsCap(s.fps);
        setFrameClock(s.fps);
        applyFrameRate(win);
        std::fprintf(stderr, "[display] fps cap %s\n", fpsLabel(s.fps).c_str());
    } else {
        s.width = kResolutions[gItem].w;
        s.height = kResolutions[gItem].h;
        std::fprintf(stderr, "[display] resolution %s after a restart\n", kResolutions[gItem].label);
    }
    saveDisplaySettings();
    gList = false;
}

void select(SDL_Window* win) {
    if (gList) return pick(win);
    if (gRow == 3) { // on or off: no drop-down
        DisplaySettings& s = displaySettings();
        s.fpsCounter = !s.fpsCounter;
        saveDisplaySettings();
        return;
    }
    if (gRow == 4) { // as closing the window does
        std::fprintf(stderr, "[window] exit from the display menu\n");
        printUnimplementedSummary();
        render::saveCaches();
        std::_Exit(0);
    }
    const DisplaySettings& s = displaySettings();
    gItem = gRow == 0   ? int(s.window)
            : gRow == 1 ? std::max(0, resIndex(s.width, s.height))
                        : std::max(0, fpsIndex(s.fps));
    gList = true;
}

void back() {
    if (gList) gList = false;
    else gOpen = false;
}

void move(int d) {
    if (gList) gItem = (gItem + d + itemCount()) % itemCount();
    else gRow = (gRow + d + kRows) % kRows;
}

} // namespace

void applyWindowMode(SDL_Window* win) {
    const WindowMode m = displaySettings().window;
    if (m == WindowMode::Fullscreen) {
        // Exclusive fullscreen at the desktop's own mode, not the window's size.
        SDL_DisplayMode dm;
        if (SDL_GetDesktopDisplayMode(SDL_GetWindowDisplayIndex(win), &dm) == 0) SDL_SetWindowDisplayMode(win, &dm);
    }
    const Uint32 flags = m == WindowMode::Fullscreen   ? SDL_WINDOW_FULLSCREEN
                         : m == WindowMode::Borderless ? SDL_WINDOW_FULLSCREEN_DESKTOP
                                                       : 0;
    if (SDL_SetWindowFullscreen(win, flags) != 0)
        std::fprintf(stderr, "[display] cannot change the window mode: %s\n", SDL_GetError());
}

void applyFrameRate(SDL_Window* win) {
    // Uncapped shows a frame at every refresh of the monitor the window is on.
    SDL_DisplayMode dm;
    // A cap above 60 runs the clock at that rate, so every vblank shows a frame.
    const int fps = displaySettings().fps;
    if (fps > 61)
        setDisplayRefresh(fps);
    else if (fps == 0 && SDL_GetCurrentDisplayMode(SDL_GetWindowDisplayIndex(win), &dm) == 0 && dm.refresh_rate > 60)
        setDisplayRefresh(dm.refresh_rate);
    else
        setDisplayRefresh(0);
}

bool displayMenuEvent(const SDL_Event& e, SDL_Window* win) {
    if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_MOVED) applyFrameRate(win); // another monitor?
    if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F11) {
        if (!e.key.repeat && !saveMenuOpen()) {
            std::lock_guard lock(gM);
            gOpen = !gOpen;
            gList = false;
        }
        return true;
    }
    if (!gOpen) return false;
    std::lock_guard lock(gM);
    if (e.type == SDL_KEYDOWN) {
        switch (e.key.keysym.scancode) {
        case SDL_SCANCODE_UP: case SDL_SCANCODE_W: move(-1); break;
        case SDL_SCANCODE_DOWN: case SDL_SCANCODE_S: move(1); break;
        case SDL_SCANCODE_RETURN: case SDL_SCANCODE_KP_ENTER: case SDL_SCANCODE_SPACE:
            if (!e.key.repeat) select(win);
            break;
        case SDL_SCANCODE_BACKSPACE: case SDL_SCANCODE_ESCAPE:
            if (!e.key.repeat) back();
            break;
        default: break;
        }
        return true; // the game gets no keys while the menu is open
    }
    if (e.type == SDL_CONTROLLERBUTTONDOWN) {
        switch (e.cbutton.button) {
        case SDL_CONTROLLER_BUTTON_DPAD_UP: move(-1); break;
        case SDL_CONTROLLER_BUTTON_DPAD_DOWN: move(1); break;
        case SDL_CONTROLLER_BUTTON_A: select(win); break;
        case SDL_CONTROLLER_BUTTON_B: case SDL_CONTROLLER_BUTTON_START: back(); break;
        default: break;
        }
        return true;
    }
    return false;
}

void drawDisplayMenu(SDL_Renderer* ren) {
    if (!gOpen) return;
    std::lock_guard lock(gM);
    const DisplaySettings& s = displaySettings();
    uint32_t activeW, activeH;
    activeRenderResolution(&activeW, &activeH);
    const bool restart = activeW != s.width || activeH != s.height;

    int w = 0, h = 0;
    SDL_GetRendererOutputSize(ren, &w, &h);
    const int px = std::max(1, h / 240);
    const int line = 12 * px;
    const std::string values[4] = {kModes[int(s.window)], resLabel(s.width, s.height), fpsLabel(s.fps),
                                   s.fpsCounter ? "ON" : "OFF"};
    const char* const labels[4] = {"WINDOW MODE", "RESOLUTION", "FPS", "FPS COUNTER"};
    const int listRows = gList ? itemCount() : 0;
    const int panelW = std::min(w - 4 * px, 230 * px), panelH = (9 + listRows + (restart ? 1 : 0)) * line;
    const int x0 = (w - panelW) / 2, y0 = (h - panelH) / 2;
    const int xv = x0 + 6 * px + menuTextWidth("FPS COUNTER  ", px);

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_Rect panel{x0, y0, panelW, panelH};
    SDL_SetRenderDrawColor(ren, 20, 20, 28, 235);
    SDL_RenderFillRect(ren, &panel);
    SDL_SetRenderDrawColor(ren, 200, 170, 90, 255);
    SDL_RenderDrawRect(ren, &panel);
    drawMenuText(ren, x0 + 6 * px, y0 + 4 * px, px, "DISPLAY");

    int y = y0 + 2 * line;
    for (int r = 0; r < 4; ++r, y += line) {
        if (r == gRow) {
            SDL_Rect hl{x0 + 3 * px, y - 2 * px, panelW - 6 * px, line - px};
            SDL_SetRenderDrawColor(ren, 90, 70, 30, 255);
            SDL_RenderFillRect(ren, &hl);
        }
        SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
        drawMenuText(ren, x0 + 6 * px, y, px, labels[r]);
        drawMenuText(ren, xv, y, px, values[r] + (r == gRow && !gList && r != 3 ? "  >" : ""));
        if (r != gRow || !gList) continue;
        // The drop-down, under its row.
        for (int i = 0; i < listRows; ++i) {
            y += line;
            const std::string item = itemLabel(i);
            SDL_Rect box{xv - 3 * px, y - 2 * px, panelW - (xv - x0) - 3 * px, line};
            SDL_SetRenderDrawColor(ren, 45, 45, 55, 255);
            SDL_RenderFillRect(ren, &box);
            if (i == gItem) {
                SDL_SetRenderDrawColor(ren, 200, 170, 90, 255);
                SDL_RenderFillRect(ren, &box);
                SDL_SetRenderDrawColor(ren, 20, 20, 28, 255);
            } else {
                SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
            }
            drawMenuText(ren, xv, y, px, item);
        }
    }
    if (gRow == 4) {
        SDL_Rect hl{x0 + 3 * px, y - 2 * px, panelW - 6 * px, line - px};
        SDL_SetRenderDrawColor(ren, 90, 70, 30, 255);
        SDL_RenderFillRect(ren, &hl);
    }
    SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
    drawMenuText(ren, x0 + 6 * px, y, px, "EXIT GAME");
    y += line;
    if (restart) {
        SDL_SetRenderDrawColor(ren, 230, 150, 90, 255);
        drawMenuText(ren, x0 + 6 * px, y + line / 2, px, "RESOLUTION CHANGES AFTER A RESTART");
        y += line;
    }
    SDL_SetRenderDrawColor(ren, 150, 150, 150, 255);
    const int hpx = std::max(1, px * 2 / 3);
    drawMenuText(ren, x0 + 6 * px, y0 + panelH - line + px, hpx, "ENTER SELECT   BACKSPACE BACK   F11 CLOSE");
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
}

void drawFpsCounter(SDL_Renderer* ren) {
    if (!displaySettings().fpsCounter) return;
    // Flips counted over half a second, so the number is steady enough to read.
    using Clock = std::chrono::steady_clock;
    static Clock::time_point from = Clock::now();
    static uint64_t fromFlips = flipCount();
    static int fps = 0;
    const auto now = Clock::now();
    const double secs = std::chrono::duration<double>(now - from).count();
    if (secs >= 0.5) {
        const uint64_t flips = flipCount();
        fps = int((flips - fromFlips) / secs + 0.5);
        from = now;
        fromFlips = flips;
    }
    int w = 0, h = 0;
    SDL_GetRendererOutputSize(ren, &w, &h);
    const int px = std::max(1, h / 360);
    const std::string text = std::to_string(fps) + " FPS";
    const int tw = menuTextWidth(text, px);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_Rect box{w - tw - 8 * px, 2 * px, tw + 6 * px, 11 * px};
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 160);
    SDL_RenderFillRect(ren, &box);
    SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
    drawMenuText(ren, w - tw - 5 * px, 4 * px, px, text);
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
}
#endif

} // namespace host
