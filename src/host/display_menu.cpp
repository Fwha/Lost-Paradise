#include "host/display_menu.h"

#include "common/ui_palette.h"

#include "host/display_settings.h"
#include "host/gpu/renderer.h"
#include "host/hle.h"
#include "host/rounded.h"
#include "host/save_menu.h"
#include "host/videoout.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

// The save menu's drawing (save_menu.h), so both menus look alike.
void text(SDL_Renderer* ren, int x, int y, int size, const std::string& s, bool bold = false) {
    menuText(ren, x, y, size, s, bold);
}
int textW(const std::string& s, int size, bool bold = false) { return menuTextW(s, size, bold); }
std::string fitText(const std::string& s, int size, int maxW) { return menuFitText(s, size, maxW); }
void fill(SDL_Renderer* ren, int x, int y, int w, int h, Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255) {
    menuFill(ren, x, y, w, h, r, g, b, a);
}
void color(SDL_Renderer* ren, Uint8 r, Uint8 g, Uint8 b) { SDL_SetRenderDrawColor(ren, r, g, b, 255); }
void fill(SDL_Renderer* ren, int x, int y, int w, int h, ui::Rgb c) { fill(ren, x, y, w, h, c.r, c.g, c.b); }
void color(SDL_Renderer* ren, ui::Rgb c) { color(ren, c.r, c.g, c.b); }
SDL_Color sdl(ui::Rgb c) { return {c.r, c.g, c.b, 255}; }
// Rounded versions of fill and outline (rounded.h); radius r.
void fillR(SDL_Renderer* ren, int x, int y, int w, int h, int r, Uint8 cr, Uint8 cg, Uint8 cb, Uint8 a = 255,
           unsigned corners = ui::kAll) {
    ui::fillRounded(ren, x, y, w, h, r, {cr, cg, cb, a}, corners);
}
void fillR(SDL_Renderer* ren, int x, int y, int w, int h, int r, ui::Rgb c, unsigned corners = ui::kAll) {
    ui::fillRounded(ren, x, y, w, h, r, sdl(c), corners);
}
void outlineR(SDL_Renderer* ren, int x, int y, int w, int h, int r, int t, ui::Rgb c) {
    ui::outlineRounded(ren, x, y, w, h, r, t, sdl(c));
}

std::mutex gM;
int gRow = 0;         // 0 window mode, 1 resolution, 2 fps, 3 fps counter, 4 exit game
bool gList = false;   // a drop-down is open under gRow
int gItem = 0;        // highlighted drop-down entry

constexpr int kRows = 5;

int resIndex(uint32_t w, uint32_t h) {
    for (int i = 0; i < kResCount; ++i)
        if (kResolutions[i].w == w && kResolutions[i].h == h) return i;
    return -1;
}

std::string resLabel(uint32_t w, uint32_t h) {
    const int i = resIndex(w, h);
    return i >= 0 ? kResolutions[i].label : std::to_string(w) + "x" + std::to_string(h);
}

int fpsIndex(int fps) {
    for (int i = 0; i < kFpsCount; ++i)
        if (kFpsChoices[i] == fps) return i;
    return -1;
}

std::string fpsLabel(int fps) { return fps ? std::to_string(fps) : "Uncapped"; }

int itemCount() { return gRow == 0 ? 3 : gRow == 1 ? kResCount : kFpsCount; }

std::string itemLabel(int i) {
    return gRow == 0 ? kWindowModeNames[i] : gRow == 1 ? kResolutions[i].label : fpsLabel(kFpsChoices[i]);
}

// A choice of a row (0..4) made, from the menu or the game's PC SETTINGS screen.
void apply(SDL_Window* win, int row, int item) {
    DisplaySettings& s = displaySettings();
    if (row == 0) {
        s.window = WindowMode(item);
        applyWindowMode(win);
        applyFrameRate(win);
    } else if (row == 2) {
        if (item >= kFpsCount) return; // the cap it has
        s.fps = kFpsChoices[item];
        setFpsCap(s.fps);
        setFrameClock(s.fps);
        applyFrameRate(win);
        std::fprintf(stderr, "[display] fps cap %s\n", fpsLabel(s.fps).c_str());
    } else if (row == 1) {
        if (item >= kResCount) return; // the size it has
        s.width = kResolutions[item].w;
        s.height = kResolutions[item].h;
        std::fprintf(stderr, "[display] resolution %s after a restart\n", kResolutions[item].label);
    } else if (row == 3) {
        s.fpsCounter = item != 0;
    } else {
        if (!item) return;
        // As closing the window does.
        std::fprintf(stderr, "[window] exit from the display menu\n");
        printUnimplementedSummary();
        render::saveCaches();
        std::_Exit(0);
    }
    saveDisplaySettings();
}

void pick(SDL_Window* win) {
    apply(win, gRow, gItem);
    gList = false;
}

void select(SDL_Window* win) {
    if (gList) return pick(win);
    if (gRow == 3) return apply(win, 3, !displaySettings().fpsCounter); // on or off: no drop-down
    if (gRow == 4) return apply(win, 4, 1);
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

namespace {
// A small down-pointing triangle, the drop-down's arrow.
void arrow(SDL_Renderer* ren, int cx, int cy, int s, SDL_Color c, bool up) {
    const float h = s * 0.55f, d = up ? -1.0f : 1.0f;
    const SDL_Vertex v[3] = {{{cx - s / 2.0f, cy - d * h / 2}, c, {0, 0}},
                             {{cx + s / 2.0f, cy - d * h / 2}, c, {0, 0}},
                             {{float(cx), cy + d * h / 2}, c, {0, 0}}};
    SDL_RenderGeometry(ren, nullptr, v, 3, nullptr, 0);
}
} // namespace

// Laid out as the save menu is (save_menu.cpp): a dark panel with a header, one row per setting
// with its value on the right, the drop-down under its row, then the controls along the bottom.
void drawDisplayMenu(SDL_Renderer* ren) {
    if (!gOpen) return;
    std::lock_guard lock(gM);
    const DisplaySettings& s = displaySettings();
    uint32_t activeW, activeH;
    activeRenderResolution(&activeW, &activeH);
    const bool restart = activeW != s.width || activeH != s.height;

    int w = 0, h = 0;
    SDL_GetRendererOutputSize(ren, &w, &h);
    const float u = std::max(0.5f, h / 720.0f); // 1 at 720p
    auto U = [u](float v) { return std::max(1, int(std::lround(v * u))); };
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    fill(ren, 0, 0, w, h, 0, 0, 0, 150);

    const std::string values[4] = {kWindowModeNames[int(s.window)], resLabel(s.width, s.height), fpsLabel(s.fps),
                                   s.fpsCounter ? "On" : "Off"};
    const char* const labels[kRows] = {"Window Mode", "Resolution", "Frame Rate Cap", "FPS Counter", "Exit Game"};
    const int listRows = gList ? itemCount() : 0;
    const int headH = U(64), rowH = U(52), gap = U(6), itemH = U(40), noteH = restart ? U(32) : 0, hintH = U(34),
              pad = U(20);
    const int panelW = std::min(w - U(32), U(640));
    const int listH = kRows * (rowH + gap) - gap + (listRows ? listRows * itemH + gap : 0);
    const int panelH = headH + pad + listH + pad + noteH + hintH;
    const int x0 = (w - panelW) / 2, y0 = (h - panelH) / 2;

    // Panel and header.
    fillR(ren, x0, y0, panelW, panelH, U(10), 16, 16, 22, 242);
    fillR(ren, x0, y0, panelW, headH, U(10), 26, 24, 30, 255, ui::kTop);
    fill(ren, x0, y0 + headH - U(2), panelW, U(2), ui::kAccent.fill);
    color(ren, 245, 240, 228);
    text(ren, x0 + pad, y0 + (headH - U(30)) / 2, U(30), "Display Settings", true);
    const std::string active = "Rendering at " + resLabel(activeW, activeH);
    color(ren, 150, 146, 138);
    text(ren, x0 + panelW - pad - textW(active, U(16)), y0 + (headH - U(16)) / 2, U(16), active);

    // Rows: the setting's name on the left, its value in a box on the right.
    const int rx = x0 + pad, rw = panelW - 2 * pad;
    const int boxW = std::min(U(300), rw / 2), boxX = rx + rw - U(10) - boxW;
    const int ls = U(20), vs = U(19);
    int y = y0 + headH + pad;
    for (int r = 0; r < kRows; ++r) {
        const bool sel = r == gRow;
        if (sel) {
            fillR(ren, rx, y, rw, rowH, U(6), ui::kAccent.tint);
            outlineR(ren, rx, y, rw, rowH, U(6), U(2), ui::kAccent.line);
        } else {
            fillR(ren, rx, y, rw, rowH, U(6), 30, 30, 38);
        }
        if (r == 4) color(ren, sel ? 255 : 230, sel ? 150 : 128, sel ? 130 : 110);
        else color(ren, 245, 240, 228);
        text(ren, rx + U(16), y + (rowH - ls) / 2, ls, labels[r], true);

        if (r == 3) {
            // On or off: a switch.
            const int tw = U(48), th = U(24), tx = rx + rw - U(16) - tw, ty = y + (rowH - th) / 2;
            if (s.fpsCounter) fillR(ren, tx, ty, tw, th, th / 2, ui::kAccent.fill);
            else fillR(ren, tx, ty, tw, th, th / 2, 58, 58, 66);
            const int k = th - U(6);
            fillR(ren, s.fpsCounter ? tx + tw - U(3) - k : tx + U(3), ty + U(3), k, k, k / 2, 245, 240, 228);
            color(ren, s.fpsCounter ? ui::kAccent.line : ui::Rgb{150, 146, 138});
            text(ren, tx - U(12) - textW(values[3], vs), y + (rowH - vs) / 2, vs, values[3]);
        } else if (r < 3) {
            const int bh = rowH - U(14), by = y + (rowH - bh) / 2;
            fillR(ren, boxX, by, boxW, bh, U(5), 12, 12, 18);
            outlineR(ren, boxX, by, boxW, bh, U(5), U(1), sel ? ui::kAccent.line : ui::Rgb{80, 78, 88});
            color(ren, ui::kAccent.line);
            const int as = U(12);
            text(ren, boxX + U(12), y + (rowH - vs) / 2, vs, fitText(values[r], vs, boxW - U(36) - as));
            arrow(ren, boxX + boxW - U(12) - as / 2, y + rowH / 2, as,
                  sel ? sdl(ui::kAccent.line) : SDL_Color{150, 146, 138, 255}, sel && gList);
        }
        y += rowH + gap;
        if (!sel || !gList) continue;

        // The drop-down, under its row, lined up with the value box.
        const int cur = r == 0 ? int(s.window) : r == 1 ? resIndex(s.width, s.height) : fpsIndex(s.fps);
        fillR(ren, boxX, y - gap, boxW, listRows * itemH + U(2), U(5), 12, 12, 18);
        outlineR(ren, boxX, y - gap, boxW, listRows * itemH + U(2), U(5), U(1), ui::kAccent.line);
        for (int i = 0; i < listRows; ++i) {
            const int iy = y - gap + U(1) + i * itemH;
            if (i == gItem) {
                fillR(ren, boxX + U(1), iy, boxW - U(2), itemH, U(4), ui::kAccent.fill);
                color(ren, ui::kAccent.on);
            } else if (i == cur) {
                color(ren, ui::kAccent.line);
            } else {
                color(ren, 225, 220, 208);
            }
            text(ren, boxX + U(12), iy + (itemH - vs) / 2, vs, fitText(itemLabel(i), vs, boxW - U(24)), i == cur);
        }
        y += listRows * itemH + gap;
    }

    if (restart) {
        color(ren, 230, 150, 90);
        text(ren, x0 + pad, y0 + panelH - hintH - noteH - U(4), U(17), "The new resolution takes effect after a restart");
    }

    // Controls, as the save menu shows them.
    fillR(ren, x0, y0 + panelH - hintH, panelW, hintH, U(10), 10, 10, 14, 255, ui::kBottom);
    const int hs = U(15), hy = y0 + panelH - hintH + (hintH - hs) / 2;
    int hx = x0 + pad;
    const bool usePad = menuControllerAttached();
    const std::pair<MenuPad, const char*> hints[] = {{MenuPad::Cross, "Select"}, {MenuPad::Circle, "Back"}};
    const char* keys[] = {"Enter", "Backspace"};
    for (size_t i = 0; i < 3; ++i) {
        const char* key = i < 2 ? keys[i] : "F11";
        const char* what = i < 2 ? hints[i].second : "Close";
        if (usePad && i < 2) {
            const int is = U(24);
            menuPadIcon(ren, hx, y0 + panelH - hintH + (hintH - is) / 2, is, hints[i].first);
            hx += is + U(7);
        } else {
            color(ren, 200, 196, 186);
            text(ren, hx, hy, hs, key, true);
            hx += textW(key, hs, true) + U(7);
        }
        color(ren, 140, 136, 128);
        text(ren, hx, hy, hs, what);
        hx += textW(what, hs) + U(26);
    }
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
    const float u = std::max(0.5f, h / 720.0f);
    auto U = [u](float v) { return std::max(1, int(std::lround(v * u))); };
    const std::string num = std::to_string(fps), unit = " FPS";
    const int ns = U(20), us = U(14);
    const int nw = textW(num, ns, true), uw = textW(unit, us, true);
    const int bw = nw + uw + U(16), bh = ns + U(10), bx = w - bw - U(8), by = U(8);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    fill(ren, bx, by, bw, bh, 0, 0, 0, 170);
    color(ren, 245, 240, 228);
    text(ren, bx + U(8), by + U(5), ns, num, true);
    color(ren, ui::kAccent.line);
    text(ren, bx + U(8) + nw, by + U(5) + ns - us - U(1), us, unit, true);
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
}

namespace {
std::vector<std::pair<int, int>> gQueued; // setDisplayOption's (row, choice), under gM

int valueOf(int row) {
    for (auto it = gQueued.rbegin(); it != gQueued.rend(); ++it)
        if (it->first == row) return it->second;
    const DisplaySettings& s = displaySettings();
    switch (row) {
    case 0: return int(s.window);
    case 1: return resIndex(s.width, s.height) >= 0 ? resIndex(s.width, s.height) : kResCount;
    case 2: return fpsIndex(s.fps) >= 0 ? fpsIndex(s.fps) : kFpsCount;
    case 3: return s.fpsCounter ? 1 : 0;
    default: return 0; // exit: not yet
    }
}
} // namespace

int displayOptionCount(int row) {
    std::lock_guard lock(gM);
    const DisplaySettings& s = displaySettings();
    switch (row) {
    case 0: return 3;
    case 1: return kResCount + (resIndex(s.width, s.height) < 0);
    case 2: return kFpsCount + (fpsIndex(s.fps) < 0);
    case 3: case 4: return 2;
    default: return 0;
    }
}

std::string displayOptionName(int row, int i) {
    std::lock_guard lock(gM);
    const DisplaySettings& s = displaySettings();
    switch (row) {
    case 0: return kWindowModeNames[std::clamp(i, 0, 2)];
    case 1: return i < kResCount ? kResolutions[i].label : std::to_string(s.width) + "x" + std::to_string(s.height);
    case 2: {
        const int fps = i < kFpsCount ? kFpsChoices[i] : s.fps;
        return fps ? std::to_string(fps) : "Uncapped";
    }
    case 3: return i ? "On" : "Off";
    default: return i ? "Yes" : "No";
    }
}

int displayOptionValue(int row) {
    std::lock_guard lock(gM);
    return valueOf(row);
}

void setDisplayOption(int row, int i) {
    std::lock_guard lock(gM);
    gQueued.emplace_back(row, i);
}

void applyDisplayOptions(SDL_Window* win) {
    std::lock_guard lock(gM);
    for (auto [row, item] : gQueued) apply(win, row, item);
    gQueued.clear();
}
#else
int displayOptionCount(int) { return 0; }
std::string displayOptionName(int, int) { return {}; }
int displayOptionValue(int) { return 0; }
void setDisplayOption(int, int) {}
#endif

} // namespace host
