#include "host/display_menu.h"

#include "host/display_settings.h"
#include "host/hle.h"
#include "host/save_menu.h"

#include <algorithm>
#include <atomic>
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
int gRow = 0;         // 0 window mode, 1 resolution, 2 exit game
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

int resIndex(uint32_t w, uint32_t h) {
    for (int i = 0; i < kResCount; ++i)
        if (kResolutions[i].w == w && kResolutions[i].h == h) return i;
    return -1;
}

std::string resLabel(uint32_t w, uint32_t h) {
    const int i = resIndex(w, h);
    return i >= 0 ? kResolutions[i].label : std::to_string(w) + "X" + std::to_string(h);
}

int itemCount() { return gRow == 0 ? 3 : kResCount; }

void pick(SDL_Window* win) {
    DisplaySettings& s = displaySettings();
    if (gRow == 0) {
        s.window = WindowMode(gItem);
        applyWindowMode(win);
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
    if (gRow == 2) { // as closing the window does
        std::fprintf(stderr, "[window] exit from the display menu\n");
        printUnimplementedSummary();
        std::_Exit(0);
    }
    const DisplaySettings& s = displaySettings();
    gItem = gRow == 0 ? int(s.window) : std::max(0, resIndex(s.width, s.height));
    gList = true;
}

void back() {
    if (gList) gList = false;
    else gOpen = false;
}

void move(int d) {
    if (gList) gItem = (gItem + d + itemCount()) % itemCount();
    else gRow = (gRow + d + 3) % 3;
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

bool displayMenuEvent(const SDL_Event& e, SDL_Window* win) {
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
    const std::string values[2] = {kModes[int(s.window)], resLabel(s.width, s.height)};
    const char* const labels[2] = {"WINDOW MODE", "RESOLUTION"};
    const int listRows = gList ? itemCount() : 0;
    const int panelW = std::min(w - 4 * px, 230 * px), panelH = (7 + listRows + (restart ? 1 : 0)) * line;
    const int x0 = (w - panelW) / 2, y0 = (h - panelH) / 2;
    const int xv = x0 + 6 * px + menuTextWidth("WINDOW MODE  ", px);

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_Rect panel{x0, y0, panelW, panelH};
    SDL_SetRenderDrawColor(ren, 20, 20, 28, 235);
    SDL_RenderFillRect(ren, &panel);
    SDL_SetRenderDrawColor(ren, 200, 170, 90, 255);
    SDL_RenderDrawRect(ren, &panel);
    drawMenuText(ren, x0 + 6 * px, y0 + 4 * px, px, "DISPLAY");

    int y = y0 + 2 * line;
    for (int r = 0; r < 2; ++r, y += line) {
        if (r == gRow) {
            SDL_Rect hl{x0 + 3 * px, y - 2 * px, panelW - 6 * px, line - px};
            SDL_SetRenderDrawColor(ren, 90, 70, 30, 255);
            SDL_RenderFillRect(ren, &hl);
        }
        SDL_SetRenderDrawColor(ren, 235, 235, 225, 255);
        drawMenuText(ren, x0 + 6 * px, y, px, labels[r]);
        drawMenuText(ren, xv, y, px, values[r] + (r == gRow && !gList ? "  >" : ""));
        if (r != gRow || !gList) continue;
        // The drop-down, under its row.
        for (int i = 0; i < listRows; ++i) {
            y += line;
            const std::string item = gRow == 0 ? kModes[i] : kResolutions[i].label;
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
    if (gRow == 2) {
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
#endif

} // namespace host
