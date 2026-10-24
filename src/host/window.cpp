#include "host/window.h"

#include "host/coverage.h"
#include "host/display_menu.h"
#include "host/gpu/renderer.h"
#include "host/hle.h"
#include "host/profile.h"
#include "host/save_menu.h"
#include "host/videoout.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef LPN_HAVE_SDL2
#include <SDL.h>
#endif

namespace host {

namespace {

bool gEnabled = false;
std::mutex gMutex;
std::condition_variable gCv;
std::vector<uint32_t> gPixels; // tightly packed copy of the latest frame
Frame gFrame;
bool gNewFrame = false;

std::mutex gPadMutex;
PadState gPad;

struct ScriptedPress {
    uint32_t button;
    uint64_t frame;
};
std::vector<ScriptedPress> gScript;
constexpr uint64_t kPressFrames = 10;

enum : uint32_t {
    kL3 = 0x2, kR3 = 0x4, kOptions = 0x8, kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80, kL2 = 0x100,
    kR2 = 0x200, kL1 = 0x400, kR1 = 0x800, kTriangle = 0x1000, kCircle = 0x2000, kCross = 0x4000,
    kSquare = 0x8000, kTouchPad = 0x100000,
    // Scripted stick pushes (not pad bits): left stick, then right stick (camera).
    kLsLeft = 1u << 24, kLsRight = 1u << 25, kLsUp = 1u << 26, kLsDown = 1u << 27,
    kRsLeft = 1u << 28, kRsRight = 1u << 29, kRsUp = 1u << 30, kRsDown = 1u << 31,
};

#ifdef LPN_HAVE_SDL2
uint8_t axisToByte(int16_t v) { return uint8_t((int32_t(v) + 32768) >> 8); }

// Keyboard: arrows = d-pad, WASD = left stick, IJKL = right stick, Enter = Cross,
// Backspace = Circle, Z = Square, X = Triangle, Q/E = L1/R1, 1/3 = L2/R2, Esc = Options.
void pollPad(SDL_GameController* gc) {
    const Uint8* k = SDL_GetKeyboardState(nullptr);
    PadState p;
    auto bit = [&](bool on, uint32_t b) { if (on) p.buttons |= b; };
    bit(k[SDL_SCANCODE_UP], kUp);
    bit(k[SDL_SCANCODE_DOWN], kDown);
    bit(k[SDL_SCANCODE_LEFT], kLeft);
    bit(k[SDL_SCANCODE_RIGHT], kRight);
    bit(k[SDL_SCANCODE_RETURN], kCross);
    bit(k[SDL_SCANCODE_BACKSPACE], kCircle);
    bit(k[SDL_SCANCODE_Z], kSquare);
    bit(k[SDL_SCANCODE_X], kTriangle);
    bit(k[SDL_SCANCODE_Q], kL1);
    bit(k[SDL_SCANCODE_E], kR1);
    bit(k[SDL_SCANCODE_ESCAPE], kOptions);
    bit(k[SDL_SCANCODE_TAB], kTouchPad);
    if (k[SDL_SCANCODE_1]) p.buttons |= kL2, p.l2 = 0xFF;
    if (k[SDL_SCANCODE_3]) p.buttons |= kR2, p.r2 = 0xFF;
    if (k[SDL_SCANCODE_A]) p.lx = 0x00;
    if (k[SDL_SCANCODE_D]) p.lx = 0xFF;
    if (k[SDL_SCANCODE_W]) p.ly = 0x00;
    if (k[SDL_SCANCODE_S]) p.ly = 0xFF;
    if (k[SDL_SCANCODE_J]) p.rx = 0x00;
    if (k[SDL_SCANCODE_L]) p.rx = 0xFF;
    if (k[SDL_SCANCODE_I]) p.ry = 0x00;
    if (k[SDL_SCANCODE_K]) p.ry = 0xFF;

    if (gc) {
        auto btn = [&](SDL_GameControllerButton b, uint32_t ps) { bit(SDL_GameControllerGetButton(gc, b), ps); };
        btn(SDL_CONTROLLER_BUTTON_A, kCross);
        btn(SDL_CONTROLLER_BUTTON_B, kCircle);
        btn(SDL_CONTROLLER_BUTTON_X, kSquare);
        btn(SDL_CONTROLLER_BUTTON_Y, kTriangle);
        btn(SDL_CONTROLLER_BUTTON_LEFTSHOULDER, kL1);
        btn(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, kR1);
        btn(SDL_CONTROLLER_BUTTON_LEFTSTICK, kL3);
        btn(SDL_CONTROLLER_BUTTON_RIGHTSTICK, kR3);
        btn(SDL_CONTROLLER_BUTTON_START, kOptions);
        btn(SDL_CONTROLLER_BUTTON_BACK, kTouchPad);
        btn(SDL_CONTROLLER_BUTTON_TOUCHPAD, kTouchPad);
        btn(SDL_CONTROLLER_BUTTON_DPAD_UP, kUp);
        btn(SDL_CONTROLLER_BUTTON_DPAD_DOWN, kDown);
        btn(SDL_CONTROLLER_BUTTON_DPAD_LEFT, kLeft);
        btn(SDL_CONTROLLER_BUTTON_DPAD_RIGHT, kRight);
        auto axis = [&](SDL_GameControllerAxis a) { return SDL_GameControllerGetAxis(gc, a); };
        auto stick = [&](int16_t v, uint8_t& out) { if (v < -8000 || v > 8000) out = axisToByte(v); };
        stick(axis(SDL_CONTROLLER_AXIS_LEFTX), p.lx);
        stick(axis(SDL_CONTROLLER_AXIS_LEFTY), p.ly);
        stick(axis(SDL_CONTROLLER_AXIS_RIGHTX), p.rx);
        stick(axis(SDL_CONTROLLER_AXIS_RIGHTY), p.ry);
        const int16_t lt = axis(SDL_CONTROLLER_AXIS_TRIGGERLEFT), rt = axis(SDL_CONTROLLER_AXIS_TRIGGERRIGHT);
        if (lt > 1000) p.l2 = uint8_t(lt >> 7), p.buttons |= kL2;
        if (rt > 1000) p.r2 = uint8_t(rt >> 7), p.buttons |= kR2;
    }
    // The game sees nothing while one of our menus is open, nor the keys still held as it closes.
    static bool suppress = false;
    if (saveMenuOpen() || displayMenuOpen()) suppress = true;
    else if (suppress && !p.buttons && p.lx == 0x80 && p.ly == 0x80 && p.rx == 0x80 && p.ry == 0x80) suppress = false;
    if (suppress) p = PadState{};
    std::lock_guard lock(gPadMutex);
    gPad = p;
}

void presenterThread() {
    profile::nameThread("lpn-window");
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) {
        std::fprintf(stderr, "[window] SDL_Init failed: %s\n", SDL_GetError());
        gEnabled = false;
        return;
    }
    SDL_Window* win = SDL_CreateWindow("Fist of the North Star: Lost Paradise", SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren = win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC) : nullptr;
    if (!ren) {
        std::fprintf(stderr, "[window] cannot create window: %s\n", SDL_GetError());
        gEnabled = false;
        return;
    }
    applyWindowMode(win);
    applyFrameRate(win);
    SDL_GameController* gc = nullptr;
    SDL_Texture* tex = nullptr;
    uint32_t texW = 0, texH = 0, texFormat = 0;

    for (;;) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (displayMenuEvent(e, win)) continue;
            if (saveMenuEvent(e)) continue;
            if (e.type == SDL_QUIT) {
                std::fprintf(stderr, "[window] closed\n");
                printUnimplementedSummary();
                render::saveCaches();
                std::_Exit(0);
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F12 && !e.key.repeat) requestCapture();
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F10 && !e.key.repeat) profile::mark();
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F9 && !e.key.repeat) coverage::onKey();
            if (e.type == SDL_CONTROLLERDEVICEADDED && !gc) gc = SDL_GameControllerOpen(e.cdevice.which);
            if (e.type == SDL_CONTROLLERDEVICEREMOVED && gc) {
                SDL_GameControllerClose(gc);
                gc = nullptr;
            }
        }
        pollPad(gc);

        std::unique_lock lock(gMutex);
        gCv.wait_for(lock, std::chrono::milliseconds(16), [] { return gNewFrame; });
        if (gNewFrame) {
            gNewFrame = false;
            // A8R8G8B8 is B,G,R,A in memory; A8B8G8R8 (format 0x...22..) is R,G,B,A.
            const uint32_t fmt = (gFrame.pixelFormat & 0x2200) == 0x2200 ? SDL_PIXELFORMAT_ABGR8888
                                                                          : SDL_PIXELFORMAT_ARGB8888;
            if (!tex || texW != gFrame.width || texH != gFrame.height || texFormat != fmt) {
                if (tex) SDL_DestroyTexture(tex);
                tex = SDL_CreateTexture(ren, fmt, SDL_TEXTUREACCESS_STREAMING, int(gFrame.width), int(gFrame.height));
                texW = gFrame.width, texH = gFrame.height, texFormat = fmt;
            }
            SDL_UpdateTexture(tex, nullptr, gPixels.data(), int(gFrame.width * 4));
        }
        lock.unlock();
        SDL_RenderClear(ren);
        if (tex) SDL_RenderCopy(ren, tex, nullptr, nullptr);
        drawSaveMenu(ren);
        drawFpsCounter(ren);
        drawDisplayMenu(ren);
        SDL_RenderPresent(ren);
    }
}
#endif

} // namespace

void startPresenter(bool headless) {
#ifdef LPN_HAVE_SDL2
    if (headless) return;
    gEnabled = true;
    std::thread(presenterThread).detach();
#else
    if (!headless) std::fprintf(stderr, "[window] built without SDL2; running headless\n");
#endif
}

bool windowShowsMenus() {
    std::lock_guard lock(gPadMutex);
    return gEnabled && gScript.empty();
}

PadState padState() {
    std::lock_guard lock(gPadMutex);
    PadState s = gPad;
    if (!gScript.empty()) {
        const uint64_t now = flipCount();
        for (const auto& p : gScript)
            if (now >= p.frame && now < p.frame + kPressFrames) s.buttons |= p.button;
        if (s.buttons & kLsLeft) s.lx = 0x00;
        if (s.buttons & kLsRight) s.lx = 0xFF;
        if (s.buttons & kLsUp) s.ly = 0x00;
        if (s.buttons & kLsDown) s.ly = 0xFF;
        if (s.buttons & kRsLeft) s.rx = 0x00;
        if (s.buttons & kRsRight) s.rx = 0xFF;
        if (s.buttons & kRsUp) s.ry = 0x00;
        if (s.buttons & kRsDown) s.ry = 0xFF;
        s.buttons &= 0xFFFFFF;
    }
    return s;
}

bool addScriptedPresses(const char* spec) {
    static const struct {
        const char* name;
        uint32_t bit;
    } names[] = {{"cross", kCross},   {"circle", kCircle}, {"square", kSquare}, {"triangle", kTriangle},
                 {"options", kOptions}, {"up", kUp},       {"down", kDown},     {"left", kLeft},
                 {"right", kRight},   {"l1", kL1},         {"r1", kR1},         {"touchpad", kTouchPad},
                 {"lsleft", kLsLeft}, {"lsright", kLsRight}, {"lsup", kLsUp},     {"lsdown", kLsDown},
                 {"rsleft", kRsLeft}, {"rsright", kRsRight}, {"rsup", kRsUp},     {"rsdown", kRsDown}};
    std::string all(spec);
    size_t pos = 0;
    while (pos <= all.size()) {
        const size_t end = std::min(all.find(',', pos), all.size());
        const std::string item = all.substr(pos, end - pos);
        const size_t at = item.find('@');
        if (at == std::string::npos) return false;
        uint32_t bit = 0;
        for (const auto& n : names)
            if (item.compare(0, at, n.name) == 0 && std::strlen(n.name) == at) bit = n.bit;
        if (!bit) return false;
        std::lock_guard lock(gPadMutex);
        gScript.push_back({bit, std::strtoull(item.c_str() + at + 1, nullptr, 10)});
        pos = end + 1;
    }
    return true;
}

void presentFrame(const Frame& f) {
    if (!gEnabled || !f.pixels || !f.width || !f.height) return;
    std::lock_guard lock(gMutex);
    gFrame = f;
    gPixels.resize(size_t(f.width) * f.height);
    // Tiled buffers are shown as if linear until the GPU work lands.
    for (uint32_t y = 0; y < f.height; ++y)
        std::memcpy(&gPixels[size_t(y) * f.width], static_cast<const uint32_t*>(f.pixels) + size_t(y) * f.pitchPixels,
                    f.width * 4);
    gNewFrame = true;
    gCv.notify_one();
}

} // namespace host
