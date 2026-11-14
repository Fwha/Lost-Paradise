#include "host/window.h"

#include "host/combo_trace.h"
#include "host/coverage.h"
#include "host/display_menu.h"
#include "host/display_settings.h"
#include "host/gpu/renderer.h"
#include "host/hle.h"
#include "host/profile.h"
#include "host/save_menu.h"
#include "host/videoout.h"

#include <algorithm>
#include <cmath>
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

// --- Motion ---------------------------------------------------------------------------

struct Quat { float x = 0, y = 0, z = 0, w = 1; };
Quat mul(const Quat& a, const Quat& b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
Quat normalized(Quat q) {
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return n > 0 ? Quat{q.x / n, q.y / n, q.z / n, q.w / n} : Quat{};
}
// World up (0, 1, 0) seen from the pad: what the accelerometer reads at rest.
void upInPad(const Quat& q, float out[3]) {
    out[0] = 2 * (q.x * q.y + q.w * q.z);
    out[1] = 1 - 2 * (q.x * q.x + q.z * q.z);
    out[2] = 2 * (q.y * q.z - q.w * q.x);
}

// Orientation from the gyro, pulled toward the accelerometer's idea of up (a Mahony filter)
// so it does not drift, as the PS4's tilt correction does.
Quat gOrient;
void integrate(Quat& q, const float gyro[3], const float accel[3], float dt) {
    float w[3] = {gyro[0], gyro[1], gyro[2]};
    const float an = std::sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
    if (an > 0.7f && an < 1.3f) { // only while the pad is not being shaken
        float v[3];
        upInPad(q, v);
        const float a[3] = {accel[0] / an, accel[1] / an, accel[2] / an};
        constexpr float kP = 2.0f;
        w[0] += kP * (a[1] * v[2] - a[2] * v[1]);
        w[1] += kP * (a[2] * v[0] - a[0] * v[2]);
        w[2] += kP * (a[0] * v[1] - a[1] * v[0]);
    }
    const Quat dq = mul(q, Quat{w[0], w[1], w[2], 0});
    q = normalized({q.x + 0.5f * dt * dq.x, q.y + 0.5f * dt * dq.y, q.z + 0.5f * dt * dq.z, q.w + 0.5f * dt * dq.w});
}

bool gHasSensors = false;
void enableSensors(SDL_GameController* gc) {
    gHasSensors = false;
    gOrient = {};
#if SDL_VERSION_ATLEAST(2, 0, 14)
    if (gc && SDL_GameControllerHasSensor(gc, SDL_SENSOR_ACCEL) && SDL_GameControllerHasSensor(gc, SDL_SENSOR_GYRO) &&
        SDL_GameControllerSetSensorEnabled(gc, SDL_SENSOR_ACCEL, SDL_TRUE) == 0 &&
        SDL_GameControllerSetSensorEnabled(gc, SDL_SENSOR_GYRO, SDL_TRUE) == 0)
        gHasSensors = true;
#endif
    if (gc)
        std::fprintf(stderr, "[pad] %s: motion from %s\n", SDL_GameControllerName(gc),
                     gHasSensors ? "its sensors" : "the right stick");
}

// The controller's own sensors.
bool readSensors(SDL_GameController* gc, PadState& p, float dt) {
#if SDL_VERSION_ATLEAST(2, 0, 14)
    if (!gc || !gHasSensors) return false;
    float a[3], g[3];
    if (SDL_GameControllerGetSensorData(gc, SDL_SENSOR_ACCEL, a, 3) != 0 ||
        SDL_GameControllerGetSensorData(gc, SDL_SENSOR_GYRO, g, 3) != 0)
        return false;
    for (int i = 0; i < 3; ++i) p.acceleration[i] = a[i] / SDL_STANDARD_GRAVITY, p.angularVelocity[i] = g[i];
    integrate(gOrient, p.angularVelocity, p.acceleration, dt);
    p.orientation[0] = gOrient.x, p.orientation[1] = gOrient.y, p.orientation[2] = gOrient.z, p.orientation[3] = gOrient.w;
    return true;
#else
    (void)gc, (void)p, (void)dt;
    return false;
#endif
}

// No sensors (an Xbox pad, the keyboard): the right stick tilts a virtual pad up to 60 degrees,
// up and down pitch it, sideways rolls it; moving the stick quickly shakes it, so
// waggling it back and forth reads as shaking the controller.
void stickMotion(PadState& p, float dt) {
    static float lastPitch = 0, lastRoll = 0, lastVel[3] = {}, lastPos[3] = {};
    constexpr float kMaxTilt = 1.0472f; // 60 degrees
    const float sx = (int(p.rx) - 128) / 127.0f, sy = (int(p.ry) - 128) / 127.0f;
    const float pitch = -sy * kMaxTilt, roll = -sx * kMaxTilt; // about x, then z
    const Quat q = mul(Quat{std::sin(pitch / 2), 0, 0, std::cos(pitch / 2)}, Quat{0, 0, std::sin(roll / 2), std::cos(roll / 2)});
    p.orientation[0] = q.x, p.orientation[1] = q.y, p.orientation[2] = q.z, p.orientation[3] = q.w;
    float up[3];
    upInPad(q, up);
    // The stick as a 12 cm swing of the pad, differentiated twice for the shake.
    const float pos[3] = {sx * 0.12f, 0, sy * 0.12f};
    for (int i = 0; i < 3; ++i) {
        const float vel = dt > 0 ? (pos[i] - lastPos[i]) / dt : 0;
        const float acc = dt > 0 ? (vel - lastVel[i]) / dt / 9.80665f : 0;
        p.acceleration[i] = std::clamp(up[i] + acc, -4.0f, 4.0f);
        lastPos[i] = pos[i], lastVel[i] = vel;
    }
    p.angularVelocity[0] = dt > 0 ? (pitch - lastPitch) / dt : 0;
    p.angularVelocity[1] = 0;
    p.angularVelocity[2] = dt > 0 ? (roll - lastRoll) / dt : 0;
    lastPitch = pitch, lastRoll = roll;
}

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
    static Uint64 lastTick = SDL_GetPerformanceCounter();
    const Uint64 tick = SDL_GetPerformanceCounter();
    const float dt = std::min(float(double(tick - lastTick) / double(SDL_GetPerformanceFrequency())), 0.1f);
    lastTick = tick;
    if (!readSensors(gc, p, dt)) stickMotion(p, dt);
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
    // 720 rows, as wide as the Aspect Ratio makes the picture.
    const int winW = int(720 * aspectValue(displaySettings().aspect) + 0.5);
    SDL_Window* win = SDL_CreateWindow("Fist of the North Star: Lost Paradise", SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED, winW, 720, SDL_WINDOW_RESIZABLE);
    bool vsync = displaySettings().vsync; // VSync: as the renderer was made or last set
    SDL_Renderer* ren =
        win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | (vsync ? SDL_RENDERER_PRESENTVSYNC : 0)) : nullptr;
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
            if (render::WarmupProgress w; render::warmupProgress(w)) { // the startup compile's screen
                const bool key = e.type == SDL_KEYDOWN && !e.key.repeat &&
                                 (e.key.keysym.scancode == SDL_SCANCODE_ESCAPE || e.key.keysym.scancode == SDL_SCANCODE_RETURN);
                const bool pad = e.type == SDL_CONTROLLERBUTTONDOWN &&
                                 (e.cbutton.button == SDL_CONTROLLER_BUTTON_A || e.cbutton.button == SDL_CONTROLLER_BUTTON_B ||
                                  e.cbutton.button == SDL_CONTROLLER_BUTTON_START);
                if (key || pad) render::skipWarmup();
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F12 && !e.key.repeat) requestCapture();
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F10 && !e.key.repeat) profile::mark();
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F9 && !e.key.repeat) coverage::onKey();
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_F8 && !e.key.repeat) combo_trace::onKey();
            if (e.type == SDL_CONTROLLERDEVICEADDED && !gc) {
                gc = SDL_GameControllerOpen(e.cdevice.which);
                enableSensors(gc);
            }
            if (e.type == SDL_CONTROLLERDEVICEREMOVED && gc) {
                SDL_GameControllerClose(gc);
                gc = nullptr;
                enableSensors(nullptr);
            }
        }
        pollPad(gc);
        applyDisplayOptions(win);
#if SDL_VERSION_ATLEAST(2, 0, 18)
        if (displaySettings().vsync != vsync) { // changed in the PC Settings screen
            vsync = displaySettings().vsync;
            if (SDL_RenderSetVSync(ren, vsync ? 1 : 0) != 0)
                std::fprintf(stderr, "[window] cannot turn vsync %s: %s\n", vsync ? "on" : "off", SDL_GetError());
            else std::fprintf(stderr, "[window] vsync %s\n", vsync ? "on" : "off");
        }
#endif

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
                // The game's 1920x1080 frame is stretched to the window: SDL's default nearest
                // scaling repeated every third row and column at 1440p, leaving stair-stepped edges
                // and blocky shading. LPN_PRESENT_NEAREST=1: nearest, as before.
                if (tex && !getenv("LPN_PRESENT_NEAREST")) SDL_SetTextureScaleMode(tex, SDL_ScaleModeLinear);
                texW = gFrame.width, texH = gFrame.height, texFormat = fmt;
            }
            SDL_UpdateTexture(tex, nullptr, gPixels.data(), int(gFrame.width * 4));
        }
        lock.unlock();
        SDL_RenderClear(ren);
        if (tex) {
            // The frame keeps its shape (the Aspect Ratio's), with black bars where the window's
            // differs; within a pixel or two of it, it fills the window.
            int ww = 0, wh = 0;
            SDL_GetRendererOutputSize(ren, &ww, &wh);
            SDL_Rect dst{0, 0, ww, wh};
            if (texW && texH && ww && wh) {
                const double fa = double(texW) / texH;
                if (ww > wh * fa + 2) dst.w = int(wh * fa + 0.5), dst.x = (ww - dst.w) / 2;
                else if (wh > ww / fa + 2) dst.h = int(ww / fa + 0.5), dst.y = (wh - dst.h) / 2;
            }
            SDL_RenderCopy(ren, tex, nullptr, &dst);
        }
        if (drawShaderWarmup(ren)) {
            SDL_RenderPresent(ren);
            continue;
        }
        drawSaveMenu(ren);
        drawFpsCounter(ren);
        drawBootCompile(ren);
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
