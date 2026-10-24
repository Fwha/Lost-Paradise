#include "host/display_settings.h"

#include "host/kernel.h"
#include "host/module.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace host {

namespace {

constexpr const char* kFile = "lpn-display.ini";
DisplaySettings gSettings;
uint32_t gActiveW = 0, gActiveH = 0;

const char* modeName(WindowMode m) {
    return m == WindowMode::Borderless ? "borderless" : m == WindowMode::Fullscreen ? "fullscreen" : "windowed";
}

} // namespace

DisplaySettings& displaySettings() { return gSettings; }

void loadDisplaySettings() {
    FILE* f = std::fopen(kFile, "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        char key[64], value[64];
        if (std::sscanf(line, " %63[^= ] = %63s", key, value) != 2) continue;
        if (!std::strcmp(key, "window")) {
            for (WindowMode m : {WindowMode::Windowed, WindowMode::Borderless, WindowMode::Fullscreen})
                if (!std::strcmp(value, modeName(m))) gSettings.window = m;
        } else if (!std::strcmp(key, "resolution")) {
            unsigned w = 0, h = 0;
            if (std::sscanf(value, "%ux%u", &w, &h) == 2 && w >= 320 && h >= 180 && w <= 7680 && h <= 4320)
                gSettings.width = w, gSettings.height = h;
            else gSettings.width = gSettings.height = 0;
        } else if (!std::strcmp(key, "fps")) {
            const int fps = std::atoi(value);
            if (!std::strcmp(value, "uncapped")) gSettings.fps = 90; // no longer offered
            else if (fps >= 10 && fps <= 240) gSettings.fps = fps;
        } else if (!std::strcmp(key, "fps_counter")) {
            gSettings.fpsCounter = !std::strcmp(value, "on");
        }
    }
    std::fclose(f);
}

void saveDisplaySettings() {
    FILE* f = std::fopen(kFile, "w");
    if (!f) {
        std::fprintf(stderr, "[display] cannot write %s\n", kFile);
        return;
    }
    std::fprintf(f, "window = %s\n", modeName(gSettings.window));
    if (gSettings.width) std::fprintf(f, "resolution = %ux%u\n", gSettings.width, gSettings.height);
    else std::fprintf(f, "resolution = default\n");
    if (gSettings.fps) std::fprintf(f, "fps = %d\n", gSettings.fps);
    else std::fprintf(f, "fps = uncapped\n");
    std::fprintf(f, "fps_counter = %s\n", gSettings.fpsCounter ? "on" : "off");
    std::fclose(f);
}

bool patchRenderResolution(Module& eboot, uint32_t width, uint32_t height) {
    if (!width || !height) return true;
    // The eboot's display setup (0x55d5e0, given argc/argv) stores the render width and height
    // (0x239d5ec/ee) and the output size (0x239d5f0/f2) as 16-bit values: 1600x896 on a base
    // PS4, 1920x1080 on a Pro, 1920x1080 output for both. Each is a `movw $imm16, (reg)`.
    struct Store {
        uint32_t at;
        uint8_t reg;     // ModRM byte
        uint16_t was;
        bool isWidth;
        bool output;
    };
    static const Store kStores[] = {
        {0x55d861, 0x00, 0x780, true, false},  {0x55d866, 0x01, 0x438, false, false}, // PS4 Pro
        {0x55d9f1, 0x00, 0x640, true, false},  {0x55d9f6, 0x01, 0x380, false, false}, // base PS4
        {0x55da09, 0x00, 0x780, true, true},   {0x55da0e, 0x01, 0x438, false, true},  // output
    };
    uint8_t* text = eboot.base();
    for (const Store& s : kStores) {
        const uint8_t* p = text + s.at;
        if (p[0] != 0x66 || p[1] != 0xC7 || p[2] != s.reg || uint16_t(p[3] | p[4] << 8) != s.was) {
            std::fprintf(stderr, "[display] unexpected eboot code at 0x%x: resolution left as the game's\n", s.at);
            return false;
        }
    }
    // Above 1080p the game also outputs at that size (its 4K modes do the same), so the window
    // gets every rendered pixel; at or below it keeps the 1920x1080 output and scales up.
    const bool bigger = width > 1920 || height > 1080;
    for (const Store& s : kStores) {
        if (s.output && !bigger) continue;
        const uint16_t v = uint16_t(s.isWidth ? width : height);
        text[s.at + 3] = uint8_t(v);
        text[s.at + 4] = uint8_t(v >> 8);
    }
    gActiveW = width, gActiveH = height;
    std::fprintf(stderr, "[display] rendering at %ux%u (output %ux%u)\n", width, height, bigger ? width : 1920,
                 bigger ? height : 1080);
    return true;
}

void activeRenderResolution(uint32_t* width, uint32_t* height) {
    *width = gActiveW;
    *height = gActiveH;
}

namespace {

struct CodePatch {
    uint32_t at;
    std::vector<uint8_t> was, now;
};

// 0 the game's clock, 1 the measured time instead of 1/60 steps (30 fps cap), 2 that and a lower
// floor (uncapped).
const CodePatch kMeasured[] = {
    // The clock: `if (fixed) dt = fixedDt` becomes `if (fixedDt > 1/60) dt = fixedDt`.
    //   mov 0x90(%r14),%rcx; test %ecx,%ecx; je measured; shr $32,%rcx
    //   -> mov 0x94(%r14),%ecx; cmp $0x3c88888a,%ecx (just above 1/60 as a float); jb measured
    {0xd9ee98,
     {0x49, 0x8b, 0x8e, 0x90, 0, 0, 0, 0x85, 0xc9, 0x74, 0x28, 0x48, 0xc1, 0xe9, 0x20},
     {0x41, 0x8b, 0x8e, 0x94, 0, 0, 0, 0x81, 0xf9, 0x8a, 0x88, 0x88, 0x3c, 0x72, 0x24}},
    // Turning the fixed step off clears the step too (movl $0,0x90 -> movq $0,0x90), so a stale
    // 1/30 is not taken for a fixed step.
    {0xd9f0f7,
     {0xc7, 0x80, 0x90, 0, 0, 0, 0, 0, 0, 0, 0xc3, 0x90},
     {0x48, 0xc7, 0x80, 0x90, 0, 0, 0, 0, 0, 0, 0, 0xc3}},
    // The flip interval the game asks for, ceil(dt * 60) while a fixed step is on, would lock it
    // to 30 fps after one slow frame. Only the 1/30 and 1/20 steps set it now, rounded:
    //   cmpl $0,0x90(%rax); je keep; ...; vroundss $0xa; vcvttss2si %xmm0,%edi
    //   -> cmpl $0x3c88888a,0x94(%rax); jb keep; ...; vcvtss2si %xmm0,%edi; nop
    {0xd993ef,
     {0x83, 0xb8, 0x90, 0, 0, 0, 0, 0x74, 0x17, 0xc5, 0xfa, 0x10, 0x40, 0x60, 0xc5, 0xfa,
      0x59, 0x05, 0x3b, 0x73, 0x27, 0x01, 0xc4, 0xe3, 0x79, 0x0a, 0xc0, 0x0a, 0xc5, 0xfa, 0x2c, 0xf8},
     {0x81, 0xb8, 0x94, 0, 0, 0, 0x8a, 0x88, 0x88, 0x3c, 0x72, 0x14, 0xc5, 0xfa, 0x10, 0x40,
      0x60, 0xc5, 0xfa, 0x59, 0x05, 0x38, 0x73, 0x27, 0x01, 0xc5, 0xfa, 0x2d, 0xf8, 0x0f, 0x1f, 0x00}},
};
// The measured time is rounded up to 1/60 s when under 17.25 ms: lower that to 2 ms (500 fps).
//   cmp $17250000,%eax; mov $16666666,%edx -> cmp $2000000,%eax; mov $2000000,%edx
const CodePatch kFloor[] = {
    {0xd9eeda, {0x3d, 0xd0, 0x36, 0x07, 0x01, 0xba, 0x2a, 0x50, 0xfe, 0x00},
     {0x3d, 0x80, 0x84, 0x1e, 0x00, 0xba, 0x80, 0x84, 0x1e, 0x00}},
};

uint8_t* gEbootText = nullptr; // set once the code is known to match
std::atomic<int> gClockWanted{0};
int gClockActive = 0;

void write(const CodePatch& p, bool on) {
    const std::vector<uint8_t>& bytes = on ? p.now : p.was;
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    uint8_t* at = gEbootText + p.at;
    uint8_t* start = reinterpret_cast<uint8_t*>(uintptr_t(at) & ~(page - 1));
    const size_t len = size_t(at + bytes.size() - start);
    mprotect(start, len, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::memcpy(at, bytes.data(), bytes.size());
    mprotect(start, len, PROT_READ | PROT_EXEC);
}

} // namespace

bool prepareFrameClock(Module& eboot) {
    auto matches = [&](const CodePatch& p) {
        if (std::memcmp(eboot.base() + p.at, p.was.data(), p.was.size()) == 0) return true;
        std::fprintf(stderr, "[display] unexpected eboot code at 0x%x: frame clock left as the game's\n", p.at);
        return false;
    };
    for (const CodePatch& p : kMeasured)
        if (!matches(p)) return false;
    for (const CodePatch& p : kFloor)
        if (!matches(p)) return false;
    gEbootText = eboot.base();
    return true;
}

// A scaled game clock (LPN_TIME_SCALE) needs the measured time unfloored, at any cap.
void setFrameClock(int fps) {
    gClockWanted = fps <= 0 || fps > 61 || gameTimeScale() != 1.0 ? 2 : fps < 59 ? 1 : 0;
}

namespace {

// LPN_PROFILE=1: a [clock] line a second with what the game's frame clock (0x2486080) stepped:
// frame times, game seconds per real second for its float time and its 1/3000 s ticks, and how
// many frames were too short to count as one 60 Hz or 30 Hz frame when rounded.
void clockStats() {
    static const bool enabled = std::getenv("LPN_PROFILE") != nullptr;
    if (!enabled) return;
    struct Clock {
        float dt;           // +0x60
        int32_t ticks;      // +0x64
        uint8_t pad0[0x10];
        float time;         // +0x78
        float rawTime;      // +0x7c
        uint32_t totalTicks; // +0x80
        uint8_t pad1[0xc];
        uint32_t fixedOn;   // +0x90
        float fixedDt;      // +0x94
        uint32_t flags;     // +0x98
    };
    static_assert(offsetof(Clock, time) == 0x18 && offsetof(Clock, fixedOn) == 0x30);
    const Clock& c = *reinterpret_cast<const Clock*>(gEbootText + 0x2486080 + 0x60);
    using Wall = std::chrono::steady_clock;
    static Wall::time_point start = Wall::now();
    static float time0 = c.time;
    static uint32_t ticks0 = c.totalTicks;
    static int frames = 0, under60 = 0, under30 = 0;
    static float dtMin = 1e9f, dtMax = 0, dtSum = 0;
    ++frames;
    dtSum += c.dt, dtMin = std::min(dtMin, c.dt), dtMax = std::max(dtMax, c.dt);
    under60 += std::lrint(c.dt * 60.0f) == 0; // as cvtss2si rounds: half to even
    under30 += std::lrint(c.dt * 30.0f) == 0;
    const double wall = std::chrono::duration<double>(Wall::now() - start).count();
    if (wall < 1.0) return;
    static int second = 0;
    std::fprintf(stderr,
                 "[clock] t=%d fps=%.0f dt=%.2f/%.2f/%.2f ms speed=%.3f ticks=%.3f round0@60=%d round0@30=%d "
                 "fixed=%u/%.4f flags=0x%x\n",
                 ++second, frames / wall, dtMin * 1e3f, dtSum / frames * 1e3f, dtMax * 1e3f, (c.time - time0) / wall,
                 (c.totalTicks - ticks0) / 3000.0 / wall, under60, under30, c.fixedOn, c.fixedDt, c.flags);
    start = Wall::now(), time0 = c.time, ticks0 = c.totalTicks;
    frames = under60 = under30 = 0;
    dtMin = 1e9f, dtMax = dtSum = 0;
}

} // namespace

void syncFrameClock() {
    if (gEbootText) clockStats();
    const int want = gClockWanted;
    if (!gEbootText || want == gClockActive) return;
    for (const CodePatch& p : kMeasured) write(p, want >= 1);
    for (const CodePatch& p : kFloor) write(p, want >= 2);
    gClockActive = want;
    std::fprintf(stderr, "[display] frame clock: %s\n",
                 want == 2 ? "measured time, uncapped" : want == 1 ? "measured time" : "the game's");
}

} // namespace host
