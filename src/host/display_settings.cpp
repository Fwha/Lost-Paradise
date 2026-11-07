#include "host/guest_abi.h"
#include "host/display_settings.h"

#include "host/kernel.h"
#include "host/module.h"
#include "host/videoout.h"

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

#include "host/mman.h"
#include "common/compat.h"
#include <unistd.h>

namespace host {

namespace {

DisplaySettings gSettings;
uint32_t gActiveW = 0, gActiveH = 0;

} // namespace

DisplaySettings& displaySettings() { return gSettings; }

void loadDisplaySettings() { readSettings(kSettingsFile, gSettings); }

void saveDisplaySettings() {
    if (!writeSettings(kSettingsFile, gSettings)) std::fprintf(stderr, "[display] cannot write %s\n", kSettingsFile);
}

namespace {

// The game builds its 3D projections for a 16:9 picture whatever size it renders at: the
// cameras' inline projection builders (f / (16/9) for the x scale, f for y, near and far at
// +0xa4/+0xa8 of the camera, the matrix stored at +0x170), the calls into the engine's
// perspective setup (0xe66a60, whose aspect argument these pass as 16/9), and the horizontal
// field of view a camera works out for culling (2 atan(tan(v/2) * 16/9)). Each of these 16/9
// floats is read by that one instruction only, so making them the picture's aspect widens the
// view at the same vertical field of view (Hor+), culling included. The other 16/9 floats the
// game reads (screen mapping for markers over 3D objects among them) are left at 16:9: the HUD
// stays a 16:9 picture in the middle (hudLayout), and those still place on it.
constexpr uint32_t kAspectFloats[] = {
    0xfb87b4,  // 0x11b48d: projection builder
    0xfc2d18,  // 0x19482a, 0x194bb1: projection builders (one passes it to 0x2358e0)
    0xfc3260,  // 0x1ae54f: aspect handed to 0x194250
    0xfd1708,  // 0x28bf03: horizontal field of view from the vertical one
    0xfd5914,  // 0x2d2328: aspect argument of 0xe66a60
    0x108937c, // 0x771518: aspect argument of 0xe66a60
    0x1097d90, // 0x838c53: projection builder
    0x109921c, // 0x856901: projection builder
    0x109ae2c, // 0x8696da: projection builder
    0x109aecc, // 0x873a81: projection builder
    0x109c504, // 0x888e1e: projection builder
    0xfdc588,  // 0x2ff84c: per-object projection behind the characters' shadows (with 9/16 below)
};
// And 9/16 where the same code takes the inverse.
constexpr uint32_t kInverseAspectFloats[] = {
    0xfdc590, // 0x2ff8ab
};
constexpr uint32_t k9By16 = 0x3f100000;
constexpr uint32_t k16By9 = 0x3fe38e39; // 16/9 as a float

uint32_t gOutW = 0, gOutH = 0; // the output size when not 16:9 (hudLayout)

bool patchAspect(uint8_t* text, float aspect) {
    for (uint32_t at : kInverseAspectFloats) {
        uint32_t v;
        std::memcpy(&v, text + at, 4);
        if (v != k9By16) {
            std::fprintf(stderr, "[display] unexpected eboot data at 0x%x: the 3D view stays 16:9\n", at);
            return false;
        }
    }
    for (uint32_t at : kAspectFloats) {
        uint32_t v;
        std::memcpy(&v, text + at, 4);
        if (v != k16By9) {
            std::fprintf(stderr, "[display] unexpected eboot data at 0x%x: the 3D view stays 16:9\n", at);
            return false;
        }
    }
    for (uint32_t at : kAspectFloats) std::memcpy(text + at, &aspect, 4);
    const float inverse = 1 / aspect;
    for (uint32_t at : kInverseAspectFloats) std::memcpy(text + at, &inverse, 4);
    return true;
}

} // namespace

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
    // gets every rendered pixel; at or below it keeps the 1920x1080 output and scales up. A picture
    // that is not 16:9 (Aspect Ratio) is output at its own size too, whatever it is: the scene
    // fills it and the HUD is drawn 16:9 in the middle (hudLayout).
    bool bigger = width > 1920 || height > 1080;
    const bool wide = std::fabs(double(width) / height - 16.0 / 9) > 0.02;
    bool sized = bigger || wide;
    // 0x55e820 fills in the setup's display size, 1920x1080 packed as width | height << 12
    // (movl $0x1438780, 0x50(%rdi)): the output size instead.
    static const uint8_t kSizeWas[] = {0xC7, 0x47, 0x50, 0x80, 0x87, 0x43, 0x01};
    // The base PS4's video memory setup (0xe7bd30) still caps the display buffers at 1920x1080
    // (the 1080p frame was then scaled back up to the window). Above 1080p it now keeps the output
    // size and reserves the extra 100 MB for the buffers, as its PS4 Pro branch does for 4K.
    struct Bytes {
        uint32_t at;
        std::vector<uint8_t> was, now;
    };
    static const Bytes kDisplay[] = {
        // movq $0x2500000, (%rsi): the display pool's size -> 0x8900000
        {0xe7beb2, {0x48, 0xC7, 0x06, 0x00, 0x00, 0x50, 0x02}, {0x48, 0xC7, 0x06, 0x00, 0x00, 0x90, 0x08}},
        // jb past the clamp to 1920x1080 -> jmp
        {0xe7beb9, {0x72, 0x13}, {0xEB, 0x13}},
    };
    if (sized) {
        bool match = width < 4096 && height < 4096 && std::equal(kSizeWas, kSizeWas + 7, text + 0x55e822);
        if (bigger)
            for (const Bytes& c : kDisplay) match = match && std::equal(c.was.begin(), c.was.end(), text + c.at);
        if (!match) {
            std::fprintf(stderr, "[display] unexpected eboot code at 0x55e822/0xe7beb2: output stays 1920x1080\n");
            if (wide) return false; // a squashed picture is worse than the game's own
            sized = bigger = false;
        }
    }
    if (wide && !patchAspect(text, float(double(width) / height))) return false;
    for (const Store& s : kStores) {
        if (s.output && !sized) continue;
        const uint16_t v = uint16_t(s.isWidth ? width : height);
        text[s.at + 3] = uint8_t(v);
        text[s.at + 4] = uint8_t(v >> 8);
    }
    if (sized) {
        if (bigger)
            for (const Bytes& c : kDisplay) std::copy(c.now.begin(), c.now.end(), text + c.at);
        const uint32_t packed = 0x1000000u | (height & 0xFFF) << 12 | (width & 0xFFF);
        for (int i = 0; i < 4; ++i) text[0x55e825 + i] = uint8_t(packed >> (8 * i));
    }
    if (wide) gOutW = width, gOutH = height;
    gActiveW = width, gActiveH = height;
    std::fprintf(stderr, "[display] rendering at %ux%u (output %ux%u)%s\n", width, height, sized ? width : 1920,
                 sized ? height : 1080, wide ? ", 3D view widened to match, HUD kept 16:9" : "");
    return true;
}

bool hudLayout(uint32_t* outWidth, uint32_t* outHeight) {
    *outWidth = gOutW, *outHeight = gOutH;
    return gOutW != 0;
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

// The clock's choice of step (0xd9ee98-0xd9eef5, measured ns in %eax, the result in %ecx) is
// handed to stepNs below:
//   mov %eax,%edi; movabs $stepNs,%rax; call *%rax; mov %eax,%ecx;
//   vmovss 0x88(%r14),%xmm0 (the raw step, which the call clobbered); jmp 0xd9eef5
constexpr uint32_t kStepAt = 0xd9ee98;
const std::vector<uint8_t> kStepWas = {0x49, 0x8b, 0x8e, 0x90, 0, 0, 0, 0x85, 0xc9, 0x74, 0x28, 0x48, 0xc1, 0xe9, 0x20,
                                       0xc5, 0xf9, 0x6e, 0xc9, 0xb9, 0x00, 0xc2, 0xeb, 0x0b, 0xc5, 0xf2, 0x5a, 0xc9,
                                       0xc5, 0xf3};
// The title screen's 30 fps flip rate, kept when uncapped with LPN_KEEP_30FPS_SCENES=1.
bool keep30FpsScenes() {
    static const bool keep = [] {
        const char* v = std::getenv("LPN_KEEP_30FPS_SCENES");
        return v && std::strcmp(v, "0") != 0;
    }();
    return keep;
}

uint8_t* gEbootText = nullptr; // set once the code is known to match
std::atomic<int> gClockWanted{0};
std::atomic<int> gClockActive{0};

// The frame clock object (0x2486080), from +0x60.
struct Clock {
    float dt;            // +0x60
    int32_t ticks;       // +0x64, dt in 1/3000 s, rounded
    uint8_t pad0[0x10];
    float time;          // +0x78
    float rawTime;       // +0x7c
    uint32_t totalTicks; // +0x80
    float overrideDt;    // +0x84, a one-shot step the clock takes and clears
    float rawDt;         // +0x88
    float pad1;
    uint32_t fixedOn;    // +0x90
    float fixedDt;       // +0x94
    uint32_t flags;      // +0x98: bit 0 paused, bit 2 step 1/60
};
static_assert(offsetof(Clock, time) == 0x18 && offsetof(Clock, overrideDt) == 0x24 && offsetof(Clock, fixedOn) == 0x30);
constexpr uint32_t kClock = 0x2486080 + 0x60;
Clock& clock() { return *reinterpret_cast<Clock*>(gEbootText + kClock); }

// How each frame's step was chosen, for the [clock] lines.
struct StepCounts {
    int fixed = 0, forced = 0, floored = 0, capped = 0, override = 0;
};
StepCounts gSteps;
int64_t gCarryNs = 0; // measured time not yet given out as whole 1/3000 s ticks

// The title sequence (made at 0x9bdd16) asks for the same 1/30 step as cutscenes. It runs from
// its push of that request (kTitleStepCaller) to its destructor's pop of it (0x9bc170 and
// 0x9bc1b0 call the pop, 0xd6e50, returning to kTitlePopCallers).
std::atomic<bool> gTitleAlive{false};
constexpr uint32_t kPopStep = 0xd6e50;
constexpr uint32_t kTitlePopCallers[] = {0x9bc1a2, 0x9bc1e2};
const uint8_t kPopWas[] = {0x48, 0x63, 0x87, 0x98, 0, 0, 0, 0xc7, 0x84, 0x87, 0x9c, 0, 0, 0};

bool titleRunning() { return gTitleAlive.load(std::memory_order_relaxed); }

// Skip intro logos: on from the first flip until the title sequence first starts.
std::atomic<bool> gTitleSeen{false};
constexpr double kSkipLimitS = 60; // in case the title is never seen
bool gStepHooks = false;            // pushStep is in place, so the title can be seen

// The 1/30 cutscene and 1/20 loading steps stay fixed. (Cutscenes stepped by the measured time
// flashed frames from later in the scene, so they stay at 30.) The game's table holds 1/60 as
// this same float.
bool fixedStepKept(const Clock& c) { return c.fixedOn && c.fixedDt > 1.0f / 60; }

constexpr uint32_t kNs60 = 16666666, kMaxFixedNs = 200000000, kMaxNs = 50000000;
constexpr uint32_t kFloorNs = 2000000; // uncapped: 500 fps

// The step, in ns, the clock takes this frame (the game's thread). The game's own rules: a fixed
// step when one is on, else 1/60 when flag bit 2 is set, else the measured time rounded up to 1/60
// below 17.25 ms and capped at 50 ms. With a 30 fps cap the 1/60 fixed step and flag bit 2 take
// the measured time instead (the title's menus set bit 2 most frames, and would run at half
// speed). Uncapped, the floor drops to 2 ms, and the step is a whole number of the
// clock's 1/3000 s ticks, carrying the rest, so that its rounded tick count (+0x64) keeps real
// time at rates such as 144 (20.8 ticks a frame, otherwise 21: 0.8% fast).
extern "C" uint32_t stepNs(uint32_t measured) {
    const Clock& c = clock();
    const bool uncapped = gClockActive.load(std::memory_order_relaxed) == 2;
    if (c.overrideDt > 0) ++gSteps.override; // the clock replaces this step after the call
    if (skippingIntro()) return std::max<uint32_t>(kMaxNs, fixedStepKept(c) ? uint32_t(double(c.fixedDt) * 1e9) : 0);
    if (fixedStepKept(c)) {
        ++gSteps.fixed;
        const double ns = double(c.fixedDt) * 1e9;
        return ns <= kMaxFixedNs ? uint32_t(ns) : kMaxFixedNs;
    }
    if (c.flags & 4) ++gSteps.forced;
    uint32_t ns = measured;
    if (!uncapped) {
        if (ns < 17250000) ns = kNs60, ++gSteps.floored;
        return std::min(ns, kMaxNs);
    }
    if (ns < kFloorNs) ns = kFloorNs, ++gSteps.floored;
    if (ns > kMaxNs) ns = kMaxNs, ++gSteps.capped, gCarryNs = 0;
    const int64_t total = gCarryNs + ns;
    const int64_t ticks = std::max<int64_t>(1, (total * 3 + 500000) / 1000000);
    const int64_t out = (ticks * 1000000 + 1) / 3;
    gCarryNs = std::clamp<int64_t>(total - out, -int64_t(kFloorNs), int64_t(kFloorNs));
    return uint32_t(out);
}

CodePatch stepPatch() {
    std::vector<uint8_t> now = {0x89, 0xc7, 0x48, 0xb8};
    const uint64_t fn = reinterpret_cast<uint64_t>(guestEntry<&stepNs>());
    for (int i = 0; i < 8; ++i) now.push_back(uint8_t(fn >> (8 * i)));
    const uint8_t tail[] = {0xff, 0xd0, 0x89, 0xc1, 0xc4, 0xc1, 0x7a, 0x10, 0x86, 0x88, 0, 0, 0, 0xe9};
    now.insert(now.end(), std::begin(tail), std::end(tail));
    const int32_t rel = int32_t(0xd9eef5 - (kStepAt + kStepWas.size()));
    for (int i = 0; i < 4; ++i) now.push_back(uint8_t(rel >> (8 * i)));
    return {kStepAt, kStepWas, now};
}

// Scenes ask for a fixed step by pushing a request on a stack (0xd6e20 pushes a step index,
// 0xd6dd0 the same with bit 31 set, which the clock takes as no fixed step; 0xd4f00 applies
// them). These replacements do the same and log who asks for what with LPN_PROFILE=1.
constexpr uint32_t kPushStep = 0xd6e20, kPushMeasured = 0xd6dd0, kWake = 0x108030, kWakeArg = 0x247e630 + 0x178;
constexpr uint32_t kTitleStepCaller = 0x9bdd60; // return address of the title sequence's push of step 2
const uint8_t kPushWas[] = {0x48, 0x63, 0x87, 0x98, 0, 0, 0};

uint64_t pushRequest(uint8_t* stack, uint32_t value, uintptr_t caller) {
    int32_t& n = *reinterpret_cast<int32_t*>(stack + 0x98);
    reinterpret_cast<uint32_t*>(stack + 0x9c)[n] = value;
    ++n;
    static const bool log = std::getenv("LPN_PROFILE") != nullptr;
    static std::atomic<int> logged{0};
    if (log && logged.fetch_add(1, std::memory_order_relaxed) < 400)
        std::fprintf(stderr, "[step] request 0x%x from eboot+0x%lx at flip %llu\n", value,
                     (unsigned long)(caller - uintptr_t(gEbootText)), (unsigned long long)flipCount());
    if (value & 0x7fffffff) return 0;
    auto wake = reinterpret_cast<uint64_t (LPN_GUEST*)(void*)>(gEbootText + kWake);
    return wake(*reinterpret_cast<void**>(gEbootText + kWakeArg));
}

extern "C" uint64_t pushStep(uint8_t* stack, uint32_t index) {
    const uintptr_t caller = uintptr_t(LPN_GUEST_RETURN_ADDRESS());
    if (caller - uintptr_t(gEbootText) == kTitleStepCaller) {
        gTitleAlive = true;
        if (!gTitleSeen.exchange(true) && gSettings.skipIntro)
            std::fprintf(stderr, "[boot] intro logos skipped: title at flip %llu\n", (unsigned long long)flipCount());
    }
    return pushRequest(stack, index, caller);
}

extern "C" void popStep(uint8_t* stack) {
    const uintptr_t at = uintptr_t(LPN_GUEST_RETURN_ADDRESS()) - uintptr_t(gEbootText);
    int32_t& n = *reinterpret_cast<int32_t*>(stack + 0x98);
    reinterpret_cast<uint32_t*>(stack + 0x9c)[n] = 0xffffffffu;
    ++n;
    if (at != kTitlePopCallers[0] && at != kTitlePopCallers[1]) return;
    gTitleAlive = false;
    static const bool log = std::getenv("LPN_PROFILE") != nullptr;
    if (log) std::fprintf(stderr, "[step] title ends at flip %llu\n", (unsigned long long)flipCount());
}

extern "C" uint64_t pushMeasured(uint8_t* stack, uint32_t index) {
    return pushRequest(stack, index ? index | 0x80000000u : 0, uintptr_t(LPN_GUEST_RETURN_ADDRESS()));
}

void jumpTo(uint32_t at, const void* fn) {
    // movabs $fn, %rax; jmp *%rax
    uint8_t jump[12] = {0x48, 0xb8};
    const uint64_t target = reinterpret_cast<uint64_t>(fn);
    std::memcpy(jump + 2, &target, 8);
    jump[10] = 0xff, jump[11] = 0xe0;
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    uint8_t* p = gEbootText + at;
    void* start = reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1));
    // Left writable: the loader still rewrites thread pointer reads on these pages, and
    // Module::protect sets the final protection afterwards.
    mprotect(start, page * 2, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::memcpy(p, jump, sizeof jump);
}

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
    if (!matches(stepPatch())) return false;
    gEbootText = eboot.base();
    if (std::memcmp(gEbootText + kPushStep, kPushWas, sizeof kPushWas) == 0 &&
        std::memcmp(gEbootText + kPushMeasured, kPushWas, sizeof kPushWas) == 0 &&
        std::memcmp(gEbootText + kPopStep, kPopWas, sizeof kPopWas) == 0) {
        jumpTo(kPushStep, guestHook<&pushStep>());
        jumpTo(kPushMeasured, guestHook<&pushMeasured>());
        jumpTo(kPopStep, guestHook<&popStep>());
        gStepHooks = true;
    }
    return true;
}

// A scaled game clock (LPN_TIME_SCALE) needs the measured time unfloored, at any cap.
void setFrameClock(int fps) {
    gClockWanted = fps <= 0 || fps > 61 || gameTimeScale() != 1.0 ? 2 : fps < 59 ? 1 : 0;
}

namespace {

// LPN_PROFILE=1: a [clock] line a second with what the game's frame clock (0x2486080) stepped:
// frame times, game seconds per real second for its float time and its 1/3000 s ticks, how many
// frames were too short to count as one 60 Hz or 30 Hz frame when rounded, and how many steps
// were fixed, asked as 1/60 by flag bit 2 (now measured), floored, capped or overridden.
void clockStats() {
    static const bool enabled = std::getenv("LPN_PROFILE") != nullptr;
    if (!enabled) return;
    const Clock& c = clock();
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
    const StepCounts n = gSteps;
    gSteps = {};
    std::fprintf(stderr,
                 "[clock] t=%d fps=%.0f dt=%.2f/%.2f/%.2f ms speed=%.3f ticks=%.3f round0@60=%d round0@30=%d "
                 "fixed=%u/%.4f steps fixed=%d bit2=%d floored=%d capped=%d override=%d\n",
                 ++second, frames / wall, dtMin * 1e3f, dtSum / frames * 1e3f, dtMax * 1e3f, (c.time - time0) / wall,
                 (c.totalTicks - ticks0) / 3000.0 / wall, under60, under30, c.fixedOn, c.fixedDt, n.fixed, n.forced,
                 n.floored, n.capped, n.override);
    start = Wall::now(), time0 = c.time, ticks0 = c.totalTicks;
    frames = under60 = under30 = 0;
    dtMin = 1e9f, dtMax = dtSum = 0;
}

} // namespace

bool frameClockUncapped() { return gClockActive.load(std::memory_order_relaxed) == 2; }

bool skippingIntro() {
    if (!gSettings.skipIntro || !gStepHooks || gTitleSeen.load(std::memory_order_relaxed)) return false;
    using Wall = std::chrono::steady_clock;
    static const Wall::time_point start = Wall::now();
    if (std::chrono::duration<double>(Wall::now() - start).count() < kSkipLimitS) return true;
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) std::fprintf(stderr, "[boot] title not seen after %.0f s: intro skip stopped\n", kSkipLimitS);
    return false;
}

namespace {

// Uncapped, the title menu runs with its 1/30 step turned off, as the clock's own off switch
// (0xd9f0f0, patched to clear the step too) leaves it: measured steps and no 30 fps flip
// interval. (Measured steps with the fixed step still on stopped the menu's options appearing.)
// The title's pop makes 0xd4f00 set the step of the scene below again.
void titleStep(bool uncapped) {
    Clock& c = clock();
    if (uncapped && titleRunning() && c.fixedOn && c.fixedDt == 1.0f / 30) c.fixedOn = 0, c.fixedDt = 0;
}

} // namespace

bool uncap30FpsScenes() {
    return frameClockUncapped() && !keep30FpsScenes() && titleRunning();
}

void syncFrameClock() {
    if (gEbootText) clockStats();
    // Skipping the intro, the host picks every step (stepNs), whatever the cap.
    const int want = skippingIntro() ? std::max(1, gClockWanted.load()) : gClockWanted.load();
    if (gEbootText) titleStep(want == 2 && gClockActive == want && !keep30FpsScenes());
    if (!gEbootText || want == gClockActive) return;
    write(kMeasured[0], want >= 1);
    write(kMeasured[1], want >= 1);
    write(stepPatch(), want >= 1);
    gCarryNs = 0;
    gClockActive = want;
    std::fprintf(stderr, "[display] frame clock: %s\n",
                 want == 2 ? "measured time, uncapped" : want == 1 ? "measured time" : "the game's");
}

} // namespace host
