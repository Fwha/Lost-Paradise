// libSceVideoOut. A host thread acts as the display: at each vblank it takes the next
// queued flip, shows the buffer and fires the flip events the game waits on.
#include "host/videoout.h"

#include "host/equeue.h"
#include "host/hle.h"
#include "host/kernel.h"
#include "host/window.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

namespace host {

namespace {

constexpr int32_t kHandle = 1;
constexpr uint64_t kEventFlip = 0, kEventVblank = 1;
constexpr int32_t kErrInvalidHandle = int32_t(0x8029000B);
constexpr int32_t kErrInvalidIndex = int32_t(0x80290005);
constexpr int32_t kErrFlipQueueFull = int32_t(0x80290012);

struct BufferAttribute {
    int32_t pixelFormat, tilingMode, aspectRatio;
    uint32_t width, height, pitchInPixel, option, reserved0;
    uint64_t reserved1;
};
static_assert(sizeof(BufferAttribute) == 40);

struct FlipStatus {
    uint64_t count, processTime, tsc;
    int64_t flipArg;
    uint64_t submitTsc, reserved0;
    int32_t gcQueueNum, flipPendingNum, currentBuffer;
    uint32_t reserved1;
};
static_assert(sizeof(FlipStatus) == 64);

struct ResolutionStatus {
    uint32_t fullWidth, fullHeight, paneWidth, paneHeight;
    uint64_t refreshRate;
    float screenSizeInInch;
    uint16_t flags, reserved0;
    uint32_t reserved1[3];
};

struct Flip {
    int32_t index;
    int64_t arg;
    uint64_t submitTime;
};

struct Display {
    std::mutex m;
    std::condition_variable vblank;
    bool open = false;
    struct {
        void* address = nullptr;
        BufferAttribute attr{};
    } buffers[16];
    std::deque<Flip> pending;
    FlipStatus status{};
    uint64_t vblankCount = 0;
    int32_t flipRate = 0; // 0: every vblank, 1: every 2nd, 2: every 3rd
} gDisplay;

double gVblankHz = 59.94;
using Clock = std::chrono::steady_clock;
const Clock::time_point gStart = Clock::now();
uint64_t nowUs() { return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - gStart).count(); }

void vblankThread() {
    auto next = Clock::now();
    for (;;) {
        next += std::chrono::nanoseconds(int64_t(1e9 / gVblankHz));
        std::this_thread::sleep_until(next);
        Frame frame;
        bool flipped = false;
        int64_t flipArg = 0;
        {
            std::lock_guard lock(gDisplay.m);
            ++gDisplay.vblankCount;
            if (!gDisplay.pending.empty() && gDisplay.vblankCount % (gDisplay.flipRate + 1) == 0) {
                const Flip f = gDisplay.pending.front();
                gDisplay.pending.pop_front();
                auto& s = gDisplay.status;
                s.count++;
                s.processTime = nowUs();
                s.tsc = s.processTime;
                s.submitTsc = f.submitTime;
                s.flipArg = f.arg;
                s.currentBuffer = f.index;
                s.flipPendingNum = int32_t(gDisplay.pending.size());
                flipArg = f.arg;
                flipped = true;
                if (f.index >= 0) {
                    const auto& b = gDisplay.buffers[f.index];
                    frame = {b.address, b.attr.width, b.attr.height, b.attr.pitchInPixel, uint32_t(b.attr.pixelFormat),
                             uint32_t(b.attr.tilingMode)};
                }
            }
        }
        gDisplay.vblank.notify_all();
        triggerEvent(kEventVblank, kFilterVideoOut, int64_t(gDisplay.vblankCount));
        if (flipped) {
            presentFrame(frame);
            triggerEvent(kEventFlip, kFilterVideoOut, flipArg);
        }
    }
}

int32_t sceVideoOutOpen(int32_t, int32_t busType, int32_t, const void*) {
    if (busType != 0) return kErrInvalidHandle;
    std::lock_guard lock(gDisplay.m);
    if (!gDisplay.open) {
        gDisplay.open = true;
        gDisplay.status.currentBuffer = -1;
        std::thread(vblankThread).detach();
    }
    return kHandle;
}

int32_t sceVideoOutClose(int32_t) { return 0; }

void sceVideoOutSetBufferAttribute(BufferAttribute* a, int32_t pixelFormat, int32_t tilingMode, int32_t aspect,
                                   uint32_t width, uint32_t height, uint32_t pitch) {
    *a = {pixelFormat, tilingMode, aspect, width, height, pitch, 0, 0, 0};
}

int32_t sceVideoOutRegisterBuffers(int32_t handle, int32_t start, void* const* addresses, int32_t num,
                                   const BufferAttribute* attr) {
    if (handle != kHandle) return kErrInvalidHandle;
    if (start < 0 || num <= 0 || start + num > 16) return kErrInvalidIndex;
    std::lock_guard lock(gDisplay.m);
    for (int32_t i = 0; i < num; ++i) gDisplay.buffers[start + i] = {addresses[i], *attr};
    std::fprintf(stderr, "[videoout] %d buffers at index %d: %ux%u pitch %u format 0x%x tiling %d\n", num, start,
                 attr->width, attr->height, attr->pitchInPixel, attr->pixelFormat, attr->tilingMode);
    return 0; // buffer set id
}

int32_t sceVideoOutUnregisterBuffers(int32_t, int32_t) { return 0; }

int32_t sceVideoOutSetFlipRate(int32_t handle, int32_t rate) {
    if (handle != kHandle) return kErrInvalidHandle;
    std::lock_guard lock(gDisplay.m);
    gDisplay.flipRate = rate;
    return 0;
}

int32_t sceVideoOutAddFlipEvent(EventQueue* q, int32_t handle, void* udata) {
    if (handle != kHandle) return kErrInvalidHandle;
    if (!q) return sceError(ps4errno::kEBADF);
    q->add(kEventFlip, kFilterVideoOut, udata);
    return 0;
}

int32_t sceVideoOutDeleteFlipEvent(EventQueue* q, int32_t) {
    if (q) q->remove(kEventFlip, kFilterVideoOut);
    return 0;
}

int32_t sceVideoOutSubmitFlip(int32_t handle, int32_t index, int32_t mode, int64_t arg) {
    return videoOutSubmitFlip(handle, index, mode, arg);
}

int32_t sceVideoOutGetFlipStatus(int32_t handle, FlipStatus* s) {
    if (handle != kHandle) return kErrInvalidHandle;
    std::lock_guard lock(gDisplay.m);
    *s = gDisplay.status;
    s->flipPendingNum = int32_t(gDisplay.pending.size());
    return 0;
}

int32_t sceVideoOutIsFlipPending(int32_t handle) {
    std::lock_guard lock(gDisplay.m);
    return handle == kHandle ? int32_t(gDisplay.pending.size()) : kErrInvalidHandle;
}

int32_t sceVideoOutWaitVblank(int32_t handle) {
    if (handle != kHandle) return kErrInvalidHandle;
    std::unique_lock lock(gDisplay.m);
    const uint64_t target = gDisplay.vblankCount + 1;
    gDisplay.vblank.wait(lock, [&] { return gDisplay.vblankCount >= target; });
    return 0;
}

int32_t sceVideoOutGetResolutionStatus(int32_t handle, ResolutionStatus* s) {
    if (handle != kHandle) return kErrInvalidHandle;
    *s = {};
    s->fullWidth = s->paneWidth = 1920;
    s->fullHeight = s->paneHeight = 1080;
    s->refreshRate = 3; // SCE_VIDEO_OUT_REFRESH_RATE_59_94HZ
    s->screenSizeInInch = 50;
    return 0;
}

int32_t ok() { return 0; }

} // namespace

int32_t videoOutSubmitFlip(int32_t handle, int32_t index, int32_t, int64_t arg) {
    if (handle != kHandle) return kErrInvalidHandle;
    if (index < -1 || index >= 16) return kErrInvalidIndex;
    std::lock_guard lock(gDisplay.m);
    if (gDisplay.pending.size() >= 16) return kErrFlipQueueFull;
    gDisplay.pending.push_back({index, arg, nowUs()});
    return 0;
}

void setVblankRate(double hz) { gVblankHz = hz; }

void registerVideoOutHle() {
    const char* v = "libSceVideoOut";
    HLE(v, sceVideoOutOpen);
    HLE(v, sceVideoOutClose);
    HLE(v, sceVideoOutSetBufferAttribute);
    HLE(v, sceVideoOutRegisterBuffers);
    HLE(v, sceVideoOutUnregisterBuffers);
    HLE(v, sceVideoOutSetFlipRate);
    HLE(v, sceVideoOutAddFlipEvent);
    HLE(v, sceVideoOutDeleteFlipEvent);
    HLE(v, sceVideoOutSubmitFlip);
    HLE(v, sceVideoOutGetFlipStatus);
    HLE(v, sceVideoOutIsFlipPending);
    HLE(v, sceVideoOutWaitVblank);
    HLE(v, sceVideoOutGetResolutionStatus);
    // Display settings and the hardware cursor don't matter on a PC window.
    for (const char* name : {"sceVideoOutAdjustColor_", "sceVideoOutColorSettingsSetGamma_",
                             "sceVideoOutConfigureOutputMode_", "sceVideoOutModeSetAny_",
                             "sceVideoOutSetWindowModeMargins", "sceVideoOutCursorEnable", "sceVideoOutCursorDisable",
                             "sceVideoOutCursorSetHotSpot", "sceVideoOutCursorSetImageAddress",
                             "sceVideoOutCursorSetPosition"})
        registerHle(v, name, reinterpret_cast<void*>(&ok));
}

} // namespace host
