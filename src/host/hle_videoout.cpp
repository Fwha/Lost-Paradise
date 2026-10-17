// libSceVideoOut. A host thread acts as the display: at each vblank it takes the next
// queued flip whose rendering the GPU has finished and fires the flip events the game waits
// on; a second thread reads the buffer back and shows it, so the vblank clock never waits.
#include "host/videoout.h"

#include "host/equeue.h"
#include "host/gpu/renderer.h"
#include "host/hle.h"
#include "host/kernel.h"
#include "host/window.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    uint64_t gpuMark; // the GPU work that renders it
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
double gFpsCap = 30;   // even 30 fps; uncapped, crowded areas swing between 20 and 30
int32_t gMinFlipRate = 1; // from the cap
uint64_t gScreenshotFlip = 0;
uint64_t gScreenshotEvery = 0;
std::string gScreenshotPath;
std::atomic<bool> gCaptureRequested{false};

// Pixels in memory order are B, G, R, A for the formats the game uses.
void writeScreenshot(const Frame& f, const std::string& path) {
    FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) return;
    std::fprintf(out, "P6\n%u %u\n255\n", f.width, f.height);
    const auto* px = static_cast<const uint8_t*>(f.pixels);
    std::vector<uint8_t> row(size_t(f.width) * 3);
    for (uint32_t y = 0; y < f.height; ++y) {
        for (uint32_t x = 0; x < f.width; ++x) {
            const uint8_t* p = px + (size_t(y) * f.pitchPixels + x) * 4;
            row[x * 3] = p[2];
            row[x * 3 + 1] = p[1];
            row[x * 3 + 2] = p[0];
        }
        std::fwrite(row.data(), 1, row.size(), out);
    }
    std::fclose(out);
    std::fprintf(stderr, "[videoout] wrote %s\n", path.c_str());
}
using Clock = std::chrono::steady_clock;
const Clock::time_point gStart = Clock::now();
uint64_t nowUs() { return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - gStart).count(); }

// A flipped frame on its way from the vblank thread to the display thread.
struct Shown {
    Frame frame;
    uint64_t flip = 0;
    bool rendered = false; // frame.pixels names a display buffer the GPU copy in `copy` reads
    render::DisplayCopy copy;
};

// The display thread's queue: one frame waiting and one being shown, each holding a readback
// slot, so the vblank thread always finds a free slot for the next.
struct {
    std::mutex m;
    std::condition_variable cv;
    bool waiting = false;
    Shown next;
    int showing = -1; // the slot the display thread reads
} gShow;

// LPN_PROFILE=1: how evenly frames reach the screen, printed every 300 flips.
const bool gPacingStats = getenv("LPN_PROFILE") != nullptr;

void showFrame(Shown& s) {
    std::vector<uint32_t> rendered;
    Frame frame = s.frame;
    if (s.rendered) {
        // Show what the GPU rendered into the buffer.
        render::fetchDisplayCopy(s.copy, frame.width, frame.height, rendered);
        frame.pixels = rendered.data();
        frame.pitchPixels = frame.width;
        frame.tilingMode = 1;
    }
    const uint64_t n = s.flip;
    if (frame.pixels && gScreenshotFlip && n == gScreenshotFlip) writeScreenshot(frame, gScreenshotPath);
    static const uint64_t captureAt = getenv("LPN_CAPTURE_AT") ? std::strtoull(getenv("LPN_CAPTURE_AT"), nullptr, 0) : 0;
    if (n == captureAt) gCaptureRequested = true; // a capture without pressing F12
    if (frame.pixels && gCaptureRequested.exchange(false)) {
        std::string dir;
        for (int i = 1; dir.empty() || std::filesystem::exists(dir); ++i) dir = "lpn-captures/capture-" + std::to_string(i);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        writeScreenshot(frame, dir + "/screenshot.ppm");
        std::fprintf(stderr, "[capture] flip %" PRIu64 " -> %s (tracing the next frame)\n", n, dir.c_str());
        render::captureNextFrame(dir);
    }
    if (frame.pixels && gScreenshotEvery && n % gScreenshotEvery == 0)
        writeScreenshot(frame, "screenshot-" + std::to_string(n) + ".ppm");
    presentFrame(frame);
}

void displayThread() {
    for (;;) {
        Shown s;
        {
            std::unique_lock lock(gShow.m);
            gShow.cv.wait(lock, [] { return gShow.waiting; });
            s = gShow.next;
            gShow.waiting = false;
            gShow.showing = s.rendered ? s.copy.slot : -1;
        }
        gShow.cv.notify_all();
        showFrame(s);
        {
            std::lock_guard lock(gShow.m);
            gShow.showing = -1;
        }
    }
}

// Hands a flipped frame to the display thread, queueing its GPU readback first. Waits only when
// the display thread is a whole frame behind.
void queueShow(const Frame& frame, uint64_t flip) {
    Shown s{frame, flip};
    std::unique_lock lock(gShow.m);
    gShow.cv.wait(lock, [] { return !gShow.waiting; });
    if (frame.pixels) {
        for (int i = 0; i < render::kDisplaySlots; ++i)
            if (i != gShow.showing) {
                s.copy.slot = i;
                break;
            }
        s.rendered = render::queueDisplayCopy(reinterpret_cast<uint64_t>(frame.pixels), frame.width, frame.height, s.copy);
    }
    gShow.next = s;
    gShow.waiting = true;
    gShow.cv.notify_all();
}

void vblankThread() {
    const auto period = std::chrono::nanoseconds(int64_t(1e9 / gVblankHz));
    auto next = Clock::now();
    uint64_t lastFlipVblank = 0, statFrom = 0;
    uint64_t intervals[5] = {}; // flips 1, 2, 3, 4, 5+ vblanks apart
    double lateMax = 0;
    for (;;) {
        next += period;
        std::this_thread::sleep_until(next);
        const auto now = Clock::now();
        if (gPacingStats) lateMax = std::max(lateMax, std::chrono::duration<double, std::milli>(now - next).count());
        if (now - next > period) next = now; // a stall: don't fire the missed vblanks in a burst
        Frame frame;
        bool flipped = false;
        int64_t flipArg = 0;
        uint64_t n = 0, vblank = 0;
        {
            std::lock_guard lock(gDisplay.m);
            vblank = ++gDisplay.vblankCount;
            if (!gDisplay.pending.empty() && gDisplay.vblankCount % (std::max(gDisplay.flipRate, gMinFlipRate) + 1) == 0 &&
                render::gpuDone(gDisplay.pending.front().gpuMark)) {
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
                n = s.count;
                if (f.index >= 0) {
                    const auto& b = gDisplay.buffers[f.index];
                    frame = {b.address, b.attr.width, b.attr.height, b.attr.pitchInPixel, uint32_t(b.attr.pixelFormat),
                             uint32_t(b.attr.tilingMode)};
                }
            }
        }
        gDisplay.vblank.notify_all();
        triggerEvent(kEventVblank, kFilterVideoOut, int64_t(vblank));
        if (flipped) {
            queueShow(frame, n);
            triggerEvent(kEventFlip, kFilterVideoOut, flipArg);
            if (gPacingStats) {
                if (lastFlipVblank) ++intervals[std::min<uint64_t>(vblank - lastFlipVblank, 5) - 1];
                lastFlipVblank = vblank;
                if (n % 300 == 0) {
                    std::fprintf(stderr,
                                 "[pacing] flips %" PRIu64 "-%" PRIu64 ": vblanks apart 1:%" PRIu64 " 2:%" PRIu64
                                 " 3:%" PRIu64 " 4:%" PRIu64 " 5+:%" PRIu64 "; vblank late by up to %.1f ms\n",
                                 statFrom, n, intervals[0], intervals[1], intervals[2], intervals[3], intervals[4], lateMax);
                    statFrom = n;
                    std::fill(std::begin(intervals), std::end(intervals), 0);
                    lateMax = 0;
                }
            }
        }
    }
}

} // namespace

void setScreenshot(uint64_t flip, const char* path) {
    gScreenshotFlip = flip;
    gScreenshotPath = path;
}

void setScreenshotEvery(uint64_t flips) { gScreenshotEvery = flips; }
void requestCapture() { gCaptureRequested = true; }

uint64_t flipCount() {
    std::lock_guard lock(gDisplay.m);
    return gDisplay.status.count;
}

namespace {

int32_t sceVideoOutOpen(int32_t, int32_t busType, int32_t, const void*) {
    if (busType != 0) return kErrInvalidHandle;
    std::lock_guard lock(gDisplay.m);
    if (!gDisplay.open) {
        gDisplay.open = true;
        gDisplay.status.currentBuffer = -1;
        std::thread(displayThread).detach();
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
    if (rate != gDisplay.flipRate)
        std::fprintf(stderr, "[videoout] flip rate %d (every %d vblanks) at flip %llu\n", rate, rate + 1,
                     (unsigned long long)gDisplay.status.count);
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
    const uint64_t mark = render::gpuMark();
    std::lock_guard lock(gDisplay.m);
    if (gDisplay.pending.size() >= 16) return kErrFlipQueueFull;
    gDisplay.pending.push_back({index, arg, nowUs(), mark});
    return 0;
}

void setVblankRate(double hz) {
    gVblankHz = hz;
    setFpsCap(gFpsCap);
}

void setFpsCap(double fps) {
    gFpsCap = fps;
    gMinFlipRate = fps > 0 ? std::max(0, int32_t(gVblankHz / fps + 0.5) - 1) : 0;
}

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
