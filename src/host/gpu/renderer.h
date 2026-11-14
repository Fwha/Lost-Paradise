// The Vulkan renderer: replays the game's draws and dispatches as they come out of the PM4
// processor, with shaders recompiled from GCN.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace host::render {

// Starts the renderer and registers it as a draw listener. Returns false (and logs why)
// when no usable Vulkan device exists; the game then runs without rendering.
// computeOnly runs compute dispatches (whose results the game may read back) but skips draws:
// much faster on a software Vulkan driver, for runs that only need the game to progress.
// Makes the Vulkan device now, before the guest's address space is reserved (Windows, with DLSS
// chosen): NVIDIA's VK_NVX_binary_import, which DLSS needs, sets up CUDA, whose address space the
// guest's reservation would otherwise take. startRenderer then uses it.
void prepareVulkan();
bool startRenderer(bool computeOnly = false);

// Reading back what the GPU rendered at a display buffer's address takes two steps, so the
// vblank clock never waits for the GPU: queueDisplayCopy records the copy into one of
// kDisplaySlots readback slots (c.slot, chosen by the caller and not in use) and returns false
// when nothing was rendered there; fetchDisplayCopy waits for it and returns the pixels in the
// buffer's memory layout, as linear rows of `width` 32-bit pixels.
constexpr int kDisplaySlots = 3;
struct DisplayCopy {
    int slot = 0;
    uint64_t seq = 0; // the batch holding the copy
    uint32_t w = 0, h = 0;
};
bool queueDisplayCopy(uint64_t address, uint32_t width, uint32_t height, DisplayCopy& c);
void fetchDisplayCopy(const DisplayCopy& c, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels);

// A mark for all GPU work recorded so far, and whether the GPU has finished it: a flip shows
// its buffer only once the frame's rendering is done, as on the console.
uint64_t gpuMark();
bool gpuDone(uint64_t mark);

// Traces the next frame (as LPN_GPU_TRACE does) into `dir`: trace.txt and every render target.
void captureNextFrame(const std::string& dir);

// Compile Shaders at Startup: while the game waits for every saved shader and recorded pipeline
// to be built, the window shows this progress (phase 1 translating shaders, 2 building
// pipelines; 0 before either has a count). False when nothing is being waited for.
struct WarmupProgress {
    int phase = 0;
    size_t done = 0, total = 0;
    double seconds = 0;
};
bool warmupProgress(WarmupProgress& p);
// The boot compile (docs/shader-precompile.md stage 4): every shader the game registers, compiled
// in the background from the title menu on. False when it is not running.
bool bootCompileProgress(size_t& done, size_t& total);
// Starts the game now; what is left builds in the background as when the setting is off.
void skipWarmup();
// The most anisotropic filtering the GPU does (1: none; 16 before the renderer starts).
int maxAnisotropy();

// The game mapped direct or flexible memory at [address, address + size): the renderer imports it into
// Vulkan ahead of the draws that read it, where a draw doing so would hitch (any thread).
void noteDirectMapping(uint64_t address, uint64_t size);

// Before exiting: writes the driver's pipeline cache, so pipelines made since the last periodic
// save are not compiled again next run.
void saveCaches();

} // namespace host::render
