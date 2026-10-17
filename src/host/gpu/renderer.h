// The Vulkan renderer: replays the game's draws and dispatches as they come out of the PM4
// processor, with shaders recompiled from GCN.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace host::render {

// Starts the renderer and registers it as a draw listener. Returns false (and logs why)
// when no usable Vulkan device exists; the game then runs without rendering.
// computeOnly runs compute dispatches (whose results the game may read back) but skips draws:
// much faster on a software Vulkan driver, for runs that only need the game to progress.
bool startRenderer(bool computeOnly = false);

// Copies the image the GPU rendered at a display buffer's address, in the buffer's memory
// layout as linear rows of `width` 32-bit pixels. False when nothing was rendered there.
bool readDisplayBuffer(uint64_t address, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels);

// Traces the next frame (as LPN_GPU_TRACE does) into `dir`: trace.txt and every render target.
void captureNextFrame(const std::string& dir);

} // namespace host::render
