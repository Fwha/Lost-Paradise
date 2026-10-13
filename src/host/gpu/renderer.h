// The Vulkan renderer: replays the game's draws and dispatches as they come out of the PM4
// processor, with shaders recompiled from GCN.
#pragma once

#include <cstdint>
#include <vector>

namespace host::render {

// Starts the renderer and registers it as a draw listener. Returns false (and logs why)
// when no usable Vulkan device exists; the game then runs without rendering.
bool startRenderer();

// Copies the image the GPU rendered at a display buffer's address, in the buffer's memory
// layout as linear rows of `width` 32-bit pixels. False when nothing was rendered there.
bool readDisplayBuffer(uint64_t address, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels);

} // namespace host::render
