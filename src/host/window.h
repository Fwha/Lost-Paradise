// The game window. One thread owns SDL and shows whatever VideoOut flips to.
#pragma once

#include <cstdint>

namespace host {

struct Frame {
    const void* pixels = nullptr;
    uint32_t width = 0, height = 0, pitchPixels = 0;
    uint32_t pixelFormat = 0; // SCE_VIDEO_OUT_PIXEL_FORMAT_*
    uint32_t tilingMode = 0;  // 0 tiled, 1 linear
};

// Open the window (or do nothing when headless or built without SDL).
void startPresenter(bool headless);
// Copy a flipped buffer for display. Safe to call from any thread.
void presentFrame(const Frame& f);

// Controller state from the keyboard or a connected game controller, in PS4 terms.
struct PadState {
    uint32_t buttons = 0; // SCE_PAD_BUTTON_* bits
    uint8_t lx = 0x80, ly = 0x80, rx = 0x80, ry = 0x80, l2 = 0, r2 = 0;
};
PadState padState();

} // namespace host
