// VideoOut: display buffers, the vblank clock and flips.
#pragma once

#include <cstdint>

namespace host {

// Queue a flip of a registered buffer (index -1 flips to blank). Returns an SCE error or 0.
int32_t videoOutSubmitFlip(int32_t handle, int32_t bufferIndex, int32_t flipMode, int64_t flipArg);

// Display refresh rate the vblank clock runs at (59.94 by default; --vblank). The game's own flip
// rates keep their PS4 timing whatever the rate.
void setVblankRate(double hz);
// The monitor's refresh rate while uncapped, 0 to go back to 59.94 (the window calls this; a
// --vblank rate wins). It can change while the game runs.
void setDisplayRefresh(double hz);
// Caps the frame rate at the refresh rate divided by a whole number (60 or 30 at 59.94 Hz):
// steadier motion than an uneven rate between them on a fixed-refresh screen. 0: no cap.
void setFpsCap(double fps);

// Write flip number `flip` (counting from 1) to `path` as a PPM image.
void setScreenshot(uint64_t flip, const char* path);
// Saves every Nth displayed frame as screenshot-<N>.ppm.
void setScreenshotEvery(uint64_t flips);
// F12: saves the next displayed frame and a GPU trace of the frame after it to
// lpn-captures/capture-<n>/.
void requestCapture();
// Frames displayed so far.
uint64_t flipCount();

void registerVideoOutHle();

} // namespace host
