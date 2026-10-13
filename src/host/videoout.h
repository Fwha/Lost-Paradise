// VideoOut: display buffers, the vblank clock and flips.
#pragma once

#include <cstdint>

namespace host {

// Queue a flip of a registered buffer (index -1 flips to blank). Returns an SCE error or 0.
int32_t videoOutSubmitFlip(int32_t handle, int32_t bufferIndex, int32_t flipMode, int64_t flipArg);

// Display refresh rate the vblank clock runs at (59.94 by default).
void setVblankRate(double hz);

void registerVideoOutHle();

} // namespace host
