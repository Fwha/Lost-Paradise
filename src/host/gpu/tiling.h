// GCN surface tiling (PS4 tile mode indices) to linear rows.
#pragma once

#include <cstddef>
#include <cstdint>

namespace host::render {

// Bytes one slice occupies in memory, or 0 if the tile mode is not supported. Sizes are in
// elements (pixels, or 4x4 blocks for compressed formats).
size_t tiledSliceSize(uint32_t tileMode, uint32_t pitch, uint32_t height, uint32_t bytesPerElement);

// Copies a width x height region of one slice into tightly packed linear rows.
bool detile(uint32_t tileMode, const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height, uint32_t pitch,
            uint32_t bytesPerElement);

} // namespace host::render
