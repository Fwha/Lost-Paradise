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

// Where one mip level of a texture sits in memory. Levels follow each other, each holding all
// of the texture's slices (array layers or cube faces) at sliceBytes apart. Sizes in elements.
struct MipLevel {
    size_t offset = 0, sliceBytes = 0;
    uint32_t pitch = 0;          // elements per row in memory
    uint32_t width = 0, height = 0; // elements to copy out of each slice
};

// Lays out levels [0, count) of a texture whose level 0 is width x height pixels with the T#'s
// pitch (pixels). Returns the bytes all levels take, or 0 if the tile mode is not supported.
// Level 0 keeps the T#'s pitch; smaller levels are aligned as GCN does for linear-aligned and
// 1D-thin textures (pow2Pad: each level's size rounded up to a power of two first).
size_t mipLayout(uint32_t tileMode, uint32_t width, uint32_t height, uint32_t pitch, bool blockCompressed,
                 uint32_t bytesPerElement, uint32_t slices, uint32_t count, bool pow2Pad, MipLevel* out);

} // namespace host::render
