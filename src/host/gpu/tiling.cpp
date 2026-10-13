#include "host/gpu/tiling.h"

#include <cstring>

namespace host::render {

namespace {

constexpr uint32_t kLinearAligned = 8, k1dThin = 13;

uint32_t align(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// Element index within an 8x8 micro tile for the THIN (non-displayable) ordering:
// bits interleave x0 y0 x1 y1 x2 y2.
uint32_t thinIndex(uint32_t x, uint32_t y) {
    return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}

} // namespace

size_t tiledSliceSize(uint32_t mode, uint32_t pitch, uint32_t height, uint32_t bpe) {
    switch (mode) {
    case kLinearAligned: return size_t(pitch) * height * bpe;
    case k1dThin: return size_t(align(pitch, 8)) * align(height, 8) * bpe;
    default: return 0;
    }
}

bool detile(uint32_t mode, const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height, uint32_t pitch,
            uint32_t bpe) {
    switch (mode) {
    case kLinearAligned:
        for (uint32_t y = 0; y < height; ++y)
            std::memcpy(dst + size_t(y) * width * bpe, src + size_t(y) * pitch * bpe, size_t(width) * bpe);
        return true;
    case k1dThin: {
        // 8x8 micro tiles laid out row by row; elements within a tile in THIN order.
        const uint32_t tilesPerRow = align(pitch, 8) / 8;
        const size_t tileBytes = 64 * size_t(bpe);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const size_t tile = size_t(y / 8) * tilesPerRow + x / 8;
                const uint8_t* s = src + tile * tileBytes + size_t(thinIndex(x & 7, y & 7)) * bpe;
                std::memcpy(dst + (size_t(y) * width + x) * bpe, s, bpe);
            }
        return true;
    }
    default: return false;
    }
}

} // namespace host::render
