#include "formats/sllz.h"

#include <cstring>

namespace formats {

namespace {

constexpr size_t kHeaderSize = 16;
constexpr size_t kMaxDistance = 0x1000;
constexpr size_t kMinMatch = 3;
constexpr size_t kMaxMatch = 0xF + kMinMatch;

uint32_t readU32(const uint8_t* p, bool bigEndian) {
    return bigEndian ? uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]
                     : uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | p[0];
}

uint16_t readU16(const uint8_t* p, bool bigEndian) {
    return bigEndian ? uint16_t(p[0] << 8 | p[1]) : uint16_t(p[1] << 8 | p[0]);
}

void putU32le(std::vector<uint8_t>& v, size_t at, uint32_t x) {
    for (int i = 0; i < 4; ++i) v[at + i] = uint8_t(x >> (8 * i));
}

} // namespace

bool isSllz(std::span<const uint8_t> data) {
    return data.size() >= kHeaderSize && std::memcmp(data.data(), "SLLZ", 4) == 0;
}

bool sllzDecompress(std::span<const uint8_t> data, std::vector<uint8_t>& out, std::string* error) {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    if (!isSllz(data)) return fail("not an SLLZ blob");

    const bool be = data[4] != 0;
    const uint8_t version = data[5];
    const uint16_t headerSize = readU16(&data[6], be);
    const uint32_t size = readU32(&data[8], be);
    const uint32_t compressedSize = readU32(&data[12], be);
    if (version != 1) return fail("unsupported SLLZ version");
    if (headerSize < kHeaderSize || compressedSize > data.size() || headerSize > compressedSize)
        return fail("bad SLLZ header");

    out.clear();
    out.reserve(size);
    size_t p = headerSize;
    const size_t end = compressedSize;
    if (size > 0 && p >= end) return fail("SLLZ stream truncated");
    // The next flag byte is read as soon as the current one is used up, before the
    // payload of the token that consumed its last bit.
    uint8_t flags = p < end ? data[p++] : 0;
    int bits = 8;
    while (out.size() < size) {
        const bool copy = flags & 0x80;
        flags <<= 1;
        if (--bits == 0) {
            flags = p < end ? data[p] : 0;
            ++p;
            bits = 8;
        }

        if (!copy) {
            if (p >= end) return fail("SLLZ stream truncated");
            out.push_back(data[p++]);
            continue;
        }
        if (p + 2 > end) return fail("SLLZ stream truncated");
        const uint16_t v = uint16_t(data[p] | data[p + 1] << 8);
        p += 2;
        const size_t distance = (v >> 4) + 1;
        const size_t length = (v & 0xF) + kMinMatch;
        if (distance > out.size()) return fail("SLLZ back-reference before start");
        for (size_t i = 0; i < length && out.size() < size; ++i)
            out.push_back(out[out.size() - distance]);
    }
    return true;
}

std::vector<uint8_t> sllzCompress(std::span<const uint8_t> data) {
    std::vector<uint8_t> out(kHeaderSize, 0);
    std::memcpy(out.data(), "SLLZ", 4);
    out[4] = 0; // little-endian header
    out[5] = 1;
    out[6] = kHeaderSize;

    size_t i = 0;
    size_t flagPos = out.size();
    out.push_back(0);
    int bits = 8;
    while (i < data.size()) {
        // Greedy search over the window. Slow, but only used by tools and tests.
        size_t bestLen = 0, bestDist = 0;
        const size_t windowStart = i > kMaxDistance ? i - kMaxDistance : 0;
        for (size_t j = windowStart; j < i; ++j) {
            size_t len = 0;
            while (len < kMaxMatch && i + len < data.size() && data[j + len] == data[i + len]) ++len;
            if (len > bestLen) {
                bestLen = len;
                bestDist = i - j;
                if (len == kMaxMatch) break;
            }
        }
        const bool copy = bestLen >= kMinMatch;
        if (copy) out[flagPos] |= uint8_t(1 << (bits - 1));
        if (--bits == 0) {
            flagPos = out.size();
            out.push_back(0);
            bits = 8;
        }
        if (copy) {
            const uint16_t v = uint16_t((bestDist - 1) << 4 | (bestLen - kMinMatch));
            out.push_back(uint8_t(v));
            out.push_back(uint8_t(v >> 8));
            i += bestLen;
        } else {
            out.push_back(data[i++]);
        }
    }
    putU32le(out, 8, uint32_t(data.size()));
    putU32le(out, 12, uint32_t(out.size()));
    return out;
}

} // namespace formats
