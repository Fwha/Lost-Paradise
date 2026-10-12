// SLLZ: the LZ77 compression used for files inside .par archives.
//
// Header (16 bytes):
//   0x00 "SLLZ"
//   0x04 u8  endianness of the header fields (0 = little, 1 = big)
//   0x05 u8  version (only 1 has been seen in Lost Paradise)
//   0x06 u16 header size
//   0x08 u32 decompressed size
//   0x0C u32 compressed size, including this header
//
// Version 1 stream: a flag byte gives 8 tokens, most significant bit first. A 0 bit is a
// literal byte; a 1 bit is a little-endian u16 back-reference where
// distance = (v >> 4) + 1 and length = (v & 0xF) + 3. The next flag byte follows
// immediately after the token that uses up the previous one, before that token's payload.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace formats {

bool isSllz(std::span<const uint8_t> data);

// Decompress a complete SLLZ blob (header included). Returns false and sets *error on failure.
bool sllzDecompress(std::span<const uint8_t> data, std::vector<uint8_t>& out, std::string* error);

// Compress with SLLZ version 1. The output is always valid but not size-optimal.
std::vector<uint8_t> sllzCompress(std::span<const uint8_t> data);

} // namespace formats
