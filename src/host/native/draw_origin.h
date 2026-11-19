// Where each of the game's draws comes from: the engine functions that called the eboot's Gnm
// draw wrappers, three levels up the frame pointer chain. The hook on the wrappers
// (draw_sites.cpp) keeps them for the draw being written, the HLE draw packet helpers put them
// into the command buffer ahead of the draw as a NOP the game's reserved space has room for, and
// the command processor hands them to the renderer with the draw (DrawCall::origin). The pass map
// lists them, which shows which engine code draws each pass: what stage 2 hooks.
#pragma once

#include <cstdint>

namespace host::native {

// Eboot offsets of return addresses, innermost first; 0 past the chain's end.
struct Origin {
    uint32_t at[3] = {};
};

// The origin of the draw this thread is about to write (set by the wrapper hook, taken by the
// packet helper).
inline thread_local Origin tPendingOrigin{};

// The NOP's three payload dwords: each offset (below 2^26) with this tag in its top 6 bits.
constexpr uint32_t kOriginTag = 0x2Du << 26;
constexpr uint32_t kOriginMask = 0x3Fu << 26;
inline bool isOriginNop(const uint32_t* body, uint32_t n) {
    return n == 3 && (body[0] & kOriginMask) == kOriginTag && (body[1] & kOriginMask) == kOriginTag &&
           (body[2] & kOriginMask) == kOriginTag;
}

} // namespace host::native
