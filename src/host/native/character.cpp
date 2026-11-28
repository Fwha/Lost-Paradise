// Character materials: s_b1dzt[skin] / sm_o1dzt[skin] and s_b1dst / sm_o1dzt.
// Both vertex programs skin four bones. SKIN selects the skin material's ramps,
// extra lobe, point-specular model and vertex-controlled directional specular.
// HAIR_HT / HAIR_CMASK add the two s_b1dt[hair] material pairs.
#include "host/native/native.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace host::native {
namespace {

// Constant-buffer images retain their original byte layouts. Separate source
// buffers have separate bases; repeated/overlapping loads share destinations.
enum : uint32_t {
    cV0 = 0,
    cV10 = 32,
    cV30 = 64,
    cV40 = 104,
    cVF0 = 105,
    cVF1 = 113,
    cP0 = 128,
    cP10 = 176,
    cP20 = 192,
    cP30 = 288,
    cP50 = 312,
    cP60 = 320,
    cP70 = 352,
    cP90 = 368,
    cView = 400,
    cFlags = 440,
    cTexScale = 441, // eight vec4s, two texture scales per vec4
    cFetch = 449,    // six fetch inputs
    cOut = 455,      // swizzles xy, clipZHalf z
    cPalette = 456,  // x: palette first dword, including Buffer::byteOffset
    cVHair = 457, // VSV[0x10]: original hair material buffer layout
    kConstants = 460,
    kDeformations = 4,
};

struct Load {
    bool vs;
    uint32_t pc, n, index, at;
};

#define V(pc, n, b, o) {true, pc, n, b + (o) / 16, ((o) / 4) & 3}
#define P(pc, n, b, o) {false, pc, n, b + (o) / 16, ((o) / 4) & 3}

// The deformation loops' loads (0x50, 0x90 and 0xd0 plus 16 per entry) run once an entry at
// one pc: the analysis keeps one entry's values, so only draws with at most one entry are drawn.
constexpr Load kVsSkin[] = {
    V(0x114, 1, cV0, 0x12c),
    V(0x2d4, 4, cV0, 0x0d0),
    V(0x2d8, 4, cV0, 0x050),
    V(0x2dc, 4, cV0, 0x090),
    V(0x358, 1, cV40, 0),
    V(0x35c, 4, cV10, 0x170),
    V(0x3ac, 4, cV30, 0x220),
    V(0x3d0, 4, cV10, 0x160),
    V(0x40c, 16, cV30, 0x030),
    V(0x410, 4, cV30, 0x220),
    V(0x488, 4, cVF1, 0x080),
    V(0x48c, 16, cVF0, 0x040),
    V(0x5b8, 8, cV30, 0),
    V(0x5bc, 4, cV30, 0x020),
};
constexpr Load kVsRigid[] = {
    V(0x114, 1, cV0, 0x12c),
    V(0x2dc, 4, cV0, 0x0d0),
    V(0x2e0, 4, cV0, 0x050),
    V(0x2e4, 4, cV0, 0x090),
    V(0x360, 1, cV40, 0),
    V(0x364, 4, cV10, 0x170),
    V(0x3b4, 4, cV30, 0x220),
    V(0x3d8, 4, cV10, 0x160),
    V(0x414, 16, cV30, 0x030),
    V(0x418, 4, cV30, 0x220),
    V(0x490, 4, cVF1, 0x080),
    V(0x494, 16, cVF0, 0x040),
    V(0x5c0, 8, cV30, 0),
    V(0x5c4, 4, cV30, 0x020),
};
constexpr Load kPsSkin[] = {
    P(0x024, 1, cP60, 0x0dc),
    P(0x074, 4, cP60, 0x0d0),
    P(0x078, 4, cP60, 0x050),
    P(0x0d4, 4, cP60, 0x090),
    P(0x120, 1, cFlags, 0),
    P(0x124, 2, cView, 0x218),
    P(0x16c, 4, cP10, 0x0f0),
    P(0x258, 1, cP70, 0x0ac),
    P(0x260, 4, cP90, 0x070),
    P(0x270, 4, cP10, 0),
    P(0x274, 1, cP70, 0x088),
    P(0x43c, 4, cP30, 0x0a0),
    P(0x440, 4, cP30, 0x020),
    P(0x56c, 4, cP30, 0x0b0),
    P(0x570, 4, cP30, 0x030),
    P(0x6a0, 4, cP30, 0x0c0),
    P(0x6a4, 4, cP30, 0x040),
    P(0x7e0, 4, cP30, 0x0d0),
    P(0x7e4, 4, cP30, 0x050),
    P(0xb2c, 1, cView, 0x22c),
    P(0xb30, 16, cP20, 0x4f0),
    P(0xb3c, 16, cP20, 0x530),
    P(0xbbc, 8, cP30, 0x080),
    P(0xbd0, 8, cP70, 0x060),
    P(0xbe4, 2, cP0, 0x040),
    P(0xbe8, 1, cP0, 0x048),
    P(0xbec, 1, cP90, 0x134),
    P(0xbf4, 4, cP30, 0x060),
    P(0xc0c, 1, cP0, 0x00c),
    P(0xc10, 4, cP30, 0x120),
    P(0xc44, 4, cP90, 0x020),
    P(0xc94, 4, cP20, 0x5b0),
    P(0xd8c, 4, cP60, 0x110),
    P(0xda8, 4, cP30, 0x100),
    P(0xe58, 8, cP30, 0),
    P(0xe64, 4, cP30, 0x0e0),
    P(0xe68, 8, cP70, 0x090),
    P(0xe6c, 8, cP60, 0),
    P(0xe70, 16, cP20, 0x570),
    P(0x12fc, 16, cP0, 0),
    P(0x143c, 4, cView, 0x220),
    P(0x1440, 8, cP10, 0x010),
    P(0x14c4, 1, cP50, 0x01c),
    P(0x14c8, 1, cP70, 0x09c),
    P(0x14cc, 1, cP60, 0x00c),
    P(0x14d0, 1, cFlags, 0),
    P(0x14d4, 4, cP10, 0x030),
    P(0x153c, 2, cView, 0x218),
    P(0x1540, 1, cP0, 0x02c),
    P(0x1544, 4, cP0, 0x284),
    P(0x1548, 1, cP0, 0x03c),
    P(0x154c, 1, cP0, 0x26c),
    P(0x1550, 1, cP0, 0x00c),
    P(0x1694, 2, cP0, 0x270),
    P(0x1698, 1, cP0, 0x054),
    P(0x1700, 2, cP0, 0x260),
    P(0x1704, 1, cP0, 0x268),
    P(0x17e0, 4, cP0, 0x2a0),
    P(0x17e4, 2, cP0, 0x278),
    P(0x17e8, 1, cP0, 0x058),
    P(0x17ec, 1, cP0, 0x280),
    P(0x18a8, 2, cP0, 0x294),
    P(0x18ac, 1, cP0, 0x29c),
    P(0x18fc, 1, cFlags, 0),
    P(0x1900, 1, cP60, 0x11c),
    P(0x1928, 4, cView, 0x230),
    P(0x19a0, 1, cView, 0x22c),
};
constexpr Load kPsRigid[] = {
    P(0x020, 1, cP60, 0x0dc),
    P(0x074, 4, cP60, 0x0d0),
    P(0x078, 4, cP60, 0x050),
    P(0x0d4, 4, cP60, 0x090),
    P(0x180, 1, cFlags, 0),
    P(0x1cc, 2, cView, 0x218),
    P(0x2c8, 4, cP10, 0x0f0),
    P(0x1ac, 1, cP70, 0x0ac),
    P(0xa24, 4, cP10, 0),
    P(0xb6c, 1, cP70, 0x088),
    P(0x378, 4, cP30, 0x0a0),
    P(0x37c, 4, cP30, 0x020),
    P(0x49c, 4, cP30, 0x0b0),
    P(0x4a0, 4, cP30, 0x030),
    P(0x5c4, 4, cP30, 0x0c0),
    P(0x5c8, 4, cP30, 0x040),
    P(0x6f8, 4, cP30, 0x0d0),
    P(0x6fc, 4, cP30, 0x050),
    P(0x908, 1, cView, 0x22c),
    P(0x8f8, 16, cP20, 0x4f0),
    P(0x900, 16, cP20, 0x530),
    P(0x9ec, 8, cP30, 0x080),
    P(0x9c4, 2, cP70, 0x068),
    P(0x914, 2, cP0, 0x040),
    P(0x9bc, 1, cP0, 0x048),
    P(0x9c8, 1, cP90, 0x134),
    P(0x9fc, 4, cP30, 0x060),
    P(0xa20, 1, cP0, 0x00c),
    P(0xa28, 4, cP30, 0x120),
    P(0xd84, 4, cP20, 0x5b0),
    P(0xb84, 4, cP60, 0x110),
    P(0x9f8, 4, cP30, 0x100),
    P(0xa10, 8, cP30, 0),
    P(0x9e8, 4, cP30, 0x0e0),
    P(0xaa4, 8, cP70, 0x090),
    P(0xb70, 8, cP60, 0),
    P(0xcc0, 16, cP20, 0x570),
    P(0x100c, 16, cP0, 0),
    P(0x114c, 4, cView, 0x220),
    P(0x1150, 8, cP10, 0x010),
    P(0x11d4, 1, cP50, 0x01c),
    P(0x11d8, 1, cP70, 0x09c),
    P(0x11dc, 1, cP60, 0x00c),
    P(0x11e0, 1, cFlags, 0),
    P(0x11e4, 4, cP10, 0x030),
    P(0x124c, 2, cView, 0x218),
    P(0x1250, 1, cP0, 0x02c),
    P(0x1254, 4, cP0, 0x284),
    P(0x1258, 1, cP0, 0x03c),
    P(0x125c, 1, cP0, 0x26c),
    P(0x1260, 1, cP0, 0x00c),
    P(0x13a4, 2, cP0, 0x270),
    P(0x13a8, 1, cP0, 0x054),
    P(0x1410, 2, cP0, 0x260),
    P(0x1414, 1, cP0, 0x268),
    P(0x14e8, 4, cP0, 0x2a0),
    P(0x14ec, 2, cP0, 0x278),
    P(0x14f0, 1, cP0, 0x058),
    P(0x14f4, 1, cP0, 0x280),
    P(0x15b8, 2, cP0, 0x294),
    P(0x15bc, 1, cP0, 0x29c),
    P(0x160c, 1, cFlags, 0),
    P(0x1610, 1, cP60, 0x11c),
    P(0x1638, 4, cView, 0x230),
    P(0x16b0, 1, cView, 0x22c),
};
// sm_p1dzt: sm_o1dzt with an alpha test inserted at 0x11b0 (0x1c bytes; two loads after it reordered).
constexpr Load kPsRigidAlpha[] = {
    P(0x020, 1, cP60, 0x0dc),
    P(0x074, 4, cP60, 0x0d0),
    P(0x078, 4, cP60, 0x050),
    P(0x0d4, 4, cP60, 0x090),
    P(0x180, 1, cFlags, 0),
    P(0x1cc, 2, cView, 0x218),
    P(0x2c8, 4, cP10, 0x0f0),
    P(0x1ac, 1, cP70, 0x0ac),
    P(0xa24, 4, cP10, 0),
    P(0xb6c, 1, cP70, 0x088),
    P(0x378, 4, cP30, 0x0a0),
    P(0x37c, 4, cP30, 0x020),
    P(0x49c, 4, cP30, 0x0b0),
    P(0x4a0, 4, cP30, 0x030),
    P(0x5c4, 4, cP30, 0x0c0),
    P(0x5c8, 4, cP30, 0x040),
    P(0x6f8, 4, cP30, 0x0d0),
    P(0x6fc, 4, cP30, 0x050),
    P(0x908, 1, cView, 0x22c),
    P(0x8f8, 16, cP20, 0x4f0),
    P(0x900, 16, cP20, 0x530),
    P(0x9ec, 8, cP30, 0x080),
    P(0x9c4, 2, cP70, 0x068),
    P(0x914, 2, cP0, 0x040),
    P(0x9bc, 1, cP0, 0x048),
    P(0x9c8, 1, cP90, 0x134),
    P(0x9fc, 4, cP30, 0x060),
    P(0xa20, 1, cP0, 0x00c),
    P(0xa28, 4, cP30, 0x120),
    P(0xd84, 4, cP20, 0x5b0),
    P(0xb84, 4, cP60, 0x110),
    P(0x9f8, 4, cP30, 0x100),
    P(0xa10, 8, cP30, 0),
    P(0x9e8, 4, cP30, 0x0e0),
    P(0xaa4, 8, cP70, 0x090),
    P(0xb70, 8, cP60, 0),
    P(0xcc0, 16, cP20, 0x570),
    P(0x100c, 16, cP0, 0),
    P(0x114c, 4, cView, 0x220),
    P(0x1150, 8, cP10, 0x010),
    P(0x11f4, 1, cP50, 0x01c),
    P(0x11dc, 1, cP70, 0x09c),
    P(0x11f8, 1, cP60, 0x00c),
    P(0x11fc, 1, cFlags, 0),
    P(0x1200, 4, cP10, 0x030),
    P(0x1268, 2, cView, 0x218),
    P(0x126c, 1, cP0, 0x02c),
    P(0x1270, 4, cP0, 0x284),
    P(0x1274, 1, cP0, 0x03c),
    P(0x1278, 1, cP0, 0x26c),
    P(0x127c, 1, cP0, 0x00c),
    P(0x13c0, 2, cP0, 0x270),
    P(0x13c4, 1, cP0, 0x054),
    P(0x142c, 2, cP0, 0x260),
    P(0x1430, 1, cP0, 0x268),
    P(0x1504, 4, cP0, 0x2a0),
    P(0x1508, 2, cP0, 0x278),
    P(0x150c, 1, cP0, 0x058),
    P(0x1510, 1, cP0, 0x280),
    P(0x15d4, 2, cP0, 0x294),
    P(0x15d8, 1, cP0, 0x29c),
    P(0x1628, 1, cFlags, 0),
    P(0x162c, 1, cP60, 0x11c),
    P(0x1654, 4, cView, 0x230),
    P(0x16cc, 1, cView, 0x22c),
    P(0x11b8, 1, cP70, 0x080), // the alpha test's reference
};

constexpr Load kVsHair[] = {
    V(0x114, 1, cV0, 0x12c),
    V(0x2dc, 4, cV0, 0x0d0),
    V(0x2e0, 4, cV0, 0x050),
    V(0x2e4, 4, cV0, 0x090),
    V(0x360, 1, cV40, 0),
    V(0x364, 4, cV10, 0x170),
    V(0x3b4, 4, cV30, 0x220),
    V(0x3d8, 4, cV10, 0x160),
    V(0x414, 16, cV30, 0x030),
    V(0x418, 4, cV30, 0x220),
    V(0x490, 4, cVF1, 0x080),
    V(0x494, 16, cVF0, 0x040),
    V(0x5c8, 8, cV30, 0),
    V(0x5cc, 4, cV30, 0x020),
    V(0x5d0, 2, cVHair, 0x028),
};
constexpr Load kPsHairHt[] = {
    P(0x028, 1, cP60, 0x0dc),
    P(0x06c, 4, cP60, 0x050),
    P(0x078, 4, cP60, 0x0d0),
    P(0x0bc, 4, cP60, 0x090),
    P(0x194, 1, cFlags, 0x000),
    P(0x1b4, 2, cView, 0x218),
    P(0x2d0, 4, cP10, 0x0f0),
    P(0x354, 4, cP30, 0x0a0),
    P(0x358, 4, cP30, 0x020),
    P(0x42c, 4, cP30, 0x0b0),
    P(0x430, 4, cP30, 0x030),
    P(0x510, 4, cP30, 0x0c0),
    P(0x514, 4, cP30, 0x040),
    P(0x5f0, 4, cP30, 0x0d0),
    P(0x5f4, 4, cP30, 0x050),
    P(0x704, 8, cP30, 0x080),
    P(0x76c, 4, cP30, 0x060),
    P(0x810, 4, cP30, 0x0e0),
    P(0x820, 4, cP30, 0x120),
    P(0x8a8, 8, cP30, 0x000),
    P(0x9a8, 4, cP30, 0x100),
    P(0x9c0, 8, cP70, 0x090),
    P(0x9c4, 2, cP70, 0x068),
    P(0x9c8, 4, cP10, 0x000),
    P(0x9cc, 1, cP70, 0x088),
    P(0xa78, 1, cView, 0x22c),
    P(0xafc, 4, cP20, 0x5b0),
    P(0xb10, 2, cP0, 0x040),
    P(0xb14, 1, cP90, 0x134),
    P(0xb18, 1, cP0, 0x048),
    P(0xb1c, 1, cP0, 0x00c),
    P(0xba0, 4, cP90, 0x030),
    P(0xbac, 16, cP20, 0x4f0),
    P(0xbf8, 16, cP20, 0x530),
    P(0xc64, 16, cP20, 0x570),
    P(0xd68, 8, cP60, 0x000),
    P(0xd6c, 4, cP60, 0x110),
    P(0xf50, 16, cP0, 0x000),
    P(0x1070, 4, cView, 0x220),
    P(0x1074, 8, cP10, 0x010),
    P(0x10d0, 1, cP70, 0x080),
    P(0x10e8, 1, cP70, 0x09c),
    P(0x1104, 1, cP50, 0x01c),
    P(0x1108, 1, cP60, 0x00c),
    P(0x110c, 4, cP10, 0x030),
    P(0x115c, 4, cView, 0x230),
};
constexpr Load kPsHairPHt[] = {
    P(0x020, 1, cP60, 0x0dc),
    P(0x074, 4, cP60, 0x050),
    P(0x078, 4, cP60, 0x0d0),
    P(0x0bc, 4, cP60, 0x090),
    P(0x174, 1, cFlags, 0x000),
    P(0x1a4, 2, cView, 0x218),
    P(0x2c0, 4, cP10, 0x0f0),
    P(0x344, 4, cP30, 0x0a0),
    P(0x348, 4, cP30, 0x020),
    P(0x41c, 4, cP30, 0x0b0),
    P(0x420, 4, cP30, 0x030),
    P(0x500, 4, cP30, 0x0c0),
    P(0x504, 4, cP30, 0x040),
    P(0x5e0, 4, cP30, 0x0d0),
    P(0x5e4, 4, cP30, 0x050),
    P(0x6c0, 8, cP30, 0x080),
    P(0x748, 4, cP30, 0x060),
    P(0x74c, 4, cP30, 0x0e0),
    P(0x7a4, 4, cP30, 0x120),
    P(0x7dc, 8, cP30, 0x000),
    P(0x82c, 8, cP70, 0x090),
    P(0x830, 1, cView, 0x22c),
    P(0x838, 4, cP10, 0x000),
    P(0x840, 4, cP30, 0x100),
    P(0x844, 2, cP70, 0x068),
    P(0x848, 1, cP70, 0x088),
    P(0x914, 1, cP0, 0x048),
    P(0x930, 1, cP90, 0x134),
    P(0x9b8, 16, cP20, 0x530),
    P(0x9cc, 4, cP20, 0x5b0),
    P(0xaa4, 2, cP0, 0x040),
    P(0xb10, 16, cP20, 0x4f0),
    P(0xb24, 1, cP0, 0x00c),
    P(0xb28, 4, cP90, 0x030),
    P(0xb30, 16, cP20, 0x570),
    P(0xb38, 8, cP60, 0x000),
    P(0xb3c, 4, cP60, 0x110),
    P(0xf1c, 16, cP0, 0x000),
    P(0x1050, 4, cView, 0x220),
    P(0x1054, 8, cP10, 0x010),
    P(0x10b0, 1, cP70, 0x080),
    P(0x10d0, 1, cP70, 0x09c),
    P(0x10ec, 1, cP50, 0x01c),
    P(0x10f0, 1, cP60, 0x00c),
    P(0x10f4, 4, cP10, 0x030),
    P(0x1140, 4, cView, 0x230),
};
constexpr Load kPsHairCmask[] = {
    P(0x01c, 1, cP60, 0x0dc),
    P(0x074, 4, cP60, 0x0d0),
    P(0x078, 4, cP60, 0x050),
    P(0x0d4, 4, cP60, 0x090),
    P(0x178, 1, cFlags, 0x000),
    P(0x1b0, 2, cView, 0x218),
    P(0x268, 4, cP10, 0x0f0),
    P(0x2f4, 4, cP30, 0x0a0),
    P(0x2f8, 4, cP30, 0x020),
    P(0x3bc, 4, cP30, 0x0b0),
    P(0x3c0, 4, cP30, 0x030),
    P(0x490, 4, cP30, 0x0c0),
    P(0x494, 4, cP30, 0x040),
    P(0x560, 4, cP30, 0x0d0),
    P(0x564, 4, cP30, 0x050),
    P(0x660, 8, cP30, 0x080),
    P(0x670, 2, cP0, 0x040),
    P(0x674, 2, cP70, 0x068),
    P(0x678, 8, cP70, 0x090),
    P(0x67c, 4, cP20, 0x5b0),
    P(0x684, 1, cP0, 0x00c),
    P(0x68c, 1, cView, 0x22c),
    P(0x690, 1, cP0, 0x048),
    P(0x6e4, 1, cP70, 0x088),
    P(0x82c, 16, cP20, 0x530),
    P(0x8f0, 16, cP20, 0x4f0),
    P(0x97c, 4, cP30, 0x0e0),
    P(0x984, 4, cP30, 0x060),
    P(0xa70, 16, cP20, 0x570),
    P(0xab4, 8, cP60, 0x000),
    P(0xac0, 4, cP30, 0x120),
    P(0xb4c, 8, cP30, 0x100),
    P(0xb64, 4, cP60, 0x110),
    P(0xb8c, 8, cP30, 0x000),
    P(0xba8, 4, cP10, 0x000),
    P(0xbe4, 1, cP90, 0x134),
    P(0xef0, 16, cP0, 0x000),
    P(0x1020, 4, cView, 0x220),
    P(0x1024, 8, cP10, 0x010),
    P(0x108c, 1, cP70, 0x080),
    P(0x10ac, 1, cP70, 0x09c),
    P(0x10c4, 1, cP50, 0x01c),
    P(0x10c8, 1, cP60, 0x00c),
    P(0x10cc, 4, cP10, 0x030),
    P(0x1118, 4, cView, 0x230),
};
constexpr Load kPsHairP[] = {
    P(0x1c, 1, cP60, 0xdc),
    P(0x74, 4, cP60, 0xd0),
    P(0x78, 4, cP60, 0x50),
    P(0xd4, 4, cP60, 0x90),
    P(0x194, 1, cFlags, 0x0),
    P(0x1b0, 2, cView, 0x218),
    P(0x26c, 4, cP10, 0xf0),
    P(0x2f8, 4, cP30, 0xa0),
    P(0x2fc, 4, cP30, 0x20),
    P(0x3c4, 4, cP30, 0xb0),
    P(0x3c8, 4, cP30, 0x30),
    P(0x49c, 4, cP30, 0xc0),
    P(0x4a0, 4, cP30, 0x40),
    P(0x570, 4, cP30, 0xd0),
    P(0x574, 4, cP30, 0x50),
    P(0x638, 8, cP30, 0x80),
    P(0x6a8, 2, cP70, 0x68),
    P(0x6ac, 8, cP70, 0x90),
    P(0x6b0, 2, cP0, 0x40),
    P(0x6b4, 1, cP0, 0x48),
    P(0x6b8, 1, cP0, 0xc),
    P(0x6c0, 4, cP20, 0x5b0),
    P(0x6f8, 1, cView, 0x22c),
    P(0x8d0, 16, cP20, 0x530),
    P(0x948, 1, cP70, 0x88),
    P(0x96c, 16, cP20, 0x4f0),
    P(0x9bc, 4, cP30, 0x60),
    P(0x9c4, 4, cP30, 0xe0),
    P(0xa84, 16, cP20, 0x570),
    P(0xad0, 4, cP30, 0x120),
    P(0xb7c, 8, cP60, 0x0),
    P(0xb80, 4, cP60, 0x110),
    P(0xba8, 8, cP30, 0x100),
    P(0xbb4, 4, cP10, 0x0),
    P(0xbc8, 1, cP90, 0x134),
    P(0xbcc, 8, cP30, 0x0),
    P(0xf20, 16, cP0, 0x0),
    P(0x105c, 4, cView, 0x220),
    P(0x1060, 8, cP10, 0x10),
    P(0x10c8, 1, cP70, 0x80),
    P(0x10e4, 1, cP70, 0x9c),
    P(0x1100, 1, cP50, 0x1c),
    P(0x1104, 1, cP60, 0xc),
    P(0x1108, 4, cP10, 0x30),
    P(0x1158, 4, cView, 0x230),
};
constexpr Load kPsRef[] = {
    P(0x20, 1, cP60, 0xdc),
    P(0x74, 4, cP60, 0xd0),
    P(0x78, 4, cP60, 0x50),
    P(0xd4, 4, cP60, 0x90),
    P(0x180, 1, cFlags, 0x0),
    P(0x1ac, 1, cP70, 0xac),
    P(0x1cc, 2, cView, 0x218),
    P(0x2c4, 4, cP10, 0xf0),
    P(0x360, 4, cP30, 0xa0),
    P(0x364, 4, cP30, 0x20),
    P(0x468, 4, cP30, 0xb0),
    P(0x46c, 4, cP30, 0x30),
    P(0x574, 4, cP30, 0xc0),
    P(0x578, 4, cP30, 0x40),
    P(0x684, 4, cP30, 0xd0),
    P(0x688, 4, cP30, 0x50),
    P(0x930, 4, cP30, 0x100),
    P(0x934, 1, cView, 0x22c),
    P(0x938, 1, cP0, 0xc),
    P(0x93c, 2, cP70, 0x68),
    P(0x940, 16, cP20, 0x530),
    P(0x974, 8, cP70, 0x90),
    P(0x98c, 2, cP0, 0x40),
    P(0x9bc, 1, cP70, 0x88),
    P(0x9c0, 4, cP30, 0xe0),
    P(0x9c4, 4, cP30, 0x60),
    P(0x9cc, 8, cP60, 0x0),
    P(0x9e0, 1, cP90, 0x134),
    P(0x9f0, 4, cP60, 0x110),
    P(0x9f8, 1, cP0, 0x48),
    P(0xa10, 4, cP20, 0x5b0),
    P(0xa2c, 4, cP10, 0x0),
    P(0xaac, 16, cP20, 0x4f0),
    P(0xaf0, 8, cP30, 0x80),
    P(0xba4, 8, cP30, 0x0),
    P(0xcc4, 16, cP20, 0x570),
    P(0xfb0, 16, cP0, 0x0),
    P(0x10f0, 4, cView, 0x220),
    P(0x10f4, 8, cP10, 0x10),
    P(0x1178, 1, cP50, 0x1c),
    P(0x117c, 1, cP70, 0x9c),
    P(0x1180, 1, cP60, 0xc),
    P(0x1184, 1, cFlags, 0x0),
    P(0x1188, 4, cP10, 0x30),
    P(0x11f0, 2, cView, 0x218),
    P(0x11f4, 1, cP0, 0x2c),
    P(0x11f8, 4, cP0, 0x284),
    P(0x11fc, 1, cP0, 0x3c),
    P(0x1200, 1, cP0, 0x26c),
    P(0x1204, 1, cP0, 0xc),
    P(0x1348, 2, cP0, 0x270),
    P(0x13a4, 2, cP0, 0x260),
    P(0x13a8, 1, cP0, 0x268),
    P(0x147c, 4, cP0, 0x2a0),
    P(0x1480, 2, cP0, 0x278),
    P(0x1484, 1, cP0, 0x280),
    P(0x153c, 2, cP0, 0x294),
    P(0x1540, 1, cP0, 0x29c),
    P(0x1590, 1, cFlags, 0x0),
    P(0x1594, 1, cP60, 0x11c),
    P(0x15bc, 4, cView, 0x230),
    P(0x1634, 1, cView, 0x22c),
};
constexpr Load kPsEye[] = {
    P(0x20, 1, cP60, 0xdc),
    P(0x74, 4, cP60, 0xd0),
    P(0x78, 4, cP60, 0x50),
    P(0xd4, 4, cP60, 0x90),
    P(0x180, 1, cFlags, 0x0),
    P(0x1a8, 2, cView, 0x218),
    P(0x264, 4, cP10, 0xf0),
    P(0x2a4, 4, cP30, 0x80),
    P(0x30c, 4, cP30, 0xa0),
    P(0x310, 4, cP30, 0x20),
    P(0x3e0, 4, cP30, 0xb0),
    P(0x3e4, 4, cP30, 0x30),
    P(0x4c4, 4, cP30, 0xc0),
    P(0x4c8, 4, cP30, 0x40),
    P(0x5a8, 4, cP30, 0xd0),
    P(0x5ac, 4, cP30, 0x50),
    P(0x68c, 4, cP30, 0xa0),
    P(0x690, 1, cP30, 0x2c),
    P(0x71c, 4, cP30, 0xb0),
    P(0x720, 1, cP30, 0x3c),
    P(0x7ac, 4, cP30, 0xc0),
    P(0x7b0, 1, cP30, 0x4c),
    P(0x848, 4, cP30, 0xd0),
    P(0x84c, 1, cP30, 0x5c),
    P(0x8f0, 16, cView, 0x30),
    P(0x8f4, 4, cP70, 0x60),
    P(0x8f8, 8, cP90, 0x40),
    P(0x8fc, 16, cP20, 0x530),
    P(0x904, 1, cView, 0x22c),
    P(0xa00, 16, cP20, 0x570),
    P(0xa94, 4, cP30, 0x90),
    P(0xb18, 16, cP20, 0x4f0),
    P(0xb20, 4, cP20, 0x5b0),
    P(0xc30, 2, cP90, 0x130),
    P(0xc70, 8, cP70, 0x90),
    P(0xcc4, 4, cView, 0x70),
    P(0xcc8, 4, cView, 0x0),
    P(0xcec, 4, cP30, 0xe0),
    P(0xcf0, 4, cP30, 0x60),
    P(0xcf4, 4, cP90, 0x60),
    P(0xd10, 1, cP0, 0x48),
    P(0xd38, 1, cP0, 0xc),
    P(0xd44, 8, cP30, 0x0),
    P(0xd8c, 2, cP0, 0x40),
    P(0xda0, 8, cP60, 0x0),
    P(0xe38, 4, cP60, 0x110),
    P(0xe68, 4, cP10, 0x0),
    P(0x110c, 16, cP0, 0x0),
    P(0x1240, 4, cView, 0x220),
    P(0x1244, 8, cP10, 0x10),
    P(0x12d8, 1, cP70, 0x9c),
    P(0x12dc, 1, cP50, 0x1c),
    P(0x12e0, 1, cP60, 0xc),
    P(0x12e4, 4, cP10, 0x30),
    P(0x1330, 4, cView, 0x230),
};
constexpr Load kVsEye[] = {
    V(0x114, 1, cV0, 0x12c),
    V(0x2e0, 4, cV0, 0xd0),
    V(0x2e4, 4, cV0, 0x50),
    V(0x2e8, 4, cV0, 0x90),
    V(0x364, 1, cV40, 0x0),
    V(0x368, 4, cV10, 0x170),
    V(0x3b8, 4, cV30, 0x220),
    V(0x3dc, 4, cV10, 0x160),
    V(0x418, 16, cV30, 0x30),
    V(0x41c, 4, cV30, 0x220),
    V(0x48c, 4, cVF1, 0x80),
    V(0x490, 16, cVF0, 0x40),
    V(0x5b4, 8, cV30, 0x0),
    V(0x5b8, 4, cV30, 0x20),
};
// The mouth VS has the same skinning and varyings, then scales param0.xyz.
constexpr Load kVsMouth[] = {
    V(0x114, 1, cV0, 0x12c),
    V(0x2dc, 4, cV0, 0x0d0),
    V(0x2e0, 4, cV0, 0x050),
    V(0x2e4, 4, cV0, 0x090),
    V(0x360, 1, cV40, 0),
    V(0x364, 4, cV10, 0x170),
    V(0x3b4, 4, cV30, 0x220),
    V(0x3d8, 4, cV10, 0x160),
    V(0x414, 16, cV30, 0x030),
    V(0x418, 4, cV30, 0x220),
    V(0x490, 4, cVF1, 0x080),
    V(0x494, 16, cVF0, 0x040),
    V(0x5c8, 8, cV30, 0),
    V(0x5cc, 4, cV30, 0x020),
    V(0x5d0, 1, cV10, 0x03c),
};
// s_o1dzt +1: PSV layouts match [ref], but all instruction PCs are its own.
constexpr Load kPsO1dzt[] = {
    P(0x20, 1, cP60, 0xdc),
    P(0x74, 4, cP60, 0xd0),
    P(0x78, 4, cP60, 0x50),
    P(0xd4, 4, cP60, 0x90),
    P(0x180, 1, cFlags, 0x0),
    P(0x1ac, 1, cP70, 0xac),
    P(0x1cc, 2, cView, 0x218),
    P(0x2c8, 4, cP10, 0xf0),
    P(0x364, 4, cP30, 0xa0),
    P(0x368, 4, cP30, 0x20),
    P(0x470, 4, cP30, 0xb0),
    P(0x474, 4, cP30, 0x30),
    P(0x580, 4, cP30, 0xc0),
    P(0x584, 4, cP30, 0x40),
    P(0x690, 4, cP30, 0xd0),
    P(0x694, 4, cP30, 0x50),
    P(0x884, 16, cP20, 0x4f0),
    P(0x89c, 16, cP20, 0x530),
    P(0x8ac, 8, cP30, 0x80),
    P(0x8b0, 4, cP30, 0xe0),
    P(0x8b4, 1, cView, 0x22c),
    P(0x8b8, 4, cP30, 0x60),
    P(0x8bc, 8, cP30, 0x0),
    P(0x8f4, 1, cP0, 0x48),
    P(0x8fc, 4, cP30, 0x100),
    P(0x900, 2, cP0, 0x40),
    P(0x9dc, 1, cP0, 0xc),
    P(0xa04, 4, cP10, 0x0),
    P(0xa14, 1, cP90, 0x134),
    P(0xa30, 2, cP70, 0x68),
    P(0xa60, 1, cP70, 0x88),
    P(0xb88, 16, cP20, 0x570),
    P(0xba4, 4, cP20, 0x5b0),
    P(0xbd8, 8, cP70, 0x90),
    P(0xbe4, 8, cP60, 0x0),
    P(0xc0c, 4, cP60, 0x110),
    P(0xf10, 16, cP0, 0x0),
    P(0x1054, 4, cView, 0x220),
    P(0x1058, 8, cP10, 0x10),
    P(0x10dc, 1, cP50, 0x1c),
    P(0x10e0, 1, cP70, 0x9c),
    P(0x10e4, 1, cP60, 0xc),
    P(0x10e8, 1, cFlags, 0x0),
    P(0x10ec, 4, cP10, 0x30),
    P(0x1154, 2, cView, 0x218),
    P(0x1158, 1, cP0, 0x2c),
    P(0x115c, 4, cP0, 0x284),
    P(0x1160, 1, cP0, 0x3c),
    P(0x1164, 1, cP0, 0x26c),
    P(0x1168, 1, cP0, 0xc),
    P(0x12ac, 2, cP0, 0x270),
    P(0x1308, 2, cP0, 0x260),
    P(0x130c, 1, cP0, 0x268),
    P(0x13e0, 4, cP0, 0x2a0),
    P(0x13e4, 2, cP0, 0x278),
    P(0x13e8, 1, cP0, 0x280),
    P(0x14a0, 2, cP0, 0x294),
    P(0x14a4, 1, cP0, 0x29c),
    P(0x14f4, 1, cFlags, 0x0),
    P(0x14f8, 1, cP60, 0x11c),
    P(0x1520, 4, cView, 0x230),
    P(0x1598, 1, cView, 0x22c),
};
#undef V
#undef P

enum {
    tNormal, tSpecular, tShadow, tControl, tBase, tScreen,
    tRamp0, tRamp1, tRampShadow,
    tDepth, tDepthPlus, tDepthMinus,
    tScreenNormal, tNormalPlus, tNormalMinus,
    kTextures
};
struct Variant {
    const char *ps, *vs, *defines;
    std::span<const Load> vsLoads, psLoads;
    uint32_t texturePc[kTextures];
    uint32_t cubePc = 0;
    bool unmatched = false;
    const char* alternateVs = nullptr;
    std::span<const Load> alternateVsLoads{};
};
const Variant kVariants[] = {
    {"sm_o1dzt[skin].fxo", "s_b1dzt[skin].fxo", "#define SKIN\n",
     kVsSkin, kPsSkin,
     {0x1c8, 0x1d0, 0x154, 0xc3c, 0xd6c, 0xda0,
      0xcbc, 0xcc4, 0xe50, 0x1560, 0x1728, 0x1740,
      0x1568, 0x1600, 0x1618}},
    {"sm_o1dzt.fxo", "s_b1dst.fxo", "",
     kVsRigid, kPsRigid,
     {0x150, 0x158, 0x2b0, 0xa04, 0xa30, 0x90c,
      0, 0, 0, 0x1270, 0x1438, 0x1450,
      0x1278, 0x1310, 0x1328}},
    {"sm_b1ht[hair][offset][dedit].fxo", "s_b1dt[hair].fxo", "#define HAIR\n#define HAIR_HT\n",
     kVsHair, kPsHairHt,
     {0x12c, 0, 0x2b8, 0xaec, 0x110, 0xd50,
      0x964, 0x974, 0x98c, 0x99c, 0, 0, 0, 0, 0}},
    {"sm_b1dzt[hair][cmask].fxo", "s_b1dt[hair].fxo", "#define HAIR\n#define HAIR_CMASK\n",
     kVsHair, kPsHairCmask,
     {0x128, 0x824, 0x250, 0xa8c, 0xa68, 0xb80,
      0x878, 0x8c8, 0x838, 0x8b0, 0, 0, 0, 0, 0}},
    {"sm_p1dzt.fxo", "s_b1dst.fxo", "#define ALPHA_TEST\n",
     kVsRigid, kPsRigidAlpha,
     {0x150, 0x158, 0x2b0, 0xa04, 0xa30, 0x90c,
      0, 0, 0, 0x128c, 0x1454, 0x146c,
      0x1294, 0x132c, 0x1344}},
    {"sm_p1dzt[hair][cmask].fxo", "s_b1dt[hair].fxo", "#define HAIR\n#define HAIR_CMASK\n#define HAIR_P\n",
     kVsHair, kPsHairP,
     {0x128,0x7f8,0x254,0xb6c,0xa7c,0xb9c,0x8b4,0x8c4,0x83c,0x88c,0,0,0,0,0}},
    {"s_o1dzt[ref].fxo", "s_b1dst.fxo", "#define REF\n",
     kVsRigid, kPsRef,
     {0x150,0x158,0x2ac,0,0x920,0x928,0,0,0,0x1214,0x13cc,0x13e4,0x121c,0x12b4,0x12cc},0x918},
    {"s_o1dzt[eye][aref].fxo", "s_o1dzt[eye][aref].fxo", "#define EYE\n",
     kVsEye, kPsEye,
     {0x12c,0xccc,0x24c,0,0xe9c,0xe48,0,0,0,0,0,0,0,0,0},0xcb8},
    {"sm_p1ht[hair][offset][dedit].fxo", "s_b1dt[hair].fxo", "#define HAIR\n#define HAIR_HT\n#define HAIR_P\n#define HAIR_P_HT\n",
     kVsHair, kPsHairPHt,
     {0x138, 0, 0x2a8, 0xaec, 0x130, 0xb08,
      0x974, 0x994, 0x97c, 0x99c, 0, 0, 0, 0, 0}},
    {"s_o1dzt.fxo", "s_b1dst.fxo", "",
     kVsRigid, kPsO1dzt,
     {0x150,0x158,0x2b0,0,0xbc0,0xb4c,0,0,0,0x1178,0x1330,0x1348,0x1180,0x1218,0x1230},
     0, true, "s_o1dzt[mouth].fxo", kVsMouth},
};
constexpr int kVariantCount = std::size(kVariants);
constexpr int kO1dztRole = 10;

constexpr const char* kHeader = R"(#version 450
layout(set = 0, binding = 7, std430) readonly buffer C { uvec4 c[]; } cb;
vec4 cf(int i) { return uintBitsToFloat(cb.c[i]); }
const int V0 = 0, V10 = 32, V30 = 64, V40 = 104, VF0 = 105, VF1 = 113;
const int P0 = 128, P10 = 176, P20 = 192, P30 = 288, P50 = 312;
const int P60 = 320, P70 = 352, P90 = 368, View = 400, Flags = 440;
uint U(int b, int o) { return cb.c[b + o / 16][(o / 4) & 3]; }
float F(int b, int o) { return uintBitsToFloat(U(b, o)); }
vec2 F2(int b, int o) { return vec2(F(b,o), F(b,o+4)); }
vec3 F3(int b, int o) { return vec3(F(b,o), F(b,o+4), F(b,o+8)); }
vec4 F4(int b, int o) { return vec4(F(b,o), F(b,o+4), F(b,o+8), F(b,o+12)); }
float B(uint x) { return uintBitsToFloat(x); }
bool has(uint f, uint m) { return (f & m) != 0u; }

// Each helper retains the original individual arithmetic instruction boundary.
// Do not replace these with fma or reassociate their callers.
#define ARITH(T) \
T add(T a, T b) { precise T r = a + b; return r; } \
T sub(T a, T b) { precise T r = a - b; return r; } \
T mul(T a, T b) { precise T r = a * b; return r; } \
T mac(T a, T b, T c) { precise T r = a * b + c; return r; } \
T sat(T a) { return clamp(a, T(0.0), T(1.0)); }
ARITH(float)
ARITH(vec2)
ARITH(vec3)
ARITH(vec4)
#undef ARITH
#define MIXED(T) \
T mul(T a, float b) { precise T r = a * b; return r; } \
T mul(float a, T b) { precise T r = a * b; return r; } \
T mac(T a, float b, T c) { precise T r = a * b + c; return r; } \
T mac(float a, T b, T c) { precise T r = a * b + c; return r; }
MIXED(vec2)
MIXED(vec3)
MIXED(vec4)
#undef MIXED
float D3(vec3 a, vec3 b) {
    float r = mul(a.x,b.x);
    r = mac(a.y,b.y,r);
    return mac(a.z,b.z,r);
}
vec3 NORM(vec3 a) { return mul(a,inversesqrt(D3(a,a))); }
float POW(float a, float e) { return exp2(mul(log2(a),e)); }
float P5(float a) {
    float a2 = mul(a,a), a4 = mul(a2,a2);
    return mul(a4,a);
}
float planeSide(vec4 p, vec3 x, vec3 o) {
    float s = mul(p.x,x.x);
    s = mac(-p.x,o.x,s);
    s = mac(p.y,x.y,s);
    s = mac(-p.y,o.y,s);
    s = mac(p.z,x.z,s);
    return mac(-p.z,o.z,s);
}
)";

constexpr const char* kVs = R"(
layout(set = 0, binding = 0, std430) readonly buffer VBuf0 { uint w[]; } v0;
layout(set = 0, binding = 1, std430) readonly buffer VBuf1 { uint w[]; } v1;
layout(set = 0, binding = 2, std430) readonly buffer VBuf2 { uint w[]; } v2;
layout(set = 0, binding = 3, std430) readonly buffer VBuf3 { uint w[]; } v3;
layout(set = 0, binding = 4, std430) readonly buffer VBuf4 { uint w[]; } v4;
layout(set = 0, binding = 5, std430) readonly buffer VBuf5 { uint w[]; } v5;
// Palette: VSV[0x20], or hair VSV[0x30]; first descriptor use PC 0x74.
layout(set = 0, binding = 6, std430) readonly buffer Palette { uint w[]; } palette;
layout(location = 0) out vec4 p0;
layout(location = 1) out vec4 p1;
layout(location = 2) out vec4 p2;
layout(location = 3) out vec4 p3;
layout(location = 4) out vec4 p4;
layout(location = 5) out vec4 p5;
#ifdef EYE
layout(location = 6) out vec4 p6;
layout(location = 7) out vec4 p7;
#endif

uint word(uint b, uint i) {
    switch (b) {
    case 0u: return v0.w[i];
    case 1u: return v1.w[i];
    case 2u: return v2.w[i];
    case 3u: return v3.w[i];
    case 4u: return v4.w[i];
    default: return v5.w[i];
    }
}
uint conv(uint raw, uint bits, uint nfmt) {
    int sx = bitfieldExtract(int(raw),0,int(bits));
    if (nfmt == 4u) return raw;
    if (nfmt == 5u) return uint(sx);
    if (nfmt == 7u) return floatBitsToUint(unpackHalf2x16(raw).x);
    if (nfmt == 0u)
        return floatBitsToUint(float(raw) / float((1u << bits) - 1u));
    return floatBitsToUint(max(float(sx) / float((1u << (bits-1u)) - 1u),-1.0));
}
// This returns VGPR bits. In particular the bone indices are never numerically
// converted to float and back. The host permits only exact uint/sint decoding.
uvec4 fetchBits(uint b) {
    uvec4 s = cb.c[449 + int(b)];
    uint i = s.x + uint(gl_VertexIndex) * s.y;
    uint dfmt = s.z & 15u, nfmt = (s.z >> 4) & 7u;
    uvec4 e = uvec4(0u,0u,0u,floatBitsToUint(1.0));
    if (dfmt == 4u || dfmt == 11u || dfmt == 13u || dfmt == 14u) {
        uint n = dfmt == 4u ? 1u : dfmt == 11u ? 2u : dfmt == 13u ? 3u : 4u;
        for (uint c = 0u; c < n; ++c) e[c] = word(b,i+c);
    } else if (dfmt == 2u || dfmt == 5u || dfmt == 12u) {
        uint n = dfmt == 2u ? 1u : dfmt == 5u ? 2u : 4u;
        for (uint c = 0u; c < n; ++c)
            e[c] = conv((word(b,i+c/2u) >> (16u*(c&1u))) & 0xffffu,16u,nfmt);
    } else {
        uint n = dfmt == 1u ? 1u : dfmt == 3u ? 2u : 4u;
        uint x = word(b,i);
        for (uint c = 0u; c < n; ++c)
            e[c] = conv((x >> (8u*c)) & 0xffu,8u,nfmt);
    }
    return e;
}
vec4 fetch(uint b) { return uintBitsToFloat(fetchBits(b)); }
vec4 boneRow(uint index, uint row) {
    uint i = cb.c[456].x + (48u*index + 16u*row)/4u;
    return uintBitsToFloat(uvec4(palette.w[i],palette.w[i+1u],
                                palette.w[i+2u],palette.w[i+3u]));
}
void main() {
    vec3 localPosition = fetch(0u).xyz;
    vec4 fw = fetch(1u);
    uvec4 indices = fetchBits(2u);
    vec4 en = fetch(3u), et = fetch(4u);
    vec2 uv = fetch(5u).xy;

    float sum = add(fw.y,fw.z);
    sum = add(sum,fw.w);
    vec4 weights = vec4(sub(1.0,sum),fw.yzw);
    vec4 bone[12];
    for (int k = 0; k < 4; ++k)
        for (int r = 0; r < 3; ++r)
            bone[3*k+r] = boneRow(indices[k],uint(r));

    vec3 M[3];
    vec3 position;
    for (int r = 0; r < 3; ++r) {
        vec3 a = mul(bone[r].xyz,weights.x);
        a = mac(bone[3+r].xyz,weights.y,a);
        a = mac(bone[6+r].xyz,weights.z,a);
        M[r] = mac(bone[9+r].xyz,weights.w,a);

        float anchor = floor(bone[r].w);
        float t = sub(bone[r].w,anchor);
        t = mac(t,weights.x,anchor);
        t = mac(weights.y,sub(bone[3+r].w,anchor),t);
        t = mac(weights.z,sub(bone[6+r].w,anchor),t);
        t = mac(weights.w,sub(bone[9+r].w,anchor),t);
        t = mac(M[r].x,localPosition.x,t);
        t = mac(M[r].y,localPosition.y,t);
        position[r] = mac(M[r].z,localPosition.z,t);
    }
    int count = int(U(V0,0x12c));
    if (count > 0) {
        vec3 displacement = vec3(0.0);
        for (int i = 0; i < count; ++i) {
            vec4 plane = F4(V0,0xd0+16*i);
            vec4 origin = F4(V0,0x50+16*i);
            vec4 shift = F4(V0,0x90+16*i);
            float selected = planeSide(plane,position,origin.xyz) >= 0.0 ? 1.0 : 0.0;
            displacement = mac(selected,mul(shift.xyz,shift.w),displacement);
        }
        position = add(displacement,position);
    }

    uint flags = U(V40,0);
    vec3 world = add(F3(V10,0x170),position);
    vec3 q = world;
    if (!(has(flags,0x80000u) && !has(flags,0x100000u)))
        q = sub(world,F3(V30,0x220));
    vec3 lp = world;
    if (has(flags,0x80000u)) lp = add(F3(V10,0x160),world);

    vec3 delta = sub(lp,F3(V30,0x220));
    vec3 away = NORM(delta);
    vec3 e = vec3(D3(F3(V30,0x40),q),
                  D3(F3(V30,0x50),q),
                  D3(F3(V30,0x60),q));
    vec4 clip;
    for (int r = 0; r < 4; ++r) {
        vec4 P = F4(V30,16*r);
        float t = mac(P.x,e.x,P.w);
        t = mac(P.y,e.y,t);
        clip[r] = mac(P.z,e.z,t);
    }
    vec4 pos = clip;
    if (cb.c[455].z != 0u) pos.z = mul(add(pos.z,pos.w),0.5);
    gl_Position = pos;

    vec3 transmission = vec3(1.0), scattering = vec3(0.0);
    if (has(flags,0x20000u)) {
        vec4 f0 = F4(VF0,0x40), f1 = F4(VF0,0x50);
        vec4 f2 = F4(VF0,0x60), f3 = F4(VF0,0x70);
        float d = D3(F3(VF1,0x80),away);
        float denominator = mac(f2.w,-d,f1.w);
        float r = inversesqrt(denominator);
        float r2 = mul(r,r), r3 = mul(r2,r);
        float angular = mac(d,d,f3.w);
        vec3 phase = mul(f1.xyz,angular);
        phase = mac(f2.xyz,r3,phase);
        vec3 tinted = mul(f3.xyz,phase);
        vec3 exponent = mul(f0.xyz,clip.w);
        exponent = mul(B(0x3fb8aa3bu),exponent);
        transmission = exp2(exponent);
        scattering = mul(f3.w,tinted);
        scattering = mac(tinted,-transmission,scattering);
    }
    if (has(flags,0x40000u)) scattering = vec3(0.0);

    float decode = B(0x40008102u);
    vec3 on = mac(decode,en.xyz,vec3(-1.0));
    vec3 ot = mac(decode,et.xyz,vec3(-1.0));
    float handed = mac(decode,et.w,-1.0);
    vec3 normal, tangent;
    for (int r = 0; r < 3; ++r) {
        normal[r] = D3(M[r],on);
        tangent[r] = D3(M[r],ot);
    }
    p0 = vec4(transmission,lp.z);
    p1 = vec4(scattering,0.0);
#ifdef SKIN
    p2 = vec4(uv,en.w,0.0);
#else
#ifdef HAIR
    p2 = vec4(uv,F2(457,0x28));
#else
    p2 = vec4(uv,0.0,0.0);
#endif
#endif
#ifdef EYE
    vec3 axis = NORM(vec3(M[0].z,M[1].z,M[2].z));
    p6 = vec4(axis,0.0);
    p7 = vec4(away,0.0);
    p3 = vec4(-NORM(add(axis,F3(V30,0x60))),lp.x);
#else
    p3 = vec4(away,lp.x);
#endif
    p4 = vec4(normal,lp.y);
    p5 = vec4(tangent,handed);
}
)";

constexpr const char* kFs = R"(
layout(set = 0, binding = 8) uniform sampler2D tex[15];
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 3) in vec4 a3;
layout(location = 4) in vec4 a4;
layout(location = 5) in vec4 a5;
#ifdef EYE
layout(location = 6) in vec4 a6;
layout(location = 7) in vec4 a7;
#endif
#if defined(REF) || defined(EYE)
layout(set = 0, binding = 9) uniform samplerCube environment;
#endif
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;

vec2 sc(int t) {
    vec4 s = cf(441+t/2);
    return (t&1) != 0 ? s.zw : s.xy;
}
vec4 at(int t, vec2 uv) { return texture(tex[t],mul(uv,sc(t))); }
float luminance(vec3 c) {
    float r = mul(B(0x3e991687u),c.r);
    r = mac(c.g,B(0x3f1645a2u),r);
    return mac(c.b,B(0x3de978d5u),r);
}
float polynomial(int channel, vec3 n) {
    int o = 0x500+0x40*channel;
    vec4 A = F4(P20,o), Bc = F4(P20,o+16);
    vec4 Cc = F4(P20,o+32), D = F4(P20,o+48);
    float b = D3(Bc.xyz,n), c = D3(Cc.xyz,n), a = D3(A.xyz,n);
    float linear = add(D.x,A.w);
    linear = add(linear,a);
    float v = D.w;
    v = mac(D.y,n.y,v);
    v = mac(D.z,n.z,v);
    v = mac(b,n.y,v);
    v = mac(Bc.w,n.y,v);
    v = mac(c,n.z,v);
    v = mac(Cc.w,n.z,v);
    v = mac(linear,n.x,v);
    return max(B(0x3d800000u),v);
}
float directionalSpec(vec3 d, float nl, vec3 reflectedV, vec3 V,
                      float exponent, float normalization, float floor_, float rest) {
    float lobe = POW(min(1.0,max(0.0,-D3(d,reflectedV))),exponent);
    lobe = mul(lobe,normalization);
    vec3 h = NORM(add(d,V));
    float f = sub(1.0,D3(h,V));
    float fresnel = mac(P5(f),rest,floor_);
    lobe = mul(lobe,fresnel);
    return mul(lobe,nl);
}
float tone(float material, float transmission) {
    float et = mul(F(P10,0),transmission);
    float x = mul(et,material);
    x = mul(2.0,x);
    float di = mac(B(0x3e19999au),x,0.5);
    float den = mac(x,di,B(0x3d75c290u));
    float ni = mac(B(0x3e19999au),x,B(0x3d4ccccdu));
    float num = mac(ni,x,B(0x3b83126fu));
    float ratio = mul(1.0/den,num);
    return sqrt(sat(mac(B(0x3fb0852eu),ratio,B(0xbdbc49cau))));
}
float normalResponse(vec3 encoded, vec3 normal) {
    float yz = mul(encoded.y,normal.y), zz = mul(encoded.z,normal.z);
    float r = mul(encoded.x,normal.x);
    r = mul(2.0,r);
    r = mac(2.0,yz,r);
    r = mac(2.0,zz,r);
    float sum = add(normal.x,normal.y);
    sum = add(sum,normal.z);
    return sub(r,sum);
}
vec4 swz(vec4 v, uint s) {
    return vec4(v[s&3u],v[(s>>2)&3u],v[(s>>4)&3u],v[(s>>6)&3u]);
}
// Compressed colour exports use f32->f16 toward zero. Start with the GLSL
// half conversion and step back one half ULP if it rounded away from zero.
// This also turns a finite overflow rounded to infinity into max finite.
float halfRTZ(float x) {
    uint h = packHalf2x16(vec2(x,0.0)) & 0xffffu;
    float y = unpackHalf2x16(h).x;
    if (!isnan(x) && abs(y) > abs(x)) h -= 1u;
    return unpackHalf2x16(h).x;
}
vec4 half4(vec4 v) {
    return vec4(halfRTZ(v.x),halfRTZ(v.y),halfRTZ(v.z),halfRTZ(v.w));
}

#ifdef HAIR
vec3 hairCross(vec3 a, vec3 b) {
    return vec3(mac(b.y,-a.z,mul(a.y,b.z)),
                mac(b.z,-a.x,mul(a.z,b.x)),
                mac(b.x,-a.y,mul(a.x,b.y)));
}
float hairPolynomial(int ch, vec3 n) {
#ifdef HAIR_HT
    int o=0x500+64*ch;
    vec4 A=F4(P20,o), Bb=F4(P20,o+16), Cc=F4(P20,o+32), D=F4(P20,o+48);
    float a=add(D3(A.xyz,n),A.w), b=add(D3(Bb.xyz,n),Bb.w);
    float c=add(D3(Cc.xyz,n),Cc.w), d=add(D3(D.xyz,n),D.w);
    float v=mul(n.x,a);
    v=mac(b,n.y,v);
    v=mac(c,n.z,v);
    return max(B(0x3d800000u),add(v,d));
#else
    return polynomial(ch,n);
#endif
}
float hairLookup(int tx, int ty, vec3 H, vec3 N, vec3 Cn, vec3 bn) {
    float x=-D3(H,N);
    float y=mac(B(0x3e4ccccdu),mul(a2.w,-D3(Cn,H)),0.5);
    float p=at(tx,vec2(x,y)).r;
    y=mac(B(0x3e4ccccdu),mul(a2.z,-D3(H,bn)),0.5);
    return mul(p,at(ty,vec2(x,y)).r);
}
void main() {
    vec3 position=vec3(a3.w,a4.w,a0.w), unfolded=position;
    for (int i=0; F(P60,0xdc)>float(i); ++i) {
        vec4 plane=F4(P60,0xd0+16*i), origin=F4(P60,0x50+16*i);
#ifdef HAIR_HT
        float side=D3(plane.xyz,sub(unfolded,origin.xyz));
#else
        float side=planeSide(plane,unfolded,origin.xyz);
#endif
        if (abs(side)<origin.w) demote;
        vec4 shift=F4(P60,0x90+16*i);
        unfolded=mac(side>=0.0?1.0:0.0,-mul(shift.xyz,shift.w),unfolded);
    }
    uint flags=U(Flags,0), viewFlags=U(View,0x22c);
    vec2 uv=a2.xy, rasterUV=mul(F2(View,0x218),gl_FragCoord.xy);
    vec3 gn=a4.xyz, tangent=a5.xyz;
    vec3 bt=mul(hairCross(gn,tangent),a5.w);
    vec2 encoded=at(0,uv).rg;
#ifdef HAIR_HT
    // CVT_OFF_F32_I4(4) is exactly 0.25, not 4.0.
    vec4 base=at(4,uv);
    float offset=add(B(0xbefefeffu),base.r);
    offset=mac(0.25,offset,B(0x3efefeffu));
    offset=add(-0.5,offset);
    offset=mac(0.5,offset,0.5);
    float om=sub(1.0,offset);
    float hi=mul(2.0,sub(1.0,encoded.g));
    hi=mac(-hi,om,1.0);
    float lo=mul(2.0,mul(encoded.g,offset));
    float ey=encoded.g<0.5?lo:hi;
    float exlo=mul(2.0,mul(0.5,encoded.r));
    float exhi=mul(2.0,sub(1.0,encoded.r));
    exhi=mac(-0.5,exhi,1.0);
    float ex=encoded.r<0.5?exlo:exhi;
    vec2 nxy=vec2(mac(2.0,ex,-1.0),mac(2.0,ey,-1.0));
    vec3 un=mul(nxy.y,bt);
    un=mac(tangent,nxy.x,un);
#else
    vec2 nxy=mac(2.0,encoded,vec2(-1.0));
    vec3 un=mul(tangent,nxy.x);
    un=mac(nxy.y,bt,un);
#endif
    float xy2=mul(nxy.x,nxy.x);
    xy2=mac(nxy.y,nxy.y,xy2);
    un=mac(gn,sqrt(sat(sub(1.0,xy2))),un);
    vec3 N=NORM(un);
    float sr=1.0, sg=1.0;
    if (has(flags,8u)) {
        vec2 sh=at(2,rasterUV).rg;
        vec4 r=F4(P10,0xf0);
        sr=mac(r.x,sh.r,r.y);
        sg=mac(sh.g,r.z,r.w);
    }
    float shadow=has(viewFlags,2u)?mul(sr,sg):min(sr,sg);
    vec3 pointDiffuse=vec3(0.0);
    float occ[4];
    for (int i=0;i<4;++i) occ[i]=0.0;
    if (has(flags,1u)) {
        for (int i=0;i<4;++i) if (has(flags,0x10000u<<i)) {
            vec4 pp=F4(P30,0xa0+16*i), pc=F4(P30,0x20+16*i);
            vec3 delta=sub(position,pp.xyz);
            float d2=D3(delta,delta), inv=inversesqrt(d2);
            float outside=max(0.0,sub(sqrt(d2),pc.w));
#ifdef HAIR_HT
            float den=mac(mul(outside,outside),pp.w,1.0);
            float att=sat(mul(B(0x3f808102u),add(B(0xbb808081u),1.0/den)));
            float nl=min(1.0,max(0.0,-D3(N,mul(delta,inv))));
#else
            float den=mac(mul(pp.w,outside),outside,1.0);
            float att=sat(mac(B(0x3f808102u),1.0/den,B(0xbb810205u)));
            float nl=sat(mul(-inv,D3(N,delta)));
#endif
            vec3 diffuse=max(vec3(0.0),mul(mul(pc.rgb,att),nl));
            if (i==0) pointDiffuse=diffuse;
            else pointDiffuse=add(pointDiffuse,diffuse);
            occ[i]=mul(nl,luminance(diffuse));
        }
    }
    float invV=inversesqrt(D3(a3.xyz,a3.xyz));
    vec3 V=mul(a3.xyz,invV);
    vec3 direction0=F3(P30,0x80), direction1=F3(P30,0x90);
    vec3 H0=NORM(mac(a3.xyz,invV,direction0));
    vec3 H1=NORM(mac(a3.xyz,invV,direction1));
    vec3 bn=mul(hairCross(N,tangent),a5.w);
    vec3 cn=hairCross(bn,N);
    float facing=mul(sat(add(1.0,D3(V,gn))),sat(add(1.0,D3(V,N))));
    float lookup0=hairLookup(6,7,H0,N,cn,bn);
    float lookup1=hairLookup(8,9,H1,N,cn,bn);
#ifdef HAIR_HT
    float exponent=F(P70,0xac);
    vec3 specMat=vec3(1.0);
#else
    vec3 specMat=at(1,uv).rgb;
    float exponent=max(1.0,mul(F(P70,0xac),specMat.b));
    vec4 base=at(4,uv);
#endif
    float normalization=mac(B(0x3d22ca2fu),exponent,B(0x3daf7aacu));
    float l0=mul(POW(abs(lookup0),exponent),normalization);
    float l1=mul(POW(abs(lookup1),exponent),normalization);
    float floor_=mul(F(P70,0x88),F(P10,0x0c));
    float rest=mac(-F(P70,0x88),F(P10,0x0c),1.0);
    float f0=sub(1.0,D3(V,H0)), f1=sub(1.0,D3(V,H1));
#ifdef HAIR_HT
    float f02=mul(f0,f0), f12=mul(f1,f1);
    float fr0=mul(rest,f02); fr0=mul(fr0,f02); fr0=mac(fr0,f0,floor_);
    float fr1=mul(rest,f12); fr1=mul(fr1,f12); fr1=mac(f1,fr1,floor_);
#else
    float fr0=mac(P5(f0),rest,floor_), fr1=mac(P5(f1),rest,floor_);
#endif
    l0=mul(l0,fr0); l1=mul(l1,fr1);
    float raw0=D3(direction0,N), raw1=D3(direction1,N);
    float nl0=min(1.0,max(0.0,-raw0)), nl1=min(1.0,max(0.0,-raw1));
    vec3 light0=F3(P30,0), light1=F3(P30,0x10);
    vec4 control=at(3,uv);
    float materialShadow=mac(control.r,sub(shadow,1.0),1.0);
    float effect=POW(facing,add(2.0,F(P70,0x6c)));
#ifdef HAIR_HT
    effect=mul(effect,add(B(0x40200000u),F(P70,0x68)));
    effect=mul(B(0x3dcccccdu),effect);
    effect=mul(F(P30,0x10c),effect);
    vec3 baseSquared=mul(F3(P90,0x30),mul(base.g,base.g));
    baseSquared=mac(control.g,mul(F3(P30,0x100),effect),baseSquared);
#else
    effect=mul(specMat.g,effect);
    effect=mul(F(P30,0x10c),effect);
    effect=mul(effect,add(B(0x40200000u),F(P70,0x68)));
    effect=mul(B(0x3dcccccdu),effect);
    vec3 baseSquared=mul(F3(P30,0x110),mul(base.rgb,base.rgb)); // 0xb98/0xbf8/0xc98: the colour mask's tint
    baseSquared=mac(effect,mul(F3(P30,0x100),control.g),baseSquared);
#endif
    float primaryLum=luminance(light0);
    float ol=mul(F(P30,0x128),max(0.0,sub(F(P30,0x124),raw0)));
#ifdef HAIR_HT
    for (int i=0;i<4;++i) ol=mac(1.0/primaryLum,occ[i],ol);
    float upper=mac(-sub(1.0,materialShadow),F(P30,0x12c),1.0);
    float ao=sub(1.0,sat(mul(F(P30,0x120),
                 mul(sub(1.0,max(0.0,min(ol,upper))),sub(1.0,control.a)))));
#else
    float po=add(occ[0],occ[1]); po=add(po,occ[2]); po=add(po,occ[3]);
    ol=mac(po,1.0/primaryLum,ol);
    float upper=F(P30,0x12c);
    upper=mac(-F(P30,0x12c),materialShadow,upper);
    upper=sub(1.0,upper);
    float ao=sub(1.0,control.a);
    ao=mac(max(0.0,min(upper,ol)),-ao,ao);
    ao=sat(mac(-F(P30,0x120),ao,1.0));
#endif
    vec3 an=NORM(N);
    vec3 poly=vec3(hairPolynomial(0,an),hairPolynomial(1,an),hairPolynomial(2,an));
    vec3 lower=F3(P30,0xe0), diff=sub(F3(P30,0x60),lower);
    float hemisphere=mac(0.5,N.y,0.5);
#ifdef HAIR_HT
    vec3 ambient=mac(hemisphere,diff,lower);
    ambient=mac(F(P20,0x4f0),poly,ambient);
#else
    vec3 ambient=mac(F(P20,0x4f0),poly,lower);
    float red=mac(diff.r,N.y,diff.r);
    ambient.gb=mac(hemisphere,diff.gb,ambient.gb);
#endif
    vec3 primary=mul(light0,nl0), secondary=mul(light1,nl1);
#ifdef HAIR_HT
    primary=mul(primary,materialShadow);
    vec3 dl=mac(F(P90,0x134),secondary,primary);
    dl=add(ambient,dl);
#else
    vec3 dl=mac(primary,materialShadow,ambient);
    dl=mac(F(P90,0x134),secondary,dl);
    dl.r=mac(0.5,red,dl.r);
#endif
    float ir=1.0/sub(F(P0,0x48),F(P0,0x44));
#ifdef HAIR_HT
    float response=sat(mul(add(-F(P0,0x44),-raw0),ir));
#else
    float response=mul(-raw0,ir);
    response=sat(mac(-F(P0,0x44),ir,response));
#endif
    float directionalResponse=mac(F(P0,0x40),add(-1.0,response),1.0);
    vec3 screen=at(5,rasterUV).rgb;
    vec3 indirect=mac(screen,1.0/F(P10,8),pointDiffuse);
    vec3 diffuseLighting=mac(dl,directionalResponse,indirect);
#ifdef HAIR_HT
    vec3 specularLighting=mul(mul(light0,l0),materialShadow);
#else
    vec3 specularLighting=mul(light0,materialShadow);
    specularLighting=mul(specularLighting,l0);
#endif
    specularLighting=mac(light1,l1,specularLighting);
#ifdef HAIR_HT
    vec3 dm=mul(F3(P70,0x90),F3(P60,0));
    vec3 sm=mul(F3(P70,0xa0),F3(P60,0x10));
    vec3 linear=mul(mul(diffuseLighting,baseSquared),dm);
    linear=mac(specularLighting,sm,linear);
    vec3 tinted=mac(-linear,ao,F3(P60,0x110));
    tinted=mul(F(P60,0x11c),tinted);
    linear=mac(vec3(ao),linear,tinted);
    vec3 x=mul(F(P10,0),mul(linear,a0.rgb));
    x=mul(x,2.0);
#else
    vec3 dm=mul(F3(P60,0),mul(F3(P70,0x90),specMat.g));
    vec3 sm=mul(F3(P70,0xa0),F3(P60,0x10));
    vec3 linear=mul(mul(dm,baseSquared),diffuseLighting);
    linear=mac(mul(sm,specularLighting),specMat.r,linear);
    linear=mul(ao,linear);
    vec3 old=linear;
    linear=mac(F3(P60,0x110),F(P60,0x11c),old);
    linear=mac(-F(P60,0x11c),old,linear);
    vec3 x=mul(mul(F(P10,0),a0.rgb),linear);
    x=mul(x,2.0);
#endif
    vec3 toned;
    for (int i=0;i<3;++i) {
        float den=mac(x[i],mac(B(0x3e19999au),x[i],0.5),B(0x3d75c290u));
        float num=mac(mac(B(0x3e19999au),x[i],B(0x3d4ccccdu)),x[i],B(0x3b83126fu));
#ifdef HAIR_HT
        float ratio=mac(num,1.0/den,B(0xbd888888u));
        toned[i]=sqrt(sat(mul(B(0x3fb0852eu),ratio)));
#else
        float ratio=mul(1.0/den,num);
        toned[i]=sqrt(sat(mac(B(0x3fb0852eu),ratio,B(0xbdbc49cau))));
#endif
    }
    if (has(U(P0,0x0c),1u)) {
#ifdef HAIR_HT
        vec3 brightness=add(add(mul(light0,nl0),mul(light1,nl1)),indirect);
        float b=mul(B(0x3eaaaaabu),brightness.r);
        b=mac(brightness.g,B(0x3eaaaaabu),b);
        b=mac(brightness.b,B(0x3eaaaaabu),b);
#else
        vec3 brightness=add(add(indirect,mul(light0,nl0)),mul(light1,nl1));
        float b=mac(add(brightness.g,brightness.b),B(0x3eaaaaabu),mul(B(0x3eaaaaabu),brightness.r));
#endif
        float om=sub(1.0,b), low=sat(mul(om,om));
        float mid=mac(-om,om,1.0); mid=sat(mac(b,-b,mid));
        float high=sat(mul(b,b));
        float black=F(P0,0x1c), irange=1.0/sub(sub(1.0,black),black);
        for (int i=0;i<3;++i) {
            vec3 controls=F3(P0,0x10+16*i);
            float e=mul(controls.x,low);
            e=mac(controls.y,mid,e);
            e=mac(controls.z,high,e);
            e=add(F(P0,4*i),e);
            toned[i]=sat(POW(sat(mac(toned[i],irange,-black)),e));
        }
    }
    vec3 color=toned;
    if (has(flags,4u)) color=add(toned,a1.rgb);
    if (has(flags,0x400000u) && !has(flags,4u)) {
        vec3 delta=sub(F3(View,0x220),position);
        float f=sat(mac(F(P10,0x28),sqrt(D3(delta,delta)),F(P10,0x2c)));
#ifdef HAIR_HT
        color=mac(f,sub(F3(P10,0x10),toned),toned);
#else
        color=mac(F3(P10,0x10),f,toned);
        color=mac(f,-toned,color);
#endif
    }
    if (F(P70,0x80)>base.a) demote;
    float alpha=mul(F(P70,0x9c),F(P50,0x1c));
    alpha=mul(F(P60,0x0c),alpha);
    alpha=mul(alpha,base.a);
    if (!has(flags,8u)) {
        color=mul(F3(P10,0x30),color);
        alpha=mul(F(P10,0x3c),alpha);
    }
#ifdef HAIR_P
    float exportAlpha=alpha;
#endif
    if (has(flags,0x800000u)) {
        vec4 b=F4(View,0x230);
#ifdef HAIR_HT
        color=mac(b.w,sub(color,b.rgb),b.rgb);
#else
        color=mac(b.w,color,b.rgb);
        color=mac(-b.rgb,b.w,color);
#endif
        alpha=mul(b.w,alpha);
    }
    uvec4 so=cb.c[455];
#ifdef HAIR_P
#ifdef HAIR_P_HT
    float depthAlpha=mul(B(0x3e4ccccdu),gl_FragCoord.z);
    depthAlpha=sat(mul(B(0x3b808081u),depthAlpha));
    float alpha0=has(viewFlags,1u)?depthAlpha:exportAlpha;
#else
    float alpha0=has(viewFlags,1u)?sat(mul(B(0x3a4d9a68u),gl_FragCoord.z)):exportAlpha;
#endif
    o0=swz(half4(vec4(color,alpha0)),so.x);
#else
    o0=swz(half4(vec4(color,alpha)),so.x);
#endif
    o1=swz(half4(vec4(0.0,0.0,0.0,alpha)),so.y);
}
#elif defined(EYE)
void main() {
    vec3 position=vec3(a3.w,a4.w,a0.w), unfolded=position;
    if (F(P60,0xdc)>0.0) {
        vec4 plane=F4(P60,0xd0), origin=F4(P60,0x50), shift=F4(P60,0x90);
        float side=planeSide(plane,unfolded,origin.xyz);
        if (abs(side)<origin.w) demote;
        unfolded=mac(side>=0.0?1.0:0.0,-mul(shift.xyz,shift.w),unfolded);
    }
    uint flags=U(Flags,0), viewFlags=U(View,0x22c);
    vec2 rasterUV=mul(F2(View,0x218),gl_FragCoord.xy);
    vec3 gn=a4.xyz, t=a5.xyz, bt;
    bt.x=mac(t.y,-gn.z,mul(gn.y,t.z));
    bt.y=mac(t.z,-gn.x,mul(gn.z,t.x));
    bt.z=mac(t.x,-gn.y,mul(gn.x,t.y));
    bt=mul(a5.w,bt);
    vec2 xy=mac(2.0,at(0,a2.xy).rg,vec2(-1.0));
    float n2=mac(xy.y,xy.y,mul(xy.x,xy.x));
    vec3 un=mul(t,xy.x);
    un=mac(bt,xy.y,un);
    un=mac(gn,sqrt(sat(sub(1.0,n2))),un);
    vec3 N=NORM(un);
    vec2 sh=vec2(1.0);
    if (has(flags,8u)) {
        vec2 raw=at(2,rasterUV).rg; vec4 remap=F4(P10,0xf0);
        sh=vec2(mac(remap.x,raw.x,remap.y),mac(raw.y,remap.z,remap.w));
    }
    float halfLight=mac(-0.5,D3(F3(P30,0x80),a6.xyz),0.5);
    vec3 point=vec3(0.0);
    for (int i=0;i<4;++i) if (has(flags,0x10000u<<i)) {
        vec4 pp=F4(P30,0xa0+16*i), pc=F4(P30,0x20+16*i);
        vec3 delta=sub(position,pp.xyz);
        float d2=D3(delta,delta);
        bool diffuse=has(flags,1u);
        float outside=max(0.0,sub(sqrt(diffuse?d2:max(B(0x3a83126fu),d2)),pc.w));
        float den=mac(mul(pp.w,outside),outside,1.0);
        float att=sat(mac(B(0x3f808102u),1.0/den,B(0xbb810205u)));
        vec3 L=diffuse?mul(delta,inversesqrt(d2)):delta;
        float candidate=mul(att,D3(a6.xyz,L));
        candidate=mac(-0.5,candidate,mul(0.5,att));
        halfLight=max(halfLight,candidate);
        if (diffuse) {
            float nl=min(1.0,max(0.0,-D3(N,L)));
            vec3 diff=max(vec3(0.0),mul(mul(pc.rgb,att),nl));
            if (i==0) point=diff; else point=add(point,diff);
        }
    }
    // 8dc..10e8, in original instruction order.
    float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51;
    vec4 sampleValue;
    r2=sh.x; r3=sh.y; r5=N.x; r8=N.y; r6=N.z;
    r7=halfLight; r11=a7.x; r13=a3.y;
    r24=position.x; r23=position.y; r22=position.z;
    r26=point.r; r51=point.g; r27=point.b;
    r28 = sat(add(0.5,r7));
    r7 = a7.y;
    r9 = F(View,0x44);
    r10 = F(P70,0x60);
    r14 = F(P90,0x40);
    r15 = F(P90,0x44);
    r14 = mac(-F(View,0x40),r10,r14);
    r12 = F(View,0x48);
    r16 = F(P90,0x48);
    r19 = F(P70,0x64);
    r15 = mac(-F(P70,0x60),r9,r15);
    r9 = a3.x;
    r14 = add(r9,r14);
    r15 = add(r13,r15);
    r10 = a3.z;
    r14 = mac(-F(View,0x50),r19,r14);
    r16 = mac(-F(P70,0x60),r12,r16);
    r16 = add(r10,r16);
    r15 = mac(-F(View,0x54),r19,r15);
    r29 = F(P90,0x50);
    r16 = mac(-F(View,0x58),r19,r16);
    r19 = F(P70,0x68);
    r29 = mac(-F(View,0x40),r19,r29);
    r12 = a7.z;
    r30 = F(P90,0x58);
    r30 = mac(-F(View,0x48),r19,r30);
    r9 = add(r9,r29);
    r29 = F(P70,0x6c);
    r9 = mac(-F(View,0x50),r29,r9);
    r10 = add(r10,r30);
    r30 = F(P90,0x54);
    r30 = mac(-F(View,0x44),r19,r30);
    r19 = mul(r14,r14);
    r19 = mac(r15,r15,r19);
    r19 = mac(r16,r16,r19);
    r19 = inversesqrt(r19);
    r15 = mul(r19,r15);
    r13 = add(r13,r30);
    r13 = mac(-F(View,0x54),r29,r13);
    r10 = mac(-F(View,0x58),r29,r10);
    r29 = mul(r19,r14);
    r14 = mul(r19,r16);
    r16 = mul(r5,r29);
    r16 = mac(r15,r8,r16);
    r16 = mac(r14,r6,r16);
    r14 = max(0.0,-r16);
    r14 = min(1.0,r14);
    r29 = log2(r14);
    r14 = mul(r11,r11);
    r14 = mac(r7,r7,r14);
    r14 = mac(r12,r12,r14);
    r14 = inversesqrt(r14);
    r33 = mul(r11,r14);
    r11 = mul(r12,r14);
    r12 = mul(r7,r14);
    r7 = mul(r5,r33);
    r15 = mac(r8,r12,r7);
    r15 = mul(mac(r11,r6,r15),2.0);
    r14 = F(P20,0x53c);
    r33 = mac(r15,-r5,r33);
    r35 = mul(r33,r33);
    r12 = mac(r15,-r8,r12);
    r35 = mac(r12,r12,r35);
    r11 = mac(r15,-r6,r11);
    r15 = mul(F(P30,0x80),r5);
    r35 = mac(r11,r11,r35);
    r19 = mac(F(P30,0x84),r8,r15);
    r15 = mul(r5,r5);
    r15 = mac(r8,r8,r15);
    r15 = mac(r6,r6,r15);
    r15 = inversesqrt(r15);
    r34 = mul(r15,r5);
    r19 = mac(F(P30,0x88),r6,r19);
    r16 = mul(F(P30,0x90),r5);
    r16 = mac(F(P30,0x94),r8,r16);
    r16 = mac(F(P30,0x98),r6,r16);
    r16 = max(0.0,-r16);
    r16 = min(1.0,r16);
    r30 = add(-1.0,r16);
    r16 = mul(r9,r9);
    r16 = mac(r13,r13,r16);
    r16 = mac(r10,r10,r16);
    r16 = inversesqrt(r16);
    r31 = mul(r16,r9);
    r7 = mul(r5,r31);
    r9 = mul(r16,r10);
    r10 = mul(r16,r13);
    r7 = mac(r10,r8,r7);
    r10 = mul(r15,r8);
    r14 = mac(F(P20,0x534),r10,r14);
    r7 = mac(r9,r6,r7);
    r9 = mul(r15,r6);
    r6 = max(0.0,-r7);
    r5 = min(1.0,r6);
    r7 = mac(F(P20,0x538),r9,r14);
    r32 = log2(r5);
    r5 = F(View,0x6c);
    r5 = mac(F(View,0x60),r24,r5);
    r31 = mac(r23,F(View,0x64),r5);
    r31 = mac(F(View,0x68),r22,r31);
    r39 = mul(F(View,0x38),r31);
    r14 = F(P20,0x5bc);
    r14 = mac(F(P20,0x5b4),r10,r14);
    r6 = mul(F(P20,0x510),r34);
    r5 = mac(F(P20,0x514),r10,r6);
    r6 = mul(F(P20,0x590),r34);
    r5 = mac(F(P20,0x518),r9,r5);
    r6 = mac(F(P20,0x594),r10,r6);
    r6 = mac(F(P20,0x598),r9,r6);
    r7 = mac(r5,r10,r7);
    r5 = mac(F(P20,0x5b8),r9,r14);
    r14 = mul(F(P20,0x520),r34);
    r14 = mac(F(P20,0x524),r10,r14);
    r14 = mac(F(P20,0x528),r9,r14);
    r5 = mac(r6,r10,r5);
    r7 = mac(F(P20,0x51c),r10,r7);
    r5 = mac(F(P20,0x59c),r10,r5);
    r7 = mac(r14,r9,r7);
    r14 = mul(F(P20,0x5a0),r34);
    r14 = mac(F(P20,0x5a4),r10,r14);
    r14 = mac(F(P20,0x5a8),r9,r14);
    r5 = mac(r14,r9,r5);
    r14 = inversesqrt(r35);
    r15 = mul(r14,r33);
    r15 = mul(-1.0,r15);
    r13 = mac(F(P20,0x52c),r9,r7);
    r7 = mul(r14,r12);
    r12 = mul(F(P20,0x580),r34);
    r33 = mac(F(P20,0x584),r10,r12);
    r33 = mac(F(P20,0x588),r9,r33);
    r12 = mul(F(P20,0x560),r34);
    r6 = mac(F(P20,0x5ac),r9,r5);
    r5 = mul(r14,r11);
    vec3 cubeDirection = vec3(r15,r7,r5);
    r14 = mul(F(P20,0x500),r34);
    r16 = mac(F(P20,0x504),r10,r14);
    r14 = min(r2,r3);
    r2 = mul(r2,r3);
    r11 = mul(F(P20,0x540),r34);
    r2 = has(viewFlags,2u)?r2:r14;
    r35 = mac(F(P20,0x544),r10,r11);
    r16 = mac(F(P20,0x508),r9,r16);
    r2 = add(-1.0,r2);
    r35 = mac(F(P20,0x548),r9,r35);
    r11 = mul(F(P20,0x550),r34);
    r3 = F(P20,0x58c);
    r3 = add(F(P20,0x5b0),r3);
    r11 = mac(F(P20,0x554),r10,r11);
    r3 = add(r3,r33);
    r11 = mac(F(P20,0x558),r9,r11);
    r12 = mac(F(P20,0x564),r10,r12);
    r6 = mac(r3,r34,r6);
    r12 = mac(F(P20,0x568),r9,r12);
    r3 = max(B(0x3d800000u),r6);
    r33 = F(P20,0x54c);
    r15 = F(P20,0x50c);
    r15 = add(F(P20,0x530),r15);
    r15 = add(r15,r16);
    r13 = mac(r15,r34,r13);
    r36 = max(B(0x3d800000u),r13);
    r13 = F(P20,0x57c);
    r13 = mac(F(P20,0x574),r10,r13);
    r13 = mac(F(P20,0x578),r9,r13);
    r13 = mac(r11,r10,r13);
    r13 = mac(F(P20,0x55c),r10,r13);
    r13 = mac(r12,r9,r13);
    r37 = mac(F(P20,0x56c),r9,r13);
    sampleValue = texture(environment,cubeDirection); r13 = sampleValue.x; r14 = sampleValue.y; r15 = sampleValue.z; r16 = sampleValue.w;
    r7 = add(F(P20,0x570),r33);
    sampleValue = at(1,a2.xy); r9 = sampleValue.x; r10 = sampleValue.y; r11 = sampleValue.z; r12 = sampleValue.w;
    r5 = mac(F(P90,0x130),r2,1.0);
    r2 = add(r7,r35);
    r6 = mac(F(P90,0x130),r30,1.0);
    r37 = mac(r2,r34,r37);
    r2 = F(View,0x34);
    r35 = add(F(View,0x30),r2);
    r2 = max(0.0,-r19);
    r2 = min(1.0,r2);
    r7 = F(View,0x7c);
    r2 = add(-1.0,r2);
    r7 = mac(F(View,0x70),r24,r7);
    r38 = mac(r23,F(View,0x74),r7);
    r7 = mac(F(P90,0x130),r2,1.0);
    r30 = F(P30,0xe0);
    r38 = mac(F(View,0x78),r22,r38);
    r37 = max(B(0x3d800000u),r37);
    r39 = mac(F(View,0x3c),r38,r39);
    r39 = mac(0.5,r35,r39);
    r40 = F(P30,0xe4);
    r34 = mac(F(P20,0x4f0),r36,r30);
    r36 = mac(F(P20,0x4f0),r37,r40);
    r37 = F(P30,0xe8);
    r40 = sub(F(P30,0x64),r40);
    r33 = mul(F(P90,0x60),r29);
    r29 = mul(F(View,0x8),r31);
    r29 = mac(F(View,0xc),r38,r29);
    r38 = mac(0.5,r8,0.5);
    r31 = mac(F(P20,0x4f0),r3,r37);
    r37 = sub(F(P30,0x68),r37);
    r3 = 1.0/r39;
    r31 = mac(r38,r37,r31);
    r35 = F(P0,0x44);
    r35 = sub(F(P0,0x48),r35);
    r35 = 1.0/r35;
    r19 = mul(-r19,r35);
    r19 = sat(mac(-F(P0,0x44),r35,r19));
    r19 = add(-1.0,r19);
    r2 = mul(F(P30,0x10),r6);
    r37 = sub(F(P30,0x60),r30);
    r30 = F(View,0x4);
    r30 = add(F(View,0x0),r30);
    r29 = mac(0.5,r30,r29);
    r30 = mul(F(P30,0x0),r7);
    r3 = mul(r3,r29);
    r29 = B(0xc1200001u);
    r32 = mul(F(P90,0x64),r32);
    r3 = sat(mac(r29,-r3,B(0xc0600001u)));
    r29 = mul(F(P30,0x8),r7);
    r31 = mac(r29,r5,r31);
    r34 = mac(r30,r5,r34);
    r36 = mac(r38,r40,r36);
    r34 = mac(F(P90,0x134),r2,r34);
    r37 = mac(r37,r8,r37);
    r48 = mac(0.5,r37,r34);
    r35 = mac(F(P0,0x40),r19,1.0);
    r19 = F(P60,0x18);
    r41 = mul(F(P70,0xa8),r19);
    r19 = F(P60,0x14);
    r40 = mul(F(P70,0xa4),r19);
    sampleValue = at(5,rasterUV); r17 = sampleValue.x; r18 = sampleValue.y; r19 = sampleValue.z;
    r34 = F(P60,0x8);
    r45 = mul(r28,r3);
    r3 = mul(F(P30,0x14),r6);
    r28 = mul(F(P30,0x18),r6);
    r44 = mac(F(P90,0x134),r28,r31);
    r31 = mul(F(P30,0x4),r7);
    r36 = mac(r31,r5,r36);
    r5 = F(P60,0x4);
    r46 = 1.0/F(P10,0x8);
    r47 = mac(F(P90,0x134),r3,r36);
    r36 = F(P60,0x10);
    r49 = mul(F(P70,0xa0),r36);
    r43 = mul(F(P70,0x94),r5);
    sampleValue = at(4,a2.xy); r5 = sampleValue.x; r6 = sampleValue.y; r7 = sampleValue.z; r8 = sampleValue.w;
    r20 = F(P60,0x0);
    r42 = mul(F(P70,0x98),r34);
    r34 = F(P60,0x11c);
    r39 = mul(F(P70,0x90),r20);
    r20 = a0.x;
    r36 = mul(F(P10,0x0),r20);
    r20 = a0.y;
    r37 = mul(F(P10,0x0),r20);
    r20 = a0.z;
    r38 = mul(F(P10,0x0),r20);
    r21 = B(0x3e19999au);
    r20 = B(0x3fb0852eu);
    r11 = mul(F(P70,0xac),r11);
    r11 = max(1.0,r11);
    r33 = mul(r33,r11);
    r32 = mul(r32,r11);
    r11 = mul(F(P90,0x6c),r12);
    r12 = max(B(0x3b808081u),r16);
    r12 = 1.0/r12;
    r12 = mul(B(0x3e800000u),r12);
    r32 = exp2(r32);
    r14 = mul(r14,r12);
    r15 = mul(r15,r12);
    r50 = mul(r11,r14);
    r16 = mul(r13,r12);
    r12 = exp2(r33);
    r12 = mul(F(P90,0x4c),r12);
    r12 = mac(F(P90,0x5c),r32,r12);
    r32 = mul(r40,r12);
    r13 = mul(r41,r12);
    r41 = mul(r11,r15);
    r33 = sat(mul(F(P30,0x4),r32));
    r32 = mul(r9,r45);
    r13 = sat(mul(F(P30,0x8),r13));
    r40 = mul(r32,r33);
    r32 = mul(r32,r13);
    r33 = mul(r11,r16);
    r11 = mul(r49,r12);
    r11 = sat(mul(F(P30,0x0),r11));
    r45 = mul(r45,r11);
    r26 = mac(r46,r17,r26);
    r51 = mac(r46,r18,r51);
    r27 = mac(r46,r19,r27);
    r19 = mac(r48,r35,r26);
    r18 = mac(r47,r35,r51);
    r17 = mac(r44,r35,r27);
    r25 = mul(r10,r18);
    r18 = mul(r10,r19);
    r17 = mul(r10,r17);
    r10 = max(r17,max(r18,r25));
    r19 = sat(add(F(P90,0x68),r10));
    r10 = mul(r45,r19);
    r6 = mul(r6,r6);
    r7 = mul(r7,r7);
    r6 = mac(r14,r50,r6);
    r7 = mac(r15,r41,r7);
    r6 = mul(r43,r6);
    r7 = mul(r42,r7);
    r6 = mul(r6,r25);
    r11 = mul(r7,r17);
    r6 = mac(r19,r40,r6);
    r11 = mac(r19,r32,r11);
    r12 = mul(r5,r5);
    r5 = mac(F(P60,0x114),r34,r6);
    r5 = mac(-F(P60,0x11c),r6,r5);
    r12 = mac(r16,r33,r12);
    r6 = mul(r39,r12);
    r7 = mul(r6,r18);
    r7 = mac(r10,r9,r7);
    r9 = mac(F(P60,0x118),r34,r11);
    r9 = mac(-F(P60,0x11c),r11,r9);
    r6 = mul(mul(r37,r5),2.0);
    r5 = mul(mul(r38,r9),2.0);
    r9 = mac(B(0x3e19999au),r5,0.5);
    r9 = mac(r5,r9,B(0x3d75c290u));
    r9 = 1.0/r9;
    r14 = mac(r21,r5,B(0x3d4ccccdu));
    r5 = mac(r14,r5,B(0x3b83126fu));
    r14 = mac(r21,r6,B(0x3d4ccccdu));
    r5 = mul(r9,r5);
    r9 = mac(F(P60,0x110),r34,r7);
    r9 = mac(-F(P60,0x11c),r7,r9);
    r7 = mul(mul(r36,r9),2.0);
    r9 = mac(B(0x3e19999au),r6,0.5);
    r9 = mac(r6,r9,B(0x3d75c290u));
    r9 = 1.0/r9;
    r6 = mac(r14,r6,B(0x3b83126fu));
    r14 = mac(r21,r7,B(0x3d4ccccdu));
    r6 = mul(r9,r6);
    r9 = mac(B(0x3e19999au),r7,0.5);
    r9 = mac(r7,r9,B(0x3d75c290u));
    r9 = 1.0/r9;
    r7 = mac(r14,r7,B(0x3b83126fu));
    r7 = mul(r9,r7);
    r7 = sat(mac(r20,r7,B(0xbdbc49cau)));
    r6 = sat(mac(r20,r6,B(0xbdbc49cau)));
    r5 = sat(mac(r20,r5,B(0xbdbc49cau)));
    r7 = sqrt(r7);
    r6 = sqrt(r6);
    r5 = sqrt(r5);
    vec3 toned=vec3(r7,r6,r5);
    if (has(U(P0,0x0c),1u)) {
        float g=add(r51,r31), b=add(r27,r29), r=add(r26,r30);
        g=add(g,r3); b=add(b,r28); r=add(r,r2);
        float brightness=mac(add(g,b),B(0x3eaaaaabu),mul(B(0x3eaaaaabu),r));
        float om=sub(1.0,brightness);
        float low=sat(mul(om,om));
        float mid=mac(-om,om,1.0); mid=sat(mac(brightness,-brightness,mid));
        float high=sat(mul(brightness,brightness));
        float black=F(P0,0x1c), invRange=1.0/sub(sub(1.0,black),black);
        for (int i=0;i<3;++i) {
            vec3 c=F3(P0,0x10+16*i);
            float e=mul(c.x,low); e=mac(c.y,mid,e); e=mac(c.z,high,e);
            e=add(F(P0,4*i),e);
            toned[i]=sat(POW(sat(mac(invRange,toned[i],-black)),e));
        }
    }
    vec3 color=toned;
    if (has(flags,4u)) color=add(toned,a1.rgb);
    if (has(flags,0x400000u) && !has(flags,4u)) {
        vec3 delta=sub(F3(View,0x220),position);
        float f=sat(mac(F(P10,0x28),sqrt(D3(delta,delta)),F(P10,0x2c)));
        color=mac(F3(P10,0x10),f,toned);
        color=mac(f,-toned,color);
    }
    float specAlpha=sat(max(0.0,r10));
    float materialAlpha=add(r8,specAlpha);
    materialAlpha=mac(specAlpha,-r8,materialAlpha);
    float alpha=mul(F(P70,0x9c),materialAlpha);
    alpha=mul(F(P50,0x1c),alpha); alpha=mul(F(P60,0x0c),alpha);
    if (!has(flags,8u)) {
        color=mul(F3(P10,0x30),color); alpha=mul(F(P10,0x3c),alpha);
    }
    float alpha1=alpha;
    if (has(flags,0x800000u)) {
        vec4 b=F4(View,0x230);
        color=mac(b.w,color,b.rgb); color=mac(-b.rgb,b.w,color);
        alpha1=mul(b.w,alpha);
    }
    float alpha0=has(viewFlags,1u)?sat(mul(B(0x3a4d9a68u),gl_FragCoord.z)):alpha;
    uvec4 so=cb.c[455];
    o0=swz(half4(vec4(color,alpha0)),so.x);
    o1=swz(half4(vec4(0.0,0.0,0.0,alpha1)),so.y);
}
#else
void main() {
    vec3 position = vec3(a3.w,a4.w,a0.w);
    vec3 unfolded = position;
    float deformationCount = F(P60,0xdc);
    for (int i = 0; deformationCount > float(i); ++i) {
        vec4 plane = F4(P60,0xd0+16*i);
        vec4 origin = F4(P60,0x50+16*i);
        float side = planeSide(plane,unfolded,origin.xyz);
        // Keep rejected lanes as derivative helpers through subsequent samples.
        if (abs(side) < origin.w) demote;
        vec4 shift = F4(P60,0x90+16*i);
        float selected = side >= 0.0 ? 1.0 : 0.0;
        unfolded = mac(selected,-mul(shift.xyz,shift.w),unfolded);
    }

    uint flags = U(Flags,0), viewFlags = U(View,0x22c);
    vec2 uv = a2.xy;
    vec2 rasterToUV = F2(View,0x218);
    vec2 rasterUV = mul(rasterToUV,gl_FragCoord.xy);
    vec3 gn = a4.xyz, tangent = a5.xyz;
    vec3 bt;
    bt.x = mul(gn.y,tangent.z);
    bt.x = mac(tangent.y,-gn.z,bt.x);
    bt.y = mul(gn.z,tangent.x);
    bt.y = mac(tangent.z,-gn.x,bt.y);
    bt.z = mul(gn.x,tangent.y);
    bt.z = mac(tangent.x,-gn.y,bt.z);
    bt = mul(a5.w,bt);

    vec2 encodedXY = at(0,uv).rg;        // 1c8 / 150
#ifdef REF
    vec4 specTex = at(1,uv);
    vec3 specMaterial = specTex.rgb;
#else
    vec3 specMaterial = at(1,uv).rgb;    // 1d0 / 158
#endif
    vec2 nxy = mac(2.0,encodedXY,vec2(-1.0));
    float xy2 = mul(nxy.x,nxy.x);
    xy2 = mac(nxy.y,nxy.y,xy2);
    float nz = sqrt(sat(sub(1.0,xy2)));
    vec3 un = mul(tangent,nxy.x);
    un = mac(bt,nxy.y,un);
    un = mac(gn,nz,un);
    vec3 N = NORM(un), V = NORM(a3.xyz);
    float exponent = max(1.0,mul(F(P70,0xac),specMaterial.b));
    float NV = D3(N,V);
    vec3 reflectedV = mac(-N,mul(2.0,NV),V);
    float normalization = mac(B(0x3d22ca2fu),exponent,B(0x3daf7aacu));
    float fresnelFloor = mul(F(P10,0x0c),F(P70,0x88));
    float fresnelRest = sub(1.0,fresnelFloor);

    float shadowR = 1.0, shadowG = 1.0;
    if (has(flags,8u)) {
        vec2 shadow = at(2,rasterUV).rg; // 154 / 2b0
        vec4 remap = F4(P10,0xf0);
        shadowR = mac(remap.x,shadow.r,remap.y);
        shadowG = mac(shadow.g,remap.z,remap.w);
    }
    float combinedShadow = has(viewFlags,2u) ? mul(shadowR,shadowG) : min(shadowR,shadowG);

    float extraSpecular = 0.0;
#ifdef SKIN
    vec4 extra = F4(P90,0x70);
    if (extra.x > 0.0) {
        vec3 d;
        d.x = mac(-extra.z,V.z,V.x);
        d.y = add(extra.w,V.y);
        d.z = mac(extra.z,V.x,V.z);
        float len2 = D3(d,d), nd = D3(d,N);
        float lobe = sat(mul(-inversesqrt(len2),nd));
        lobe = log2(lobe);
        lobe = mul(extra.y,lobe);
        lobe = mul(lobe,exponent);
        lobe = exp2(lobe);
        extraSpecular = mul(extra.x,lobe);
    }
#endif

    vec3 pointDiffuse = vec3(0.0), pointSpecular = vec3(0.0);
    float pointOcclusionLight = 0.0;
    if (has(flags,1u)) {
        vec3 colors[4], L[4], H[4];
        float lambert[4], occ[4];
        for (int i = 0; i < 4; ++i) {
            colors[i] = vec3(0.0);
            L[i] = vec3(0.0);
            H[i] = vec3(0.0);
            lambert[i] = 0.0;
            occ[i] = 0.0;
            if (has(flags,0x10000u<<i)) {
                vec4 pp = F4(P30,0xa0+16*i);
                vec4 pc = F4(P30,0x20+16*i);
                vec3 delta = sub(position,pp.xyz);
                float distance2 = D3(delta,delta);
                float outside = max(0.0,sub(sqrt(distance2),pc.w));
                float falloff = mul(pp.w,outside);
                float den = mac(falloff,outside,1.0);
                float attenuation = sat(mac(B(0x3f808102u),1.0/den,B(0xbb810205u)));
                float invDistance = inversesqrt(distance2);
                L[i] = mul(delta,invDistance);
                lambert[i] = min(1.0,max(0.0,-D3(N,L[i])));
                colors[i] = mul(pc.rgb,attenuation);
                vec3 diffuse = max(vec3(0.0),mul(lambert[i],colors[i]));
                pointDiffuse = add(pointDiffuse,diffuse);
                occ[i] = mul(lambert[i],luminance(diffuse));
                H[i] = NORM(mac(delta,invDistance,V));
            }
        }
        for (int i = 0; i < 4; ++i) {
            float s;
#ifdef SKIN
            float lobe = POW(min(1.0,max(0.0,-D3(reflectedV,L[i]))),exponent);
            lobe = mul(lobe,normalization);
            float fa = sub(1.0,D3(H[i],V));
            float fresnel = mac(P5(fa),fresnelRest,fresnelFloor);
            s = mul(lobe,fresnel);
            s = mul(s,lambert[i]);
#else
            s = POW(min(1.0,max(0.0,-D3(H[i],N))),exponent);
#endif
            if (i == 0) pointSpecular = mul(colors[i],s);
            else pointSpecular = mac(colors[i],s,pointSpecular);
        }
        pointSpecular = max(vec3(0.0),pointSpecular);
        pointOcclusionLight = add(occ[0],occ[1]);
        pointOcclusionLight = add(pointOcclusionLight,occ[2]);
        pointOcclusionLight = add(pointOcclusionLight,occ[3]);
    }

    vec3 ambientN = NORM(N);
    vec3 poly = vec3(polynomial(0,ambientN),polynomial(1,ambientN),polynomial(2,ambientN));
    vec3 lower = F3(P30,0xe0), upper = F3(P30,0x60);
    vec3 ambient = mac(F(P20,0x4f0),poly,lower);
    vec3 difference = sub(upper,lower);
    float hemisphere = mac(0.5,N.y,0.5);
#if defined(SKIN) || defined(REF)
    float red = difference.r;
    red = mac(difference.r,N.y,red);
#ifndef REF
    ambient.r = mac(0.5,red,ambient.r);
#endif
    ambient.gb = mac(hemisphere,difference.gb,ambient.gb);
#else
    ambient = mac(hemisphere,difference,ambient);
#endif

    vec3 direction0 = F3(P30,0x80), direction1 = F3(P30,0x90);
    vec3 light0 = F3(P30,0), light1 = F3(P30,0x10);
    float secondaryGain = F(P90,0x134);
    float rawNL0 = D3(direction0,N), rawNL1 = D3(direction1,N);
    float nl0 = min(1.0,max(0.0,-rawNL0));
    float nl1 = min(1.0,max(0.0,-rawNL1));
#ifdef REF
    vec4 control = vec4(1.0,1.0,0.0,1.0);
#else
    vec4 control = at(3,uv);              // c3c / a04
#endif
    vec4 base = at(4,uv);                 // d6c / a30
#ifdef ALPHA_TEST
    if (F(P70,0x80) > base.a) demote; // sm_p1dzt 0x11c0: unscaled base alpha under the reference
#endif
    // The image at this PC supplies its own entry s[4:7] sampler.
    vec4 screen = at(5,rasterUV);         // da0 / 90c
    float invScreen = 1.0/F(P10,8);
    vec3 indirectDiffuse = mac(screen.rgb,invScreen,pointDiffuse);
    vec3 indirectSpecular = mac(invScreen,vec3(screen.a),pointSpecular);
#ifdef REF
    float materialShadow = combinedShadow;
#else
    float materialShadow = mac(control.r,sub(combinedShadow,1.0),1.0);
#endif
    vec3 primary = mul(light0,nl0);
    primary = mul(materialShadow,primary);
    vec3 secondary = mul(light1,nl1);
    vec3 directionalDiffuse = mac(secondaryGain,secondary,primary);
#ifdef REF
    vec3 diffuseLighting = mac(mul(light0,nl0),materialShadow,ambient);
    diffuseLighting = mac(secondaryGain,secondary,diffuseLighting);
    diffuseLighting.r = mac(0.5,red,diffuseLighting.r);
#else
    vec3 diffuseLighting = add(ambient,directionalDiffuse);
#endif

#ifdef SKIN
    float rampY = mac(sub(F(P70,0x78),F(P70,0x7c)),a2.z,F(P70,0x7c));
    vec3 ramp0 = at(6,vec2(mac(-0.5,rawNL0,0.5),rampY)).rgb; // cbc
    vec3 ramp1 = at(7,vec2(mac(-0.5,rawNL1,0.5),rampY)).rgb; // cc4
    float rb = mul(control.r,sub(shadowR,1.0));
    float modified = mac(-rawNL0,rb,-rawNL0);
    float terminator = mul(F(P90,0x2c),-rawNL0);
    float shadowX = mac(0.5,min(terminator,modified),0.5);
    vec3 shadowRamp = at(8,vec2(shadowX,rampY)).rgb;          // e50
    vec3 adjusted = mac(ramp0,materialShadow,-shadowRamp);
    adjusted = mac(nl0,adjusted,shadowRamp);
    vec3 rampDiffuse = mul(light0,adjusted);
    rampDiffuse = mac(light1,ramp1,rampDiffuse);
    vec3 correction = sub(rampDiffuse,directionalDiffuse);
    float gb = mul(control.r,sub(shadowG,1.0));
    float gain = mac(F(P90,0x24),gb,F(P90,0x24));
    diffuseLighting = mac(correction,gain,diffuseLighting);
#endif
    float invResponse = 1.0/sub(F(P0,0x48),F(P0,0x44));
    float response = mul(-rawNL0,invResponse);
    response = sat(mac(-F(P0,0x44),invResponse,response));
    float directionalResponse = mac(F(P0,0x40),sub(response,1.0),1.0);
    diffuseLighting = mac(diffuseLighting,directionalResponse,indirectDiffuse);

    float ds0 = directionalSpec(direction0,nl0,reflectedV,V,exponent,
                                normalization,fresnelFloor,fresnelRest);
    float ds1 = directionalSpec(direction1,nl1,reflectedV,V,exponent,
                                normalization,fresnelFloor,fresnelRest);
    vec3 directionalSpecular = mul(light0,materialShadow);
    directionalSpecular = mac(directionalSpecular,ds0,vec3(extraSpecular));
    directionalSpecular = mac(light1,ds1,directionalSpecular);
    vec3 specMultiplier = mul(F3(P60,0x10),F3(P70,0xa0));
    vec3 specularLighting = mul(specMultiplier,directionalSpecular);
#ifdef SKIN
    float sf = sat(sub(specMaterial.g,a2.z));
    float sw = mac(materialShadow,nl0,sf);
    float shadowedNL = mul(materialShadow,nl0);
    sw = mac(shadowedNL,-sf,sw);
    specularLighting = mac(specularLighting,sw,indirectSpecular);
#else
    specularLighting = add(specularLighting,indirectSpecular);
#endif

    float geometricFacing = sat(add(1.0,D3(gn,V)));
    float normalFacing = sat(add(1.0,NV));
    float facingProduct = mul(geometricFacing,normalFacing);
    float effectExponent = add(2.0,F(P70,0x6c));
    float effect = log2(facingProduct);
    effect = mul(effect,effectExponent);
    effect = exp2(effect);
    effect = mul(specMaterial.g,effect);
    effect = mul(F(P30,0x10c),effect);
#ifdef SKIN
    effect = mul(effect,add(2.0,F(P70,0x68)));
#else
    effect = mul(effect,add(B(0x40200000u),F(P70,0x68)));
    effect = mul(B(0x3dcccccdu),effect);
#endif
    vec3 baseSquared = mul(base.rgb,base.rgb);
#ifdef REF
    vec3 cubeDirection = NORM(reflectedV);
    vec4 env = texture(environment,vec3(-cubeDirection.x,cubeDirection.y,cubeDirection.z)); // 0x89c-0x918
    float envScale = mul(B(0x3e800000u),1.0/max(B(0x3b808081u),env.a));
    vec3 envColor = mul(env.rgb,envScale);
    baseSquared = mac(specTex.a,mac(envColor,envColor,-baseSquared),baseSquared);
#endif
    vec3 effectColor = mul(F3(P30,0x100),control.g);
    baseSquared = mac(effect,effectColor,baseSquared);
#ifndef REF

    float primaryLuminance = luminance(light0);
    float occlusionLight = max(0.0,sub(F(P30,0x124),rawNL0));
    occlusionLight = mul(F(P30,0x128),occlusionLight);
    occlusionLight = mac(pointOcclusionLight,1.0/primaryLuminance,occlusionLight);
    float remainingShadow = mac(-F(P30,0x12c),materialShadow,F(P30,0x12c));
    float upperLimit = sub(1.0,remainingShadow);
    float suppression = max(0.0,min(upperLimit,occlusionLight));
    float materialOcclusion = sub(1.0,control.a);
    materialOcclusion = mac(suppression,-materialOcclusion,materialOcclusion);
    float occlusion = sat(mac(-F(P30,0x120),materialOcclusion,1.0));
#endif

    vec3 diffuseMultiplier = mul(F3(P70,0x90),specMaterial.g);
    diffuseMultiplier = mul(F3(P60,0),diffuseMultiplier);
    vec3 diffuseMaterial = mul(diffuseMultiplier,baseSquared);
    vec3 linearColor = mul(diffuseMaterial,diffuseLighting);
    linearColor = mac(specularLighting,specMaterial.r,linearColor);
#ifndef REF
    linearColor = mul(occlusion,linearColor);
#endif
    vec3 oldColor = linearColor;
    float linearWeight = F(P60,0x11c);
    linearColor = mac(F3(P60,0x110),linearWeight,oldColor);
    linearColor = mac(-linearWeight,oldColor,linearColor);
    vec3 toned = vec3(tone(linearColor.r,a0.r),
                      tone(linearColor.g,a0.g),
                      tone(linearColor.b,a0.b));

    uint maskFlags = U(P0,0x0c);
    if (has(maskFlags,1u)) {
        vec3 brightnessRGB = add(indirectDiffuse,indirectSpecular);
        brightnessRGB = add(brightnessRGB,mul(light0,nl0));
        brightnessRGB = add(brightnessRGB,mul(light1,nl1));
        float gb = add(brightnessRGB.g,brightnessRGB.b);
        float third = mul(B(0x3eaaaaabu),brightnessRGB.r);
        float brightness = mac(gb,B(0x3eaaaaabu),third);
        float oneMinus = sub(1.0,brightness);
        float lowWeight = sat(mul(oneMinus,oneMinus));
        float middleWeight = mac(-oneMinus,oneMinus,1.0);
        middleWeight = sat(mac(brightness,-brightness,middleWeight));
        float highWeight = sat(mul(brightness,brightness));
        vec3 baseExponent = F3(P0,0);
        vec4 red = F4(P0,0x10), green = F4(P0,0x20), blue = F4(P0,0x30);
        float black = red.w;
        float invRange = sub(1.0,black);
        invRange = sub(invRange,black);
        invRange = 1.0/invRange;
        vec3 gradeExponent;
        for (int c = 0; c < 3; ++c) {
            vec3 controls = c == 0 ? red.xyz : c == 1 ? green.xyz : blue.xyz;
            float e = mul(controls.x,lowWeight);
            e = mac(controls.y,middleWeight,e);
            e = mac(controls.z,highWeight,e);
            gradeExponent[c] = add(baseExponent[c],e);
        }
        vec3 adjusted = sat(mac(invRange,toned,vec3(-black)));
        toned = sat(exp2(mul(log2(adjusted),gradeExponent)));
    }

    vec3 color = toned;
    float fogAmount = 0.0;
    if (has(flags,4u)) color = add(toned,a1.rgb);
    if (has(flags,0x400000u) && !has(flags,4u)) {
        vec3 delta = sub(F3(View,0x220),position);
        float distance = sqrt(D3(delta,delta));
        fogAmount = sat(mac(F(P10,0x28),distance,F(P10,0x2c)));
        color = mac(F3(P10,0x10),fogAmount,toned);
        color = mac(fogAmount,-toned,color);
    }
    float alpha = mul(F(P70,0x9c),F(P50,0x1c));
    alpha = mul(F(P60,0x0c),alpha);
    alpha = mul(alpha,base.a);
    if (!has(flags,8u)) {
        color = mul(F3(P10,0x30),color);
        alpha = mul(F(P10,0x3c),alpha);
    }

    float mask = 0.0;
    if (!has(flags,0x2000000u)) {
        vec2 maskScale = F2(View,0x218);
        vec2 maskUV = mul(maskScale,gl_FragCoord.xy);
        float depthAdd = F(P0,0x2c), depthNumerator = F(P0,0x3c);
        vec4 dc = F4(P0,0x284);
        float start = dc.x, near_ = dc.y, end_ = dc.z, minRadius = dc.w;
        float depthSample = at(9,maskUV).r;                    // 1560 / 1270
        float invDepth = 1.0/add(depthAdd,depthSample);
        float depthMinusStart = mac(depthNumerator,invDepth,-start);
        float negativeDepth = mul(-depthNumerator,invDepth);
        float radiusFade = sat(mac(-depthMinusStart,1.0/sub(end_,start),1.0));
        float nearInverse = 1.0/sub(start,near_);
        float nearRamp = mul(-negativeDepth,nearInverse);
        nearRamp = sat(mac(-near_,nearInverse,nearRamp));

        // Each image carries its own direct entry s[0:3] sampler.
        vec3 encoded = at(12,maskUV).rgb;                     // 1568 / 1278
        vec3 screenNormal = mac(2.0,encoded,vec3(-1.0));
        vec2 screenDirection = mul(maskScale,vec2(screenNormal.x,-screenNormal.y));
        float normalRadius = mul(F(P0,0x26c),radiusFade);
        normalRadius = max(minRadius,normalRadius);
        normalRadius = mul(nearRamp,normalRadius);
        vec3 np = at(13,mac(screenDirection,normalRadius,maskUV)).rgb;  // 1600 / 1310
        vec3 nm = at(14,mac(-screenDirection,normalRadius,maskUV)).rgb; // 1618 / 1328
        float rp = normalResponse(np,screenNormal);
        float rm = normalResponse(nm,screenNormal);
        float normalEdge = 0.0, depthEdge = 0.0, facingMask = 0.0;
        if (has(maskFlags,4u)) {
            vec2 k = F2(P0,0x270);
            float agreement = mac(0.5,rm,0.5);
            agreement = mac(rp,agreement,agreement);
            agreement = mul(0.5,agreement);
            float tested = mac(agreement,agreement,B(0x3b808081u));
            float threshold = mac(-k.y,nearRamp,1.0);
            float passed = tested >= threshold ? 1.0 : 0.0;
            float gate = sat(mac(agreement,agreement,passed));
            float suppress = mac(sub(1.0,k.x),gate,k.x);
            normalEdge = mac(-nearRamp,suppress,nearRamp);
#ifndef REF
            normalEdge = mul(mac(-control.b,F(P0,0x54),1.0),normalEdge);
#endif
        }
        if (has(maskFlags,8u)) {
            vec2 k = F2(P0,0x260);
            float radius = mul(k.x,radiusFade);
            radius = max(minRadius,radius);
            radius = mul(nearRamp,radius);
            float zp = at(10,mac(screenDirection,radius,maskUV)).r;  // 1728 / 1438
            float zm = at(11,mac(-screenDirection,radius,maskUV)).r; // 1740 / 1450
            float r = mul(2.0,negativeDepth);
            r = mac(depthNumerator,1.0/add(depthAdd,zp),r);
            r = mac(depthNumerator,1.0/add(depthAdd,zm),r);
            r = mac(-0.5,r,1.0);
            float threshold = mac(-F(P0,0x268),nearRamp,1.0);
            float passed = add(r,B(0x3b808081u)) >= threshold ? 1.0 : 0.0;
            float gate = sat(add(passed,r));
            float suppress = mac(sub(1.0,k.y),gate,k.y);
            depthEdge = mac(-nearRamp,suppress,nearRamp);
        }
        if (has(maskFlags,2u)) {
            vec4 df = F4(P0,0x2a0);
            vec2 fc = F2(P0,0x278);
            float facing = min(1.0,max(0.0,-D3(N,V)));
            float farDistance = df.x;
            float nearDistance = min(df.y,add(farDistance,B(0xba83126fu)));
            float invRange = 1.0/sub(farDistance,nearDistance);
            float fd = mul(-negativeDepth,invRange);
            fd = sat(mac(nearDistance,-invRange,fd));
            float width = mac(fd,sub(fc.y,df.z),df.z);
#ifdef REF
            float widthRamp = nearRamp;
#else
            float widthRamp = mac(-control.b,F(P0,0x58),1.0);
            widthRamp = mul(widthRamp,nearRamp);
#endif
            float den = mac(widthRamp,width,B(0x3b808081u));
            float normalizedFacing = sat(mul(1.0/den,add(facing,B(0x3b808081u))));
            float gain = mul(sub(1.0,fc.x),normalizedFacing);
            float suppress = sat(mac(gain,normalizedFacing,fc.x));
            float threshold = mac(sub(F(P0,0x280),df.w),fd,df.w);
            float r = mac(-rp,rm,1.0);
            float candidate = mac(suppress,-nearRamp,nearRamp);
            float gated = mul(r >= threshold ? 1.0 : 0.0,candidate);
            facingMask = threshold > 0.0 ? gated : candidate;
        }
        mask = max(depthEdge,max(facingMask,normalEdge));
        vec2 fade = F2(P0,0x294);
        float invFade = 1.0/sub(fade.x,start);
        float distanceFade = sat(mac(-depthMinusStart,invFade,1.0));
        distanceFade = max(sub(1.0,fade.y),distanceFade);
        mask = mul(distanceFade,mask);
        mask = mac(mask,-fogAmount,mask);
        mask = mul(F(P0,0x29c) <= mask ? 1.0 : 0.0,mask);
    }
    mask = mul(sub(1.0,F(P60,0x11c)),mask);

    float alpha1 = alpha;
    if (has(flags,0x800000u)) {
        vec4 blend = F4(View,0x230);
        color = mac(blend.w,color,blend.rgb);
        color = mac(-blend.rgb,blend.w,color);
        mask = mul(blend.w,mask);
        alpha1 = mul(blend.w,alpha);
    }
    float rasterAlpha = sat(mul(B(0x3a4d9a68u),gl_FragCoord.z));
    float alpha0 = has(U(View,0x22c),1u) ? rasterAlpha : alpha;
    uvec4 so = cb.c[455];
    o0 = swz(half4(vec4(color,alpha0)),so.x);
    o1 = swz(half4(vec4(mask,0.0,0.0,alpha1)),so.y);
}
#endif // HAIR
)";

// Only the new role renames main. The nine established shader compositions
// still use exactly kHeader + their original defines + kVs / kFs.
constexpr const char* kO1dztVsSuffix = R"(
#undef main
void main() {
    characterMain();
    if (cb.c[455].w != 0u) p0.xyz = mul(F(V10,0x3c),p0.xyz);
}
)";

// PS 0x53c661700. Keep the translator's scalar instruction expressions and
// branch/sample order. In particular, these operations may contract, and
// opcode 303 uses packHalf2x16 (nearest), rather than the older halfRTZ helper.
// Identity float/uint round trips are removed from VGPRs. Keep SGPR bits and
// packed exports as uints. Descriptor loads become native bindings; buffer
// loads retain their PCs above.
constexpr const char* kO1dztFs = R"(
layout(set=0,binding=8) uniform sampler2D tex[15];
layout(location=0) in vec4 a0;
layout(location=1) in vec4 a1;
layout(location=2) in vec4 a2;
layout(location=3) in vec4 a3;
layout(location=4) in vec4 a4;
layout(location=5) in vec4 a5;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
float F(uint x) { return uintBitsToFloat(x); }
vec2 sc(int t) { vec4 s=cf(441+t/2); return (t&1)!=0 ? s.zw : s.xy; }
vec4 swz(vec4 v,uint s) {
    return vec4(v[s&3u],v[(s>>2)&3u],v[(s>>4)&3u],v[(s>>6)&3u]);
}
void main() {
    uint e=0u, eh=0u, m0=0u, s0=0u, s1=0u, s2=0u, s3=0u, s4=0u; uint s5=0u, s6=0u, s7=0u, s8=0u, s9=0u, s10=0u, s14=0u, s15=0u;
    uint s16=0u, s17=0u, s18=0u, s19=0u, s20=0u, s21=0u, s22=0u, s23=0u; uint s24=0u, s25=0u, s26=0u, s27=0u, s28=0u, s29=0u, s30=0u, s31=0u;
    uint s32=0u, s33=0u, s34=0u, s35=0u, s36=0u, s37=0u, s38=0u, s39=0u; uint s40=0u, s41=0u, s42=0u, s43=0u, s44=0u, s45=0u, s46=0u, s47=0u;
    uint s48=0u, s49=0u, s50=0u, s51=0u, s52=0u, s53=0u, s54=0u, s55=0u; uint s56=0u, s57=0u, s58=0u, s59=0u, s60=0u, s61=0u, s62=0u, s63=0u;
    uint s64=0u, s65=0u, s66=0u, s67=0u, s68=0u, s69=0u, s70=0u, s71=0u; uint s72=0u, s73=0u, s74=0u, s75=0u, s76=0u, s77=0u, s78=0u, s79=0u;
    uint s84=0u, s85=0u, s86=0u, s87=0u, s88=0u, s89=0u, s90=0u, s91=0u; uint s92=0u, s93=0u, s94=0u, s95=0u, s100=0u, s101=0u, vc=0u, vh=0u;
    uint h0=0u,h1=0u,h2=0u,h3=0u; float v0=0.0, v1=0.0, v2=0.0, v3=0.0, v4=0.0, v5=0.0, v6=0.0, v7=0.0;
    float v8=0.0, v9=0.0, v10=0.0, v11=0.0, v12=0.0, v13=0.0, v14=0.0, v15=0.0;
    float v16=0.0, v17=0.0, v18=0.0, v19=0.0, v20=0.0, v21=0.0, v22=0.0, v23=0.0;
    float v24=0.0, v25=0.0, v26=0.0, v27=0.0, v28=0.0, v29=0.0, v30=0.0, v31=0.0;
    float v32=0.0, v33=0.0, v34=0.0, v35=0.0, v36=0.0, v37=0.0, v38=0.0, v39=0.0;
    float v40=0.0, v41=0.0, v42=0.0, v43=0.0, v44=0.0, v45=0.0, v46=0.0, v47=0.0;
    float v48=0.0, v49=0.0, v50=0.0, v51=0.0, v52=0.0, v53=0.0, v54=0.0, v55=0.0; float v56=0.0, v57=0.0, v58=0.0, v59=0.0; bool cc = false;
    uint pc = 0x0u; e = 1u; v2 = gl_FragCoord.x; v3 = gl_FragCoord.y; v4 = gl_FragCoord.w;
    {
        pc = 0x0u;
    }
    if (pc == 0x0u) {
        vh = 0x2cbu; m0 = s14;
        { uint lo_ = e; uint hi_ = eh; s14 = lo_; s15 = hi_; }
        { uint lo_ = e; uint hi_ = eh; e = lo_; eh = hi_; }
        s28 = U(P60,0xdc);
        if(e!=0u) {
            v21 = a0[3]; v23 = a3[3]; v22 = a4[3]; v5 = (v23); v6 = (v22); v7 = (v21);
        }
        s29 = 0u; pc = 0x48u;
    }
    for (uint guard0 = 0u; guard0 < 100000u; ++guard0) {
        if (pc == 0x48u) {
            if(e!=0u) {
                v8 = float(int(s29));
            }
            vc = ((e & 1u) != 0u && (F(s28) > v8)) ? 1u : 0u; vh = 0u; pc = ((vc & 1u) == 0u) ? 0x10cu : 0x58u;
        }
        if (pc == 0x58u) {
            s30 = s29 * 16u;
            { uint a_ = s30; uint b_ = 0xd0u; uint r_ = a_ + b_; cc = ((~(a_ ^ b_) & (a_ ^ r_)) >> 31) != 0u; s20 = r_; }
            { uint a_ = s30; uint b_ = 0x50u; uint r_ = a_ + b_; cc = ((~(a_ ^ b_) & (a_ ^ r_)) >> 31) != 0u; s24 = r_; }
            s20 = U(P60,0xd0); s21 = U(P60,0xd4); s22 = U(P60,0xd8); s23 = U(P60,0xdc); s24 = U(P60,0x50); s25 = U(P60,0x54); s26 = U(P60,0x58);
            s27 = U(P60,0x5c);
            if(e!=0u) {
                v8 = F(s20) * v5; v9 = F(s24); v8 = (-F(s20)) * v9 + v8; v9 = F(s25); v8 = F(s21) * v6 + v8; v8 = (-F(s21)) * v9 + v8; v9 = F(s26);
                v8 = F(s22) * v7 + v8; v8 = (-F(s22)) * v9 + v8;
            }
            vc = ((e & 1u) != 0u && (abs(v8) < F(s27))) ? 1u : 0u; vh = 0u;
            { uint lo_ = s14 & ~vc; uint hi_ = s15 & ~vh; s14 = lo_; s15 = hi_; }
            cc = (s14 | s15) != 0u; pc = (!cc) ? 0x15ccu : 0xc0u;
        }
        if (pc == 0xc0u) {
            vc = ((e & 1u) != 0u && (0.0 <= v8)) ? 1u : 0u; vh = 0u;
            if(e!=0u) {
                v8 = (vc & 1u) != 0u ? 1.0 : 0.0;
            }
            { uint a_ = s30; uint b_ = 0x90u; uint r_ = a_ + b_; cc = ((~(a_ ^ b_) & (a_ ^ r_)) >> 31) != 0u; s20 = r_; }
            s16 = U(P60,0x90); s17 = U(P60,0x94); s18 = U(P60,0x98); s19 = U(P60,0x9c);
            { uint a_ = s29; uint b_ = 1u; uint r_ = a_ + b_; cc = ((~(a_ ^ b_) & (a_ ^ r_)) >> 31) != 0u; s29 = r_; }
            if(e!=0u) {
                v11 = F(s19); v9 = F(s16) * v11; v10 = F(s17) * v11; v11 = F(s18) * v11; v5 = v8 * (-v9) + v5; v6 = v8 * (-v10) + v6; v7 = v8 * (-v11) + v7;
            }
            pc = 0x48u;
        }
        if (pc == 0x48u) continue;
        break;
    }
    if (pc == 0x10cu) {
        if(e!=0u) {
            v5 = a2[0]; v6 = a2[1]; v29 = a4[1]; v19 = a5[2];
        }
        {
            vec2 uv_ = vec2(v5, v6); uv_ *= sc(0);
            { vec4 t_ = vec4(texture(tex[0], uv_));
                if(e!=0u) {
                    v11 = t_[0]; v12 = t_[1];
                }
            }
        }
        {
            vec2 uv_ = vec2(v5, v6); uv_ *= sc(1);
            { vec4 t_ = vec4(texture(tex[1], uv_));
                if(e!=0u) {
                    v13 = t_[0]; v14 = t_[1]; v15 = t_[2];
                }
            }
        }
        if(e!=0u) {
            v26 = a5[0]; v9 = v29 * v19; v28 = a4[2]; v20 = a5[1];
        }
        s101 = U(Flags,0x0);
        if(e!=0u) {
            v8 = a5[3]; v30 = a4[0]; v10 = v28 * v26; v9 = v20 * (-v28) + v9; v27 = v8 * v9; v9 = v30 * v20; v10 = v19 * (-v30) + v10; v25 = v8 * v10;
        }
        s20 = U(P70,0xac);
        if(e!=0u) {
            v9 = v26 * (-v29) + v9; v24 = v8 * v9; v7 = a3[0];
        }
        s16 = U(View,0x218); s17 = U(View,0x21c);
        if(e!=0u) {
            v10 = v7 * v7; v9 = a3[1];
        }
        s100 = s101 & 8u; cc = s100 != 0u;
        if(e!=0u) {
            v8 = a3[2];
        }
        cc = s100 != 0u;
        if(e!=0u) {
            v10 = v9 * v9 + v10;
        }
        { uint lo_ = cc ? e : 0u; uint hi_ = cc ? eh : 0u; s28 = lo_; s29 = hi_; }
        if(e!=0u) {
            v10 = v8 * v8 + v10; v10 = inversesqrt(v10);
        }
        { uint lo_ = s28; uint hi_ = s29; vc = lo_; vh = hi_; }
        if(e!=0u) {
            v16 = v7 * v10; v17 = v9 * v10; v18 = v8 * v10; v7 = F(s16) * v2; v8 = F(s17) * v3; v9 = F(0x3f800000u); v10 = F(0x3f800000u);
            v31 = 2.0 * v11 + -1.0; v32 = v31 * v31; v11 = 2.0 * v12 + -1.0; v32 = v11 * v11 + v32; v12 = sat(1.0 - v32); v26 = v26 * v31;
            v12 = sqrt(v12); v32 = v27 * v11 + v26; v20 = v20 * v31; v19 = v19 * v31; v32 = v12 * v30 + v32; v20 = v11 * v25 + v20; v25 = v32 * v32;
            v20 = v12 * v29 + v20; v19 = v11 * v24 + v19; v11 = v28 * v12 + v19; v25 = v20 * v20 + v25; v25 = v11 * v11 + v25; v12 = inversesqrt(v25);
            v24 = F(s20) * v15; v15 = v12 * v32; v19 = v12 * v20; v20 = v12 * v11; v27 = max(1.0, v24);
        }
        { uint lo_ = vc; uint hi_ = vh; vc = lo_; vh = hi_; }
        pc = ((vc & 1u) == 0u) ? 0x2b8u : 0x2a4u;
    }
    if (pc == 0x2a4u) {
        {
            vec2 uv_ = vec2(v7, v8); uv_ *= sc(2);
            { vec4 t_ = vec4(texture(tex[2], uv_));
                if(e!=0u) {
                    v9 = t_[0]; v10 = t_[1];
                }
            }
        }
        pc = 0x2b8u;
    }
    if (pc == 0x2b8u) {
        { uint lo_ = e & s28; uint hi_ = eh & s29; vc = lo_; vh = hi_; }
        cc = (vc | vh) != 0u; pc = ((vc & 1u) == 0u) ? 0x2ecu : 0x2c0u;
    }
    if (pc == 0x2c0u) {
        s16 = U(P10,0xf0); s17 = U(P10,0xf4); s18 = U(P10,0xf8); s19 = U(P10,0xfc);
        if(e!=0u) {
            v11 = F(s17); v12 = F(s19); v9 = F(s16) * v9 + v11; v10 = v10 * F(s18) + v12;
        }
        pc = 0x2ecu;
    }
    if (pc == 0x2ecu) {
        if(e!=0u) {
            v31 = 0.0;
        }
        vc = s101 & 0x10000u; cc = vc != 0u; vh = s101 & 1u; cc = vh != 0u; cc = vc != 0u;
        if(e!=0u) {
            v59 = 0.0; v32 = 0.0; v58 = 0.0; v57 = 0.0; v24 = 0.0;
        }
        { uint lo_ = cc ? e : 0u; uint hi_ = cc ? eh : 0u; s16 = lo_; s17 = hi_; }
        cc = vh != 0u; pc = (!cc) ? 0x844u : 0x320u;
    }
    if (pc == 0x320u) {
        if(e!=0u) {
            v58 = 0.0;
        }
        { uint lo_ = e & s16; uint hi_ = eh & s17; vc = lo_; vh = hi_; }
        cc = (vc | vh) != 0u;
        if(e!=0u) {
            v57 = 0.0; v24 = 0.0; v11 = 0.0; v12 = 0.0; v31 = 0.0; v36 = 0.0; v37 = 0.0; v35 = 0.0;
        }
        pc = ((vc & 1u) == 0u) ? 0x434u : 0x34cu;
    }
    if (pc == 0x34cu) {
        if(e!=0u) {
            v12 = F(0x3f808102u);
        }
        vc = 0xbb810205u; s16 = U(P30,0xa0); s17 = U(P30,0xa4); s18 = U(P30,0xa8); s19 = U(P30,0xac); s20 = U(P30,0x20); s21 = U(P30,0x24);
        s22 = U(P30,0x28); s23 = U(P30,0x2c);
        if(e!=0u) {
            v26 = v23 - F(s16); v25 = v26 * v26; v24 = v22 - F(s17); v11 = v21 - F(s18); v25 = v24 * v24 + v25; v25 = v11 * v11 + v25; v31 = sqrt(v25);
            v31 = v31 - F(s23); v25 = inversesqrt(v25); v31 = max(0.0, v31); v33 = v26 * v25; v32 = F(s19) * v31; v33 = v15 * v33; v34 = v24 * v25;
            v31 = v32 * v31 + 1.0; v35 = v19 * v34 + v33; v34 = v11 * v25; v32 = v26 * v25 + v16; v26 = 1.0 / v31; v36 = v32 * v32;
            v33 = v24 * v25 + v17; v35 = v34 * v20 + v35; v24 = sat(v12 * v26 + F(vc)); v12 = max(0.0, (-v35)); v34 = v11 * v25 + v18;
            v36 = v33 * v33 + v36; v11 = F(s20) * v24; v25 = min(1.0, v12); v12 = F(s21) * v24; v31 = F(s22) * v24; v35 = v34 * v34 + v36;
            v26 = v25 * v11; v24 = v25 * v12; v36 = v25 * v31; v35 = inversesqrt(v35); v58 = max(0.0, v26); v57 = max(0.0, v24); v24 = max(0.0, v36);
            v36 = v35 * v32; v37 = v35 * v33; v35 = v35 * v34;
        }
        pc = 0x434u;
    }
    if (pc == 0x434u) {
        if(e!=0u) {
            v32 = 0.0;
        }
        vc = s101 & 0x20000u; cc = vc != 0u;
        if(e!=0u) {
            v33 = 0.0; v34 = 0.0; v38 = 0.0; v39 = 0.0; v40 = 0.0;
        }
        pc = (!cc) ? 0x544u : 0x458u;
    }
    if (pc == 0x458u) {
        if(e!=0u) {
            v33 = F(0x3f808102u);
        }
        vc = 0xbb810205u; s16 = U(P30,0xb0); s17 = U(P30,0xb4); s18 = U(P30,0xb8); s19 = U(P30,0xbc); s20 = U(P30,0x30); s21 = U(P30,0x34);
        s22 = U(P30,0x38); s23 = U(P30,0x3c);
        if(e!=0u) {
            v34 = v23 - F(s16); v25 = v34 * v34; v32 = v22 - F(s17); v38 = v21 - F(s18); v25 = v32 * v32 + v25; v25 = v38 * v38 + v25; v40 = sqrt(v25);
            v40 = v40 - F(s23); v39 = inversesqrt(v25); v40 = max(0.0, v40); v42 = v34 * v39; v41 = F(s19) * v40; v42 = v15 * v42; v43 = v32 * v39;
            v40 = v41 * v40 + 1.0; v42 = v43 * v19 + v42; v43 = v38 * v39; v41 = 1.0 / v40; v40 = v34 * v39 + v16; v42 = v43 * v20 + v42;
            v34 = sat(v33 * v41 + F(vc)); v33 = max(0.0, (-v42)); v25 = v40 * v40; v41 = v32 * v39 + v17; v32 = F(s20) * v34; v43 = min(1.0, v33);
            v33 = F(s21) * v34; v34 = F(s22) * v34; v42 = v38 * v39 + v18; v25 = v41 * v41 + v25; v45 = v43 * v32; v38 = v43 * v33; v39 = v43 * v34;
            v25 = v42 * v42 + v25; v44 = max(0.0, v45); v38 = max(0.0, v38); v39 = max(0.0, v39); v43 = inversesqrt(v25); v58 = v58 + v44;
            v57 = v57 + v38; v24 = v24 + v39; v38 = v43 * v40; v39 = v43 * v41; v40 = v43 * v42;
        }
        pc = 0x544u;
    }
    if (pc == 0x544u) {
        if(e!=0u) {
            v42 = 0.0;
        }
        vc = s101 & 0x40000u; cc = vc != 0u;
        if(e!=0u) {
            v43 = 0.0; v41 = 0.0; v44 = 0.0; v45 = 0.0; v46 = 0.0;
        }
        pc = (!cc) ? 0x654u : 0x568u;
    }
    if (pc == 0x568u) {
        if(e!=0u) {
            v45 = F(0x3f808102u);
        }
        vc = 0xbb810205u; s16 = U(P30,0xc0); s17 = U(P30,0xc4); s18 = U(P30,0xc8); s19 = U(P30,0xcc); s20 = U(P30,0x40); s21 = U(P30,0x44);
        s22 = U(P30,0x48); s23 = U(P30,0x4c);
        if(e!=0u) {
            v42 = v23 - F(s16); v25 = v42 * v42; v41 = v22 - F(s17); v43 = v21 - F(s18); v25 = v41 * v41 + v25; v25 = v43 * v43 + v25; v46 = sqrt(v25);
            v46 = v46 - F(s23); v47 = max(0.0, v46); v48 = F(s19) * v47; v44 = inversesqrt(v25); v46 = v41 * v44; v47 = v48 * v47 + 1.0;
            v48 = v42 * v44; v25 = v15 * v48; v25 = v46 * v19 + v25; v47 = 1.0 / v47; v49 = sat(v45 * v47 + F(vc)); v47 = v42 * v44 + v16;
            v45 = v41 * v44 + v17; v41 = v43 * v44; v25 = v41 * v20 + v25; v41 = F(s22) * v49; v42 = F(s20) * v49; v46 = v43 * v44 + v18;
            v43 = max(0.0, (-v25)); v48 = min(1.0, v43); v44 = v47 * v47; v43 = F(s21) * v49; v44 = v45 * v45 + v44; v44 = v46 * v46 + v44;
            v49 = inversesqrt(v44); v44 = v49 * v47; v47 = v48 * v41; v47 = max(0.0, v47); v24 = v24 + v47; v47 = v48 * v43; v48 = v48 * v42;
            v48 = max(0.0, v48); v47 = max(0.0, v47); v58 = v58 + v48; v57 = v57 + v47; v45 = v49 * v45; v46 = v49 * v46;
        }
        pc = 0x654u;
    }
    if (pc == 0x654u) {
        if(e!=0u) {
            v51 = 0.0;
        }
        vc = s101 & 0x80000u; cc = vc != 0u;
        if(e!=0u) {
            v52 = 0.0; v50 = 0.0; v48 = 0.0; v49 = 0.0; v47 = 0.0;
        }
        pc = (!cc) ? 0x76cu : 0x678u;
    }
    if (pc == 0x678u) {
        if(e!=0u) {
            v49 = F(0x3f808102u);
        }
        vc = 0xbb810205u; s16 = U(P30,0xd0); s17 = U(P30,0xd4); s18 = U(P30,0xd8); s19 = U(P30,0xdc); s20 = U(P30,0x50); s21 = U(P30,0x54);
        s22 = U(P30,0x58); s23 = U(P30,0x5c);
        if(e!=0u) {
            v51 = v23 - F(s16); v25 = v51 * v51; v47 = v22 - F(s17); v48 = v21 - F(s18); v25 = v47 * v47 + v25; v25 = v48 * v48 + v25; v52 = sqrt(v25);
            v52 = v52 - F(s23); v52 = max(0.0, v52); v53 = F(s19) * v52; v52 = v53 * v52 + 1.0; v54 = 1.0 / v52; v53 = inversesqrt(v25);
            v52 = v47 * v53; v50 = v48 * v53 + v18; v48 = v48 * v53; v49 = sat(v49 * v54 + F(vc)); v54 = v51 * v53; v54 = v15 * v54;
            v51 = v51 * v53 + v16; v25 = v19 * v52 + v54; v52 = v47 * v53 + v17; v47 = v51 * v51; v26 = v52 * v52 + v47; v25 = v48 * v20 + v25;
            v48 = max(0.0, (-v25)); v26 = v50 * v50 + v26; v53 = min(1.0, v48); v48 = F(s20) * v49; v54 = inversesqrt(v26); v47 = F(s22) * v49;
            v49 = F(s21) * v49; v51 = v54 * v51; v52 = v54 * v52; v50 = v54 * v50; v54 = v53 * v48; v54 = max(0.0, v54); v58 = v58 + v54;
            v54 = v53 * v49; v53 = v53 * v47; v53 = max(0.0, v53); v54 = max(0.0, v54); v24 = v24 + v53; v57 = v57 + v54;
        }
        pc = 0x76cu;
    }
    if (pc == 0x76cu) {
        if(e!=0u) {
            v36 = v36 * v15; v25 = v37 * v19 + v36; v37 = v51 * v15; v26 = v52 * v19 + v37; v25 = v20 * v35 + v25; v35 = max(0.0, (-v25));
            v26 = v20 * v50 + v26; v35 = min(1.0, v35); v36 = max(0.0, (-v26)); v35 = log2(v35); v36 = min(1.0, v36); v37 = v35 * v27; v36 = log2(v36);
            v35 = v36 * v27; v36 = exp2(v37); v37 = v38 * v15; v37 = v19 * v39 + v37; v37 = v20 * v40 + v37; v37 = max(0.0, (-v37));
            v37 = min(1.0, v37); v37 = log2(v37); v37 = v37 * v27; v11 = v36 * v11; v37 = exp2(v37); v12 = v36 * v12; v31 = v36 * v31; v36 = v44 * v15;
            v36 = v19 * v45 + v36; v36 = v20 * v46 + v36; v36 = max(0.0, (-v36)); v36 = min(1.0, v36); v36 = log2(v36); v36 = v36 * v27;
            v36 = exp2(v36); v11 = v32 * v37 + v11; v12 = v33 * v37 + v12; v31 = v34 * v37 + v31; v35 = exp2(v35); v25 = v36 * v41 + v31;
            v11 = v42 * v36 + v11; v12 = v43 * v36 + v12; v11 = v48 * v35 + v11; v12 = v49 * v35 + v12; v25 = v47 * v35 + v25; v31 = max(0.0, v11);
            v59 = max(0.0, v12); v32 = max(0.0, v25);
        }
        pc = 0x844u;
    }
    if (pc == 0x844u) {
        if(e!=0u) {
            v11 = v30 * v16; v11 = v17 * v29 + v11; v11 = v18 * v28 + v11; v12 = v15 * v16; v41 = v19 * v17 + v12; v41 = v18 * v20 + v41;
            v43 = sat(1.0 + v11); v11 = v15 * v15;
        }
        s16 = U(P20,0x4f0); s17 = U(P20,0x4f4); s18 = U(P20,0x4f8); s19 = U(P20,0x4fc); s20 = U(P20,0x500); s21 = U(P20,0x504); s22 = U(P20,0x508);
        s23 = U(P20,0x50c); s24 = U(P20,0x510); s25 = U(P20,0x514); s26 = U(P20,0x518); s27 = U(P20,0x51c); s28 = U(P20,0x520); s29 = U(P20,0x524);
        s30 = U(P20,0x528); s31 = U(P20,0x52c);
        if(e!=0u) {
            v11 = v19 * v19 + v11; v11 = v20 * v20 + v11; v12 = inversesqrt(v11); v36 = v12 * v15;
        }
        s32 = U(P20,0x530); s33 = U(P20,0x534); s34 = U(P20,0x538); s35 = U(P20,0x53c); s36 = U(P20,0x540); s37 = U(P20,0x544); s38 = U(P20,0x548);
        s39 = U(P20,0x54c); s40 = U(P20,0x550); s41 = U(P20,0x554); s42 = U(P20,0x558); s43 = U(P20,0x55c); s44 = U(P20,0x560); s45 = U(P20,0x564);
        s46 = U(P20,0x568); s47 = U(P20,0x56c);
        if(e!=0u) {
            v39 = v12 * v19; v37 = v12 * v20;
        }
        s48 = U(P30,0x80); s49 = U(P30,0x84); s50 = U(P30,0x88); s51 = U(P30,0x8c); s52 = U(P30,0x90); s53 = U(P30,0x94); s54 = U(P30,0x98);
        s55 = U(P30,0x9c); s76 = U(P30,0xe0); s77 = U(P30,0xe4); s78 = U(P30,0xe8); s79 = U(P30,0xec); s88 = U(View,0x22c); s84 = U(P30,0x60);
        s85 = U(P30,0x64); s86 = U(P30,0x68); s87 = U(P30,0x6c); s56 = U(P30,0x0); s57 = U(P30,0x4); s58 = U(P30,0x8); s59 = U(P30,0xc);
        s60 = U(P30,0x10); s61 = U(P30,0x14); s62 = U(P30,0x18); s63 = U(P30,0x1c);
        if(e!=0u) {
            v52 = 2.0 * v41; v55 = (-v15) * v52 + v16; v56 = (-v19) * v52 + v17; v52 = (-v20) * v52 + v18; v11 = F(s35); v12 = F(s33) * v39 + v11;
            v12 = F(s34) * v37 + v12;
        }
        s55 = U(P0,0x48); vc = s88 & 2u; cc = vc != 0u; s88 = U(P30,0x100); s89 = U(P30,0x104); s90 = U(P30,0x108); s91 = U(P30,0x10c);
        s70 = U(P0,0x40); s71 = U(P0,0x44); cc = vc != 0u;
        if(e!=0u) {
            v11 = min(v9, v10); v9 = v9 * v10; v10 = F(s24) * v36; v10 = F(s25) * v39 + v10; v10 = F(s26) * v37 + v10; v12 = v10 * v39 + v12;
            v10 = F(s28) * v36; v28 = F(s29) * v39 + v10; v28 = F(s30) * v37 + v28; v10 = F(s23); v10 = F(s32) + v10; v12 = F(s27) * v39 + v12;
            v12 = v28 * v37 + v12; v28 = F(s20) * v36; v28 = F(s21) * v39 + v28; v28 = F(s22) * v37 + v28; v10 = v10 + v28; v12 = F(s31) * v37 + v12;
            v12 = v10 * v36 + v12; v12 = max(F(0x3d800000u), v12);
        }
        { uint lo_ = cc ? e : 0u; uint hi_ = cc ? eh : 0u; vc = lo_; vh = hi_; }
        if(e!=0u) {
            v30 = (vc & 1u) != 0u ? v9 : v11; v9 = F(s52) * v15; v9 = F(s53) * v19 + v9; v9 = F(s54) * v20 + v9; v10 = max(0.0, (-v9)); v9 = F(s76);
            v11 = F(s16) * v12 + v9; v9 = F(s84) - v9; v9 = v9 * v19 + v9; v12 = F(s52) + v16; v54 = F(s52) * v55; v55 = F(s48) * v55;
            v54 = F(s53) * v56 + v54; v55 = F(s49) * v56 + v55; v54 = F(s54) * v52 + v54; v54 = max(0.0, (-v54)); v55 = F(s50) * v52 + v55;
            v55 = max(0.0, (-v55)); v54 = min(1.0, v54); v56 = F(0x3d22ca2fu); v56 = v56 * v27 + F(0x3daf7aacu); v53 = F(s48) + v16;
        }
        s51 = U(P0,0xc);
        if(e!=0u) {
            v48 = v12 * v12; v49 = v12 * v16; v52 = log2(v54); v54 = min(1.0, v55); v54 = log2(v54); v54 = v54 * v27; v55 = v53 * v53;
        }
        s92 = U(P10,0x0); s93 = U(P10,0x4); s94 = U(P10,0x8); s95 = U(P10,0xc);
        if(e!=0u) {
            v52 = v52 * v27; v46 = F(s39); v52 = exp2(v52);
        }
        s17 = U(P90,0x134);
        if(e!=0u) {
            v52 = v52 * v56; v27 = exp2(v54); v54 = F(s50) + v18; v34 = min(1.0, v10);
        }
        s68 = U(P70,0x68); s69 = U(P70,0x6c);
        if(e!=0u) {
            v10 = F(s48) * v15; v10 = F(s49) * v19 + v10; v10 = F(s50) * v20 + v10; v12 = max(0.0, (-v10)); v44 = 2.0 + F(s69); v38 = min(1.0, v12);
            v29 = F(s56) * v38; v11 = v29 * v30 + v11;
        }
        s39 = U(P70,0x88);
        if(e!=0u) {
            v27 = v27 * v56; v56 = F(s49) + v17; v55 = v56 * v56 + v55; v55 = v54 * v54 + v55; v55 = inversesqrt(v55); v53 = v55 * v53; v53 = v16 * v53;
            v56 = v55 * v56; v54 = v55 * v54; v55 = F(s53) + v17; v48 = v55 * v55 + v48; v49 = v17 * v55 + v49; v55 = F(s54) + v18;
            v48 = v55 * v55 + v48; v48 = inversesqrt(v48); v49 = v18 * v55 + v49; v55 = (-v48) * v49 + 1.0; v53 = v56 * v17 + v53; v49 = v55 * v55;
            v48 = F(s40) * v36; v53 = v54 * v18 + v53; v53 = 1.0 - v53; v49 = v49 * v49; v48 = F(s41) * v39 + v48; v54 = v53 * v53;
            v48 = F(s42) * v37 + v48; v54 = v54 * v54; v53 = v54 * v53; v54 = F(s39); v56 = (-F(s95)) * v54 + 1.0; v54 = F(s95) * v54;
            v53 = v53 * v56 + v54; v56 = v49 * v56; v28 = F(s60) * v34; v27 = v27 * v53; v54 = v55 * v56 + v54; v42 = F(s17) * v28 + v11;
            v40 = 1.0 / F(s94); v27 = v27 * v38; v53 = F(s56) * v30; v45 = F(s77); v53 = v53 * v27; v42 = 0.5 * v9 + v42; v9 = F(s71); v9 = F(s55) - v9;
            v9 = 1.0 / v9; v10 = (-v10) * v9; v10 = sat((-F(s71)) * v9 + v10); v9 = -1.0 + v10; v33 = F(s70) * v9 + 1.0;
        }
        {
            vec2 uv_ = vec2(v7, v8); uv_ *= sc(5);
            { vec4 t_ = vec4(texture(tex[5], uv_));
                if(e!=0u) {
                    v9 = t_[0]; v10 = t_[1]; v11 = t_[2]; v12 = t_[3];
                }
            }
        }
        if(e!=0u) {
            v7 = sat(1.0 + v41); v7 = v43 * v7; v7 = log2(v7); v7 = v7 * v44; v7 = exp2(v7); v7 = v14 * v7; v8 = F(s68); v7 = F(s91) * v7;
            v8 = F(0x40200000u) + v8; v7 = v7 * v8;
        }
        s20 = U(P20,0x570); s21 = U(P20,0x574); s22 = U(P20,0x578); s23 = U(P20,0x57c); s24 = U(P20,0x580); s25 = U(P20,0x584); s26 = U(P20,0x588);
        s27 = U(P20,0x58c); s28 = U(P20,0x590); s29 = U(P20,0x594); s30 = U(P20,0x598); s31 = U(P20,0x59c); s32 = U(P20,0x5a0); s33 = U(P20,0x5a4);
        s34 = U(P20,0x5a8); s35 = U(P20,0x5ac);
        if(e!=0u) {
            v41 = F(0x3dcccccdu) * v7; v44 = F(s78);
        }
        vc = s51 & 1u; cc = vc != 0u; s48 = U(P20,0x5b0); s49 = U(P20,0x5b4); s50 = U(P20,0x5b8); s51 = U(P20,0x5bc); vc = 0x3e19999au;
        vh = 0xbdbc49cau;
        {
            vec2 uv_ = vec2(v5, v6); uv_ *= sc(4);
            { vec4 t_ = vec4(texture(tex[4], uv_));
                if(e!=0u) {
                    v5 = t_[0]; v6 = t_[1]; v7 = t_[2]; v8 = t_[3];
                }
            }
        }
        if(e!=0u) {
            v51 = F(s23); v51 = F(s21) * v39 + v51; v49 = F(s22) * v37 + v51;
        }
        s64 = U(P70,0x90); s65 = U(P70,0x94); s66 = U(P70,0x98); s67 = U(P70,0x9c); s68 = U(P70,0xa0); s69 = U(P70,0xa4); s70 = U(P70,0xa8);
        s71 = U(P70,0xac);
        if(e!=0u) {
            v50 = F(s20) + v46; v49 = v48 * v39 + v49;
        }
        s72 = U(P60,0x0); s73 = U(P60,0x4); s74 = U(P60,0x8); s75 = U(P60,0xc); s76 = U(P60,0x10); s77 = U(P60,0x14); s78 = U(P60,0x18);
        s79 = U(P60,0x1c);
        if(e!=0u) {
            v51 = F(s44) * v36; v55 = F(s51); v49 = F(s43) * v39 + v49; v51 = F(s45) * v39 + v51; v55 = F(s49) * v39 + v55; v25 = F(s50) * v37 + v55;
            v26 = F(s32) * v36; v51 = F(s46) * v37 + v51;
        }
        s4 = U(P60,0x110); s5 = U(P60,0x114); s6 = U(P60,0x118); s7 = U(P60,0x11c);
        if(e!=0u) {
            v48 = v37 * v51 + v49; v49 = F(s36) * v36; v49 = F(s37) * v39 + v49; v43 = F(s24) * v36; v49 = F(s38) * v37 + v49; v26 = F(s33) * v39 + v26;
            v51 = v52 * v54; v49 = v50 + v49; v50 = F(s28) * v36; v52 = F(s29) * v39 + v50; v52 = F(s30) * v37 + v52; v25 = v52 * v39 + v25;
            v46 = F(s25) * v39 + v43; v43 = F(s64) * v14; v39 = F(s31) * v39 + v25; v47 = F(s27); v48 = F(s47) * v37 + v48; v26 = F(s34) * v37 + v26;
            v43 = F(s72) * v43; v25 = v37 * v26 + v39; v47 = F(s48) + v47; v56 = F(s76); v48 = v49 * v36 + v48; v46 = F(s26) * v37 + v46;
            v56 = F(s68) * v56; v48 = max(F(0x3d800000u), v48); v39 = v47 + v46; v25 = F(s35) * v37 + v25; v26 = a0[0]; v48 = F(s16) * v48 + v45;
            v49 = F(s85) - v45; v45 = v51 * v34; v53 = F(s60) * v45 + v53; v47 = 0.5 * v19 + 0.5; v25 = v39 * v36 + v25; v52 = F(s57) * v38;
            v39 = F(s58) * v30; v35 = F(s7); v48 = v47 * v49 + v48; v55 = F(s61) * v34; v37 = F(s65) * v14; v49 = F(s57) * v30; v50 = F(s86) - v44;
            v39 = v39 * v27; v48 = v52 * v30 + v48; v46 = F(s92) * v26; v26 = max(F(0x3d800000u), v25); v36 = F(s16) * v26 + v44; v37 = F(s73) * v37;
            v49 = v49 * v27; v54 = F(s58) * v38; v48 = F(s17) * v55 + v48; v36 = v47 * v50 + v36; v27 = F(s61) * v45 + v49; v26 = F(s62) * v34;
            v38 = F(s66) * v14; v36 = v54 * v30 + v36; v14 = F(s74) * v38; v44 = F(s78); v36 = F(s17) * v26 + v36; v25 = F(s62) * v45 + v39;
            v39 = F(s77); v39 = F(s69) * v39; v44 = F(s70) * v44; v38 = a0[2]; v38 = F(s92) * v38; v45 = F(0x3fb0852eu); v58 = v40 * v9 + v58;
            v51 = v42 * v33 + v58; v57 = v40 * v10 + v57; v34 = v48 * v33 + v57; v59 = v40 * v12 + v59; v27 = v39 * v27 + v59; v11 = v11 * v40 + v24;
            v24 = v36 * v33 + v11; v39 = a0[1]; v39 = F(s92) * v39; v5 = v5 * v5; v5 = F(s88) * v41 + v5; v5 = v43 * v5; v43 = v5 * v51; v10 = v6 * v6;
            v6 = v12 * v40 + v31; v9 = v7 * v7; v10 = F(s89) * v41 + v10; v42 = v37 * v10; v9 = F(s90) * v41 + v9; v5 = v12 * v40 + v32;
            v32 = v56 * v53 + v6; v43 = v32 * v13 + v43; v12 = F(s4) * v35 + v43; v12 = (-F(s7)) * v43 + v12; v12 = (v46 * v12) * 2.0;
            v32 = F(0x3e19999au); v41 = v32 * v12 + F(0x3d4ccccdu); v31 = F(vc) * v12 + 0.5; v31 = v12 * v31 + F(0x3d75c290u); v31 = 1.0 / v31;
            v12 = v41 * v12 + F(0x3b83126fu); v41 = v14 * v9; v9 = v42 * v34; v9 = v27 * v13 + v9; v10 = F(s5) * v35 + v9; v34 = v44 * v25 + v5;
            v10 = (-F(s7)) * v9 + v10; v9 = v41 * v24; v9 = v34 * v13 + v9; v27 = (v39 * v10) * 2.0; v10 = F(s6) * v35 + v9; v10 = (-F(s7)) * v9 + v10;
            v13 = (v38 * v10) * 2.0; v24 = F(vc) * v27 + 0.5; v30 = F(vc) * v13 + 0.5; v33 = v27 * v24 + F(0x3d75c290u);
            v24 = v32 * v27 + F(0x3d4ccccdu); v30 = v13 * v30 + F(0x3d75c290u); v32 = v32 * v13 + F(0x3d4ccccdu); v33 = 1.0 / v33;
            v24 = v24 * v27 + F(0x3b83126fu); v27 = 1.0 / v30; v30 = v32 * v13 + F(0x3b83126fu); v31 = v31 * v12; v13 = v33 * v24; v24 = v27 * v30;
            v49 = sat(v45 * v31 + F(vh)); v31 = sat(v45 * v13 + F(vh)); v27 = sat(v45 * v24 + F(vh)); v49 = sqrt(v49); v47 = sqrt(v31); v27 = sqrt(v27);
        }
        pc = (!cc) ? 0xfe8u : 0xef0u;
    }
    if (pc == 0xef0u) {
        if(e!=0u) {
            v10 = v57 + v59; v5 = v11 + v5; v6 = v58 + v6; v7 = v10 + v52; v5 = v5 + v54; v6 = v6 + v29;
        }
        s16 = U(P0,0x0); s17 = U(P0,0x4); s18 = U(P0,0x8); s19 = U(P0,0xc); s20 = U(P0,0x10); s21 = U(P0,0x14); s22 = U(P0,0x18); s23 = U(P0,0x1c);
        s24 = U(P0,0x20); s25 = U(P0,0x24); s26 = U(P0,0x28); s27 = U(P0,0x2c); s28 = U(P0,0x30); s29 = U(P0,0x34); s30 = U(P0,0x38);
        s31 = U(P0,0x3c);
        if(e!=0u) {
            v7 = v7 + v55; v5 = v5 + v26; v6 = v6 + v28; v5 = v7 + v5; v6 = F(0x3eaaaaabu) * v6; v5 = v5 * F(0x3eaaaaabu) + v6; v6 = 1.0 - v5;
            v7 = (-v6) * v6 + 1.0; v10 = sat(v6 * v6); v6 = 1.0 - F(s23); v11 = F(s20) * v10; v6 = v6 - F(s23); v12 = F(s24) * v10; v14 = F(s28) * v10;
            v7 = sat(v5 * (-v5) + v7); v25 = F(s21) * v7 + v11; v5 = sat(v5 * v5); v6 = 1.0 / v6; v12 = F(s25) * v7 + v12; v14 = F(s29) * v7 + v14;
            v9 = sat(v6 * v49 + (-F(s23))); v11 = sat(v6 * v47 + (-F(s23))); v6 = sat(v6 * v27 + (-F(s23))); v25 = F(s22) * v5 + v25;
            v12 = F(s26) * v5 + v12; v14 = F(s30) * v5 + v14; v7 = F(s16) + v25; v9 = log2(v9); v10 = F(s17) + v12; v11 = log2(v11); v5 = F(s18) + v14;
            v6 = log2(v6); v7 = v9 * v7; v9 = v11 * v10; v5 = v6 * v5; v49 = sat(exp2(v7)); v47 = sat(exp2(v9)); v27 = sat(exp2(v5));
        }
        pc = 0xfe8u;
    }
    if (pc == 0xfe8u) {
        s4 = s101 & 4u; cc = s4 != 0u; vc = s101 & 0x400000u; cc = vc != 0u; cc = vc != 0u;
        vc = ((e & 1u) != 0u && (s4 != 0u)) ? 1u : 0u; vh = 0u;
        if(e!=0u) {
            v6 = a1[0]; v5 = a1[1]; v0 = a1[2]; v1 = v6 + v49; v5 = v5 + v47; v6 = v0 + v27; v0 = (vc & 1u) != 0u ? v1 : v49;
            v1 = (vc & 1u) != 0u ? v5 : v47; v5 = (vc & 1u) != 0u ? v6 : v27;
        }
        vh = cc ? 0u : 1u; cc = s4 != 0u; vc = cc ? 1u : 0u; vc = vh | vc; cc = vc != 0u;
        if(e!=0u) {
            v14 = 0.0;
        }
        pc = (cc) ? 0x10b8u : 0x1048u;
    }
    if (pc == 0x1048u) {
        s4 = U(View,0x220); s5 = U(View,0x224); s6 = U(View,0x228); s7 = U(View,0x22c); s16 = U(P10,0x10); s17 = U(P10,0x14); s18 = U(P10,0x18);
        s19 = U(P10,0x1c); s20 = U(P10,0x20); s21 = U(P10,0x24); s22 = U(P10,0x28); s23 = U(P10,0x2c);
        if(e!=0u) {
            v0 = F(s4) - v23; v0 = v0 * v0; v1 = F(s5) - v22; v0 = v1 * v1 + v0; v1 = F(s6) - v21; v0 = v1 * v1 + v0; v0 = sqrt(v0); v14 = F(s23);
            v14 = sat(F(s22) * v0 + v14); v0 = F(s16) * v14 + v49; v1 = F(s17) * v14 + v47; v5 = F(s18) * v14 + v27; v0 = v14 * (-v49) + v0;
            v1 = v14 * (-v47) + v1; v5 = v14 * (-v27) + v5;
        }
        pc = 0x10b8u;
    }
    if (pc == 0x10b8u) {
        if(e!=0u) {
            v6 = 0.0;
        }
        cc = s100 != 0u;
        { uint lo_ = cc ? e : 0u; uint hi_ = cc ? eh : 0u; vc = lo_; vh = hi_; }
        s28 = U(P50,0x1c); s24 = U(P70,0x9c); s20 = U(P60,0xc); s16 = U(Flags,0x0); s4 = U(P10,0x30); s5 = U(P10,0x34); s6 = U(P10,0x38);
        s7 = U(P10,0x3c);
        if(e!=0u) {
            v7 = F(s28); v7 = F(s24) * v7; v7 = F(s20) * v7; v10 = v7 * v8; v11 = F(s7) * v10; v9 = F(s4) * v0; v8 = F(s5) * v1; v7 = F(s6) * v5;
            v13 = (vc & 1u) != 0u ? v10 : v11; v21 = (vc & 1u) != 0u ? v0 : v9; v22 = (vc & 1u) != 0u ? v1 : v8; v23 = (vc & 1u) != 0u ? v5 : v7;
        }
        vc = s16 & 0x2000000u; cc = vc != 0u; cc = vc != 0u;
        { uint lo_ = cc ? e : 0u; uint hi_ = cc ? eh : 0u; vc = lo_; vh = hi_; }
        pc = ((vc & 1u) != 0u) ? 0x14e4u : 0x1138u;
    }
    if (pc == 0x1138u) {
        if(e!=0u) {
            v25 = 0.0;
        }
        s40 = U(View,0x218); s41 = U(View,0x21c); s42 = U(P0,0x2c); s32 = U(P0,0x284); s33 = U(P0,0x288); s34 = U(P0,0x28c); s35 = U(P0,0x290);
        s43 = U(P0,0x3c); s44 = U(P0,0x26c); s36 = U(P0,0xc);
        if(e!=0u) {
            v11 = F(s40) * v2; v12 = F(s41) * v3;
        }
        {
            vec2 uv_ = vec2(v11, v12); uv_ *= sc(9);
            { vec4 t_ = vec4(texture(tex[9], uv_));
                if(e!=0u) {
                    v7 = t_[0];
                }
            }
        }
        {
            vec2 uv_ = vec2(v11, v12); uv_ *= sc(12);
            { vec4 t_ = vec4(texture(tex[12], uv_));
                if(e!=0u) {
                    v0 = t_[0]; v1 = t_[1]; v2 = t_[2];
                }
            }
        }
        if(e!=0u) {
            v3 = F(s32); v6 = F(s34) - v3; v24 = F(s32); v5 = F(s33); v6 = 1.0 / v6; v5 = F(s32) - v5; v5 = 1.0 / v5;
        }
        vc = s36 & 4u; cc = vc != 0u;
        if(e!=0u) {
            v7 = F(s42) + v7; v7 = 1.0 / v7; v24 = F(s43) * v7 + (-v24); v28 = sat((-v24) * v6 + 1.0); v27 = (-F(s43)) * v7; v7 = F(s44) * v28;
            v26 = (-v27) * v5; v31 = 2.0 * v1 + -1.0; v1 = max(F(s35), v7); v32 = 2.0 * v0 + -1.0; v26 = sat((-F(s33)) * v5 + v26); v8 = v26 * v1;
            v30 = F(s40) * v32; v29 = F(s41) * (-v31); v0 = v30 * v8 + v11; v1 = v29 * v8 + v12;
        }
        {
            vec2 uv_ = vec2(v0, v1); uv_ *= sc(13);
            { vec4 t_ = vec4(texture(tex[13], uv_));
                if(e!=0u) {
                    v5 = t_[0]; v6 = t_[1]; v7 = t_[2];
                }
            }
        }
        if(e!=0u) {
            v0 = (-v30) * v8 + v11; v1 = (-v29) * v8 + v12;
        }
        {
            vec2 uv_ = vec2(v0, v1); uv_ *= sc(14);
            { vec4 t_ = vec4(texture(tex[14], uv_));
                if(e!=0u) {
                    v8 = t_[0]; v9 = t_[1]; v10 = t_[2];
                }
            }
        }
        if(e!=0u) {
            v1 = 2.0 * v2 + -1.0; v0 = v32 + v31; v0 = v0 + v1; v2 = v6 * v31; v33 = (v5 * v32) * 2.0; v6 = v7 * v1; v5 = v9 * v31;
            v8 = (v8 * v32) * 2.0; v1 = v10 * v1; v33 = 2.0 * v2 + v33; v8 = 2.0 * v5 + v8; v33 = 2.0 * v6 + v33; v8 = 2.0 * v1 + v8; v2 = v33 - v0;
            v5 = v8 - v0;
        }
        pc = (!cc) ? 0x12e8u : 0x128cu;
    }
    if (pc == 0x128cu) {
        if(e!=0u) {
            v0 = 0.5 * v5 + 0.5; v0 = (v2 * v0 + v0) * 0.5; v1 = v0 * v0 + F(0x3b808081u);
        }
        s0 = U(P0,0x270); s1 = U(P0,0x274);
        if(e!=0u) {
            v6 = (-F(s1)) * v26 + 1.0;
        }
        vc = ((e & 1u) != 0u && (v1 >= v6)) ? 1u : 0u; vh = 0u;
        if(e!=0u) {
            v6 = (vc & 1u) != 0u ? 1.0 : 0.0; v1 = 1.0 - F(s0); v6 = sat(v0 * v0 + v6); v0 = v1 * v6 + F(s0); v25 = (-v26) * v0 + v26;
        }
        pc = 0x12e8u;
    }
    if (pc == 0x12e8u) {
        if(e!=0u) {
            v0 = 0.0;
        }
        vc = s36 & 8u; cc = vc != 0u; pc = (!cc) ? 0x13b4u : 0x12f4u;
    }
    if (pc == 0x12f4u) {
        if(e!=0u) {
            v6 = 2.0 * v27;
        }
        s8 = U(P0,0x260); s9 = U(P0,0x264); s10 = U(P0,0x268);
        if(e!=0u) {
            v0 = F(s8) * v28; v0 = max(F(s35), v0); v8 = v26 * v0; v0 = v30 * v8 + v11; v1 = v29 * v8 + v12;
        }
        {
            vec2 uv_ = vec2(v0, v1); uv_ *= sc(10);
            { vec4 t_ = vec4(texture(tex[10], uv_));
                if(e!=0u) {
                    v7 = t_[0];
                }
            }
        }
        if(e!=0u) {
            v0 = (-v30) * v8 + v11; v1 = (-v29) * v8 + v12;
        }
        {
            vec2 uv_ = vec2(v0, v1); uv_ *= sc(11);
            { vec4 t_ = vec4(texture(tex[11], uv_));
                if(e!=0u) {
                    v8 = t_[0];
                }
            }
        }
        if(e!=0u) {
            v1 = (-F(s10)) * v26 + 1.0; v0 = 1.0 - F(s9); v7 = F(s42) + v7; v7 = 1.0 / v7; v6 = F(s43) * v7 + v6; v7 = F(s42) + v8; v7 = 1.0 / v7;
            v6 = F(s43) * v7 + v6; v6 = -0.5 * v6 + 1.0; v7 = F(0x3b808081u) + v6;
        }
        vc = ((e & 1u) != 0u && (v7 >= v1)) ? 1u : 0u; vh = 0u;
        if(e!=0u) {
            v1 = (vc & 1u) != 0u ? 1.0 : 0.0; v1 = sat(v1 + v6); v0 = v0 * v1 + F(s9); v0 = (-v26) * v0 + v26;
        }
        pc = 0x13b4u;
    }
    if (pc == 0x13b4u) {
        if(e!=0u) {
            v1 = 0.0;
        }
        vc = s36 & 2u; cc = vc != 0u; pc = (!cc) ? 0x1490u : 0x13c0u;
    }
    if (pc == 0x13c0u) {
        if(e!=0u) {
            v1 = v15 * v16; v1 = v17 * v19 + v1; v1 = v18 * v20 + v1; v1 = max(0.0, (-v1)); v1 = min(1.0, v1);
        }
        s0 = U(P0,0x2a0); s1 = U(P0,0x2a4); s2 = U(P0,0x2a8); s3 = U(P0,0x2ac); s8 = U(P0,0x278); s9 = U(P0,0x27c); s4 = U(P0,0x280);
        if(e!=0u) {
            v6 = F(0x3b808081u) + v1; v1 = (-v2) * v5 + 1.0; v2 = F(s0); v2 = F(0xba83126fu) + v2; v2 = min(F(s1), v2); v5 = F(s0) - v2; v5 = 1.0 / v5;
            v8 = (-v27) * v5; v7 = F(s2); v8 = sat(v2 * (-v5) + v8); v5 = F(s9) - v7; v5 = v8 * v5 + F(s2); v5 = v26 * v5 + F(0x3b808081u);
            v5 = 1.0 / v5; v7 = F(s3); v5 = sat(v5 * v6); v6 = 1.0 - F(s8); v7 = F(s4) - v7; v6 = v6 * v5; v2 = v7 * v8 + F(s3);
            v5 = sat(v6 * v5 + F(s8));
        }
        vc = ((e & 1u) != 0u && (v1 >= v2)) ? 1u : 0u; vh = 0u;
        if(e!=0u) {
            v26 = v5 * (-v26) + v26; v5 = (vc & 1u) != 0u ? 1.0 : 0.0; v5 = v5 * v26;
        }
        vc = ((e & 1u) != 0u && (0.0 < v2)) ? 1u : 0u; vh = 0u;
        if(e!=0u) {
            v1 = (vc & 1u) != 0u ? v5 : v26;
        }
        pc = 0x1490u;
    }
    if (pc == 0x1490u) {
        if(e!=0u) {
            v0 = max(max(v0, v1), v25);
        }
        s4 = U(P0,0x294); s5 = U(P0,0x298); s0 = U(P0,0x29c);
        if(e!=0u) {
            v1 = F(s4) - v3; v1 = 1.0 / v1; v1 = sat((-v24) * v1 + 1.0); v2 = 1.0 - F(s5); v1 = max(v2, v1); v0 = v1 * v0; v0 = v0 * (-v14) + v0;
        }
        vc = ((e & 1u) != 0u && (F(s0) <= v0)) ? 1u : 0u; vh = 0u;
        if(e!=0u) {
            v1 = (vc & 1u) != 0u ? 1.0 : 0.0; v6 = v1 * v0;
        }
        pc = 0x14e4u;
    }
    if (pc == 0x14e4u) {
        if(e!=0u) {
            v1 = (v13);
        }
        s4 = U(Flags,0x0); s0 = U(P60,0x11c);
        if(e!=0u) {
            v0 = 1.0 - F(s0);
        }
        vc = s4 & 0x800000u; cc = vc != 0u;
        if(e!=0u) {
            v0 = v0 * v6;
        }
        pc = (!cc) ? 0x1570u : 0x1518u;
    }
    if (pc == 0x1518u) {
        s0 = U(View,0x230); s1 = U(View,0x234); s2 = U(View,0x238); s3 = U(View,0x23c);
        if(e!=0u) {
            v2 = F(s0); v3 = F(s1); v1 = F(s2); v21 = F(s3) * v21 + v2; v22 = F(s3) * v22 + v3; v23 = F(s3) * v23 + v1; v5 = F(s3); v0 = F(s3) * v0;
            v1 = F(s3) * v13; v21 = (-F(s0)) * v5 + v21; v22 = (-F(s1)) * v5 + v22; v23 = (-F(s2)) * v5 + v23;
        }
        pc = 0x1570u;
    }
    if (pc == 0x1570u) {
        if(e!=0u) {
            h0 = packHalf2x16(vec2(v0, 0.0));
        }
        vc = 0x3a4d9a68u;
        if(e!=0u) {
            v2 = sat(F(vc) * v4); h1 = packHalf2x16(vec2(0.0, v1)); h3 = packHalf2x16(vec2(v21, v22));
        }
        s0 = U(View,0x22c); vc = s0 & 1u; cc = vc != 0u; cc = vc != 0u;
        { uint lo_ = cc ? e : 0u; uint hi_ = cc ? eh : 0u; vc = lo_; vh = hi_; }
        if(e!=0u) {
            v2 = (vc & 1u) != 0u ? v2 : v13; h2 = packHalf2x16(vec2(v23, v2));
        }
        { uint lo_ = s14; uint hi_ = s15; e = lo_; eh = hi_; }
        if(e==0u) discard;
        o1 = swz(vec4(unpackHalf2x16(h0), unpackHalf2x16(h1)),cb.c[455].y); o0 = swz(vec4(unpackHalf2x16(h3), unpackHalf2x16(h2)),cb.c[455].x);
        return;
        pc = 0x15ccu;
    }
    if (pc == 0x15ccu) {
        { uint lo_ = 0u; uint hi_ = 0u; e = lo_; eh = hi_; }
        if(e==0u) discard;
        o1 = swz(vec4(unpackHalf2x16(h0), unpackHalf2x16(h0)),cb.c[455].y);
        return;
        return;
    }
}
)";

bool named(const std::string& name, const char* wanted) {
    return name == wanted || name.rfind(std::string(wanted) + " +", 0) == 0;
}

uint32_t formatComponents(uint32_t dfmt) {
    switch (dfmt) {
    case 1: case 2: case 4: return 1;
    case 3: case 5: case 11: return 2;
    case 13: return 3;
    case 10: case 12: case 14: return 4;
    default: return 0;
    }
}

bool decodable(const VertexBuffer& v, bool integer) {
    if (!v.data.buffer || v.indexVgpr || v.stride % 4 ||
        (uint64_t(v.data.byteOffset) + v.offset) % 4)
        return false;
    if (!formatComponents(v.dfmt)) return false;
    const bool wide = v.dfmt == 4 || v.dfmt == 11 || v.dfmt == 13 || v.dfmt == 14;
    if (integer) return v.nfmt == 4 || v.nfmt == 5;
    if (wide) return v.nfmt == 7; // no normalized 32-bit approximation
    if (v.nfmt == 7) return v.dfmt == 2 || v.dfmt == 5 || v.dfmt == 12;
    return v.nfmt == 0 || v.nfmt == 1;
}

class Character final : public Pass {
public:
    const char* name() const override { return "character"; }
    int claims(const std::string& psName) const override {
        for (int i = 0; i < kVariantCount; ++i)
            if (named(psName, kVariants[i].ps)) return i + 1;
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        if (role < 1 || role > kVariantCount) return false;
        const Variant& var = kVariants[role - 1];
        if (var.unmatched && !d.compareCopy)
            return decline(role, "not yet matched (drawn into compare copies only)");
        const bool alternate = d.vsName && var.alternateVs && named(*d.vsName, var.alternateVs);
        const auto vsLoads = alternate ? var.alternateVsLoads : var.vsLoads;
        if (!d.vsName || (!named(*d.vsName, var.vs) && !alternate))
            return decline(role, "another vertex shader");
        if (d.upscaler) return decline(role, "an upscaler is on");
        if (d.pixelPositions) return decline(role, "positions in pixels");
        if (d.depthBias) return decline(role, "a polygon offset");
        if (d.colorCount != 2 || !d.color[0].view || !d.color[1].view || !d.depth.view)
            return decline(role, "not two colour targets with depth");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        if (d.indexed && (!d.index.buffer ||
                         (d.indexType != VK_INDEX_TYPE_UINT16 && d.indexType != VK_INDEX_TYPE_UINT32)))
            return decline(role, "an unsupported index buffer");

        uint32_t c[kConstants][4] = {};
        uint64_t loaded[(kConstants * 4 + 63) / 64] = {};
        const auto& offs = offsets(role, d, vsLoads);
        const size_t vsCount = vsLoads.size();
        for (size_t k = 0; k < vsCount + var.psLoads.size(); ++k) {
            const Load& l = k < vsCount ? vsLoads[k] : var.psLoads[k - vsCount];
            const uint32_t* data = l.vs ? d.vsData : d.psData;
            const size_t size = l.vs ? d.vsSize : d.psSize;
            std::array<uint32_t, 16> temporary{};
            const uint32_t* source = nullptr;
            if (data && offs[k] >= 0 && size_t(offs[k]) <= size &&
                l.n <= size - size_t(offs[k])) {
                source = data + offs[k];
            } else {
                const auto& load = l.vs ? d.vsLoad : d.load;
                if (load && load(l.pc, temporary.data(), l.n)) source = temporary.data();
            }
            if (!source) {
                char why[80];
                std::snprintf(why, sizeof why, "no %s load window at 0x%x (%u dwords)",
                              l.vs ? "vertex" : "pixel", l.pc, l.n);
                return decline(role, why);
            }
            for (uint32_t i = 0; i < l.n; ++i) {
                const uint32_t at = l.index * 4 + l.at + i, value = source[i];
                // Untaken branch captures contain zeros. They must not erase
                // a live copy from another load of the same source component.
                if (value || !(loaded[at / 64] >> (at % 64) & 1))
                    c[at / 4][at % 4] = value;
                loaded[at / 64] |= uint64_t(1) << (at % 64);
            }
        }
        int32_t vsCountValue;
        std::memcpy(&vsCountValue, &c[cV0 + 0x12c / 16][3], sizeof vsCountValue);
        float psCountValue;
        std::memcpy(&psCountValue, &c[cP60 + 0xdc / 16][3], sizeof psCountValue);
        if (vsCountValue > 1 || psCountValue > 1.0f)
            return decline(role, "more than one deformation entry");

        // All four VSs use v4, v8, v12, v16, v20 and v24.
        // VSV[0x20] is deliberately found separately by its first use PC.
        const VertexBuffer* slot[6] = {};
        const VertexBuffer* palette = nullptr;
        for (const VertexBuffer& v : d.vertex) {
            if (v.slot >= 0 && v.slot < 6 && v.components) {
                if (slot[v.slot]) return decline(role, "duplicate vertex input");
                slot[v.slot] = &v;
            } else if (v.slot < 0 && v.pc == 0x74) {
                if (palette) return decline(role, "ambiguous palette");
                palette = &v;
            }
        }
        const uint32_t used[] = {3, 4, 4, role == 1 ? 4u : 3u, 4, 2};
        for (int i = 0; i < 6; ++i) {
            const VertexBuffer* v = slot[i];
            if (!v) return decline(role, "a vertex input missing");
            bool plain = v->components >= used[i] && formatComponents(v->dfmt) >= used[i];
            for (uint32_t j = 0; j < used[i]; ++j)
                plain = plain && ((v->dstSel >> (3 * j)) & 7) == 4 + j;
            if (!plain || !decodable(*v, i == 2)) {
                char why[144];
                std::snprintf(why, sizeof why,
                              "input %d: dfmt %u nfmt %u stride %u sel 0x%x components %u index v%u",
                              i, v->dfmt, v->nfmt, v->stride, v->dstSel, v->components, v->indexVgpr);
                return decline(role, why);
            }
            const uint64_t offset = uint64_t(v->data.byteOffset) + v->offset;
            if (offset > UINT32_MAX) return decline(role, "a vertex offset overflow");
            c[cFetch + i][0] = uint32_t(offset / 4);
            c[cFetch + i][1] = v->stride / 4;
            c[cFetch + i][2] = v->dfmt | v->nfmt << 4;
        }
        if (!palette || !palette->data.buffer) return decline(role, "no palette at vertex PC 0x74");
        // The typed instruction explicitly selects float4. Its element index
        // addressing is equivalent to 48*index+16*r only for a 16-byte stride.
        if (palette->stride != 16 || palette->offset || palette->data.byteOffset % 4)
            return decline(role, "a palette address layout it does not read exactly");
        c[cPalette][0] = palette->data.byteOffset / 4;

        std::array<VkDescriptorImageInfo, kTextures> images{};
        for (int t = 0; t < kTextures; ++t) {
            const uint32_t pc = var.texturePc[t] ? var.texturePc[t] : var.texturePc[tBase];
            const Texture* x = nullptr;
            for (const Texture& candidate : d.textures)
                if (candidate.pc == pc) x = &candidate;
            if (!x || !x->view || !x->sampler) {
                char why[64];
                std::snprintf(why, sizeof why, "no texture and sampler at 0x%x", pc);
                return decline(role, why);
            }
            images[t] = {x->sampler, x->view, VK_IMAGE_LAYOUT_GENERAL};
            std::memcpy(&c[cTexScale + t / 2][(t & 1) * 2], x->scale, sizeof x->scale);
        }
        VkDescriptorImageInfo cube{};
        if (var.cubePc) {
            const Texture* x = nullptr;
            for (const Texture& candidate : d.textures)
                if (candidate.pc == var.cubePc) x = &candidate;
            if (!x || !x->view || !x->sampler)
                return decline(role, "no environment cube texture and sampler");
            if (x->scale[0] != 1.0f || x->scale[1] != 1.0f)
                return decline(role, "a scaled cube texture");
            cube = {x->sampler,x->view,VK_IMAGE_LAYOUT_GENERAL};
        }
        for (int i = 0; i < 2; ++i) {
            int32_t sw[4];
            swizzle(d.color[i], sw);
            c[cOut][i] = uint32_t(sw[0] | sw[1] << 2 | sw[2] << 4 | sw[3] << 6);
        }
        c[cOut][2] = d.clipZHalf ? 1 : 0;
        c[cOut][3] = alternate ? 1 : 0;

        Buffer constants;
        if (!gpu.upload(c, sizeof c, constants)) return decline(role, "no room for constants");
        const VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (kCompareOn) describe(c, slot, *palette, d);
        if (!logged_[role].exchange(true))
            std::fprintf(stderr, "[native] character: drawing %s natively\n", var.ps);

        std::array<VkDescriptorBufferInfo, 8> buffers{};
        for (int i = 0; i < 6; ++i)
            buffers[i] = {slot[i]->data.buffer, slot[i]->data.offset, slot[i]->data.range};
        buffers[6] = {palette->data.buffer, palette->data.offset, palette->data.range};
        buffers[7] = {constants.buffer, constants.offset, constants.range};
        const VkViewport viewport = d.vkViewport;
        const VkRect2D scissor{{d.scissor[0], d.scissor[1]},
                              {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        std::array<uint32_t, 6> st;
        for (int i = 0; i < 2; ++i) st[i * 3] = d.stencilRef[i], st[i * 3 + 1] = d.stencilCompare[i], st[i * 3 + 2] = d.stencilWrite[i];
        gpu.beginRendering(d.color, 2, d.depth, d.hasStencil);
        gpu.record([pipe, layout = layout_, buffers, images, cube, viewport, scissor,
                    index = d.index, firstIndex = d.firstIndex, indexed = d.indexed,
                    indexType = d.indexType, count = d.count, st, bc = std::to_array(d.blendConstants),
                    pushSet = gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetBlendConstants(cmd, bc.data());
            for (int i = 0; i < 2; ++i) {
                const VkStencilFaceFlags face = i ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
                vkCmdSetStencilReference(cmd, face, st[i * 3]);
                vkCmdSetStencilCompareMask(cmd, face, st[i * 3 + 1]);
                vkCmdSetStencilWriteMask(cmd, face, st[i * 3 + 2]);
            }
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            VkWriteDescriptorSet writes[10]{};
            for (uint32_t i = 0; i < 8; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &buffers[i];
            }
            writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[8].dstBinding = 8;
            writes[8].descriptorCount = kTextures;
            writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[8].pImageInfo = images.data();
            if (cube.imageView) {
                writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[9].dstBinding = 9;
                writes[9].descriptorCount = 1;
                writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[9].pImageInfo = &cube;
            }
            pushSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, cube.imageView ? 10 : 9, writes);
            if (indexed) {
                vkCmdBindIndexBuffer(cmd, index.buffer, index.offset + index.byteOffset, indexType);
                vkCmdDrawIndexed(cmd, count, 1, firstIndex, 0, 0);
            } else {
                vkCmdDraw(cmd, count, 1, 0, 0);
            }
        });
        return true;
    }

    std::string lastDrawInfo() const override { return info_; }

private:
    struct PairKey {
        int role;
        const void *vs, *ps;
        bool operator==(const PairKey&) const = default;
    };
    struct PairHash {
        size_t operator()(const PairKey& k) const {
            return std::hash<const void*>()(k.vs) * 31 ^
                   std::hash<const void*>()(k.ps) ^ size_t(k.role);
        }
    };
    static int64_t resolve(const Draw& d, const Load& l) {
        const auto& offset = l.vs ? d.vsOffset : d.psOffset;
        return offset ? offset(l.pc) : -1;
    }
    const std::vector<int64_t>& offsets(int role, const Draw& d, std::span<const Load> vsLoads) {
        const Variant& var = kVariants[role - 1];
        const auto loadAt = [&](size_t k) -> const Load& {
            return k < vsLoads.size() ? vsLoads[k] : var.psLoads[k - vsLoads.size()];
        };
        const size_t count = vsLoads.size() + var.psLoads.size();
        const PairKey key{role, d.vsId, d.psId};
        if (!d.vsId || !d.psId || !d.vsOffset || !d.psOffset) {
            thread_local std::vector<int64_t> current;
            current.resize(count);
            for (size_t k = 0; k < count; ++k) current[k] = resolve(d, loadAt(k));
            return current;
        }
        {
            std::shared_lock lock(offsetsMutex_);
            if (auto it = offsets_.find(key); it != offsets_.end()) {
                if (kCompareOn) {
                    for (size_t k = 0; k < count; ++k) {
                        const Load& l = loadAt(k);
                        const int64_t now = resolve(d, l);
                        if (now != it->second[k] && cacheWarned_.fetch_add(1) < 32)
                            std::fprintf(stderr,
                                         "[native] character: role %d cached %s load 0x%x at %lld, now %lld\n",
                                         role, l.vs ? "vertex" : "pixel", l.pc,
                                         (long long)it->second[k], (long long)now);
                    }
                }
                return it->second;
            }
        }
        std::vector<int64_t> result(count);
        for (size_t k = 0; k < count; ++k) result[k] = resolve(d, loadAt(k));
        std::unique_lock lock(offsetsMutex_);
        return offsets_.try_emplace(key, std::move(result)).first->second;
    }

    static void describe(const uint32_t (&c)[kConstants][4],
                         const VertexBuffer* const* slots, const VertexBuffer& palette,
                         const Draw& d) {
        std::string result;
        char b[384];
        std::snprintf(b, sizeof b,
                      "vs %s; ps %s; flags 0x%x view 0x%x vertex 0x%x mask 0x%x; "
                      "swizzles 0x%x 0x%x clipZHalf %u; formats %d %d; "
                      "%s %u firstIndex %u index type %d; cull %u front %d; depth %u/%u/%d",
                      d.vsName ? d.vsName->c_str() : "?",
                      d.psName ? d.psName->c_str() : "?",
                      c[cFlags][0], c[cView + 0x22c / 16][3], c[cV40][0],
                      c[cP0][3], c[cOut][0], c[cOut][1], c[cOut][2],
                      int(d.color[0].format), int(d.color[1].format),
                      d.indexed ? "indices" : "vertices", d.count, d.firstIndex, int(d.indexType),
                      unsigned(d.cull), int(d.frontFace), d.depthStencil.depthTestEnable,
                      d.depthStencil.depthWriteEnable, int(d.depthStencil.depthCompareOp));
        result += b;
        std::snprintf(b, sizeof b, "\n    viewport %g %g %g %g; scissor %d %d %d %d",
                      d.vkViewport.x, d.vkViewport.y, d.vkViewport.width, d.vkViewport.height,
                      d.scissor[0], d.scissor[1], d.scissor[2], d.scissor[3]);
        result += b;
        for (uint32_t i = 0; i < kConstants; ++i) {
            if (!(c[i][0] | c[i][1] | c[i][2] | c[i][3])) continue;
            float f[4];
            std::memcpy(f, c[i], sizeof f);
            std::snprintf(b, sizeof b,
                          "\n    c%u: %g %g %g %g [0x%08x 0x%08x 0x%08x 0x%08x]",
                          i, f[0], f[1], f[2], f[3], c[i][0], c[i][1], c[i][2], c[i][3]);
            result += b;
        }
        for (int i = 0; i < 7; ++i) {
            const VertexBuffer& v = i == 6 ? palette : *slots[i];
            std::snprintf(b, sizeof b,
                          "\n    %s %d: pc 0x%x dfmt %u nfmt %u stride %u sel 0x%x "
                          "components %u index v%u offset %u+%u records %u range %llu",
                          i == 6 ? "palette" : "input", i, v.pc, v.dfmt, v.nfmt,
                          v.stride, v.dstSel, v.components, v.indexVgpr,
                          v.data.byteOffset, v.offset, v.records, (unsigned long long)v.data.range);
            result += b;
        }
        for (const Texture& t : d.textures) {
            std::snprintf(b, sizeof b, "\n    texture pc 0x%x: %ux%u at 0x%llx scale %g %g",
                          t.pc, t.width, t.height, (unsigned long long)t.address,
                          t.scale[0], t.scale[1]);
            result += b;
        }
        info_ = std::move(result);
    }

    bool decline(int role, const char* why) {
        std::lock_guard lock(m_);
        if (declined_[role].insert(why).second && declined_[role].size() <= 16)
            std::fprintf(stderr, "[native] character: a %s draw left to the translated renderer (%s)\n",
                         kVariants[role - 1].ps, why);
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        const auto& z = d.depthStencil;
        // Blending and stencil as the translated pipeline has them (hair blends and marks stencil).
        uint64_t blendKey = 0, stencilKey = 0;
        for (int i = 0; i < 2; ++i) {
            const VkPipelineColorBlendAttachmentState& b = d.blendState[i];
            const uint64_t v = uint64_t(b.blendEnable) | uint64_t(b.srcColorBlendFactor) << 1 | uint64_t(b.dstColorBlendFactor) << 6 |
                               uint64_t(b.colorBlendOp) << 11 | uint64_t(b.srcAlphaBlendFactor) << 14 |
                               uint64_t(b.dstAlphaBlendFactor) << 19 | uint64_t(b.alphaBlendOp) << 24;
            blendKey |= v << (32 * i);
        }
        const bool stencil = d.stencil && d.hasStencil && z.stencilTestEnable;
        if (stencil)
            for (int i = 0; i < 2; ++i) {
                const VkStencilOpState& o = i ? z.back : z.front;
                stencilKey |= uint64_t(o.failOp | o.passOp << 3 | o.depthFailOp << 6 | o.compareOp << 9) << (16 * i);
            }
        stencilKey |= uint64_t(stencil) << 40;
        const auto key = std::make_tuple(role, int(d.color[0].format), int(d.color[1].format),
                                         int(d.writeMask & 0xff), int(d.depth.format), bool(d.hasStencil),
                                         z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace), blendKey, stencilKey);
        thread_local std::remove_const_t<decltype(key)> lastKey{};
        thread_local VkPipeline lastPipe = VK_NULL_HANDLE;
        thread_local const Character* lastPass = nullptr;
        if (lastPipe && lastPass == this && lastKey == key) return lastPipe;
        std::lock_guard lock(m_);
        if (auto it = pipes_.find(key); it != pipes_.end()) {
            if (it->second) lastKey = key, lastPipe = it->second, lastPass = this;
            return it->second;
        }
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        if (!layout_) {
            VkDescriptorSetLayoutBinding bindings[10]{};
            for (uint32_t i = 0; i < 8; ++i)
                bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                               VkShaderStageFlags(i < 7 ? VK_SHADER_STAGE_VERTEX_BIT :
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT), nullptr};
            bindings[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTextures,
                           VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            bindings[9] = {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            dl.bindingCount = 10;
            dl.pBindings = bindings;
            if (!setLayout_ && vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayout_) != VK_SUCCESS)
                return out = VK_NULL_HANDLE;
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayout_;
            if (vkCreatePipelineLayout(dev, &pl, nullptr, &layout_) != VK_SUCCESS)
                return out = VK_NULL_HANDLE;
        }
        const std::string header = kHeader;
        const std::string defines = kVariants[role - 1].defines;
        const std::string vsSource = role == kO1dztRole
            ? header + "#define main characterMain\n" + kVs + kO1dztVsSuffix
            : header + defines + kVs;
        const VkShaderModule vs = gpu.shader(vsSource, Gpu::Stage::Vertex);
        const std::string fsHeader =
            "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\n" +
            header.substr(header.find('\n') + 1);
        const VkShaderModule fs = gpu.shader(fsHeader + defines + (role == kO1dztRole ? kO1dztFs : kFs), Gpu::Stage::Fragment);
        if (!vs || !fs) return out = VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = d.cull;
        rs.frontFace = d.frontFace;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = z.depthTestEnable;
        ds.depthWriteEnable = z.depthWriteEnable;
        ds.depthCompareOp = z.depthCompareOp;
        if (stencil) ds.stencilTestEnable = VK_TRUE, ds.front = z.front, ds.back = z.back;
        VkPipelineColorBlendAttachmentState attachments[2] = {d.blendState[0], d.blendState[1]};
        attachments[0].colorWriteMask = d.writeMask & 0xf;
        attachments[1].colorWriteMask = (d.writeMask >> 4) & 0xf;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 2;
        blend.pAttachments = attachments;
        const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                          VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                          VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 6;
        dy.pDynamicStates = dynamic;
        const VkFormat colors[] = {d.color[0].format, d.color[1].format};
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 2;
        rendering.pColorAttachmentFormats = colors;
        rendering.depthAttachmentFormat = d.depth.format;
        if (d.hasStencil) rendering.stencilAttachmentFormat = d.depth.format;
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.pNext = &rendering;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &blend;
        pi.pDynamicState = &dy;
        pi.layout = layout_;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, nullptr, &out) != VK_SUCCESS)
            out = VK_NULL_HANDLE;
        if (out) lastKey = key, lastPipe = out, lastPass = this;
        return out;
    }

    std::map<std::tuple<int, int, int, int, int, bool, VkBool32, VkBool32, int, int, int, uint64_t, uint64_t>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::mutex m_;
    std::set<std::string> declined_[kVariantCount + 1];
    std::atomic<bool> logged_[kVariantCount + 1]{};
    std::shared_mutex offsetsMutex_;
    std::unordered_map<PairKey, std::vector<int64_t>, PairHash> offsets_;
    std::atomic<int> cacheWarned_{0};
    static inline const bool kCompareOn = std::getenv("LPN_NATIVE_COMPARE") != nullptr;
    static inline thread_local std::string info_;
};

} // namespace

Pass* characterPass() {
    static Character pass;
    return &pass;
}

} // namespace host::native
