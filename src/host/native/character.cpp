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
};

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
    p3 = vec4(away,lp.x);
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
    vec3 baseSquared=mul(base.rgb,base.rgb);
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
    o0=swz(half4(vec4(color,alpha)),so.x);
    o1=swz(half4(vec4(0.0,0.0,0.0,alpha)),so.y);
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
    vec3 specMaterial = at(1,uv).rgb;    // 1d0 / 158
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
#ifdef SKIN
    float red = difference.r;
    red = mac(difference.r,N.y,red);
    ambient.r = mac(0.5,red,ambient.r);
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
    vec4 control = at(3,uv);              // c3c / a04
    vec4 base = at(4,uv);                 // d6c / a30
    // The image at this PC supplies its own entry s[4:7] sampler.
    vec4 screen = at(5,rasterUV);         // da0 / 90c
    float invScreen = 1.0/F(P10,8);
    vec3 indirectDiffuse = mac(screen.rgb,invScreen,pointDiffuse);
    vec3 indirectSpecular = mac(invScreen,vec3(screen.a),pointSpecular);
    float materialShadow = mac(control.r,sub(combinedShadow,1.0),1.0);
    vec3 primary = mul(light0,nl0);
    primary = mul(materialShadow,primary);
    vec3 secondary = mul(light1,nl1);
    vec3 directionalDiffuse = mac(secondaryGain,secondary,primary);
    vec3 diffuseLighting = add(ambient,directionalDiffuse);

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
    vec3 effectColor = mul(F3(P30,0x100),control.g);
    baseSquared = mac(effect,effectColor,baseSquared);

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

    vec3 diffuseMultiplier = mul(F3(P70,0x90),specMaterial.g);
    diffuseMultiplier = mul(F3(P60,0),diffuseMultiplier);
    vec3 diffuseMaterial = mul(diffuseMultiplier,baseSquared);
    vec3 linearColor = mul(diffuseMaterial,diffuseLighting);
    linearColor = mac(specularLighting,specMaterial.r,linearColor);
    linearColor = mul(occlusion,linearColor);
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
            normalEdge = mul(mac(-control.b,F(P0,0x54),1.0),normalEdge);
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
            float widthRamp = mac(-control.b,F(P0,0x58),1.0);
            widthRamp = mul(widthRamp,nearRamp);
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
        for (int i = 0; i < 4; ++i)
            if (named(psName, kVariants[i].ps)) return i + 1;
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        if (role < 1 || role > 4) return false;
        const Variant& var = kVariants[role - 1];
        if (!d.vsName || !named(*d.vsName, var.vs))
            return decline(role, "another vertex shader");
        if (d.upscaler) return decline(role, "an upscaler is on");
        if (d.pixelPositions) return decline(role, "positions in pixels");
        if (d.depthBias) return decline(role, "a polygon offset");
        if (d.colorCount != 2 || !d.color[0].view || !d.color[1].view || !d.depth.view)
            return decline(role, "not two colour targets with depth");
        if (d.blend || d.stencil || d.depthStencil.stencilTestEnable)
            return decline(role, "blending or stencil");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        if (d.indexed && (!d.index.buffer ||
                         (d.indexType != VK_INDEX_TYPE_UINT16 && d.indexType != VK_INDEX_TYPE_UINT32)))
            return decline(role, "an unsupported index buffer");

        uint32_t c[kConstants][4] = {};
        uint64_t loaded[(kConstants * 4 + 63) / 64] = {};
        const auto& offs = offsets(role, d);
        const size_t vsCount = var.vsLoads.size();
        for (size_t k = 0; k < vsCount + var.psLoads.size(); ++k) {
            const Load& l = k < vsCount ? var.vsLoads[k] : var.psLoads[k - vsCount];
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
        for (int i = 0; i < 2; ++i) {
            int32_t sw[4];
            swizzle(d.color[i], sw);
            c[cOut][i] = uint32_t(sw[0] | sw[1] << 2 | sw[2] << 4 | sw[3] << 6);
        }
        c[cOut][2] = d.clipZHalf ? 1 : 0;

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
        gpu.beginRendering(d.color, 2, d.depth, d.hasStencil);
        gpu.record([pipe, layout = layout_, buffers, images, viewport, scissor,
                    index = d.index, firstIndex = d.firstIndex, indexed = d.indexed,
                    indexType = d.indexType, count = d.count,
                    pushSet = gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            VkWriteDescriptorSet writes[9]{};
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
            pushSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 9, writes);
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
    const std::vector<int64_t>& offsets(int role, const Draw& d) {
        const Variant& var = kVariants[role - 1];
        const auto loadAt = [&](size_t k) -> const Load& {
            return k < var.vsLoads.size() ? var.vsLoads[k] : var.psLoads[k - var.vsLoads.size()];
        };
        const size_t count = var.vsLoads.size() + var.psLoads.size();
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
        const auto key = std::make_tuple(role, int(d.color[0].format), int(d.color[1].format),
                                         int(d.writeMask & 0xff), int(d.depth.format), bool(d.hasStencil),
                                         z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace));
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
            VkDescriptorSetLayoutBinding bindings[9]{};
            for (uint32_t i = 0; i < 8; ++i)
                bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                               VkShaderStageFlags(i < 7 ? VK_SHADER_STAGE_VERTEX_BIT :
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT), nullptr};
            bindings[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTextures,
                           VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = 9;
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
        const VkShaderModule vs = gpu.shader(header + defines + kVs, Gpu::Stage::Vertex);
        const std::string fsHeader =
            "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\n" +
            header.substr(header.find('\n') + 1);
        const VkShaderModule fs = gpu.shader(fsHeader + defines + kFs, Gpu::Stage::Fragment);
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
        VkPipelineColorBlendAttachmentState attachments[2]{};
        attachments[0].colorWriteMask = d.writeMask & 0xf;
        attachments[1].colorWriteMask = (d.writeMask >> 4) & 0xf;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 2;
        blend.pAttachments = attachments;
        const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
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

    std::map<std::tuple<int, int, int, int, int, bool, VkBool32, VkBool32, int, int, int>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::mutex m_;
    std::set<std::string> declined_[5];
    std::atomic<bool> logged_[5]{};
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
