// The material pass's largest pair: r_b1dst_a(vr)2d.fxo (vertex) with r_o1dst_m(vr)2ds.fxo (pixel),
// about 570 of the frame's 3,350 forward material draws, rebuilt from the disassembly (Codex's
// reading in docs/shaders/r_o1dst_m(vr)2ds.md, checked against it here).
//
// Vertex: the position as the prepass computes it (the same operations in the same order: the
// depth this pass tests against is the prepass's), vertex fog (transmission and scattering), the
// world-space normal and tangent, baked light, two sets of coordinates. Pixel: a normal map in the
// tangent frame, two directional lights with shadow and tint, up to four point lights, screen-space
// lighting and shadow, the tone curve, distance fog, and a screen-space edge mask into target 1.
//
// Every constant comes from the game's own loads (by their offsets in the shaders), uploaded per
// draw; LPN_NATIVE_COMPARE=material checks the result against the translated shaders.
#include "host/native/native.h"

#include <array>
#include <span>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <set>
#include <string>
#include <tuple>

namespace host::native {

namespace {

// The constants, as vec4s (indices into the buffer the shaders read).
enum : uint32_t {
    // vertex
    cWorld = 0,     // 3 rows, translation in w (VSV[0x20])
    cOffset = 3,    // 0x170 of VSV[0x10]
    cOffset2 = 4,   // 0x160 of VSV[0x10]: added to the lighting position with flag 0x80000
    cCamera = 5,    // 0x220 of VSV[0x30]
    cProj = 6,      // 4 rows: x, y, z (0x0 of VSV[0x30]), w (0x30)
    cRot = 10,      // 3 rows (0x40 of VSV[0x30]): loaded with the projection's w row
    cUvOffset = 13, // 0x0 of VSV[0x00]: uv0 xy, uv1 zw
    cVsMisc = 14,   // 0x18, 0x1c of VSV[0x00]; z: the vertex flags (0x0 of VSV[0x40]) as bits
    cFog = 15,      // 4: 0x40..0x7f of the buffer in s[4:7]
    cFogDir = 19,   // 0x80 of the buffer in s[8:11]
    // pixel
    cPsFlags = 20,   // x: pixel flags (bits), yz: raster to uv (PSView 0x218), w: PSView 0x22c (bits)
    cUvScale = 21,   // PSV[0x70] 0x40: specular uv scale xy, normal uv scale zw
    cShadowMap = 22, // PSV[0x10] 0x90
    cTintBase = 23,  // PSV[0x30] 0x70
    cTint = 24,      // PSV[0x10] 0xb0
    cPointPos = 25,  // 4: PSV[0x30] 0xa0..
    cPointColor = 29, // 4: PSV[0x30] 0x20..
    cPsCamera = 33,  // PSView 0x220
    cMul = 34,       // 2: PSV[0x60] 0x0 (diffuse multiplier, alpha factor 1, specular multiplier, ambient gain)
    cMisc = 36,      // PSV[0x00] 0x4c, PSV[0x30] 0xf0, PSV[0x50] 0x1c, PSV[0x70] 0x9c
    cMaterial = 37,  // 2: PSV[0x70] 0x90 (diffuse colour, alpha factor 2, specular colour, exponent)
    cExposure = 39,  // PSV[0x10] 0x0
    cAmbient = 40,   // PSV[0x10] 0x100
    cDir = 41,       // 2: PSV[0x30] 0x80, 0x90
    cAmbientTint = 43, // PSV[0x70] 0xb0
    cMisc2 = 44,     // PSV[0x70] 0x50 xy, PSV[0x60] 0xc, PSV[0x00] 0xc (bits)
    cLight = 45,     // 2: PSV[0x30] 0x0, 0x10
    cDistFog = 47,   // 2: PSV[0x10] 0x10..0x2f
    cOutScale = 49,  // PSV[0x10] 0x30
    cMask0 = 50,     // PSV[0x00] 0x2c, 0x3c, 0x2bc, 0x2b8
    cMask1 = 51,     // PSV[0x00] 0x2d4..0x2e0
    cMask2 = 52,     // PSV[0x00] 0x2c0, 0x2c4, 0x2b0, 0x2b4
    cMask3 = 53,     // PSV[0x00] 0x2f0..0x2fc
    cMask4 = 54,     // PSV[0x00] 0x2c8, 0x2cc, 0x2d0, 0x2ec
    cMask5 = 55,     // PSV[0x00] 0x2e4, 0x2e8
    cBlend = 56,     // PSView 0x230
    cTexScale = 57,  // 5: the textures' coordinate scales (native::Texture::scale), two a vec4
    cFetch = 62,     // 7: the vertex inputs (first word, stride in words, dfmt | nfmt << 4)
    cOut = 69,       // the targets' swizzles, 1 to map clip z from [-w, w]
    cUvBlock = 70,   // (SIMPLE_MATERIAL: 70.x is the alpha test's reference, PSV[0x70] 0x80)   // 2 (dst variants): PSV[0x70] 0x40..0x5f; 71: specular detail xy, normal detail zw
    kConstants = 72,
};

// A scalar load of the game's shaders and where its dwords go: vec4 `index`, from component `at`.
struct Load {
    bool vs;
    uint32_t pc, n, index, at;
};
constexpr Load kVsLoads[] = {
    {true, 0x1c, 4, cOffset, 0},
    {true, 0x20, 8, cWorld, 0},
    {true, 0x24, 4, cWorld + 2, 0},
    {true, 0x28, 1, cVsMisc, 2},
    {true, 0xb8, 4, cCamera, 0},
    {true, 0xdc, 4, cOffset2, 0},
    {true, 0x118, 16, cProj + 3, 0},
    {true, 0x168, 16, cFog, 0},
    {true, 0x16c, 4, cFogDir, 0},
    {true, 0x2fc, 8, cProj, 0},
    {true, 0x300, 4, cProj + 2, 0},
    {true, 0x304, 2, cVsMisc, 0},
    {true, 0x308, 4, cUvOffset, 0},
};

// Each pixel shader's loads (the variants' pcs from Codex's reading of their disassembly).
constexpr Load kPsLoadsMds[] = {
    {false, 0x18, 1, cPsFlags, 0},
    {false, 0x1c, 2, cPsFlags, 1},
    {false, 0x101c, 1, cPsFlags, 3},
    {false, 0x34, 4, cUvScale, 0},
    {false, 0x19c, 4, cShadowMap, 0},
    {false, 0x208, 4, cTintBase, 0},
    {false, 0x260, 4, cTint, 0},
    {false, 0x2cc, 4, cPointPos, 0},
    {false, 0x334, 4, cPointColor, 0},
    {false, 0x384, 4, cPointPos + 1, 0},
    {false, 0x3ec, 4, cPointColor + 1, 0},
    {false, 0x43c, 4, cPointPos + 2, 0},
    {false, 0x4a4, 4, cPointColor + 2, 0},
    {false, 0x4f4, 4, cPointPos + 3, 0},
    {false, 0x55c, 4, cPointColor + 3, 0},
    {false, 0x59c, 4, cPsCamera, 0},
    {false, 0x600, 8, cMul, 0},
    {false, 0x604, 1, cMisc, 0},
    {false, 0x670, 1, cMisc, 1},
    {false, 0xba8, 1, cMisc, 2},
    {false, 0xbac, 1, cMisc, 3},
    {false, 0x650, 8, cMaterial, 0},
    {false, 0x654, 4, cExposure, 0},
    {false, 0x660, 4, cAmbient, 0},
    {false, 0x68c, 8, cDir, 0},
    {false, 0x690, 4, cAmbientTint, 0},
    {false, 0x694, 2, cMisc2, 0},
    {false, 0xbb0, 1, cMisc2, 2},
    {false, 0xc48, 1, cMisc2, 3},
    {false, 0x730, 8, cLight, 0},
    {false, 0xb38, 8, cDistFog, 0},
    {false, 0xbb4, 4, cOutScale, 0},
    {false, 0xc38, 1, cMask0, 0},
    {false, 0xc40, 1, cMask0, 1},
    {false, 0xc44, 1, cMask0, 2},
    {false, 0xdd4, 1, cMask0, 3},
    {false, 0xc3c, 4, cMask1, 0},
    {false, 0xd74, 2, cMask2, 0},
    {false, 0xdd0, 2, cMask2, 2},
    {false, 0xea8, 4, cMask3, 0},
    {false, 0xeac, 2, cMask4, 0},
    {false, 0xeb0, 1, cMask4, 2},
    {false, 0xf6c, 1, cMask4, 3},
    {false, 0xf68, 2, cMask5, 0},
    {false, 0xfc4, 4, cBlend, 0},
};
constexpr Load kPsLoadsMdst[] = {
    {false, 0x20,   1, cPsFlags,       0},
    {false, 0x24,   2, cPsFlags,       1},
    {false, 0x3c,   4, cUvScale,       0},
    {false, 0x3c,   8, cUvBlock,       0},
    {false, 0x1b4,  4, cShadowMap,     0},
    {false, 0x220,  4, cTintBase,      0},
    {false, 0x278,  4, cTint,          0},
    {false, 0x2e4,  4, cPointPos,      0},
    {false, 0x34c,  4, cPointColor,    0},
    {false, 0x39c,  4, cPointPos + 1,  0},
    {false, 0x404,  4, cPointColor + 1,0},
    {false, 0x454,  4, cPointPos + 2,  0},
    {false, 0x4bc,  4, cPointColor + 2,0},
    {false, 0x50c,  4, cPointPos + 3,  0},
    {false, 0x574,  4, cPointColor + 3,0},
    {false, 0x5b4,  4, cPsCamera,      0},
    {false, 0x5fc,  8, cDir,           0},
    {false, 0x608,  1, cMisc,          1},
    {false, 0x740,  8, cLight,         0},
    {false, 0x758,  8, cMaterial,      0},
    {false, 0x76c,  4, cAmbientTint,   0},
    {false, 0x7b4,  4, cExposure,      0},
    {false, 0x7b8,  4, cAmbient,       0},
    {false, 0x7bc,  8, cMul,           0},
    {false, 0x7dc,  1, cMisc,          0},
    {false, 0xb40,  8, cDistFog,       0},
    {false, 0xbb0,  1, cMisc,          2},
    {false, 0xbb4,  1, cMisc,          3},
    {false, 0xbb8,  1, cMisc2,         2},
    {false, 0xbbc,  4, cOutScale,      0},
    {false, 0xc40,  1, cMask0,         0},
    {false, 0xc44,  4, cMask1,         0},
    {false, 0xc48,  1, cMask0,         1},
    {false, 0xc4c,  1, cMask0,         2},
    {false, 0xc50,  1, cMisc2,         3},
    {false, 0xd7c,  2, cMask2,         0},
    {false, 0xdd8,  2, cMask2,         2},
    {false, 0xddc,  1, cMask0,         3},
    {false, 0xeb0,  4, cMask3,         0},
    {false, 0xeb4,  2, cMask4,         0},
    {false, 0xeb8,  1, cMask4,         2},
    {false, 0xf70,  2, cMask5,         0},
    {false, 0xf74,  1, cMask4,         3},
    {false, 0xfcc,  4, cBlend,         0},
    {false, 0x1024, 1, cPsFlags,       3},
};
constexpr Load kPsLoadsBdst[] = {
    {false, 0x28,  1, cPsFlags,       0},
    {false, 0x2c,  2, cPsFlags,       1},
    {false, 0x44,  4, cUvScale,       0},
    {false, 0x44,  8, cUvBlock,       0},
    {false, 0x1cc, 4, cShadowMap,     0},
    {false, 0x238, 4, cTintBase,      0},
    {false, 0x290, 4, cTint,          0},
    {false, 0x2fc, 4, cPointPos,      0},
    {false, 0x364, 4, cPointColor,    0},
    {false, 0x3b4, 4, cPointPos + 1,  0},
    {false, 0x41c, 4, cPointColor + 1,0},
    {false, 0x46c, 4, cPointPos + 2,  0},
    {false, 0x4d4, 4, cPointColor + 2,0},
    {false, 0x524, 4, cPointPos + 3,  0},
    {false, 0x58c, 4, cPointColor + 3,0},
    {false, 0x5dc, 4, cPsCamera,      0},
    {false, 0x634, 8, cMul,           0},
    {false, 0x644, 4, cExposure,      0},
    {false, 0x65c, 8, cMaterial,      0},
    {false, 0x660, 4, cAmbientTint,   0},
    {false, 0x664, 4, cAmbient,       0},
    {false, 0x66c, 8, cLight,         0},
    {false, 0x684, 8, cDir,           0},
    {false, 0x730, 1, cMisc,          1},
    {false, 0x76c, 1, cMisc,          0},
    {false, 0xb18, 8, cDistFog,       0},
    {false, 0xb88, 1, cMisc,          2},
    {false, 0xb8c, 1, cMisc,          3},
    {false, 0xb90, 1, cMisc2,         2},
    {false, 0xb94, 4, cOutScale,      0},
    {false, 0xc18, 1, cMask0,         0},
    {false, 0xc1c, 4, cMask1,         0},
    {false, 0xc20, 1, cMask0,         1},
    {false, 0xc24, 1, cMask0,         2},
    {false, 0xc28, 1, cMisc2,         3},
    {false, 0xd54, 2, cMask2,         0},
    {false, 0xdb0, 2, cMask2,         2},
    {false, 0xdb4, 1, cMask0,         3},
    {false, 0xe88, 4, cMask3,         0},
    {false, 0xe8c, 2, cMask4,         0},
    {false, 0xe90, 1, cMask4,         2},
    {false, 0xf48, 2, cMask5,         0},
    {false, 0xf4c, 1, cMask4,         3},
    {false, 0xfa4, 4, cBlend,         0},
    {false, 0xffc, 1, cPsFlags,       3},
};
constexpr Load kPsLoadsBds[] = {
    {false, 0x20,  1, cPsFlags,       0},
    {false, 0x24,  2, cPsFlags,       1},
    {false, 0x3c,  4, cUvScale,       0},
    {false, 0x1c4, 4, cShadowMap,     0},
    {false, 0x230, 4, cTintBase,      0},
    {false, 0x288, 4, cTint,          0},
    {false, 0x2f4, 4, cPointPos,      0},
    {false, 0x35c, 4, cPointColor,    0},
    {false, 0x3ac, 4, cPointPos + 1,  0},
    {false, 0x414, 4, cPointColor + 1,0},
    {false, 0x464, 4, cPointPos + 2,  0},
    {false, 0x4cc, 4, cPointColor + 2,0},
    {false, 0x51c, 4, cPointPos + 3,  0},
    {false, 0x584, 4, cPointColor + 3,0},
    {false, 0x5c8, 4, cPsCamera,      0},
    {false, 0x610, 8, cDir,           0},
    {false, 0x620, 4, cExposure,      0},
    {false, 0x624, 4, cAmbient,       0},
    {false, 0x628, 1, cMisc,          0},
    {false, 0x62c, 4, cAmbientTint,   0},
    {false, 0x630, 1, cMisc,          1},
    {false, 0x6b4, 2, cMisc2,         0},
    {false, 0x6c0, 8, cMaterial,      0},
    {false, 0x754, 8, cMul,           0},
    {false, 0x758, 8, cLight,         0},
    {false, 0xb18, 8, cDistFog,       0},
    {false, 0xb88, 1, cMisc,          2},
    {false, 0xb8c, 1, cMisc,          3},
    {false, 0xb90, 1, cMisc2,         2},
    {false, 0xb94, 4, cOutScale,      0},
    {false, 0xc18, 1, cMask0,         0},
    {false, 0xc1c, 4, cMask1,         0},
    {false, 0xc20, 1, cMask0,         1},
    {false, 0xc24, 1, cMask0,         2},
    {false, 0xc28, 1, cMisc2,         3},
    {false, 0xd54, 2, cMask2,         0},
    {false, 0xdb0, 2, cMask2,         2},
    {false, 0xdb4, 1, cMask0,         3},
    {false, 0xe88, 4, cMask3,         0},
    {false, 0xe8c, 2, cMask4,         0},
    {false, 0xe90, 1, cMask4,         2},
    {false, 0xf48, 2, cMask5,         0},
    {false, 0xf4c, 1, cMask4,         3},
    {false, 0xfa4, 4, cBlend,         0},
    {false, 0xffc, 1, cPsFlags,       3},
};


// The pixel shader's textures, by the sample that reads each (the offset samples read the same).
enum Tex { tNormal, tShadow, tDetail, tScreenLight, tBase, tSpecular, tSpecDetail, tDepth, tScreenNormal, tNormalDetail, kTextures };

// The family's pixel shaders handled: "m" multiplies the detail textures in by the vertex's
// weight, "b" blends toward them by the detail texture's alpha too; "dst" adds a detail normal
// map blended toward (without it, slot 9 repeats the normal map, unread).
// The r_b1dst family (vertex shader r_b1dst: one set of coordinates, no vertex detail weight),
// from Codex's reading of its disassembly: no detail textures; "u" takes the specular map's
// green alone, "p1" drops pixels whose base alpha is under a reference.
constexpr Load kVsLoadsSimple[] = {
    {true, 0x01c,  4, cOffset,     0},
    {true, 0x020,  8, cWorld,      0},
    {true, 0x024,  4, cWorld + 2,  0},
    {true, 0x028,  1, cVsMisc,     2},
    {true, 0x0b8,  4, cCamera,     0},
    {true, 0x0dc,  4, cOffset2,    0},
    {true, 0x118, 16, cProj + 3,   0},
    {true, 0x168, 16, cFog,    0},
    {true, 0x16c,  4, cFogDir,     0},
    {true, 0x174,  4, cCamera,     0},
    {true, 0x2e8,  8, cProj,       0},
    {true, 0x2ec,  4, cProj + 2,   0},
    {true, 0x2f0,  1, cVsMisc,     1},
    {true, 0x2f4,  2, cUvOffset,   0},
};
constexpr Load kPsLoadsOdst[] = {
    {false, 0x030, 4, cUvScale,       0},
    {false, 0x094, 1, cPsFlags,       0},
    {false, 0x0c8, 2, cPsFlags,       1},
    {false, 0x190, 4, cShadowMap,     0},
    {false, 0x1c4, 4, cShadowMap,     0},
    {false, 0x1fc, 4, cTintBase,      0},
    {false, 0x254, 4, cTint,      0},
    {false, 0x2c0, 4, cPointPos,      0},
    {false, 0x328, 4, cPointColor,    0},
    {false, 0x378, 4, cPointPos + 1,  0},
    {false, 0x3e0, 4, cPointColor + 1,0},
    {false, 0x430, 4, cPointPos + 2,  0},
    {false, 0x498, 4, cPointColor + 2,0},
    {false, 0x4e8, 4, cPointPos + 3,  0},
    {false, 0x550, 4, cPointColor + 3,0},
    {false, 0x5cc, 4, cPsCamera,      0},
    {false, 0x5d0, 8, cDir,       0},
    {false, 0x5d4, 8, cMaterial,      0},
    {false, 0x5d8, 8, cMul,       0},
    {false, 0x5dc, 4, cExposure,      0},
    {false, 0x5e0, 4, cAmbientTint,   0},
    {false, 0x5e4, 1, cMisc,      1},
    {false, 0x5e8, 4, cAmbient,       0},
    {false, 0x5ec, 8, cLight,     0},
    {false, 0x748, 1, cMisc,      0},
    {false, 0xa7c, 8, cDistFog,       0},
    {false, 0xaf0, 1, cMisc,      2},
    {false, 0xaf4, 1, cMisc,      3},
    {false, 0xaf8, 1, cMisc2,     2},
    {false, 0xafc, 1, cPsFlags,       0},
    {false, 0xb00, 4, cOutScale,      0},
    {false, 0xb68, 2, cPsFlags,       1},
    {false, 0xb6c, 1, cMask0,     0},
    {false, 0xb70, 4, cMask1,     0},
    {false, 0xb74, 1, cMask0,     1},
    {false, 0xb78, 1, cMask0,     2},
    {false, 0xb7c, 1, cMisc2,     3},
    {false, 0xcc0, 2, cMask2,     0},
    {false, 0xd1c, 2, cMask2,     2},
    {false, 0xd20, 1, cMask0,     3},
    {false, 0xdf0, 4, cMask3,     0},
    {false, 0xdf4, 2, cMask4,     0},
    {false, 0xdf8, 1, cMask4,     2},
    {false, 0xeb4, 2, cMask5,     0},
    {false, 0xeb8, 1, cMask4,     3},
    {false, 0xf04, 1, cPsFlags,       0},
    {false, 0xf20, 4, cBlend,     0},
    {false, 0xf94, 1, cPsFlags,       3},
};
constexpr Load kPsLoadsOdut[] = {
    {false, 0x030, 4, cUvScale,       0},
    {false, 0x094, 1, cPsFlags,       0},
    {false, 0x0c8, 2, cPsFlags,       1},
    {false, 0x194, 4, cShadowMap,     0},
    {false, 0x1c8, 4, cShadowMap,     0},
    {false, 0x200, 4, cTintBase,      0},
    {false, 0x258, 4, cTint,      0},
    {false, 0x2c4, 4, cPointPos,      0},
    {false, 0x32c, 4, cPointColor,    0},
    {false, 0x37c, 4, cPointPos + 1,  0},
    {false, 0x3e4, 4, cPointColor + 1,0},
    {false, 0x434, 4, cPointPos + 2,  0},
    {false, 0x49c, 4, cPointColor + 2,0},
    {false, 0x4ec, 4, cPointPos + 3,  0},
    {false, 0x554, 4, cPointColor + 3,0},
    {false, 0x5c4, 4, cPsCamera,      0},
    {false, 0x5c8, 8, cDir,       0},
    {false, 0x5cc, 8, cMul,       0},
    {false, 0x5d0, 8, cMaterial,      0},
    {false, 0x5d4, 1, cMisc,      1},
    {false, 0x5d8, 4, cExposure,      0},
    {false, 0x5e8, 4, cAmbientTint,   0},
    {false, 0x5ec, 4, cAmbient,       0},
    {false, 0x5f0, 8, cLight,     0},
    {false, 0x62c, 1, cMisc,      0},
    {false, 0xa80, 8, cDistFog,       0},
    {false, 0xaf4, 1, cMisc,      2},
    {false, 0xaf8, 1, cMisc,      3},
    {false, 0xafc, 1, cMisc2,     2},
    {false, 0xb00, 1, cPsFlags,       0},
    {false, 0xb04, 4, cOutScale,      0},
    {false, 0xb6c, 2, cPsFlags,       1},
    {false, 0xb70, 1, cMask0,     0},
    {false, 0xb74, 4, cMask1,     0},
    {false, 0xb78, 1, cMask0,     1},
    {false, 0xb7c, 1, cMask0,     2},
    {false, 0xb80, 1, cMisc2,     3},
    {false, 0xcc4, 2, cMask2,     0},
    {false, 0xd20, 2, cMask2,     2},
    {false, 0xd24, 1, cMask0,     3},
    {false, 0xdf4, 4, cMask3,     0},
    {false, 0xdf8, 2, cMask4,     0},
    {false, 0xdfc, 1, cMask4,     2},
    {false, 0xeb8, 2, cMask5,     0},
    {false, 0xebc, 1, cMask4,     3},
    {false, 0xf08, 1, cPsFlags,       0},
    {false, 0xf24, 4, cBlend,     0},
    {false, 0xf98, 1, cPsFlags,       3},
};
constexpr Load kPsLoadsPdut[] = {
    {false, 0x030, 4, cUvScale,       0},
    {false, 0x094, 1, cPsFlags,       0},
    {false, 0x0c8, 2, cPsFlags,       1},
    {false, 0x194, 4, cShadowMap,     0},
    {false, 0x1c8, 4, cShadowMap,     0},
    {false, 0x200, 4, cTintBase,      0},
    {false, 0x258, 4, cTint,      0},
    {false, 0x2c4, 4, cPointPos,      0},
    {false, 0x32c, 4, cPointColor,    0},
    {false, 0x37c, 4, cPointPos + 1,  0},
    {false, 0x3e4, 4, cPointColor + 1,0},
    {false, 0x434, 4, cPointPos + 2,  0},
    {false, 0x49c, 4, cPointColor + 2,0},
    {false, 0x4ec, 4, cPointPos + 3,  0},
    {false, 0x554, 4, cPointColor + 3,0},
    {false, 0x5c4, 4, cPsCamera,      0},
    {false, 0x5c8, 8, cDir,       0},
    {false, 0x5cc, 8, cMul,       0},
    {false, 0x5d0, 8, cMaterial,      0},
    {false, 0x5d4, 1, cMisc,      1},
    {false, 0x5d8, 4, cExposure,      0},
    {false, 0x5e8, 4, cAmbientTint,   0},
    {false, 0x5ec, 4, cAmbient,       0},
    {false, 0x5f0, 8, cLight,     0},
    {false, 0x62c, 1, cMisc,      0},
    {false, 0xa80, 8, cDistFog,       0},
    {false, 0xad8, 1, cUvBlock,      0},
    {false, 0xafc, 1, cMisc,      3},
    {false, 0xb10, 1, cMisc,      2},
    {false, 0xb14, 1, cMisc2,     2},
    {false, 0xb18, 1, cPsFlags,       0},
    {false, 0xb1c, 4, cOutScale,      0},
    {false, 0xb88, 2, cPsFlags,       1},
    {false, 0xb8c, 1, cMask0,     0},
    {false, 0xb90, 4, cMask1,     0},
    {false, 0xb94, 1, cMask0,     1},
    {false, 0xb98, 1, cMask0,     2},
    {false, 0xb9c, 1, cMisc2,     3},
    {false, 0xce0, 2, cMask2,     0},
    {false, 0xd3c, 2, cMask2,     2},
    {false, 0xd40, 1, cMask0,     3},
    {false, 0xe10, 4, cMask3,     0},
    {false, 0xe14, 2, cMask4,     0},
    {false, 0xe18, 1, cMask4,     2},
    {false, 0xed4, 2, cMask5,     0},
    {false, 0xed8, 1, cMask4,     3},
    {false, 0xf24, 1, cPsFlags,       0},
    {false, 0xf40, 4, cBlend,     0},
    {false, 0xfb4, 1, cPsFlags,       3},
};

// Green-specular siblings of the b variants (Codex's reading): green alone from both specular maps;
// 2du keeps the normal map unflattened while its detail weight still blends the textures.
constexpr Load kPsLoadsOdutBdut[] = {
    {false, 0x028, 1, cPsFlags,        0},
    {false, 0x02c, 2, cPsFlags,        1},
    {false, 0x040, 4, cUvScale,        0},
    {false, 0x040, 8, cUvBlock,        0},
    {false, 0x1cc, 4, cShadowMap,      0},
    {false, 0x200, 4, cShadowMap,      0},
    {false, 0x238, 4, cTintBase,       0},
    {false, 0x290, 4, cTint,           0},
    {false, 0x2fc, 4, cPointPos,       0},
    {false, 0x364, 4, cPointColor,     0},
    {false, 0x3b4, 4, cPointPos + 1,   0},
    {false, 0x41c, 4, cPointColor + 1, 0},
    {false, 0x46c, 4, cPointPos + 2,   0},
    {false, 0x4d4, 4, cPointColor + 2, 0},
    {false, 0x524, 4, cPointPos + 3,   0},
    {false, 0x58c, 4, cPointColor + 3, 0},
    {false, 0x5cc, 4, cPsCamera,       0},
    {false, 0x614, 8, cDir,            0},
    {false, 0x6a8, 4, cAmbientTint,    0},
    {false, 0x6ac, 1, cMisc,           1},
    {false, 0x6e8, 8, cMaterial,       0},
    {false, 0x738, 8, cMul,            0},
    {false, 0x758, 8, cLight,          0},
    {false, 0x768, 1, cMisc,           0},
    {false, 0x7f8, 4, cExposure,       0},
    {false, 0x7fc, 4, cAmbient,        0},
    {false, 0xb10, 8, cDistFog,        0},
    {false, 0xb74, 1, cPsFlags,        0},
    {false, 0xb84, 1, cMisc,           2},
    {false, 0xb88, 1, cMisc,           3},
    {false, 0xb8c, 1, cMisc2,          2},
    {false, 0xb90, 4, cOutScale,       0},
    {false, 0xbe0, 2, cPsFlags,        1},
    {false, 0xc14, 1, cMask0,          0},
    {false, 0xc18, 4, cMask1,          0},
    {false, 0xc1c, 1, cMask0,          1},
    {false, 0xc20, 1, cMask0,          2},
    {false, 0xc24, 1, cMisc2,          3},
    {false, 0xd50, 2, cMask2,          0},
    {false, 0xdac, 2, cMask2,          2},
    {false, 0xdb0, 1, cMask0,          3},
    {false, 0xe80, 4, cMask3,          0},
    {false, 0xe84, 2, cMask4,          0},
    {false, 0xe88, 1, cMask4,          2},
    {false, 0xf44, 2, cMask5,          0},
    {false, 0xf48, 1, cMask4,          3},
    {false, 0xf8c, 1, cPsFlags,        0},
    {false, 0xfa0, 4, cBlend,          0},
    {false, 0xff8, 1, cPsFlags,        3},
};
constexpr Load kPsLoadsPdutBdut[] = {
    {false, 0x028,  1, cPsFlags,        0},
    {false, 0x02c,  2, cPsFlags,        1},
    {false, 0x040,  4, cUvScale,        0},
    {false, 0x040,  8, cUvBlock,        0},
    {false, 0x1cc,  4, cShadowMap,      0},
    {false, 0x200,  4, cShadowMap,      0},
    {false, 0x238,  4, cTintBase,       0},
    {false, 0x290,  4, cTint,           0},
    {false, 0x2fc,  4, cPointPos,       0},
    {false, 0x364,  4, cPointColor,     0},
    {false, 0x3b4,  4, cPointPos + 1,   0},
    {false, 0x41c,  4, cPointColor + 1, 0},
    {false, 0x46c,  4, cPointPos + 2,   0},
    {false, 0x4d4,  4, cPointColor + 2, 0},
    {false, 0x524,  4, cPointPos + 3,   0},
    {false, 0x58c,  4, cPointColor + 3, 0},
    {false, 0x5cc,  4, cPsCamera,       0},
    {false, 0x614,  8, cDir,            0},
    {false, 0x6a8,  4, cAmbientTint,    0},
    {false, 0x6ac,  1, cMisc,           1},
    {false, 0x6e8,  8, cMaterial,       0},
    {false, 0x738,  8, cMul,            0},
    {false, 0x758,  8, cLight,          0},
    {false, 0x768,  1, cMisc,           0},
    {false, 0x7f8,  4, cExposure,       0},
    {false, 0x7fc,  4, cAmbient,        0},
    {false, 0xb10,  8, cDistFog,        0},
    {false, 0xb68,  1, cUvBlock,        0},
    {false, 0xb88,  1, cMisc,           3},
    {false, 0xb90,  1, cPsFlags,        0},
    {false, 0xba4,  1, cMisc,           2},
    {false, 0xba8,  1, cMisc2,          2},
    {false, 0xbac,  4, cOutScale,       0},
    {false, 0xbfc,  2, cPsFlags,        1},
    {false, 0xc30,  1, cMask0,          0},
    {false, 0xc34,  4, cMask1,          0},
    {false, 0xc38,  1, cMask0,          1},
    {false, 0xc3c,  1, cMask0,          2},
    {false, 0xc40,  1, cMisc2,          3},
    {false, 0xd6c,  2, cMask2,          0},
    {false, 0xdc8,  2, cMask2,          2},
    {false, 0xdcc,  1, cMask0,          3},
    {false, 0xe9c,  4, cMask3,          0},
    {false, 0xea0,  2, cMask4,          0},
    {false, 0xea4,  1, cMask4,          2},
    {false, 0xf60,  2, cMask5,          0},
    {false, 0xf64,  1, cMask4,          3},
    {false, 0xfa8,  1, cPsFlags,        0},
    {false, 0xfbc,  4, cBlend,          0},
    {false, 0x1014, 1, cPsFlags,        3},
};
constexpr Load kPsLoadsOdutBdu[] = {
    {false, 0x018, 1, cPsFlags,        0},
    {false, 0x01c, 2, cPsFlags,        1},
    {false, 0x034, 4, cUvScale,        0},
    {false, 0x178, 4, cShadowMap,      0},
    {false, 0x1ac, 4, cShadowMap,      0},
    {false, 0x1e4, 4, cTintBase,       0},
    {false, 0x23c, 4, cTint,           0},
    {false, 0x2a8, 4, cPointPos,       0},
    {false, 0x310, 4, cPointColor,     0},
    {false, 0x360, 4, cPointPos + 1,   0},
    {false, 0x3c8, 4, cPointColor + 1, 0},
    {false, 0x418, 4, cPointPos + 2,   0},
    {false, 0x480, 4, cPointColor + 2, 0},
    {false, 0x4d0, 4, cPointPos + 3,   0},
    {false, 0x538, 4, cPointColor + 3, 0},
    {false, 0x580, 4, cPsCamera,       0},
    {false, 0x5d4, 4, cAmbientTint,    0},
    {false, 0x5e8, 2, cMisc2,          0},
    {false, 0x5f8, 8, cDir,            0},
    {false, 0x638, 8, cMaterial,       0},
    {false, 0x654, 1, cMisc,           0},
    {false, 0x69c, 8, cMul,            0},
    {false, 0x6e8, 4, cExposure,       0},
    {false, 0x710, 4, cAmbient,        0},
    {false, 0x7b0, 8, cLight,          0},
    {false, 0x7c0, 1, cMisc,           1},
    {false, 0xaf8, 8, cDistFog,        0},
    {false, 0xb58, 1, cPsFlags,        0},
    {false, 0xb68, 1, cMisc,           2},
    {false, 0xb6c, 1, cMisc,           3},
    {false, 0xb70, 1, cMisc2,          2},
    {false, 0xb74, 4, cOutScale,       0},
    {false, 0xbc4, 2, cPsFlags,        1},
    {false, 0xbf8, 1, cMask0,          0},
    {false, 0xbfc, 4, cMask1,          0},
    {false, 0xc00, 1, cMask0,          1},
    {false, 0xc04, 1, cMask0,          2},
    {false, 0xc08, 1, cMisc2,          3},
    {false, 0xd34, 2, cMask2,          0},
    {false, 0xd90, 2, cMask2,          2},
    {false, 0xd94, 1, cMask0,          3},
    {false, 0xe68, 4, cMask3,          0},
    {false, 0xe6c, 2, cMask4,          0},
    {false, 0xe70, 1, cMask4,          2},
    {false, 0xf28, 2, cMask5,          0},
    {false, 0xf2c, 1, cMask4,          3},
    {false, 0xf70, 1, cPsFlags,        0},
    {false, 0xf84, 4, cBlend,          0},
    {false, 0xfdc, 1, cPsFlags,        3},
};

struct Variant {
    const char* ps;
    const char* defines;
    const Load* loads;
    size_t loadCount;
    uint32_t texturePc[kTextures]; // 0: not read (bound to another of the draw's textures)
    bool simple = false;           // the r_b1dst vertex shader and SIMPLE_MATERIAL
};
template <size_t N> constexpr size_t count(const Load (&)[N]) { return N; }
const Variant kVariants[] = {
    {"r_o1dst_m(vr)2ds.fxo", "", kPsLoadsMds, count(kPsLoadsMds), {0x54, 0x184, 0x5f0, 0x640, 0x684, 0x8c0, 0x91c, 0xc28, 0xc30, 0x54}},
    {"r_o1dst_m(vr)2dst.fxo", "#define DETAIL_NORMAL\n", kPsLoadsMdst, count(kPsLoadsMdst), {0x74, 0x19c, 0x78c, 0x7a4, 0x7ac, 0x778, 0x75c, 0xc30, 0xc38, 0x6c}},
    {"r_o1dst_b(vr)2dst.fxo", "#define DETAIL_NORMAL\n#define DETAIL_BLEND\n", kPsLoadsBdst, count(kPsLoadsBdst), {0x7c, 0x1b4, 0x84, 0x67c, 0x618, 0x89c, 0x654, 0xc08, 0xc10, 0x74}},
    {"r_o1dst_b(vr)2ds.fxo", "#define DETAIL_BLEND\n", kPsLoadsBds, count(kPsLoadsBds), {0x6c, 0x1ac, 0x58, 0x7d8, 0x74c, 0x6b8, 0x7cc, 0xc08, 0xc10, 0x6c}},
    {"r_o1dst.fxo", "#define SIMPLE_MATERIAL\n", kPsLoadsOdst, count(kPsLoadsOdst), {0x04c, 0x178, 0, 0x714, 0x758, 0x81c, 0, 0xb8c, 0xb94, 0}, true},
    {"r_o1dut.fxo", "#define SIMPLE_MATERIAL\n#define SPECULAR_GREEN\n", kPsLoadsOdut, count(kPsLoadsOdut), {0x04c, 0x17c, 0, 0x78c, 0x7d4, 0x7e8, 0, 0xb90, 0xb98, 0}, true},
    {"r_o1dut_b(vr)2dut.fxo", "#define DETAIL_NORMAL\n#define DETAIL_BLEND\n#define SPECULAR_GREEN\n", kPsLoadsOdutBdut, count(kPsLoadsOdutBdut), {0x07c, 0x1b4, 0x084, 0x7f0, 0x770, 0x790, 0x750, 0xc04, 0xc0c, 0x074}},
    {"r_p1dut_b(vr)2dut.fxo", "#define DETAIL_NORMAL\n#define DETAIL_BLEND\n#define SPECULAR_GREEN\n#define MATERIAL_ALPHA_TEST\n", kPsLoadsPdutBdut, count(kPsLoadsPdutBdut), {0x07c, 0x1b4, 0x084, 0x7f0, 0x770, 0x790, 0x750, 0xc20, 0xc28, 0x074}},
    {"r_o1dut_b(vr)2du.fxo", "#define DETAIL_BLEND\n#define SPECULAR_GREEN\n#define KEEP_PRIMARY_NORMAL\n", kPsLoadsOdutBdu, count(kPsLoadsOdutBdu), {0x054, 0x160, 0x5e0, 0x5f0, 0x6c8, 0x778, 0x7a8, 0xbe8, 0xbf0, 0}},
    {"r_p1dut.fxo", "#define SIMPLE_MATERIAL\n#define SPECULAR_GREEN\n#define MATERIAL_ALPHA_TEST\n", kPsLoadsPdut, count(kPsLoadsPdut), {0x04c, 0x17c, 0, 0x78c, 0x7d4, 0x7e8, 0, 0xbac, 0xbb4, 0}, true},
};
constexpr int kVariantCount = 10;

constexpr const char* kHeader = R"(#version 450
layout(set = 0, binding = 7, std430) readonly buffer C { uvec4 c[]; } cb;
vec4 cf(int i) { return uintBitsToFloat(cb.c[i]); }
)";

constexpr const char* kVs = R"(
layout(set = 0, binding = 0, std430) readonly buffer V0 { uint w[]; } v0;
layout(set = 0, binding = 1, std430) readonly buffer V1 { uint w[]; } v1;
layout(set = 0, binding = 2, std430) readonly buffer V2 { uint w[]; } v2;
layout(set = 0, binding = 3, std430) readonly buffer V3 { uint w[]; } v3;
layout(set = 0, binding = 4, std430) readonly buffer V4 { uint w[]; } v4;
layout(set = 0, binding = 5, std430) readonly buffer V5 { uint w[]; } v5;
layout(set = 0, binding = 6, std430) readonly buffer V6 { uint w[]; } v6;
layout(location = 0) out vec4 p0;
layout(location = 1) out vec4 p1;
layout(location = 2) out vec4 p2;
layout(location = 3) out vec4 p3;
layout(location = 4) out vec4 p4;
layout(location = 5) out vec4 p5;
layout(location = 6) out vec4 p6;
layout(location = 7) out vec4 p7;
uint word(uint b, uint i) {
    switch (b) {
    case 0u: return v0.w[i];
    case 1u: return v1.w[i];
    case 2u: return v2.w[i];
    case 3u: return v3.w[i];
    case 4u: return v4.w[i];
    case 5u: return v5.w[i];
    default: return v6.w[i];
    }
}
// As the translator's fetch decodes elements (prepass.cpp's).
uint conv(uint raw, uint bits, uint nfmt) {
    float maxv = float((1u << bits) - 1u), half_ = float((1u << (bits - 1u)) - 1u);
    int sx = bitfieldExtract(int(raw), 0, int(bits));
    if (nfmt == 0u) return floatBitsToUint(float(raw) / maxv);
    if (nfmt == 1u) return floatBitsToUint(max(float(sx) / half_, -1.0));
    if (nfmt == 7u && bits == 16u) return floatBitsToUint(unpackHalf2x16(raw).x);
    if (nfmt == 5u) return uint(sx);
    return raw;
}
vec4 fetch(uint b) {
    uvec4 s = cb.c[62 + int(b)];
    uint i = s.x + uint(gl_VertexIndex) * s.y, dfmt = s.z & 15u, nfmt = (s.z >> 4) & 7u;
    uvec4 e = uvec4(0u, 0u, 0u, floatBitsToUint(1.0));
    if (dfmt == 4u || dfmt == 11u || dfmt == 13u || dfmt == 14u) {
        uint c_ = dfmt == 4u ? 1u : dfmt == 11u ? 2u : dfmt == 13u ? 3u : 4u;
        for (uint c = 0u; c < c_; ++c) e[c] = word(b, i + c);
    } else if (dfmt == 2u || dfmt == 5u || dfmt == 12u) {
        uint c_ = dfmt == 2u ? 1u : dfmt == 5u ? 2u : 4u;
        for (uint c = 0u; c < c_; ++c) e[c] = conv((word(b, i + c / 2u) >> (16u * (c & 1u))) & 0xffffu, 16u, nfmt);
    } else {
        uint c_ = dfmt == 1u ? 1u : dfmt == 3u ? 2u : 4u, x = word(b, i);
        for (uint c = 0u; c < c_; ++c) e[c] = conv((x >> (8u * c)) & 0xffu, 8u, nfmt);
    }
    return uintBitsToFloat(e);
}
float mac(float a, float b, float c) { precise float r = a * b + c; return r; }
float mul(float a, float b) { precise float r = a * b; return r; }
float dot3(vec4 a, vec3 x) { return mac(a.z, x.z, mac(a.y, x.y, mul(a.x, x.x))); }
void main() {
    uint flags = cb.c[14].z;
    vec4 W[3] = vec4[3](cf(0), cf(1), cf(2));
    vec3 off = cf(3).xyz, cam = cf(5).xyz;
    vec3 x = fetch(0u).xyz;
    // The position: as the prepass computes it, operation for operation.
    vec3 w;
    for (int r = 0; r < 3; ++r) {
        precise float s = W[r].w + off[r];
        s = mac(W[r].x, x.x, s);
        s = mac(W[r].y, x.y, s);
        w[r] = mac(W[r].z, x.z, s);
    }
    vec3 q = w;
    if (!((flags & 0x80000u) != 0u && (flags & 0x100000u) == 0u))
        for (int r = 0; r < 3; ++r) { precise float d = w[r] - cam[r]; q[r] = d; }
    vec3 e = vec3(dot3(cf(10), q), dot3(cf(11), q), dot3(cf(12), q));
    vec4 c;
    for (int r = 0; r < 4; ++r) { vec4 P = cf(6 + r); c[r] = mac(P.z, e.z, mac(P.y, e.y, mac(P.x, e.x, P.w))); }
    vec4 pos = c;
    if (cb.c[69].z != 0u) pos.z = (pos.z + pos.w) * 0.5;
    gl_Position = pos;
    vec3 lp = w;
    if ((flags & 0x80000u) != 0u) lp = cf(4).xyz + w;
    // Vertex fog: transmission by the clip w, scattering by the angle to the fog's direction.
    vec3 T = vec3(1.0), scatter = vec3(0.0);
    if ((flags & 0x20000u) != 0u) {
        vec4 f0 = cf(15), f1 = cf(16), f2 = cf(17), f3 = cf(18);
        T = exp2(f0.xyz * c.w * uintBitsToFloat(0x3fb8aa3bu));
        vec3 d = lp - cam;
        float cosine = -dot(cf(19).xyz, d) * inversesqrt(dot(d, d));
        float r = inversesqrt(f1.w + f2.w * cosine);
        vec3 phase = f1.xyz * (cosine * cosine + f3.w) + f2.xyz * (r * r * r);
        scatter = (vec3(f3.w) - T) * (f3.xyz * phase);
    }
    if ((flags & 0x40000u) != 0u) scatter = vec3(0.0);
    const float k = uintBitsToFloat(0x40008102u);
#ifdef SIMPLE_MATERIAL
    vec4 n = fetch(1u), t = fetch(2u), col = fetch(3u), b = fetch(5u);
#else
    vec4 n = fetch(1u), t = fetch(2u), col = fetch(3u), b = fetch(6u);
#endif
    // Rounded as the game's shader (each product, then each sum), as the prepass does.
    vec3 on = vec3(mac(k, n.x, -1.0), mac(k, n.y, -1.0), mac(k, n.z, -1.0));
    vec3 ot = vec3(mac(k, t.x, -1.0), mac(k, t.y, -1.0), mac(k, t.z, -1.0));
    float handed = mac(k, t.w, -1.0);
#ifdef SIMPLE_MATERIAL
    float flatten = 0.0;
#else
    float flatten = cf(14).x * (1.0 - clamp(n.w * uintBitsToFloat(0x3f808102u), 0.0, 1.0));
#endif
    // As the shaders round it: 1 - w, its reciprocal, the product, a quarter, squared.
    precise float bd = 1.0 - b.w;
    precise float bi = 1.0 / bd;
    precise vec3 b0 = b.xyz * bi;
    precise vec3 b1 = 0.25 * b0;
    precise vec3 baked = b1 * b1;
    vec4 uvo = cf(13);
#ifdef SIMPLE_MATERIAL
    p0 = vec4(fetch(4u).xy + uvo.xy, 0.0, 0.0);
#else
    p0 = vec4(fetch(4u).xy + uvo.xy, fetch(5u).xy + uvo.zw);
#endif
    p1 = vec4(col.rgb * T, col.a * cf(14).y);
    p2 = vec4(scatter, flatten);
    p3 = vec4(dot3(W[0], on), dot3(W[1], on), dot3(W[2], on), 0.0);
    p4 = vec4(baked, 0.0);
    p5 = vec4(T, 0.0);
    p6 = vec4(dot3(W[0], ot), dot3(W[1], ot), dot3(W[2], ot), handed);
    p7 = vec4(lp, 0.0);
}
)";

constexpr const char* kFs = R"(
layout(set = 0, binding = 8) uniform sampler2D tex[10];
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 3) in vec4 a3;
layout(location = 4) in vec4 a4;
layout(location = 5) in vec4 a5;
layout(location = 6) in vec4 a6;
layout(location = 7) in vec4 a7;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
vec2 sc(int t) { vec4 s = cf(57 + t / 2); return (t & 1) != 0 ? s.zw : s.xy; }
vec4 at(int t, vec2 uv) { return texture(tex[t], uv * sc(t)); }
bool has(uint f, uint m) { return (f & m) != 0u; }
float tone(float v, float exposure) {
    float x = 2.0 * exposure * v;
    float num = x * (uintBitsToFloat(0x3e19999au) * x + uintBitsToFloat(0x3d4ccccdu)) + uintBitsToFloat(0x3b83126fu);
    float den = x * (uintBitsToFloat(0x3e19999au) * x + 0.5) + uintBitsToFloat(0x3d75c290u);
    return sqrt(clamp(uintBitsToFloat(0x3fb0852eu) * (num / den) + uintBitsToFloat(0xbdbc49cau), 0.0, 1.0));
}
float depthAt(vec2 uv) { vec4 m = cf(50); return m.y / (at(7, uv).r + m.x); }
vec4 swz(vec4 v, uint s) { return vec4(v[s & 3u], v[(s >> 2) & 3u], v[(s >> 4) & 3u], v[(s >> 6) & 3u]); }
vec4 half4(vec4 v) { return vec4(unpackHalf2x16(packHalf2x16(v.xy)), unpackHalf2x16(packHalf2x16(v.zw))); }
void main() {
    uvec4 pf = cb.c[20];
    uint flags = pf.x;
    vec2 rasterToUV = uintBitsToFloat(pf.yz);
    vec2 uv0 = a0.xy, uv1 = a0.zw;
    vec2 ruv = gl_FragCoord.xy * rasterToUV;
    vec3 P = a7.xyz, gn = a3.xyz, tg = a6.xyz;
    vec3 bt = cross(gn, tg) * a6.w;
#ifdef SIMPLE_MATERIAL
    // No detail textures: their weight 0 leaves every step below as the shaders without them.
    float flatten = 0.0;
#else
    float flatten = a2.w;
#endif
#ifdef SIMPLE_MATERIAL
    vec4 detail4 = vec4(1.0);
    float dw = 0.0;
#elif defined(DETAIL_BLEND)
    // Toward the detail textures by their alpha too.
    vec4 detail4 = at(2, uv1);
    precise float dw = detail4.a * flatten;
#else
    vec4 detail4 = vec4(at(2, uv1).rgb, 1.0);
    float dw = flatten;
#endif
    vec2 s = at(0, uv0 * cf(21).zw).rg;
#ifdef DETAIL_NORMAL
    // Toward the detail normal map (its encoded x, y).
    vec2 sd = at(9, uv1 * cf(71).zw).rg;
    precise vec2 de = sd - s;
    precise vec2 enc = de * dw + s;
    vec2 nxy = 2.0 * enc - 1.0;
#elif defined(KEEP_PRIMARY_NORMAL)
    precise vec2 nxy = 2.0 * s - 1.0;
#else
    // Toward (0.5, 0.5) in the shader's two steps: q = 0.5 w + s, then q - w s.
    precise vec2 q = 0.5 * dw + s;
    precise vec2 q2 = dw * -s + q;
    precise vec2 nxy = 2.0 * q2 - 1.0;
#endif
    float nz = sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0));
    vec3 N = normalize(tg * nxy.x + bt * nxy.y + gn * nz);

    float shR = 1.0, shG = 1.0, cover = 1.0;
    vec4 rm = cf(22);
    if (has(flags, 8u)) {
        vec4 g = at(1, ruv);
        shR = g.r, shG = g.g, cover = g.a;
        for (int i = 0; i < 2; ++i) { shR = rm.x * shR + rm.y; shG = shG * rm.z + rm.w; }
    }
    vec3 tb = cf(23).xyz;
    vec3 dirTint = tb + (1.0 - tb) * (shR * shG);
    vec3 indTint = dirTint;
    if (has(flags, 8u)) { vec4 tt = cf(24); indTint = (tt.z + tt.w) * (dirTint * tt.x + vec3(tt.y)); }

    vec3 point = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        if (!has(flags, 0x10000u << uint(i))) continue;
        vec4 pp = cf(25 + i), pc = cf(29 + i);
        vec3 d = P - pp.xyz;
        float d2 = dot(d, d);
        float lam = clamp(-dot(N, d) * inversesqrt(d2), 0.0, 1.0);
        if (lam > 0.0) {
            float o = max(0.0, sqrt(d2) - pc.w);
            float att = clamp(uintBitsToFloat(0x3f808102u) / (1.0 + pp.w * o * o) + uintBitsToFloat(0xbb810205u), 0.0, 1.0);
            point += pc.rgb * (lam * att);
        }
    }

    vec4 camv = cf(33);
    vec3 cs = P - camv.xyz;
    float cd2 = dot(cs, cs);
    vec3 away = cs * inversesqrt(cd2);
    vec3 d0 = cf(41).xyz, d1 = cf(42).xyz;
    float nl0 = clamp(-dot(d0, N), 0.0, 1.0), nl1 = clamp(-dot(d1, N), 0.0, 1.0);
    vec3 h0 = away + d0, h1 = away + vec3(0.0, uintBitsToFloat(0xbe99999au), 0.0);
    vec4 m0 = cf(37), m1 = cf(38), mul0 = cf(34), mul1 = cf(35), misc = cf(36);
    float lobe0 = exp2(m1.w * log2(clamp(-dot(N, h0) * inversesqrt(dot(h0, h0)), 0.0, 1.0)));
    float lobe1 = exp2(m1.w * log2(clamp(-dot(N, h1) * inversesqrt(dot(h1, h1)), 0.0, 1.0)));
    vec3 lc0 = cf(45).xyz, lc1 = cf(46).xyz;
    vec3 diffuseColor = m0.xyz * mul0.xyz, specularColor = m1.xyz * mul1.xyz;
    vec3 ambientTex = cf(43).xyz * mul1.w;
    vec4 base = at(4, uv0);
#ifdef MATERIAL_ALPHA_TEST
    // Base alpha under the reference: dropped (kept for the quad's derivatives).
    if (cf(70).x > base.a) demote;
#endif
    vec3 detail = detail4.rgb;
#ifdef SPECULAR_GREEN
    vec3 specMat = vec3(at(5, uv0 * cf(21).xy).g);
#else
    vec3 specMat = at(5, uv0 * cf(21).xy).rgb;
#endif
#if defined(SIMPLE_MATERIAL)
    vec3 specDetail = vec3(1.0);
#elif defined(DETAIL_NORMAL)
    vec3 specDetail = at(6, uv1 * cf(71).xy).rgb;
#else
    vec3 specDetail = at(6, uv1 * cf(44).xy).rgb;
#endif
#ifdef SPECULAR_GREEN
    specDetail = vec3(specDetail.g);
#endif
    vec4 screenLight = at(3, ruv) / cf(39).z;
#ifdef DETAIL_BLEND
    precise vec3 baseSq = base.rgb * base.rgb;
    precise vec3 dd = detail * detail - baseSq;
    precise vec3 adjBase = dw * dd + baseSq;
    precise vec3 sdd = specDetail - specMat;
    precise vec3 adjSpec = dw * sdd + specMat;
#else
    vec3 adjBase = base.rgb * base.rgb * (1.0 - dw * (1.0 - detail * detail));
    vec3 adjSpec = specMat * (1.0 - dw * (1.0 - specDetail));
#endif
    vec3 vtint = a1.rgb * cover, itint = indTint * cover;
    vec3 bp = a4.xyz + point, T = a5.xyz;
    vec3 dirDiffuse = lc0 * dirTint * nl0 + lc1 * shG * nl1;
    // The ambient texture's term is weighted by the base colour here and again with the rest
    // (m1.s 0x800, then 0x90c): adjBase squared in all (Codex's review).
    vec3 diffuse = vtint * (ambientTex * adjBase + cf(40).xyz + diffuseColor * dirDiffuse);
    vec3 indDiffuse = T * diffuseColor * bp + screenLight.rgb * shG;
    float blendBase = misc.x;
    diffuse += (blendBase + (1.0 - blendBase) * shG) * itint * indDiffuse;
    vec3 dirSpec = lc0 * dirTint * lobe0 + lc1 * (2.0 * misc.y) * lobe1 * nl1;
    vec3 spec = vtint * specularColor * dirSpec;
    vec3 indSpec = T * specularColor * bp + vec3(screenLight.a * shG);
    spec += itint * lobe1 * indSpec;
    vec3 lin = adjBase * diffuse + adjSpec * spec;

    float exposure = cf(39).x;
    vec3 tc = vec3(tone(lin.r, exposure), tone(lin.g, exposure), tone(lin.b, exposure));
    vec3 color = tc;
    float fogAmount = 0.0;
    if (has(flags, 4u)) color += a2.rgb;
    vec4 df0 = cf(47), df1 = cf(48);
    if (has(flags, 0x400000u) && !has(flags, 4u)) {
        fogAmount = clamp(df1.z * sqrt(cd2) + df1.w, 0.0, 1.0);
        color = tc * (1.0 - fogAmount) + df0.xyz * fogAmount;
    }
    float alpha = misc.z * misc.w * mul0.w * base.a;
    if (!has(flags, 8u)) { vec4 os = cf(49); color *= os.xyz; alpha *= os.w; }

    float mask = 0.0;
    if (!has(flags, 0x2000000u)) {
        uint mf = cb.c[44].w;
        vec4 k0 = cf(50), k1 = cf(51), k2 = cf(52), k3 = cf(53), k4 = cf(54), k5 = cf(55);
        float z = depthAt(ruv);
        vec3 sn = 2.0 * at(8, ruv).rgb - 1.0;
        float dStart = k1.x, dNear = k1.y, dEnd = k1.z, minRadius = k1.w;
        float radiusFade = clamp(1.0 - (z - dStart) / (dEnd - dStart), 0.0, 1.0);
        float nearRamp = clamp((z - dNear) / (dStart - dNear), 0.0, 1.0);
        vec2 sdir = rasterToUV * vec2(sn.x, -sn.y);
        float nr = nearRamp * max(minRadius, k0.z * radiusFade);
        vec3 np = at(8, ruv + sdir * nr).rgb, nm = at(8, ruv - sdir * nr).rgb;
        float rp = dot(sn, 2.0 * np - 1.0), rmn = dot(sn, 2.0 * nm - 1.0);
        float normalEdge = 0.0, depthEdge = 0.0, facingMask = 0.0;
        const float eps = uintBitsToFloat(0x3b808081u);
        if (has(mf, 4u)) {
            float ag = 0.25 * (rp + 1.0) * (rmn + 1.0);
            float ag2 = ag * ag;
            float passed = ag2 + eps >= 1.0 - k2.y * nearRamp ? 1.0 : 0.0;
            float sup = k2.x + (1.0 - k2.x) * clamp(passed + ag2, 0.0, 1.0);
            normalEdge = nearRamp * (1.0 - sup);
        }
        if (has(mf, 8u)) {
            float dr = nearRamp * max(minRadius, k2.z * radiusFade);
            float dp = depthAt(ruv + sdir * dr), dm = depthAt(ruv - sdir * dr);
            float resp = 1.0 - 0.5 * (-2.0 * z + dp + dm);
            float passed = resp + eps >= 1.0 - k0.w * nearRamp ? 1.0 : 0.0;
            float sup = k2.w + (1.0 - k2.w) * clamp(passed + resp, 0.0, 1.0);
            depthEdge = nearRamp * (1.0 - sup);
        }
        if (has(mf, 2u)) {
            float facing = clamp(-dot(N, away), 0.0, 1.0);
            float far_ = k3.x, near_ = min(k3.y, far_ + uintBitsToFloat(0xba83126fu));
            float ramp = clamp((z - near_) / (far_ - near_), 0.0, 1.0);
            float width = k3.z + ramp * (k4.y - k3.z);
            float nf = clamp((facing + eps) / (nearRamp * width + eps), 0.0, 1.0);
            float fs = clamp(k4.x + (1.0 - k4.x) * nf * nf, 0.0, 1.0);
            float cand = nearRamp * (1.0 - fs);
            float thr = k3.w + ramp * (k4.z - k3.w);
            float resp = 1.0 - rp * rmn;
            facingMask = thr > 0.0 ? cand * (resp >= thr ? 1.0 : 0.0) : cand;
        }
        mask = max(depthEdge, max(facingMask, normalEdge));
        mask *= max(1.0 - k5.y, clamp(1.0 - (z - dStart) / (k5.x - dStart), 0.0, 1.0));
        mask *= 1.0 - fogAmount;
        mask *= k4.w <= mask ? 1.0 : 0.0;
    }

    float alpha1 = alpha;
    if (has(flags, 0x800000u)) {
        vec4 bl = cf(56);
        color = bl.xyz + bl.w * color - bl.xyz * bl.w;
        mask *= bl.w;
        alpha1 = alpha * bl.w;
    }
    float alpha0 = has(pf.w, 1u) ? clamp(uintBitsToFloat(0x3a4d9a68u) * gl_FragCoord.z, 0.0, 1.0) : alpha;
    uvec4 so = cb.c[69];
    o0 = swz(half4(vec4(color, alpha0)), so.x);
    o1 = swz(half4(vec4(mask, 0.0, 0.0, alpha1)), so.y);
}
)";

bool decodable(const VertexBuffer& v) {
    static constexpr uint32_t kDfmts[] = {1, 2, 3, 4, 5, 10, 11, 12, 13, 14};
    bool dfmt = false;
    for (uint32_t f : kDfmts) dfmt = dfmt || v.dfmt == f;
    const bool nfmt = v.nfmt == 0 || v.nfmt == 1 || v.nfmt == 7;
    return dfmt && nfmt && v.stride % 4 == 0 && (v.data.byteOffset + v.offset) % 4 == 0 && v.indexVgpr == 0;
}

class Material final : public Pass {
public:
    const char* name() const override { return "material"; }

    int claims(const std::string& psName) const override {
        // An archive name, with " +N" when N other entries have the same code (shader_names.cpp).
        for (int i = 0; i < kVariantCount; ++i) {
            const std::string name = kVariants[i].ps;
            if (psName == name || psName.rfind(name + " +", 0) == 0) return i + 1;
        }
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        if (role < 1 || role > kVariantCount) return false;
        const Variant& var = kVariants[role - 1];
        const char* vsWanted = var.simple ? "r_b1dst.fxo" : "r_b1dst_a(vr)2d.fxo";
        if (!d.vsName || (*d.vsName != vsWanted && d.vsName->rfind(std::string(vsWanted) + " +", 0) != 0))
            return decline(role, "another vertex shader");
        if (d.upscaler) return decline(role, "an upscaler is on");
        if (d.pixelPositions) return decline(role, "positions in pixels");
        if (d.depthBias) return decline(role, "a polygon offset");
        if (d.colorCount != 2 || !d.color[0].view || !d.color[1].view || !d.depth.view)
            return decline(role, "not two colour targets with depth");
        if (d.blend || d.stencil) return decline(role, "blending or stencil");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        uint32_t c[kConstants][4] = {};
        // The loads' offsets in the shaders' data, resolved once per shader pair.
        const Load* vsList = var.simple ? kVsLoadsSimple : kVsLoads;
        const size_t vsCount = var.simple ? count(kVsLoadsSimple) : count(kVsLoads);
        const std::vector<int64_t>& offs = offsets(role, d, vsList, vsCount, var.loads, var.loadCount);
        for (size_t k = 0; k < vsCount + var.loadCount; ++k) {
            const Load& l = k < vsCount ? vsList[k] : var.loads[k - vsCount];
            const uint32_t* data = l.vs ? d.vsData : d.psData;
            const size_t size = l.vs ? d.vsSize : d.psSize;
            if (offs[k] < 0 || size_t(offs[k]) + l.n > size) {
                static thread_local char why[64];
                std::snprintf(why, sizeof why, "no %s load at 0x%x", l.vs ? "vertex" : "pixel", l.pc);
                return decline(role, why);
            }
            for (uint32_t i = 0; i < l.n; ++i) c[l.index + (l.at + i) / 4][(l.at + i) % 4] = data[offs[k] + i];
        }
        // The vertex inputs: v4 position, v8 normal, v12 tangent, v16 colour, v20 uv0, v24 uv1, v28 baked light.
        // r_b1dst: v4 position, v8 normal (xyz), v12 tangent, v16 colour, v20 uv, v24 baked light.
        static constexpr uint32_t kUsedFull[7] = {3, 4, 4, 4, 2, 2, 4}, kUsedSimple[7] = {3, 3, 4, 4, 2, 4, 0};
        const uint32_t* kUsed = var.simple ? kUsedSimple : kUsedFull;
        const int inputs = var.simple ? 6 : 7;
        const VertexBuffer* slot[7] = {};
        for (const VertexBuffer& v : d.vertex)
            if (v.slot >= 0 && v.slot < 7 && v.components) slot[v.slot] = &v;
        for (int i = 0; i < inputs; ++i) {
            const VertexBuffer* v = slot[i];
            if (!v) return decline(role, "a vertex input missing");
            bool plain = v->components >= kUsed[i];
            for (uint32_t k = 0; k < kUsed[i]; ++k) plain = plain && ((v->dstSel >> (3 * k)) & 7) == 4 + k;
            if (!decodable(*v) || !plain) {
                char why[112];
                std::snprintf(why, sizeof why, "vertex input %d format dfmt %u nfmt %u stride %u sel 0x%x components %u", i, v->dfmt,
                              v->nfmt, v->stride, v->dstSel, v->components);
                return decline(role, why);
            }
            c[cFetch + i][0] = (v->data.byteOffset + v->offset) / 4, c[cFetch + i][1] = v->stride / 4;
            c[cFetch + i][2] = v->dfmt | v->nfmt << 4;
        }
        if (slot[0]->nfmt != 7 || (slot[0]->dfmt != 13 && slot[0]->dfmt != 14 && slot[0]->dfmt != 12))
            return decline(role, "a position format it does not read exactly");
        VkDescriptorImageInfo images[kTextures]{};
        for (int t = 0; t < kTextures; ++t) {
            const Texture* x = nullptr;
            const uint32_t pc = var.texturePc[t] ? var.texturePc[t] : var.texturePc[tBase]; // unread: the base's
            for (const Texture& y : d.textures)
                if (y.pc == pc) x = &y;
            if (!x || !x->view || !x->sampler) {
                char why[48];
                std::snprintf(why, sizeof why, "no texture at 0x%x", var.texturePc[t]);
                return decline(role, why);
            }
            images[t] = {x->sampler, x->view, VK_IMAGE_LAYOUT_GENERAL};
            std::memcpy(&c[cTexScale + t / 2][(t & 1) * 2], x->scale, 8);
        }
        for (int i = 0; i < 2; ++i) {
            int32_t sw[4];
            swizzle(d.color[i], sw);
            c[cOut][i] = uint32_t(sw[0] | sw[1] << 2 | sw[2] << 4 | sw[3] << 6);
        }
        c[cOut][2] = d.clipZHalf ? 1 : 0;
        Buffer consts;
        if (!gpu.upload(c, sizeof c, consts)) return decline(role, "no room for the constants");
        VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role].exchange(true)) {
            std::fprintf(stderr, "[native] material: drawing %s natively (targets 0x%" PRIx64 ", 0x%" PRIx64 ")\n", var.ps,
                         d.color[0].address, d.color[1].address);
        }
        gpu.beginRendering(d.color, 2, d.depth, d.hasStencil);
        VkDescriptorBufferInfo bufs[8];
        for (int i = 0; i < 7; ++i) {
            const VertexBuffer* v = slot[i] && i < inputs ? slot[i] : slot[0]; // an unread binding: the position's
            bufs[i] = {v->data.buffer, v->data.offset, v->data.range};
        }
        bufs[7] = {consts.buffer, consts.offset, consts.range};
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Buffer index = d.index;
        std::array<VkDescriptorImageInfo, kTextures> imgs;
        std::memcpy(imgs.data(), images, sizeof images);
        std::array<VkDescriptorBufferInfo, 8> bs;
        std::memcpy(bs.data(), bufs, sizeof bufs);
        gpu.record([pipe, layout = layout_, bs, imgs, vp, sc, index, indexed = d.indexed, indexType = d.indexType, count = d.count,
                    pushSet = gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            VkWriteDescriptorSet w[9]{};
            for (uint32_t i = 0; i < 8; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo = &bs[i];
            }
            w[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[8].dstBinding = 8;
            w[8].descriptorCount = kTextures;
            w[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[8].pImageInfo = imgs.data();
            pushSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 9, w);
            if (indexed) {
                vkCmdBindIndexBuffer(cmd, index.buffer, index.offset + index.byteOffset, indexType);
                vkCmdDrawIndexed(cmd, count, 1, 0, 0, 0);
            } else {
                vkCmdDraw(cmd, count, 1, 0, 0);
            }
        });
        return true;
    }

private:
    // Per shader pair (and role): each load's offset in the shader's data, or -1.
    struct PairKey {
        int role;
        const void *vs, *ps;
        bool operator==(const PairKey&) const = default;
    };
    struct PairHash {
        size_t operator()(const PairKey& k) const {
            return std::hash<const void*>()(k.vs) * 31 ^ std::hash<const void*>()(k.ps) ^ size_t(k.role);
        }
    };
    const std::vector<int64_t>& offsets(int role, const Draw& d, const Load* vs, size_t vsCount, const Load* ps, size_t psCount) {
        const PairKey key{role, d.vsId, d.psId};
        if (!d.vsId || !d.psId) { // not cacheable: this draw's own
            thread_local std::vector<int64_t> v;
            v.resize(vsCount + psCount);
            for (size_t k = 0; k < v.size(); ++k) {
                const Load& l = k < vsCount ? vs[k] : ps[k - vsCount];
                v[k] = l.vs ? d.vsOffset(l.pc) : d.psOffset(l.pc);
            }
            return v;
        }
        {
            std::shared_lock lock(offsetsMutex_);
            if (auto it = offsets_.find(key); it != offsets_.end()) return it->second;
        }
        std::vector<int64_t> v(vsCount + psCount);
        for (size_t k = 0; k < v.size(); ++k) {
            const Load& l = k < vsCount ? vs[k] : ps[k - vsCount];
            v[k] = l.vs ? d.vsOffset(l.pc) : d.psOffset(l.pc);
        }
        std::unique_lock lock(offsetsMutex_);
        return offsets_.try_emplace(key, std::move(v)).first->second; // unordered_map: references stay valid
    }
    std::shared_mutex offsetsMutex_;
    std::unordered_map<PairKey, std::vector<int64_t>, PairHash> offsets_;

    bool decline(int role, const char* why) {
        std::lock_guard lock(m_);
        if (declined_[role].insert(why).second && declined_[role].size() <= 16)
            std::fprintf(stderr, "[native] material: a %s draw left to the translated renderer (%s)\n", kVariants[role - 1].ps, why);
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        const VkPipelineDepthStencilStateCreateInfo& z = d.depthStencil;
        const auto key = std::make_tuple(role, int(d.color[0].format), int(d.color[1].format), int(d.writeMask & 0xFF), int(d.depth.format),
                                         bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace));
        // Most draws repeat the last pipeline on their recorder thread: no lock, no lookup.
        thread_local std::remove_const_t<decltype(key)> lastKey{};
        thread_local VkPipeline lastPipe = VK_NULL_HANDLE;
        thread_local const Material* lastPass = nullptr;
        if (lastPipe && lastPass == this && lastKey == key) return lastPipe;
        std::lock_guard lock(m_);
        if (auto it = pipes_.find(key); it != pipes_.end()) {
            if (it->second) lastKey = key, lastPipe = it->second, lastPass = this;
            return it->second;
        }
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        if (!layout_) {
            VkDescriptorSetLayoutBinding b[9];
            for (uint32_t i = 0; i < 8; ++i)
                b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VkShaderStageFlags(i < 7 ? VK_SHADER_STAGE_VERTEX_BIT
                                                                                           : VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
                        nullptr};
            b[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = 9;
            dl.pBindings = b;
            if (vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayout_) != VK_SUCCESS) return out = VK_NULL_HANDLE;
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayout_;
            if (vkCreatePipelineLayout(dev, &pl, nullptr, &layout_) != VK_SUCCESS) {
                layout_ = VK_NULL_HANDLE;
                return out = VK_NULL_HANDLE;
            }
        }
        const Variant& var = kVariants[role - 1];
        const std::string header = kHeader;
        const std::string vsDefs = var.simple ? "#define SIMPLE_MATERIAL\n" : "";
        const VkShaderModule vsm = gpu.shader(header + vsDefs + kVs, Gpu::Stage::Vertex);
        // The fragment stage's extension, ahead of everything else.
        const std::string fsHeader = "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\n" + header.substr(header.find('\n') + 1);
        const VkShaderModule fsm = gpu.shader(fsHeader + var.defines + kFs, Gpu::Stage::Fragment);
        if (!vsm || !fsm) return out = VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vsm, "main", nullptr};
        st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fsm, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = d.cull;
        rs.frontFace = d.frontFace;
        rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = z.depthTestEnable;
        ds.depthWriteEnable = z.depthWriteEnable;
        ds.depthCompareOp = z.depthCompareOp;
        VkPipelineColorBlendAttachmentState att[2]{};
        att[0].colorWriteMask = d.writeMask & 0xF;
        att[1].colorWriteMask = (d.writeMask >> 4) & 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 2;
        cb.pAttachments = att;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        const VkFormat colors[2] = {d.color[0].format, d.color[1].format};
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rinfo.colorAttachmentCount = 2;
        rinfo.pColorAttachmentFormats = colors;
        rinfo.depthAttachmentFormat = d.depth.format;
        if (d.hasStencil) rinfo.stencilAttachmentFormat = d.depth.format;
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.pNext = &rinfo;
        pi.stageCount = 2;
        pi.pStages = st;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vps;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dy;
        pi.layout = layout_;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, nullptr, &out) != VK_SUCCESS) out = VK_NULL_HANDLE;
        return out;
    }

    std::map<std::tuple<int, int, int, int, int, bool, VkBool32, VkBool32, int, int, int>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::atomic<bool> logged_[kVariantCount + 1] = {};
    // Recorder threads draw concurrently: the pipelines and logs are guarded.
    std::mutex m_;
    std::set<std::string> declined_[kVariantCount + 1];
};

} // namespace

Pass* materialPass() {
    static Material pass;
    return &pass;
}

} // namespace host::native
