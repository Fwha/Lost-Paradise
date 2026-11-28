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
    cReflectVs = 72, // x: the reflection coefficient (VSV[0x10] 0x0, [ref] vertex shaders)
    cReflectPs = 73, // PSV[0x30] 0xf0 (the secondary specular gain) and 0xf4 (the reflection light's floor)
    cDiffuseView = 74, // 2: PSView 0x210; raster scale zw, camera in next xyz
    cWind = 76,     // plant: byte offset, stride, records, bound byte size of VS pc 0x50
    cStageFetch = 77, // two extra fetch slots, without moving the shared constants
    cStageVsMap = 79, // SPI_PS_INPUT_CNTL export mapping, nine inputs
    cStageDefaults = 80, // PS default mask and two packed default-value words
    kConstants = 142,
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

// The reflective [ref] family (Codex's reading): r_b1dst[ref] / r_b1dut[ref] vertex shaders (a
// Fresnel-like weight in p3.w), an environment cube map added to the lit colour.
constexpr Load kVsLoadsRefSt[] = {
    {true, 0x01c,  4, cOffset,    0},
    {true, 0x020,  8, cWorld,     0},
    {true, 0x024,  4, cWorld + 2, 0},
    {true, 0x028,  1, cVsMisc,    2},
    {true, 0x0b8,  4, cCamera,    0},
    {true, 0x0dc,  4, cOffset2,   0},
    {true, 0x118, 16, cProj + 3,  0},
    {true, 0x168, 16, cFog,       0},
    {true, 0x16c,  4, cFogDir,    0},
    {true, 0x174,  4, cCamera,    0},
    {true, 0x2f0,  8, cProj,      0},
    {true, 0x2f4,  4, cProj + 2,  0},
    {true, 0x2f8,  4, cCamera,    0},
    {true, 0x2fc,  1, cVsMisc,    1},
    {true, 0x300,  2, cUvOffset,  0},
    {true, 0x304,  1, cReflectVs, 0},
};
constexpr Load kVsLoadsRefUt[] = {
    {true, 0x01c,  4, cOffset,    0},
    {true, 0x020,  8, cWorld,     0},
    {true, 0x024,  4, cWorld + 2, 0},
    {true, 0x028,  1, cVsMisc,    2},
    {true, 0x0b8,  4, cCamera,    0},
    {true, 0x0dc,  4, cOffset2,   0},
    {true, 0x118, 16, cProj + 3,  0},
    {true, 0x168, 16, cFog,       0},
    {true, 0x16c,  4, cFogDir,    0},
    {true, 0x174,  4, cCamera,    0},
    {true, 0x2ec,  8, cProj,      0},
    {true, 0x2f0,  4, cProj + 2,  0},
    {true, 0x2f4,  4, cCamera,    0},
    {true, 0x2f8,  2, cVsMisc,    0},
    {true, 0x2fc,  2, cUvOffset,  0},
    {true, 0x300,  1, cReflectVs, 0},
};
constexpr Load kPsLoadsRefOdst[] = {
    {false, 0x034, 4, cUvScale,        0},
    {false, 0x088, 1, cPsFlags,        0},
    {false, 0x0c0, 2, cPsFlags,        1},
    {false, 0x184, 4, cShadowMap,      0},
    {false, 0x1c4, 4, cShadowMap,      0},
    {false, 0x1ec, 4, cTintBase,       0},
    {false, 0x244, 4, cTint,           0},
    {false, 0x2b0, 4, cPointPos,       0},
    {false, 0x318, 4, cPointColor,     0},
    {false, 0x368, 4, cPointPos + 1,   0},
    {false, 0x3d0, 4, cPointColor + 1, 0},
    {false, 0x420, 4, cPointPos + 2,   0},
    {false, 0x488, 4, cPointColor + 2, 0},
    {false, 0x4d8, 4, cPointPos + 3,   0},
    {false, 0x540, 4, cPointColor + 3, 0},
    {false, 0x5cc, 4, cPsCamera,       0},
    {false, 0x5d0, 8, cMaterial,       0},
    {false, 0x5d4, 4, cAmbientTint,    0},
    {false, 0x5d8, 1, cMisc,           0},
    {false, 0x6b4, 8, cDir,            0},
    {false, 0x6b8, 2, cReflectPs,      0},
    {false, 0x6bc, 8, cLight,          0},
    {false, 0x700, 8, cMul,            0},
    {false, 0x7bc, 4, cExposure,       0},
    {false, 0x7c8, 4, cAmbient,        0},
    {false, 0xb54, 8, cDistFog,        0},
    {false, 0xbc4, 1, cMisc,           2},
    {false, 0xbc8, 1, cMisc,           3},
    {false, 0xbcc, 1, cMisc2,          2},
    {false, 0xbd0, 1, cPsFlags,        0},
    {false, 0xbd4, 4, cOutScale,       0},
    {false, 0xc3c, 2, cPsFlags,        1},
    {false, 0xc40, 1, cMask0,          0},
    {false, 0xc44, 4, cMask1,          0},
    {false, 0xc48, 1, cMask0,          1},
    {false, 0xc4c, 1, cMask0,          2},
    {false, 0xc50, 1, cMisc2,          3},
    {false, 0xd94, 2, cMask2,          0},
    {false, 0xdf0, 2, cMask2,          2},
    {false, 0xdf4, 1, cMask0,          3},
    {false, 0xec8, 4, cMask3,          0},
    {false, 0xecc, 2, cMask4,          0},
    {false, 0xed0, 1, cMask4,          2},
    {false, 0xf88, 2, cMask5,          0},
    {false, 0xf8c, 1, cMask4,          3},
    {false, 0xfd8, 1, cPsFlags,        0},
    {false, 0xff4, 4, cBlend,          0},
    {false, 0x106c,1, cPsFlags,        3},
};
constexpr Load kPsLoadsRefOdut[] = {
    {false, 0x034, 4, cUvScale,        0},
    {false, 0x094, 1, cPsFlags,        0},
    {false, 0x0c0, 2, cPsFlags,        1},
    {false, 0x184, 4, cShadowMap,      0},
    {false, 0x1c4, 4, cShadowMap,      0},
    {false, 0x1ec, 4, cTintBase,       0},
    {false, 0x244, 4, cTint,           0},
    {false, 0x2b0, 4, cPointPos,       0},
    {false, 0x310, 4, cPointColor,     0},
    {false, 0x360, 4, cPointPos + 1,   0},
    {false, 0x3c0, 4, cPointColor + 1, 0},
    {false, 0x410, 4, cPointPos + 2,   0},
    {false, 0x470, 4, cPointColor + 2, 0},
    {false, 0x4c0, 4, cPointPos + 3,   0},
    {false, 0x520, 4, cPointColor + 3, 0},
    {false, 0x5bc, 4, cPsCamera,       0},
    {false, 0x5c0, 8, cMaterial,       0},
    {false, 0x5c4, 2, cReflectPs,      0},
    {false, 0x5c8, 4, cAmbientTint,    0},
    {false, 0x5cc, 1, cMisc,           0},
    {false, 0x684, 8, cDir,            0},
    {false, 0x688, 8, cMul,            0},
    {false, 0x768, 8, cLight,          0},
    {false, 0x7bc, 4, cExposure,       0},
    {false, 0x7c0, 4, cAmbient,        0},
    {false, 0xb38, 8, cDistFog,        0},
    {false, 0xba4, 1, cMisc,           2},
    {false, 0xba8, 1, cMisc,           3},
    {false, 0xbac, 1, cMisc2,          2},
    {false, 0xbb0, 4, cOutScale,       0},
    {false, 0xbb4, 1, cPsFlags,        0},
    {false, 0xc20, 2, cPsFlags,        1},
    {false, 0xc24, 4, cMask1,          0},
    {false, 0xc28, 1, cMask0,          0},
    {false, 0xc2c, 1, cMask0,          1},
    {false, 0xc30, 1, cMask0,          2},
    {false, 0xc34, 1, cMisc2,          3},
    {false, 0xd78, 2, cMask2,          0},
    {false, 0xdd0, 2, cMask2,          2},
    {false, 0xdd4, 1, cMask0,          3},
    {false, 0xeb0, 4, cMask3,          0},
    {false, 0xeb4, 2, cMask4,          0},
    {false, 0xeb8, 1, cMask4,          2},
    {false, 0xf70, 2, cMask5,          0},
    {false, 0xf74, 1, cMask4,          3},
    {false, 0xfc0, 1, cPsFlags,        0},
    {false, 0xfdc, 4, cBlend,          0},
    {false, 0x1050,1, cPsFlags,        3},
};
constexpr Load kPsLoadsRefOdu[] = {
    {false, 0x024, 1, cPsFlags,        0},
    {false, 0x030, 2, cPsFlags,        1},
    {false, 0x0ac, 4, cShadowMap,      0},
    {false, 0x0ec, 4, cShadowMap,      0},
    {false, 0x114, 4, cTintBase,       0},
    {false, 0x16c, 4, cTint,           0},
    {false, 0x1d8, 4, cPointPos,       0},
    {false, 0x240, 4, cPointColor,     0},
    {false, 0x290, 4, cPointPos + 1,   0},
    {false, 0x2f8, 4, cPointColor + 1, 0},
    {false, 0x348, 4, cPointPos + 2,   0},
    {false, 0x3b0, 4, cPointColor + 2, 0},
    {false, 0x400, 4, cPointPos + 3,   0},
    {false, 0x468, 4, cPointColor + 3, 0},
    {false, 0x508, 4, cPsCamera,       0},
    {false, 0x50c, 8, cDir,            0},
    {false, 0x510, 2, cReflectPs,      0},
    {false, 0x514, 1, cMisc,           0},
    {false, 0x630, 2, cUvScale,        0},
    {false, 0x654, 8, cMaterial,       0},
    {false, 0x6b8, 8, cMul,            0},
    {false, 0x6dc, 4, cAmbientTint,    0},
    {false, 0x6ec, 8, cLight,          0},
    {false, 0x6f0, 4, cExposure,       0},
    {false, 0x6f4, 4, cAmbient,        0},
    {false, 0xa94, 8, cDistFog,        0},
    {false, 0xb04, 1, cMisc,           2},
    {false, 0xb08, 1, cMisc,           3},
    {false, 0xb0c, 1, cMisc2,          2},
    {false, 0xb10, 1, cPsFlags,        0},
    {false, 0xb14, 4, cOutScale,       0},
    {false, 0xb7c, 2, cPsFlags,        1},
    {false, 0xb80, 1, cMask0,          0},
    {false, 0xb84, 4, cMask1,          0},
    {false, 0xb88, 1, cMask0,          1},
    {false, 0xb8c, 1, cMask0,          2},
    {false, 0xb90, 1, cMisc2,          3},
    {false, 0xcd4, 2, cMask2,          0},
    {false, 0xd30, 2, cMask2,          2},
    {false, 0xd34, 1, cMask0,          3},
    {false, 0xe08, 4, cMask3,          0},
    {false, 0xe0c, 2, cMask4,          0},
    {false, 0xe10, 1, cMask4,          2},
    {false, 0xec8, 2, cMask5,          0},
    {false, 0xecc, 1, cMask4,          3},
    {false, 0xf18, 1, cPsFlags,        0},
    {false, 0xf34, 4, cBlend,          0},
    {false, 0xfac, 1, cPsFlags,        3},
};
constexpr Load kPsLoadsRefPdut[] = {
    {false, 0x034, 4, cUvScale,        0},
    {false, 0x094, 1, cPsFlags,        0},
    {false, 0x0c0, 2, cPsFlags,        1},
    {false, 0x184, 4, cShadowMap,      0},
    {false, 0x1c4, 4, cShadowMap,      0},
    {false, 0x1ec, 4, cTintBase,       0},
    {false, 0x244, 4, cTint,           0},
    {false, 0x2b0, 4, cPointPos,       0},
    {false, 0x310, 4, cPointColor,     0},
    {false, 0x360, 4, cPointPos + 1,   0},
    {false, 0x3c0, 4, cPointColor + 1, 0},
    {false, 0x410, 4, cPointPos + 2,   0},
    {false, 0x470, 4, cPointColor + 2, 0},
    {false, 0x4c0, 4, cPointPos + 3,   0},
    {false, 0x520, 4, cPointColor + 3, 0},
    {false, 0x5bc, 4, cPsCamera,       0},
    {false, 0x5c0, 8, cMaterial,       0},
    {false, 0x5c4, 2, cReflectPs,      0},
    {false, 0x5c8, 4, cAmbientTint,    0},
    {false, 0x5cc, 1, cMisc,           0},
    {false, 0x684, 8, cDir,            0},
    {false, 0x688, 8, cMul,            0},
    {false, 0x768, 8, cLight,          0},
    {false, 0x7bc, 4, cExposure,       0},
    {false, 0x7c0, 4, cAmbient,        0},
    {false, 0xb38, 8, cDistFog,        0},
    {false, 0xb8c, 1, cUvBlock,        0},
    {false, 0xbac, 1, cMisc,           3},
    {false, 0xbc4, 1, cMisc,           2},
    {false, 0xbc8, 1, cMisc2,          2},
    {false, 0xbcc, 4, cOutScale,       0},
    {false, 0xbd0, 1, cPsFlags,        0},
    {false, 0xc3c, 2, cPsFlags,        1},
    {false, 0xc40, 4, cMask1,          0},
    {false, 0xc44, 1, cMask0,          0},
    {false, 0xc48, 1, cMask0,          1},
    {false, 0xc4c, 1, cMask0,          2},
    {false, 0xc50, 1, cMisc2,          3},
    {false, 0xd94, 2, cMask2,          0},
    {false, 0xdec, 2, cMask2,          2},
    {false, 0xdf0, 1, cMask0,          3},
    {false, 0xecc, 4, cMask3,          0},
    {false, 0xed0, 2, cMask4,          0},
    {false, 0xed4, 1, cMask4,          2},
    {false, 0xf8c, 2, cMask5,          0},
    {false, 0xf90, 1, cMask4,          3},
    {false, 0xfdc, 1, cPsFlags,        0},
    {false, 0xff8, 4, cBlend,          0},
    {false, 0x106c,1, cPsFlags,        3},
};

// r_o1dut_a(vr)2d[ref] / r_o1dut_m(vr)2d[ref].
// VS descriptor-table entries for offset/world/view/flags are shifted by 0x10 from m2ds.
// VSV[0x10] supplies the reflection coefficient; VSV[0x00] still supplies UVs and misc.
// Conditional position/fog camera loads and the unconditional reflection camera load
// must all land in cCamera so the existing nonzero merge retains the value.
constexpr Load kVsLoadsRefM2d[] = {
    {true, 0x01c,  4, cOffset,    0},
    {true, 0x020,  8, cWorld,     0},
    {true, 0x024,  4, cWorld + 2, 0},
    {true, 0x028,  1, cVsMisc,    2},
    {true, 0x0b8,  4, cCamera,    0},
    {true, 0x0dc,  4, cOffset2,   0},
    {true, 0x118, 16, cProj + 3,  0},
    {true, 0x168, 16, cFog,       0},
    {true, 0x16c,  4, cFogDir,    0},
    {true, 0x174,  4, cCamera,    0},
    {true, 0x2f4,  8, cProj,      0},
    {true, 0x2f8,  4, cProj + 2,  0},
    {true, 0x2fc,  4, cCamera,    0},
    {true, 0x300,  2, cVsMisc,    0},
    {true, 0x304,  4, cUvOffset,  0},
    {true, 0x308,  1, cReflectVs, 0},
};
constexpr Load kPsLoadsRefM2d[] = {
    {false, 0x018, 1, cPsFlags,        0},
    {false, 0x01c, 2, cPsFlags,        1},
    {false, 0x034, 4, cUvScale,        0},
    {false, 0x178, 4, cShadowMap,      0},
    {false, 0x1b8, 4, cShadowMap,      0},
    {false, 0x1e0, 4, cTintBase,       0},
    {false, 0x238, 4, cTint,           0},
    {false, 0x2a4, 4, cPointPos,       0},
    {false, 0x30c, 4, cPointColor,     0},
    {false, 0x35c, 4, cPointPos + 1,   0},
    {false, 0x3c4, 4, cPointColor + 1, 0},
    {false, 0x414, 4, cPointPos + 2,   0},
    {false, 0x47c, 4, cPointColor + 2, 0},
    {false, 0x4cc, 4, cPointPos + 3,   0},
    {false, 0x534, 4, cPointColor + 3, 0},
    {false, 0x574, 4, cPsCamera,       0},
    {false, 0x5d8, 2, cReflectPs,      0},
    {false, 0x634, 1, cMisc,           0},
    {false, 0x6ac, 8, cDir,            0},
    {false, 0x6b0, 8, cMaterial,       0},
    {false, 0x784, 8, cLight,          0},
    {false, 0x788, 8, cMul,            0},
    {false, 0x7d4, 4, cAmbientTint,    0},
    {false, 0x7fc, 4, cExposure,       0},
    {false, 0x800, 4, cAmbient,        0},
    {false, 0xbb8, 8, cDistFog,        0},
    {false, 0xc18, 1, cPsFlags,        0},
    {false, 0xc28, 1, cMisc,           2},
    {false, 0xc2c, 1, cMisc,           3},
    {false, 0xc30, 1, cMisc2,          2},
    {false, 0xc34, 4, cOutScale,       0},
    {false, 0xc84, 2, cPsFlags,        1},
    {false, 0xcb8, 1, cMask0,          0},
    {false, 0xcbc, 4, cMask1,          0},
    {false, 0xcc0, 1, cMask0,          1},
    {false, 0xcc4, 1, cMask0,          2},
    {false, 0xcc8, 1, cMisc2,          3},
    {false, 0xdf4, 2, cMask2,          0},
    {false, 0xe50, 2, cMask2,          2},
    {false, 0xe54, 1, cMask0,          3},
    {false, 0xf28, 4, cMask3,          0},
    {false, 0xf2c, 2, cMask4,          0},
    {false, 0xf30, 1, cMask4,          2},
    {false, 0xfe8, 2, cMask5,          0},
    {false, 0xfec, 1, cMask4,          3},
    {false, 0x1030,1, cPsFlags,        0},
    {false, 0x1044,4, cBlend,          0},
    {false, 0x109c,1, cPsFlags,        3},
};
// r_b1d / r_o1d[ag]: five vertex inputs, geometric normal, diffuse lighting only.
constexpr Load kVsLoadsD[] = {
    {true, 0x01c, 4, cOffset, 0},
    {true, 0x020, 8, cWorld, 0},
    {true, 0x024, 4, cWorld + 2, 0},
    {true, 0x028, 1, cVsMisc, 2},
    {true, 0x0b8, 4, cCamera, 0},
    {true, 0x0dc, 4, cOffset2, 0},
    {true, 0x118, 16, cProj + 3, 0},
    {true, 0x168, 16, cFog, 0},
    {true, 0x16c, 4, cFogDir, 0},
    {true, 0x174, 4, cCamera, 0},
    {true, 0x2cc, 4, cProj + 2, 0},
    {true, 0x30c, 8, cProj, 0},
    {true, 0x310, 1, cVsMisc, 1},
    {true, 0x314, 2, cUvOffset, 0},
};
constexpr Load kPsLoadsD[] = {
    {false, 0x014, 8, cDiffuseView, 0},
    {false, 0x054, 1, cPsFlags, 0},
    {false, 0x0f0, 4, cShadowMap, 0},
    {false, 0x124, 4, cShadowMap, 0},
    {false, 0x15c, 4, cTintBase, 0},
    {false, 0x1b4, 4, cTint, 0},
    {false, 0x208, 4, cPointPos, 0},
    {false, 0x268, 4, cPointColor, 0},
    {false, 0x2c0, 4, cPointPos + 1, 0},
    {false, 0x320, 4, cPointColor + 1, 0},
    {false, 0x378, 4, cPointPos + 2, 0},
    {false, 0x3d8, 4, cPointColor + 2, 0},
    {false, 0x430, 4, cPointPos + 3, 0},
    {false, 0x490, 4, cPointColor + 3, 0},
    {false, 0x4f4, 8, cDir, 0},
    {false, 0x4f8, 8, cLight, 0},
    {false, 0x510, 1, cMul + 1, 3},
    {false, 0x514, 4, cMul, 0},
    {false, 0x530, 4, cAmbientTint, 0},
    {false, 0x59c, 4, cMaterial, 0},
    {false, 0x5ac, 4, cAmbient, 0},
    {false, 0x5b0, 4, cExposure, 0},
    {false, 0x620, 1, cMisc, 0},
    {false, 0x858, 8, cDistFog, 0},
    {false, 0x8c0, 1, cMisc, 2},
    {false, 0x8c4, 1, cMisc, 3},
    {false, 0x8c8, 1, cMisc2, 2},
    {false, 0x8cc, 4, cOutScale, 0},
    {false, 0x8d0, 1, cPsFlags, 0},
    {false, 0x940, 2, cPsFlags, 1},
    {false, 0x944, 1, cMask0, 0},
    {false, 0x948, 4, cMask1, 0},
    {false, 0x94c, 1, cMask0, 1},
    {false, 0x950, 1, cMask0, 2},
    {false, 0x954, 1, cMisc2, 3},
    {false, 0xa90, 2, cMask2, 0},
    {false, 0xaf0, 2, cMask2, 2},
    {false, 0xaf4, 1, cMask0, 3},
    {false, 0xbbc, 4, cMask3, 0},
    {false, 0xbc0, 2, cMask4, 0},
    {false, 0xc2c, 1, cMask4, 2},
    {false, 0xc90, 2, cMask5, 0},
    {false, 0xc94, 1, cMask4, 3},
    {false, 0xce0, 1, cPsFlags, 0},
    {false, 0xcfc, 4, cBlend, 0},
    {false, 0xd60, 1, cPsFlags, 3},
};
constexpr Load kPsLoadsB1dst[] = {
    {false, 0x30, 4, cUvScale, 0},
    {false, 0x94, 1, cPsFlags, 0},
    {false, 0xc8, 2, cPsFlags, 1},
    {false, 0x190, 4, cShadowMap, 0},
    {false, 0x1c4, 4, cShadowMap, 0},
    {false, 0x1fc, 4, cTintBase, 0},
    {false, 0x254, 4, cTint, 0},
    {false, 0x2c0, 4, cPointPos, 0},
    {false, 0x328, 4, cPointColor, 0},
    {false, 0x378, 4, cPointPos + 1, 0},
    {false, 0x3e0, 4, cPointColor + 1, 0},
    {false, 0x430, 4, cPointPos + 2, 0},
    {false, 0x498, 4, cPointColor + 2, 0},
    {false, 0x4e8, 4, cPointPos + 3, 0},
    {false, 0x550, 4, cPointColor + 3, 0},
    {false, 0x5cc, 4, cPsCamera, 0},
    {false, 0x5d0, 8, cDir, 0},
    {false, 0x5d4, 8, cMaterial, 0},
    {false, 0x5d8, 8, cMul, 0},
    {false, 0x5dc, 1, cMisc, 1},
    {false, 0x5e0, 4, cExposure, 0},
    {false, 0x5e4, 4, cAmbientTint, 0},
    {false, 0x5ec, 4, cAmbient, 0},
    {false, 0x5f0, 8, cLight, 0},
    {false, 0x744, 1, cMisc, 0},
    {false, 0xa78, 8, cDistFog, 0},
    {false, 0xae0, 1, cMisc, 2},
    {false, 0xae4, 1, cMisc, 3},
    {false, 0xae8, 1, cMisc2, 2},
    {false, 0xaec, 1, cUvBlock, 0},
    {false, 0xb30, 1, cPsFlags, 0},
    {false, 0xb34, 4, cOutScale, 0},
    {false, 0xb8c, 2, cPsFlags, 1},
    {false, 0xb90, 4, cMask1, 0},
    {false, 0xb94, 1, cMask0, 2},
    {false, 0xb98, 1, cMisc2, 3},
    {false, 0xcb4, 2, cMask2, 0},
    {false, 0xd18, 4, cMask3, 0},
    {false, 0xd1c, 2, cMask4, 0},
    {false, 0xd20, 1, cMask4, 2},
    {false, 0xdd8, 2, cMask5, 0},
    {false, 0xddc, 1, cMask4, 3},
    {false, 0xe24, 1, cPsFlags, 0},
    {false, 0xe40, 4, cBlend, 0},
};
constexpr Load kPsLoadsNte[] = {
    {false, 0x40, 8, cUvBlock, 0},
    {false, 0x40, 4, cUvScale, 0},
    {false, 0xd4, 1, cPsFlags, 0},
    {false, 0x108, 2, cPsFlags, 1},
    {false, 0x1e0, 4, cShadowMap, 0},
    {false, 0x214, 4, cShadowMap, 0},
    {false, 0x24c, 4, cTintBase, 0},
    {false, 0x2a4, 4, cTint, 0},
    {false, 0x310, 4, cPointPos, 0},
    {false, 0x378, 4, cPointColor, 0},
    {false, 0x3c8, 4, cPointPos + 1, 0},
    {false, 0x430, 4, cPointColor + 1, 0},
    {false, 0x480, 4, cPointPos + 2, 0},
    {false, 0x4e8, 4, cPointColor + 2, 0},
    {false, 0x538, 4, cPointPos + 3, 0},
    {false, 0x5a0, 4, cPointColor + 3, 0},
    {false, 0x628, 1, cMisc, 1},
    {false, 0x634, 4, cAmbient, 0},
    {false, 0x638, 4, cExposure, 0},
    {false, 0x63c, 1, cMisc, 0},
    {false, 0x640, 8, cDir, 0},
    {false, 0x658, 4, cPsCamera, 0},
    {false, 0x65c, 8, cMul, 0},
    {false, 0x6d0, 8, cMaterial, 0},
    {false, 0x730, 4, cAmbientTint, 0},
    {false, 0x7e8, 8, cLight, 0},
    {false, 0xb38, 8, cDistFog, 0},
    {false, 0xbac, 1, cMisc, 2},
    {false, 0xbb0, 1, cMisc, 3},
    {false, 0xbb4, 1, cMisc2, 2},
    {false, 0xbb8, 4, cOutScale, 0},
    {false, 0xc04, 4, cBlend, 0},
    {false, 0xc64, 1, cPsFlags, 3},
};
constexpr Load kPsLoadsB2dutRef[] = {
    {false, 0x28, 1, cPsFlags, 0},
    {false, 0x2c, 2, cPsFlags, 1},
    {false, 0x44, 8, cUvBlock, 0},
    {false, 0x44, 4, cUvScale, 0},
    {false, 0x1cc, 4, cShadowMap, 0},
    {false, 0x20c, 4, cShadowMap, 0},
    {false, 0x234, 4, cTintBase, 0},
    {false, 0x28c, 4, cTint, 0},
    {false, 0x2f8, 4, cPointPos, 0},
    {false, 0x360, 4, cPointColor, 0},
    {false, 0x3b0, 4, cPointPos + 1, 0},
    {false, 0x418, 4, cPointColor + 1, 0},
    {false, 0x468, 4, cPointPos + 2, 0},
    {false, 0x4d0, 4, cPointColor + 2, 0},
    {false, 0x520, 4, cPointPos + 3, 0},
    {false, 0x588, 4, cPointColor + 3, 0},
    {false, 0x5c8, 4, cPsCamera, 0},
    {false, 0x6f4, 8, cMaterial, 0},
    {false, 0x6fc, 4, cAmbientTint, 0},
    {false, 0x704, 1, cMisc, 0},
    {false, 0x710, 8, cDir, 0},
    {false, 0x714, 2, cReflectPs, 0},
    {false, 0x718, 4, cExposure, 0},
    {false, 0x71c, 4, cAmbient, 0},
    {false, 0x7e0, 8, cLight, 0},
    {false, 0x808, 8, cMul, 0},
    {false, 0xbf4, 8, cDistFog, 0},
    {false, 0xc54, 1, cPsFlags, 0},
    {false, 0xc64, 1, cMisc, 2},
    {false, 0xc68, 1, cMisc, 3},
    {false, 0xc6c, 1, cMisc2, 2},
    {false, 0xc70, 4, cOutScale, 0},
    {false, 0xcc0, 2, cPsFlags, 1},
    {false, 0xcf4, 1, cMask0, 0},
    {false, 0xcf8, 4, cMask1, 0},
    {false, 0xcfc, 1, cMask0, 1},
    {false, 0xd00, 1, cMask0, 2},
    {false, 0xd04, 1, cMisc2, 3},
    {false, 0xe30, 2, cMask2, 0},
    {false, 0xe8c, 2, cMask2, 2},
    {false, 0xe90, 1, cMask0, 3},
    {false, 0xf64, 4, cMask3, 0},
    {false, 0xf68, 2, cMask4, 0},
    {false, 0xf6c, 1, cMask4, 2},
    {false, 0x1024, 2, cMask5, 0},
    {false, 0x1028, 1, cMask4, 3},
    {false, 0x106c, 1, cPsFlags, 0},
    {false, 0x1080, 4, cBlend, 0},
    {false, 0x10d8, 1, cPsFlags, 3},
};
constexpr Load kPsLoadsM2duRef[] = {
    {false, 0x18, 1, cPsFlags, 0},
    {false, 0x1c, 2, cPsFlags, 1},
    {false, 0x34, 4, cUvScale, 0},
    {false, 0x178, 4, cShadowMap, 0},
    {false, 0x1b8, 4, cShadowMap, 0},
    {false, 0x1e0, 4, cTintBase, 0},
    {false, 0x238, 4, cTint, 0},
    {false, 0x2a4, 4, cPointPos, 0},
    {false, 0x30c, 4, cPointColor, 0},
    {false, 0x35c, 4, cPointPos + 1, 0},
    {false, 0x3c4, 4, cPointColor + 1, 0},
    {false, 0x414, 4, cPointPos + 2, 0},
    {false, 0x47c, 4, cPointColor + 2, 0},
    {false, 0x4cc, 4, cPointPos + 3, 0},
    {false, 0x534, 4, cPointColor + 3, 0},
    {false, 0x574, 4, cPsCamera, 0},
    {false, 0x5c8, 2, cMisc2, 0},
    {false, 0x60c, 8, cMaterial, 0},
    {false, 0x6d8, 4, cAmbientTint, 0},
    {false, 0x6e0, 4, cExposure, 0},
    {false, 0x6e8, 4, cAmbient, 0},
    {false, 0x704, 8, cDir, 0},
    {false, 0x708, 8, cMul, 0},
    {false, 0x710, 1, cMisc, 0},
    {false, 0x714, 2, cReflectPs, 0},
    {false, 0x7ac, 8, cLight, 0},
    {false, 0xbe8, 8, cDistFog, 0},
    {false, 0xc48, 1, cPsFlags, 0},
    {false, 0xc58, 1, cMisc, 2},
    {false, 0xc5c, 1, cMisc, 3},
    {false, 0xc60, 1, cMisc2, 2},
    {false, 0xc64, 4, cOutScale, 0},
    {false, 0xcb4, 2, cPsFlags, 1},
    {false, 0xce8, 1, cMask0, 0},
    {false, 0xcec, 4, cMask1, 0},
    {false, 0xcf0, 1, cMask0, 1},
    {false, 0xcf4, 1, cMask0, 2},
    {false, 0xcf8, 1, cMisc2, 3},
    {false, 0xe24, 2, cMask2, 0},
    {false, 0xe80, 2, cMask2, 2},
    {false, 0xe84, 1, cMask0, 3},
    {false, 0xf58, 4, cMask3, 0},
    {false, 0xf5c, 2, cMask4, 0},
    {false, 0xf60, 1, cMask4, 2},
    {false, 0x1018, 2, cMask5, 0},
    {false, 0x101c, 1, cMask4, 3},
    {false, 0x1060, 1, cPsFlags, 0},
    {false, 0x1074, 4, cBlend, 0},
    {false, 0x10cc, 1, cPsFlags, 3},
};
constexpr Load kPsLoadsO1d[] = {
    {false, 0x20, 1, cPsFlags, 0},
    {false, 0x44, 2, cPsFlags, 1},
    {false, 0xb8, 4, cShadowMap, 0},
    {false, 0xec, 4, cShadowMap, 0},
    {false, 0x124, 4, cTintBase, 0},
    {false, 0x17c, 4, cTint, 0},
    {false, 0x1e8, 4, cPointPos, 0},
    {false, 0x250, 4, cPointColor, 0},
    {false, 0x2a0, 4, cPointPos + 1, 0},
    {false, 0x308, 4, cPointColor + 1, 0},
    {false, 0x358, 4, cPointPos + 2, 0},
    {false, 0x3c0, 4, cPointColor + 2, 0},
    {false, 0x410, 4, cPointPos + 3, 0},
    {false, 0x478, 4, cPointColor + 3, 0},
    {false, 0x4f0, 4, cPsCamera, 0},
    {false, 0x4f4, 8, cDir, 0},
    {false, 0x4f8, 8, cMaterial, 0},
    {false, 0x4fc, 4, cExposure, 0},
    {false, 0x500, 4, cAmbientTint, 0},
    {false, 0x504, 1, cMisc, 1},
    {false, 0x508, 4, cAmbient, 0},
    {false, 0x534, 8, cMul, 0},
    {false, 0x614, 8, cLight, 0},
    {false, 0x694, 1, cMisc, 0},
    {false, 0x984, 8, cDistFog, 0},
    {false, 0x9f8, 1, cMisc, 2},
    {false, 0x9fc, 1, cMisc, 3},
    {false, 0xa00, 1, cMisc2, 2},
    {false, 0xa04, 1, cPsFlags, 0},
    {false, 0xa08, 4, cOutScale, 0},
    {false, 0xa70, 2, cPsFlags, 1},
    {false, 0xa74, 1, cMask0, 0},
    {false, 0xa78, 4, cMask1, 0},
    {false, 0xa7c, 1, cMask0, 1},
    {false, 0xa80, 1, cMask0, 2},
    {false, 0xa84, 1, cMisc2, 3},
    {false, 0xbc8, 2, cMask2, 0},
    {false, 0xc24, 2, cMask2, 2},
    {false, 0xc28, 1, cMask0, 3},
    {false, 0xcf8, 4, cMask3, 0},
    {false, 0xcfc, 2, cMask4, 0},
    {false, 0xd00, 1, cMask4, 2},
    {false, 0xdbc, 2, cMask5, 0},
    {false, 0xdc0, 1, cMask4, 3},
    {false, 0xe0c, 1, cPsFlags, 0},
    {false, 0xe28, 4, cBlend, 0},
    {false, 0xe9c, 1, cPsFlags, 3},
};
constexpr Load kPsLoadsA2d[] = {
    {false, 0x30, 4, cUvScale, 0},
    {false, 0x94, 1, cPsFlags, 0},
    {false, 0xc8, 2, cPsFlags, 1},
    {false, 0x190, 4, cShadowMap, 0},
    {false, 0x1c4, 4, cShadowMap, 0},
    {false, 0x1fc, 4, cTintBase, 0},
    {false, 0x254, 4, cTint, 0},
    {false, 0x2c0, 4, cPointPos, 0},
    {false, 0x328, 4, cPointColor, 0},
    {false, 0x378, 4, cPointPos + 1, 0},
    {false, 0x3e0, 4, cPointColor + 1, 0},
    {false, 0x430, 4, cPointPos + 2, 0},
    {false, 0x498, 4, cPointColor + 2, 0},
    {false, 0x4e8, 4, cPointPos + 3, 0},
    {false, 0x550, 4, cPointColor + 3, 0},
    {false, 0x5c0, 4, cPsCamera, 0},
    {false, 0x5c4, 8, cDir, 0},
    {false, 0x5c8, 8, cMaterial, 0},
    {false, 0x5cc, 8, cMul, 0},
    {false, 0x5d0, 4, cExposure, 0},
    {false, 0x5d8, 4, cAmbientTint, 0},
    {false, 0x5dc, 1, cMisc, 1},
    {false, 0x5e4, 4, cAmbient, 0},
    {false, 0x684, 8, cLight, 0},
    {false, 0x760, 1, cMisc, 0},
    {false, 0xac8, 8, cDistFog, 0},
    {false, 0xb3c, 1, cMisc, 2},
    {false, 0xb40, 1, cMisc, 3},
    {false, 0xb44, 1, cMisc2, 2},
    {false, 0xb48, 1, cPsFlags, 0},
    {false, 0xb4c, 4, cOutScale, 0},
    {false, 0xbb4, 2, cPsFlags, 1},
    {false, 0xbb8, 1, cMask0, 0},
    {false, 0xbbc, 4, cMask1, 0},
    {false, 0xbc0, 1, cMask0, 1},
    {false, 0xbc4, 1, cMask0, 2},
    {false, 0xbc8, 1, cMisc2, 3},
    {false, 0xd0c, 2, cMask2, 0},
    {false, 0xd68, 2, cMask2, 2},
    {false, 0xd6c, 1, cMask0, 3},
    {false, 0xe3c, 4, cMask3, 0},
    {false, 0xe40, 2, cMask4, 0},
    {false, 0xe44, 1, cMask4, 2},
    {false, 0xf00, 2, cMask5, 0},
    {false, 0xf04, 1, cMask4, 3},
    {false, 0xf50, 1, cPsFlags, 0},
    {false, 0xf6c, 4, cBlend, 0},
    {false, 0xfe0, 1, cPsFlags, 3},
};

constexpr Load kVsLoadsD2uRef[] = {
    {true, 0x01c,  4, cOffset,          0},
    {true, 0x020,  8, cWorld,           0},
    {true, 0x024,  4, cWorld + 2,       0},
    {true, 0x028,  1, cVsMisc,          2},
    {true, 0x0b8,  4, cCamera,          0},
    {true, 0x0dc,  4, cOffset2,         0},
    {true, 0x118, 16, cProj + 3,        0},
    {true, 0x168, 16, cFog,             0},
    {true, 0x16c,  4, cFogDir,          0},
    {true, 0x174,  4, cCamera,          0},
    {true, 0x2e0,  8, cProj,            0},
    {true, 0x2e4,  4, cProj + 2,        0},
    {true, 0x2e8,  4, cCamera,          0},
    {true, 0x2ec,  1, cVsMisc,          1},
    {true, 0x2f0,  4, cUvOffset,        0},
    {true, 0x2f4,  1, cReflectVs,       0},
};
constexpr Load kPsLoadsD2uRef[] = {
    {false, 0x024,  1, cPsFlags,         0},
    {false, 0x030,  2, cPsFlags,         1},
    {false, 0x0ac,  4, cShadowMap,       0},
    {false, 0x0ec,  4, cShadowMap,       0},
    {false, 0x114,  4, cTintBase,        0},
    {false, 0x16c,  4, cTint,            0},
    {false, 0x1d8,  4, cPointPos,        0},
    {false, 0x240,  4, cPointColor,      0},
    {false, 0x290,  4, cPointPos + 1,    0},
    {false, 0x2f8,  4, cPointColor + 1,  0},
    {false, 0x348,  4, cPointPos + 2,    0},
    {false, 0x3b0,  4, cPointColor + 2,  0},
    {false, 0x400,  4, cPointPos + 3,    0},
    {false, 0x468,  4, cPointColor + 3,  0},
    {false, 0x50c,  4, cPsCamera,        0},
    {false, 0x510,  8, cDir,             0},
    {false, 0x514,  8, cMul,             0},
    {false, 0x518,  2, cUvScale,         0},
    {false, 0x51c,  1, cMisc,            0},
    {false, 0x520,  2, cReflectPs,       0},
    {false, 0x5e8,  4, cAmbientTint,     0},
    {false, 0x5f0,  8, cMaterial,        0},
    {false, 0x648,  4, cExposure,        0},
    {false, 0x64c,  4, cAmbient,         0},
    {false, 0x6ac,  8, cLight,           0},
    {false, 0xa98,  8, cDistFog,         0},
    {false, 0xb08,  1, cMisc,            2},
    {false, 0xb0c,  1, cMisc,            3},
    {false, 0xb10,  1, cMisc2,           2},
    {false, 0xb14,  1, cPsFlags,         0},
    {false, 0xb18,  4, cOutScale,        0},
    {false, 0xb80,  2, cPsFlags,         1},
    {false, 0xb84,  1, cMask0,           0},
    {false, 0xb88,  4, cMask1,           0},
    {false, 0xb8c,  1, cMask0,           1},
    {false, 0xb90,  1, cMask0,           2},
    {false, 0xb94,  1, cMisc2,           3},
    {false, 0xcd8,  2, cMask2,           0},
    {false, 0xd34,  2, cMask2,           2},
    {false, 0xd38,  1, cMask0,           3},
    {false, 0xe0c,  4, cMask3,           0},
    {false, 0xe10,  2, cMask4,           0},
    {false, 0xe14,  1, cMask4,           2},
    {false, 0xecc,  2, cMask5,           0},
    {false, 0xed0,  1, cMask4,           3},
    {false, 0xf1c,  1, cPsFlags,         0},
    {false, 0xf38,  4, cBlend,           0},
    {false, 0xfb0,  1, cPsFlags,         3},
};

constexpr Load kVsLoadsPlant[] = {
    {true, 0x58, 8, cWorld, 0},
    {true, 0x5c, 4, cWorld + 2, 0},
    {true, 0x60, 4, cProj + 3, 0},
    {true, 0x64, 1, cVsMisc, 2},
    {true, 0x130, 16, cFog, 0},
    {true, 0x134, 4, cFogDir, 0},
    {true, 0x13c, 4, cCamera, 0},
    {true, 0x2a0, 8, cProj, 0},
    {true, 0x2a4, 4, cProj + 2, 0},
    {true, 0x2a8, 2, cVsMisc, 0},
    {true, 0x2ac, 2, cUvOffset, 0},
};
constexpr Load kPsLoadsPlant[] = {
    {false, 0x30, 4, cUvScale, 0},
    {false, 0x94, 1, cPsFlags, 0},
    {false, 0xc8, 2, cPsFlags, 1},
    {false, 0x190, 4, cShadowMap, 0},
    {false, 0x1c4, 4, cShadowMap, 0},
    {false, 0x1fc, 4, cTintBase, 0},
    {false, 0x254, 4, cTint, 0},
    {false, 0x2c0, 4, cPointPos, 0},
    {false, 0x328, 4, cPointColor, 0},
    {false, 0x378, 4, cPointPos + 1, 0},
    {false, 0x3e0, 4, cPointColor + 1, 0},
    {false, 0x430, 4, cPointPos + 2, 0},
    {false, 0x498, 4, cPointColor + 2, 0},
    {false, 0x4e8, 4, cPointPos + 3, 0},
    {false, 0x550, 4, cPointColor + 3, 0},
    {false, 0x5d0, 4, cPsCamera, 0},
    {false, 0x5d4, 8, cDir, 0},
    {false, 0x5d8, 8, cMaterial, 0},
    {false, 0x5dc, 1, cMisc, 1},
    {false, 0x5e0, 8, cMul, 0},
    {false, 0x5e4, 4, cExposure, 0},
    {false, 0x5e8, 4, cAmbient, 0},
    {false, 0x5ec, 8, cLight, 0},
    {false, 0x674, 1, cReflectPs, 0},
    {false, 0x718, 4, cAmbientTint, 0},
    {false, 0x748, 1, cMisc, 0},
    {false, 0xaa0, 8, cDistFog, 0},
    {false, 0xaf8, 1, cUvBlock, 0},
    {false, 0xb1c, 1, cMisc, 3},
    {false, 0xb30, 1, cMisc, 2},
    {false, 0xb34, 1, cMisc2, 2},
    {false, 0xb38, 1, cPsFlags, 0},
    {false, 0xb3c, 4, cOutScale, 0},
    {false, 0xba8, 2, cPsFlags, 1},
    {false, 0xbac, 1, cMask0, 0},
    {false, 0xbb0, 4, cMask1, 0},
    {false, 0xbb4, 1, cMask0, 1},
    {false, 0xbb8, 1, cMask0, 2},
    {false, 0xbbc, 1, cMisc2, 3},
    {false, 0xd00, 2, cMask2, 0},
    {false, 0xd5c, 2, cMask2, 2},
    {false, 0xd60, 1, cMask0, 3},
    {false, 0xe30, 4, cMask3, 0},
    {false, 0xe34, 2, cMask4, 0},
    {false, 0xe38, 1, cMask4, 2},
    {false, 0xef4, 2, cMask5, 0},
    {false, 0xef8, 1, cMask4, 3},
    {false, 0xf44, 1, cPsFlags, 0},
    {false, 0xf60, 4, cBlend, 0},
    {false, 0xfd4, 1, cPsFlags, 3},
};

// Packed scalar loads for the four-layer stage (indices 81..141).
constexpr Load kVsLoadsStage[] = {
    {true, 0x1c, 4, 81, 0},
    {true, 0x20, 8, 82, 0},
    {true, 0x24, 4, 84, 0},
    {true, 0x28, 1, 85, 0},
    {true, 0xb8, 4, 85, 1},
    {true, 0xdc, 4, 86, 1},
    {true, 0x118, 16, 87, 1},
    {true, 0x168, 16, 91, 1},
    {true, 0x16c, 4, 95, 1},
    {true, 0x174, 4, 96, 1},
    {true, 0x2e8, 8, 97, 1},
    {true, 0x2ec, 4, 99, 1},
    {true, 0x2f0, 4, 100, 1},
    {true, 0x2f4, 1, 101, 1},
    {true, 0x2f8, 2, 101, 2},
    {true, 0x2fc, 4, 102, 0},
    {true, 0x300, 1, 103, 0},
};
constexpr Load kPsLoadsStage[] = {
    {false, 0x14, 1, 103, 1},
    {false, 0x18, 2, 103, 2},
    {false, 0x174, 4, 104, 0},
    {false, 0x1b4, 4, 105, 0},
    {false, 0x1f0, 2, 106, 0},
    {false, 0x1f4, 4, 106, 2},
    {false, 0x274, 4, 107, 2},
    {false, 0x2f0, 4, 108, 2},
    {false, 0x358, 4, 109, 2},
    {false, 0x3a8, 4, 110, 2},
    {false, 0x410, 4, 111, 2},
    {false, 0x460, 4, 112, 2},
    {false, 0x4c8, 4, 113, 2},
    {false, 0x518, 4, 114, 2},
    {false, 0x580, 4, 115, 2},
    {false, 0x5cc, 4, 116, 2},
    {false, 0x620, 2, 117, 2},
    {false, 0x664, 8, 118, 0},
    {false, 0x6d8, 2, 120, 0},
    {false, 0x720, 2, 120, 2},
    {false, 0x80c, 8, 121, 0},
    {false, 0x814, 4, 123, 0},
    {false, 0x818, 8, 124, 0},
    {false, 0x890, 4, 126, 0},
    {false, 0x8f4, 4, 127, 0},
    {false, 0x904, 1, 128, 0},
    {false, 0x928, 8, 128, 1},
    {false, 0xc94, 8, 130, 1},
    {false, 0xcf4, 1, 132, 1},
    {false, 0xd04, 1, 132, 2},
    {false, 0xd08, 1, 132, 3},
    {false, 0xd0c, 1, 133, 0},
    {false, 0xd10, 4, 133, 1},
    {false, 0xd60, 2, 134, 1},
    {false, 0xd94, 1, 134, 3},
    {false, 0xd98, 4, 135, 0},
    {false, 0xd9c, 1, 136, 0},
    {false, 0xda0, 1, 136, 1},
    {false, 0xda4, 1, 136, 2},
    {false, 0xed0, 2, 136, 3},
    {false, 0xf2c, 2, 137, 1},
    {false, 0xf30, 1, 137, 3},
    {false, 0x1004, 4, 138, 0},
    {false, 0x1008, 2, 139, 0},
    {false, 0x100c, 1, 139, 2},
    {false, 0x10c4, 2, 139, 3},
    {false, 0x10c8, 1, 140, 1},
    {false, 0x110c, 1, 140, 2},
    {false, 0x1120, 4, 140, 3},
    {false, 0x1178, 1, 141, 3},
};

struct VertexShaderInfo {
    const char* name;
    const char* defines;
    const Load* loads;
    size_t loadCount;
    uint32_t used[9]; // components read from each fetch slot (0: not read)
    int inputs;
};

struct Variant {
    const char* ps;
    const char* defines;
    const Load* loads;
    size_t loadCount;
    uint32_t texturePc[kTextures]; // 0: not read (bound to another of the draw's textures)
    uint8_t vs = 0;                // its vertex shader (kVertexShaders); 1 and up: SIMPLE_MATERIAL
    uint32_t cubePc = 0;           // the environment cube map's sample ([ref])
    bool unmatched = false;       // restricted to comparison copies until image-matched
    uint32_t volumePc = 0;         // stage volume sample, binding 13
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
    {"r_o1dst[ref].fxo", "#define SIMPLE_MATERIAL\n#define REFLECT_MATERIAL\n", kPsLoadsRefOdst, count(kPsLoadsRefOdst), {0x54, 0x16c, 0, 0x860, 0x868, 0x870, 0, 0xc60, 0xc68, 0}, 2, 0x698},
    {"r_o1dut[ref].fxo", "#define SIMPLE_MATERIAL\n#define SPECULAR_GREEN\n#define REFLECT_MATERIAL\n", kPsLoadsRefOdut, count(kPsLoadsRefOdut), {0x54, 0x16c, 0, 0x838, 0x830, 0x6f4, 0, 0xc44, 0xc68, 0}, 3, 0x67c},
    {"r_o1du[ref].fxo", "#define SIMPLE_MATERIAL\n#define SPECULAR_GREEN\n#define REFLECT_MATERIAL\n#define GEOMETRIC_NORMAL\n", kPsLoadsRefOdu, count(kPsLoadsRefOdu), {0, 0x94, 0, 0x7dc, 0x7ec, 0x640, 0, 0xba0, 0xba8, 0}, 3, 0x608},
    {"r_p1dut[ref].fxo", "#define SIMPLE_MATERIAL\n#define SPECULAR_GREEN\n#define MATERIAL_ALPHA_TEST\n#define REFLECT_MATERIAL\n", kPsLoadsRefPdut, count(kPsLoadsRefPdut), {0x54, 0x16c, 0, 0x838, 0x830, 0x6f4, 0, 0xc60, 0xc84, 0}, 3, 0x67c},
    {"r_o1dut_m(vr)2d[ref].fxo", "#define SPECULAR_GREEN\n#define REFLECT_MATERIAL\n#define KEEP_PRIMARY_NORMAL\n#define DIFFUSE_DETAIL_ONLY\n", kPsLoadsRefM2d, count(kPsLoadsRefM2d), {0x54, 0x160, 0x7f0, 0x844, 0x894, 0x710, 0, 0xca8, 0xcb0, 0}, 4, 0x6a4},
    {"r_o1d[ag].fxo", "#define SIMPLE_MATERIAL\n#define GEOMETRIC_NORMAL\n#define DIFFUSE_MATERIAL\n", kPsLoadsD, count(kPsLoadsD), {0, 0xd8, 0, 0x610, 0x590, 0, 0, 0x964, 0x96c, 0}, 5},
    {"r_b1dst.fxo", "#define SIMPLE_MATERIAL\n#define MODULATED_ALPHA_TEST\n#define RASTER_DEPTH_MASK\n#define BLEND_ALPHA_BOTH\n", kPsLoadsB1dst, count(kPsLoadsB1dst), {0x4c, 0x178, 0, 0x710, 0x754, 0x810, 0, 0, 0xba8, 0}, 1},
    {"r_o1dst_b(vr)2dst[nte].fxo", "#define DETAIL_NORMAL\n#define DETAIL_BLEND\n#define NO_EDGE_MASK\n", kPsLoadsNte, count(kPsLoadsNte), {0x7c, 0x1c8, 0x84, 0x7e0, 0x6c4, 0x834, 0x890, 0, 0, 0x74}, 0},
    {"r_o1dut_b(vr)2dut[ref].fxo", "#define DETAIL_NORMAL\n#define DETAIL_BLEND\n#define SPECULAR_GREEN\n#define REFLECT_MATERIAL\n#define REFLECT_DETAIL_BLEND\n", kPsLoadsB2dutRef, count(kPsLoadsB2dutRef), {0x7c, 0x1b4, 0x84, 0x884, 0x8c4, 0x6e8, 0x6e0, 0xce4, 0xcec, 0x74}, 4, 0x6d0, true},
    {"r_o1dut_m(vr)2du[ref].fxo", "#define SPECULAR_GREEN\n#define REFLECT_MATERIAL\n#define KEEP_PRIMARY_NORMAL\n#define MULTIPLY_DETAIL_REF\n", kPsLoadsM2duRef, count(kPsLoadsM2duRef), {0x54, 0x160, 0x6bc, 0x8ac, 0x8dc, 0x6fc, 0x604, 0xcd8, 0xce0, 0}, 4, 0x6b4},
    {"r_o1d.fxo", "#define SIMPLE_MATERIAL\n#define GEOMETRIC_NORMAL\n#define BARE_MATERIAL\n", kPsLoadsO1d, count(kPsLoadsO1d), {0, 0xa0, 0, 0x664, 0x6b0, 0, 0, 0xa94, 0xa9c, 0}, 5},
    {"r_o1dut_a(vr)2d.fxo", "#define KEEP_PRIMARY_NORMAL\n#define SPECULAR_GREEN\n#define AMBIENT_DETAIL_ADD\n", kPsLoadsA2d, count(kPsLoadsA2d), {0x4c, 0x178, 0x818, 0x74c, 0x7a0, 0x7c8, 0, 0xbd8, 0xbe0, 0}, 0},
    {"r_o1d2u[ref].fxo", "#define SIMPLE_MATERIAL\n#define SPECULAR_GREEN\n#define REFLECT_MATERIAL\n#define GEOMETRIC_NORMAL\n", kPsLoadsD2uRef, count(kPsLoadsD2uRef), {0, 0x94, 0, 0x5e0, 0x6a4, 0x65c, 0, 0xba4, 0xbac, 0}, 6, 0x750}, // compare: within 1/255 (~560 draws)
    {"r_p1dst[tree][plant].fxo", "#define SIMPLE_MATERIAL\n", kPsLoadsPlant, count(kPsLoadsPlant), {0x4c, 0x178, 0, 0x704, 0x758, 0x82c, 0, 0xbcc, 0xbd4, 0}, 7, 0}, // compare: 0 px (84 draws)
    {"r_o1d2u3t_m4d[nref][sign].fxo", "#define REFLECT_MATERIAL\n", kPsLoadsStage, count(kPsLoadsStage),
     {0x38, 0x15c, 0x1e8, 0x8e8, 0x8e0, 0x9ac, 0xe3c, 0xd84, 0xd8c, 0}, 8, 0x804, false, 0x644}, // compare: within 1/255 (~1,500 draws)
};

constexpr int kVariantCount = 25;
constexpr VertexShaderInfo kVertexShaders[9] = {
    {"r_b1dst_a(vr)2d.fxo", "", kVsLoads, count(kVsLoads), {3, 4, 4, 4, 2, 2, 4}, 7},
    {"r_b1dst.fxo", "#define SIMPLE_MATERIAL\n", kVsLoadsSimple, count(kVsLoadsSimple), {3, 3, 4, 4, 2, 4, 0}, 6},
    {"r_b1dst[ref].fxo", "#define SIMPLE_MATERIAL\n#define REFLECT_MATERIAL\n", kVsLoadsRefSt, count(kVsLoadsRefSt), {3, 3, 4, 4, 2, 4, 0}, 6},
    {"r_b1dut[ref].fxo", "#define SIMPLE_MATERIAL\n#define REFLECT_MATERIAL\n#define REFLECT_VERTEX_U\n", kVsLoadsRefUt, count(kVsLoadsRefUt), {3, 4, 4, 4, 2, 4, 0}, 6},
    {"r_o1dut_a(vr)2d[ref].fxo", "#define REFLECT_MATERIAL\n#define REFLECT_VERTEX_U\n", kVsLoadsRefM2d, count(kVsLoadsRefM2d), {3, 4, 4, 4, 2, 2, 4}, 7},
    {"r_b1d.fxo", "#define SIMPLE_MATERIAL\n#define DIFFUSE_MATERIAL\n", kVsLoadsD, count(kVsLoadsD), {3, 3, 4, 2, 4, 0, 0}, 5},
    {"r_o1d2u[ref].fxo", "#define SIMPLE_MATERIAL\n#define DIFFUSE_MATERIAL\n#define REFLECT_MATERIAL\n", kVsLoadsD2uRef, count(kVsLoadsD2uRef), {3, 3, 4, 2, 2, 4, 0}, 6},
    {"r_o1dst[tree][plant].fxo", "#define SIMPLE_MATERIAL\n", kVsLoadsPlant, count(kVsLoadsPlant), {3, 1, 4, 4, 4, 2, 4}, 7},
    {"r_o1d2s3t_m4d[ref].fxo", "", kVsLoadsStage, count(kVsLoadsStage), {3, 3, 4, 4, 2, 2, 2, 2, 4}, 9},
};

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
#ifdef DIFFUSE_MATERIAL
        float dd = mac(d.z, d.z, mac(d.y, d.y, mul(d.x, d.x)));
        float fd = mac(cf(19).z, d.z, mac(cf(19).y, d.y, mul(cf(19).x, d.x)));
        cosine = mul(-inversesqrt(dd), fd);
        r = inversesqrt(mac(f2.w, cosine, f1.w));
        float rr = mul(mul(r, r), r);
        float cc = mac(cosine, cosine, f3.w);
        for (int i = 0; i < 3; ++i) {
            float ph = mac(f2[i], rr, mul(f1[i], cc));
            float amp = mul(f3[i], ph);
            scatter[i] = mac(amp, -T[i], mul(f3.w, amp));
        }
#else
        scatter = (vec3(f3.w) - T) * (f3.xyz * phase);
#endif
    }
    if ((flags & 0x40000u) != 0u) scatter = vec3(0.0);
    const float k = uintBitsToFloat(0x40008102u);
#ifdef DIFFUSE_MATERIAL
    vec4 n = fetch(1u), col = fetch(2u), b = fetch(4u);
#elif defined(SIMPLE_MATERIAL)
    vec4 n = fetch(1u), t = fetch(2u), col = fetch(3u), b = fetch(5u);
#else
    vec4 n = fetch(1u), t = fetch(2u), col = fetch(3u), b = fetch(6u);
#endif
    // Rounded as the game's shader (each product, then each sum), as the prepass does.
    vec3 on = vec3(mac(k, n.x, -1.0), mac(k, n.y, -1.0), mac(k, n.z, -1.0));
#ifndef DIFFUSE_MATERIAL
    vec3 ot = vec3(mac(k, t.x, -1.0), mac(k, t.y, -1.0), mac(k, t.z, -1.0));
    float handed = mac(k, t.w, -1.0);
#endif
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
#ifdef DIFFUSE_MATERIAL
    p0 = vec4(fetch(3u).xy + uvo.xy, 0.0, 0.0);
#elif defined(SIMPLE_MATERIAL)
    p0 = vec4(fetch(4u).xy + uvo.xy, 0.0, 0.0);
#else
    p0 = vec4(fetch(4u).xy + uvo.xy, fetch(5u).xy + uvo.zw);
#endif
    p1 = vec4(col.rgb * T, col.a * cf(14).y);
#ifdef REFLECT_VERTEX_U
    p2 = vec4(scatter, mac(-cf(14).x, clamp(n.w * uintBitsToFloat(0x3f808102u), 0.0, 1.0), cf(14).x));
#else
    p2 = vec4(scatter, flatten);
#endif
    vec3 wn = vec3(dot3(W[0], on), dot3(W[1], on), dot3(W[2], on));
#ifdef REFLECT_MATERIAL
    // The reflection's weight: 1 + (normal . view) / |view| (the normal as it is, unnormalised),
    // saturated, to the fourth, scaled (VSV[0x10] 0x0).
    float rw;
    {
        vec3 dv = vec3(lp.x - cam.x, lp.y - cam.y, lp.z - cam.z);
        float d2 = mac(dv.z, dv.z, mac(dv.y, dv.y, mul(dv.x, dv.x)));
        float nd = mac(wn.z, dv.z, mac(wn.y, dv.y, mul(wn.x, dv.x)));
        float f = clamp(mac(inversesqrt(d2), nd, 1.0), 0.0, 1.0);
        float f2 = f * f, f4 = f2 * f2;
        rw = 0.5 * mac(cf(72).x, f4, cf(72).x);
    }
    p3 = vec4(wn, rw);
#else
    p3 = vec4(wn, 0.0);
#endif
    p4 = vec4(baked, 0.0);
    p5 = vec4(T, 0.0);
#ifdef DIFFUSE_MATERIAL
    p6 = vec4(lp, 0.0);
    p7 = vec4(0.0);
#else
    p6 = vec4(dot3(W[0], ot), dot3(W[1], ot), dot3(W[2], ot), handed);
    p7 = vec4(lp, 0.0);
#endif
}
)";

constexpr const char* kFs = R"(
layout(set = 0, binding = 8) uniform sampler2D tex[10];
#ifdef REFLECT_MATERIAL
layout(set = 0, binding = 9) uniform samplerCube env;
#endif
float mac(float a, float b, float c) { precise float r = a * b + c; return r; }
float mul(float a, float b) { precise float r = a * b; return r; }
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
#ifdef DIFFUSE_MATERIAL
float depthAt(vec2 uv) { vec4 m = cf(50); return mul(m.y, 1.0 / (m.x + at(7, uv).r)); }
#else
float depthAt(vec2 uv) { vec4 m = cf(50); return m.y / (at(7, uv).r + m.x); }
#endif
vec4 swz(vec4 v, uint s) { return vec4(v[s & 3u], v[(s >> 2) & 3u], v[(s >> 4) & 3u], v[(s >> 6) & 3u]); }
vec4 half4(vec4 v) { return vec4(unpackHalf2x16(packHalf2x16(v.xy)), unpackHalf2x16(packHalf2x16(v.zw))); }
void main() {
    uvec4 pf = cb.c[20];
    uint flags = pf.x;
#ifdef DIFFUSE_MATERIAL
    vec2 rasterToUV = cf(74).zw;
#else
    vec2 rasterToUV = uintBitsToFloat(pf.yz);
#endif
    vec2 uv0 = a0.xy, uv1 = a0.zw;
    vec2 ruv = gl_FragCoord.xy * rasterToUV;
#ifdef DIFFUSE_MATERIAL
    vec3 P = a6.xyz, gn = a3.xyz;
    float ni = inversesqrt(mac(gn.z, gn.z, mac(gn.y, gn.y, mul(gn.x, gn.x))));
    vec3 N = vec3(mul(gn.x, ni), mul(gn.y, ni), mul(gn.z, ni));
#else
#ifdef BARE_MATERIAL
    vec3 P = a6.xyz, gn = a3.xyz;
    vec3 N = normalize(gn);
    vec4 detail4 = vec4(1.0);
    float dw = 0.0;
#else
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
#elif defined(MULTIPLY_DETAIL_REF)
    vec4 detail4 = at(2, uv1);
    precise float dw = detail4.a * flatten;
#elif defined(DETAIL_BLEND)
    // Toward the detail textures by their alpha too.
    vec4 detail4 = at(2, uv1);
    precise float dw = detail4.a * flatten;
#elif defined(AMBIENT_DETAIL_ADD)
    vec4 detail4 = at(2, uv1);
    float dw = flatten;
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
#ifdef GEOMETRIC_NORMAL
    vec3 N = normalize(gn); // r_o1du[ref]: no normal map
#else
    vec3 N = normalize(tg * nxy.x + bt * nxy.y + gn * nz);
#endif
#endif
#endif

    float shR = 1.0, shG = 1.0, cover = 1.0, shR1 = 1.0, shG1 = 1.0;
    vec4 rm = cf(22);
    if (has(flags, 8u)) {
        vec4 g = at(1, ruv);
        shR = g.r, shG = g.g, cover = g.a;
        for (int i = 0; i < 2; ++i) {
#ifdef DIFFUSE_MATERIAL
            shR = mac(rm.x, shR, rm.y); shG = mac(shG, rm.z, rm.w);
#else
            shR = rm.x * shR + rm.y; shG = shG * rm.z + rm.w;
#endif
            if (i == 0) shR1 = shR, shG1 = shG; // [ref]'s reflection light takes the first remap
        }
    }
    vec3 tb = cf(23).xyz;
#ifdef DIFFUSE_MATERIAL
    vec3 dirTint;
    for (int i = 0; i < 3; ++i) dirTint[i] = mac(1.0 - tb[i], mul(shR, shG), tb[i]);
#else
    vec3 dirTint = tb + (1.0 - tb) * (shR * shG);
#endif
    vec3 indTint = dirTint;
#ifdef DIFFUSE_MATERIAL
    if (has(flags, 8u)) {
        vec4 tt = cf(24);
        for (int i = 0; i < 3; ++i) indTint[i] = mul(tt.z + tt.w, mac(dirTint[i], tt.x, tt.y));
    }
#else
    if (has(flags, 8u)) { vec4 tt = cf(24); indTint = (tt.z + tt.w) * (dirTint * tt.x + vec3(tt.y)); }
#endif

    vec3 point = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        if (!has(flags, 0x10000u << uint(i))) continue;
        vec4 pp = cf(25 + i), pc = cf(29 + i);
        vec3 d = P - pp.xyz;
#ifdef DIFFUSE_MATERIAL
        float d2 = mac(d.z, d.z, mac(d.y, d.y, mul(d.x, d.x)));
        float nd = mac(d.z, N.z, mac(N.y, d.y, mul(N.x, d.x)));
        float lam = clamp(mul(-inversesqrt(d2), nd), 0.0, 1.0);
#else
        float d2 = dot(d, d);
        float lam = clamp(-dot(N, d) * inversesqrt(d2), 0.0, 1.0);
#endif
        if (lam > 0.0) {
            float o = max(0.0, sqrt(d2) - pc.w);
#ifdef DIFFUSE_MATERIAL
            float att = clamp(mac(uintBitsToFloat(0x3f808102u),
                                  1.0 / mac(mul(pp.w, o), o, 1.0),
                                  uintBitsToFloat(0xbb810205u)), 0.0, 1.0);
            float weight = mul(lam, att);
            for (int j = 0; j < 3; ++j)
                point[j] = i == 0 ? mul(pc[j], weight) : mac(pc[j], weight, point[j]);
#else
            float att = clamp(uintBitsToFloat(0x3f808102u) / (1.0 + pp.w * o * o) + uintBitsToFloat(0xbb810205u), 0.0, 1.0);
            point += pc.rgb * (lam * att);
#endif
        }
    }
#ifdef DIFFUSE_MATERIAL
    vec4 camv = cf(75);
#else

    vec4 camv = cf(33);
#endif
    vec3 cs = P - camv.xyz;
#ifdef DIFFUSE_MATERIAL
    float cd2 = mac(cs.z, cs.z, mac(cs.y, cs.y, mul(cs.x, cs.x)));
#else
    float cd2 = dot(cs, cs);
#endif
    vec3 away = cs * inversesqrt(cd2);
    vec3 d0 = cf(41).xyz, d1 = cf(42).xyz;
#ifdef DIFFUSE_MATERIAL
    float nl0 = min(1.0, max(0.0, -mac(d0.z, N.z, mac(d0.y, N.y, mul(d0.x, N.x)))));
    float nl1 = min(1.0, max(0.0, -mac(d1.z, N.z, mac(d1.y, N.y, mul(d1.x, N.x)))));
#else
    float nl0 = clamp(-dot(d0, N), 0.0, 1.0), nl1 = clamp(-dot(d1, N), 0.0, 1.0);
#endif
#ifndef DIFFUSE_MATERIAL
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
#ifdef BARE_MATERIAL
    vec3 specMat = vec3(1.0);
#else
#ifdef SPECULAR_GREEN
    vec4 specSample = at(5, uv0 * cf(21).xy);
    vec3 specMat = vec3(specSample.g);
#else
#ifdef REFLECT_MATERIAL
    vec4 specSample = at(5, uv0 * cf(21).xy);
    vec3 specMat = specSample.rgb;
#else
    vec3 specMat = at(5, uv0 * cf(21).xy).rgb;
#endif
#endif
#endif
#ifdef REFLECT_DETAIL_BLEND
    vec4 specDetailSample = at(6, uv1 * cf(71).xy);
#endif
#ifdef MULTIPLY_DETAIL_REF
    vec4 specDetailSample = at(6, uv1 * cf(44).xy);
#endif
#if defined(SIMPLE_MATERIAL) || defined(DIFFUSE_DETAIL_ONLY) || defined(AMBIENT_DETAIL_ADD)
    vec3 specDetail = vec3(1.0);
#elif defined(DETAIL_NORMAL)
#ifdef REFLECT_DETAIL_BLEND
    vec3 specDetail = specDetailSample.rgb;
#else
    vec3 specDetail = at(6, uv1 * cf(71).xy).rgb;
#endif
#else
#ifdef MULTIPLY_DETAIL_REF
    vec3 specDetail = specDetailSample.rgb;
#else
    vec3 specDetail = at(6, uv1 * cf(44).xy).rgb;
#endif
#endif
#ifdef SPECULAR_GREEN
    specDetail = vec3(specDetail.g);
#endif
    vec4 screenLight = at(3, ruv) / cf(39).z;
#ifdef AMBIENT_DETAIL_ADD
    vec3 adjBase = base.rgb * base.rgb;
    vec3 adjSpec = specMat;
#elif defined(MULTIPLY_DETAIL_REF)
    vec3 adjBase;
    for (int i = 0; i < 3; ++i) {
        float fade = mac(-mul(detail[i], detail[i]), dw, dw);
        float b2 = mul(base[i], base[i]);
        adjBase[i] = mac(b2, -fade, b2);
    }
    float sf = mac(-specDetailSample.g, dw, dw);
    vec3 adjSpec = vec3(mac(-specSample.g, sf, specSample.g));
#elif defined(DIFFUSE_DETAIL_ONLY)
    // new_ps: 0x8f0/0x938 squares red detail first. Green (0x984/0x988)
    // and blue (0x8f4/0x8f8) multiply the weight first.
    vec3 fade;
    fade.r = mac(-mul(detail.r, detail.r), dw, dw);
    fade.g = mac(-mul(dw, detail.g), detail.g, dw);
    fade.b = mac(-mul(dw, detail.b), detail.b, dw);
    vec3 adjBase;
    for (int i = 0; i < 3; ++i) {
        float b2 = mul(base[i], base[i]);
        adjBase[i] = mac(b2, -fade[i], b2);
    }
    vec3 adjSpec = specMat;
#elif defined(DETAIL_BLEND)
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
#ifdef REFLECT_MATERIAL
    float secondaryGain = cf(73).x; // its load lands with the floor at 0xf4
#else
    float secondaryGain = misc.y;
#endif
    vec3 dirSpec = lc0 * dirTint * lobe0 + lc1 * (2.0 * secondaryGain) * lobe1 * nl1;
    vec3 spec = vtint * specularColor * dirSpec;
    vec3 indSpec = T * specularColor * bp + vec3(screenLight.a * shG);
    spec += itint * lobe1 * indSpec;
    vec3 lin = adjBase * diffuse + adjSpec * spec;
#ifdef AMBIENT_DETAIL_ADD
    for (int i = 0; i < 3; ++i)
        lin[i] = mac(mul(detail4.a, dw), mul(detail[i], detail[i]), lin[i]);
#endif
#ifdef REFLECT_MATERIAL
    // The environment: reflected about N (its x mirrored, as the shader's cube lookup takes it),
    // decoded (rgb / (4 a), squared), lit by the first shadow remap and the light's angle.
    {
        float dd2 = mac(away.z, N.z, mac(away.y, N.y, mul(away.x, N.x)));
        float twice = 2.0 * dd2;
        vec3 cd = vec3(mac(N.x, twice, -away.x), mac(-N.y, twice, away.y), mac(-N.z, twice, away.z));
        vec4 cube = texture(env, cd);
        float inva = 1.0 / max(cube.a, uintBitsToFloat(0x3b808081u));
        vec3 dec = cube.rgb * (0.25 * inva);
        vec3 envLin = dec * dec;
        float angular = clamp(0.75 - dot(d0, N), 0.0, 1.0);
        float rlight = max(shR1 * shG1 * angular, cf(73).y);
#ifdef SPECULAR_GREEN
#ifdef REFLECT_DETAIL_BLEND
        float red = mac(specDetailSample.r - specSample.r, dw, specSample.r);
        float weight = mul(mul(red, a3.w), rlight);
        for (int i = 0; i < 3; ++i)
            lin[i] = mac(dec[i], mul(weight, dec[i]), lin[i]);
#elif defined(MULTIPLY_DETAIL_REF)
        float rf = mac(-specDetailSample.r, dw, dw);
        float red = mac(-specSample.r, rf, specSample.r);
        float weight = mul(mul(red, a3.w), rlight);
        for (int i = 0; i < 3; ++i)
            lin[i] = mac(dec[i], mul(weight, dec[i]), lin[i]);
#elif defined(DIFFUSE_DETAIL_ONLY)
        // new_ps 0x88c..0x890, then 0x8bc/0xa0c, 0xa60/0xa9c, 0xaa0/0xb18.
        float weight = mul(mul(specSample.r, a3.w), rlight);
        for (int i = 0; i < 3; ++i)
            lin[i] = mac(dec[i], mul(weight, dec[i]), lin[i]);
#else
        lin += envLin * (specSample.r * a3.w * rlight);
#endif
#else
        // 0x870 loads the specular texture's alpha over the point light's red (v12); the reflection goes
        // into the specular sum, multiplied by the specular colour (0x938.., 0x96c..).
        lin += adjSpec * envLin * rlight * (a3.w * specSample.a);
#endif
    }
#endif

    float exposure = cf(39).x;
    vec3 tc = vec3(tone(lin.r, exposure), tone(lin.g, exposure), tone(lin.b, exposure));
#else
    vec4 mul0 = cf(34), misc = cf(36);
    vec4 base = at(4, uv0);
    vec3 screen = at(3, ruv).rgb;
    float invExposure = 1.0 / cf(39).z;
    float blendBase = mac(1.0 - misc.x, shG, misc.x);
    vec3 tc;
    // new_ps 0x500..0x7f0: products before multiply-adds, exposure before lighting.
    for (int i = 0; i < 3; ++i) {
        float ld = mul(mul(cf(45)[i], dirTint[i]), nl0);
        ld = mac(mul(cf(46)[i], shG), nl1, ld);
        float b2 = mul(base[i], base[i]);
        float ambient = mac(mul(cf(35).w, cf(43)[i]), b2, cf(40)[i]);
        float dc = mul(cf(37)[i], mul0[i]);
        float direct = mul(mul(a1[i], cover), mac(dc, ld, ambient));
        float indirect = mul(mul(a5[i], dc), a4[i] + point[i]);
        indirect = mac(mul(screen[i], invExposure), shG, indirect);
        indirect = mul(mul(cover, indTint[i]), indirect);
        direct = mac(blendBase, indirect, direct);
        float tx = mul(mul(mul(cf(39).x, b2), direct), 2.0);
        float den = mac(uintBitsToFloat(0x3e19999au), tx, 0.5);
        den = mac(tx, den, uintBitsToFloat(0x3d75c290u));
        float num = mac(uintBitsToFloat(0x3e19999au), tx, uintBitsToFloat(0x3d4ccccdu));
        num = mac(num, tx, uintBitsToFloat(0x3b83126fu));
        float curve = mul(1.0 / den, num);
        tc[i] = sqrt(clamp(mac(uintBitsToFloat(0x3fb0852eu), curve,
                               uintBitsToFloat(0xbdbc49cau)), 0.0, 1.0));
    }
#endif
    vec3 color = tc;
    float fogAmount = 0.0;
    if (has(flags, 4u)) color += a2.rgb;
    vec4 df0 = cf(47), df1 = cf(48);
    if (has(flags, 0x400000u) && !has(flags, 4u)) {
#ifdef DIFFUSE_MATERIAL
        fogAmount = clamp(mac(df1.z, sqrt(cd2), df1.w), 0.0, 1.0);
#else
        fogAmount = clamp(df1.z * sqrt(cd2) + df1.w, 0.0, 1.0);
#endif
#ifdef DIFFUSE_MATERIAL
        for (int i = 0; i < 3; ++i) color[i] = mac(fogAmount, -tc[i], mac(df0[i], fogAmount, tc[i]));
#else
        color = tc * (1.0 - fogAmount) + df0.xyz * fogAmount;
#endif
    }
#ifdef MODULATED_ALPHA_TEST
    float alphaGain = mul(mul(misc.z, misc.w), cf(44).z);
    alphaGain = mul(alphaGain, base.a);
    if (mac(alphaGain, a1.w, -cf(70).x) < 0.0) demote;
    float alpha = mul(alphaGain, a1.w);
#else
    float alpha = misc.z * misc.w * mul0.w * base.a;
#endif
    if (!has(flags, 8u)) { vec4 os = cf(49); color *= os.xyz; alpha *= os.w; }

    float mask = 0.0;
#ifndef NO_EDGE_MASK
    if (!has(flags, 0x2000000u)) {
        uint mf = cb.c[44].w;
        vec4 k0 = cf(50), k1 = cf(51), k2 = cf(52), k3 = cf(53), k4 = cf(54), k5 = cf(55);
#ifdef RASTER_DEPTH_MASK
        float z = gl_FragCoord.z;
#else
        float z = depthAt(ruv);
#endif
        vec3 sn = 2.0 * at(8, ruv).rgb - 1.0;
        float dStart = k1.x, dNear = k1.y, dEnd = k1.z, minRadius = k1.w;
#ifdef DIFFUSE_MATERIAL
        float radiusFade = clamp(mac(-(z - dStart), 1.0 / (dEnd - dStart), 1.0), 0.0, 1.0);
#else
        float radiusFade = clamp(1.0 - (z - dStart) / (dEnd - dStart), 0.0, 1.0);
#endif
#ifdef DIFFUSE_MATERIAL
        float ir = 1.0 / (dStart - dNear);
        float nearRamp = clamp(mac(-dNear, ir, mul(z, ir)), 0.0, 1.0);
#else
        float nearRamp = clamp((z - dNear) / (dStart - dNear), 0.0, 1.0);
#endif
        vec2 sdir = rasterToUV * vec2(sn.x, -sn.y);
        float nr = nearRamp * max(minRadius, k0.z * radiusFade);
        vec3 np = at(8, ruv + sdir * nr).rgb, nm = at(8, ruv - sdir * nr).rgb;
#ifdef DIFFUSE_MATERIAL
        float snsum = (sn.x + sn.y) + sn.z;
        float rp = mul(mul(np.x, sn.x), 2.0);
        rp = mac(2.0, mul(np.y, sn.y), rp);
        rp = mac(2.0, mul(np.z, sn.z), rp) - snsum;
        float rmn = mul(mul(nm.x, sn.x), 2.0);
        rmn = mac(2.0, mul(nm.y, sn.y), rmn);
        rmn = mac(2.0, mul(nm.z, sn.z), rmn) - snsum;
#else
        float rp = dot(sn, 2.0 * np - 1.0), rmn = dot(sn, 2.0 * nm - 1.0);
#endif
        float normalEdge = 0.0, depthEdge = 0.0, facingMask = 0.0;
        const float eps = uintBitsToFloat(0x3b808081u);
        if (has(mf, 4u)) {
#ifdef DIFFUSE_MATERIAL
            float ag = mac(0.5, rmn, 0.5);
            ag = mul(mac(rp, ag, ag), 0.5);
#else
            float ag = 0.25 * (rp + 1.0) * (rmn + 1.0);
#endif
            float ag2 = ag * ag;
#ifdef DIFFUSE_MATERIAL
            float passed = ag2 + eps >= mac(-k2.y, nearRamp, 1.0) ? 1.0 : 0.0;
#else
            float passed = ag2 + eps >= 1.0 - k2.y * nearRamp ? 1.0 : 0.0;
#endif
#ifdef DIFFUSE_MATERIAL
            float sup = mac(1.0 - k2.x, clamp(passed + ag2, 0.0, 1.0), k2.x);
            normalEdge = mac(-nearRamp, sup, nearRamp);
#else
            float sup = k2.x + (1.0 - k2.x) * clamp(passed + ag2, 0.0, 1.0);
            normalEdge = nearRamp * (1.0 - sup);
#endif
        }
#ifndef RASTER_DEPTH_MASK
        if (has(mf, 8u)) {
            float dr = nearRamp * max(minRadius, k2.z * radiusFade);
            float dp = depthAt(ruv + sdir * dr), dm = depthAt(ruv - sdir * dr);
#ifdef DIFFUSE_MATERIAL
            float resp = mac(-0.5, (mul(-2.0, z) + dp) + dm, 1.0);
            float passed = resp + eps >= mac(-k0.w, nearRamp, 1.0) ? 1.0 : 0.0;
            float sup = mac(1.0 - k2.w, clamp(passed + resp, 0.0, 1.0), k2.w);
            depthEdge = mac(-nearRamp, sup, nearRamp);
#else
            float resp = 1.0 - 0.5 * (-2.0 * z + dp + dm);
            float passed = resp + eps >= 1.0 - k0.w * nearRamp ? 1.0 : 0.0;
            float sup = k2.w + (1.0 - k2.w) * clamp(passed + resp, 0.0, 1.0);
            depthEdge = nearRamp * (1.0 - sup);
#endif
        }
#endif
        if (has(mf, 2u)) {
#ifdef DIFFUSE_MATERIAL
            float facing = clamp(-mac(away.z, N.z, mac(away.y, N.y, mul(N.x, away.x))), 0.0, 1.0);
#else
            float facing = clamp(-dot(N, away), 0.0, 1.0);
#endif
            float far_ = k3.x, near_ = min(k3.y, far_ + uintBitsToFloat(0xba83126fu));
#ifdef DIFFUSE_MATERIAL
            float invRange = 1.0 / (far_ - near_);
            float ramp = clamp(mac(near_, -invRange, mul(z, invRange)), 0.0, 1.0);
            float width = mac(ramp, k4.y - k3.z, k3.z);
            float nf = clamp(mul(1.0 / mac(nearRamp, width, eps), facing + eps), 0.0, 1.0);
            float fs = clamp(mac(mul(1.0 - k4.x, nf), nf, k4.x), 0.0, 1.0);
            float cand = mac(nearRamp, -fs, nearRamp);
            float thr = mac(k4.z - k3.w, ramp, k3.w);
            float resp = mac(-rp, rmn, 1.0);
#else
            float ramp = clamp((z - near_) / (far_ - near_), 0.0, 1.0);
            float width = k3.z + ramp * (k4.y - k3.z);
            float nf = clamp((facing + eps) / (nearRamp * width + eps), 0.0, 1.0);
            float fs = clamp(k4.x + (1.0 - k4.x) * nf * nf, 0.0, 1.0);
            float cand = nearRamp * (1.0 - fs);
            float thr = k3.w + ramp * (k4.z - k3.w);
            float resp = 1.0 - rp * rmn;
#endif
            facingMask = thr > 0.0 ? cand * (resp >= thr ? 1.0 : 0.0) : cand;
        }
        mask = max(depthEdge, max(facingMask, normalEdge));
#ifdef DIFFUSE_MATERIAL
        mask = mul(max(1.0 - k5.y, clamp(mac(-(z - dStart), 1.0 / (k5.x - dStart), 1.0), 0.0, 1.0)), mask);
        mask = mac(mask, -fogAmount, mask);
#else
        mask *= max(1.0 - k5.y, clamp(1.0 - (z - dStart) / (k5.x - dStart), 0.0, 1.0));
        mask *= 1.0 - fogAmount;
#endif
        mask *= k4.w <= mask ? 1.0 : 0.0;
    }
#endif

    float alpha1 = alpha;
    if (has(flags, 0x800000u)) {
        vec4 bl = cf(56);
#ifdef DIFFUSE_MATERIAL
        for (int i = 0; i < 3; ++i) color[i] = mac(-bl[i], bl.w, mac(bl.w, color[i], bl[i]));
#else
        color = bl.xyz + bl.w * color - bl.xyz * bl.w;
#endif
        mask *= bl.w;
        alpha1 = alpha * bl.w;
#ifdef BLEND_ALPHA_BOTH
        alpha = alpha1;
#endif
    }
    float alpha0 = has(pf.w, 1u) ? clamp(uintBitsToFloat(0x3a4d9a68u) * gl_FragCoord.z, 0.0, 1.0) : alpha;
    uvec4 so = cb.c[69];
    o0 = swz(half4(vec4(color, alpha0)), so.x);
    o1 = swz(half4(vec4(mask, 0.0, 0.0, alpha1)), so.y);
}
)";

// r_o1d2u[ref], ps 0x53c570c00. The register names and forward block labels
// retain the translated GLSL's operation boundaries and contraction opportunities.
// No material alpha test occurs in this program: the translator's final discard
// only restores the raster execution mask. Alpha is exported, without a demote.
// Samples: shadow, screen light, UV1 specular RG, UV0 base, cube, then edge samples.
constexpr const char* kFsD2uRefMain = R"(
float F(uint x) { return uintBitsToFloat(x); }
// v_cube*: face id, face s/t and 2x the major axis of a direction (GCN face order +x -x +y -y +z -z).
float cubeId(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return d.z < 0.0 ? 5.0 : 4.0;
    if (a.y >= a.x) return d.y < 0.0 ? 3.0 : 2.0;
    return d.x < 0.0 ? 1.0 : 0.0;
}
float cubeSc(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return d.z < 0.0 ? -d.x : d.x;
    if (a.y >= a.x) return d.x;
    return d.x < 0.0 ? d.z : -d.z;
}
float cubeTc(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return -d.y;
    if (a.y >= a.x) return d.y < 0.0 ? -d.z : d.z;
    return -d.y;
}
float cubeMa(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return 2.0 * d.z;
    if (a.y >= a.x) return 2.0 * d.y;
    return 2.0 * d.x;
}
// Cube image coordinates (s, t in [1, 2] as v_cube* produce them, face id) back to a direction.
vec3 cubeDir(vec3 c) {
    float u = 2.0 * c.x - 3.0, v = 2.0 * c.y - 3.0;
    switch (int(c.z + 0.5) & 7) {
    case 0: return vec3(1.0, -v, -u);
    case 1: return vec3(-1.0, -v, u);
    case 2: return vec3(u, 1.0, v);
    case 3: return vec3(u, -1.0, -v);
    case 4: return vec3(u, -v, 1.0);
    default: return vec3(-u, -v, -1.0);
    }
}
void main() {
uint exec_hi = 0u; uint exec_lo = 0u; uint s0 = 0u; uint s1 = 0u; uint s10 = 0u; uint s16 = 0u; uint s17 = 0u;
uint s18 = 0u; uint s2 = 0u; uint s20 = 0u; uint s21 = 0u; uint s22 = 0u; uint s23 = 0u; uint s24 = 0u; uint s25 = 0u;
uint s26 = 0u; uint s28 = 0u; uint s29 = 0u; uint s3 = 0u; uint s30 = 0u; uint s31 = 0u; uint s32 = 0u; uint s33 = 0u;
uint s34 = 0u; uint s35 = 0u; uint s36 = 0u; uint s37 = 0u; uint s38 = 0u; uint s4 = 0u; uint s40 = 0u; uint s41 = 0u;
uint s42 = 0u; uint s43 = 0u; uint s44 = 0u; uint s48 = 0u; uint s49 = 0u; uint s5 = 0u; uint s50 = 0u; uint s52 = 0u;
uint s53 = 0u; uint s54 = 0u; uint s56 = 0u; uint s57 = 0u; uint s58 = 0u; uint s6 = 0u; uint s68 = 0u; uint s69 = 0u;
uint s7 = 0u; uint s70 = 0u; uint s76 = 0u; uint s77 = 0u; uint s78 = 0u; uint s79 = 0u; uint s8 = 0u; uint s80 = 0u;
uint s82 = 0u; uint s88 = 0u; uint s89 = 0u; uint s9 = 0u; uint s90 = 0u; uint s91 = 0u; float v0 = 0.0;
float v1 = 0.0; float v10 = 0.0; float v11 = 0.0; float v12 = 0.0; float v13 = 0.0; float v14 = 0.0; float v15 = 0.0;
float v16 = 0.0; float v17 = 0.0; float v18 = 0.0; float v19 = 0.0; float v2 = 0.0; float v20 = 0.0; float v21 = 0.0;
float v22 = 0.0; float v23 = 0.0; float v24 = 0.0; float v25 = 0.0; float v26 = 0.0; float v27 = 0.0; float v28 = 0.0;
float v29 = 0.0; float v3 = 0.0; float v30 = 0.0; float v31 = 0.0; float v32 = 0.0; float v33 = 0.0; float v34 = 0.0;
float v35 = 0.0; float v36 = 0.0; float v37 = 0.0; float v38 = 0.0; float v39 = 0.0; float v4 = 0.0; float v40 = 0.0;
float v41 = 0.0; float v42 = 0.0; float v43 = 0.0; float v44 = 0.0; float v45 = 0.0; float v46 = 0.0; float v47 = 0.0;
float v48 = 0.0; float v49 = 0.0; float v5 = 0.0; float v50 = 0.0; float v51 = 0.0; float v52 = 0.0; float v53 = 0.0;
float v54 = 0.0; float v55 = 0.0; float v56 = 0.0; float v57 = 0.0; float v58 = 0.0; float v59 = 0.0; float v6 = 0.0;
float v60 = 0.0; float v61 = 0.0; float v62 = 0.0; float v63 = 0.0; float v7 = 0.0; float v8 = 0.0; float v9 = 0.0;
uint vcc_hi = 0u; uint vcc_lo = 0u; bool scc = false; uint pc = 0x00000000u;
exec_lo = 1u;
v2 = gl_FragCoord.x; v3 = gl_FragCoord.y; v4 = gl_FragCoord.w;
{
pc = 0x00000000u;
}
// Geometric normal, shadow remaps, and the four point lights.
s91 = cb.c[20].x; s90 = s91 & 8u;
vec3 gn = a3.xyz;
float ni = inversesqrt(gn.z * gn.z + (gn.y * gn.y + gn.x * gn.x)); v20 = gn.x * ni; v21 = gn.y * ni; v22 = gn.z * ni;
v5 = F(cb.c[20].y) * v2; v6 = F(cb.c[20].z) * v3; v17 = 1.0; v18 = 1.0; v19 = 1.0;
if (s90 != 0u) {
    vec4 shadow = at(1, vec2(v5, v6));
    v17 = shadow.r; v18 = shadow.g; v19 = shadow.a;
    vec4 rm = cf(22);
    v17 = rm.x * v17 + rm.y; v18 = v18 * rm.z + rm.w;
}
v7 = v17; v29 = v18;
if (s90 != 0u) {
    vec4 rm = cf(22);
    v7 = rm.x * v17 + rm.y; v29 = rm.z * v18 + rm.w;
}
v7 = v7 * v29;
vec3 tb = cf(23).xyz;
v33 = (1.0 - tb.r) * v7 + tb.r; v31 = (1.0 - tb.g) * v7 + tb.g; v32 = (1.0 - tb.b) * v7 + tb.b;
v14 = v33; v7 = v31; v8 = v32;
if (s90 != 0u) {
    vec4 tt = cf(24);
    float gain = tt.z + tt.w;
    v14 = gain * (v33 * tt.x + tt.y);
    v7 = gain * (v31 * tt.x + tt.y);
    v8 = gain * (tt.x * v32 + tt.y);
}
v28 = a6.x; v27 = a6.y; v26 = a6.z; v15 = 0.0; v13 = 0.0; v30 = 0.0;
for (int i = 0; i < 4; ++i) {
    if ((s91 & (0x10000u << uint(i))) == 0u) continue;
    vec4 pp = cf(25 + i);
    float dx = v28 - pp.x, dy = v27 - pp.y, dz = v26 - pp.z;
    float dd = dx * dx, nd = v20 * dx;
    dd = dy * dy + dd; nd = v21 * dy + nd;
    dd = dz * dz + dd;
    float invd = inversesqrt(dd);
    nd = dz * v22 + nd;
    float lam = clamp(-invd * nd, 0.0, 1.0);
    if (lam > 0.0) {
        float distance_ = sqrt(dd);
        vec4 light = cf(29 + i);
        float o = max(0.0, distance_ - light.w);
        float po = pp.w * o;
        float attenuation = 1.0 / (po * o + 1.0);
        attenuation = clamp(F(0x3f808102u) * attenuation + F(0xbb810205u), 0.0, 1.0);
        float weight = lam * attenuation;
        if (i == 0) {
            v15 = light.r * weight; v13 = light.g * weight; v30 = light.b * weight;
        } else {
            v15 = light.r * weight + v15;
            v13 = light.g * weight + v13;
            v30 = light.b * weight + v30;
        }
    }
}
pc = 0x000004a4u;
if (pc == 0x000004a4u) {
v10 = a0[3];
vcc_lo = 0x3f400000u;
v9 = a0[2]; v16 = a4[1]; v43 = a5[0]; v51 = (v19 * v14); v38 = (v17 * v18); s56 = cb.c[33][0]; s57 = cb.c[33][1];
s58 = cb.c[33][2]; s48 = cb.c[41][0]; s49 = cb.c[41][1]; s50 = cb.c[41][2]; s52 = cb.c[42][0]; s53 = cb.c[42][1];
s54 = cb.c[42][2]; s16 = cb.c[34][0]; s17 = cb.c[34][1]; s18 = cb.c[34][2]; s20 = cb.c[35][0]; s21 = cb.c[35][1];
s22 = cb.c[35][2]; s23 = cb.c[35][3]; s88 = cb.c[21][0]; s89 = cb.c[21][1]; s76 = cb.c[36][0]; s78 = cb.c[73][0];
s79 = cb.c[73][1]; v34 = (v19 * v8); v41 = a3[3]; v35 = (v19 * v7); v8 = (v28 - F(s56)); v7 = (v8 * v8);
v11 = (v27 - F(s57)); v12 = (v26 - F(s58)); v7 = (v11 * v11 + v7); v7 = (v12 * v12 + v7); v17 = (inversesqrt(v7));
v36 = (v17 * v8 + F(s48)); v24 = (v17 * v8); v7 = (v17 * v12 + F(s50)); v18 = (v17 * v11 + F(0xbe99999au));
v23 = (v17 * v12); v12 = (v17 * v11 + F(s49)); v8 = (F(s48) * v20); v39 = (F(s49) * v21 + v8); v25 = (v17 * v11);
v11 = (v36 * v36); v11 = (v12 * v12 + v11); v11 = (v7 * v7 + v11); v39 = (F(s50) * v22 + v39);
v14 = clamp(F(vcc_lo) - v39, 0.0, 1.0); v8 = (F(s89) * v10); v10 = (v36 * v20); v12 = (v12 * v21 + v10);
v10 = (inversesqrt(v11)); v12 = (v22 * v7 + v12); v7 = clamp((-v10) * v12, 0.0, 1.0); v36 = (log2(v7));
v7 = (F(s88) * v9); v10 = (F(s52) * v20); v10 = (F(s53) * v21 + v10); v10 = (F(s54) * v22 + v10);
v17 = (max(F(0u), (-v10)));
{
vec2 uv_ = vec2(v5, v6);
uv_.xy *= sc(3);
{ vec4 t_ = vec4(texture(tex[3], uv_));
v9 = t_[0]; v10 = t_[1]; v11 = t_[2]; v12 = t_[3];
}
}
s4 = cb.c[43][0]; s5 = cb.c[43][1]; s6 = cb.c[43][2]; s7 = cb.c[43][3]; v5 = (v24 * v20); s24 = cb.c[37][0];
s25 = cb.c[37][1]; s26 = cb.c[37][2]; s28 = cb.c[38][0]; s29 = cb.c[38][1]; s30 = cb.c[38][2]; s31 = cb.c[38][3];
v6 = (v24 * v24); v6 = (v18 * v18 + v6); v18 = (v18 * v21 + v5); v6 = (v23 * v23 + v6); v37 = (inversesqrt(v6));
v18 = (v22 * v23 + v18); v18 = clamp((-v37) * v18, 0.0, 1.0); v6 = (F(s31) * v36); v18 = (log2(v18));
v42 = (v21 * v25 + v5); v5 = F(s16); v18 = (F(s31) * v18); v36 = (exp2(v18)); v42 = ((v22 * v23 + v42) * 2.0);
v55 = ((-v22) * v42 + v23); s80 = cb.c[39][0]; s82 = cb.c[39][2]; s68 = cb.c[40][0]; s69 = cb.c[40][1];
s70 = cb.c[40][2]; v47 = (exp2(v6)); v6 = F(s17); v37 = (min(1.0, v17));
{
vec2 uv_ = vec2(v7, v8);
uv_.xy *= sc(5);
{ vec4 t_ = vec4(texture(tex[5], uv_));
v17 = t_[0]; v18 = t_[1];
}
}
v45 = (F(s25) * v6); v6 = a0[1]; v46 = (F(s24) * v5); v5 = F(s4); v48 = (1.0 / F(s82)); v50 = (F(s23) * v5);
v5 = a1[0]; v49 = (v5 * v19); v5 = a4[0]; v44 = (v5 + v15); v5 = a0[0];
{
vec2 uv_ = vec2(v5, v6);
uv_.xy *= sc(4);
{ vec4 t_ = vec4(texture(tex[4], uv_));
v5 = t_[0]; v6 = t_[1]; v7 = t_[2]; v8 = t_[3];
}
}
s32 = cb.c[45][0]; s33 = cb.c[45][1]; s34 = cb.c[45][2]; s35 = cb.c[45][3]; s36 = cb.c[46][0]; s37 = cb.c[46][1];
s38 = cb.c[46][2]; v40 = (v43 * v46); v52 = (v40 * v44); v40 = (v38 * v14); v14 = (max(F(0u), (-v39)));
v39 = (v20 * v42 + (-v24)); v15 = (F(s36) * v29); v54 = (min(1.0, v14)); v14 = ((-v21) * v42 + v25);
v60 = (F(s33) * v47); v53 = (F(s38) * v29); v38 = (v16 + v13); v13 = (F(s32) * v33);
v16 = (cubeTc(vec3(v39, v14, v55))); v56 = (v13 * v54); v13 = (F(s36) * v36); v57 = (v13 * v37);
v13 = (cubeMa(vec3(v39, v14, v55))); v58 = (1.0 / abs(v13)); v13 = (cubeSc(vec3(v39, v14, v55)));
v13 = (v13 * v58 + F(0x3fc00000u)); v56 = (v37 * v15 + v56); v15 = (1.0 - F(s76)); v42 = (v15 * v29 + F(s76));
v15 = (cubeId(vec3(v39, v14, v55))); v14 = (v16 * v58 + F(0x3fc00000u)); v16 = F(s20); v55 = (F(s28) * v16);
{
vec3 uv_ = vec3(v13, v14, v15);
uv_ = cubeDir(uv_);
{ vec4 t_ = vec4(texture(env, uv_));
v13 = t_[0]; v14 = t_[1]; v15 = t_[2]; v16 = t_[3];
}
}
v58 = (v49 * v55); v39 = a5[1]; v55 = (v43 * v55); v43 = (F(s32) * v47); v59 = (v43 * v33); v33 = F(s22);
v43 = (F(s30) * v33); v33 = (2.0 * F(s78)); v59 = (v33 * v57 + v59); v40 = (max(F(s79), v40)); v57 = (v58 * v59);
v58 = (F(s34) * v47); v59 = (F(s37) * v29); v47 = a1[1]; v47 = (v47 * v19); v63 = (v58 * v32); v32 = (F(s34) * v32);
v62 = (v32 * v54); v58 = F(s18); v58 = (F(s26) * v58); v62 = (v37 * v53 + v62); v32 = (v60 * v31);
v31 = (F(s33) * v31); v60 = a4[2]; v53 = (v31 * v54); v54 = (F(s38) * v36); v54 = (v54 * v37); v31 = F(s5);
v61 = (F(s23) * v31); v60 = (v60 + v30); v30 = F(s6); v30 = (F(s23) * v30); v53 = (v37 * v59 + v53);
v59 = (v36 * v51); v63 = (v33 * v54 + v63); v54 = a5[2]; s77 = 0x3e19999au;
vcc_hi = s91 & 4u;
scc = vcc_hi != 0u;
vcc_lo = s91 & 0x00400000u;
scc = vcc_lo != 0u;
s4 = 0xbdbc49cau;
scc = vcc_lo != 0u;
v31 = a2[0]; s7 = scc ? 0u : 1u;
scc = vcc_hi != 0u;
vcc_lo = ((exec_lo & 1u) != 0u && (vcc_hi != 0u)) ? 1u : 0u; vcc_hi = 0u;
v9 = (v9 * v48); v52 = (v9 * v29 + v52); v52 = (v51 * v52); v10 = (v10 * v48); v9 = (v54 * v43); v51 = (v5 * v5);
v7 = (v7 * v7); v50 = (v50 * v51 + F(s68)); v30 = (v30 * v7 + F(s70)); v5 = (v6 * v6); v6 = (v39 * v45);
v50 = (v56 * v46 + v50); v30 = (v62 * v58 + v30); v49 = (v49 * v50); v46 = (v61 * v5 + F(s69)); v50 = (v6 * v38);
v6 = a1[2]; v19 = (v6 * v19); v6 = (v12 * v48); v12 = (v11 * v48); v6 = (v29 * v6); v48 = (v55 * v44 + v6);
v44 = (v54 * v58); v49 = (v42 * v52 + v49); v46 = (v53 * v45 + v46); v50 = (v10 * v29 + v50); v11 = (v47 * v46);
v46 = (v51 * v49); v57 = (v48 * v59 + v57); v48 = (v35 * v50); v10 = (v36 * v35); v35 = (v17 * v41);
v46 = (v57 * v18 + v46); v11 = (v42 * v48 + v11); v11 = (v5 * v11); v5 = a2[1]; v0 = a2[2];
v1 = (max(F(0x3b808081u), v16)); v1 = (1.0 / v1); v1 = (F(0x3e800000u) * v1); v13 = (v13 * v1); v16 = (v19 * v30);
v30 = (v44 * v60); v19 = (v19 * v43); v30 = (v12 * v29 + v30); v12 = (v36 * v34); v45 = (v19 * v63);
v19 = (v35 * v40); v29 = (v9 * v60 + v6); v9 = (v34 * v30); v16 = (v42 * v9 + v16); v9 = (v14 * v1); v1 = (v15 * v1);
v15 = (v19 * v13); v14 = (v7 * v16); v7 = (F(s37) * v36); v7 = (v7 * v37); v32 = (v33 * v7 + v32);
v46 = (v13 * v15 + v46); v45 = (v29 * v12 + v45); v12 = F(s21); v12 = (F(s29) * v12); v17 = (v47 * v12);
v12 = (v39 * v12); v13 = (v17 * v32); v6 = (v38 * v12 + v6); v13 = (v6 * v10 + v13); v6 = ((F(s80) * v46) * 2.0);
v15 = (v19 * v9); v7 = (v19 * v1); v11 = (v13 * v18 + v11); v14 = (v45 * v18 + v14); v12 = (F(s77) * v6 + 0.5);
v10 = (v15 * v9 + v11); v9 = F(0x3e19999au); v14 = (v1 * v7 + v14); v12 = (v6 * v12 + F(0x3d75c290u));
v11 = (v9 * v6 + F(0x3d4ccccdu)); v1 = ((F(s80) * v10) * 2.0); v7 = ((F(s80) * v14) * 2.0); v10 = (F(s77) * v1 + 0.5);
v6 = (v11 * v6 + F(0x3b83126fu)); v11 = (1.0 / v12); v6 = (v11 * v6); v11 = (v9 * v1 + F(0x3d4ccccdu));
v9 = (v9 * v7 + F(0x3d4ccccdu)); v11 = (v11 * v1 + F(0x3b83126fu)); v1 = (v1 * v10 + F(0x3d75c290u)); v1 = (1.0 / v1);
v9 = (v9 * v7 + F(0x3b83126fu)); v1 = (v1 * v11); v10 = F(0x3fb0852eu); v6 = clamp(v10 * v6 + F(s4), 0.0, 1.0);
v11 = clamp(v10 * v1 + F(s4), 0.0, 1.0); v1 = (sqrt(v6)); v6 = (sqrt(v11)); v11 = (v31 + v1); v12 = (v5 + v6);
v5 = (F(s77) * v7 + 0.5); v5 = (v7 * v5 + F(0x3d75c290u)); v5 = (1.0 / v5); v5 = (v5 * v9);
v5 = clamp(v10 * v5 + F(s4), 0.0, 1.0); v5 = (sqrt(v5)); v0 = (v0 + v5); v7 = (vcc_lo & 1u) != 0u ? v11 : v1;
v9 = (vcc_lo & 1u) != 0u ? v12 : v6; v10 = (vcc_lo & 1u) != 0u ? v0 : v5;
vcc_lo = scc ? 1u : 0u;
vcc_lo = s7 | vcc_lo;
scc = vcc_lo != 0u;
v13 = F(0u);
pc = (scc) ? 0x00000ae4u : 0x00000a7cu;
}
if (pc == 0x00000a7cu) {
v0 = (F(s56) - v28); v0 = (v0 * v0); v7 = (F(s57) - v27); v0 = (v7 * v7 + v0); v7 = (F(s58) - v26); s16 = cb.c[47][0];
s17 = cb.c[47][1]; s18 = cb.c[47][2]; s20 = cb.c[48][0]; s21 = cb.c[48][1]; s22 = cb.c[48][2]; s23 = cb.c[48][3];
v0 = (v7 * v7 + v0); v0 = (sqrt(v0)); v13 = F(s23); v13 = clamp(F(s22) * v0 + v13, 0.0, 1.0);
v7 = (F(s16) * v13 + v1); v9 = (F(s17) * v13 + v6); v10 = (F(s18) * v13 + v5); v7 = (v13 * (-v1) + v7);
v9 = (v13 * (-v6) + v9); v10 = (v13 * (-v5) + v10);
pc = 0x00000ae4u;
}
if (pc == 0x00000ae4u) {
v0 = F(0u);
scc = s90 != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; }
s28 = cb.c[36][2]; s24 = cb.c[36][3]; s20 = cb.c[44][2]; s16 = cb.c[20][0]; s4 = cb.c[49][0]; s5 = cb.c[49][1];
s6 = cb.c[49][2]; s7 = cb.c[49][3]; v1 = F(s28); v1 = (F(s24) * v1); v1 = (F(s20) * v1); v8 = (v1 * v8);
v11 = (F(s7) * v8); v6 = (F(s4) * v7); v5 = (F(s5) * v9); v1 = (F(s6) * v10); v14 = (vcc_lo & 1u) != 0u ? v8 : v11;
v15 = (vcc_lo & 1u) != 0u ? v7 : v6; v16 = (vcc_lo & 1u) != 0u ? v9 : v5; v17 = (vcc_lo & 1u) != 0u ? v10 : v1;
vcc_lo = s16 & 0x02000000u;
scc = vcc_lo != 0u;
scc = vcc_lo != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; }
pc = ((vcc_lo & 1u) != 0u) ? 0x00000f10u : 0x00000b64u;
}
if (pc == 0x00000b64u) {
v19 = F(0u); s40 = cb.c[20][1]; s41 = cb.c[20][2]; s42 = cb.c[50][0]; s32 = cb.c[51][0]; s33 = cb.c[51][1];
s34 = cb.c[51][2]; s35 = cb.c[51][3]; s43 = cb.c[50][1]; s44 = cb.c[50][2]; s36 = cb.c[44][3]; v11 = (F(s40) * v2);
v12 = (F(s41) * v3);
{
vec2 uv_ = vec2(v11, v12);
uv_.xy *= sc(7);
{ vec4 t_ = vec4(texture(tex[7], uv_));
v7 = t_[0];
}
}
{
vec2 uv_ = vec2(v11, v12);
uv_.xy *= sc(8);
{ vec4 t_ = vec4(texture(tex[8], uv_));
v0 = t_[0]; v1 = t_[1]; v2 = t_[2];
}
}
v3 = F(s32); v6 = (F(s34) - v3); v18 = F(s32); v5 = F(s33); v6 = (1.0 / v6); v5 = (F(s32) - v5); v5 = (1.0 / v5);
vcc_lo = s36 & 4u;
scc = vcc_lo != 0u;
v7 = (F(s42) + v7); v7 = (1.0 / v7); v18 = (F(s43) * v7 + (-v18)); v28 = clamp((-v18) * v6 + 1.0, 0.0, 1.0);
v27 = ((-F(s43)) * v7); v7 = (F(s44) * v28); v26 = ((-v27) * v5); v31 = (2.0 * v1 + -1.0); v1 = (max(F(s35), v7));
v32 = (2.0 * v0 + -1.0); v26 = clamp((-F(s33)) * v5 + v26, 0.0, 1.0); v8 = (v26 * v1); v30 = (F(s40) * v32);
v29 = (F(s41) * (-v31)); v0 = (v30 * v8 + v11); v1 = (v29 * v8 + v12);
{
vec2 uv_ = vec2(v0, v1);
uv_.xy *= sc(8);
{ vec4 t_ = vec4(texture(tex[8], uv_));
v5 = t_[0]; v6 = t_[1]; v7 = t_[2];
}
}
v0 = ((-v30) * v8 + v11); v1 = ((-v29) * v8 + v12);
{
vec2 uv_ = vec2(v0, v1);
uv_.xy *= sc(8);
{ vec4 t_ = vec4(texture(tex[8], uv_));
v8 = t_[0]; v9 = t_[1]; v10 = t_[2];
}
}
v1 = (2.0 * v2 + -1.0); v0 = (v32 + v31); v0 = (v0 + v1); v2 = (v6 * v31); v33 = ((v5 * v32) * 2.0); v6 = (v7 * v1);
v5 = (v9 * v31); v8 = ((v8 * v32) * 2.0); v1 = (v10 * v1); v33 = (2.0 * v2 + v33); v8 = (2.0 * v5 + v8);
v33 = (2.0 * v6 + v33); v8 = (2.0 * v1 + v8); v2 = (v33 - v0); v5 = (v8 - v0);
pc = (!scc) ? 0x00000d14u : 0x00000cb8u;
}
if (pc == 0x00000cb8u) {
v0 = (0.5 * v5 + 0.5); v0 = ((v2 * v0 + v0) * 0.5); v1 = (v0 * v0 + F(0x3b808081u)); s0 = cb.c[52][0];
s1 = cb.c[52][1]; v6 = ((-F(s1)) * v26 + 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (v1 >= v6)) ? 1u : 0u; vcc_hi = 0u;
v6 = (vcc_lo & 1u) != 0u ? F(0x3f800000u) : F(0u); v1 = (1.0 - F(s0)); v6 = clamp(v0 * v0 + v6, 0.0, 1.0);
v0 = (v1 * v6 + F(s0)); v19 = ((-v26) * v0 + v26);
pc = 0x00000d14u;
}
if (pc == 0x00000d14u) {
v0 = F(0u);
vcc_lo = s36 & 8u;
scc = vcc_lo != 0u;
pc = (!scc) ? 0x00000de0u : 0x00000d20u;
}
if (pc == 0x00000d20u) {
v6 = (2.0 * v27); s8 = cb.c[52][2]; s9 = cb.c[52][3]; s10 = cb.c[50][3]; v0 = (F(s8) * v28); v0 = (max(F(s35), v0));
v8 = (v26 * v0); v0 = (v30 * v8 + v11); v1 = (v29 * v8 + v12);
{
vec2 uv_ = vec2(v0, v1);
uv_.xy *= sc(7);
{ vec4 t_ = vec4(texture(tex[7], uv_));
v7 = t_[0];
}
}
v0 = ((-v30) * v8 + v11); v1 = ((-v29) * v8 + v12);
{
vec2 uv_ = vec2(v0, v1);
uv_.xy *= sc(7);
{ vec4 t_ = vec4(texture(tex[7], uv_));
v8 = t_[0];
}
}
v1 = ((-F(s10)) * v26 + 1.0); v0 = (1.0 - F(s9)); v7 = (F(s42) + v7); v7 = (1.0 / v7); v6 = (F(s43) * v7 + v6);
v7 = (F(s42) + v8); v7 = (1.0 / v7); v6 = (F(s43) * v7 + v6); v6 = (-0.5 * v6 + 1.0); v7 = (F(0x3b808081u) + v6);
vcc_lo = ((exec_lo & 1u) != 0u && (v7 >= v1)) ? 1u : 0u; vcc_hi = 0u;
v1 = (vcc_lo & 1u) != 0u ? F(0x3f800000u) : F(0u); v1 = clamp(v1 + v6, 0.0, 1.0); v0 = (v0 * v1 + F(s9));
v0 = ((-v26) * v0 + v26);
pc = 0x00000de0u;
}
if (pc == 0x00000de0u) {
v1 = F(0u);
vcc_lo = s36 & 2u;
scc = vcc_lo != 0u;
pc = (!scc) ? 0x00000ebcu : 0x00000decu;
}
if (pc == 0x00000decu) {
v1 = (v20 * v24); v1 = (v25 * v21 + v1); v1 = (v23 * v22 + v1); v1 = (max(F(0u), (-v1))); v1 = (min(1.0, v1));
s0 = cb.c[53][0]; s1 = cb.c[53][1]; s2 = cb.c[53][2]; s3 = cb.c[53][3]; s8 = cb.c[54][0]; s9 = cb.c[54][1];
s4 = cb.c[54][2]; v6 = (F(0x3b808081u) + v1); v1 = ((-v2) * v5 + 1.0); v2 = F(s0); v2 = (F(0xba83126fu) + v2);
v2 = (min(F(s1), v2)); v5 = (F(s0) - v2); v5 = (1.0 / v5); v8 = ((-v27) * v5); v7 = F(s2);
v8 = clamp(v2 * (-v5) + v8, 0.0, 1.0); v5 = (F(s9) - v7); v5 = (v8 * v5 + F(s2)); v5 = (v26 * v5 + F(0x3b808081u));
v5 = (1.0 / v5); v7 = F(s3); v5 = clamp(v5 * v6, 0.0, 1.0); v6 = (1.0 - F(s8)); v7 = (F(s4) - v7); v6 = (v6 * v5);
v2 = (v7 * v8 + F(s3)); v5 = clamp(v6 * v5 + F(s8), 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (v1 >= v2)) ? 1u : 0u; vcc_hi = 0u;
v26 = (v5 * (-v26) + v26); v5 = (vcc_lo & 1u) != 0u ? F(0x3f800000u) : F(0u); v5 = (v5 * v26);
vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v2)) ? 1u : 0u; vcc_hi = 0u;
v1 = (vcc_lo & 1u) != 0u ? v5 : v26;
pc = 0x00000ebcu;
}
if (pc == 0x00000ebcu) {
v0 = (max(max(v0, v1), v19)); s4 = cb.c[55][0]; s5 = cb.c[55][1]; s0 = cb.c[54][3]; v1 = (F(s4) - v3);
v1 = (1.0 / v1); v1 = clamp((-v18) * v1 + 1.0, 0.0, 1.0); v2 = (1.0 - F(s5)); v1 = (max(v2, v1)); v0 = (v1 * v0);
v0 = (v0 * (-v13) + v0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(s0) <= v0)) ? 1u : 0u; vcc_hi = 0u;
v1 = (vcc_lo & 1u) != 0u ? F(0x3f800000u) : F(0u); v0 = (v1 * v0);
pc = 0x00000f10u;
}
if (pc == 0x00000f10u) {
v2 = (v14); s0 = cb.c[20][0];
vcc_lo = s0 & 0x00800000u;
scc = vcc_lo != 0u;
pc = (!scc) ? 0x00000f88u : 0x00000f30u;
}
if (pc == 0x00000f30u) {
s0 = cb.c[56][0]; s1 = cb.c[56][1]; s2 = cb.c[56][2]; s3 = cb.c[56][3]; v2 = F(s0); v3 = F(s1); v1 = F(s2);
v15 = (F(s3) * v15 + v2); v16 = (F(s3) * v16 + v3); v17 = (F(s3) * v17 + v1); v5 = F(s3); v0 = (F(s3) * v0);
v2 = (F(s3) * v14); v15 = ((-F(s0)) * v5 + v15); v16 = ((-F(s1)) * v5 + v16); v17 = ((-F(s2)) * v5 + v17);
pc = 0x00000f88u;
}
if (pc == 0x00000f88u) {
    float alpha0 = (cb.c[20].w & 1u) != 0u
        ? clamp(F(0x3a4d9a68u) * v4, 0.0, 1.0) : v14;
    o1 = swz(half4(vec4(v0, 0.0, 0.0, v2)), cb.c[69].y);
    o0 = swz(half4(vec4(v15, v16, v17, alpha0)), cb.c[69].x);
    return;
}
}
)";

// Specialize copies: neither a shared GLSL string nor an existing variant's
// composed shader text changes. The vertex path shares the diffuse fog/normal
// arithmetic and reflection weight; its second UV precedes the baked-light stream.
std::string d2uRefVertexSource() {
    std::string source = kVs;
    const auto replace = [&](const char* from, const char* to) {
        source.replace(source.find(from), std::strlen(from), to);
    };
    replace("vec4 n = fetch(1u), col = fetch(2u), b = fetch(4u);",
            "vec4 n = fetch(1u), col = fetch(2u), b = fetch(5u);");
    replace("p0 = vec4(fetch(3u).xy + uvo.xy, 0.0, 0.0);",
            "p0 = vec4(fetch(3u).xy + uvo.xy, fetch(4u).xy + uvo.zw);");
    replace("T = exp2(f0.xyz * c.w * uintBitsToFloat(0x3fb8aa3bu));",
            "for (int i = 0; i < 3; ++i) T[i] = exp2(mul(mul(f0[i], c.w), uintBitsToFloat(0x3fb8aa3bu)));");
    replace("float f2 = f * f, f4 = f2 * f2;",
            "float f2 = mul(f, f), f4 = mul(f2, f2);");
    replace("rw = 0.5 * mac(cf(72).x, f4, cf(72).x);",
            "rw = mul(0.5, mac(cf(72).x, f4, cf(72).x));");
    return source;
}

std::string d2uRefFragmentSource() {
    const std::string shared = kFs;
    return shared.substr(0, shared.find("void main()")) + kFsD2uRefMain;
}

// Plant wind deformation and late alpha rejection (VS 0x53c599300, PS 0x53c5f7a00).
// The VS keeps precise operation boundaries; the PS retains the translated
// program's forward blocks, sample order, and unconstrained multiply-adds.
constexpr const char* kVsPlantMain = R"(
layout(set = 0, binding = 10, std430) readonly buffer Wind { uint w[]; } wind;
void main() {
    vec4 x = fetch(0u), n = fetch(2u), t = fetch(3u), col = fetch(4u), b = fetch(6u);
    vec3 axis = vec3(mac(2.0, b.y, -1.0), mac(2.0, b.z, -1.0), mac(2.0, b.w, -1.0));
    uint idx = floatBitsToUint(fetch(1u).x);
    uvec4 wi = cb.c[76];
    uint a = wi.x + idx * wi.y;
    vec4 windValue = vec4(0.0);
    if ((wi.y != 0u ? idx < wi.z : wi.z != 0u) && a < wi.w) {
        uint i = a >> 2;
        windValue = uintBitsToFloat(uvec4(wind.w[i], wind.w[i + 1u], wind.w[i + 2u], wind.w[i + 3u]));
    }
    vec4 W[3] = vec4[3](cf(0), cf(1), cf(2));
    vec3 world;
    for (int i = 0; i < 3; ++i)
        world[i] = mac(W[i].w, x.w, mac(W[i].z, x.z, mac(W[i].y, x.y, mul(W[i].x, x.x))));
    float along = mac(axis.z, windValue.z, mac(axis.y, windValue.y, mul(windValue.x, axis.x)));
    vec3 perpendicular;
    for (int i = 0; i < 3; ++i) perpendicular[i] = mac(along, -axis[i], windValue[i]);
    float length2 = mac(windValue.w, windValue.w,
                        mac(perpendicular.z, perpendicular.z,
                            mac(perpendicular.y, perpendicular.y, mul(perpendicular.x, perpendicular.x))));
    float bend = mul(uintBitsToFloat(0x3e4ccccdu), sqrt(length2));
    for (int i = 0; i < 3; ++i)
        world[i] = mac(mac(bend, axis[i], perpendicular[i]), b.x, world[i]);
    float homogeneous = mac(b.x, windValue.w, x.w);
    vec4 pos;
    for (int i = 0; i < 4; ++i) {
        vec4 P = cf(6 + i);
        pos[i] = mac(P.w, homogeneous, mac(P.z, world.z, mac(P.y, world.y, mul(P.x, world.x))));
    }
    vec3 T = vec3(1.0), scatter = vec3(0.0);
    uint flags = cb.c[14].z;
    if ((flags & 0x20000u) != 0u) {
        vec4 f0 = cf(15), f1 = cf(16), f2 = cf(17), f3 = cf(18);
        for (int i = 0; i < 3; ++i)
            T[i] = exp2(mul(uintBitsToFloat(0x3fb8aa3bu), mul(f0[i], pos.w)));
        precise vec3 d = world - cf(5).xyz;
        float dd = mac(d.z, d.z, mac(d.y, d.y, mul(d.x, d.x)));
        float fd = mac(cf(19).z, d.z, mac(cf(19).y, d.y, mul(cf(19).x, d.x)));
        float cosine = mul(-inversesqrt(dd), fd);
        float r = inversesqrt(mac(f2.w, cosine, f1.w));
        float rr = mul(mul(r, r), r);
        float cc = mac(cosine, cosine, f3.w);
        for (int i = 0; i < 3; ++i) {
            float phase = mac(f2[i], rr, mul(f1[i], cc));
            float amp = mul(f3[i], phase);
            scatter[i] = mac(amp, -T[i], mul(f3.w, amp));
        }
    }
    if ((flags & 0x40000u) != 0u) scatter = vec3(0.0);
    const float k = uintBitsToFloat(0x40008102u);
    vec3 normal = vec3(mac(k, n.x, -1.0), mac(k, n.y, -1.0), mac(k, n.z, -1.0));
    vec3 tangent = vec3(mac(k, t.x, -1.0), mac(k, t.y, -1.0), mac(k, t.z, -1.0));
    gl_Position = pos;
    if (cb.c[69].z != 0u) gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;
    precise vec2 uv = cf(13).xy + fetch(5u).xy;
    p0 = vec4(uv, 0.0, 0.0);
    p1 = vec4(mul(col.r, T.r), mul(T.g, col.g), mul(T.b, col.b), mul(cf(14).y, col.a));
    p2 = vec4(scatter, mac(-cf(14).x, clamp(mul(uintBitsToFloat(0x3f808102u), n.w), 0.0, 1.0), cf(14).x));
    p3 = vec4(dot3(W[0], normal), dot3(W[1], normal), dot3(W[2], normal), 0.0);
    p4 = vec4(0.0);
    p5 = vec4(T, 0.0);
    p6 = vec4(dot3(W[0], tangent), dot3(W[1], tangent), dot3(W[2], tangent), mac(k, t.w, -1.0));
    p7 = vec4(world, 0.0);
}
)";

constexpr const char* kFsPlantMain = R"(
float F(uint x) { return uintBitsToFloat(x); }
void main() {
uint exec_hi = 0u; uint exec_lo = 0u; uint m0 = 0u; uint s0 = 0u; uint s1 = 0u; uint s10 = 0u; uint s14 = 0u;
uint s15 = 0u; uint s16 = 0u; uint s17 = 0u; uint s18 = 0u; uint s19 = 0u; uint s2 = 0u; uint s20 = 0u; uint s21 = 0u;
uint s22 = 0u; uint s23 = 0u; uint s24 = 0u; uint s25 = 0u; uint s28 = 0u; uint s29 = 0u; uint s3 = 0u; uint s32 = 0u;
uint s33 = 0u; uint s34 = 0u; uint s35 = 0u; uint s36 = 0u; uint s37 = 0u; uint s38 = 0u; uint s39 = 0u; uint s4 = 0u;
uint s40 = 0u; uint s41 = 0u; uint s42 = 0u; uint s43 = 0u; uint s44 = 0u; uint s45 = 0u; uint s46 = 0u; uint s47 = 0u;
uint s48 = 0u; uint s49 = 0u; uint s5 = 0u; uint s50 = 0u; uint s51 = 0u; uint s52 = 0u; uint s53 = 0u; uint s54 = 0u;
uint s55 = 0u; uint s56 = 0u; uint s6 = 0u; uint s64 = 0u; uint s65 = 0u; uint s66 = 0u; uint s67 = 0u; uint s68 = 0u;
uint s69 = 0u; uint s7 = 0u; uint s70 = 0u; uint s71 = 0u; uint s76 = 0u; uint s77 = 0u; uint s78 = 0u; uint s79 = 0u;
uint s8 = 0u; uint s84 = 0u; uint s85 = 0u; uint s86 = 0u; uint s87 = 0u; uint s9 = 0u; float v0 = 0.0; float v1 = 0.0;
float v10 = 0.0; float v106 = 0.0; float v11 = 0.0; float v12 = 0.0; float v13 = 0.0; float v14 = 0.0; float v15 = 0.0;
float v16 = 0.0; float v17 = 0.0; float v18 = 0.0; float v19 = 0.0; float v2 = 0.0; float v20 = 0.0; float v21 = 0.0;
float v22 = 0.0; float v23 = 0.0; float v24 = 0.0; float v25 = 0.0; float v26 = 0.0; float v27 = 0.0; float v28 = 0.0;
float v29 = 0.0; float v3 = 0.0; float v30 = 0.0; float v31 = 0.0; float v32 = 0.0; float v33 = 0.0; float v34 = 0.0;
float v35 = 0.0; float v36 = 0.0; float v37 = 0.0; float v38 = 0.0; float v39 = 0.0; float v4 = 0.0; float v40 = 0.0;
float v41 = 0.0; float v42 = 0.0; float v43 = 0.0; float v44 = 0.0; float v45 = 0.0; float v46 = 0.0; float v47 = 0.0;
float v5 = 0.0; float v6 = 0.0; float v7 = 0.0; float v8 = 0.0; float v9 = 0.0; uint vcc_hi = 0u; uint vcc_lo = 0u;
bool scc = false; uint pc = 0x00000000u; exec_lo = 1u; v2 = gl_FragCoord.x; v3 = gl_FragCoord.y; v4 = gl_FragCoord.w;
{
pc = 0x00000000u;
}
if (pc == 0x00000000u) {
vcc_hi = 0x00000212u; m0 = s14; { uint lo_ = exec_lo; uint hi_ = exec_hi; s14 = lo_; s15 = hi_; }
{ uint lo_ = exec_lo; uint hi_ = exec_hi; exec_lo = lo_; exec_hi = hi_; } s64 = cb.c[21][0]; s65 = cb.c[21][1];
s66 = cb.c[21][2]; s67 = cb.c[21][3]; v15 = a0[0]; v16 = a0[1]; v5 = (F(s66) * v15); v6 = (F(s67) * v16);
{
vec2 uv_ = vec2(v5, v6); uv_ *= sc(0); { vec4 t_ = vec4(texture(tex[0], uv_)); v7 = t_[0]; v8 = t_[1];
}
}
v10 = a3[1]; v11 = a6[2]; v20 = a6[0]; v13 = (v10 * v11); v9 = a3[2]; v17 = a6[1]; v5 = a6[3]; v12 = a3[0];
v14 = (v9 * v20); s67 = cb.c[20][0]; v13 = (v17 * (-v9) + v13); v21 = (v5 * v13); v6 = (v12 * v17);
v14 = (v11 * (-v12) + v14); v19 = (v5 * v14); v6 = (v20 * (-v10) + v6); v18 = (v5 * v6); s66 = s67 & 8u; scc = s66 != 0u;
s24 = cb.c[20][1]; s25 = cb.c[20][2]; scc = s66 != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; s28 = lo_; s29 = hi_; }
{ uint lo_ = s28; uint hi_ = s29; vcc_lo = lo_; vcc_hi = hi_; } v5 = (F(s24) * v2); v6 = (F(s25) * v3);
v13 = F(0x3f800000u); v14 = F(0x3f800000u); v22 = (2.0 * v7 + -1.0); v23 = (v22 * v22); v7 = (2.0 * v8 + -1.0);
v23 = (v7 * v7 + v23); v8 = clamp(1.0 - v23, 0.0, 1.0); v20 = (v20 * v22); v8 = (sqrt(v8)); v17 = (v17 * v22);
v20 = (v7 * v21 + v20); v20 = (v8 * v12 + v20); v12 = (v19 * v7 + v17); v17 = (v11 * v22); v11 = (v20 * v20);
v12 = (v8 * v10 + v12); v17 = (v7 * v18 + v17); v10 = (v12 * v12 + v11); v7 = (v9 * v8 + v17); v10 = (v7 * v7 + v10);
v8 = (inversesqrt(v10)); v17 = (v8 * v20); v18 = (v8 * v12); v19 = (v8 * v7);
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = lo_; vcc_hi = hi_; }
pc = ((vcc_lo & 1u) == 0u) ? 0x00000180u : 0x0000016cu;
}
if (pc == 0x0000016cu) {
{
vec2 uv_ = vec2(v5, v6); uv_ *= sc(1); { vec4 t_ = vec4(texture(tex[1], uv_)); v13 = t_[0]; v14 = t_[1];
}
}
pc = 0x00000180u;
}
if (pc == 0x00000180u) {
{ uint lo_ = exec_lo & s28; uint hi_ = exec_hi & s29; vcc_lo = lo_; vcc_hi = hi_; } scc = (vcc_lo | vcc_hi) != 0u;
pc = ((vcc_lo & 1u) == 0u) ? 0x000001b4u : 0x00000188u;
}
if (pc == 0x00000188u) {
s16 = cb.c[22][0]; s17 = cb.c[22][1]; s18 = cb.c[22][2]; s19 = cb.c[22][3]; v7 = F(s17); v8 = F(s19);
v13 = (F(s16) * v13 + v7); v14 = (v14 * F(s18) + v8); pc = 0x000001b4u;
}
if (pc == 0x000001b4u) {
{ uint lo_ = exec_lo & s28; uint hi_ = exec_hi & s29; vcc_lo = lo_; vcc_hi = hi_; } scc = (vcc_lo | vcc_hi) != 0u;
pc = ((vcc_lo & 1u) == 0u) ? 0x000001e8u : 0x000001bcu;
}
if (pc == 0x000001bcu) {
s16 = cb.c[22][0]; s17 = cb.c[22][1]; s18 = cb.c[22][2]; s19 = cb.c[22][3]; v7 = F(s17); v8 = F(s19);
v13 = (F(s16) * v13 + v7); v14 = (v14 * F(s18) + v8); pc = 0x000001e8u;
}
if (pc == 0x000001e8u) {
v8 = (v13 * v14); { uint lo_ = exec_lo & s28; uint hi_ = exec_hi & s29; vcc_lo = lo_; vcc_hi = hi_; }
scc = (vcc_lo | vcc_hi) != 0u; s16 = cb.c[23][0]; s17 = cb.c[23][1]; s18 = cb.c[23][2]; s19 = cb.c[23][3];
v7 = (1.0 - F(s16)); v10 = (1.0 - F(s17)); v9 = (1.0 - F(s18)); v7 = (v7 * v8 + F(s16)); v31 = (v10 * v8 + F(s17));
v29 = (v9 * v8 + F(s18)); v32 = (v7); v28 = (v31); v26 = (v29);
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = lo_; vcc_hi = hi_; }
pc = ((vcc_lo & 1u) == 0u) ? 0x00000288u : 0x0000024cu;
}
if (pc == 0x0000024cu) {
s16 = cb.c[24][0]; s17 = cb.c[24][1]; s18 = cb.c[24][2]; s19 = cb.c[24][3]; v8 = F(s19); v10 = F(s17);
v8 = (F(s18) + v8); v11 = (v7 * F(s16) + v10); v9 = (v31 * F(s16) + v10); v10 = (F(s16) * v29 + v10); v32 = (v8 * v11);
v28 = (v8 * v9); v26 = (v8 * v10); pc = 0x00000288u;
}
if (pc == 0x00000288u) {
    v25 = a7.x; v24 = a7.y; v23 = a7.z;
    v8 = 0.0; v30 = 0.0; v27 = 0.0;
    for (int i = 0; i < 4; ++i) {
        if ((s67 & (0x10000u << uint(i))) == 0u) continue;
        vec4 pp = cf(25 + i), light = cf(29 + i);
        float dx = v25 - pp.x, dy = v24 - pp.y, dz = v23 - pp.z;
        float dd = dx * dx, nd = v17 * dx;
        dd = dy * dy + dd; nd = v18 * dy + nd;
        dd = dz * dz + dd;
        float invd = inversesqrt(dd);
        nd = dz * v19 + nd;
        float lam = clamp(-invd * nd, 0.0, 1.0);
        if (lam > 0.0) {
            float distance_ = sqrt(dd);
            float o = max(0.0, distance_ - light.w);
            float po = pp.w * o;
            float att = 1.0 / (po * o + 1.0);
            att = clamp(F(0x3f808102u) * att + F(0xbb810205u), 0.0, 1.0);
            float weight = lam * att;
            if (i == 0) {
                v8 = light.r * weight; v30 = light.g * weight; v27 = light.b * weight;
            } else {
                v8 = light.r * weight + v8;
                v30 = light.g * weight + v30;
                v27 = light.b * weight + v27;
            }
        }
    }
    pc = 0x0000058cu;
}
if (pc == 0x0000058cu) {
v38 = a1[0]; v41 = a5[0]; vcc_lo = 0x3e19999au; vcc_hi = 0xbdbc49cau; s68 = cb.c[33][0]; s69 = cb.c[33][1];
s70 = cb.c[33][2]; s71 = cb.c[33][3]; s40 = cb.c[41][0]; s41 = cb.c[41][1]; s42 = cb.c[41][2]; s43 = cb.c[41][3];
s44 = cb.c[42][0]; s45 = cb.c[42][1]; s46 = cb.c[42][2]; s47 = cb.c[42][3]; s16 = cb.c[37][0]; s17 = cb.c[37][1];
s18 = cb.c[37][2]; s19 = cb.c[37][3]; s20 = cb.c[38][0]; s21 = cb.c[38][1]; s22 = cb.c[38][2]; s23 = cb.c[38][3];
s56 = cb.c[36][1]; s32 = cb.c[34][0]; s33 = cb.c[34][1]; s34 = cb.c[34][2]; s35 = cb.c[34][3]; s36 = cb.c[35][0];
s37 = cb.c[35][1]; s38 = cb.c[35][2]; s39 = cb.c[35][3]; s76 = cb.c[39][0]; s77 = cb.c[39][1]; s78 = cb.c[39][2];
s79 = cb.c[39][3]; s84 = cb.c[40][0]; s85 = cb.c[40][1]; s86 = cb.c[40][2]; s87 = cb.c[40][3]; s48 = cb.c[45][0];
s49 = cb.c[45][1]; s50 = cb.c[45][2]; s51 = cb.c[45][3]; s52 = cb.c[46][0]; s53 = cb.c[46][1]; s54 = cb.c[46][2];
s55 = cb.c[46][3]; v12 = (v25 - F(s68)); v10 = (v12 * v12); v9 = (v24 - F(s69)); v11 = (v23 - F(s70));
v10 = (v9 * v9 + v10); v10 = (v11 * v11 + v10); v10 = (inversesqrt(v10)); v13 = (v10 * v12 + F(s40));
v22 = (v10 * v9 + F(s41)); v21 = (v10 * v12); v12 = (v13 * v17); v13 = (v13 * v13); v20 = (v10 * v11);
v33 = (v22 * v18 + v12); v12 = (v22 * v22 + v13); v22 = (v10 * v11 + F(s42)); v12 = (v22 * v22 + v12);
v12 = (inversesqrt(v12)); v33 = (v19 * v22 + v33); v11 = clamp((-v12) * v33, 0.0, 1.0);
v12 = (v10 * v9 + F(0xbe99999au)); v11 = (log2(v11)); v33 = (2.0 * F(s56)); s19 = cb.c[73][0]; v39 = (1.0 / F(s78));
s35 = s67 & 4u; scc = s35 != 0u; v22 = (v10 * v9); v10 = (v21 * v21); v9 = (v21 * v17); v10 = (v12 * v12 + v10);
v9 = (v18 * v12 + v9); v10 = (v20 * v20 + v10); v10 = (inversesqrt(v10)); v9 = (v19 * v20 + v9);
v9 = clamp((-v10) * v9, 0.0, 1.0); v9 = (log2(v9)); v10 = (F(s23) * v9); v9 = (F(s23) * v11); v44 = (exp2(v9));
v35 = (exp2(v10)); v10 = (F(s44) * v17); v10 = (F(s45) * v18 + v10); v10 = (F(s46) * v19 + v10);
v10 = (max(F(0u), (-v10))); v9 = (F(s48) * v44); v34 = (min(1.0, v10)); v10 = (F(s52) * v35); v12 = F(s36);
v9 = (v9 * v7); v11 = (v10 * v34); v10 = (F(s20) * v12); v9 = (v33 * v11 + v9); v11 = (v38 * v10); v7 = (F(s48) * v7);
v13 = (v41 * v10); v40 = (v11 * v9);
{
vec2 uv_ = vec2(v5, v6); uv_ *= sc(3); { vec4 t_ = vec4(texture(tex[3], uv_)); v9 = t_[0]; v10 = t_[1]; v11 = t_[2];
v12 = t_[3];
}
}
v5 = a4[0]; v6 = (F(s52) * v14); s4 = cb.c[43][0]; s5 = cb.c[43][1]; s6 = cb.c[43][2]; s7 = cb.c[43][3]; v46 = (v5 + v8);
v5 = (F(s40) * v17); v5 = (F(s41) * v18 + v5); v5 = (F(s42) * v19 + v5); v8 = (max(F(0u), (-v5))); v47 = F(s4);
v42 = (min(1.0, v8)); v45 = (v7 * v42); s4 = cb.c[36][0]; v45 = (v34 * v6 + v45); v36 = clamp(max(F(0u), v5), 0.0, 1.0);
{
vec2 uv_ = vec2(v15, v16); uv_ *= sc(4); { vec4 t_ = vec4(texture(tex[4], uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2];
v8 = t_[3];
}
}
v43 = (v35 * v32); v12 = (v12 * v39); v37 = (v14 * v12); v9 = (v9 * v39); v12 = (v13 * v46 + v37); v13 = (F(s65) * v16);
v43 = (v43 * v12 + v40); v12 = (F(s64) * v15); v15 = (F(s39) * v47); v16 = (F(s50) * v44); v40 = (v10 * v39);
v10 = F(s32); v47 = (F(s16) * v10); v10 = (v41 * v47); v10 = (v10 * v46); v10 = (v9 * v14 + v10); v9 = (F(s49) * v44);
v44 = (v9 * v31); v5 = (v5 * v5); v15 = (v15 * v5 + F(s84)); v15 = (v45 * v47 + v15); v46 = (F(s49) * v31); v9 = F(s5);
v41 = (v6 * v6); v6 = (v32 * v10); v10 = (F(s53) * v35); v10 = (v10 * v34); v44 = (v33 * v10 + v44); v9 = (F(s39) * v9);
v32 = (v9 * v41 + F(s85)); v10 = a5[2]; v9 = (v7 * v7); v7 = F(s19); v7 = (F(s48) * v7); v15 = (v36 * v7 + v15);
v7 = (1.0 - F(s4)); v31 = (v38 * v15); v15 = (v7 * v14 + F(s4)); v31 = (v15 * v6 + v31); v47 = (v5 * v31);
{
vec2 uv_ = vec2(v12, v13); uv_ *= sc(5); { vec4 t_ = vec4(texture(tex[5], uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2];
}
}
v12 = F(0x3e19999au); v38 = a5[1]; v13 = (v11 * v39); v11 = (F(s50) * v29); v39 = (v46 * v42); v31 = (v16 * v29);
v16 = a4[1]; v30 = (v16 + v30); v29 = (v11 * v42); v11 = F(s37); v45 = (F(s21) * v11); v46 = (v38 * v45);
v11 = F(0x3fb0852eu); v16 = a2[0]; v47 = (v43 * v5 + v47); v5 = ((F(s76) * v47) * 2.0); v43 = (F(vcc_lo) * v5 + 0.5);
v42 = (v12 * v5 + F(0x3d4ccccdu)); v43 = (v5 * v43 + F(0x3d75c290u)); v43 = (1.0 / v43);
v5 = (v42 * v5 + F(0x3b83126fu)); v42 = a1[1]; v5 = (v43 * v5); v43 = (v42 * v45); v47 = (v43 * v44); v43 = F(s33);
v43 = (F(s17) * v43); v44 = (F(s53) * v14); v38 = (v38 * v43); v44 = (v44 * v34 + v39); v32 = (v44 * v43 + v32);
v45 = (v46 * v30 + v37); v30 = (v38 * v30); v38 = F(s49); v38 = (F(s19) * v38); v32 = (v36 * v38 + v32);
v30 = (v40 * v14 + v30); v32 = (v42 * v32); v46 = (v35 * v28); v28 = (v28 * v30); v32 = (v15 * v28 + v32);
v28 = (v41 * v32); v47 = (v45 * v46 + v47); v28 = (v47 * v6 + v28); v6 = ((F(s76) * v28) * 2.0);
v30 = (v12 * v6 + F(0x3d4ccccdu)); v30 = (v30 * v6 + F(0x3b83126fu)); v28 = (F(vcc_lo) * v6 + 0.5);
v28 = (v6 * v28 + F(0x3d75c290u)); v6 = a1[2]; v28 = (1.0 / v28); v28 = (v28 * v30); v30 = (F(s54) * v35);
v30 = (v30 * v34); v31 = (v33 * v30 + v31); v30 = F(s38); v30 = (F(s22) * v30); v38 = (v6 * v30); v33 = (v10 * v30);
v30 = a4[2]; v30 = (v30 + v27); v27 = F(s6); v27 = (F(s39) * v27); v32 = (v27 * v9 + F(s86)); v38 = (v38 * v31);
v31 = (v35 * v26); v37 = (v30 * v33 + v37); v38 = (v37 * v31 + v38); v5 = clamp(v11 * v5 + F(vcc_hi), 0.0, 1.0);
v33 = clamp(v11 * v28 + F(vcc_hi), 0.0, 1.0); v5 = (sqrt(v5)); v0 = (sqrt(v33)); v27 = a2[1]; v28 = a2[2];
v1 = (F(s54) * v14); v1 = (v1 * v34 + v29); v29 = F(s34); v29 = (F(s18) * v29); v10 = (v10 * v29); v10 = (v10 * v30);
v32 = (v1 * v29 + v32); v10 = (v13 * v14 + v10); v13 = F(s50); v13 = (F(s19) * v13); v32 = (v36 * v13 + v32);
v1 = (v6 * v32); v6 = (v26 * v10); v1 = (v15 * v6 + v1); v1 = (v9 * v1); v1 = (v38 * v7 + v1);
v1 = ((F(s76) * v1) * 2.0); v6 = (F(vcc_lo) * v1 + 0.5); v6 = (v1 * v6 + F(0x3d75c290u));
v7 = (v12 * v1 + F(0x3d4ccccdu)); v6 = (1.0 / v6); v1 = (v7 * v1 + F(0x3b83126fu)); v1 = (v6 * v1);
vcc_lo = s67 & 0x00400000u; scc = vcc_lo != 0u; v1 = clamp(v11 * v1 + F(vcc_hi), 0.0, 1.0); scc = vcc_lo != 0u;
vcc_lo = ((exec_lo & 1u) != 0u && (s35 != 0u)) ? 1u : 0u; vcc_hi = 0u; v1 = (sqrt(v1)); v16 = (v16 + v5);
v7 = (v27 + v0); v9 = (v28 + v1); v6 = (vcc_lo & 1u) != 0u ? v16 : v5; v7 = (vcc_lo & 1u) != 0u ? v7 : v0;
v9 = (vcc_lo & 1u) != 0u ? v9 : v1; vcc_hi = scc ? 0u : 1u; scc = s35 != 0u; vcc_lo = scc ? 1u : 0u;
vcc_lo = vcc_hi | vcc_lo; scc = vcc_lo != 0u; v13 = F(0u); pc = (scc) ? 0x00000af0u : 0x00000a88u;
}
if (pc == 0x00000a88u) {
v6 = (F(s68) - v25); v6 = (v6 * v6); v7 = (F(s69) - v24); v6 = (v7 * v7 + v6); s16 = cb.c[47][0]; s17 = cb.c[47][1];
s18 = cb.c[47][2]; s19 = cb.c[47][3]; s20 = cb.c[48][0]; s21 = cb.c[48][1]; s22 = cb.c[48][2]; s23 = cb.c[48][3];
v7 = (F(s70) - v23); v6 = (v7 * v7 + v6); v6 = (sqrt(v6)); v13 = F(s23); v13 = clamp(F(s22) * v6 + v13, 0.0, 1.0);
v6 = (F(s16) * v13 + v5); v7 = (F(s17) * v13 + v0); v9 = (F(s18) * v13 + v1); v6 = (v13 * (-v5) + v6);
v7 = (v13 * (-v0) + v7); v9 = (v13 * (-v1) + v9); pc = 0x00000af0u;
}
if (pc == 0x00000af0u) {
s4 = cb.c[70][0]; vcc_lo = ((exec_lo & 1u) != 0u && (F(s4) > v8)) ? 1u : 0u; vcc_hi = 0u;
{ uint lo_ = s14 & ~vcc_lo; uint hi_ = s15 & ~vcc_hi; s14 = lo_; s15 = hi_; } scc = (s14 | s15) != 0u;
pc = (!scc) ? 0x0000100cu : 0x00000b0cu;
}
if (pc == 0x00000b0cu) {
s24 = cb.c[36][3]; { uint lo_ = s14; uint hi_ = s15; exec_lo = lo_; exec_hi = hi_; } scc = s66 != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; } s25 = cb.c[36][2];
s20 = cb.c[44][2]; s16 = cb.c[20][0]; s4 = cb.c[49][0]; s5 = cb.c[49][1]; s6 = cb.c[49][2]; s7 = cb.c[49][3]; v0 = F(0u);
v1 = F(s25); v1 = (F(s24) * v1); v1 = (F(s20) * v1); v10 = (v1 * v8); v11 = (F(s7) * v10); v8 = (F(s4) * v6);
v5 = (F(s5) * v7); v1 = (F(s6) * v9); v14 = (vcc_lo & 1u) != 0u ? v10 : v11; v15 = (vcc_lo & 1u) != 0u ? v6 : v8;
v16 = (vcc_lo & 1u) != 0u ? v7 : v5; v23 = (vcc_lo & 1u) != 0u ? v9 : v1; vcc_lo = s16 & 0x02000000u; scc = vcc_lo != 0u;
scc = vcc_lo != 0u; { uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; }
pc = ((vcc_lo & 1u) != 0u) ? 0x00000f38u : 0x00000b8cu;
}
if (pc == 0x00000b8cu) {
v25 = F(0u); s40 = cb.c[20][1]; s41 = cb.c[20][2]; s42 = cb.c[50][0]; s32 = cb.c[51][0]; s33 = cb.c[51][1];
s34 = cb.c[51][2]; s35 = cb.c[51][3]; s43 = cb.c[50][1]; s44 = cb.c[50][2]; s36 = cb.c[44][3]; v11 = (F(s40) * v2);
v12 = (F(s41) * v3);
{
vec2 uv_ = vec2(v11, v12); uv_ *= sc(7); { vec4 t_ = vec4(texture(tex[7], uv_)); v7 = t_[0];
}
}
{
vec2 uv_ = vec2(v11, v12); uv_ *= sc(8); { vec4 t_ = vec4(texture(tex[8], uv_)); v0 = t_[0]; v1 = t_[1]; v2 = t_[2];
}
}
v3 = F(s32); v6 = (F(s34) - v3); v24 = F(s32); v5 = F(s33); v6 = (1.0 / v6); v5 = (F(s32) - v5); v5 = (1.0 / v5);
vcc_lo = s36 & 4u; scc = vcc_lo != 0u; v7 = (F(s42) + v7); v7 = (1.0 / v7); v24 = (F(s43) * v7 + (-v24));
v28 = clamp((-v24) * v6 + 1.0, 0.0, 1.0); v27 = ((-F(s43)) * v7); v7 = (F(s44) * v28); v26 = ((-v27) * v5);
v31 = (2.0 * v1 + -1.0); v1 = (max(F(s35), v7)); v32 = (2.0 * v0 + -1.0); v26 = clamp((-F(s33)) * v5 + v26, 0.0, 1.0);
v8 = (v26 * v1); v30 = (F(s40) * v32); v29 = (F(s41) * (-v31)); v0 = (v30 * v8 + v11); v1 = (v29 * v8 + v12);
{
vec2 uv_ = vec2(v0, v1); uv_ *= sc(8); { vec4 t_ = vec4(texture(tex[8], uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2];
}
}
v0 = ((-v30) * v8 + v11); v1 = ((-v29) * v8 + v12);
{
vec2 uv_ = vec2(v0, v1); uv_ *= sc(8); { vec4 t_ = vec4(texture(tex[8], uv_)); v8 = t_[0]; v9 = t_[1]; v10 = t_[2];
}
}
v1 = (2.0 * v2 + -1.0); v0 = (v32 + v31); v0 = (v0 + v1); v2 = (v6 * v31); v33 = ((v5 * v32) * 2.0); v6 = (v7 * v1);
v5 = (v9 * v31); v8 = ((v8 * v32) * 2.0); v1 = (v10 * v1); v33 = (2.0 * v2 + v33); v8 = (2.0 * v5 + v8);
v33 = (2.0 * v6 + v33); v8 = (2.0 * v1 + v8); v2 = (v33 - v0); v5 = (v8 - v0); pc = (!scc) ? 0x00000d3cu : 0x00000ce0u;
}
if (pc == 0x00000ce0u) {
v0 = (0.5 * v5 + 0.5); v0 = ((v2 * v0 + v0) * 0.5); v1 = (v0 * v0 + F(0x3b808081u)); s0 = cb.c[52][0]; s1 = cb.c[52][1];
v6 = ((-F(s1)) * v26 + 1.0); vcc_lo = ((exec_lo & 1u) != 0u && (v1 >= v6)) ? 1u : 0u; vcc_hi = 0u;
v6 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0; v1 = (1.0 - F(s0)); v6 = clamp(v0 * v0 + v6, 0.0, 1.0); v0 = (v1 * v6 + F(s0));
v25 = ((-v26) * v0 + v26); pc = 0x00000d3cu;
}
if (pc == 0x00000d3cu) {
v0 = F(0u); vcc_lo = s36 & 8u; scc = vcc_lo != 0u; pc = (!scc) ? 0x00000e08u : 0x00000d48u;
}
if (pc == 0x00000d48u) {
v6 = (2.0 * v27); s8 = cb.c[52][2]; s9 = cb.c[52][3]; s10 = cb.c[50][3]; v0 = (F(s8) * v28); v0 = (max(F(s35), v0));
v8 = (v26 * v0); v0 = (v30 * v8 + v11); v1 = (v29 * v8 + v12);
{
vec2 uv_ = vec2(v0, v1); uv_ *= sc(7); { vec4 t_ = vec4(texture(tex[7], uv_)); v7 = t_[0];
}
}
v0 = ((-v30) * v8 + v11); v1 = ((-v29) * v8 + v12);
{
vec2 uv_ = vec2(v0, v1); uv_ *= sc(7); { vec4 t_ = vec4(texture(tex[7], uv_)); v8 = t_[0];
}
}
v1 = ((-F(s10)) * v26 + 1.0); v0 = (1.0 - F(s9)); v7 = (F(s42) + v7); v7 = (1.0 / v7); v6 = (F(s43) * v7 + v6);
v7 = (F(s42) + v8); v7 = (1.0 / v7); v6 = (F(s43) * v7 + v6); v6 = (-0.5 * v6 + 1.0); v7 = (F(0x3b808081u) + v6);
vcc_lo = ((exec_lo & 1u) != 0u && (v7 >= v1)) ? 1u : 0u; vcc_hi = 0u; v1 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0;
v1 = clamp(v1 + v6, 0.0, 1.0); v0 = (v0 * v1 + F(s9)); v0 = ((-v26) * v0 + v26); pc = 0x00000e08u;
}
if (pc == 0x00000e08u) {
v1 = F(0u); vcc_lo = s36 & 2u; scc = vcc_lo != 0u; pc = (!scc) ? 0x00000ee4u : 0x00000e14u;
}
if (pc == 0x00000e14u) {
v1 = (v17 * v21); v1 = (v22 * v18 + v1); v1 = (v20 * v19 + v1); v1 = (max(F(0u), (-v1))); s0 = cb.c[53][0];
s1 = cb.c[53][1]; s2 = cb.c[53][2]; s3 = cb.c[53][3]; s8 = cb.c[54][0]; s9 = cb.c[54][1]; s4 = cb.c[54][2];
v1 = (min(1.0, v1)); v1 = (F(0x3b808081u) + v1); v2 = ((-v2) * v5 + 1.0); v6 = F(s0); v6 = (F(0xba83126fu) + v6);
v6 = (min(F(s1), v6)); v7 = (F(s0) - v6); v7 = (1.0 / v7); v10 = ((-v27) * v7); v8 = F(s2);
v10 = clamp(v6 * (-v7) + v10, 0.0, 1.0); v7 = (F(s9) - v8); v7 = (v10 * v7 + F(s2)); v7 = (v26 * v7 + F(0x3b808081u));
v8 = (1.0 / v7); v7 = F(s3); v1 = clamp(v8 * v1, 0.0, 1.0); v8 = (1.0 - F(s8)); v9 = (F(s4) - v7); v7 = (v8 * v1);
v6 = (v9 * v10 + F(s3)); v1 = clamp(v7 * v1 + F(s8), 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (v2 >= v6)) ? 1u : 0u; vcc_hi = 0u; v2 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0;
v26 = (v1 * (-v26) + v26); v2 = (v2 * v26); vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v6)) ? 1u : 0u; vcc_hi = 0u;
v1 = (vcc_lo & 1u) != 0u ? v2 : v26; pc = 0x00000ee4u;
}
if (pc == 0x00000ee4u) {
v0 = (max(max(v0, v1), v25)); s4 = cb.c[55][0]; s5 = cb.c[55][1]; s0 = cb.c[54][3]; v1 = (F(s4) - v3); v1 = (1.0 / v1);
v1 = clamp((-v24) * v1 + 1.0, 0.0, 1.0); v2 = (1.0 - F(s5)); v1 = (max(v2, v1)); v0 = (v1 * v0); v0 = (v0 * (-v13) + v0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(s0) <= v0)) ? 1u : 0u; vcc_hi = 0u; v1 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0;
v0 = (v1 * v0); pc = 0x00000f38u;
}
if (pc == 0x00000f38u) {
v2 = (v14); s0 = cb.c[20][0]; vcc_lo = s0 & 0x00800000u; scc = vcc_lo != 0u; pc = (!scc) ? 0x00000fb0u : 0x00000f58u;
}
if (pc == 0x00000f58u) {
s0 = cb.c[56][0]; s1 = cb.c[56][1]; s2 = cb.c[56][2]; s3 = cb.c[56][3]; v2 = F(s0); v3 = F(s1); v1 = F(s2);
v15 = (F(s3) * v15 + v2); v16 = (F(s3) * v16 + v3); v23 = (F(s3) * v23 + v1); v5 = F(s3); v0 = (F(s3) * v0);
v2 = (F(s3) * v14); v15 = ((-F(s0)) * v5 + v15); v16 = ((-F(s1)) * v5 + v16); v23 = ((-F(s2)) * v5 + v23);
pc = 0x00000fb0u;
}
if (pc == 0x00000fb0u) {
float alpha0 = (cb.c[20].w & 1u) != 0u ? clamp(F(0x3a4d9a68u) * v4, 0.0, 1.0) : v14; if ((s14 & 1u) == 0u) discard;
o1 = swz(half4(vec4(v0, 0.0, 0.0, v2)), cb.c[69].y); o0 = swz(half4(vec4(v15, v16, v23, alpha0)), cb.c[69].x); return;
}
if (pc == 0x0000100cu) discard;
}
)";

std::string plantVertexSource() {
    const std::string shared = kVs;
    std::string source = shared.substr(0, shared.find("void main()"));
    const std::string end = "return uintBitsToFloat(e);";
    source.replace(source.find(end), end.size(),
                   "for (uint c = (s.z >> 8) & 7u; c < 4u; ++c) e[c] = 0u;\n    return uintBitsToFloat(e);");
    return source + kVsPlantMain;
}

std::string plantFragmentSource() {
    const std::string shared = kFs;
    return shared.substr(0, shared.find("void main()")) + kFsPlantMain;
}

// Four-layer stage: VS 0x53c55ef00, PS 0x53c569d00. Isolated source copies
// retain every previous variant verbatim. VS operations remain precise; PS
// multiply-adds retain the translator's contraction opportunities and sample order.
// The four temporary EXEC masks only skip forward point-light blocks; every
// executed vector operation has lane 0 enabled. Exports round through binary16.
constexpr const char* kStageVs = R"(
float F(uint x) { return uintBitsToFloat(x); }
void stageExport(uint n, vec4 v) {
    if (((cb.c[79][0] >> 0u) & 0xffu) == n) p0 = v;
    if (((cb.c[79][0] >> 8u) & 0xffu) == n) p1 = v;
    if (((cb.c[79][0] >> 16u) & 0xffu) == n) p2 = v;
    if (((cb.c[79][0] >> 24u) & 0xffu) == n) p3 = v;
    if (((cb.c[79][1] >> 0u) & 0xffu) == n) p4 = v;
    if (((cb.c[79][1] >> 8u) & 0xffu) == n) p5 = v;
    if (((cb.c[79][1] >> 16u) & 0xffu) == n) p6 = v;
    if (((cb.c[79][1] >> 24u) & 0xffu) == n) p7 = v;
    if (((cb.c[79][2] >> 0u) & 0xffu) == n) p8 = v;
}
void main() { uint exec_hi = 0u; uint exec_lo = 0u; uint s0 = 0u; uint s1 = 0u; uint s10 = 0u; uint s11 = 0u;
uint s12 = 0u; uint s13 = 0u; uint s14 = 0u; uint s15 = 0u; uint s16 = 0u; uint s17 = 0u; uint s18 = 0u; uint s19 = 0u;
uint s2 = 0u; uint s20 = 0u; uint s21 = 0u; uint s22 = 0u; uint s23 = 0u; uint s24 = 0u; uint s25 = 0u; uint s26 = 0u;
uint s27 = 0u; uint s28 = 0u; uint s29 = 0u; uint s3 = 0u; uint s30 = 0u; uint s31 = 0u; uint s32 = 0u; uint s33 = 0u;
uint s34 = 0u; uint s35 = 0u; uint s36 = 0u; uint s37 = 0u; uint s38 = 0u; uint s39 = 0u; uint s4 = 0u; uint s40 = 0u;
uint s41 = 0u; uint s42 = 0u; uint s43 = 0u; uint s5 = 0u; uint s6 = 0u; uint s7 = 0u; uint s8 = 0u; uint s9 = 0u;
float v0 = 0.0; float v1 = 0.0; float v10 = 0.0; float v11 = 0.0; float v12 = 0.0; float v13 = 0.0; float v14 = 0.0;
float v15 = 0.0; float v16 = 0.0; float v17 = 0.0; float v18 = 0.0; float v19 = 0.0; float v2 = 0.0; float v20 = 0.0;
float v21 = 0.0; float v22 = 0.0; float v23 = 0.0; float v24 = 0.0; float v25 = 0.0; float v26 = 0.0; float v27 = 0.0;
float v28 = 0.0; float v29 = 0.0; float v3 = 0.0; float v30 = 0.0; float v31 = 0.0; float v32 = 0.0; float v33 = 0.0;
float v34 = 0.0; float v35 = 0.0; float v36 = 0.0; float v37 = 0.0; float v38 = 0.0; float v39 = 0.0; float v4 = 0.0;
float v5 = 0.0; float v6 = 0.0; float v7 = 0.0; float v8 = 0.0; float v9 = 0.0; uint vcc_hi = 0u; uint vcc_lo = 0u;
bool scc = false; uint pc = 0x00000000u; { vec4 f = fetch(0u); v4 = f[0]; v5 = f[1]; v6 = f[2]; v7 = f[3]; }
{ vec4 f = fetch(1u); v8 = f[0]; v9 = f[1]; v10 = f[2]; v11 = f[3]; }
{ vec4 f = fetch(2u); v12 = f[0]; v13 = f[1]; v14 = f[2]; v15 = f[3]; }
{ vec4 f = fetch(3u); v16 = f[0]; v17 = f[1]; v18 = f[2]; v19 = f[3]; }
{ vec4 f = fetch(4u); v20 = f[0]; v21 = f[1]; v22 = f[2]; v23 = f[3]; }
{ vec4 f = fetch(5u); v24 = f[0]; v25 = f[1]; v26 = f[2]; v27 = f[3]; }
{ vec4 f = fetch(6u); v28 = f[0]; v29 = f[1]; v30 = f[2]; v31 = f[3]; }
{ vec4 f = fetch(7u); v32 = f[0]; v33 = f[1]; v34 = f[2]; v35 = f[3]; }
{ vec4 f = fetch(8u); v36 = f[0]; v37 = f[1]; v38 = f[2]; v39 = f[3]; } exec_lo = 1u; pc = 0xcu;
for (;;) { switch (pc) { case 0x0000000cu: s16 = cb.c[81][0]; s17 = cb.c[81][1]; s18 = cb.c[81][2]; s19 = cb.c[81][3];
s32 = cb.c[82][0]; s33 = cb.c[82][1]; s34 = cb.c[82][2]; s35 = cb.c[82][3]; s36 = cb.c[83][0]; s37 = cb.c[83][1];
s38 = cb.c[83][2]; s39 = cb.c[83][3]; s40 = cb.c[84][0]; s41 = cb.c[84][1]; s42 = cb.c[84][2]; s43 = cb.c[84][3];
s14 = cb.c[85][0]; v1 = F(s35); v2 = F(s17); v0 = F(s18); { precise float r_ = F(s16) + v1; v1 = (r_); }
{ precise float r_ = F(s39) + v2; v2 = (r_); } { precise float r_ = F(s43) + v0; v0 = (r_); } s15 = s14 & 0x00080000u;
scc = s15 != 0u; { precise float r_ = v4 * F(s40) + v0; v3 = (r_); } scc = s15 != 0u;
{ precise float r_ = F(s32) * v4 + v1; v1 = (r_); } { precise float r_ = F(s36) * v4 + v2; v2 = (r_); }
{ precise float r_ = F(s33) * v5 + v1; v0 = (r_); } { precise float r_ = F(s37) * v5 + v2; v1 = (r_); }
vcc_hi = scc ? 1u : 0u; vcc_lo = s14 & 0x00100000u; scc = vcc_lo != 0u; scc = vcc_lo != 0u; vcc_lo = scc ? 0u : 1u;
{ precise float r_ = F(s41) * v5 + v3; v3 = (r_); } { precise float r_ = F(s42) * v6 + v3; v2 = (r_); }
vcc_lo = vcc_hi & vcc_lo; scc = vcc_lo != 0u; { precise float r_ = F(s34) * v6 + v0; v0 = (r_); }
{ precise float r_ = F(s38) * v6 + v1; v1 = (r_); } v7 = (v0); v22 = (v1); v11 = (v2);
if (scc) { pc = 0x000000ccu; break; } case 0x000000b0u: s0 = cb.c[85][1]; s1 = cb.c[85][2]; s2 = cb.c[85][3];
s3 = cb.c[86][0]; { precise float r_ = v0 - F(s0); v7 = (r_); } { precise float r_ = v1 - F(s1); v22 = (r_); }
{ precise float r_ = v2 - F(s2); v11 = (r_); } case 0x000000ccu: scc = s15 != 0u;
if (!scc) { pc = 0x000000f0u; break; } case 0x000000d4u: s0 = cb.c[86][1]; s1 = cb.c[86][2]; s2 = cb.c[86][3];
s3 = cb.c[87][0]; { precise float r_ = F(s0) + v0; v0 = (r_); } { precise float r_ = F(s1) + v1; v1 = (r_); }
{ precise float r_ = F(s2) + v2; v2 = (r_); } case 0x000000f0u: v3 = F(0x3f800000u); vcc_lo = s14 & 0x00020000u;
scc = vcc_lo != 0u; v4 = F(0x3f800000u); v5 = F(0x3f800000u); v23 = F(0u); v26 = F(0u); v27 = F(0u); s16 = cb.c[87][1];
s17 = cb.c[87][2]; s18 = cb.c[87][3]; s19 = cb.c[88][0]; s20 = cb.c[88][1]; s21 = cb.c[88][2]; s22 = cb.c[88][3];
s23 = cb.c[89][0]; s24 = cb.c[89][1]; s25 = cb.c[89][2]; s26 = cb.c[89][3]; s27 = cb.c[90][0]; s28 = cb.c[90][1];
s29 = cb.c[90][2]; s30 = cb.c[90][3]; s31 = cb.c[91][0]; { precise float r_ = F(s20) * v7; v6 = (r_); }
{ precise float r_ = F(s24) * v7; v30 = (r_); } { precise float r_ = F(s21) * v22 + v6; v6 = (r_); }
{ precise float r_ = F(s28) * v7; v31 = (r_); } v34 = F(s19); { precise float r_ = F(s22) * v11 + v6; v6 = (r_); }
{ precise float r_ = F(s25) * v22 + v30; v30 = (r_); } { precise float r_ = F(s26) * v11 + v30; v7 = (r_); }
{ precise float r_ = F(s16) * v6 + v34; v34 = (r_); } { precise float r_ = F(s29) * v22 + v31; v31 = (r_); }
{ precise float r_ = F(s30) * v11 + v31; v11 = (r_); } { precise float r_ = F(s17) * v7 + v34; v34 = (r_); }
{ precise float r_ = F(s18) * v11 + v34; v22 = (r_); } if (!scc) { pc = 0x00000240u; break; } case 0x00000164u:
s16 = cb.c[91][1]; s17 = cb.c[91][2]; s18 = cb.c[91][3]; s19 = cb.c[92][0]; s20 = cb.c[92][1]; s21 = cb.c[92][2];
s22 = cb.c[92][3]; s23 = cb.c[93][0]; s24 = cb.c[93][1]; s25 = cb.c[93][2]; s26 = cb.c[93][3]; s27 = cb.c[94][0];
s28 = cb.c[94][1]; s29 = cb.c[94][2]; s30 = cb.c[94][3]; s31 = cb.c[95][0]; s4 = cb.c[95][1]; s5 = cb.c[95][2];
s6 = cb.c[95][3]; s7 = cb.c[96][0]; s0 = cb.c[96][1]; s1 = cb.c[96][2]; s2 = cb.c[96][3]; s3 = cb.c[97][0];
{ precise float r_ = F(s18) * v22; v3 = (r_); } v23 = F(s23); { precise float r_ = F(s16) * v22; v4 = (r_); }
{ precise float r_ = F(0x3fb8aa3bu) * v4; v4 = (r_); } { precise float r_ = F(0x3fb8aa3bu) * v3; v5 = (r_); }
{ precise float r_ = F(s17) * v22; v3 = (r_); } { precise float r_ = F(0x3fb8aa3bu) * v3; v26 = (r_); }
{ precise float r_ = exp2(v4); v3 = (r_); } { precise float r_ = exp2(v26); v4 = (r_); }
{ precise float r_ = exp2(v5); v5 = (r_); } { precise float r_ = v0 - F(s0); v26 = (r_); }
{ precise float r_ = v1 - F(s1); v27 = (r_); } { precise float r_ = v26 * v26; v30 = (r_); }
{ precise float r_ = F(s4) * v26; v31 = (r_); } { precise float r_ = v2 - F(s2); v26 = (r_); }
{ precise float r_ = v27 * v27 + v30; v30 = (r_); } { precise float r_ = F(s5) * v27 + v31; v31 = (r_); }
{ precise float r_ = v26 * v26 + v30; v30 = (r_); } { precise float r_ = inversesqrt(v30); v27 = (r_); }
{ precise float r_ = F(s6) * v26 + v31; v31 = (r_); } { precise float r_ = (-v27) * v31; v26 = (r_); }
{ precise float r_ = F(s27) * v26 + v23; v23 = (r_); } { precise float r_ = inversesqrt(v23); v23 = (r_); }
{ precise float r_ = v23 * v23; v27 = (r_); } { precise float r_ = v26 * v26 + F(s31); v26 = (r_); }
{ precise float r_ = v27 * v23; v23 = (r_); } { precise float r_ = F(s20) * v26; v30 = (r_); }
{ precise float r_ = F(s21) * v26; v27 = (r_); } { precise float r_ = F(s22) * v26; v31 = (r_); }
{ precise float r_ = F(s24) * v23 + v30; v30 = (r_); } { precise float r_ = F(s25) * v23 + v27; v27 = (r_); }
{ precise float r_ = F(s26) * v23 + v31; v31 = (r_); } { precise float r_ = F(s28) * v30; v23 = (r_); }
{ precise float r_ = F(s29) * v27; v26 = (r_); } { precise float r_ = F(s30) * v31; v27 = (r_); }
{ precise float r_ = F(s31) * v23; v34 = (r_); } { precise float r_ = F(s31) * v26; v31 = (r_); }
{ precise float r_ = F(s31) * v27; v30 = (r_); } { precise float r_ = (-v3) * v23 + v34; v23 = (r_); }
{ precise float r_ = (-v4) * v26 + v31; v26 = (r_); } { precise float r_ = (-v5) * v27 + v30; v27 = (r_); }
case 0x00000240u: { precise float r_ = v16 * v3; v16 = (r_); } s26 = 0x40008102u;
{ precise float r_ = F(s26) * v8 + -1.0; v8 = (r_); } { precise float r_ = F(s26) * v9 + -1.0; v9 = (r_); }
{ precise float r_ = F(s40) * v8; v31 = (r_); } { precise float r_ = F(s41) * v9 + v31; v31 = (r_); }
{ precise float r_ = F(s26) * v14 + -1.0; v14 = (r_); } { precise float r_ = F(s32) * v8; v30 = (r_); }
{ precise float r_ = F(s26) * v10 + -1.0; v10 = (r_); } { precise float r_ = F(s36) * v8; v34 = (r_); }
{ precise float r_ = F(s33) * v9 + v30; v30 = (r_); } { precise float r_ = F(s34) * v10 + v30; v8 = (r_); }
{ precise float r_ = F(s37) * v9 + v34; v34 = (r_); } { precise float r_ = F(s38) * v10 + v34; v9 = (r_); }
{ precise float r_ = F(s42) * v10 + v31; v10 = (r_); } { precise float r_ = 1.0 - v39; v31 = (r_); }
{ precise float r_ = 1.0 / v31; v39 = (r_); } { precise float r_ = v36 * v39; v35 = (r_); }
{ precise float r_ = v37 * v39; v37 = (r_); } { precise float r_ = F(0x3e800000u) * v35; v35 = (r_); }
{ precise float r_ = F(s26) * v12 + -1.0; v30 = (r_); } vcc_lo = s14 & 0x00040000u; scc = vcc_lo != 0u;
{ precise float r_ = F(s32) * v30; v12 = (r_); } { precise float r_ = F(s26) * v13 + -1.0; v13 = (r_); }
{ precise float r_ = F(s36) * v30; v31 = (r_); } { precise float r_ = F(s40) * v30; v34 = (r_); } s0 = cb.c[97][1];
s1 = cb.c[97][2]; s2 = cb.c[97][3]; s3 = cb.c[98][0]; s4 = cb.c[98][1]; s5 = cb.c[98][2]; s6 = cb.c[98][3];
s7 = cb.c[99][0]; s12 = cb.c[99][1]; s13 = cb.c[99][2]; s14 = cb.c[99][3]; s15 = cb.c[100][0]; s20 = cb.c[100][1];
s21 = cb.c[100][2]; s22 = cb.c[100][3]; s23 = cb.c[101][0]; s27 = cb.c[101][1]; s24 = cb.c[101][2]; s25 = cb.c[101][3];
s16 = cb.c[102][0]; s17 = cb.c[102][1]; s18 = cb.c[102][2]; s19 = cb.c[102][3]; s8 = cb.c[103][0];
{ precise float r_ = v38 * v39; v30 = (r_); } scc = vcc_lo != 0u;
{ precise float r_ = F(s37) * v13 + v31; v39 = (r_); } { precise float r_ = F(0x3e800000u) * v37; v31 = (r_); }
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; }
{ precise float r_ = F(s33) * v13 + v12; v12 = (r_); } { precise float r_ = F(s41) * v13 + v34; v13 = (r_); }
{ precise float r_ = F(0x3e800000u) * v30; v34 = (r_); } { precise float r_ = F(s26) * v15 + -1.0; v15 = (r_); }
{ precise float r_ = v35 * v35; v30 = (r_); } { precise float r_ = v4 * v17; v17 = (r_); }
{ precise float r_ = v5 * v18; v18 = (r_); } { precise float r_ = F(s34) * v14 + v12; v12 = (r_); }
{ precise float r_ = F(s38) * v14 + v39; v39 = (r_); } { precise float r_ = F(s42) * v14 + v13; v13 = (r_); }
v36 = F(s3); v37 = F(s7); v14 = F(s15); { precise float r_ = F(s0) * v6 + v36; v35 = (r_); }
{ precise float r_ = F(s4) * v6 + v37; v36 = (r_); } { precise float r_ = F(s12) * v6 + v14; v14 = (r_); }
{ precise float r_ = F(s1) * v7 + v35; v35 = (r_); } { precise float r_ = F(s5) * v7 + v36; v36 = (r_); }
{ precise float r_ = F(s13) * v7 + v14; v14 = (r_); } { precise float r_ = F(s2) * v11 + v35; v7 = (r_); }
{ precise float r_ = F(s6) * v11 + v36; v36 = (r_); } { precise float r_ = F(s14) * v11 + v14; v14 = (r_); }
gl_Position = vec4(v7, v36, v14, v22); if (cb.c[69].z != 0u) gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;
{ precise float r_ = v0 - F(s20); v6 = (r_); } { precise float r_ = v1 - F(s21); v35 = (r_); }
{ precise float r_ = v6 * v6; v11 = (r_); } { precise float r_ = v8 * v6; v37 = (r_); }
{ precise float r_ = v2 - F(s22); v6 = (r_); } { precise float r_ = v35 * v35 + v11; v11 = (r_); }
{ precise float r_ = v9 * v35 + v37; v35 = (r_); } { precise float r_ = v6 * v6 + v11; v11 = (r_); }
{ precise float r_ = v10 * v6 + v35; v6 = (r_); } { precise float r_ = inversesqrt(v11); v11 = (r_); }
{ precise float r_ = v11 * v6 + 1.0; v6 = clamp(r_, 0.0, 1.0); } { precise float r_ = v6 * v6; v6 = (r_); }
{ precise float r_ = v6 * v6; v11 = (r_); } { precise float r_ = v31 * v31; v6 = (r_); }
{ precise float r_ = v34 * v34; v31 = (r_); } { precise float r_ = (F(s8) * v11 + F(s8)) * 0.5; v11 = (r_); }
v34 = (vcc_lo & 1u) != 0u ? 0u : v23; v23 = (vcc_lo & 1u) != 0u ? 0u : v26; v26 = (vcc_lo & 1u) != 0u ? 0u : v27;
{ precise float r_ = F(s27) * v19; v19 = (r_); } { precise float r_ = F(s24) + v28; v27 = (r_); }
{ precise float r_ = F(s25) + v29; v28 = (r_); } { precise float r_ = F(s16) + v20; v20 = (r_); }
{ precise float r_ = F(s17) + v21; v21 = (r_); } { precise float r_ = F(s18) + v24; v24 = (r_); }
{ precise float r_ = F(s19) + v25; v25 = (r_); } v29 = F(0u); stageExport(8u, vec4(v0, v1, v2, v29));
stageExport(7u, vec4(v12, v39, v13, v15)); stageExport(6u, vec4(v3, v4, v5, v29));
stageExport(5u, vec4(v30, v6, v31, v29)); stageExport(4u, vec4(v8, v9, v10, v11));
stageExport(3u, vec4(v34, v23, v26, v29)); stageExport(2u, vec4(v16, v17, v18, v19));
stageExport(1u, vec4(v27, v28, v32, v33)); stageExport(0u, vec4(v20, v21, v24, v25)); return; return; default: return;
} } }
)";
constexpr const char* kStageFs = R"(
layout(set = 0, binding = 13) uniform sampler3D volume;
layout(location = 8) in vec4 a8;
float F(uint x) { return uintBitsToFloat(x); }
float cubeId(vec3 d) { vec3 a = abs(d); if (a.z >= a.x && a.z >= a.y) return d.z < 0.0 ? 5.0 : 4.0;
if (a.y >= a.x) return d.y < 0.0 ? 3.0 : 2.0; return d.x < 0.0 ? 1.0 : 0.0; } float cubeSc(vec3 d) { vec3 a = abs(d);
if (a.z >= a.x && a.z >= a.y) return d.z < 0.0 ? -d.x : d.x; if (a.y >= a.x) return d.x; return d.x < 0.0 ? d.z : -d.z;
} float cubeTc(vec3 d) { vec3 a = abs(d); if (a.z >= a.x && a.z >= a.y) return -d.y;
if (a.y >= a.x) return d.y < 0.0 ? -d.z : d.z; return -d.y; } float cubeMa(vec3 d) { vec3 a = abs(d);
if (a.z >= a.x && a.z >= a.y) return 2.0 * d.z; if (a.y >= a.x) return 2.0 * d.y; return 2.0 * d.x; }
vec3 cubeDir(vec3 c) { float u = 2.0 * c.x - 3.0, v = 2.0 * c.y - 3.0; switch (int(c.z + 0.5) & 7) {
case 0: return vec3(1.0, -v, -u); case 1: return vec3(-1.0, -v, u); case 2: return vec3(u, 1.0, v);
case 3: return vec3(u, -1.0, -v); case 4: return vec3(u, -v, 1.0); default: return vec3(-u, -v, -1.0); } }
float psInput(uint a, uint c, float v) {
    if (((cb.c[80].x >> a) & 1u) == 0u) return v;
    uint d = (cb.c[80][1u + a / 16u] >> ((a % 16u) * 2u)) & 3u;
    return float(c == 3u ? d & 1u : d >> 1);
}
void main() { uint exec_hi = 0u; uint exec_lo = 0u; uint m0 = 0u; uint s0 = 0u; uint s1 = 0u; uint s10 = 0u;
uint s11 = 0u; uint s12 = 0u; uint s13 = 0u; uint s14 = 0u; uint s15 = 0u; uint s16 = 0u; uint s17 = 0u; uint s18 = 0u;
uint s19 = 0u; uint s2 = 0u; uint s20 = 0u; uint s21 = 0u; uint s22 = 0u; uint s23 = 0u; uint s24 = 0u; uint s25 = 0u;
uint s26 = 0u; uint s27 = 0u; uint s28 = 0u; uint s29 = 0u; uint s3 = 0u; uint s30 = 0u; uint s31 = 0u; uint s32 = 0u;
uint s33 = 0u; uint s34 = 0u; uint s35 = 0u; uint s36 = 0u; uint s37 = 0u; uint s38 = 0u; uint s39 = 0u; uint s4 = 0u;
uint s40 = 0u; uint s41 = 0u; uint s42 = 0u; uint s43 = 0u; uint s44 = 0u; uint s45 = 0u; uint s46 = 0u; uint s47 = 0u;
uint s48 = 0u; uint s49 = 0u; uint s5 = 0u; uint s50 = 0u; uint s51 = 0u; uint s52 = 0u; uint s53 = 0u; uint s54 = 0u;
uint s55 = 0u; uint s56 = 0u; uint s57 = 0u; uint s58 = 0u; uint s59 = 0u; uint s6 = 0u; uint s60 = 0u; uint s61 = 0u;
uint s62 = 0u; uint s63 = 0u; uint s64 = 0u; uint s65 = 0u; uint s66 = 0u; uint s67 = 0u; uint s68 = 0u; uint s69 = 0u;
uint s7 = 0u; uint s70 = 0u; uint s71 = 0u; uint s72 = 0u; uint s73 = 0u; uint s74 = 0u; uint s75 = 0u; uint s76 = 0u;
uint s77 = 0u; uint s78 = 0u; uint s79 = 0u; uint s8 = 0u; uint s80 = 0u; uint s81 = 0u; uint s82 = 0u; uint s83 = 0u;
uint s84 = 0u; uint s85 = 0u; uint s86 = 0u; uint s87 = 0u; uint s88 = 0u; uint s89 = 0u; uint s9 = 0u; uint s90 = 0u;
uint s91 = 0u; uint s92 = 0u; uint s93 = 0u; uint s94 = 0u; uint s95 = 0u; uint s96 = 0u; uint s97 = 0u; uint s98 = 0u;
uint s99 = 0u; float v0 = 0.0; float v1 = 0.0; float v10 = 0.0; float v106 = 0.0; float v11 = 0.0; float v12 = 0.0;
float v13 = 0.0; float v14 = 0.0; float v15 = 0.0; float v16 = 0.0; float v17 = 0.0; float v18 = 0.0; float v19 = 0.0;
float v2 = 0.0; float v20 = 0.0; float v21 = 0.0; float v22 = 0.0; float v23 = 0.0; float v24 = 0.0; float v25 = 0.0;
float v26 = 0.0; float v27 = 0.0; float v28 = 0.0; float v29 = 0.0; float v3 = 0.0; float v30 = 0.0; float v31 = 0.0;
float v32 = 0.0; float v33 = 0.0; float v34 = 0.0; float v35 = 0.0; float v36 = 0.0; float v37 = 0.0; float v38 = 0.0;
float v39 = 0.0; float v4 = 0.0; float v40 = 0.0; float v41 = 0.0; float v42 = 0.0; float v43 = 0.0; float v44 = 0.0;
float v45 = 0.0; float v46 = 0.0; float v47 = 0.0; float v48 = 0.0; float v49 = 0.0; float v5 = 0.0; float v50 = 0.0;
float v51 = 0.0; float v52 = 0.0; float v53 = 0.0; float v54 = 0.0; float v55 = 0.0; float v56 = 0.0; float v57 = 0.0;
float v58 = 0.0; float v59 = 0.0; float v6 = 0.0; float v60 = 0.0; float v61 = 0.0; float v62 = 0.0; float v63 = 0.0;
float v7 = 0.0; float v8 = 0.0; float v9 = 0.0; uint vcc_hi = 0u; uint vcc_lo = 0u; bool scc = false;
uint pc = 0x00000000u; exec_lo = 1u; v2 = gl_FragCoord.x; v3 = gl_FragCoord.y; v4 = gl_FragCoord.w; { pc = 0x00000000u;
} if (pc == 0x00000000u) { vcc_hi = 0x00000247u; m0 = s14; s93 = cb.c[103][1]; s28 = cb.c[103][2]; s29 = cb.c[103][3];
{ uint lo_ = exec_lo; uint hi_ = exec_hi; s14 = lo_; s15 = hi_; }
{ uint lo_ = exec_lo; uint hi_ = exec_hi; exec_lo = lo_; exec_hi = hi_; } v5 = psInput(1u, 0u, a1[0]);
v6 = psInput(1u, 1u, a1[1]); { vec2 uv_ = vec2(v5, v6); uv_.xy *= sc(0); { vec4 t_ = vec4(texture(tex[0], uv_));
v7 = t_[0]; v8 = t_[1]; } } v12 = psInput(4u, 1u, a4[1]); v16 = psInput(7u, 2u, a7[2]); v21 = psInput(7u, 0u, a7[0]);
v9 = (v12 * v16); v11 = psInput(4u, 2u, a4[2]); v18 = psInput(7u, 1u, a7[1]); v5 = psInput(7u, 3u, a7[3]);
v17 = psInput(4u, 0u, a4[0]); v10 = (v11 * v21); v9 = (v18 * (-v11) + v9); v22 = (v5 * v9); v6 = (v17 * v18);
v10 = (v16 * (-v17) + v10); v20 = (v5 * v10); v6 = (v21 * (-v12) + v6); v19 = (v5 * v6); s92 = s93 & 8u;
scc = s92 != 0u; scc = s92 != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; s36 = lo_; s37 = hi_; }
{ uint lo_ = s36; uint hi_ = s37; vcc_lo = lo_; vcc_hi = hi_; } v5 = psInput(1u, 2u, a1[2]);
v6 = psInput(1u, 3u, a1[3]); v9 = (F(s28) * v2); v10 = (F(s29) * v3); v13 = F(0x3f800000u); v14 = F(0x3f800000u);
v15 = F(0x3f800000u); v23 = (2.0 * v7 + -1.0); v24 = (v23 * v23); v7 = (2.0 * v8 + -1.0); v24 = (v7 * v7 + v24);
v8 = clamp(1.0 - v24, 0.0, 1.0); v21 = (v21 * v23); v8 = (sqrt(v8)); v18 = (v18 * v23); v21 = (v7 * v22 + v21);
v21 = (v8 * v17 + v21); v17 = (v20 * v7 + v18); v18 = (v16 * v23); v16 = (v21 * v21); v17 = (v8 * v12 + v17);
v18 = (v7 * v19 + v18); v7 = (v17 * v17 + v16); v18 = (v8 * v11 + v18); v7 = (v18 * v18 + v7); v8 = (inversesqrt(v7));
v19 = (v8 * v21); v20 = (v8 * v17); v21 = (v8 * v18); pc = ((vcc_lo & 1u) == 0u) ? 0x00000164u : 0x00000150u; }
if (pc == 0x00000150u) { { vec2 uv_ = vec2(v9, v10); uv_.xy *= sc(1); { vec4 t_ = vec4(texture(tex[1], uv_));
v13 = t_[0]; v14 = t_[1]; v15 = t_[3]; } } pc = 0x00000164u; } if (pc == 0x00000164u) {
{ uint lo_ = exec_lo & s36; uint hi_ = exec_hi & s37; vcc_lo = lo_; vcc_hi = hi_; } scc = (vcc_lo | vcc_hi) != 0u;
pc = ((vcc_lo & 1u) == 0u) ? 0x00000198u : 0x0000016cu; } if (pc == 0x0000016cu) { s16 = cb.c[104][0];
s17 = cb.c[104][1]; s18 = cb.c[104][2]; s19 = cb.c[104][3]; v7 = F(s17); v8 = F(s19); v13 = (F(s16) * v13 + v7);
v14 = (v14 * F(s18) + v8); pc = 0x00000198u; } if (pc == 0x00000198u) { v7 = (v13);
{ uint lo_ = exec_lo & s36; uint hi_ = exec_hi & s37; vcc_lo = lo_; vcc_hi = hi_; } scc = (vcc_lo | vcc_hi) != 0u;
v28 = (v14); pc = ((vcc_lo & 1u) == 0u) ? 0x000001ccu : 0x000001acu; } if (pc == 0x000001acu) { s16 = cb.c[105][0];
s17 = cb.c[105][1]; s18 = cb.c[105][2]; s19 = cb.c[105][3]; v7 = F(s17); v28 = F(s19); v7 = (F(s16) * v13 + v7);
v28 = (F(s18) * v14 + v28); pc = 0x000001ccu; } if (pc == 0x000001ccu) { v8 = (v7 * v28);
{ uint lo_ = exec_lo & s36; uint hi_ = exec_hi & s37; vcc_lo = lo_; vcc_hi = hi_; } scc = (vcc_lo | vcc_hi) != 0u; {
vec2 uv_ = vec2(v5, v6); uv_.xy *= sc(2); { vec4 t_ = vec4(texture(tex[2], uv_)); v16 = t_[0]; v17 = t_[1]; } }
s20 = cb.c[106][0]; s21 = cb.c[106][1]; s16 = cb.c[106][2]; s17 = cb.c[106][3]; s18 = cb.c[107][0]; s19 = cb.c[107][1];
v11 = F(s21); v7 = (1.0 - F(s16)); v6 = (1.0 - F(s17)); v5 = (1.0 - F(s18)); v18 = (v7 * v8 + F(s16));
v38 = (v6 * v8 + F(s17)); v37 = (v5 * v8 + F(s18)); v11 = (F(s20) * v17 + v11); v8 = (v16 * v11);
v33 = (v7 * v8 + F(s16)); v31 = (v6 * v8 + F(s17)); v29 = (v5 * v8 + F(s18)); v34 = (v33 * v18); v35 = (v31 * v38);
v36 = (v29 * v37); { uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = lo_; vcc_hi = hi_; }
pc = ((vcc_lo & 1u) == 0u) ? 0x000002b8u : 0x0000026cu; } if (pc == 0x0000026cu) { s16 = cb.c[107][2];
s17 = cb.c[107][3]; s18 = cb.c[108][0]; s19 = cb.c[108][1]; v12 = F(s17); v7 = F(s19); v8 = (F(s18) * v33 + v7);
v11 = (v18 * F(s16) + v12); v5 = (F(s18) * v31 + v7); v6 = (v38 * F(s16) + v12); v7 = (F(s18) * v29 + v7);
v12 = (F(s16) * v37 + v12); v34 = (v8 * v11); v35 = (v5 * v6); v36 = (v7 * v12); pc = 0x000002b8u; }
if (pc == 0x000002b8u) { vcc_lo = s93 & 0x00010000u; scc = vcc_lo != 0u; v27 = psInput(8u, 0u, a8[0]);
v26 = psInput(8u, 1u, a8[1]); v25 = psInput(8u, 2u, a8[2]); v39 = F(0u); v32 = F(0u); v30 = F(0u);
pc = (!scc) ? 0x00000394u : 0x000002e8u; } if (pc == 0x000002e8u) { s16 = cb.c[108][2]; s17 = cb.c[108][3];
s18 = cb.c[109][0]; s19 = cb.c[109][1]; v5 = (v27 - F(s16)); v6 = (v26 - F(s17)); v7 = (v5 * v5); v8 = (v19 * v5);
v5 = (v25 - F(s18)); v11 = (v6 * v6 + v7); v6 = (v20 * v6 + v8); v11 = (v5 * v5 + v11); v7 = (inversesqrt(v11));
v6 = (v5 * v21 + v6); v5 = clamp((-v7) * v6, 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v5)) ? 1u : 0u; vcc_hi = 0u;
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = exec_lo; vcc_hi = exec_hi; exec_lo = lo_ & exec_lo; exec_hi = hi_ & exec_hi; scc = (exec_lo & 1u) != 0u; }
pc = ((exec_lo & 1u) == 0u) ? 0x00000390u : 0x0000033cu; } if (pc == 0x0000033cu) { v7 = (sqrt(v11));
v6 = F(0x3f808102u); s16 = 0xbb810205u; s20 = cb.c[109][2]; s21 = cb.c[109][3]; s22 = cb.c[110][0]; s23 = cb.c[110][1];
v7 = (v7 - F(s23)); v7 = (max(F(0u), v7)); v8 = (F(s19) * v7); v7 = (v8 * v7 + 1.0); v7 = (1.0 / v7);
v6 = clamp(v6 * v7 + F(s16), 0.0, 1.0); v5 = (v5 * v6); v39 = (F(s20) * v5); v32 = (F(s21) * v5); v30 = (F(s22) * v5);
pc = 0x00000390u; } if (pc == 0x00000390u) { { uint lo_ = vcc_lo; uint hi_ = vcc_hi; exec_lo = lo_; exec_hi = hi_; }
pc = 0x00000394u; } if (pc == 0x00000394u) { vcc_lo = s93 & 0x00020000u; scc = vcc_lo != 0u;
pc = (!scc) ? 0x0000044cu : 0x000003a0u; } if (pc == 0x000003a0u) { s16 = cb.c[110][2]; s17 = cb.c[110][3];
s18 = cb.c[111][0]; s19 = cb.c[111][1]; v5 = (v27 - F(s16)); v6 = (v26 - F(s17)); v7 = (v5 * v5); v8 = (v19 * v5);
v5 = (v25 - F(s18)); v11 = (v6 * v6 + v7); v6 = (v20 * v6 + v8); v11 = (v5 * v5 + v11); v7 = (inversesqrt(v11));
v6 = (v5 * v21 + v6); v5 = clamp((-v7) * v6, 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v5)) ? 1u : 0u; vcc_hi = 0u;
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = exec_lo; vcc_hi = exec_hi; exec_lo = lo_ & exec_lo; exec_hi = hi_ & exec_hi; scc = (exec_lo & 1u) != 0u; }
pc = ((exec_lo & 1u) == 0u) ? 0x00000448u : 0x000003f4u; } if (pc == 0x000003f4u) { v7 = (sqrt(v11));
v6 = F(0x3f808102u); s16 = 0xbb810205u; s20 = cb.c[111][2]; s21 = cb.c[111][3]; s22 = cb.c[112][0]; s23 = cb.c[112][1];
v7 = (v7 - F(s23)); v7 = (max(F(0u), v7)); v8 = (F(s19) * v7); v7 = (v8 * v7 + 1.0); v7 = (1.0 / v7);
v6 = clamp(v6 * v7 + F(s16), 0.0, 1.0); v5 = (v5 * v6); v39 = (F(s20) * v5 + v39); v32 = (F(s21) * v5 + v32);
v30 = (F(s22) * v5 + v30); pc = 0x00000448u; } if (pc == 0x00000448u) {
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; exec_lo = lo_; exec_hi = hi_; } pc = 0x0000044cu; } if (pc == 0x0000044cu) {
vcc_lo = s93 & 0x00040000u; scc = vcc_lo != 0u; pc = (!scc) ? 0x00000504u : 0x00000458u; } if (pc == 0x00000458u) {
s16 = cb.c[112][2]; s17 = cb.c[112][3]; s18 = cb.c[113][0]; s19 = cb.c[113][1]; v5 = (v27 - F(s16));
v6 = (v26 - F(s17)); v7 = (v5 * v5); v8 = (v19 * v5); v5 = (v25 - F(s18)); v11 = (v6 * v6 + v7); v6 = (v20 * v6 + v8);
v11 = (v5 * v5 + v11); v7 = (inversesqrt(v11)); v6 = (v5 * v21 + v6); v5 = clamp((-v7) * v6, 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v5)) ? 1u : 0u; vcc_hi = 0u;
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = exec_lo; vcc_hi = exec_hi; exec_lo = lo_ & exec_lo; exec_hi = hi_ & exec_hi; scc = (exec_lo & 1u) != 0u; }
pc = ((exec_lo & 1u) == 0u) ? 0x00000500u : 0x000004acu; } if (pc == 0x000004acu) { v7 = (sqrt(v11));
v6 = F(0x3f808102u); s16 = 0xbb810205u; s20 = cb.c[113][2]; s21 = cb.c[113][3]; s22 = cb.c[114][0]; s23 = cb.c[114][1];
v7 = (v7 - F(s23)); v7 = (max(F(0u), v7)); v8 = (F(s19) * v7); v7 = (v8 * v7 + 1.0); v7 = (1.0 / v7);
v6 = clamp(v6 * v7 + F(s16), 0.0, 1.0); v5 = (v5 * v6); v39 = (F(s20) * v5 + v39); v32 = (F(s21) * v5 + v32);
v30 = (F(s22) * v5 + v30); pc = 0x00000500u; } if (pc == 0x00000500u) {
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; exec_lo = lo_; exec_hi = hi_; } pc = 0x00000504u; } if (pc == 0x00000504u) {
vcc_lo = s93 & 0x00080000u; scc = vcc_lo != 0u; pc = (!scc) ? 0x000005bcu : 0x00000510u; } if (pc == 0x00000510u) {
s16 = cb.c[114][2]; s17 = cb.c[114][3]; s18 = cb.c[115][0]; s19 = cb.c[115][1]; v5 = (v27 - F(s16));
v6 = (v26 - F(s17)); v7 = (v5 * v5); v8 = (v19 * v5); v5 = (v25 - F(s18)); v11 = (v6 * v6 + v7); v6 = (v20 * v6 + v8);
v11 = (v5 * v5 + v11); v7 = (inversesqrt(v11)); v6 = (v5 * v21 + v6); v5 = clamp((-v7) * v6, 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v5)) ? 1u : 0u; vcc_hi = 0u;
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; vcc_lo = exec_lo; vcc_hi = exec_hi; exec_lo = lo_ & exec_lo; exec_hi = hi_ & exec_hi; scc = (exec_lo & 1u) != 0u; }
pc = ((exec_lo & 1u) == 0u) ? 0x000005b8u : 0x00000564u; } if (pc == 0x00000564u) { v7 = (sqrt(v11));
v6 = F(0x3f808102u); s16 = 0xbb810205u; s20 = cb.c[115][2]; s21 = cb.c[115][3]; s22 = cb.c[116][0]; s23 = cb.c[116][1];
v7 = (v7 - F(s23)); v7 = (max(F(0u), v7)); v8 = (F(s19) * v7); v7 = (v8 * v7 + 1.0); v7 = (1.0 / v7);
v6 = clamp(v6 * v7 + F(s16), 0.0, 1.0); v5 = (v5 * v6); v30 = (F(s22) * v5 + v30); v32 = (F(s21) * v5 + v32);
v39 = (F(s20) * v5 + v39); pc = 0x000005b8u; } if (pc == 0x000005b8u) {
{ uint lo_ = vcc_lo; uint hi_ = vcc_hi; exec_lo = lo_; exec_hi = hi_; } pc = 0x000005bcu; } if (pc == 0x000005bcu) {
s64 = cb.c[116][2]; s65 = cb.c[116][3]; s66 = cb.c[117][0]; s67 = cb.c[117][1]; v11 = psInput(2u, 0u, a2[0]);
v11 = (v11 * v15); v13 = (v13 * v14); v48 = psInput(6u, 0u, a6[0]); vcc_lo = 0x3f400000u; v44 = psInput(4u, 3u, a4[3]);
s95 = s93 & 4u; scc = s95 != 0u; s88 = cb.c[117][2]; s89 = cb.c[117][3]; v14 = (v27 - F(s64)); v22 = (v14 * v14);
v12 = (v26 - F(s65)); v17 = (v25 - F(s66)); v5 = (F(s88) * v27); v6 = (F(s88) * v26); v7 = (F(s88) * v25); {
vec3 uv_ = vec3(v5, v6, v7); { vec4 t_ = vec4(texture(volume, uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2]; v8 = t_[3]; }
} v22 = (v12 * v12 + v22); v22 = (v17 * v17 + v22); v41 = (inversesqrt(v22)); v23 = (v41 * v14); v24 = (v23 * v23);
v22 = (v41 * v17); s48 = cb.c[118][0]; s49 = cb.c[118][1]; s50 = cb.c[118][2]; s51 = cb.c[118][3]; s52 = cb.c[119][0];
s53 = cb.c[119][1]; s54 = cb.c[119][2]; s55 = cb.c[119][3]; v40 = (v41 * v12 + F(0xbe99999au)); v43 = (v23 * v19);
v42 = (v40 * v40 + v24); v24 = (v41 * v12); v43 = (v20 * v40 + v43); v42 = (v22 * v22 + v42);
v49 = (v41 * v12 + F(s49)); v40 = (inversesqrt(v42)); v43 = (v21 * v22 + v43); v47 = (v41 * v17 + F(s50));
v41 = (v41 * v14 + F(s48)); v14 = (v41 * v41); v41 = (v41 * v19); v17 = clamp((-v40) * v43, 0.0, 1.0);
v45 = (F(s52) * v19); v14 = (v49 * v49 + v14); v46 = (log2(v17)); v41 = (v20 * v49 + v41); v14 = (v47 * v47 + v14);
v45 = (F(s53) * v20 + v45); s52 = cb.c[120][0]; s53 = cb.c[120][1]; v14 = (inversesqrt(v14)); v41 = (v21 * v47 + v41);
v45 = (F(s54) * v21 + v45); v14 = clamp((-v14) * v41, 0.0, 1.0); v54 = (max(F(0u), (-v45))); v14 = (log2(v14));
v40 = (v11 * v33); v11 = psInput(0u, 2u, a0[2]); v12 = psInput(0u, 3u, a0[3]); v17 = psInput(0u, 0u, a0[0]);
v51 = (F(s48) * v19); s72 = cb.c[120][2]; s73 = cb.c[120][3]; v42 = psInput(5u, 0u, a5[0]); v42 = (v42 + v39);
v43 = (v16 * v15); s94 = 0xbdbc49cau; v8 = (2.0 * v8 + -1.0); v5 = (2.0 * v5 + (-v8)); v5 = (-1.0 + v5);
v5 = (F(s89) * v5 + v19); v7 = (2.0 * v7 + (-v8)); v8 = (2.0 * v6 + (-v8)); v6 = (-1.0 + v8); v8 = (v5 * v5);
v6 = (F(s89) * v6 + v20); v7 = (-1.0 + v7); v7 = (F(s89) * v7 + v21); v8 = (v6 * v6 + v8); v8 = (v7 * v7 + v8);
v39 = (inversesqrt(v8)); v5 = (v39 * v5); v6 = (v39 * v6); v8 = (v39 * v7); v7 = (v23 * v5); v39 = (v24 * v6 + v7);
v39 = ((v8 * v22 + v39) * 2.0); v5 = (v5 * v39 + (-v23)); v7 = ((-v6) * v39 + v24); v8 = ((-v8) * v39 + v22);
v6 = (cubeMa(vec3(v5, v7, v8))); v39 = (1.0 / abs(v6)); v50 = (cubeSc(vec3(v5, v7, v8)));
v6 = (cubeTc(vec3(v5, v7, v8))); v6 = (v6 * v39 + F(0x3fc00000u)); v7 = (cubeId(vec3(v5, v7, v8)));
v5 = (v50 * v39 + F(0x3fc00000u)); { vec3 uv_ = vec3(v5, v6, v7); uv_ = cubeDir(uv_);
{ vec4 t_ = vec4(texture(env, uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2]; v8 = t_[3]; } } s40 = cb.c[121][0];
s41 = cb.c[121][1]; s42 = cb.c[121][2]; s43 = cb.c[121][3]; s44 = cb.c[122][0]; s45 = cb.c[122][1]; s46 = cb.c[122][2];
s47 = cb.c[122][3]; s84 = cb.c[123][0]; s85 = cb.c[123][1]; s86 = cb.c[123][2]; s87 = cb.c[123][3]; s56 = cb.c[124][0];
s57 = cb.c[124][1]; s58 = cb.c[124][2]; s59 = cb.c[124][3]; s60 = cb.c[125][0]; s61 = cb.c[125][1]; s62 = cb.c[125][2];
s63 = cb.c[125][3]; v52 = (F(s47) * v46); v39 = (exp2(v52)); v55 = (F(s57) * v38); v41 = (F(s47) * v14);
v46 = (F(s49) * v20 + v51); v46 = (F(s50) * v21 + v46); v56 = (max(F(0u), (-v46))); v45 = (min(1.0, v56));
v52 = (F(s61) * v39); v56 = (v55 * v45); v55 = (min(1.0, v54)); v57 = (F(s58) * v37); v58 = (v52 * v55);
v57 = (v57 * v45); v63 = F(s84); v53 = (F(s56) * v18); v53 = (v53 * v45); v51 = clamp(F(vcc_lo) - v46, 0.0, 1.0);
v49 = (v13 * v51); v13 = (exp2(v41)); v14 = (F(s56) * v13); v62 = (v14 * v18); v18 = (F(s60) * v39);
v14 = (F(s57) * v13); v13 = (F(s58) * v13); s56 = cb.c[126][0]; s57 = cb.c[126][1]; s58 = cb.c[126][2];
s59 = cb.c[126][3]; v60 = (v18 * v55); v18 = psInput(0u, 1u, a0[1]); v41 = (F(s60) * v28); v54 = (v41 * v55 + v53);
v47 = (max(F(s53), v49)); v51 = (F(s61) * v28); v53 = (v51 * v55 + v56); v56 = (1.0 / F(s58)); v38 = (v14 * v38);
v14 = (F(s62) * v39); v46 = (F(s62) * v28); v51 = (v46 * v55 + v57); v57 = (v14 * v55); v37 = (v13 * v37); {
vec2 uv_ = vec2(v11, v12); uv_.xy *= sc(4); { vec4 t_ = vec4(texture(tex[4], uv_)); v13 = t_[0]; v14 = t_[1]; } } {
vec2 uv_ = vec2(v9, v10); uv_.xy *= sc(3); { vec4 t_ = vec4(texture(tex[3], uv_)); v9 = t_[0]; v10 = t_[1];
v11 = t_[2]; v12 = t_[3]; } } s28 = cb.c[127][0]; s29 = cb.c[127][1]; s30 = cb.c[127][2]; s31 = cb.c[127][3];
vcc_lo = 0x3e19999au; s24 = cb.c[128][0]; v8 = (max(F(0x3b808081u), v8)); v8 = (1.0 / v8); v59 = (F(0x3e800000u) * v8);
v8 = (2.0 * F(s52)); s48 = cb.c[128][1]; s49 = cb.c[128][2]; s50 = cb.c[128][3]; s51 = cb.c[129][0];
s52 = cb.c[129][1]; s53 = cb.c[129][2]; s54 = cb.c[129][3]; s55 = cb.c[130][0]; v62 = (v8 * v60 + v62);
v49 = (v5 * v59); v50 = (v58 * v8 + v38); v58 = F(s73); v61 = F(s50); v5 = F(s54); v63 = (F(s55) * v63); v52 = F(s53);
v52 = (F(s45) * v52); v45 = (v6 * v59); v6 = F(s52); v46 = (F(s42) * v61); v61 = F(s48); v55 = (F(s40) * v61);
v61 = F(s72); v61 = (F(s55) * v61); v38 = (F(s46) * v5); v37 = (v8 * v57 + v37); v8 = (F(s44) * v6); v6 = (v40 * v8);
v60 = (v48 * v8); v57 = (v63 * F(s72) + v58); v63 = (v43 * v34); v41 = (v7 * v59); v7 = (v48 * v55); v59 = (v6 * v62);
v6 = F(s49); v62 = (v7 * v42); v48 = (F(s41) * v6); { vec2 uv_ = vec2(v17, v18); uv_.xy *= sc(5);
{ vec4 t_ = vec4(texture(tex[5], uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2]; v8 = t_[3]; } }
v18 = psInput(6u, 1u, a6[1]); v17 = (v43 * v35); v35 = (v39 * v63); v44 = (v13 * v44); v34 = (v10 * v56);
v10 = (v11 * v56); v11 = (v12 * v56); v12 = (v9 * v56); v9 = (v28 * v11); v11 = (v60 * v42 + v9);
v62 = (v12 * v28 + v62); v42 = (v16 * v33); v42 = (v42 * v44); v42 = (v42 * v47); v59 = (v11 * v35 + v59);
v11 = (1.0 - F(s24)); v11 = (v11 * v28 + F(s24)); v42 = (v42 * v49); v35 = psInput(5u, 1u, a5[1]); v35 = (v35 + v32);
v60 = psInput(6u, 2u, a6[2]); v13 = (v43 * v36); v43 = (v63 * v62); v62 = psInput(2u, 1u, a2[1]);
v36 = (v61 * F(s85) + v58); v63 = (v18 * v48); v62 = (v62 * v15); v63 = (v63 * v35); v62 = (v62 * v31);
v34 = (v28 * v34 + v63); v63 = (v17 * v34); v17 = (v39 * v17); v58 = (F(s86) * v61 + v58);
v61 = psInput(5u, 2u, a5[2]); v30 = (v61 + v30); v56 = (v5 * v5); v5 = (v56 * v57 + F(s28)); v5 = (v54 * v55 + v5);
v12 = (v40 * v5); v12 = (v11 * v43 + v12); v12 = (v56 * v12); v12 = (v59 * v14 + v12); v12 = (v49 * v42 + v12);
v40 = ((F(s56) * v12) * 2.0); v5 = F(0x3e19999au); v42 = (v5 * v40 + F(0x3d4ccccdu)); v33 = (v6 * v6);
v36 = (v33 * v36 + F(s29)); v36 = (v53 * v48 + v36); v6 = (v7 * v7); v32 = (v6 * v58 + F(s30));
v32 = (v51 * v46 + v32); v7 = (F(vcc_lo) * v40 + 0.5); v7 = (v40 * v7 + F(0x3d75c290u)); v7 = (1.0 / v7);
v12 = (v42 * v40 + F(0x3b83126fu)); v42 = (v16 * v31); v42 = (v42 * v44); v31 = (v62 * v36); v36 = (v62 * v52);
v40 = (v36 * v50); v31 = (v11 * v63 + v31); v36 = (v42 * v47); v42 = (v18 * v52); v34 = (v42 * v35 + v9);
v40 = (v34 * v17 + v40); v17 = (v33 * v31); v31 = (v36 * v45); v17 = (v40 * v14 + v17); v17 = (v45 * v31 + v17);
v18 = ((F(s56) * v17) * 2.0); v31 = (v5 * v18 + F(0x3d4ccccdu)); v16 = (v16 * v29); v16 = (v16 * v44);
v16 = (v16 * v47); v17 = (F(vcc_lo) * v18 + 0.5); v17 = (v18 * v17 + F(0x3d75c290u)); v17 = (1.0 / v17);
v18 = (v31 * v18 + F(0x3b83126fu)); v31 = psInput(2u, 2u, a2[2]); v15 = (v31 * v15); v15 = (v15 * v29);
v31 = (v60 * v38); v60 = (v60 * v46); v29 = (v15 * v32); v61 = (v31 * v30 + v9); v30 = (v60 * v30);
v10 = (v28 * v10 + v30); v31 = (v15 * v38); v10 = (v13 * v10); v31 = (v31 * v37); v37 = (v39 * v13);
v29 = (v11 * v10 + v29); v6 = (v6 * v29); v31 = (v61 * v37 + v31); v10 = (v16 * v41); v6 = (v31 * v14 + v6);
v6 = (v41 * v10 + v6); v6 = ((F(s56) * v6) * 2.0); v9 = (F(vcc_lo) * v6 + 0.5); v9 = (v6 * v9 + F(0x3d75c290u));
v5 = (v5 * v6 + F(0x3d4ccccdu)); v9 = (1.0 / v9); v6 = (v5 * v6 + F(0x3b83126fu)); v5 = (v7 * v12); v10 = (v17 * v18);
v9 = (v9 * v6); vcc_lo = s93 & 0x00400000u; scc = vcc_lo != 0u; v37 = F(0x3fb0852eu);
v12 = clamp(v37 * v9 + F(s94), 0.0, 1.0); scc = vcc_lo != 0u;
vcc_lo = ((exec_lo & 1u) != 0u && (s95 != 0u)) ? 1u : 0u; vcc_hi = 0u; v39 = psInput(3u, 0u, a3[0]);
v60 = psInput(3u, 1u, a3[1]); v0 = (sqrt(v12)); v61 = psInput(3u, 2u, a3[2]); v1 = clamp(v37 * v5 + F(s94), 0.0, 1.0);
v5 = clamp(v37 * v10 + F(s94), 0.0, 1.0); v1 = (sqrt(v1)); v5 = (sqrt(v5)); v7 = (v39 + v1); v10 = (v60 + v5);
v9 = (v61 + v0); v6 = (vcc_lo & 1u) != 0u ? v7 : v1; v7 = (vcc_lo & 1u) != 0u ? v10 : v5;
v9 = (vcc_lo & 1u) != 0u ? v9 : v0; vcc_hi = scc ? 0u : 1u; scc = s95 != 0u; vcc_lo = scc ? 1u : 0u;
vcc_lo = vcc_hi | vcc_lo; scc = vcc_lo != 0u; v38 = F(0u); pc = (scc) ? 0x00000ce0u : 0x00000c78u; }
if (pc == 0x00000c78u) { v6 = (F(s64) - v27); v6 = (v6 * v6); v7 = (F(s65) - v26); v6 = (v7 * v7 + v6);
v7 = (F(s66) - v25); s16 = cb.c[130][1]; s17 = cb.c[130][2]; s18 = cb.c[130][3]; s19 = cb.c[131][0];
s20 = cb.c[131][1]; s21 = cb.c[131][2]; s22 = cb.c[131][3]; s23 = cb.c[132][0]; v6 = (v7 * v7 + v6); v6 = (sqrt(v6));
v13 = F(s23); v38 = clamp(F(s22) * v6 + v13, 0.0, 1.0); v6 = (F(s16) * v38 + v1); v7 = (F(s17) * v38 + v5);
v9 = (F(s18) * v38 + v0); v6 = (v38 * (-v1) + v6); v7 = (v38 * (-v5) + v7); v9 = (v38 * (-v0) + v9); pc = 0x00000ce0u;
} if (pc == 0x00000ce0u) { v0 = F(0u); s32 = cb.c[132][1]; scc = s92 != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; } s28 = cb.c[132][2];
s24 = cb.c[132][3]; s20 = cb.c[133][0]; s16 = cb.c[133][1]; s17 = cb.c[133][2]; s18 = cb.c[133][3]; s19 = cb.c[134][0];
v1 = F(s28); v1 = (F(s24) * v1); v1 = (F(s20) * v1); v10 = (v1 * v8); v11 = (F(s19) * v10); v8 = (F(s16) * v6);
v5 = (F(s17) * v7); v1 = (F(s18) * v9); v14 = (vcc_lo & 1u) != 0u ? v10 : v11; v15 = (vcc_lo & 1u) != 0u ? v6 : v8;
v16 = (vcc_lo & 1u) != 0u ? v7 : v5; v17 = (vcc_lo & 1u) != 0u ? v9 : v1; vcc_lo = s32 & 0x02000000u;
scc = vcc_lo != 0u; scc = vcc_lo != 0u;
{ uint lo_ = scc ? exec_lo : 0u; uint hi_ = scc ? exec_hi : 0u; vcc_lo = lo_; vcc_hi = hi_; }
pc = ((vcc_lo & 1u) != 0u) ? 0x00001108u : 0x00000d5cu; } if (pc == 0x00000d5cu) { v25 = F(0u); s44 = cb.c[134][1];
s45 = cb.c[134][2]; v11 = (F(s44) * v2); v12 = (F(s45) * v3); { vec2 uv_ = vec2(v11, v12); uv_.xy *= sc(7);
{ vec4 t_ = vec4(texture(tex[7], uv_)); v7 = t_[0]; } } { vec2 uv_ = vec2(v11, v12); uv_.xy *= sc(8);
{ vec4 t_ = vec4(texture(tex[8], uv_)); v0 = t_[0]; v1 = t_[1]; v2 = t_[2]; } } s40 = cb.c[134][3]; s24 = cb.c[135][0];
s25 = cb.c[135][1]; s26 = cb.c[135][2]; s27 = cb.c[135][3]; s41 = cb.c[136][0]; s28 = cb.c[136][1]; s36 = cb.c[136][2];
v3 = F(s24); v6 = (F(s26) - v3); v18 = F(s24); v5 = F(s25); v6 = (1.0 / v6); v5 = (F(s24) - v5); v5 = (1.0 / v5);
vcc_lo = s36 & 4u; scc = vcc_lo != 0u; v7 = (F(s40) + v7); v7 = (1.0 / v7); v18 = (F(s41) * v7 + (-v18));
v28 = clamp((-v18) * v6 + 1.0, 0.0, 1.0); v27 = ((-F(s41)) * v7); v7 = (F(s28) * v28); v26 = ((-v27) * v5);
v31 = (2.0 * v1 + -1.0); v1 = (max(F(s27), v7)); v32 = (2.0 * v0 + -1.0); v26 = clamp((-F(s25)) * v5 + v26, 0.0, 1.0);
v8 = (v26 * v1); v30 = (F(s44) * v32); v29 = (F(s45) * (-v31)); v0 = (v30 * v8 + v11); v1 = (v29 * v8 + v12); {
vec2 uv_ = vec2(v0, v1); uv_.xy *= sc(6); { vec4 t_ = vec4(texture(tex[6], uv_)); v5 = t_[0]; v6 = t_[1]; v7 = t_[2]; }
} v0 = ((-v30) * v8 + v11); v1 = ((-v29) * v8 + v12); { vec2 uv_ = vec2(v0, v1); uv_.xy *= sc(6);
{ vec4 t_ = vec4(texture(tex[6], uv_)); v8 = t_[0]; v9 = t_[1]; v10 = t_[2]; } } v1 = (2.0 * v2 + -1.0);
v0 = (v32 + v31); v0 = (v0 + v1); v2 = (v6 * v31); v33 = ((v5 * v32) * 2.0); v6 = (v7 * v1); v5 = (v9 * v31);
v8 = ((v8 * v32) * 2.0); v1 = (v10 * v1); v33 = (2.0 * v2 + v33); v8 = (2.0 * v5 + v8); v33 = (2.0 * v6 + v33);
v8 = (2.0 * v1 + v8); v2 = (v33 - v0); v5 = (v8 - v0); pc = (!scc) ? 0x00000f0cu : 0x00000eb0u; }
if (pc == 0x00000eb0u) { v0 = (0.5 * v5 + 0.5); v0 = ((v2 * v0 + v0) * 0.5); v1 = (v0 * v0 + F(0x3b808081u));
s16 = cb.c[136][3]; s17 = cb.c[137][0]; v6 = ((-F(s17)) * v26 + 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (v1 >= v6)) ? 1u : 0u; vcc_hi = 0u; v6 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0;
v1 = (1.0 - F(s16)); v6 = clamp(v0 * v0 + v6, 0.0, 1.0); v0 = (v1 * v6 + F(s16)); v25 = ((-v26) * v0 + v26);
pc = 0x00000f0cu; } if (pc == 0x00000f0cu) { v0 = F(0u); vcc_lo = s36 & 8u; scc = vcc_lo != 0u;
pc = (!scc) ? 0x00000fd8u : 0x00000f18u; } if (pc == 0x00000f18u) { v6 = (2.0 * v27); s8 = cb.c[137][1];
s9 = cb.c[137][2]; s10 = cb.c[137][3]; v0 = (F(s8) * v28); v0 = (max(F(s27), v0)); v8 = (v26 * v0);
v0 = (v30 * v8 + v11); v1 = (v29 * v8 + v12); { vec2 uv_ = vec2(v0, v1); uv_.xy *= sc(7);
{ vec4 t_ = vec4(texture(tex[7], uv_)); v7 = t_[0]; } } v0 = ((-v30) * v8 + v11); v1 = ((-v29) * v8 + v12); {
vec2 uv_ = vec2(v0, v1); uv_.xy *= sc(7); { vec4 t_ = vec4(texture(tex[7], uv_)); v8 = t_[0]; } }
v1 = ((-F(s10)) * v26 + 1.0); v0 = (1.0 - F(s9)); v7 = (F(s40) + v7); v7 = (1.0 / v7); v6 = (F(s41) * v7 + v6);
v7 = (F(s40) + v8); v7 = (1.0 / v7); v6 = (F(s41) * v7 + v6); v6 = (-0.5 * v6 + 1.0); v7 = (F(0x3b808081u) + v6);
vcc_lo = ((exec_lo & 1u) != 0u && (v7 >= v1)) ? 1u : 0u; vcc_hi = 0u; v1 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0;
v1 = clamp(v1 + v6, 0.0, 1.0); v0 = (v0 * v1 + F(s9)); v0 = ((-v26) * v0 + v26); pc = 0x00000fd8u; }
if (pc == 0x00000fd8u) { v1 = F(0u); vcc_lo = s36 & 2u; scc = vcc_lo != 0u; pc = (!scc) ? 0x000010b4u : 0x00000fe4u; }
if (pc == 0x00000fe4u) { v1 = (v19 * v23); v1 = (v24 * v20 + v1); v1 = (v22 * v21 + v1); v1 = (max(F(0u), (-v1)));
v1 = (min(1.0, v1)); s8 = cb.c[138][0]; s9 = cb.c[138][1]; s10 = cb.c[138][2]; s11 = cb.c[138][3]; s20 = cb.c[139][0];
s21 = cb.c[139][1]; s16 = cb.c[139][2]; v6 = (F(0x3b808081u) + v1); v1 = ((-v2) * v5 + 1.0); v2 = F(s8);
v2 = (F(0xba83126fu) + v2); v2 = (min(F(s9), v2)); v5 = (F(s8) - v2); v5 = (1.0 / v5); v8 = ((-v27) * v5); v7 = F(s10);
v8 = clamp(v2 * (-v5) + v8, 0.0, 1.0); v5 = (F(s21) - v7); v5 = (v8 * v5 + F(s10)); v5 = (v26 * v5 + F(0x3b808081u));
v5 = (1.0 / v5); v7 = F(s11); v5 = clamp(v5 * v6, 0.0, 1.0); v6 = (1.0 - F(s20)); v7 = (F(s16) - v7); v6 = (v6 * v5);
v2 = (v7 * v8 + F(s11)); v5 = clamp(v6 * v5 + F(s20), 0.0, 1.0);
vcc_lo = ((exec_lo & 1u) != 0u && (v1 >= v2)) ? 1u : 0u; vcc_hi = 0u; v26 = (v5 * (-v26) + v26);
v5 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0; v5 = (v5 * v26);
vcc_lo = ((exec_lo & 1u) != 0u && (F(0u) < v2)) ? 1u : 0u; vcc_hi = 0u; v1 = (vcc_lo & 1u) != 0u ? v5 : v26;
pc = 0x000010b4u; } if (pc == 0x000010b4u) { v0 = (max(max(v0, v1), v25)); s12 = cb.c[139][3]; s13 = cb.c[140][0];
s8 = cb.c[140][1]; v1 = (F(s12) - v3); v1 = (1.0 / v1); v1 = clamp((-v18) * v1 + 1.0, 0.0, 1.0); v2 = (1.0 - F(s13));
v1 = (max(v2, v1)); v0 = (v1 * v0); v0 = (v0 * (-v38) + v0);
vcc_lo = ((exec_lo & 1u) != 0u && (F(s8) <= v0)) ? 1u : 0u; vcc_hi = 0u; v1 = (vcc_lo & 1u) != 0u ? 1.0 : 0.0;
v0 = (v1 * v0); pc = 0x00001108u; } if (pc == 0x00001108u) { v1 = (v14); s0 = cb.c[140][2]; vcc_lo = s0 & 0x00800000u;
scc = vcc_lo != 0u; pc = (!scc) ? 0x00001170u : 0x00001120u; } if (pc == 0x00001120u) { s0 = cb.c[140][3];
s1 = cb.c[141][0]; s2 = cb.c[141][1]; s3 = cb.c[141][2]; v2 = F(s0); v3 = F(s1); v1 = F(s2); v15 = (F(s3) * v15 + v2);
v16 = (F(s3) * v16 + v3); v17 = (F(s3) * v17 + v1); v5 = F(s3); v0 = (F(s3) * v0); v1 = (F(s3) * v14);
v15 = ((-F(s0)) * v5 + v15); v16 = ((-F(s1)) * v5 + v16); v17 = ((-F(s2)) * v5 + v17); pc = 0x00001170u; }
if (pc == 0x00001170u) { s0 = cb.c[141][3]; float alpha = (s0 & 1u) != 0u ? clamp(F(0x3a4d9a68u) * v4, 0.0, 1.0) : v14;
if ((s14 & 1u) == 0u) discard; o1 = swz(half4(vec4(v0, 0.0, 0.0, v1)), cb.c[69].y);
o0 = swz(half4(vec4(v15, v16, v17, alpha)), cb.c[69].x); return; } }
)";

std::string stageVertexSource() {
    const std::string shared = kVs;
    std::string source = shared.substr(0, shared.find("void main()"));
    const auto replace = [&](const char* from, const char* to) {
        source.replace(source.find(from), std::strlen(from), to);
    };
    replace("default: return v6.w[i];", "case 6u: return v6.w[i]; case 7u: return v7.w[i]; default: return v8.w[i];");
    replace("cb.c[62 + int(b)]", "cb.c[b < 7u ? 62 + int(b) : 77 + int(b) - 7]");
    replace("return uintBitsToFloat(e);", "for (uint c = (s.z >> 8) & 7u; c < 4u; ++c) e[c] = 0u; return uintBitsToFloat(e);");
    source.insert(0, "layout(set = 0, binding = 11, std430) readonly buffer V7 { uint w[]; } v7;\n"
                     "layout(set = 0, binding = 12, std430) readonly buffer V8 { uint w[]; } v8;\n"
                     "layout(location = 8) out vec4 p8;\n");
    return source + kStageVs;
}
std::string stageFragmentSource() {
    const std::string shared = kFs;
    return shared.substr(0, shared.find("void main()")) + kStageFs;
}

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
        if (var.unmatched && !d.compareCopy) return decline(role, "not yet matched (drawn into compare copies only)");
        const VertexShaderInfo& vsi = kVertexShaders[var.vs];
        const char* vsWanted = vsi.name;
        if (!d.vsName || (*d.vsName != vsWanted && d.vsName->rfind(std::string(vsWanted) + " +", 0) != 0))
            return decline(role, "another vertex shader");
        if (d.upscaler) return decline(role, "an upscaler is on");
        if (d.pixelPositions) return decline(role, "positions in pixels");
        if (d.depthBias) return decline(role, "a polygon offset");
        if (d.colorCount != 2 || !d.color[0].view || !d.color[1].view || !d.depth.view)
            return decline(role, "not two colour targets with depth");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        uint32_t c[kConstants][4] = {};
        uint64_t loaded[(kConstants * 4 + 63) / 64] = {};
        // The loads' offsets in the shaders' data, resolved once per shader pair.
        const Load* vsList = vsi.loads;
        const size_t vsCount = vsi.loadCount;
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
            // Some values are loaded at more than one place in a shader (the camera: for the position,
            // for fog, for the reflection), each in its own branch. A load in a branch this draw
            // did not take reads as 0: it must not undo a load the draw did take (the same value).
            for (uint32_t i = 0; i < l.n; ++i) {
                const uint32_t at = l.index * 4 + l.at + i, v = data[offs[k] + i];
                if (v || !(loaded[at / 64] >> (at % 64) & 1)) c[at / 4][at % 4] = v;
                loaded[at / 64] |= uint64_t(1) << (at % 64);
            }
        }
        // The vertex inputs: v4 position, v8 normal, v12 tangent, v16 colour, v20 uv0, v24 uv1, v28 baked light.
        // r_b1dst: v4 position, v8 normal (xyz), v12 tangent, v16 colour, v20 uv, v24 baked light.
        const uint32_t* kUsed = vsi.used;
        const int inputs = vsi.inputs;
        const VertexBuffer* slot[9] = {};
        for (const VertexBuffer& v : d.vertex)
            if (v.slot >= 0 && v.slot < 9 && v.components) slot[v.slot] = &v;
        for (int i = 0; i < inputs; ++i) {
            const VertexBuffer* v = slot[i];
            if (!v) return decline(role, "a vertex input missing");
            bool plain = v->components >= kUsed[i];
            for (uint32_t k = 0; k < kUsed[i]; ++k) plain = plain && ((v->dstSel >> (3 * k)) & 7) == 4 + k;
            VertexBuffer decoded = *v;
            if (var.vs == 7 && i == 1 && (v->nfmt == 4 || v->nfmt == 5)) decoded.nfmt = 7;
            if (!decodable(decoded) || !plain) {
                char why[112];
                std::snprintf(why, sizeof why, "vertex input %d format dfmt %u nfmt %u stride %u sel 0x%x components %u", i, v->dfmt,
                              v->nfmt, v->stride, v->dstSel, v->components);
                return decline(role, why);
            }
            const uint32_t at = i < 7 ? cFetch + i : cStageFetch + i - 7;
            c[at][0] = (v->data.byteOffset + v->offset) / 4, c[at][1] = v->stride / 4;
            c[at][2] = v->dfmt | v->nfmt << 4;
            if (var.vs == 7 || var.vs == 8) c[at][2] |= v->components << 8;
        }
        if (slot[0]->nfmt != 7 || (slot[0]->dfmt != 13 && slot[0]->dfmt != 14 && slot[0]->dfmt != 12))
            return decline(role, "a position format it does not read exactly");
        VkDescriptorBufferInfo wind{};
        if (var.vs == 7) {
            const VertexBuffer* v = nullptr;
            for (const VertexBuffer& b : d.vertex)
                if (b.slot < 0 && b.pc == 0x50) v = &b;
            if (!v || !v->data.buffer || v->dfmt != 14 || v->nfmt != 7 || v->offset ||
                v->stride % 4 || v->data.byteOffset % 4 || !v->data.range ||
                v->data.range == VK_WHOLE_SIZE || v->data.range > UINT32_MAX)
                return decline(role, "a wind buffer layout it does not read exactly");
            const uint64_t bytes = v->records
                ? uint64_t(v->data.byteOffset) + (v->stride ? uint64_t(v->records - 1) * v->stride : 0) + 16
                : v->data.byteOffset;
            if (bytes > v->data.range) return decline(role, "a truncated wind buffer");
            c[cWind][0] = v->data.byteOffset;
            c[cWind][1] = v->stride;
            c[cWind][2] = v->records;
            c[cWind][3] = uint32_t(v->data.range);
            wind = {v->data.buffer, v->data.offset, v->data.range};
        }
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
        VkDescriptorImageInfo cube{}; // [ref]: the environment cube map, at its own binding
        if (var.cubePc) {
            const Texture* x = nullptr;
            for (const Texture& y : d.textures)
                if (y.pc == var.cubePc) x = &y;
            if (!x || !x->view || !x->sampler) return decline(role, "no environment cube map");
            cube = {x->sampler, x->view, VK_IMAGE_LAYOUT_GENERAL};
        }
        VkDescriptorImageInfo volume{};
        if (var.volumePc) {
            const Texture* x = nullptr;
            for (const Texture& y : d.textures)
                if (y.pc == var.volumePc) x = &y;
            if (!x || !x->view || !x->sampler) return decline(role, "no stage volume texture");
            volume = {x->sampler, x->view, VK_IMAGE_LAYOUT_GENERAL};
            if (!d.vsData || d.vsSize < 24 || !d.psData || d.psSize < 23)
                return decline(role, "no stage varying mapping/defaults");
            std::memcpy(c[cStageVsMap], d.vsData + 21, 12);
            std::memcpy(c[cStageDefaults], d.psData + 20, 12);
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
        if (kCompareOn) describe(c, slot, inputs, d);
        if (!logged_[role].exchange(true)) {
            std::fprintf(stderr, "[native] material: drawing %s natively (targets 0x%" PRIx64 ", 0x%" PRIx64 ")\n", var.ps,
                         d.color[0].address, d.color[1].address);
        }
        gpu.beginRendering(d.color, 2, d.depth, d.hasStencil);
        VkDescriptorBufferInfo bufs[10];
        for (int i = 0; i < 7; ++i) {
            const VertexBuffer* v = slot[i] && i < inputs ? slot[i] : slot[0]; // an unread binding: the position's
            bufs[i] = {v->data.buffer, v->data.offset, v->data.range};
        }
        bufs[7] = {consts.buffer, consts.offset, consts.range};
        for (int i = 7; i < 9; ++i) {
            const VertexBuffer* v = i < inputs ? slot[i] : slot[0];
            bufs[i + 1] = {v->data.buffer, v->data.offset, v->data.range};
        }
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Buffer index = d.index;
        const uint32_t firstIndex = d.firstIndex;
        std::array<VkDescriptorImageInfo, kTextures> imgs;
        std::memcpy(imgs.data(), images, sizeof images);
        std::array<VkDescriptorBufferInfo, 10> bs;
        std::memcpy(bs.data(), bufs, sizeof bufs);
        std::array<uint32_t, 6> st;
        for (int i = 0; i < 2; ++i) st[i * 3] = d.stencilRef[i], st[i * 3 + 1] = d.stencilCompare[i], st[i * 3 + 2] = d.stencilWrite[i];
        gpu.record([pipe, layout = layout_, bs, imgs, cube, wind, volume, vp, sc, index, firstIndex, indexed = d.indexed, indexType = d.indexType, count = d.count,
                    st, bc = std::to_array(d.blendConstants), pushSet = gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetBlendConstants(cmd, bc.data());
            for (int i = 0; i < 2; ++i) {
                const VkStencilFaceFlags face = i ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
                vkCmdSetStencilReference(cmd, face, st[i * 3]);
                vkCmdSetStencilCompareMask(cmd, face, st[i * 3 + 1]);
                vkCmdSetStencilWriteMask(cmd, face, st[i * 3 + 2]);
            }
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            VkWriteDescriptorSet w[14]{};
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
            uint32_t n = 9;
            if (cube.imageView) {
                w[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[9].dstBinding = 9;
                w[9].descriptorCount = 1;
                w[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[9].pImageInfo = &cube;
                n = 10;
            }
            if (wind.buffer) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstBinding = 10;
                w[n].descriptorCount = 1;
                w[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[n].pBufferInfo = &wind;
                ++n;
            }
            if (volume.imageView) {
                for (uint32_t i = 0; i < 2; ++i) {
                    w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[n].dstBinding = 11 + i;
                    w[n].descriptorCount = 1;
                    w[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    w[n].pBufferInfo = &bs[8 + i];
                    ++n;
                }
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstBinding = 13;
                w[n].descriptorCount = 1;
                w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[n].pImageInfo = &volume;
                ++n;
            }
            pushSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, n, w);
            if (indexed) {
                vkCmdBindIndexBuffer(cmd, index.buffer, index.offset + index.byteOffset, indexType);
                vkCmdDrawIndexed(cmd, count, 1, firstIndex, 0, 0);
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
            if (auto it = offsets_.find(key); it != offsets_.end()) {
                if (kCompareOn) { // compare mode: the cached offsets checked against this draw's own
                    for (size_t k = 0; k < it->second.size(); ++k) {
                        const Load& l = k < vsCount ? vs[k] : ps[k - vsCount];
                        const int64_t now = l.vs ? d.vsOffset(l.pc) : d.psOffset(l.pc);
                        if (now != it->second[k] && cacheWarned_.fetch_add(1) < 32)
                            std::fprintf(stderr, "[native] material: role %d's cached %s load 0x%x is at %lld, this draw's at %lld (%s)\n", role,
                                         l.vs ? "vertex" : "pixel", l.pc, (long long)it->second[k], (long long)now,
                                         d.vsName ? d.vsName->c_str() : "?");
                    }
                }
                return it->second;
            }
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
    std::atomic<int> cacheWarned_{0};
    std::unordered_map<PairKey, std::vector<int64_t>, PairHash> offsets_;

    std::string lastDrawInfo() const override { return info_; }
    // Compare mode's log: the constants a wrong pixel can come from, and the vertex inputs.
    static void describe(const uint32_t (&c)[kConstants][4], const VertexBuffer* const* slot, int inputs, const Draw& d) {
        std::string s;
        char b[200];
        std::snprintf(b, sizeof b, "vs %s; ps %s; ps flags 0x%x, 0x22c 0x%x, vs flags 0x%x, mask flags 0x%x, out swizzles 0x%x 0x%x",
                      d.vsName ? d.vsName->c_str() : "?", d.psName ? d.psName->c_str() : "?", c[cPsFlags][0], c[cPsFlags][3], c[cVsMisc][2], c[cMisc2][3], c[cOut][0], c[cOut][1]);
        s += b;
        std::snprintf(b, sizeof b, "; target formats %d %d", int(d.color[0].format), int(d.color[1].format));
        s += b;
        std::snprintf(b, sizeof b, "; %s %u (index offset %u, type %d), cull %d front %d, depth test %d write %d op %d", d.indexed ? "indices" : "vertices",
                      d.count, d.index.byteOffset, int(d.indexType), int(d.cull), int(d.frontFace), int(d.depthStencil.depthTestEnable),
                      int(d.depthStencil.depthWriteEnable), int(d.depthStencil.depthCompareOp));
        s += b;
        std::snprintf(b, sizeof b, "; %zu buffers, viewport %g %g %g %g", d.vertex.size(), d.viewport[0], d.viewport[1], d.viewport[2], d.viewport[3]);
        s += b;
        const int show[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73};
        for (int i : show) {
            float f[4];
            std::memcpy(f, c[i], sizeof f);
            std::snprintf(b, sizeof b, "\n    c%d: %g %g %g %g", i, f[0], f[1], f[2], f[3]);
            s += b;
        }
        for (int i = 0; i < inputs; ++i)
            if (slot[i]) {
                std::snprintf(b, sizeof b, "\n    input %d: dfmt %u nfmt %u stride %u sel 0x%x components %u index v%u offset %u+%u records %u", i,
                              slot[i]->dfmt, slot[i]->nfmt, slot[i]->stride, slot[i]->dstSel, slot[i]->components, slot[i]->indexVgpr,
                              slot[i]->data.byteOffset, slot[i]->offset, slot[i]->records);
                s += b;
            }
        for (const Texture& t : d.textures) {
            std::snprintf(b, sizeof b, "\n    texture pc 0x%x: %ux%u at 0x%llx scale %g %g", t.pc, t.width, t.height, (unsigned long long)t.address, t.scale[0], t.scale[1]);
            s += b;
        }
        info_ = std::move(s);
    }
    static inline const bool kCompareOn = std::getenv("LPN_NATIVE_COMPARE") != nullptr;
    static inline thread_local std::string info_;
    bool decline(int role, const char* why) {
        std::lock_guard lock(m_);
        if (declined_[role].insert(why).second && declined_[role].size() <= 16)
            std::fprintf(stderr, "[native] material: a %s draw left to the translated renderer (%s)\n", kVariants[role - 1].ps, why);
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        const VkPipelineDepthStencilStateCreateInfo& z = d.depthStencil;
        // Blending and stencil as the translated pipeline has them.
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
        const auto key = std::make_tuple(role, int(d.color[0].format), int(d.color[1].format), int(d.writeMask & 0xFF), int(d.depth.format),
                                         bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace), blendKey, stencilKey);
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
            VkDescriptorSetLayoutBinding b[14];
            for (uint32_t i = 0; i < 8; ++i)
                b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VkShaderStageFlags(i < 7 ? VK_SHADER_STAGE_VERTEX_BIT
                                                                                           : VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
                        nullptr};
            b[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            b[9] = {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // [ref]'s cube map
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            b[10] = {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            b[11] = {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            b[12] = {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            b[13] = {13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            dl.bindingCount = 14;
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
        const std::string vsDefs = kVertexShaders[var.vs].defines;
        const VkShaderModule vsm = gpu.shader(header + vsDefs + (var.vs == 8 ? stageVertexSource() : var.vs == 7 ? plantVertexSource() : var.vs == 6 ? d2uRefVertexSource() : std::string(kVs)), Gpu::Stage::Vertex);
        // The fragment stage's extension, ahead of everything else.
        const std::string fsHeader = "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\n" + header.substr(header.find('\n') + 1);
        const VkShaderModule fsm = gpu.shader(fsHeader + var.defines + (var.vs == 8 ? stageFragmentSource() : var.vs == 7 ? plantFragmentSource() : var.vs == 6 ? d2uRefFragmentSource() : std::string(kFs)), Gpu::Stage::Fragment);
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
        if (stencil) ds.stencilTestEnable = VK_TRUE, ds.front = z.front, ds.back = z.back;
        VkPipelineColorBlendAttachmentState att[2] = {d.blendState[0], d.blendState[1]};
        att[0].colorWriteMask = d.writeMask & 0xF;
        att[1].colorWriteMask = (d.writeMask >> 4) & 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 2;
        cb.pAttachments = att;
        const VkDynamicState dyn[6] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                       VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                       VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 6;
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

    std::map<std::tuple<int, int, int, int, int, bool, VkBool32, VkBool32, int, int, int, uint64_t, uint64_t>, VkPipeline> pipes_;
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
