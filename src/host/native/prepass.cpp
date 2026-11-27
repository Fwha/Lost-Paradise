// The depth and normal prepass (vs_normal_rigid[_pt].vso with ps_normal[_pt].pso: about 2,700
// draws a frame into the 2560x1472 normal target and the scene's depth, which SSAO and the
// material passes read), rebuilt from the disassembly of the four shaders.
//
// Vertex: the position (fetch slot 0) through the object's 3x4 world matrix (0x0 of the buffer in
// s[8:11]) plus an offset (0x170 of s[4:7]), minus the camera (0x220 of the view's buffer at 0x0
// of the table in s[12:13]) unless the object's flags (0x0 of the table's 0x10 buffer) have
// 0x80000 without 0x100000, then the view rotation (rows at 0x40, 0x50, 0x60 of the view's
// buffer) and the projection (rows at 0x0). The normal (slot 1, stored as n / 2 + 0.5 in 8-bit
// unorm: 512/255 n - 1) goes through the world matrix's 3x3 and the view rotation. _pt passes
// the texture coordinates (slot 2) on.
//
// The material passes after it test their depth against this one's, so the position is computed
// with the game's own operations in its order, each rounded (precise, no fused multiply-adds), as
// the translated shaders do: folding the matrices into one would move depths by a rounding.
//
// Pixel: the view-space normal as n * 0.498 + 0.498 to x, y, z (w 0). _pt first drops pixels
// where the second texture's w times the first's z is below a threshold (0xa0 of the buffer at
// 0x30 of the table in s[12:13]).
#include "host/native/native.h"
#include "host/native/skin_fetch.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

namespace host::native {

namespace {

enum Role { kOpaque = 1, kAlphaTest = 2, kSkinOpaque = 3, kSkinHair = 4, kRoleCount };

bool skinRole(int role) { return role == kSkinOpaque || role == kSkinHair; }
bool alphaRole(int role) { return role == kAlphaTest || role == kSkinHair; }
const char* roleName(int role) {
    switch (role) {
    case kOpaque: return "ps_normal.pso";
    case kAlphaTest: return "ps_normal_pt.pso";
    case kSkinOpaque: return "ps_normal_mask_n.pso";
    case kSkinHair: return "ps_normal_mask_n_hair_pt.pso";
    default: return "unknown";
    }
}

// 240 bytes: more than the 128 every device has, so the pass needs a device with 256.
struct Push {
    float proj[16];   // rows
    float rot[12];    // rows 0..2 (w unused)
    float world[12];  // rows 0..2, translation in w
    float off[4];     // the offset; w: the alpha threshold
    float cam[4];     // the camera; w: 1 to subtract it
    uint32_t pos[4];  // first word, stride in words, dfmt | nfmt << 4; w: the target's swizzle
    uint32_t nrm[4]; // w: 1 to map clip z from [-w, w]
    uint32_t uv[4];
};
static_assert(sizeof(Push) == 240);

constexpr const char* kHeader = R"(#version 450
layout(push_constant) uniform P {
    vec4 proj[4];
    vec4 rot[3];
    vec4 world[3];
    vec4 off;
    vec4 cam;
    uvec4 pos;
    uvec4 nrm;
    uvec4 uv;
} p;
)";

constexpr const char* kVs = R"(
layout(set = 0, binding = 0, std430) readonly buffer V0 { uint w[]; } v0;
layout(set = 0, binding = 1, std430) readonly buffer V1 { uint w[]; } v1;
#ifdef ALPHA_TEST
layout(set = 0, binding = 2, std430) readonly buffer V2 { uint w[]; } v2;
#endif
// As the game's shader exports them (param0: the normal and clip w; param1: the coordinates and
// zeros): interpolated through another interface, a share of the _pt pixels came out a step off.
layout(location = 0) out vec4 n;
#ifdef ALPHA_TEST
layout(location = 1) out vec4 t;
#endif
uint word(uint b, uint i) {
#ifdef ALPHA_TEST
    if (b == 2u) return v2.w[i];
#endif
    return b == 0u ? v0.w[i] : v1.w[i];
}
// An element as the GCN fetch decodes it, as raw bits (floats as their bits), by data format
// (bits 0-3) and number format (bits 4-6): 32-bit, 16-bit and 8-bit channels (as shadow.cpp's).
uint conv(uint raw, uint bits, uint nfmt) {
    float maxv = float((1u << bits) - 1u), half_ = float((1u << (bits - 1u)) - 1u);
    int sx = bitfieldExtract(int(raw), 0, int(bits));
    if (nfmt == 0u) return floatBitsToUint(float(raw) / maxv);
    if (nfmt == 1u) return floatBitsToUint(max(float(sx) / half_, -1.0));
    if (nfmt == 7u && bits == 16u) return floatBitsToUint(unpackHalf2x16(raw).x);
    if (nfmt == 5u) return uint(sx);
    return raw;
}
vec4 fetch(uint b, uvec4 s) {
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
// v_mac / v_mad as the translated shaders compute them: a rounded product, then a rounded sum.
float mac(float a, float b, float c) { precise float r = a * b + c; return r; }
float mul(float a, float b) { precise float r = a * b; return r; }
// a . x in the shaders' order: ((a.x x.x + a.y x.y) + a.z x.z), the first a product alone.
float dot3(vec4 a, vec3 x) { return mac(a.z, x.z, mac(a.y, x.y, mul(a.x, x.x))); }
void main() {
    vec3 x = fetch(0u, p.pos).xyz;
    vec3 w;
    for (int r = 0; r < 3; ++r) {
        precise float s = p.world[r].w + p.off[r];
        s = mac(p.world[r].x, x.x, s);
        s = mac(p.world[r].y, x.y, s);
        w[r] = mac(p.world[r].z, x.z, s);
        if (p.cam.w != 0.0) { precise float d = w[r] - p.cam[r]; w[r] = d; }
    }
    vec3 e = vec3(dot3(p.rot[0], w), dot3(p.rot[1], w), dot3(p.rot[2], w));
    vec4 c;
    for (int r = 0; r < 4; ++r) c[r] = mac(p.proj[r].z, e.z, mac(p.proj[r].y, e.y, mac(p.proj[r].x, e.x, p.proj[r].w)));
    // As the translated export: clip z from [-w, w] to [0, w] (PA_CL_CLIP_CNTL bit 19 clear).
    if (p.nrm.w != 0u) c.z = (c.z + c.w) * 0.5;
    gl_Position = c;
    vec3 m = fetch(1u, p.nrm).xyz;
    const float k = uintBitsToFloat(0x40008102u); // the shader's literal: 255/127, not 2
    m = vec3(mac(k, m.x, -1.0), mac(k, m.y, -1.0), mac(k, m.z, -1.0));
    vec3 o = vec3(dot3(p.world[0], m), dot3(p.world[1], m), dot3(p.world[2], m));
    n = vec4(dot3(p.rot[0], o), dot3(p.rot[1], o), dot3(p.rot[2], o), c.w);
#ifdef ALPHA_TEST
    t = vec4(fetch(2u, p.uv).xy, 0.0, 0.0);
#endif
}
)";

constexpr const char* kFs = R"(
layout(location = 0) in vec4 n;
#ifdef ALPHA_TEST
layout(location = 1) in vec4 t;
layout(set = 0, binding = 3) uniform sampler2D mask0;
layout(set = 0, binding = 4) uniform sampler2D mask1;
#endif
layout(location = 0) out vec4 color;
void main() {
#ifdef ALPHA_TEST
    if (0.0 > texture(mask1, t.xy).w * texture(mask0, t.xy).z - p.off.w) discard;
#endif
    const float k = 0.49803922;
    vec3 e = n.xyz * k + k;
    // v_cvt_pkrtz_f16_f32 before the compressed export, as the translator rounds it.
    vec4 v = vec4(unpackHalf2x16(packHalf2x16(e.xy)), unpackHalf2x16(packHalf2x16(vec2(e.z, 0.0))));
    uint s = p.pos.w;
    color = vec4(v[s & 3u], v[(s >> 2) & 3u], v[(s >> 4) & 3u], v[(s >> 6) & 3u]);
}
)";

// The skin VSs share fetch decoding with the character material passes, but
// have their own translation order and exports. The two hair PS binaries
// (0x53c41d400 and 0x53c41c800) and their translated GLSL are identical.
constexpr const char* kSkinVs = R"(
// v4/v8/v12/v16/v20/v24, followed by the palette and compact fetch constants.
layout(set = 0, binding = 0, std430) readonly buffer V0 { uint w[]; } v0;
layout(set = 0, binding = 1, std430) readonly buffer V1 { uint w[]; } v1;
layout(set = 0, binding = 2, std430) readonly buffer V2 { uint w[]; } v2;
layout(set = 0, binding = 3, std430) readonly buffer V3 { uint w[]; } v3;
layout(set = 0, binding = 4, std430) readonly buffer V4 { uint w[]; } v4;
layout(set = 0, binding = 5, std430) readonly buffer V5 { uint w[]; } v5;
layout(set = 0, binding = 6, std430) readonly buffer Palette { uint w[]; } palette;
layout(set = 0, binding = 7, std430) readonly buffer Fetch { uvec4 c[]; } cb;
layout(location = 0) out vec4 n;
layout(location = 1) out vec4 t;
layout(location = 2) out vec4 tangent;
// Every floating VS instruction in these translations is precise. Pixel arithmetic
// deliberately has separate helpers below: it permits contraction.
float add(float a, float b) { precise float r = a + b; return r; }
float sub(float a, float b) { precise float r = a - b; return r; }
float mul(float a, float b) { precise float r = a * b; return r; }
float mac(float a, float b, float c) { precise float r = a * b + c; return r; }
float dot3(vec4 a, vec3 b) {
    return mac(a.z, b.z, mac(a.y, b.y, mul(a.x, b.x)));
}
)";

constexpr const char* kSkinMain = R"(
void main() {
    vec3 x = fetch(0u).xyz;
    vec4 fw = fetch(1u);
    uvec4 indices = fetchBits(2u);
    vec3 en = fetch(3u).xyz;
    vec4 et = fetch(4u);
    vec2 uv = fetch(5u).xy;
    float sum = add(fw.y, fw.z);
    sum = add(sum, fw.w);
    vec4 weights = vec4(sub(1.0, sum), fw.yzw);
    vec4 bone[12];
    for (int k = 0; k < 4; ++k)
        for (int r = 0; r < 3; ++r)
            bone[3*k+r] = boneRow(indices[k], uint(r));
    vec3 M[3];
    vec3 w;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float a = mul(bone[r][c], weights.x);
            // These MADs have the palette component first in both VSs.
            if ((r == 2 && c == 0) || (r == 1 && c == 1)) a = mac(bone[3+r][c], weights.y, a);
            else a = mac(weights.y, bone[3+r][c], a);
#ifdef ALPHA_TEST
            if (r == 1 && c == 1) a = mac(bone[6+r][c], weights.z, a);
            else
#endif
            a = mac(weights.z, bone[6+r][c], a);
#ifdef ALPHA_TEST
            if (r == 0 && c == 2) M[r][c] = mac(bone[9+r][c], weights.w, a);
            else
#endif
            if (r == 1 && c == 2) M[r][c] = mac(bone[9+r][c], weights.w, a);
            else M[r][c] = mac(weights.w, bone[9+r][c], a);
        }
        precise float anchor = floor(bone[r].w);
        float a = sub(bone[r].w, anchor);
        // Unlike the material VS, the offset is added BEFORE the weighted
        // fractional translation. Moving it after skinning changes depths.
        float s = add(p.off[r], anchor);
        s = mac(weights.x, a, s);
        s = mac(weights.y, sub(bone[3+r].w, anchor), s);
        s = mac(weights.z, sub(bone[6+r].w, anchor), s);
        s = mac(weights.w, sub(bone[9+r].w, anchor), s);
        s = mac(M[r].x, x.x, s);
        s = mac(M[r].y, x.y, s);
        w[r] = mac(M[r].z, x.z, s);
        if (p.cam.w != 0.0) w[r] = sub(w[r], p.cam[r]);
    }
    vec3 e = vec3(dot3(p.rot[0], w), dot3(p.rot[1], w), dot3(p.rot[2], w));
    vec4 clip;
    for (int r = 0; r < 4; ++r)
        clip[r] = mac(p.proj[r].z, e.z,
                      mac(p.proj[r].y, e.y, mac(p.proj[r].x, e.x, p.proj[r].w)));
#ifdef ALPHA_TEST
    // s_buffer_load at 0x2dc, added to clip z at 0x3f4 (before clip conversion).
    clip.z = add(p.world[0].x, clip.z);
#endif
    vec4 pos = clip;
    // Export adjustment is NOT precise in the translator.
    if (p.nrm.w != 0u) pos.z = (pos.z + pos.w) * 0.5;
    gl_Position = pos;
    const float k = uintBitsToFloat(0x40008102u);
    vec3 on, ot;
    for (int c = 0; c < 3; ++c) {
        on[c] = mac(k, en[c], -1.0);
        ot[c] = mac(k, et[c], -1.0);
    }
    vec3 normal, tan;
    for (int r = 0; r < 3; ++r) {
        float a = mul(M[r].x, on.x);
        // The opaque r2 normal MAD has M.y first; hair's MAC has on.y first.
#ifndef ALPHA_TEST
        if (r == 2) a = mac(M[r].y, on.y, a);
        else
#endif
        a = mac(on.y, M[r].y, a);
#ifdef ALPHA_TEST
        if (r == 0) normal[r] = mac(M[r].z, on.z, a);
        else
#endif
#ifndef ALPHA_TEST
        if (r == 1) normal[r] = mac(M[r].z, on.z, a);
        else
#endif
        normal[r] = mac(on.z, M[r].z, a);
        a = mul(M[r].x, ot.x);
#ifdef ALPHA_TEST
        if (r == 1) a = mac(M[r].y, ot.y, a);
        else
#endif
        a = mac(ot.y, M[r].y, a);
        tan[r] = mac(ot.z, M[r].z, a);
    }
    vec3 vn, vt;
    for (int r = 0; r < 3; ++r) {
        // The shader's last multiply is rotation.z * input.z, not a dot builtin.
        vn[r] = dot3(p.rot[r], normal);
        vt[r] = dot3(p.rot[r], tan);
    }
    n = vec4(vn, clip.w);
    t = vec4(uv, 0.0, 0.0);
    tangent = vec4(vt, mac(k, et.w, -1.0));
}
)";

constexpr const char* kSkinFs = R"(
layout(location = 0) in vec4 n;
layout(location = 1) in vec4 t;
layout(location = 2) in vec4 tangent;
layout(set = 0, binding = 8) uniform sampler2D normalMap;
#ifdef ALPHA_TEST
layout(set = 0, binding = 9) uniform sampler2D mask0;
layout(set = 0, binding = 10) uniform sampler2D mask1;
#endif
layout(location = 0) out vec4 color;
float F(uint x) { return uintBitsToFloat(x); }
uint U(float x) { return floatBitsToUint(x); }
uint sat(float x, bool c) { return U(c ? clamp(x, 0.0, 1.0) : x); }
void main() {
#ifdef ALPHA_TEST
    // No precise here: like every arithmetic instruction in these PS translations.
    uint alpha = sat(texture(mask1, t.xy).w * texture(mask0, t.xy).z + (-p.off.w), false);
    // PS 0x68 restores WQM for the normal-map sample at 0x70. Capture
    // gradients before killing alpha-failing lanes: their UVs still take
    // part in the surviving lanes' implicit LOD in the translated shader.
    vec2 dx = dFdx(t.xy), dy = dFdy(t.xy);
    vec2 sampleRG = textureGrad(normalMap, t.xy, dx, dy).xy;
    if (F(0u) > F(alpha)) discard;
#else
    vec2 sampleRG = texture(normalMap, t.xy).xy;
#endif
    uint v0, v1, v2 = U(sampleRG.x), v3 = U(sampleRG.y);
    uint v4 = U(n.z), v5 = U(n.y), v6 = U(tangent.z);
    uint v7 = U(tangent.y), v8 = U(n.x), v9 = U(tangent.x);
    uint v10, v11, v12 = U(tangent.w), v13;
v10 = sat(F(v5) * F(v6), false);
v1 = sat(F(v4) * F(v9), false);
v10 = sat(F(v7) * (-F(v4)) + F(v10), false);
v11 = sat(F(v12) * F(v10), false);
v0 = sat(F(v8) * F(v7), false);
v1 = sat(F(v6) * (-F(v8)) + F(v1), false);
v10 = sat(F(v12) * F(v1), false);
v0 = sat(F(v9) * (-F(v5)) + F(v0), false);
v1 = sat(F(v12) * F(v0), false);
v0 = 0x3efefeffu;
v12 = sat(2.0 * F(v2) + -1.0, false);
v13 = sat(F(v12) * F(v12), false);
v2 = sat(2.0 * F(v3) + -1.0, false);
v13 = sat(F(v2) * F(v2) + F(v13), false);
v3 = sat(1.0 - F(v13), true);
v9 = sat(F(v9) * F(v12), false);
v3 = sat(sqrt(F(v3)), false);
v13 = sat(F(v11) * F(v2) + F(v9), false);
v7 = sat(F(v7) * F(v12), false);
v13 = sat(F(v3) * F(v8) + F(v13), false);
v8 = sat(F(v6) * F(v12), false);
v7 = sat(F(v2) * F(v10) + F(v7), false);
v6 = sat(F(v13) * F(v13), false);
v7 = sat(F(v3) * F(v5) + F(v7), false);
v8 = sat(F(v2) * F(v1) + F(v8), false);
v6 = sat(F(v7) * F(v7) + F(v6), false);
v8 = sat(F(v3) * F(v4) + F(v8), false);
v6 = sat(F(v8) * F(v8) + F(v6), false);
v2 = sat(inversesqrt(F(v6)), false);
v3 = sat(F(v2) * F(v13), false);
v4 = sat(F(v2) * F(v7), false);
v1 = sat(F(v2) * F(v8), false);
v2 = sat(F(v0) * F(v3) + F(0x3efefeffu), false);
v3 = sat(F(v0) * F(v4) + F(0x3efefeffu), false);
v0 = sat(F(v0) * F(v1) + F(0x3efefeffu), false);
v1 = packHalf2x16(vec2(F(v2), F(v3)));
v0 = packHalf2x16(vec2(F(v0), 1.0));
    vec4 v = vec4(unpackHalf2x16(v1), unpackHalf2x16(v0));
    uint s = p.pos.w;
    color = vec4(v[s & 3u], v[(s >> 2) & 3u], v[(s >> 4) & 3u], v[(s >> 6) & 3u]);
}
)";

// Where the shaders' scalar loads are (offsets from the vertex shader's start).
struct Loads {
    uint32_t offset, world, world2, flags, camera, rotation, rotation2, projection;
};
constexpr Loads kOpaqueLoads{0x10, 0x14, 0x18, 0x20, 0xa0, 0xe0, 0xe8, 0xe4};
constexpr Loads kAlphaLoads{0x10, 0x14, 0x18, 0x20, 0xa0, 0x10c, 0x114, 0x110};

bool decodable(const VertexBuffer& v) {
    static constexpr uint32_t kDfmts[] = {1, 2, 3, 4, 5, 10, 11, 12, 13, 14};
    bool dfmt = false;
    for (uint32_t f : kDfmts) dfmt = dfmt || v.dfmt == f;
    const bool nfmt = v.nfmt == 0 || v.nfmt == 1 || v.nfmt == 7;
    return dfmt && nfmt && v.stride % 4 == 0 && (v.data.byteOffset + v.offset) % 4 == 0 && v.indexVgpr == 0;
}

class Prepass final : public Pass {
public:
    const char* name() const override { return "prepass"; }

    int claims(const std::string& psName) const override {
        if (psName == "ps_normal.pso") return kOpaque;
        if (psName == "ps_normal_pt.pso") return kAlphaTest;
        // Name aliases (+1 in capture-5) identify the same program role.
        if (psName == "ps_normal_mask_n.pso" || psName == "ps_normal_mask_n.pso +1") return kSkinOpaque;
        if (psName == "ps_normal_mask_n_hair_pt.pso" || psName == "ps_normal_mask_n_hair_pt.pso +1") return kSkinHair;
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        const bool skin = skinRole(role), alpha = alphaRole(role);
        if (skin && !d.compareCopy) return decline(role, "not yet matched (drawn into compare copies only)");
        const char* vs = skin ? (alpha ? "vs_normal_skin_n_pt.vso" : "vs_normal_skin_n.vso") :
                               (alpha ? "vs_normal_rigid_pt.vso" : "vs_normal_rigid.vso");
        if (!d.vsName || *d.vsName != vs) return decline(role, "another vertex shader");
        // The depths must be the translated ones: with an upscaler the scene's draws are jittered
        // and the camera read from the first of them, which stays with the translated renderer.
        if (d.upscaler) return decline(role, "an upscaler is on");
        if (d.pixelPositions) return decline(role, "positions in pixels");
        if (d.depthBias) return decline(role, "a polygon offset");
        if (gpu.maxPushConstants() < sizeof(Push)) return decline(role, "a device with fewer than 240 bytes of push constants");
        if (d.colorCount != 1 || !d.color[0].view || !d.depth.view) return decline(role, "not one colour target with depth");
        if (d.blend || d.stencil) return decline(role, "blending or stencil");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        if (skin) return drawSkin(role, d, gpu);
        // Fetch slots: the position (v4: x, y, z), the normal (v8: x, y, z), the coordinates (v12: x, y).
        const VertexBuffer* slot[3] = {};
        for (const VertexBuffer& v : d.vertex)
            if (v.slot >= 0 && v.slot < 3 && v.components) slot[v.slot] = &v;
        const int slots = alpha ? 3 : 2;
        for (int i = 0; i < slots; ++i) {
            const VertexBuffer* v = slot[i];
            if (!v) return decline(role, "a vertex input missing");
            const uint32_t used = i == 2 ? 2 : 3;
            bool plain = v->components >= used;
            for (uint32_t c = 0; c < used; ++c) plain = plain && ((v->dstSel >> (3 * c)) & 7) == 4 + c;
            if (!decodable(*v) || !plain) {
                char why[112];
                std::snprintf(why, sizeof why, "vertex input %d format dfmt %u nfmt %u stride %u offset %u sel 0x%x index v%u", i,
                              v->dfmt, v->nfmt, v->stride, v->data.byteOffset + v->offset, v->dstSel, v->indexVgpr);
                return decline(role, why);
            }
        }
        // Positions as stored: 32-bit floats or halves (the depths must match the game's).
        if (slot[0]->nfmt != 7 || (slot[0]->dfmt != 13 && slot[0]->dfmt != 14 && slot[0]->dfmt != 12))
            return decline(role, "a position format it does not read exactly");
        Push push{};
        const Loads& l = alpha ? kAlphaLoads : kOpaqueLoads;
        auto get = [&](uint32_t pc, float* v, uint32_t n) { return d.vsLoad(pc, reinterpret_cast<uint32_t*>(v), n); };
        uint32_t flags = 0;
        if (!get(l.offset, push.off, 4) || !get(l.world, push.world, 8) || !get(l.world2, push.world + 8, 4) ||
            !d.vsLoad(l.flags, &flags, 1) || !get(l.rotation, push.rot, 8) || !get(l.rotation2, push.rot + 8, 4) ||
            !get(l.projection, push.proj, 16))
            return decline(role, "the vertex shader's constants are missing");
        const bool subtract = !((flags & 0x80000) && !(flags & 0x100000));
        if (subtract && !get(l.camera, push.cam, 4)) return decline(role, "no camera constants");
        push.cam[3] = subtract ? 1.0f : 0.0f;
        auto put = [](const VertexBuffer& v, uint32_t* s) {
            s[0] = (v.data.byteOffset + v.offset) / 4, s[1] = v.stride / 4, s[2] = v.dfmt | v.nfmt << 4;
        };
        put(*slot[0], push.pos), put(*slot[1], push.nrm);
        if (alpha) put(*slot[2], push.uv);
        int32_t sw[4];
        swizzle(d.color[0], sw);
        push.pos[3] = uint32_t(sw[0] | sw[1] << 2 | sw[2] << 4 | sw[3] << 6);
        push.nrm[3] = d.clipZHalf ? 1 : 0;
        VkDescriptorBufferInfo bufs[3]{};
        for (int i = 0; i < slots; ++i) bufs[i] = {slot[i]->data.buffer, slot[i]->data.offset, slot[i]->data.range};
        VkDescriptorImageInfo images[2]{};
        if (alpha) {
            const Texture* t[2] = {};
            for (const Texture& x : d.textures) {
                if (x.pc == 0x34) t[0] = &x;
                if (x.pc == 0x3c) t[1] = &x;
            }
            for (int i = 0; i < 2; ++i) {
                if (!t[i] || !t[i]->view || !t[i]->sampler) return decline(role, "no mask texture");
                if (t[i]->scale[0] != 1 || t[i]->scale[1] != 1) return decline(role, "a mask texture with a coordinate scale");
                images[i] = {t[i]->sampler, t[i]->view, VK_IMAGE_LAYOUT_GENERAL};
            }
            uint32_t thr;
            if (!d.load(0x44, &thr, 1)) return decline(role, "no alpha threshold");
            std::memcpy(&push.off[3], &thr, 4);
        }
        VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role].exchange(true)) {
            std::fprintf(stderr, "[native] prepass: drawing %s natively (target 0x%" PRIx64 " %ux%u, depth 0x%" PRIx64 ")\n",
                         alpha ? "ps_normal_pt.pso" : "ps_normal.pso", d.color[0].address, d.color[0].width, d.color[0].height,
                         d.depth.address);
        }
        gpu.beginRendering(d.color, 1, d.depth, d.hasStencil);
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Buffer index = d.index;
        const uint32_t firstIndex = d.firstIndex;
        gpu.record([pipe, layout = layouts_[role], push, bufs, images, alpha, vp, sc, index, firstIndex, indexed = d.indexed,
                    indexType = d.indexType, count = d.count, pushSet = gpu.pushDescriptorSet()](VkCommandBuffer c) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(c, 0, 1, &vp);
            vkCmdSetScissor(c, 0, 1, &sc);
            VkWriteDescriptorSet w[5]{};
            uint32_t n = 0;
            const uint32_t buffers = alpha ? 3 : 2;
            for (uint32_t i = 0; i < buffers; ++i, ++n) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstBinding = i;
                w[n].descriptorCount = 1;
                w[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[n].pBufferInfo = &bufs[i];
            }
            if (alpha)
                for (uint32_t i = 0; i < 2; ++i, ++n) {
                    w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[n].dstBinding = 3 + i;
                    w[n].descriptorCount = 1;
                    w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w[n].pImageInfo = &images[i];
                }
            pushSet(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, n, w);
            vkCmdPushConstants(c, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, &push);
            if (indexed) {
                vkCmdBindIndexBuffer(c, index.buffer, index.offset + index.byteOffset, indexType);
                vkCmdDrawIndexed(c, count, 1, firstIndex, 0, 0);
            } else {
                vkCmdDraw(c, count, 1, 0, 0);
            }
        });
        return true;
    }

private:

    bool drawSkin(int role, const Draw& d, Gpu& gpu) {
        const bool alpha = role == kSkinHair;
        const auto& z = d.depthStencil;
        // Test/write enables and compare op are copied into the pipeline and
        // included in its key, just as for the rigid roles.
        if (!d.indexed || !d.index.buffer ||
            (d.indexType != VK_INDEX_TYPE_UINT16 && d.indexType != VK_INDEX_TYPE_UINT32))
            return decline(role, "not an indexed triangle list");
        if (d.color[0].format != VK_FORMAT_R8G8B8A8_UNORM ||
            d.color[0].swap != 1 || d.color[0].comps != 4 || d.writeMask != 0xf ||
            d.depth.format != VK_FORMAT_D32_SFLOAT_S8_UINT || !d.hasStencil)
            return decline(role, "another target or depth format");
        if (z.depthBoundsTestEnable || z.stencilTestEnable)
            return decline(role, "another depth state");
        const VertexBuffer* slot[6] = {};
        const VertexBuffer* palette = nullptr;
        constexpr uint32_t palettePc[] = {0x3c, 0x44, 0x4c, 0x84, 0x8c, 0x94, 0x9c, 0xc4, 0xcc, 0xe8, 0xf0, 0xfc};
        uint32_t seen = 0;
        for (const VertexBuffer& v : d.vertex) {
            if (v.slot >= 0 && v.slot < 6 && v.components) {
                if (slot[v.slot]) return decline(role, "duplicate vertex input");
                slot[v.slot] = &v;
                continue;
            }
            if (v.slot >= 0) return decline(role, "another fetch slot");
            int at = -1;
            for (int i = 0; i < 12; ++i) if (v.pc == palettePc[i]) at = i;
            if (at < 0 || (seen & (1u << at))) return decline(role, "another palette load");
            seen |= 1u << at;
            if (v.stride != 16 || v.records != 2097152 || v.offset || v.data.byteOffset % 4 ||
                v.dfmt != 14 || v.nfmt != 7 || v.dstSel != 0xfac || !v.data.buffer ||
                v.data.byteOffset > v.data.range || v.data.range - v.data.byteOffset < 256 * 48)
                return decline(role, "another palette layout");
            if (palette && (palette->data.buffer != v.data.buffer || palette->data.offset != v.data.offset ||
                            palette->data.byteOffset != v.data.byteOffset || palette->data.range != v.data.range))
                return decline(role, "palette loads disagree");
            palette = &v;
        }
        if (seen != 0xfff || !palette) return decline(role, "a palette load missing");
        // Only components consumed by these VSs must have an identity DST_SEL.
        // In particular the stored weight x is ignored (reconstructed from yzw),
        // and normal.w is unused.
        constexpr uint32_t used[] = {3, 4, 4, 3, 4, 2};
        uint32_t constants[7][4]{};
        VkDescriptorBufferInfo buffers[8]{};
        for (int i = 0; i < 6; ++i) {
            const VertexBuffer* v = slot[i];
            if (!v) return decline(role, "a skin vertex input missing");
            uint32_t channels = 0, bytes = 0;
            switch (v->dfmt) {
            case 1: channels = 1; bytes = 4; break;
            case 2: channels = 1; bytes = 4; break;
            case 3: channels = 2; bytes = 4; break;
            case 4: channels = 1; bytes = 4; break;
            case 5: channels = 2; bytes = 4; break;
            case 10: channels = 4; bytes = 4; break;
            case 11: channels = 2; bytes = 8; break;
            case 12: channels = 4; bytes = 8; break;
            case 13: channels = 3; bytes = 12; break;
            case 14: channels = 4; bytes = 16; break;
            }
            const bool wide = v->dfmt == 4 || v->dfmt == 11 || v->dfmt == 13 || v->dfmt == 14;
            const bool half = v->dfmt == 2 || v->dfmt == 5 || v->dfmt == 12;
            const bool format = i == 2 ? (v->nfmt == 4 || v->nfmt == 5) :
                (wide ? v->nfmt == 7 : (v->nfmt == 0 || v->nfmt == 1 || (half && v->nfmt == 7)));
            bool plain = v->components >= used[i] && channels >= used[i];
            for (uint32_t c = i == 1 ? 1 : 0; c < used[i]; ++c)
                plain = plain && ((v->dstSel >> (3 * c)) & 7) == 4 + c;
            const uint64_t offset = uint64_t(v->data.byteOffset) + v->offset;
            if (!v->data.buffer || v->indexVgpr || v->stride % 4 || offset % 4 ||
                offset > UINT32_MAX || !format || !plain || !v->records)
                return decline(role, "a skin fetch format it does not read exactly");
            // The renderer may cap the descriptor range independently of
            // NUM_RECORDS; do not require the entire declared stream to fit.
            if (offset > v->data.range || bytes > v->data.range - offset)
                return decline(role, "a skin fetch outside its buffer");
            constants[i][0] = uint32_t(offset / 4);
            constants[i][1] = v->stride / 4;
            constants[i][2] = v->dfmt | v->nfmt << 4;
            constants[i][3] = v->records;
            buffers[i] = {v->data.buffer, v->data.offset, v->data.range};
        }
        constants[6][0] = palette->data.byteOffset / 4;
        buffers[6] = {palette->data.buffer, palette->data.offset, palette->data.range};
        Push push{};
        auto get = [&](uint32_t pc, float* out, uint32_t n) {
            return d.vsLoad && d.vsLoad(pc, reinterpret_cast<uint32_t*>(out), n);
        };
        uint32_t flags;
        if (!get(alpha ? 0x104 : 0x108, push.off, 4) || !d.vsLoad || !d.vsLoad(0x11c, &flags, 1) ||
            !get(alpha ? 0x354 : 0x340, push.rot, 8) || !get(alpha ? 0x35c : 0x348, push.rot + 8, 4) ||
            !get(alpha ? 0x358 : 0x344, push.proj, 16))
            return decline(role, "the skin vertex constants are missing");
        const bool subtract = !((flags & 0x80000) && !(flags & 0x100000));
        if (subtract && !get(alpha ? 0x2c4 : 0x2bc, push.cam, 4)) return decline(role, "no skin camera constants");
        push.cam[3] = subtract ? 1.0f : 0.0f;
        if (alpha && !get(0x2dc, push.world, 1)) return decline(role, "no hair depth offset");
        VkDescriptorImageInfo images[3]{};
        const uint32_t texturePc[] = {alpha ? 0x70u : 0x24u, 0x34, 0x3c};
        const uint32_t textures = alpha ? 3 : 1;
        for (uint32_t i = 0; i < textures; ++i) {
            const Texture* t = nullptr;
            for (const Texture& candidate : d.textures) if (candidate.pc == texturePc[i]) {
                if (t) return decline(role, "duplicate skin texture");
                t = &candidate;
            }
            if (!t || !t->view || !t->sampler) return decline(role, "no skin texture");
            if (t->scale[0] != 1.0f || t->scale[1] != 1.0f) return decline(role, "a scaled skin texture");
            images[i] = {t->sampler, t->view, VK_IMAGE_LAYOUT_GENERAL};
        }
        if (alpha) {
            uint32_t threshold;
            if (!d.load || !d.load(0x44, &threshold, 1)) return decline(role, "no hair alpha threshold");
            std::memcpy(&push.off[3], &threshold, sizeof threshold);
        }
        int32_t sw[4];
        swizzle(d.color[0], sw);
        push.pos[3] = uint32_t(sw[0] | sw[1] << 2 | sw[2] << 4 | sw[3] << 6);
        push.nrm[3] = d.clipZHalf ? 1 : 0;
        Buffer uploaded;
        if (!gpu.upload(constants, sizeof constants, uploaded)) return decline(role, "no room for skin fetch constants");
        buffers[7] = {uploaded.buffer, uploaded.offset, uploaded.range};
        const VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role].exchange(true))
            std::fprintf(stderr, "[native] prepass: drawing %s into compare copies only\n", roleName(role));
        gpu.beginRendering(d.color, 1, d.depth, d.hasStencil);
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        gpu.record([pipe, layout = layouts_[role], push, buffers, images, textures, vp = d.vkViewport, sc,
                    index = d.index, indexType = d.indexType, first = d.firstIndex, count = d.count,
                    pushSet = gpu.pushDescriptorSet()](VkCommandBuffer c) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(c, 0, 1, &vp);
            vkCmdSetScissor(c, 0, 1, &sc);
            VkWriteDescriptorSet writes[11]{};
            for (uint32_t i = 0; i < 8 + textures; ++i) {
                auto& w = writes[i];
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstBinding = i;
                w.descriptorCount = 1;
                if (i < 8) {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    w.pBufferInfo = &buffers[i];
                } else {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w.pImageInfo = &images[i - 8];
                }
            }
            pushSet(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 8 + textures, writes);
            vkCmdPushConstants(c, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, &push);
            vkCmdBindIndexBuffer(c, index.buffer, index.offset + index.byteOffset, indexType);
            vkCmdDrawIndexed(c, count, 1, first, 0, 0);
        });
        return true;
    }

    // Logs each reason once.
    bool decline(int role, const char* why) {
        std::lock_guard lock(m_);
        if (declined_[role].insert(why).second && declined_[role].size() <= 16)
            std::fprintf(stderr, "[native] prepass: a %s draw left to the translated renderer (%s)\n",
                         roleName(role), why);
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        std::lock_guard lock(m_);
        const VkPipelineDepthStencilStateCreateInfo& z = d.depthStencil;
        const auto key = std::make_tuple(role, int(d.color[0].format), int(d.writeMask & 0xF), int(d.depth.format),
                                         bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace));
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        const bool skin = skinRole(role), alpha = alphaRole(role);
        if (!layouts_[role]) {
            VkDescriptorSetLayoutBinding b[11] = {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            if (skin) {
                for (uint32_t i = 0; i < 8; ++i)
                    b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
                for (uint32_t i = 8; i < (alpha ? 11u : 9u); ++i)
                    b[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            }
            dl.bindingCount = skin ? (alpha ? 11 : 9) : (alpha ? 5 : 2);
            dl.pBindings = b;
            if (vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
            const VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, uint32_t(sizeof(Push))};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayouts_[role];
            pl.pushConstantRangeCount = 1;
            pl.pPushConstantRanges = &pr;
            if (vkCreatePipelineLayout(dev, &pl, nullptr, &layouts_[role]) != VK_SUCCESS) {
                layouts_[role] = VK_NULL_HANDLE;
                return out = VK_NULL_HANDLE;
            }
        }
        const std::string defs = std::string(alpha ? "#define ALPHA_TEST\n" : "") + (skin ? "#define PREPASS_SKIN\n" : "");
        const std::string vertex = skin ? std::string(kSkinVs) + kSkinFetch + kSkinMain : std::string(kVs);
        const VkShaderModule vsm = gpu.shader(std::string(kHeader) + defs + vertex, Gpu::Stage::Vertex);
        const VkShaderModule fsm = gpu.shader(std::string(kHeader) + defs + (skin ? kSkinFs : kFs), Gpu::Stage::Fragment);
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
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = d.writeMask & 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &att;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        const VkFormat color = d.color[0].format;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rinfo.colorAttachmentCount = 1;
        rinfo.pColorAttachmentFormats = &color;
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
        pi.layout = layouts_[role];
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, nullptr, &out) != VK_SUCCESS) out = VK_NULL_HANDLE;
        return out;
    }

    std::map<std::tuple<int, int, int, int, bool, VkBool32, VkBool32, int, int, int>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayouts_[kRoleCount] = {};
    VkPipelineLayout layouts_[kRoleCount] = {};
    std::atomic<bool> logged_[kRoleCount] = {};
    // Recorder threads draw concurrently: the pipelines and logs are guarded.
    std::mutex m_;
    std::set<std::string> declined_[kRoleCount];
};

} // namespace

Pass* prepassPass() {
    static Prepass pass;
    return &pass;
}

} // namespace host::native
