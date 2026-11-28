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

enum Role { kOpaque = 1, kAlphaTest = 2, kSkinOpaque = 3, kSkinHair = 4,
            kShadowReceiver = 5, kShadowReceiverPt = 6, kSkinDepth = 7, kTreePt = 8, kRoleCount };

bool skinRole(int role) { return role == kSkinOpaque || role == kSkinHair; }
bool alphaRole(int role) { return role == kAlphaTest || role == kSkinHair; }
const char* roleName(int role) {
    switch (role) {
    case kOpaque: return "ps_normal.pso";
    case kAlphaTest: return "ps_normal_pt.pso";
    case kSkinOpaque: return "ps_normal_mask_n.pso";
    case kSkinHair: return "ps_normal_mask_n_hair_pt.pso";
    case kShadowReceiver: return "ps_sdw_cas3_char1.pso";
    case kShadowReceiverPt: return "ps_sdw_cas3_char1_pt.pso";
    case kSkinDepth: return "ps_depth_pt.pso";
    case kTreePt: return "ps_normal_tree_pt.pso";
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

// Compare-only longtail families. Fetch c[0..6] follows skin_fetch.h;
// c[7..18]: cascade rows, c[19]: secondary object offset,
// c[20]: cascade weights, c[21]: PCF radii, c[22..24]: character rows,
// c[25].x: pixel flags. Existing shader strings above remain unchanged.
constexpr const char* kLongtailVs = R"(
layout(set=0,binding=0,std430) readonly buffer V0 { uint w[]; } v0;
layout(set=0,binding=1,std430) readonly buffer V1 { uint w[]; } v1;
layout(set=0,binding=2,std430) readonly buffer V2 { uint w[]; } v2;
layout(set=0,binding=3,std430) readonly buffer V3 { uint w[]; } v3;
layout(set=0,binding=4,std430) readonly buffer V4 { uint w[]; } v4;
layout(set=0,binding=5,std430) readonly buffer V5 { uint w[]; } v5;
layout(set=0,binding=6,std430) readonly buffer Palette { uint w[]; } palette;
layout(set=0,binding=7,std430) readonly buffer Fetch { uvec4 c[]; } cb;
layout(location=0) out vec4 a0;
#ifndef DEPTH_ONLY
layout(location=1) out vec4 a1;
layout(location=2) out vec4 a2;
layout(location=3) out vec4 a3;
layout(location=4) out vec4 a4;
layout(location=5) out vec4 a5;
#endif
float add(float a,float b) { precise float r=a+b; return r; }
float sub(float a,float b) { precise float r=a-b; return r; }
float mul(float a,float b) { precise float r=a*b; return r; }
float mac(float a,float b,float c) { precise float r=a*b+c; return r; }
float dot3(vec4 a,vec3 b) { return mac(a.z,b.z,mac(a.y,b.y,mul(a.x,b.x))); }
)";
constexpr const char* kLongtailMain = R"(
void main() {
    vec3 x=fetch(0u).xyz, w;
    vec2 uv=vec2(0.0);
#ifdef FAMILY_SKIN
    vec4 fw=fetch(1u);
    uvec4 indices=fetchBits(2u);
    uv=fetch(3u).xy;
    float sum=add(fw.y,fw.z);
    sum=add(sum,fw.w);
    vec4 weights=vec4(sub(1.0,sum),fw.yzw);
    vec4 bone[12];
    for (int k=0;k<4;++k)
        for (int r=0;r<3;++r) bone[3*k+r]=boneRow(indices[k],uint(r));
    for (int r=0;r<3;++r) {
        vec3 M;
        for (int c=0;c<3;++c) {
            float a=mul(bone[r][c],weights.x);
#ifdef DEPTH_ONLY
            if (r==2 && c==0) a=mac(bone[3+r][c],weights.y,a);
#else
            if (r==1 && c==0) a=mac(bone[3+r][c],weights.y,a);
#endif
            else a=mac(weights.y,bone[3+r][c],a);
#ifdef DEPTH_ONLY
            if ((r==1 && c==2) || (r==2 && c==2)) a=mac(bone[6+r][c],weights.z,a);
#else
            if (r==2 && c==2) a=mac(bone[6+r][c],weights.z,a);
#endif
            else a=mac(weights.z,bone[6+r][c],a);
            M[c]=mac(weights.w,bone[9+r][c],a);
        }
        precise float anchor=floor(bone[r].w);
        float s=add(p.off[r],anchor);
        s=mac(weights.x,sub(bone[r].w,anchor),s);
        s=mac(weights.y,sub(bone[3+r].w,anchor),s);
        s=mac(weights.z,sub(bone[6+r].w,anchor),s);
        s=mac(weights.w,sub(bone[9+r].w,anchor),s);
        s=mac(M.x,x.x,s);
        s=mac(M.y,x.y,s);
        w[r]=mac(M.z,x.z,s);
    }
#else
#ifdef ALPHA_TEST
    uv=fetch(1u).xy;
#endif
    for (int r=0;r<3;++r) {
        float s=r==0?add(p.off[r],p.world[r].w):add(p.world[r].w,p.off[r]);
        s=r==2?mac(x.x,p.world[r].x,s):mac(p.world[r].x,x.x,s);
        s=mac(p.world[r].y,x.y,s);
        w[r]=mac(p.world[r].z,x.z,s);
    }
#endif
    vec3 q=w;
    if (p.cam.w!=0.0)
        for (int r=0;r<3;++r) q[r]=sub(w[r],p.cam[r]);
    vec3 e=vec3(dot3(p.rot[0],q),dot3(p.rot[1],q),dot3(p.rot[2],q));
    vec4 clip;
    for (int r=0;r<4;++r) {
        float base=p.proj[r].w;
#ifdef FAMILY_SKIN
        // Both skinned programs add the bias to the projection constant first.
        if (r==2) base=add(p.world[0].x,base);
#endif
        clip[r]=mac(p.proj[r].z,e.z,mac(p.proj[r].y,e.y,mac(p.proj[r].x,e.x,base)));
    }
    gl_Position=clip;
    // Export conversion is not precise in the translator.
    if (p.nrm.w!=0u) gl_Position.z=(gl_Position.z+gl_Position.w)*0.5;
#ifdef DEPTH_ONLY
    a0=vec4(uv,clip.w,0.0);
#else
    vec3 shifted=w;
    if (p.uv.w!=0u)
        for (int r=0;r<3;++r) shifted[r]=add(uintBitsToFloat(cb.c[19][r]),w[r]);
    vec3 shadowPosition=p.uv.w!=0u?q:shifted;
    vec4 cascades[3];
    for (int k=0;k<3;++k)
        for (int r=0;r<4;++r) {
            vec4 m=uintBitsToFloat(cb.c[7+4*k+r]);
            cascades[k][r]=mac(m.z,shadowPosition.z,mac(m.y,shadowPosition.y,mac(m.x,shadowPosition.x,m.w)));
        }
    a0=cascades[2];
    a1=cascades[1];
    a2=cascades[0];
    a3=vec4(mac(uintBitsToFloat(0x3c800000u),-clip.w,1.0),-e.z,uv);
    a4=vec4(shadowPosition,1.0);
    a5=vec4(shifted,0.0);
#endif
}
)";
constexpr const char* kLongtailFs = R"(
layout(set=0,binding=7,std430) readonly buffer Constants { uvec4 c[]; } cb;
layout(location=0) in vec4 a0;
#ifdef DEPTH_ONLY
layout(set=0,binding=8) uniform sampler2D mask0;
layout(set=0,binding=9) uniform sampler2D mask1;
#else
layout(location=1) in vec4 a1;
layout(location=2) in vec4 a2;
layout(location=3) in vec4 a3;
layout(location=4) in vec4 a4;
layout(set=0,binding=8) uniform sampler2DShadow shadow0;
layout(set=0,binding=9) uniform sampler2DShadow shadow1;
layout(set=0,binding=10) uniform sampler2DShadow shadow2;
layout(set=0,binding=11) uniform sampler2DShadow shadow3;
#ifdef ALPHA_TEST
layout(set=0,binding=12) uniform sampler2D mask0;
layout(set=0,binding=13) uniform sampler2D mask1;
#endif
layout(location=0) out vec4 color;
#endif
// PS register boundaries, without precise: the translator permits contraction.
float F(uint x) { return uintBitsToFloat(x); }
uint U(float x) { return floatBitsToUint(x); }
uint sat(float x,bool c) { return U(c?clamp(x,0.0,1.0):x); }
#ifndef DEPTH_ONLY
bool inside(uvec2 uv) {
    uint y=sat(max(F(0u),F(uv.y)),true);
    uint x=sat(max(F(0u),F(uv.x)),true);
    return F(uv.y)==F(y) && F(uv.x)==F(x);
}
// Collapse a four-PC group only when all four resolved bindings agree.
#define PCF(name,tex) \
uint name(uvec3 q,uint radius) { \
    if (!inside(q.xy)) return U(1.0); \
    uint z=sat(min(1.0,F(q.z)),false); \
    uint yp=sat(F(radius)+F(q.y),false); \
    uint xp=sat(F(radius)+F(q.x),false); \
    uint xm=sat(F(q.x)-F(radius),false); \
    uint ym=sat(F(q.y)-F(radius),false); \
    uint t0=U(texture(tex,vec3(F(xp),F(yp),F(z)))); \
    uint t1=U(texture(tex,vec3(F(xm),F(yp),F(z)))); \
    uint t2=U(texture(tex,vec3(F(xp),F(ym),F(z)))); \
    uint t3=U(texture(tex,vec3(F(xm),F(ym),F(z)))); \
    uint sum=sat(F(t0)+F(t1),false); \
    sum=sat(F(sum)+F(t2),false); \
    uint last=sat(F(0x3e800000u)*F(t3),false); \
    return sat(F(sum)*F(0x3e800000u)+F(last),false); \
}
PCF(nearPcf,shadow0)
PCF(middlePcf,shadow1)
PCF(farPcf,shadow2)
PCF(characterPcf,shadow3)
#endif
void main() {
#ifdef DEPTH_ONLY
    uint b=U(texture(mask0,a0.xy).z);
    uint a=U(texture(mask1,a0.xy).w);
    uint test=sat(F(a)*F(b)+(-p.off.w),false);
    if (F(0u)>F(test)) discard;
#else
    vec4 weights=uintBitsToFloat(cb.c[20]);
    uint nearWeight=sat(weights.x*weights.y,false);
    nearWeight=sat((-weights.y)*a3.y+F(nearWeight),true);
    uint ns=U(1.0);
    if (F(0u)<F(nearWeight)) ns=nearPcf(floatBitsToUint(a0.xyz),cb.c[21].x);
    uint farWeight=sat(weights.w*a3.y,false);
    uint remainder=sat(1.0-F(nearWeight),false);
    farWeight=sat((-weights.z)*weights.w+F(farWeight),true);
    uint middleWeight=sat(F(remainder)-F(farWeight),false);
    uint ms=U(1.0);
    if (F(remainder)>F(farWeight)) ms=middlePcf(floatBitsToUint(a1.xyz),cb.c[21].y);
    uint fs=U(1.0);
    if (F(0u)<F(farWeight)) fs=farPcf(floatBitsToUint(a2.xyz),cb.c[21].z);
    vec4 row0=uintBitsToFloat(cb.c[22]),row1=uintBitsToFloat(cb.c[23]);
    uint x=sat(row0.x*a4.x,false);
    uint y=sat(row1.x*a4.x,false);
    y=sat(a4.y*row1.y+F(y),false);
    x=sat(row0.y*a4.y+F(x),false);
    x=sat(a4.z*row0.z+F(x),false);
    y=sat(row1.z*a4.z+F(y),false);
    x=sat(row0.w*a4.w+F(x),false);
#ifdef ALPHA_TEST
    y=sat(a4.w*row1.w+F(y),false);
#else
    y=sat(row1.w*a4.w+F(y),false);
#endif
    uint character=U(1.0);
    if (inside(uvec2(x,y))) {
        vec4 row2=uintBitsToFloat(cb.c[24]);
        uint z=sat(row2.x*a4.x,false);
        z=sat(row2.y*a4.y+F(z),false);
        z=sat(row2.z*a4.z+F(z),false);
        z=sat(row2.w*a4.w+F(z),false);
        character=characterPcf(uvec3(x,y,z),cb.c[21].w);
    }
#ifdef ALPHA_TEST
    // 0x414 samples blue; 0x41c samples alpha. Preserve their product.
    uint b=U(texture(mask0,a3.zw).z);
    uint a=U(texture(mask1,a3.zw).w);
    uint test=sat(F(a)*F(b)+(-p.off.w),false);
    if (F(0u)>F(test)) discard;
#endif
    uint v=sat(F(middleWeight)+F(nearWeight),false);
    v=sat(F(v)+F(farWeight),false);
    v=sat(F(nearWeight)*(-F(ns))+F(v),false);
    v=sat(F(ms)*(-F(middleWeight))+F(v),false);
    v=sat(F(farWeight)*(-F(fs))+F(v),false);
    character=sat(F(v)*(-F(character))+F(character),false);
    uint fade=sat(max(F(0u),a3.x),true);
    fade=sat(F(fade)*F(fade),false);
    bool flag=(cb.c[25].x&8u)!=0u;
    uint rg=packHalf2x16(vec2(flag?1.0:F(character),flag?F(character):1.0));
#ifdef ALPHA_TEST
    uint alpha=sat(F(a)*F(b),false);
    uint ba=packHalf2x16(vec2(F(fade),F(alpha)));
#else
    uint ba=packHalf2x16(vec2(F(fade),1.0));
#endif
    vec4 outValue=vec4(unpackHalf2x16(rg),unpackHalf2x16(ba));
    uint s=p.pos.w;
    color=vec4(outValue[s&3u],outValue[(s>>2)&3u],outValue[(s>>4)&3u],outValue[(s>>6)&3u]);
#endif
}
)";

// Tree prepass (VS 0x53c3cb400, PS 0x53c41da00). Its indexed float4
// wind load uses c[6] in bytes; the four fetch descriptors use c[0..3].
constexpr const char* kTreeVs = R"(
layout(set=0,binding=0,std430) readonly buffer V0 { uint w[]; } v0;
layout(set=0,binding=1,std430) readonly buffer V1 { uint w[]; } v1;
layout(set=0,binding=2,std430) readonly buffer V2 { uint w[]; } v2;
layout(set=0,binding=3,std430) readonly buffer V3 { uint w[]; } v3;
layout(set=0,binding=4,std430) readonly buffer V4 { uint w[]; } v4;
layout(set=0,binding=5,std430) readonly buffer V5 { uint w[]; } v5;
layout(set=0,binding=6,std430) readonly buffer Wind { uint w[]; } palette;
layout(set=0,binding=7,std430) readonly buffer Fetch { uvec4 c[]; } cb;
layout(location=0) out vec4 a0;
float mul(float a,float b) { precise float r=a*b; return r; }
float mac(float a,float b,float c) { precise float r=a*b+c; return r; }
)";
constexpr const char* kTreeMain = R"(
uvec4 treeFetch(uint slot) {
    uvec4 r=fetchBits(slot);
    uint count=(cb.c[slot].z>>8)&7u;
    for (uint i=count;i<4u;++i) r[i]=0u;
    return r;
}
void main() {
    vec4 x=uintBitsToFloat(treeFetch(0u));
    uint index=treeFetch(1u).x;
    vec2 uv=uintBitsToFloat(treeFetch(2u)).xy;
    vec4 b=uintBitsToFloat(treeFetch(3u));
    uvec4 wi=cb.c[6];
    uint a=wi.x+index*wi.y;
    vec4 wind=vec4(0.0);
    if ((wi.y!=0u?index<wi.z:wi.z!=0u) && a<wi.w) {
        uint i=a>>2;
        wind=uintBitsToFloat(uvec4(palette.w[i],palette.w[i+1u],palette.w[i+2u],palette.w[i+3u]));
    }
    vec3 axis=vec3(mac(2.0,b.y,-1.0),mac(2.0,b.z,-1.0),mac(2.0,b.w,-1.0));
    vec3 world;
    for (int i=0;i<3;++i)
        world[i]=mac(p.world[i].w,x.w,mac(p.world[i].z,x.z,mac(p.world[i].y,x.y,mul(p.world[i].x,x.x))));
    float along=mul(wind.x,axis.x);
    along=mac(wind.y,axis.y,along);
    along=mac(axis.z,wind.z,along);
    vec3 perpendicular;
    perpendicular.x=mac(-axis.x,along,wind.x);
    perpendicular.y=mac(along,-axis.y,wind.y);
    perpendicular.z=mac(-axis.z,along,wind.z);
    float length2=mul(perpendicular.x,perpendicular.x);
    length2=mac(perpendicular.y,perpendicular.y,length2);
    length2=mac(perpendicular.z,perpendicular.z,length2);
    length2=mac(wind.w,wind.w,length2);
    precise float length_=sqrt(length2);
    float bend=mul(uintBitsToFloat(0x3e4ccccdu),length_);
    perpendicular.x=mac(bend,axis.x,perpendicular.x);
    perpendicular.y=mac(axis.y,bend,perpendicular.y);
    world.x=mac(perpendicular.x,b.x,world.x);
    world.y=mac(perpendicular.y,b.x,world.y);
    perpendicular.z=mac(bend,axis.z,perpendicular.z);
    world.z=mac(perpendicular.z,b.x,world.z);
    float homogeneous=mac(b.x,wind.w,x.w);
    vec4 clip;
    for (int i=0;i<4;++i)
        clip[i]=mac(p.proj[i].w,homogeneous,mac(p.proj[i].z,world.z,mac(p.proj[i].y,world.y,mul(p.proj[i].x,world.x))));
    gl_Position=clip;
    if (p.nrm.w!=0u) gl_Position.z=(gl_Position.z+gl_Position.w)*0.5;
    a0=vec4(uv,clip.w,0.0);
}
)";
constexpr const char* kTreeFs = R"(
layout(location=0) in vec4 a0;
layout(set=0,binding=8) uniform sampler2D mask0;
layout(location=0) out vec4 color;
void main() {
    float alpha=texture(mask0,a0.xy).w;
    if (p.off.w>alpha) discard;
    vec4 v=vec4(unpackHalf2x16(0x38003800u),unpackHalf2x16(packHalf2x16(vec2(0.0,alpha))));
    uint s=p.pos.w;
    color=vec4(v[s&3u],v[(s>>2)&3u],v[(s>>4)&3u],v[(s>>6)&3u]);
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
        if (psName == "ps_sdw_cas3_char1.pso") return kShadowReceiver;
        if (psName == "ps_sdw_cas3_char1_pt.pso") return kShadowReceiverPt;
        if (psName == "ps_depth_pt.pso") return kSkinDepth;
        if (psName == "ps_normal_tree_pt.pso") return kTreePt;
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        if (role >= kShadowReceiver) return drawLongtail(role, d, gpu);
        const bool skin = skinRole(role), alpha = alphaRole(role);
        // The skinned opaque role matched in Fwha's compare run (~20,000 draws: depth exact, ~40 px at
        // 1/255) and is drawn (LPN_PREPASS_SKIN=0 leaves it to the translated renderer); the hair's
        // normals still differ: compare copies only.
        static const bool skinOn = [] { const char* e = std::getenv("LPN_PREPASS_SKIN"); return !e || *e != '0'; }();
        if (role == kSkinOpaque && !skinOn && !d.compareCopy) return decline(role, "off (LPN_PREPASS_SKIN=0)");
        if (role == kSkinHair && !d.compareCopy) return decline(role, "not yet matched (drawn into compare copies only)");
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
                offset > UINT32_MAX || !format || !plain || !v->records) {
                char why[160];
                std::snprintf(why, sizeof why, "a fetch format it does not read exactly: slot %d dfmt %u nfmt %u components %u dst_sel 0x%x stride %u offset %u",
                              i, unsigned(v->dfmt), unsigned(v->nfmt), unsigned(v->components), unsigned(v->dstSel), unsigned(v->stride), unsigned(v->offset));
                return decline(role, why);
            }
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
            std::fprintf(stderr, "[native] prepass: drawing %s natively (or into compare copies)\n", roleName(role));
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

    bool drawLongtail(int role, const Draw& d, Gpu& gpu) {
        const bool tree=role==kTreePt;
        if (tree && !d.compareCopy)
            return decline(role,"not yet matched (drawn into compare copies only)");
        // Matched in Fwha's compare run (shadow receivers 0 px and within 1/255, skinned depth 0 px over
        // ~15,000 draws): drawn by default; LPN_PREPASS_LONGTAIL=0 leaves them to the translated renderer.
        static const bool on = [] { const char* e = std::getenv("LPN_PREPASS_LONGTAIL"); return !e || *e != '0'; }();
        if (!on && !d.compareCopy) return decline(role, "off (LPN_PREPASS_LONGTAIL=0)");
        const bool depthOnly=role==kSkinDepth, alpha=role!=kShadowReceiver;
        const bool skin=depthOnly || (d.vsName && *d.vsName=="vs_sdw_cas3_skin_pt.vso");
        const char* vs=tree?"lin_depth_rigid_tree_pt.vso +1":depthOnly?"lin_depth_skin_pt.vso +1":
            skin?"vs_sdw_cas3_skin_pt.vso":alpha?"vs_sdw_cas3_normal_pt.vso":"vs_sdw_cas3_normal.vso";
        if (!d.vsName || *d.vsName!=vs || (skin && !alpha))
            return decline(role,"another vertex shader");
        if (d.upscaler || d.pixelPositions || d.depthBias || d.instances!=1 || d.primitive!=4 ||
            d.blend || d.stencil || !d.vsTextures.empty())
            return decline(role,"unsupported raster state or vertex textures");
        if (gpu.maxPushConstants()<sizeof(Push)) return decline(role,"too few push constants");
        if (!d.depth.view || (depthOnly?d.colorCount!=0:d.colorCount!=1 || !d.color[0].view))
            return decline(role,"another attachment shape");
        const auto& z = d.depthStencil;
        // Test/write enables and compare op are copied into the pipeline and
        // included in its key, just as for the rigid roles.
        if (!d.indexed || !d.index.buffer ||
            (d.indexType != VK_INDEX_TYPE_UINT16 && d.indexType != VK_INDEX_TYPE_UINT32))
            return decline(role, "not an indexed triangle list");
        if ((!depthOnly && (d.color[0].format!=VK_FORMAT_R8G8B8A8_UNORM ||
                d.color[0].swap!=1 || d.color[0].comps!=4 || d.writeMask!=(tree?0x7u:0xfu))) ||
            d.depth.format!=VK_FORMAT_D32_SFLOAT_S8_UINT || !d.hasStencil)
            return decline(role,"another target or depth format");
        // DB_DEPTH_CONTROL 0x732: depth test, LEQUAL, no writes or stencil.
        // 0x736 adds depth writes; the disabled stencil functions are ALWAYS.
        if (!z.depthTestEnable || z.depthWriteEnable!=VkBool32(depthOnly || tree) ||
            z.depthCompareOp!=VK_COMPARE_OP_LESS_OR_EQUAL || z.depthBoundsTestEnable || z.stencilTestEnable)
            return decline(role,"another depth state");
        const VertexBuffer* slot[6] = {};
        const VertexBuffer* palette = nullptr;
        constexpr uint32_t depthPalettePc[]={0x3c,0x44,0x4c,0x84,0x8c,0x94,0x9c,0xc4,0xcc,0xe8,0xf0,0xfc};
        constexpr uint32_t receiverPalettePc[]={0x74,0x7c,0x84,0x8c,0x94,0x9c,0xa4,0xc0,0xc8,0xe4,0xec,0xf8};
        const uint32_t* palettePc=depthOnly?depthPalettePc:receiverPalettePc;
        const int slots=(skin || tree)?4:alpha?2:1;
        uint32_t seen = 0;
        for (const VertexBuffer& v : d.vertex) {
            if (v.slot>=0 && !v.components) continue;
            if (v.slot >= 0 && v.slot < slots && v.components) {
                if (slot[v.slot]) return decline(role, "duplicate vertex input");
                slot[v.slot] = &v;
                continue;
            }
            if (tree && v.slot<0) {
                if (palette || v.pc!=0xc || v.offset ||
                    v.stride!=16 || v.dfmt!=14 || v.nfmt!=7 || v.dstSel!=0xfac ||
                    !v.data.buffer || v.data.byteOffset%4 || !v.records ||
                    v.data.range==VK_WHOLE_SIZE || v.data.range>UINT32_MAX ||
                    v.data.byteOffset>v.data.range || v.data.range-v.data.byteOffset<16 ||
                    (v.data.range-v.data.byteOffset)%16)
                    return decline(role,"another wind buffer layout");
                palette=&v;
                continue;
            }
            if (v.slot >= 0 || !skin) return decline(role, "another fetch slot or palette");
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
        if (skin && (seen != 0xfff || !palette)) return decline(role, "a palette load missing");
        if (tree && !palette) return decline(role,"missing tree wind buffer");
        const uint32_t used[]={3,tree?1u:skin?4u:2u,tree?2u:4u,tree?4u:2u};
        uint32_t constants[26][4]{};
        VkDescriptorBufferInfo buffers[8]{};
        for (int i = 0; i < slots; ++i) {
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
            const bool format = tree && i == 1 ?
                (wide ? (v->nfmt==4 || v->nfmt==5 || v->nfmt==7) :
                    (v->nfmt==4 || v->nfmt==5)) : skin && i == 2 ? (v->dfmt == 10 && v->nfmt == 4) :
                (wide ? v->nfmt == 7 : (v->nfmt == 0 || v->nfmt == 1 || (half && v->nfmt == 7)));
            bool plain = v->components >= used[i] && channels >= used[i];
            if (tree) {
                plain = plain && v->components<=4 && v->stride!=0;
                if (i==0 && v->components==4) plain = plain && (((v->dstSel>>9)&7)==7 || ((v->dstSel>>9)&7)==1); // W (absent in a 3-channel format: 1) or the constant 1
            }
            for (uint32_t c = skin && i == 1 ? 1 : 0; c < used[i]; ++c)
                plain = plain && ((v->dstSel >> (3 * c)) & 7) == 4 + c;
            const uint64_t offset = uint64_t(v->data.byteOffset) + v->offset;
            if (!v->data.buffer || v->indexVgpr || v->stride % 4 || offset % 4 ||
                offset > UINT32_MAX || !format || !plain || !v->records) {
                char why[160];
                std::snprintf(why, sizeof why, "a fetch format it does not read exactly: slot %d dfmt %u nfmt %u components %u dst_sel 0x%x stride %u offset %u",
                              i, unsigned(v->dfmt), unsigned(v->nfmt), unsigned(v->components), unsigned(v->dstSel), unsigned(v->stride), unsigned(v->offset));
                return decline(role, why);
            }
            // The renderer may cap the descriptor range independently of
            // NUM_RECORDS; do not require the entire declared stream to fit.
            if (offset > v->data.range || bytes > v->data.range - offset)
                return decline(role, "a skin fetch outside its buffer");
            constants[i][0] = uint32_t(offset / 4);
            constants[i][1] = v->stride / 4;
            constants[i][2] = v->dfmt | v->nfmt << 4;
            if (tree) constants[i][2] |= v->components << 8;
            constants[i][3] = v->records;
            buffers[i] = {v->data.buffer, v->data.offset, v->data.range};
        }
        // Dummy descriptors for streams the selected VS does not read.
        for (int i=slots;i<7;++i) buffers[i]=buffers[0];
        if (skin) {
            constants[6][0]=palette->data.byteOffset/4;
            buffers[6]={palette->data.buffer,palette->data.offset,palette->data.range};
        }
        if (tree) {
            constants[6][0]=palette->data.byteOffset;
            constants[6][1]=palette->stride;
            constants[6][2]=palette->records;
            constants[6][3]=uint32_t(palette->data.range);
            buffers[6]={palette->data.buffer,palette->data.offset,palette->data.range};
        }
        Push push{};
        auto get=[&](uint32_t pc,float* out,uint32_t n) {
            return d.vsLoad && d.vsLoad(pc,reinterpret_cast<uint32_t*>(out),n);
        };
        auto load=[&](uint32_t pc,uint32_t* out,uint32_t n) {
            return d.load && d.load(pc,out,n);
        };
        uint32_t flags=0;
        if (tree) {
            if (!get(0x2c,push.world,8) || !get(0x30,push.world+8,4) ||
                !get(0x34,push.proj,16))
                return decline(role,"missing tree vertex constants");
        } else {
            if (!get(skin?0x104:0x10,push.off,4) ||
                !d.vsLoad || !d.vsLoad(skin?(depthOnly?0x11c:0x118):0x24,&flags,1) ||
                !get(skin?(depthOnly?0x2e0:0x32c):0x11c,push.rot,8) ||
                !get(skin?(depthOnly?0x2e8:0x334):0x124,push.rot+8,4) ||
                !get(skin?(depthOnly?0x2e4:0x330):0x120,push.proj,16))
                return decline(role,"missing longtail vertex constants");
            const bool subtract=!((flags&0x80000) && !(flags&0x100000));
            if (subtract && !get(skin?(depthOnly?0x2bc:0x2c0):0xb4,push.cam,4))
                return decline(role,"missing longtail camera");
            push.cam[3]=subtract?1.0f:0.0f;
            if (skin) {
                if (!get(depthOnly?0x2d8:0x2f8,push.world,1)) return decline(role,"missing object depth offset");
            } else if (!get(0x1c,push.world,8) || !get(0x20,push.world+8,4))
                return decline(role,"missing rigid world matrix");
            if (!depthOnly) {
                for (int i=0;i<3;++i)
                    if (!d.vsLoad(skin?0x2fc+4*i:0xec+4*i,constants[7+4*i],16))
                        return decline(role,"missing receiver cascade matrix");
                if ((flags&0x80000) && !d.vsLoad(skin?0x2dc:0xd0,constants[19],4))
                    return decline(role,"missing receiver secondary offset");
                if (!load(0x14,constants[20],4) ||
                    !load(0x94,&constants[21][0],1) || !load(0x180,&constants[21][1],1) ||
                    !load(0x254,&constants[21][2],1) ||
                    !load(alpha?0x370:0x364,&constants[21][3],1) ||
                    !load(0x2dc,constants[22],8) || !load(alpha?0x36c:0x360,constants[24],4) ||
                    !load(alpha?0x44c:0x3f4,constants[25],1))
                    return decline(role,"missing receiver pixel constants");
            }
        }
        if (alpha) {
            uint32_t threshold;
            if (!load(tree?0xc:depthOnly?0x44:0x424,&threshold,1)) return decline(role,"missing alpha threshold");
            std::memcpy(&push.off[3],&threshold,4);
        }
        if (!depthOnly) {
            int32_t sw[4];
            swizzle(d.color[0],sw);
            push.pos[3]=uint32_t(sw[0]|sw[1]<<2|sw[2]<<4|sw[3]<<6);
        }
        push.nrm[3]=d.clipZHalf?1:0;
        push.uv[3]=(flags&0x80000)?1:0;
        VkDescriptorImageInfo images[6]{};
        const uint32_t textures=tree?1:depthOnly?2:alpha?6:4;
        auto texture=[&](uint32_t pc)->const Texture* {
            const Texture* result=nullptr;
            for (const Texture& t:d.textures) if (t.pc==pc) {
                if (result) return nullptr;
                result=&t;
            }
            return result;
        };
        const uint32_t groups[4][4]={
            {0xc0,0xc8,0xd4,0xe0},{0x1ac,0x1b4,0x1c0,0x1cc},{0x280,0x288,0x294,0x2a0},
            {alpha?0x3a0u:0x394u,alpha?0x3a8u:0x39cu,alpha?0x3b4u:0x3a8u,alpha?0x3c0u:0x3b4u}
        };
        if (d.textures.size()!=(tree?1u:depthOnly?2u:alpha?18u:16u))
            return decline(role,"another texture instruction shape");
        for (uint32_t i=0;i<textures;++i) {
            const uint32_t pc=tree?0x28:depthOnly?(i==0?0x34:0x3c):
                i<4?groups[i][0]:i==4?0x414:0x41c;
            const Texture* t=texture(pc);
            if (!t || !t->view || !t->sampler || t->scale[0]!=1 || t->scale[1]!=1 ||
                t->address==d.depth.address || (!depthOnly && t->address==d.color[0].address))
                return decline(role,"missing, scaled or aliased texture");
            if (!tree && !depthOnly && i<4) {
                if (t->width!=1024 || t->height!=1024) return decline(role,"another shadow map size");
                for (int j=1;j<4;++j) {
                    const Texture* other=texture(groups[i][j]);
                    if (!other || other->view!=t->view || other->sampler!=t->sampler ||
                        other->address!=t->address || other->width!=t->width || other->height!=t->height ||
                        other->scale[0]!=1 || other->scale[1]!=1)
                        return decline(role,"PCF bindings disagree");
                }
            }
            // resolveImage supplies comparison samplers and depth-aspect views,
            // using im.compare, as for post's kShadow role.
            images[i]={t->sampler,t->view,VK_IMAGE_LAYOUT_GENERAL};
        }
        Buffer uploaded;
        if (!gpu.upload(constants, sizeof constants, uploaded)) return decline(role, "no room for skin fetch constants");
        if (uploaded.byteOffset) return decline(role,"nonzero constant upload offset");
        buffers[7] = {uploaded.buffer, uploaded.offset, uploaded.range};
        const VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role].exchange(true))
            std::fprintf(stderr, "[native] prepass: drawing %s natively (or into compare copies)\n", roleName(role));
        gpu.beginRendering(d.color, d.colorCount, d.depth, d.hasStencil);
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        gpu.record([pipe, layout = layouts_[role], push, buffers, images, textures, vp = d.vkViewport, sc,
                    index = d.index, indexType = d.indexType, first = d.firstIndex, count = d.count,
                    pushSet = gpu.pushDescriptorSet()](VkCommandBuffer c) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(c, 0, 1, &vp);
            vkCmdSetScissor(c, 0, 1, &sc);
            VkWriteDescriptorSet writes[14]{};
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
        const bool family = role >= kShadowReceiver;
        const bool familySkin = role == kSkinDepth ||
            (family && d.vsName && *d.vsName == "vs_sdw_cas3_skin_pt.vso");
        // The same _pt PS claims both rigid and skinned receiver VSs.
        const auto key = std::make_tuple(role + (familySkin ? 256 : 0), int(d.color[0].format), int(d.writeMask & 0xF), int(d.depth.format),
                                         bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace));
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        const bool skin = skinRole(role), alpha = alphaRole(role);
        if (!layouts_[role]) {
            VkDescriptorSetLayoutBinding b[14] = {
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
            if (family) {
                for (uint32_t i=0;i<8;++i)
                    b[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,
                          VkShaderStageFlags(i==7?VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT),nullptr};
                const uint32_t images=role==kTreePt?1:role==kSkinDepth?2:role==kShadowReceiverPt?6:4;
                for (uint32_t i=8;i<8+images;++i)
                    b[i]={i,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            }
            dl.bindingCount = family ? (role==kTreePt?9:role==kSkinDepth?10:role==kShadowReceiverPt?14:12) :
                skin ? (alpha ? 11 : 9) : (alpha ? 5 : 2);
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
        const std::string defs = family ?
            std::string("#define PREPASS_SKIN\n") +
            (role!=kShadowReceiver ? "#define ALPHA_TEST\n" : "") +
            (familySkin ? "#define FAMILY_SKIN\n" : "") +
            (role==kSkinDepth ? "#define DEPTH_ONLY\n" : "") :
            std::string(alpha ? "#define ALPHA_TEST\n" : "") + (skin ? "#define PREPASS_SKIN\n" : "");
        const std::string vertex = role==kTreePt ? std::string(kTreeVs) + kSkinFetch + kTreeMain : family ? std::string(kLongtailVs) + kSkinFetch + kLongtailMain :
            skin ? std::string(kSkinVs) + kSkinFetch + kSkinMain : std::string(kVs);
        const VkShaderModule vsm = gpu.shader(std::string(kHeader) + defs + vertex, Gpu::Stage::Vertex);
        const VkShaderModule fsm = gpu.shader(std::string(kHeader) + defs +
            (role==kTreePt ? kTreeFs : family ? kLongtailFs : skin ? kSkinFs : kFs), Gpu::Stage::Fragment);
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
        cb.attachmentCount = d.colorCount;
        cb.pAttachments = &att;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        const VkFormat color = d.color[0].format;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rinfo.colorAttachmentCount = d.colorCount;
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
