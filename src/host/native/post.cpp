// Bloom/downsampling, SSAO and outlines, from the drawn GCN programs in
// dis/post, dis/ssao and dis/outline.
// Fetch the quad rather than assuming its UVs cover [0,1]: glow_pass1 and SSAO
// remap them, and SSAO's named vertex shaders also export view rays.
// The luminance VS is capture-5's vs-53c2a1500.bin (fx_depth_to_shadow_unuse_map.fxo +8).
#include "host/native/native.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

namespace host::native {
namespace {

enum Role { kDown2 = 1, kDown4, kLumi, kBlur, kCompose, kSsao, kTemporal, kSsaoBlur, kUpsample,
            kHandDraw2, kHandDrawFinalize, kToonEdgeFxaa, kHaze };
static_assert(kHaze<24); // renderer.cpp's kCompareRoles.
struct Variant {
    const char* ps;
    const char* vs;
    std::array<uint32_t, 18> samples;
};
constexpr Variant kRoles[] = {
    {},
    {"ps_down_sample_2x4.pso", "vs_down_sample_2x4.vso", {0x34, 0x3c}},
    {"ps_down_sample_4x4.pso", "vs_down_sample_4x4.vso", {0x54, 0x5c, 0x64, 0x6c}},
    {"ps_glow_pass0_lumi.pso", "fx_depth_to_shadow_unuse_map.fxo +8", {0x38, 0x80}},
    {"ps_glow_pass1_ca.pso", "vs_glow_pass1.vso",
     {0xf8, 0x100, 0x108, 0x110, 0x150, 0x158, 0x160, 0x168, 0x180,
      0x1b8, 0x1c0, 0x1c8, 0x218, 0x220, 0x228, 0x248, 0x250, 0x258}},
    {"ps_glow_pass2_5.pso", "vs_glow_pass2.vso", {0x40, 0x64, 0x84, 0xa4, 0xb8, 0xec}},
    {"fx_ssao_test[dnrm].fxo", "fx_ssao.fxo +4",
     {0x50, 0x98, 0x314, 0x3e8, 0x4dc, 0x558, 0x70c, 0x818, 0x96c, 0x994, 0xbec, 0xd1c, 0xda0, 0xde4}},
    {"fx_ssao_temporal_filter.fxo", "fx_ssao_temporal_filter.fxo",
     {0x34, 0x1d0, 0x1d8, 0x208, 0x210}},
    {"ps_ssao_blur.pso", nullptr,
     {0x88, 0x90, 0x98, 0xa0, 0xac, 0xb4, 0xbc, 0xc4, 0x110, 0x118, 0x120, 0x128, 0x130, 0x138, 0x140, 0x148}},
    {"ps_ssao_upsampling.pso", nullptr,
     {0xf4, 0xfc, 0x118, 0x120, 0x128, 0x130, 0x138, 0x140, 0x148}},
    {"ps_hand_draw_2.pso", "fx_depth_to_shadow_unuse_map.fxo +8", {0x28}},
    {"fx_hand_draw_finalize.fxo", "fx_hand_draw_finalize.fxo", {0x3c, 0x50}},
    {"ps_toon_edge_fxaa.pso", nullptr, {0x50, 0x58, 0x60, 0x6c, 0x74, 0x110, 0x128, 0x140, 0x158}},
    {"fx_haze_pass0.fxo +3", nullptr, {0x24}},
};

// All 49 MIMG instructions in hand_draw_2 read s[0:7], s[8:11].
// Validate that binding at every PC before sharing its descriptor and scale.
constexpr uint32_t kHandDrawSamples[] = {
    0x28, 0x74, 0x7c, 0x140, 0x148, 0x210, 0x218, 0x2cc, 0x2d4,
    0x394, 0x39c, 0x458, 0x460, 0x50c, 0x514, 0x5cc, 0x5d4, 0x6d0,
    0x6d8, 0x6e0, 0x6f0, 0x700, 0x79c, 0x7cc, 0x7d4, 0x890, 0x898,
    0x8a8, 0x8b0, 0x8b8, 0x968, 0x998, 0x9a0, 0xa4c, 0xa54, 0xb14,
    0xb1c, 0xbdc, 0xbe4, 0xca0, 0xca8, 0xd84, 0xda8, 0xdb0, 0xdc0,
    0xdcc, 0xde0, 0xde8, 0xebc,
};

// c0: VS position transform; c1..3: VS constants; c4..31: PS scalar loads;
// c32..49: texture scales, one per MIMG instruction, even for the same T#.
// SSAO's scalar-load PCs and destination rows are listed together in draw().
// Luminance uses c4=0x44, c5..6=0x90 (when enabled), c7.x=0x40 flags.
using Constants = std::array<std::array<uint32_t, 4>, 50>;
static_assert(sizeof(Constants) == 50*16);
struct Push {
    uint32_t fetch[2][4]; // word offset, word stride, stored component count, DST_SEL
    int32_t swizzle[4];
    uint32_t clipZHalf, pad[3];
};
static_assert(sizeof(Push) == 64);
constexpr VkShaderStageFlags kStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

constexpr const char* kHeader = R"(#version 450
layout(push_constant) uniform P {
    uvec4 fetch[2];
    ivec4 swizzle;
    uvec4 flags;
} p;
layout(set=0, binding=2, std430) readonly buffer Constants { vec4 c[]; };
float mul(float a, float b) { precise float r=a*b; return r; }
float mac(float a, float b, float d) { precise float r=a*b+d; return r; }
vec2 mac(vec2 a, vec2 b, vec2 d) { precise vec2 r=a*b+d; return r; }
vec4 mac(vec4 a, vec4 b, vec4 d) { precise vec4 r=a*b+d; return r; }
)";

constexpr const char* kVs = R"(
layout(set=0, binding=0, std430) readonly buffer V0 { uint w[]; } v0;
layout(set=0, binding=1, std430) readonly buffer V1 { uint w[]; } v1;
layout(location=0) out vec4 a0;
layout(location=1) out vec4 a1;
layout(location=2) out vec4 a2;
vec4 fetch(int slot) {
    uvec4 f=p.fetch[slot];
    uint index=uint(gl_VertexIndex);
    vec4 r=vec4(0.0,0.0,0.0,1.0);
    uint base=f.x+index*f.y;
    for (uint j=0u;j<f.z;++j)
        r[j]=uintBitsToFloat(slot==0?v0.w[base+j]:v1.w[base+j]);
    vec4 result;
    for (uint j=0u;j<4u;++j) {
        uint s=(f.w>>(3u*j))&7u;
        result[j]=s>=4u?r[s-4u]:(s==1u?1.0:0.0);
    }
    return result;
}
void main() {
    vec4 pos=fetch(0);
    vec2 uv=vec2(0.0);
#if ROLE < 8
    uv=fetch(1).xy;
#endif
    gl_Position=vec4(mac(c[0].xy,pos.xy,c[0].zw),pos.zw);
    if (p.flags.x!=0u) gl_Position.z=mul(gl_Position.z+gl_Position.w,0.5);
    a0=a1=a2=vec4(0.0);
#if ROLE == 1
    a0=vec4(uv.x,uv.y-c[1].x,uv.x,uv.y+c[1].x);
#elif ROLE == 2
    a0=vec4(uv.x-c[1].x,uv.y-c[1].y,
            mac(vec2(-1.0),c[1].xx,uv.xx).x,uv.y+c[1].y);
    a1=vec4(uv.x+c[1].x,uv.y-c[1].y,uv.x+c[1].x,uv.y+c[1].y);
#elif ROLE == 3
    // Captured vs-53c2a1500: load 0x10, param0=(v8,v9,0,0).
    a0=vec4(uv,0.0,0.0);
#elif ROLE == 4
    // madmk's printed third operand is the addend; the literal is the multiplier.
    a0=vec4(mac(c[1].xy,vec2(-2.5),uv),mac(c[1].xy,vec2(-1.5),uv));
    a1=vec4(mac(vec2(-0.5),c[1].xy,uv),mac(vec2(0.5),c[1].xy,uv));
    a2=vec4(mac(c[1].xy,vec2(1.5),uv),mac(c[1].xy,vec2(2.5),uv));
#elif ROLE == 5
    a0=vec4(c[1].xy+uv,c[1].zw+uv);
    a1=vec4(c[2].xy+uv,c[2].zw+uv);
    a2=vec4(c[3].xy+uv,c[3].zw+uv);
#elif ROLE == 6
    // 0x50 param1: UV and the normal-map UV; 0x58 param0: view ray.
    a1=vec4(uv,mac(c[2].xy,uv,c[2].zw));
    a0=vec4(c[1].xy*gl_Position.xy,c[1].z,0.0);
#elif ROLE == 7
    a1=vec4(uv,0.0,0.0);
    a0=vec4(c[1].xy*gl_Position.xy,c[1].z,0.0);
#endif
}
)";

// Same corner selection and fourth-vertex arithmetic as renderer.cpp's expandShader.
constexpr const char* kGs = R"(#version 450
layout(triangles) in;
layout(triangle_strip,max_vertices=4) out;
in gl_PerVertex { vec4 gl_Position; } gl_in[];
out gl_PerVertex { vec4 gl_Position; };
layout(location=0) in vec4 a0[];
layout(location=1) in vec4 a1[];
layout(location=2) in vec4 a2[];
layout(location=0) out vec4 b0;
layout(location=1) out vec4 b1;
layout(location=2) out vec4 b2;
void emit(int i) {
    gl_Position=gl_in[i].gl_Position;
    b0=a0[i]; b1=a1[i]; b2=a2[i];
    EmitVertex();
}
void main() {
    vec2 p0=gl_in[0].gl_Position.xy/gl_in[0].gl_Position.w;
    vec2 p1=gl_in[1].gl_Position.xy/gl_in[1].gl_Position.w;
    vec2 p2=gl_in[2].gl_Position.xy/gl_in[2].gl_Position.w;
    float d0=abs(dot(p1-p0,p2-p0)),d1=abs(dot(p0-p1,p2-p1)),d2=abs(dot(p0-p2,p1-p2));
    int c=(d0<=d1 && d0<=d2)?0:(d1<=d2?1:2);
    int a=(c+1)%3,b=(c+2)%3;
    emit(c); emit(a); emit(b);
    gl_Position=gl_in[a].gl_Position+gl_in[b].gl_Position-gl_in[c].gl_Position;
    b0=a0[a]+a0[b]-a0[c];
    b1=a1[a]+a1[b]-a1[c];
    b2=a2[a]+a2[b]-a2[c];
    EmitVertex(); EndPrimitive();
}
)";

constexpr const char* kFs = R"(
layout(location=0) in vec4 a0;
layout(location=1) in vec4 a1;
layout(location=2) in vec4 a2;
layout(location=0) out vec4 outColour;
layout(set=0,binding=3) uniform sampler2D tex[TEXTURES];
// Literal indices keep this independent of sampled-image array dynamic indexing features.
#define sampleAt(i,uv) texture(tex[i],(uv)*c[32+(i)].xy)
void put(vec4 v) {
    // Match the translated compressed export's half conversion before COMP_SWAP.
    v=vec4(unpackHalf2x16(packHalf2x16(v.xy)),unpackHalf2x16(packHalf2x16(v.zw)));
    outColour=vec4(v[p.swizzle.x],v[p.swizzle.y],v[p.swizzle.z],v[p.swizzle.w]);
}
float luminance(vec3 v) {
    precise float r=mul(uintBitsToFloat(0x3e990afeu),v.r);
    r=v.g*uintBitsToFloat(0x3f162c23u)+r;
    r=v.b*uintBitsToFloat(0x3dea7371u)+r;
    return r;
}
void main() {
#if ROLE == 1
    precise vec4 x=sampleAt(0,a0.zw),y=sampleAt(1,a0.xy);
    put((x+y)*0.5);
#elif ROLE == 2
    precise vec4 x=sampleAt(0,a1.zw),y=sampleAt(1,a1.xy);
    precise vec4 z=sampleAt(2,a0.zw),w=sampleAt(3,a0.xy);
    // Red and GBA have different addition trees in the program.
    precise float r=((x.r+y.r)+z.r)+w.r;
    precise vec3 gba=((w.gba+z.gba)+y.gba)+x.gba;
    put(vec4(r,gba)*0.25);
#elif ROLE == 3
    precise vec3 base=sampleAt(0,a0.xy).rgb;
    precise vec4 v=vec4(c[4].rgb*base,mul(c[4].a,luminance(base)));
#if TEXTURES == 2
    precise vec3 colour=sampleAt(1,a0.xy).rgb;
    precise float l=luminance(colour);
    precise float delta=max(0.0,l-c[5].x);
    precise float denominator=l+uintBitsToFloat(0x3a83126fu);
    precise float factor=mul(1.0/denominator,delta);
    precise vec3 extra=c[6].yzw*colour;
    extra=clamp(extra*factor,0.0,1.0);
    precise float alpha=clamp(mul(c[6].x,delta),0.0,1.0);
    v+=vec4(extra,alpha);
#endif
    put(v);
#elif ROLE == 4
    vec2 uv[6]=vec2[6](a0.xy,a0.zw,a1.xy,a1.zw,a2.xy,a2.zw);
    precise vec2 dr=c[7].zw-c[7].xy;
    precise vec2 dg=c[8].zw-c[8].xy;
    precise vec2 db=c[9].zw-c[9].xy;
    // Each sample keeps its own sampler and scale, identified by MIMG PC.
    precise vec4 t0=sampleAt(0,mac(uv[4],dr,c[7].xy));
    precise vec4 t1=sampleAt(1,mac(uv[5],dg,c[8].xy));
    precise vec4 t2=sampleAt(2,mac(uv[4],dg,c[8].xy));
    precise vec4 t3=sampleAt(3,mac(uv[1],dg,c[8].xy));
    precise vec4 t4=sampleAt(4,mac(uv[1],dr,c[7].xy));
    precise vec4 t5=sampleAt(5,mac(uv[0],dr,c[7].xy));
    precise vec4 t6=sampleAt(6,mac(uv[3],dr,c[7].xy));
    precise vec4 t7=sampleAt(7,mac(uv[2],dr,c[7].xy));
    precise vec4 t8=sampleAt(8,mac(uv[0],dg,c[8].xy));
    precise vec4 t9=sampleAt(9,mac(uv[5],dr,c[7].xy));
    precise vec4 t10=sampleAt(10,mac(uv[3],dg,c[8].xy));
    precise vec4 t11=sampleAt(11,mac(uv[2],dg,c[8].xy));
    precise vec4 t12=sampleAt(12,mac(uv[5],db,c[9].xy));
    precise vec4 t13=sampleAt(13,mac(uv[0],db,c[9].xy));
    precise vec4 t14=sampleAt(14,mac(uv[3],db,c[9].xy));
    precise vec4 t15=sampleAt(15,mac(uv[4],db,c[9].xy));
    precise vec4 t16=sampleAt(16,mac(uv[1],db,c[9].xy));
    precise vec4 t17=sampleAt(17,mac(uv[2],db,c[9].xy));
    precise vec4 v;
    v.r=c[6].r*(t9.r+t5.r);
    v.g=c[6].g*(t8.g+t1.g);
    v.a=c[6].a*(t5.a+t9.a);
    v.r=c[5].r*(t0.r+t4.r)+v.r;
    v.g=c[5].g*(t3.g+t2.g)+v.g;
    v.a=c[5].a*(t4.a+t0.a)+v.a;
    v.r=c[4].r*(t6.r+t7.r)+v.r;
    v.a=c[4].a*(t7.a+t6.a)+v.a;
    v.g=c[4].g*(t11.g+t10.g)+v.g;
    v.b=c[6].b*(t13.b+t12.b);
    v.b=c[5].b*(t16.b+t15.b)+v.b;
    v.b=c[4].b*(t17.b+t14.b)+v.b;
    put(v);
#elif ROLE == 5
    precise vec4 t0=sampleAt(0,a0.zw),t1=sampleAt(1,a0.xy);
    precise vec4 t2=sampleAt(2,a1.xy),t3=sampleAt(3,a1.zw);
    precise vec4 t4=sampleAt(4,a2.xy),t5=sampleAt(5,a2.zw);
    precise float alpha=(((t1.a+t0.a)+t2.a)+t3.a)+t4.a;
    alpha=mul(c[4].x,alpha);
    alpha=mul(alpha,uintBitsToFloat(0x3e23d70au));
    precise float r=(((t4.r+t3.r)+t2.r)+t0.r)+t1.r;
    precise float g=(((t1.g+t0.g)+t2.g)+t3.g)+t4.g;
    precise float b=(((t1.b+t0.b)+t2.b)+t3.b)+t4.b;
    precise vec3 extra=c[5].xxx*t5.rgb;
    extra=vec3(0.2)*extra;
    precise vec3 rgb=vec3(r,g,b)*uintBitsToFloat(0x3e23d70au)+extra;
    rgb=clamp(c[4].yzw*rgb+vec3(alpha),0.0,1.0);
    put(vec4(rgb,max(max(rgb.b,rgb.r),rgb.g)));
#endif
}
)";

// The SSAO ports keep the captured register names where interleaved arithmetic matters.
// Scalar loads and MIMG instructions are identified by PC; samples are separate even
// when they read the same T#, so each instruction retains the game's sampler and scale.
constexpr const char* kSsaoFs = R"(
layout(location=0) in vec4 a0;
layout(location=1) in vec4 a1;
layout(location=2) in vec4 a2;
layout(location=0) out vec4 outColour;
layout(set=0,binding=3) uniform sampler2D tex[TEXTURES];
#define sampleAt(i,uv) texture(tex[i],(uv)*c[32+(i)].xy)
float B(uint x) { return uintBitsToFloat(x); }
// Pixel arithmetic follows translator.cpp's pixel fres(): contraction is allowed.
// Keep kHeader's noncontracting helpers for vertex/glow shaders.
float ssaoMul(float a, float b) { return a*b; }
float ssaoMac(float a, float b, float d) { return a*b+d; }
#define mul ssaoMul
#define mac ssaoMac
void put(vec4 v) {
    // Match the current translator's opcode 303 (packHalf2x16, round to nearest).
    // GCN's RTZ conversion differs; changing that belongs in both paths together.
    v=vec4(unpackHalf2x16(packHalf2x16(v.xy)),unpackHalf2x16(packHalf2x16(v.zw)));
    outColour=vec4(v[p.swizzle.x],v[p.swizzle.y],v[p.swizzle.z],v[p.swizzle.w]);
}
#if ROLE == 9
// ps_ssao_upsampling.pso's complete literal table (the short code dump cuts it off).
// The V# starts at PC 0x334; load 0x150 addresses 48 - 16*(xParity+2*yParity).
const vec4 weights[4]=vec4[4](
    vec4(0.5625,0.1875,0.1875,0.0625),
    vec4(0.1875,0.5625,0.0625,0.1875),
    vec4(0.1875,0.0625,0.5625,0.1875),
    vec4(0.0625,0.1875,0.1875,0.5625));
#endif
void main() {
#if ROLE == 6
    float v0,v1,v2,v3,v4,v5,v6,v7;
    float v8,v9,v10,v11,v12,v13,v14,v15;
    float v16,v17,v18,v19,v20,v21,v22,v23;
    float v24,v25,v26,v27,v28,v29,v30,v31;
    float v32,v33,v34,v35,v36,v37,v38,v39;
    float v40,v41,v42,v43,v44,v45,v46,v47;
    float s12,s13,s14,s15,s16,s17,s18,s19;
    float s20,s21,s22,s23,s24,s25,s26,s27;
    float s28,s29,s30,s31,s32,s33,s34,s35;
    float s36,s37,s38,s39,s40,s41,s42,s43;
    float s48,s50,s51;
    bool b28,b36,b44,b46;
    v2=gl_FragCoord.x; v3=gl_FragCoord.y;
    s48=B(0x3c800000u);
    // Scalar load 0x2c.
    s40=c[13].x; s41=c[13].y; v6=B(0xbe800000u)+v2; v7=B(0xbe800000u)+v3; v4=mul(s40,v6);
    v5=mul(s41,v7);
    // MIMG 0x50: image_sample, dmask 0x1.
    v5=sampleAt(0,vec2(v4,v5)).x;
    // PS attr0 is VS param1 (the remapped normal-map UV in .zw).
    // Scalar load 0x58.
    s12=c[4].x; s13=c[4].y; s14=c[4].z; s15=c[4].w; s16=c[5].x; s17=c[5].y; s18=c[5].z; s19=c[5].w;
    s20=c[6].x; s21=c[6].y; s22=c[6].z; s23=c[6].w; s24=c[7].x; s25=c[7].y; s26=c[7].z; s27=c[7].w;
    v0=mac(s41,v7,-0.5); v2=a1.z; v3=a1.w; v1=mac(s40,v6,-0.5); v4=1.0/(s17); v8=mul(v4,-v0);
    v0=1.0/(s12); v6=mul(v0,-v1);
    // MIMG 0x98: image_sample, dmask 0x7.
    { vec4 t=sampleAt(1,vec2(v2,v3));
      v0=t.x;   v1=t.y;   v2=t.z;
    }
    // Scalar load 0xa0.
    s28=c[8].x; s29=c[8].y; s30=c[8].z; s31=c[8].w; s32=c[9].x; s33=c[9].y; s34=c[9].z; s35=c[9].w;
    s36=c[10].x; s37=c[10].y; s38=c[10].z; s39=c[10].w; s40=c[11].x; s41=c[11].y; s42=c[11].z;
    s43=c[11].w; v9=s27;
    // Scalar load 0xa8.
    s50=c[12].x; s51=c[12].y; b44=s28>0.5; b28=s29>0.5; v17=s41; v13=s33; b46=s37>0.5; b36=s36>0.5;
    v3=s22+v5; v3=1.0/(v3); v10=mul(-s23,v3); v11=subgroupQuadBroadcast(v10,2u); v6=mul(v6,v10);
    v4=subgroupQuadBroadcast(v10,1u); v7=subgroupQuadBroadcast(v10,0u); v18=log2(abs(v5));
    v5=mul(v8,v10); v4=v4-v7; v7=v11-v7; v11=mul(-2.0,v5); v12=subgroupQuadBroadcast(v11,1u);
    v8=subgroupQuadBroadcast(v11,0u); v14=subgroupQuadBroadcast(v11,2u); v0=mac(2.0,v0,-1.0);
    v1=mac(2.0,v1,-1.0); v15=v12-v8; v12=mul(2.0,v6); v16=subgroupQuadBroadcast(v12,1u);
    v19=subgroupQuadBroadcast(v12,0u); v20=subgroupQuadBroadcast(v12,2u); v14=v14-v8; v8=mul(v4,v14);
    v16=v16-v19; v19=v20-v19; v8=mac(v7,-v15,v8); v20=mul(v15,v19); v20=mac(v14,-v16,v20);
    v14=mul(v16,v7); v7=mul(v8,v8); v14=mac(v19,-v4,v14); v7=mac(v14,v14,v7); v7=mac(v20,v20,v7);
    v15=inversesqrt(v7); v7=mul(v15,v14); v14=mac(2.0,v2,-1.0); v21=b28?0.0:v7; v4=mul(v15,v20);
    v8=mul(v15,v8); v15=mul(v0,v0); v15=mac(v1,v1,v15); v15=mac(v14,v14,v15); v16=inversesqrt(v15);
    v15=mul(v16,v14); v14=mul(v16,v0); v16=mul(v16,v1); v1=b44?v15:0.0; v23=b44?v16:0.0;
    v27=mul(v1,v21); v26=b28?1.0:v4; v24=b28?0.0:v8; v29=b44?v14:0.0; v0=mul(s31,-v10);
    v28=mul(v1,v24); v1=b44?v2:1.0; v1=mul(s30,v1); v19=max(v1,v0); v1=B(0x3f3504f3u)+v23;
    v0=B(0x3f3504f3u)+v29; v30=mac(-v26,v0,v28); v38=mul(v21,v0); v0=mac(-v24,v1,v38);
    v1=mac(v26,v1,-v27); v20=mul(v1,v1); v20=mac(v30,v30,v20); v20=mac(v0,v0,v20);
    v20=inversesqrt(v20); v22=mul(v1,v20); v36=mac(v22,v19,v12); v1=mul(v30,v20); v35=mac(v1,v19,v11);
    v0=mul(v0,v20); v20=mul(s12,v36); v25=mac(v0,v19,v10); v20=mac(s13,v35,v20); v20=mac(s14,v25,v20);
    v33=mac(-v22,v19,v12); v31=mac(-v1,v19,v11); v1=mac(s24,v36,v9); v1=mac(s25,v35,v1);
    v1=mac(s26,v25,v1); v1=1.0/(v1); v37=mul(s16,v33); v22=mac(-v0,v19,v10); v0=s15+v20;
    v20=mul(s16,v36); v20=mac(s17,v35,v20); v20=mac(s18,v25,v20); v20=s19+v20; v0=mul(v1,v0);
    v1=mul(v1,v20); v0=mac(0.5,v0,0.5); v1=mac(-0.5,v1,0.5); v37=mac(s17,v31,v37);
    v37=mac(s18,v22,v37); v37=s19+v37;
    // MIMG 0x314: image_sample, dmask 0x1.
    v34=sampleAt(2,vec2(v0,v1)).x; v0=mul(s35,v18); v1=exp2(v0); v0=mul(s43,v18); v20=mac(s34,v1,v13);
    v1=exp2(v0); v0=mul(s12,v33); v0=mac(s13,v31,v0); v0=mac(s14,v22,v0); v0=s15+v0;
    v39=mac(-v8,v20,v33); v39=mac(-2.0,v6,v39); v36=mac(v20,-v8,v36); v13=mul(-v4,v20);
    v18=mac(-2.0,v6,v36); v35=mac(v20,-v7,v35); v36=mac(2.0,v5,v35); v35=mul(v18,v18); v18=mul(v18,v8);
    v35=mac(v36,v36,v35); v36=mac(v36,v7,v18); v42=mul(v39,v8); v15=b36?v15:0.0; v14=b36?v14:0.0;
    v14=B(0x3f3504f3u)+v14; v18=b36?v2:1.0; v18=mul(s38,v18); v2=mac(s42,v1,v17); v1=mac(s24,v33,v9);
    v1=mac(s25,v31,v1); v1=mac(s26,v22,v1); v1=1.0/(v1); v0=mul(v1,v0); v1=mul(v1,v37);
    v0=mac(0.5,v0,0.5); v1=mac(-0.5,v1,0.5);
    // MIMG 0x3e8: image_sample, dmask 0x1.
    v41=sampleAt(3,vec2(v0,v1)).x; v0=B(0xbf3504f3u)+v23; v1=mac(v26,v0,-v27); v38=mac(v0,-v24,v38);
    v0=mul(v1,v1); v0=mac(v30,v30,v0); v0=mac(v38,v38,v0); v32=inversesqrt(v0); v37=mul(v1,v32);
    v0=mac(v37,v19,v12); v1=mul(v30,v32); v17=mul(v38,v32); v40=mac(s24,v0,v9); v30=mac(v17,v19,v10);
    v33=mac(-v7,v20,v31); v31=mul(s12,v0); v32=mac(-v37,v19,v12); v37=mac(v1,v19,v11);
    v43=mac(s13,v37,v31); v31=mul(s16,v0); v40=mac(s25,v37,v40); v31=mac(s17,v37,v31);
    v31=mac(s18,v30,v31); v40=mac(s26,v30,v40); v43=mac(s14,v30,v43); v45=s19+v31; v33=mac(2.0,v5,v33);
    v17=mac(-v17,v19,v10); v31=mac(-v7,v20,v37); v37=1.0/(v40); v40=mac(-v8,v20,v0);
    v31=mac(2.0,v5,v31); v38=mul(v39,v39); v39=mac(-2.0,v6,v40); v40=mul(s16,v32);
    v38=mac(v33,v33,v38); v42=mac(v7,v33,v42); v33=mac(-v1,v19,v11); v1=s15+v43; v1=mul(v37,v1);
    v37=mul(v37,v45); v0=mac(0.5,v1,0.5); v1=mac(-0.5,v37,0.5);
    // MIMG 0x4dc: image_sample, dmask 0x1.
    v37=sampleAt(4,vec2(v0,v1)).x; v1=mul(v39,v8); v0=mac(s24,v32,v9); v43=mac(s25,v33,v0);
    v40=mac(s17,v33,v40); v40=mac(s18,v17,v40); v43=mac(s26,v17,v43); v0=s19+v40; v43=1.0/(v43);
    v0=mul(v43,v0); v39=mul(v39,v39); v40=mac(v31,v7,v1); v1=mul(s12,v32); v1=mac(s13,v33,v1);
    v1=mac(s14,v17,v1); v1=s15+v1; v32=mac(v20,-v8,v32); v32=mac(-2.0,v6,v32); v33=mac(v20,-v7,v33);
    v39=mac(v31,v31,v39); v31=mul(v43,v1); v1=mac(-0.5,v0,0.5); v0=mac(0.5,v31,0.5);
    // MIMG 0x558: image_sample, dmask 0x1.
    v31=sampleAt(5,vec2(v0,v1)).x; v0=1.0+v23; v43=mul(v32,v8); v1=mul(v29,v21); v1=mac(v0,-v24,v1);
    v0=mac(v26,v0,-v27); v32=mul(v32,v32); v33=mac(2.0,v5,v33); v32=mac(v33,v33,v32);
    v33=mac(v33,v7,v43); v43=mac(-v29,v26,v28); v29=1.0+v29; v21=mul(v21,v29); v27=mac(v23,v26,-v27);
    v26=mac(-v26,v29,v28); v29=s22+v34; v29=1.0/(v29); v25=mac(s23,v29,v25); v29=mac(-s23,v29,v13);
    v25=clamp(mac(-s50,abs(v25),1.0),0.0,1.0); v29=mac(s23,v3,v29); v25=log2(v25);
    v35=mac(v29,v29,v35); v29=mac(v29,v4,v36); v23=mac(-v23,v24,v21); v21=mul(v0,v0);
    v21=mac(v43,v43,v21); v21=mac(v1,v1,v21); v24=inversesqrt(v21); v21=mul(v0,v24);
    v0=mac(v21,v19,v12); v1=mul(v1,v24); v24=mul(v43,v24); v28=mul(s16,v0); v34=s22+v41;
    v41=inversesqrt(v35); v34=1.0/(v34); v43=mac(-s23,v34,v13); v43=mac(s23,v3,v43);
    v38=mac(v43,v43,v38); v36=inversesqrt(v38); v35=mac(s23,v34,v22); v42=mac(v4,v43,v42);
    v45=b46?0.0:v7; v22=clamp(mul(v41,v29),0.0,1.0); v29=mul(s51,v25); v29=exp2(v29);
    v25=mac(v24,v19,v11); v22=mul(v29,v22); v29=mul(s12,v0); v41=mac(s13,v25,v29);
    v29=mac(s17,v25,v28); v28=clamp(mac(-s50,abs(v35),1.0),0.0,1.0); v28=log2(v28); v28=mul(s51,v28);
    v28=exp2(v28); v35=clamp(mul(v36,v42),0.0,1.0); v36=mac(-v21,v19,v12); v42=mac(-v24,v19,v11);
    v24=mac(v1,v19,v10); v41=mac(s14,v24,v41); v34=s15+v41; v29=mac(s18,v24,v29); v29=s19+v29;
    v21=mac(-v1,v19,v10); v1=mac(s24,v0,v9); v1=mac(s25,v25,v1); v1=mac(s26,v24,v1); v1=1.0/(v1);
    v25=mac(v20,-v7,v25); v25=mac(2.0,v5,v25); v22=mac(v35,v28,v22); v35=mac(-v8,v20,v0);
    v0=mul(v1,v34); v0=mac(0.5,v0,0.5); v1=mul(v1,v29); v1=mac(-0.5,v1,0.5);
    // MIMG 0x70c: image_sample, dmask 0x1.
    v29=sampleAt(6,vec2(v0,v1)).x; v34=s22+v37; v37=1.0/(v34); v34=mac(-s23,v37,v13);
    v37=mac(s23,v37,v30); v0=mac(s24,v36,v9); v34=mac(s23,v3,v34); v0=mac(s25,v42,v0);
    v38=mac(v34,v34,v39); v0=mac(s26,v21,v0); v39=mac(v34,v4,v40); v1=1.0/(v0); v0=mul(s12,v36);
    v0=mac(s13,v42,v0); v0=mac(s14,v21,v0); v0=s15+v0; v0=mul(v1,v0); v0=mac(0.5,v0,0.5);
    v35=mac(-2.0,v6,v35); v34=mul(v35,v35); v34=mac(v25,v25,v34); v41=mul(v35,v8);
    v30=mac(-v8,v20,v36); v47=s22+v31; v47=1.0/(v47); v36=mul(s16,v36); v17=mac(s23,v47,v17);
    v47=mac(-s23,v47,v13); v17=clamp(mac(-s50,abs(v17),1.0),0.0,1.0); v40=mac(s17,v42,v36);
    v36=clamp(mac(-s50,abs(v37),1.0),0.0,1.0); v36=log2(v36); v47=mac(s23,v3,v47);
    v37=inversesqrt(v38); v36=mul(s51,v36); v17=log2(v17); v32=mac(v47,v47,v32); v40=mac(s18,v21,v40);
    v37=clamp(mul(v37,v39),0.0,1.0); v36=exp2(v36); v32=inversesqrt(v32); v17=mul(s51,v17);
    v28=s19+v40; v33=mac(v4,v47,v33); v36=mul(v36,v37); v32=clamp(mul(v32,v33),0.0,1.0); v17=exp2(v17);
    v1=mul(v1,v28); v33=mul(v27,v27); v1=mac(-0.5,v1,0.5);
    // MIMG 0x818: image_sample, dmask 0x1.
    v28=sampleAt(7,vec2(v0,v1)).x; v33=mac(v26,v26,v33); v36=mac(v32,v17,v36); v1=mac(-0.5,v36,1.0);
    v33=mac(v23,v23,v33); v1=log2(abs(v1)); v33=inversesqrt(v33); v1=mul(s32,v1); v27=mul(v27,v33);
    v1=exp2(v1); v38=mac(v27,v19,v12); v0=mul(v23,v33); v23=mul(v26,v33); v26=mul(s16,v38);
    v37=mac(-v27,v19,v12); v27=mul(s16,v37); v36=mac(-v23,v19,v11); v39=mac(s24,v37,v9);
    v27=mac(s17,v36,v27); v39=mac(s25,v36,v39); v47=b36?v16:0.0; v32=b46?1.0:v4; v40=mul(v45,v14);
    v42=mac(v20,-v7,v42); v42=mac(2.0,v5,v42); v41=mac(v7,v25,v41); v25=mul(s12,v38);
    v30=mac(-2.0,v6,v30); v31=mul(v30,v8); v31=mac(v7,v42,v31); v30=mul(v30,v30); v30=mac(v42,v42,v30);
    v22=mac(-0.5,v22,1.0); v22=log2(abs(v22)); v22=mul(s32,v22); v22=exp2(v22); v22=v22+v1;
    v1=mac(v23,v19,v11); v23=mac(v0,v19,v10); v35=mac(s13,v1,v25); v35=mac(s14,v23,v35);
    v25=mac(s17,v1,v26); v25=mac(s18,v23,v25); v25=s19+v25; v26=mul(s12,v37); v19=mac(-v0,v19,v10);
    v0=s15+v35; v35=mac(s24,v38,v9); v35=mac(s25,v1,v35); v35=mac(s26,v23,v35); v26=mac(s13,v36,v26);
    v27=mac(s18,v19,v27); v39=mac(s26,v19,v39); v26=mac(s14,v19,v26); v27=s19+v27;
    v38=mac(v20,-v8,v38); v38=mac(-2.0,v6,v38); v33=mac(-v7,v20,v1); v1=1.0/(v35); v25=mul(v1,v25);
    v0=mul(v1,v0); v1=mac(-0.5,v25,0.5); v0=mac(0.5,v0,0.5);
    // MIMG 0x96c: image_sample, dmask 0x1.
    v25=sampleAt(8,vec2(v0,v1)).x; v1=1.0/(v39); v0=s15+v26; v0=mul(v1,v0); v1=mul(v1,v27);
    v1=mac(-0.5,v1,0.5); v0=mac(0.5,v0,0.5);
    // MIMG 0x994: image_sample, dmask 0x1.
    v27=sampleAt(9,vec2(v0,v1)).x; v1=b46?0.0:v8; v0=s22+v29; v0=1.0/(v0); v35=B(0xbf3504f3u)+v47;
    v24=mac(s23,v0,v24); v0=mac(-s23,v0,v13); v0=mac(s23,v3,v0); v34=mac(v0,v0,v34);
    v16=inversesqrt(v34); v41=mac(v4,v0,v41); v0=clamp(mul(v16,v41),0.0,1.0);
    v24=clamp(mac(-s50,abs(v24),1.0),0.0,1.0); v24=log2(v24); v24=mul(s51,v24); v26=mul(v38,v38);
    v33=mac(2.0,v5,v33); v16=exp2(v24); v26=mac(v33,v33,v26); v43=mul(v38,v8); v38=mac(-v8,v20,v37);
    v38=mac(-2.0,v6,v38); v37=mac(-v7,v20,v36); v36=mul(v38,v8); v20=mul(v38,v38); v37=mac(2.0,v5,v37);
    v29=mac(v37,v37,v20); v20=mul(v15,v1); v15=mul(v15,v45); v20=mac(v14,-v32,v20); v14=mul(s39,-v10);
    v28=s22+v28; v42=max(v18,v14); v18=mac(v32,v35,-v15); v43=mac(v7,v33,v43); v33=mul(v16,v0);
    v24=mac(v37,v7,v36); v36=B(0x3f3504f3u)+v47; v39=mac(v32,v36,-v15); v15=mul(v39,v39);
    v14=mac(-v1,v36,v40); v32=1.0/(v28); v21=mac(s23,v32,v21);
    v21=clamp(mac(-s50,abs(v21),1.0),0.0,1.0); v36=log2(v21); v36=mul(s51,v36); v36=exp2(v36);
    v47=mac(-v1,v35,v40); v1=mac(v20,v20,v15); v1=mac(v14,v14,v1); v35=inversesqrt(v1);
    v34=mul(v39,v35); v21=mac(v34,v42,v12); v15=mul(v20,v35); v38=mac(v15,v42,v11); v28=mul(v14,v35);
    v35=mac(s24,v21,v9); v37=mul(s12,v21); v14=mac(v28,v42,v10); v35=mac(s25,v38,v35);
    v37=mac(s13,v38,v37); v1=mac(-s23,v32,v13); v40=mac(s14,v14,v37); v35=mac(s26,v14,v35);
    v39=1.0/(v35); v0=s15+v40; v1=mac(s23,v3,v1); v0=mul(v39,v0); v31=mac(v4,v1,v31);
    v0=mac(0.5,v0,0.5); v30=mac(v1,v1,v30); v30=inversesqrt(v30); v30=clamp(mul(v30,v31),0.0,1.0);
    v31=mul(s16,v21); v31=mac(s17,v38,v31); v31=mac(s18,v14,v31); v46=s19+v31; v1=mul(v39,v46);
    v1=mac(-0.5,v1,0.5); v33=mac(v30,v36,v33); v30=mac(-0.5,v33,1.0); v30=log2(abs(v30));
    v30=mul(s32,v30); v36=mac(-v15,v42,v11); v31=mul(v18,v18); v15=mac(-v28,v42,v10);
    v31=mac(v20,v20,v31); v21=mac(v2,-v8,v21); v31=mac(v47,v47,v31); v44=inversesqrt(v31);
    v41=exp2(v30); v30=mac(-v34,v42,v12); v28=mul(s16,v30); v28=mac(s17,v36,v28); v28=mac(s18,v15,v28);
    v37=s19+v28; v33=mul(v18,v44); v22=v22+v41; v18=mac(v33,v42,v12); v31=mul(v20,v44);
    v47=mul(v47,v44); v12=mac(v42,-v33,v12); v33=mul(s12,v18); v32=mul(s16,v18); v38=mac(v2,-v7,v38);
    v17=mul(-v4,v2); v34=mac(2.0,v5,v38); v38=mac(-2.0,v6,v21);
    // MIMG 0xbec: image_sample, dmask 0x1.
    v21=sampleAt(10,vec2(v0,v1)).x; v0=s22+v25; v0=1.0/(v0); v1=s22+v27; v23=mac(s23,v0,v23);
    v0=mac(-s23,v0,v13); v0=mac(s23,v3,v0); v26=mac(v0,v0,v26); v20=inversesqrt(v26); v25=mul(s12,v30);
    v25=mac(s13,v36,v25); v25=mac(s14,v15,v25); v43=mac(v4,v0,v43); v0=clamp(mul(v20,v43),0.0,1.0);
    v20=1.0/(v1); v1=clamp(mac(-s50,abs(v23),1.0),0.0,1.0); v23=mac(s23,v20,v19);
    v23=clamp(mac(-s50,abs(v23),1.0),0.0,1.0); v27=mac(-s23,v20,v13); v20=log2(v1); v26=mul(s51,v20);
    v19=log2(v23); v23=exp2(v26); v19=mul(s51,v19); v27=mac(s23,v3,v27); v19=exp2(v19); v20=s15+v25;
    v29=mac(v27,v27,v29); v13=inversesqrt(v29); v24=mac(v4,v27,v24); v26=mul(v23,v0);
    v23=mac(s24,v30,v9); v23=mac(s25,v36,v23); v23=mac(s26,v15,v23); v23=1.0/(v23); v1=mul(v23,v37);
    v1=mac(-0.5,v1,0.5); v20=mul(v23,v20); v13=clamp(mul(v13,v24),0.0,1.0); v0=mac(0.5,v20,0.5);
    v27=mac(-v31,v42,v11); v26=mac(v13,v19,v26); v19=mac(v47,v42,v10); v13=mac(-v47,v42,v10);
    v30=mac(v2,-v8,v30); v24=mul(v38,v38); v35=mac(-v7,v2,v36); v30=mac(-2.0,v6,v30); v20=mul(v30,v30);
    v24=mac(v34,v34,v24); v35=mac(2.0,v5,v35); v23=mul(v30,v8); v20=mac(v35,v35,v20);
    v23=mac(v7,v35,v23); v25=mul(v38,v8); v25=mac(v7,v34,v25); v30=mac(-0.5,v26,1.0);
    // MIMG 0xd1c: image_sample, dmask 0x1.
    v26=sampleAt(11,vec2(v0,v1)).x; v1=mac(v31,v42,v11); v33=mac(s13,v1,v33); v33=mac(s14,v19,v33);
    v0=s15+v33; v32=mac(s17,v1,v32); v32=mac(s18,v19,v32); v29=s19+v32; v16=mac(-v7,v2,v1);
    v16=mac(2.0,v5,v16); v11=log2(abs(v30)); v11=mul(s32,v11); v11=exp2(v11); v11=v22+v11;
    v22=mac(s24,v18,v9); v22=mac(s25,v1,v22); v22=mac(s26,v19,v22); v1=1.0/(v22); v9=mac(s24,v12,v9);
    v0=mul(v1,v0); v22=mac(s25,v27,v9); v0=mac(0.5,v0,0.5); v22=mac(s26,v13,v22); v1=mul(v1,v29);
    v1=mac(-0.5,v1,0.5);
    // MIMG 0xda0: image_sample, dmask 0x1.
    v9=sampleAt(12,vec2(v0,v1)).x; v0=mul(s16,v12); v0=mac(s17,v27,v0); v0=mac(s18,v13,v0); v0=s19+v0;
    v1=1.0/(v22); v22=mul(s12,v12); v22=mac(s13,v27,v22); v22=mac(s14,v13,v22); v22=s15+v22;
    v0=mul(v1,v0); v22=mul(v1,v22); v1=mac(-0.5,v0,0.5); v0=mac(0.5,v22,0.5);
    // MIMG 0xde4: image_sample, dmask 0x1.
    v0=sampleAt(13,vec2(v0,v1)).x; v12=mac(v2,-v8,v12); v12=mac(-2.0,v6,v12); v18=mac(v2,-v8,v18);
    v18=mac(-2.0,v6,v18); v27=mac(v2,-v7,v27); v6=mul(v18,v8); v1=mul(v12,v12); v27=mac(2.0,v5,v27);
    v5=mul(v18,v18); v5=mac(v16,v16,v5); v2=mul(v12,v8); v6=mac(v7,v16,v6); v16=mac(v27,v27,v1);
    v2=mac(v7,v27,v2); v7=s22+v21; v7=1.0/(v7); v14=mac(s23,v7,v14);
    v8=clamp(mac(-s50,abs(v14),1.0),0.0,1.0); v8=log2(v8); v8=mul(s51,v8); v8=exp2(v8);
    v7=mac(-s23,v7,v17); v7=mac(s23,v3,v7); v24=mac(v7,v7,v24); v12=inversesqrt(v24);
    v25=mac(v4,v7,v25); v7=clamp(mul(v12,v25),0.0,1.0); v7=mul(v8,v7);
    v45=clamp(mac(s48,v10,1.0),0.0,1.0); v8=s22+v26; v8=1.0/(v8); v15=mac(s23,v8,v15);
    v8=mac(-s23,v8,v17); v8=mac(s23,v3,v8); v20=mac(v8,v8,v20); v14=inversesqrt(v20);
    v12=clamp(mac(-s50,abs(v15),1.0),0.0,1.0); v23=mac(v4,v8,v23); v8=clamp(mul(v14,v23),0.0,1.0);
    v12=log2(v12); v12=mul(s51,v12); v12=exp2(v12); v7=mac(v8,v12,v7); v7=mac(-0.5,v7,1.0);
    v7=log2(abs(v7)); v7=mul(s40,v7); v7=exp2(v7); v8=s22+v9; v8=1.0/(v8); v12=mac(-s23,v8,v17);
    v19=mac(s23,v8,v19); v9=clamp(mac(-s50,abs(v19),1.0),0.0,1.0); v12=mac(s23,v3,v12); v9=log2(v9);
    v5=mac(v12,v12,v5); v5=inversesqrt(v5); v9=mul(s51,v9); v6=mac(v4,v12,v6);
    v5=clamp(mul(v5,v6),0.0,1.0); v8=exp2(v9); v5=mul(v8,v5); v0=s22+v0; v0=1.0/(v0);
    v17=mac(-s23,v0,v17); v13=mac(s23,v0,v13); v17=mac(s23,v3,v17);
    v3=clamp(mac(-s50,abs(v13),1.0),0.0,1.0); v1=log2(v3); v16=mac(v17,v17,v16); v3=inversesqrt(v16);
    v1=mul(s51,v1); v2=mac(v4,v17,v2); v0=clamp(mul(v3,v2),0.0,1.0); v1=exp2(v1); v5=mac(v0,v1,v5);
    v0=mac(-0.5,v5,1.0); v0=log2(abs(v0)); v0=mul(s40,v0); v0=exp2(v0); v0=v7+v0; v11=mac(2.0,v0,v11);
    v0=mul(B(0x3e000000u),v11); put(vec4(v45,v0,v45,v0));
#elif ROLE == 7
    float v0,v1,v2,v3,v4,v5,v6,v7;
    float v8,v9,v10,v11;
    float s0,s1,s2,s3,s4,s5,s6,s7;
    float s8,s9,s10,s11,s12,s24,s25,s26,s27;
    float s28,s29,s30,s31,s40,s41,s42,s43;
    v2=a1.x; v3=a1.y;
    // MIMG 0x34: image_sample, dmask 0x1.
    v5=sampleAt(0,vec2(v2,v3)).x;
    // Scalar load 0x40.
    s0=c[4].x; s1=c[4].y; s2=c[4].z; s3=c[4].w; s4=c[5].x; s5=c[5].y; s6=c[5].z; s7=c[5].w;
    // Scalar load 0x50.
    s24=c[6].x; s25=c[6].y; s26=c[6].z; s27=c[6].w; s28=c[7].x; s29=c[7].y; s30=c[7].z; s31=c[7].w;
    v4=a0.x; v2=a0.y; v3=a0.z;
    // Scalar load 0x68.
    s40=c[8].x; s41=c[8].y; s42=c[8].z; s43=c[8].w; v0=s0+v5; v0=1.0/(v0); v5=mul(s1,v0);
    v0=mul(v4,v5); v1=mul(v2,v5); v2=mul(v3,v5); v3=mul(s24,v0); v5=mac(s25,v1,v3); v5=mac(s26,v2,v5);
    // Scalar load 0x9c.
    s0=c[9].x; s1=c[9].y; s2=c[9].z; s3=c[9].w; v4=mul(s28,v0); v4=mac(s29,v1,v4); v4=mac(s30,v2,v4);
    v3=s31+v4; v4=s27+v5;
    // Scalar load 0xb4.
    s24=c[10].x; s25=c[10].y; s26=c[10].z; s27=c[10].w; s28=c[11].x; s29=c[11].y; s30=c[11].z;
    s31=c[11].w; v6=s27; v5=mac(s24,v0,v6); v5=mac(s25,v1,v5); v5=mac(s26,v2,v5); v5=1.0/(v5);
    v3=mul(v5,v3); v4=mul(v5,v4); v5=mul(s28,v0); v5=mac(s29,v1,v5); v5=mac(s30,v2,v5); v6=s31+v5;
    // Scalar load 0xec.
    s24=c[12].x; s25=c[12].y; s26=c[12].z; s27=c[12].w; s28=c[13].x; s29=c[13].y; s30=c[13].z;
    s31=c[13].w; v5=mul(s0,v0); v7=s27; v7=mac(s24,v0,v7); v7=mac(s25,v1,v7); v7=mac(s26,v2,v7);
    v7=1.0/(v7); v5=mac(s1,v1,v5); v5=mac(s2,v2,v5); v5=s3+v5;
    // Scalar load 0x118.
    s0=c[14].x; s1=c[14].y; s2=c[14].z; s3=c[14].w; v3=mac(-0.5,v3,0.5); v8=mul(s0,v0);
    v8=mac(s1,v1,v8); v8=mac(s2,v2,v8); v5=mul(v7,v5); v6=mul(v7,v6); v7=mul(s28,v0);
    v7=mac(s29,v1,v7); v7=mac(s30,v2,v7); v7=s31+v7;
    // Scalar load 0x14c.
    s24=c[15].x; s25=c[15].y; s26=c[15].z; s27=c[15].w; s28=c[16].x; s29=c[16].y; s30=c[16].z;
    s31=c[16].w; v8=s3+v8;
    // Scalar load 0x154.
    s0=c[17].x; s1=c[17].y; s2=c[17].z; s3=c[17].w;
    // Scalar load 0x158.
    s8=c[18].x; s9=c[18].y; s10=c[18].z; s11=c[18].w; v9=s27; v10=s11; v10=mac(s8,v0,v10);
    v9=mac(s24,v0,v9); v10=mac(s9,v1,v10); v9=mac(s25,v1,v9); v10=mac(s10,v2,v10); v9=mac(s26,v2,v9);
    v9=1.0/(v9); v8=mul(v9,v8); v9=mul(v9,v7); v7=mul(s28,v0); v0=mul(s0,v0); v0=mac(s1,v1,v0);
    v7=mac(s29,v1,v7); v0=mac(s2,v2,v0); v7=mac(s30,v2,v7); v2=1.0/(v10); v1=s3+v0; v7=s31+v7;
    v0=mac(0.5,v6,0.5); v10=mul(v2,v1); v1=mac(-0.5,v5,0.5); v11=mul(v2,v7); v2=mac(0.5,v4,0.5);
    // MIMG 0x1d0: image_sample, dmask 0xc.
    { vec4 t=sampleAt(1,vec2(v2,v3));
      v6=t.z;   v7=t.w;
    }
    // MIMG 0x1d8: image_sample, dmask 0x3.
    { vec4 t=sampleAt(2,vec2(v0,v1));
      v4=t.x;   v5=t.y;
    }
    v1=mac(-0.5,v8,0.5); v0=mac(0.5,v9,0.5); v3=mac(-0.5,v10,0.5); v2=mac(0.5,v11,0.5);
    // MIMG 0x208: image_sample, dmask 0xc.
    { vec4 t=sampleAt(3,vec2(v0,v1));
      v0=t.z;   v1=t.w;
    }
    // MIMG 0x210: image_sample, dmask 0x3.
    { vec4 t=sampleAt(4,vec2(v2,v3));
      v2=t.x;   v3=t.y;
    }
    s12=B(0x42800000u); v8=mul(s4,v6); v9=mul(s4,v7); v8=mac(s5,v4,v8); v9=mac(s5,v5,v9);
    v8=mac(s6,v0,v8); v11=mac(s6,v1,v9); v8=mac(s7,v2,v8); v10=v8-v6; v11=mac(s7,v3,v11);
    v9=clamp(mul(s12,abs(v10)),0.0,1.0); v10=v11-v7; v4=v8-v4; v4=clamp(mul(s12,abs(v4)),0.0,1.0);
    v7=mac(v10,v9,v7); v9=v11-v5; v10=v8-v0; v0=mul(s40,v7); v4=mac(v4,v9,v5);
    v5=clamp(mul(s12,abs(v10)),0.0,1.0); v7=v11-v1; v2=v8-v2; v2=clamp(mul(s12,abs(v2)),0.0,1.0);
    v0=mac(s41,v4,v0); v4=v11-v3; v1=mac(v7,v5,v1); v0=mac(s42,v1,v0); v3=mac(v4,v2,v3);
    v0=mac(s43,v3,v0); put(vec4(v0));
#elif ROLE == 8
    float v0,v1,v2,v3,v4,v5,v6,v7;
    float v8,v9,v10,v11,v12,v13,v14,v15;
    float v16,v17,v18,v19,v20,v21;
    float s20,s21,s22,s23,s24,s25,s26,s27;
    float s32,s33,s34;
    v2=gl_FragCoord.x; v3=gl_FragCoord.y;
    s34=B(0x3e218d1du);
    // Scalar load 0x24.
    s32=c[4].x; s33=c[4].y;
    // Scalar load 0x28.
    s20=c[5].x; s21=c[5].y; s22=c[5].z; s23=c[5].w; s24=c[6].x; s25=c[6].y; s26=c[6].z; s27=c[6].w;
    v11=B(0xbe800000u)+v2; v13=B(0xbe800000u)+v3; v4=mul(s32,v11); v5=mul(s33,v13);
    v0=mac(-4.0,s22,v4); v1=mac(-4.0,s23,v5); v6=mac(-2.0,s22,v4); v2=s23; v10=s22;
    v8=mac(v10,B(0xc0400000u),v4); v9=mac(v2,B(0xc0400000u),v5); v7=mac(-2.0,s23,v5);
    // MIMG 0x88: image_sample, dmask 0x1.
    v21=sampleAt(0,vec2(v0,v1)).x;
    // MIMG 0x90: image_sample, dmask 0x1.
    v20=sampleAt(1,vec2(v4,v5)).x;
    // MIMG 0x98: image_sample, dmask 0x1.
    v18=sampleAt(2,vec2(v8,v9)).x;
    // MIMG 0xa0: image_sample, dmask 0x1.
    v15=sampleAt(3,vec2(v6,v7)).x; v2=s23;
    // MIMG 0xac: image_sample, dmask 0x8.
    v12=sampleAt(4,vec2(v6,v7)).w;
    // MIMG 0xb4: image_sample, dmask 0x8.
    v16=sampleAt(5,vec2(v0,v1)).w;
    // MIMG 0xbc: image_sample, dmask 0xa.
    { vec4 t=sampleAt(6,vec2(v4,v5));
      v0=t.y;   v1=t.w;
    }
    // MIMG 0xc4: image_sample, dmask 0x8.
    v14=sampleAt(7,vec2(v8,v9)).w; v6=s22; v6=mac(s32,v11,-v6); v7=mac(s33,v13,-v2);
    v8=mac(s32,v11,v10); v9=mac(s33,v13,v2); v3=mac(2.0,s23,v5); v10=mac(v10,B(0x40400000u),v4);
    v11=mac(v2,B(0x40400000u),v5); v2=mac(2.0,s22,v4);
    // MIMG 0x110: image_sample, dmask 0x1.
    v19=sampleAt(8,vec2(v6,v7)).x;
    // MIMG 0x118: image_sample, dmask 0x1.
    v13=sampleAt(9,vec2(v8,v9)).x;
    // MIMG 0x120: image_sample, dmask 0x1.
    v17=sampleAt(10,vec2(v2,v3)).x;
    // MIMG 0x128: image_sample, dmask 0x1.
    v5=sampleAt(11,vec2(v10,v11)).x;
    // MIMG 0x130: image_sample, dmask 0x8.
    v6=sampleAt(12,vec2(v6,v7)).w;
    // MIMG 0x138: image_sample, dmask 0x8.
    v4=sampleAt(13,vec2(v8,v9)).w;
    // MIMG 0x140: image_sample, dmask 0x8.
    v3=sampleAt(14,vec2(v2,v3)).w;
    // MIMG 0x148: image_sample, dmask 0x8.
    v2=sampleAt(15,vec2(v10,v11)).w; v7=s20+v21; v8=1.0/(v7); v7=s20+v20; v7=1.0/(v7); v7=mul(s21,v7);
    v8=mac(s21,v8,-v7); v8=clamp(mac(-s24,abs(v8),1.0),0.0,1.0); v8=log2(v8); v8=mul(s25,v8);
    v10=exp2(v8); v8=s20+v18; v8=1.0/(v8); v8=mac(s21,v8,-v7); v11=s20+v19; v11=1.0/(v11);
    v8=clamp(mac(-s24,abs(v8),1.0),0.0,1.0); v11=mac(s21,v11,-v7); v8=log2(v8);
    v11=clamp(mac(-s24,abs(v11),1.0),0.0,1.0); v8=mul(s25,v8); v11=log2(v11); v8=exp2(v8);
    v18=mul(s25,v11); v5=s20+v5; v9=mul(B(0x3e218d1du),v10); v10=mac(s34,v10,1.0);
    v10=mac(v8,B(0x3eb53268u),v10); v5=1.0/(v5); v9=mac(v16,v9,v0); v0=s20+v15; v0=1.0/(v0);
    v0=mac(s21,v0,-v7); v0=clamp(mac(-s24,abs(v0),1.0),0.0,1.0); v0=log2(v0); v0=mul(s25,v0);
    v11=mul(B(0x3eb53268u),v8); v8=exp2(v18); v15=exp2(v0); v16=mac(v14,v11,v9);
    v11=mul(B(0x3f215716u),v15); v0=mac(v15,B(0x3f215716u),v10); v10=s20+v17; v10=1.0/(v10);
    v10=mac(s21,v10,-v7); v10=clamp(mac(-s24,abs(v10),1.0),0.0,1.0); v14=log2(v10); v9=mul(s25,v14);
    v0=mac(v8,B(0x3f64184du),v0); v16=mac(v11,v12,v16); v12=s20+v13; v11=mul(B(0x3f64184du),v8);
    v8=1.0/(v12); v8=mac(s21,v8,-v7); v8=clamp(mac(-s24,abs(v8),1.0),0.0,1.0); v8=log2(v8);
    v7=mac(s21,v5,-v7); v8=mul(s25,v8); v5=clamp(mac(-s24,abs(v7),1.0),0.0,1.0); v8=exp2(v8);
    v7=log2(v5); v5=exp2(v9); v7=mul(s25,v7); v9=mac(v8,B(0x3f64184du),v0); v16=mac(v11,v6,v16);
    v1=v16+v1; v6=mul(B(0x3f64184du),v8); v0=exp2(v7); v7=mac(v5,B(0x3f215716u),v9);
    v5=mul(B(0x3f215716u),v5); v1=mac(v6,v4,v1); v4=mac(v0,B(0x3eb53268u),v7); v5=mac(v3,v5,v1);
    v0=mul(B(0x3eb53268u),v0); v3=1.0+v4; v1=1.0/(v3); v5=mac(v0,v2,v5); v0=mul(v1,v5); put(vec4(v0));
#elif ROLE == 9
    float v0,v1,v2,v3,v4,v5,v6,v7;
    float v8,v9,v10,v11,v12,v13,v14,v15;
    float v16;
    float s12,s13,s14,s15,s32,s33;
    bool cc;
    v2=gl_FragCoord.x; v3=gl_FragCoord.y;
    // Scalar load 0x2c.
    s32=c[4].x; s33=c[4].y; v0=B(0xbe800000u)+v2; v1=floor(v0);
    v6=B(0xbe800000u)+v3; v5=intBitsToFloat(int(v1));
    v1=uintBitsToFloat(0xffffffffu+floatBitsToUint(v5)); v4=uintBitsToFloat(1u&floatBitsToUint(v1));
    v5=uintBitsToFloat(1u&floatBitsToUint(v5));
    v4=uintBitsToFloat(floatBitsToUint(v1)-floatBitsToUint(v4)); v9=mul(0.5,s32); v0=mul(0.5,s33);
    v2=-0.5+v2; v1=-0.5+v3; v3=floor(v6); v3=intBitsToFloat(int(v3));
    v6=uintBitsToFloat(0xffffffffu+floatBitsToUint(v3)); v3=uintBitsToFloat(1u&floatBitsToUint(v3));
    v3=uintBitsToFloat((floatBitsToUint(v3)&0xffffffu)*2u+floatBitsToUint(v5));
    v5=uintBitsToFloat(1u&floatBitsToUint(v6));
    v7=uintBitsToFloat(floatBitsToUint(v6)-floatBitsToUint(v5));
    v5=uintBitsToFloat(2u+floatBitsToUint(v4)); v6=float(floatBitsToInt(v5));
    v5=uintBitsToFloat(2u+floatBitsToUint(v7)); v7=float(floatBitsToInt(v7)); v7=mac(v7,s33,v0);
    v8=float(floatBitsToInt(v4)); v4=uintBitsToFloat(0u-floatBitsToUint(v3)); v6=mac(v6,s32,v9);
    v3=float(floatBitsToInt(v5)); v5=mac(s33,v1,v0); v1=mac(v3,s33,v0); v3=v1;
    v13=uintBitsToFloat(floatBitsToUint(v4)<<4u); v4=mac(s32,v2,v9);
    // MIMG 0xf4: image_sample, dmask 0x1.
    v11=sampleAt(0,vec2(v4,v5)).x;
    // MIMG 0xfc: image_sample, dmask 0x1.
    v14=sampleAt(1,vec2(v6,v7)).x; v4=mac(v8,s32,v9); v5=v7; v0=v4; v2=v6;
    // MIMG 0x118: image_sample, dmask 0x1.
    v12=sampleAt(2,vec2(v4,v5)).x;
    // MIMG 0x120: image_sample, dmask 0x1.
    v10=sampleAt(3,vec2(v0,v1)).x;
    // MIMG 0x128: image_sample, dmask 0x1.
    v9=sampleAt(4,vec2(v2,v3)).x;
    // MIMG 0x130: image_sample, dmask 0x8.
    v6=sampleAt(5,vec2(v6,v7)).w;
    // MIMG 0x138: image_sample, dmask 0x8.
    v7=sampleAt(6,vec2(v2,v3)).w;
    // MIMG 0x140: image_sample, dmask 0x8.
    v8=sampleAt(7,vec2(v0,v1)).w;
    // MIMG 0x148: image_sample, dmask 0x8.
    v4=sampleAt(8,vec2(v4,v5)).w;
    // 0x150: shader-local float4 table at PC 0x334, byte offset 48 - 16*parity.
    { vec4 t=weights[int((floatBitsToUint(v13)+48u)>>4u)];
      v0=t.x; v1=t.y; v2=t.z; v3=t.w; }
    // Scalar load 0x158.
    s12=c[5].x; s13=c[5].y; s14=c[5].z; s15=c[5].w; v5=s12+v11; v5=1.0/(v5); v11=mul(-s13,v5);
    v5=s12+v14; v5=1.0/(v5); v12=s12+v12; v12=1.0/(v12); v10=s12+v10; v5=mac(s13,v5,v11);
    v14=1.0/(v10); v13=s12+v9; v9=mac(s13,v12,v11); v12=abs(v9)+abs(v5); v13=1.0/(v13);
    v10=mac(s13,v13,v11); v11=mac(s13,v14,v11); v12=v12+abs(v11); cc=v12<=-abs(v10); v14=v12+abs(v10);
    v13=abs(v5); v12=abs(v9);
    if (cc) {
        v0=mul(v0,v4); v0=mac(v6,v1,v0); v0=mac(v8,v2,v0); v0=mac(v7,v3,v0);
    } else {
        v15=mul(v0,abs(v9)); v14=1.0/(v14); v16=mul(abs(v5),v1); v0=mac(v14,-v15,v0); v15=mac(-v16,v14,v1);
        v1=mul(abs(v11),v2); v16=mul(v0,v4); v1=mac(-v1,v14,v2); v2=mul(abs(v10),v3); v0=v0+v15; v0=v0+v1;
        v16=mac(v6,v15,v16); v3=mac(v14,-v2,v3); v0=v0+v3; v16=mac(v8,v1,v16); v0=1.0/(v0);
        v16=mac(v7,v3,v16); v0=mul(v0,v16);
    }
    cc=abs(v9)>abs(v5);
    if (cc) {
        v12=v13; v4=v6;
    }
    cc=abs(v11)>abs(v10); v1=min(abs(v11),abs(v10)); v2=cc?v7:v8; cc=v12>v1; v1=cc?v2:v4; v0=v0-v1;
    v1=mac(s14,v0,v1); put(vec4(v1));
#endif
}
)";

constexpr const char* kOutlineVs = R"(
layout(set=0, binding=0, std430) readonly buffer V0 { uint w[]; } v0;
layout(set=0, binding=1, std430) readonly buffer V1 { uint w[]; } v1;
layout(location=0) out vec4 a0;
layout(location=1) out vec4 a1;
layout(location=2) out vec4 a2;
vec4 fetch(int slot) {
    uvec4 f=p.fetch[slot];
    uint index=uint(gl_VertexIndex);
    vec4 r=vec4(0.0,0.0,0.0,1.0);
    uint base=f.x+index*f.y;
    for (uint j=0u;j<f.z;++j)
        r[j]=uintBitsToFloat(slot==0?v0.w[base+j]:v1.w[base+j]);
    vec4 result;
    for (uint j=0u;j<4u;++j) {
        uint s=(f.w>>(3u*j))&7u;
        result[j]=s>=4u?r[s-4u]:(s==1u?1.0:0.0);
    }
    return result;
}
void main() {
    vec4 pos=fetch(0);
    gl_Position=vec4(mac(c[0].xy,pos.xy,c[0].zw),pos.zw);
    if (p.flags.x!=0u) gl_Position.z=mul(gl_Position.z+gl_Position.w,0.5);
    a0=a1=a2=vec4(0.0);
#if ROLE == 10
    a0=vec4(fetch(1).xy,0.0,0.0);
#elif ROLE == 11
    vec2 uv=fetch(1).xy;
    a0=vec4(uv,c[1].xy*uv);
#endif
}
)";

// Capture-5's drawn outline programs. Preserve register reuse and operation order,
// including both branches of hand_draw_2; these are not interchangeable blur kernels.
constexpr const char* kOutlineFs = R"(
layout(location=0) in vec4 a0;
layout(location=0) out vec4 outColour;
layout(set=0,binding=3) uniform sampler2D tex[TEXTURES];
#define sampleAt(i,uv) texture(tex[i],(uv)*c[32+(i)].xy)
float B(uint x) { return uintBitsToFloat(x); }
// As kSsaoFs and the translated pixel shaders: allow contraction, export half RNE.
float omul(float a,float b) { return a*b; }
float omad(float a,float b,float d) { return a*b+d; }
void put(vec4 v) {
    v=vec4(unpackHalf2x16(packHalf2x16(v.xy)),unpackHalf2x16(packHalf2x16(v.zw)));
    outColour=vec4(v[p.swizzle.x],v[p.swizzle.y],v[p.swizzle.z],v[p.swizzle.w]);
}
void main() {
#if ROLE == 10
    float v0,v1,v2,v3,v4,v5,v6,v7;
    float v8,v9,v10,v11,v12,v13,v14,v15;
    float v16,v17,v18,v19,v20,v21,v22,v23;
    float v24,v25,v26,v27,v28,v29,v30,v31;
    float v32,v33,v34,v35;
    float s13=c[4].y,s14=c[4].z,s15=c[4].w;
    bool enabled=s13>0.0,cc;
    v21=a0.x; v22=a0.y;
    // MIMG 0x28.
    { vec4 t=sampleAt(0,vec2(v21,v22));
      v0=t.x; v1=t.y; v2=t.z; v3=t.w; }
    v4=s13; v33=omul(-2.0,s14); v32=omul(-2.0,s15); v31=omad(v4,B(0x3a4d9a68u),v3);
    if (enabled) {
        v19=v21+v33; v20=v22+v32; v15=v21-v33; v16=v22-v32;
        // MIMG 0x74.
        { vec4 t=sampleAt(0,vec2(v19,v20));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x7c.
        { vec4 t=sampleAt(0,vec2(v15,v16));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v17=v15; v18=v20; v11=v19; v12=v16; v13=v19; v24=v5-v2; v14=v3-v0; v23=v4-v1; v25=omul(v14,v14); v14=v8-v1;
        v25=omad(v23,v23,v25); v25=omad(v24,v24,v25); v24=v7-v0; v23=omul(v24,v24); v24=v9-v2; v23=omad(v14,v14,v23);
        v23=omad(v24,v24,v23); cc=v25<v23;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v14=omad(v7,v6,v3); v23=omad(v8,v6,v4); v24=omad(v9,v6,v5); v25=cc?2.0:1.0;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v14=omad(v3,v6,v7); v23=omad(v4,v6,v8); v24=omad(v5,v6,v9); v25=cc?2.0:1.0;
        }
        // MIMG 0x140.
        { vec4 t=sampleAt(0,vec2(v17,v18));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x148.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v17=v5-v2; v11=v3-v0; v12=v4-v1; v18=omul(v11,v11); v11=v8-v1; v18=omad(v12,v12,v18); v18=omad(v17,v17,v18);
        v17=v7-v0; v12=omul(v17,v17); v17=v9-v2; v12=omad(v11,v11,v12); v12=omad(v17,v17,v12); cc=v18<v12;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v10=v25+v6; v3=v3+v14; v25=v23+v4; v26=v24+v5; v23=1.0+v10; v24=omad(v7,v6,v3);
            v25=omad(v6,v8,v25); v26=omad(v6,v9,v26);
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v10=v25+v6; v7=v7+v14; v25=v23+v8; v26=v24+v9; v23=1.0+v10; v24=omad(v3,v6,v7);
            v25=omad(v6,v4,v25); v26=omad(v6,v5,v26);
        }
        v3=s14+v33; v11=v21+v3; v12=v20; v17=v21-v3; v18=v16;
        // MIMG 0x210.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x218.
        { vec4 t=sampleAt(0,vec2(v17,v18));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v18=v20; v12=v16; v20=v5-v2; v14=v3-v0; v16=v4-v1; v27=omul(v14,v14); v14=v8-v1; v27=omad(v16,v16,v27);
        v27=omad(v20,v20,v27); v20=v7-v0; v16=omul(v20,v20); v20=v9-v2; v16=omad(v14,v14,v16); v16=omad(v20,v20,v16);
        cc=v27<v16;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v16=omad(v7,v6,v3); v20=omad(v8,v6,v4); v27=omad(v9,v6,v5); v14=cc?2.0:1.0;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v16=omad(v3,v6,v7); v20=omad(v4,v6,v8); v27=omad(v5,v6,v9); v14=cc?2.0:1.0;
        }
        // MIMG 0x2cc.
        { vec4 t=sampleAt(0,vec2(v17,v18));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x2d4.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v28=v5-v2; v12=v3-v0; v18=v4-v1; v29=omul(v12,v12); v12=v8-v1; v29=omad(v18,v18,v29); v29=omad(v28,v28,v29);
        v28=v7-v0; v18=omul(v28,v28); v28=v9-v2; v18=omad(v12,v12,v18); v18=omad(v28,v28,v18); cc=v29<v18;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v10=v3+v16; v29=v20+v4; v30=v27+v5; v5=v14+v6; v28=1.0+v5; v27=omad(v7,v6,v10);
            v29=omad(v6,v8,v29); v30=omad(v6,v9,v30);
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v10=v7+v16; v29=v20+v8; v30=v27+v9; v27=omad(v3,v6,v10); v3=v14+v6; v28=1.0+v3;
            v29=omad(v6,v4,v29); v30=omad(v6,v5,v30);
        }
        v3=s15+v32; v20=v22+v3; v16=v22-v3;
        // MIMG 0x394.
        { vec4 t=sampleAt(0,vec2(v19,v20));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x39c.
        { vec4 t=sampleAt(0,vec2(v15,v16));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v14=v16; v12=v20; v18=v16; v16=v20; v34=v5-v2; v19=v3-v0; v20=v4-v1; v35=omul(v19,v19); v19=v8-v1;
        v35=omad(v20,v20,v35); v35=omad(v34,v34,v35); v34=v7-v0; v20=omul(v34,v34); v34=v9-v2; v20=omad(v19,v19,v20);
        v20=omad(v34,v34,v20); cc=v35<v20;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v20=omad(v7,v6,v3); v19=omad(v8,v6,v4); v34=omad(v9,v6,v5); v35=cc?2.0:1.0;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v20=omad(v3,v6,v7); v19=omad(v4,v6,v8); v34=omad(v5,v6,v9); v35=cc?2.0:1.0;
        }
        // MIMG 0x458.
        { vec4 t=sampleAt(0,vec2(v15,v16));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x460.
        { vec4 t=sampleAt(0,vec2(v13,v14));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v15=v5-v2; v13=v3-v0; v14=v4-v1; v16=omul(v13,v13); v13=v8-v1; v16=omad(v14,v14,v16); v16=omad(v15,v15,v16);
        v15=v7-v0; v14=omul(v15,v15); v15=v9-v2; v14=omad(v13,v13,v14); v14=omad(v15,v15,v14); cc=v16<v14;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v10=v35+v6; v16=v3+v20; v19=v19+v4; v20=v34+v5; v15=1.0+v10; v16=omad(v6,v7,v16);
            v19=omad(v6,v8,v19); v20=omad(v6,v9,v20);
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v16=v7+v20; v19=v19+v8; v20=v34+v9; v16=omad(v6,v3,v16); v3=v35+v6; v15=1.0+v3;
            v19=omad(v6,v4,v19); v20=omad(v6,v5,v20);
        }
        // MIMG 0x50c.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x514.
        { vec4 t=sampleAt(0,vec2(v17,v18));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v13=v17; v14=v12; v12=v18; v34=v5-v2; v17=v3-v0; v18=v4-v1; v35=omul(v17,v17); v17=v8-v1;
        v35=omad(v18,v18,v35); v35=omad(v34,v34,v35); v34=v7-v0; v18=omul(v34,v34); v34=v9-v2; v18=omad(v17,v17,v18);
        v18=omad(v34,v34,v18); cc=v35<v18;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v18=omad(v7,v6,v3); v34=omad(v8,v6,v4); v35=omad(v9,v6,v5); v17=cc?2.0:1.0;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v18=omad(v3,v6,v7); v34=omad(v4,v6,v8); v35=omad(v5,v6,v9); v17=cc?2.0:1.0;
        }
        // MIMG 0x5cc.
        { vec4 t=sampleAt(0,vec2(v13,v14));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0x5d4.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v13=v5-v2; v11=v3-v0; v12=v4-v1; v14=omul(v11,v11); v11=v8-v1; v14=omad(v12,v12,v14); v14=omad(v13,v13,v14);
        v13=v7-v0; v12=omul(v13,v13); v13=v9-v2; v12=omad(v11,v11,v12); v12=omad(v13,v13,v12); cc=v14<v12;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v11=v3+v18; v12=v34+v4; v13=v35+v5; v11=omad(v6,v7,v11); v12=omad(v6,v8,v12);
            v13=omad(v6,v9,v13); v6=v17+v6; v6=1.0+v6;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v11=v7+v18; v12=v34+v8; v13=v35+v9; v11=omad(v6,v3,v11); v12=omad(v6,v4,v12);
            v13=omad(v6,v5,v13); v6=v17+v6; v6=1.0+v6;
        }
        v7=v24+v27; v8=v25+v29; v9=v26+v30; v10=v23+v28; v7=v16+v7; v8=v19+v8; v9=v20+v9; v10=v15+v10; v26=v11+v7;
        v27=v12+v8; v28=v13+v9; v35=v6+v10;
    } else {
        v3=s14+v33; v19=v22+v32; v23=v21-v3; v18=v21+v3; v3=v23; v4=v19; v24=v22-v32;
        // MIMG 0x6d0.
        { vec4 t=sampleAt(0,vec2(v18,v19));
          v12=t.x; v13=t.y; v14=t.z; }
        // MIMG 0x6d8.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v6=t.x; v7=t.y; v8=t.z; }
        // MIMG 0x6e0.
        { vec4 t=sampleAt(0,vec2(v23,v24));
          v15=t.x; v16=t.y; v17=t.z; }
        v3=v18; v4=v24;
        // MIMG 0x6f0.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v9=t.x; v10=t.y; v11=t.z; }
        v25=v21+v33; v26=v19;
        // MIMG 0x700.
        { vec4 t=sampleAt(0,vec2(v25,v26));
          v3=t.x; v4=t.y; v5=t.z; }
        v28=v24; v27=v21-v33; v26=v24; v20=s15+v32; v29=v25; v34=v14-v2; v24=v12-v0; v30=v13-v1; v35=omul(v24,v24);
        v24=v16-v1; v35=omad(v30,v30,v35); v35=omad(v34,v34,v35); v34=v15-v0; v30=omul(v34,v34); v34=v17-v2;
        v30=omad(v24,v24,v30); v30=omad(v34,v34,v30); cc=v35<v30; v24=cc?v14:v17; v14=v10-v1; v15=cc?v12:v15;
        v12=v6-v0; v12=omul(v12,v12); v16=cc?v13:v16; v13=v7-v1; v12=omad(v13,v13,v12); v13=v8-v2;
        v12=omad(v13,v13,v12); v13=v9-v0; v13=omul(v13,v13); v13=omad(v14,v14,v13); v14=v11-v2;
        v13=omad(v14,v14,v13); cc=v12<v13;
        // MIMG 0x79c.
        { vec4 t=sampleAt(0,vec2(v27,v28));
          v12=t.x; v13=t.y; v14=t.z; }
        v28=v19; v10=cc?v7:v10; v6=cc?v6:v9; v9=cc?v8:v11; v8=v3-v0; v19=omul(v8,v8); v17=v6+v15; v16=v10+v16;
        v15=v9+v24;
        // MIMG 0x7cc.
        { vec4 t=sampleAt(0,vec2(v27,v28));
          v9=t.x; v10=t.y; v11=t.z; }
        // MIMG 0x7d4.
        { vec4 t=sampleAt(0,vec2(v25,v26));
          v6=t.x; v7=t.y; v8=t.z; }
        v26=v22+v20; v28=v22-v20; v20=v4-v1; v19=omad(v20,v20,v19); v20=v5-v2; v19=omad(v20,v20,v19); v30=v28;
        v35=8.0; v20=v12-v0; v20=omul(v20,v20); v24=v13-v1; v20=omad(v24,v24,v20); v24=v14-v2; v20=omad(v24,v24,v20);
        cc=v19<v20; v12=cc?v3:v12; v3=cc?v4:v13; v4=cc?v5:v14; v14=v9-v0; v13=v10-v1; v19=omul(v14,v14); v14=v11-v2;
        v5=v7-v1; v19=omad(v13,v13,v19); v19=omad(v14,v14,v19); v14=v6-v0; v13=omul(v14,v14); v14=v8-v2;
        v13=omad(v5,v5,v13); v13=omad(v14,v14,v13); cc=v19<v13; v5=cc?v9:v6; v6=v5+v12; v5=cc?v11:v8; v19=v26;
        v5=v5+v4; v4=cc?v10:v7; v3=v4+v3; v4=v26; v24=v28; v20=v5+v15; v34=v6+v17;
        // MIMG 0x890.
        { vec4 t=sampleAt(0,vec2(v25,v26));
          v12=t.x; v13=t.y; v14=t.z; }
        // MIMG 0x898.
        { vec4 t=sampleAt(0,vec2(v29,v30));
          v6=t.x; v7=t.y; v8=t.z; }
        v29=v3+v16; v3=v27;
        // MIMG 0x8a8.
        { vec4 t=sampleAt(0,vec2(v27,v28));
          v15=t.x; v16=t.y; v17=t.z; }
        // MIMG 0x8b0.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v9=t.x; v10=t.y; v11=t.z; }
        // MIMG 0x8b8.
        { vec4 t=sampleAt(0,vec2(v18,v19));
          v3=t.x; v4=t.y; v5=t.z; }
        v27=v12-v0; v30=omul(v27,v27); v19=v13-v1; v27=v14-v2; v30=omad(v19,v19,v30); v19=omad(v27,v27,v30);
        v27=v15-v0; v27=omul(v27,v27); v30=v16-v1; v27=omad(v30,v30,v27); v30=v17-v2; v27=omad(v30,v30,v27);
        cc=v19<v27; v16=cc?v13:v16; v13=v10-v1; v15=cc?v12:v15; v12=cc?v14:v17; v17=v9-v0; v19=omul(v17,v17);
        v17=v11-v2; v14=v7-v1; v19=omad(v13,v13,v19); v19=omad(v17,v17,v19); v17=v6-v0; v13=omul(v17,v17); v17=v8-v2;
        v13=omad(v14,v14,v13); v13=omad(v17,v17,v13); cc=v19<v13; v7=cc?v10:v7; v6=cc?v9:v6; v9=v6+v15; v6=cc?v11:v8;
        v8=v7+v16; v7=v4-v1; v19=v28; v6=v6+v12;
        // MIMG 0x968.
        { vec4 t=sampleAt(0,vec2(v23,v24));
          v12=t.x; v13=t.y; v14=t.z; }
        v24=v26; v15=v34+v9; v16=v29+v8; v8=v3-v0; v9=omul(v8,v8); v8=v5-v2; v9=omad(v7,v7,v9); v26=omad(v8,v8,v9);
        v17=v20+v6;
        // MIMG 0x998.
        { vec4 t=sampleAt(0,vec2(v23,v24));
          v6=t.x; v7=t.y; v8=t.z; }
        // MIMG 0x9a0.
        { vec4 t=sampleAt(0,vec2(v18,v19));
          v9=t.x; v10=t.y; v11=t.z; }
        v19=v12-v0; v20=omul(v19,v19); v18=v13-v1; v19=v14-v2; v20=omad(v18,v18,v20); v20=omad(v19,v19,v20);
        cc=v26<v20; v12=cc?v3:v12; v3=cc?v4:v13; v4=cc?v5:v14; v13=v6-v0; v5=v9-v0; v14=omul(v13,v13); v18=v7-v1;
        v5=omul(v5,v5); v13=v10-v1; v14=omad(v18,v18,v14); v18=v8-v2; v5=omad(v13,v13,v5); v13=v11-v2;
        v14=omad(v18,v18,v14); v5=omad(v13,v13,v5); cc=v14<v5; v5=cc?v6:v9; v5=v5+v12; v6=cc?v8:v11; v4=v6+v4;
        v26=v15+v5; v5=cc?v7:v10; v3=v5+v3; v27=v16+v3; v28=v17+v4;
    }
    if (enabled) {
        v3=v21+v33; v4=v22; v7=v21-v33; v8=v22;
        // MIMG 0xa4c.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0xa54.
        { vec4 t=sampleAt(0,vec2(v7,v8));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v14=v22+v32; v13=v21; v12=v22-v32; v11=v21; v17=v5-v2; v15=v3-v0; v16=v4-v1; v18=omul(v15,v15); v15=v8-v1;
        v18=omad(v16,v16,v18); v18=omad(v17,v17,v18); v17=v7-v0; v16=omul(v17,v17); v17=v9-v2; v16=omad(v15,v15,v16);
        v16=omad(v17,v17,v16); cc=v18<v16;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v15=omad(v7,v6,v3); v16=omad(v8,v6,v4); v17=omad(v9,v6,v5); v18=cc?2.0:1.0;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v15=omad(v3,v6,v7); v16=omad(v4,v6,v8); v17=omad(v5,v6,v9); v18=cc?2.0:1.0;
        }
        // MIMG 0xb14.
        { vec4 t=sampleAt(0,vec2(v13,v14));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0xb1c.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v13=v5-v2; v11=v3-v0; v12=v4-v1; v14=omul(v11,v11); v11=v8-v1; v14=omad(v12,v12,v14); v14=omad(v13,v13,v14);
        v13=v7-v0; v12=omul(v13,v13); v13=v9-v2; v12=omad(v11,v11,v12); v12=omad(v13,v13,v12); cc=v14<v12;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v10=v18+v6; v14=v3+v15; v15=v16+v4; v16=v17+v5; v13=1.0+v10; v14=omad(v6,v7,v14);
            v15=omad(v6,v8,v15); v16=omad(v6,v9,v16);
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v10=v18+v6; v14=v7+v15; v15=v16+v8; v16=v17+v9; v13=1.0+v10; v14=omad(v6,v3,v14);
            v15=omad(v6,v4,v15); v16=omad(v6,v5,v16);
        }
        v5=s14+v33; v3=v21+v5; v4=v22; v7=v21-v5; v8=v22;
        // MIMG 0xbdc.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0xbe4.
        { vec4 t=sampleAt(0,vec2(v7,v8));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v11=s15+v32; v12=v22+v11; v22=v22-v11; v11=v21; v19=v5-v2; v17=v3-v0; v18=v4-v1; v20=omul(v17,v17);
        v17=v8-v1; v20=omad(v18,v18,v20); v20=omad(v19,v19,v20); v19=v7-v0; v18=omul(v19,v19); v19=v9-v2;
        v18=omad(v17,v17,v18); v18=omad(v19,v19,v18); cc=v20<v18;
        if (cc) {
            cc=v10>=v31; v6=cc?1.0:0.0; v17=omad(v7,v6,v3); v18=omad(v8,v6,v4); v19=omad(v9,v6,v5); v20=cc?2.0:1.0;
        } else {
            cc=v6>=v31; v6=cc?1.0:0.0; v17=omad(v3,v6,v7); v18=omad(v4,v6,v8); v19=omad(v5,v6,v9); v20=cc?2.0:1.0;
        }
        // MIMG 0xca0.
        { vec4 t=sampleAt(0,vec2(v11,v12));
          v3=t.x; v4=t.y; v5=t.z; v6=t.w; }
        // MIMG 0xca8.
        { vec4 t=sampleAt(0,vec2(v21,v22));
          v7=t.x; v8=t.y; v9=t.z; v10=t.w; }
        v21=v5-v2; v11=v3-v0; v12=v4-v1; v22=omul(v11,v11); v11=v8-v1; v22=omad(v12,v12,v22); v22=omad(v21,v21,v22);
        v21=v7-v0; v12=omul(v21,v21); v21=v9-v2; v12=omad(v11,v11,v12); v12=omad(v21,v21,v12); cc=v22<v12;
        if (cc) {
            cc=v10>=v31; v10=cc?1.0:0.0; v6=v20+v10; v11=v3+v17; v12=v18+v4; v17=v19+v5; v6=1.0+v6; v11=omad(v10,v7,v11);
            v12=omad(v10,v8,v12); v17=omad(v10,v9,v17);
        } else {
            cc=v6>=v31; v10=cc?1.0:0.0; v6=v20+v10; v11=v7+v17; v12=v18+v8; v17=v19+v9; v6=1.0+v6; v11=omad(v10,v3,v11);
            v12=omad(v10,v4,v12); v17=omad(v10,v5,v17);
        }
        v7=v14+v26; v8=v15+v27; v9=v16+v28; v10=v13+v35; v3=v11+v7; v4=v12+v8; v5=v17+v9; v25=v6+v10;
    } else {
        v3=v21+v33; v4=v22;
        // MIMG 0xd84.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v15=t.x; v16=t.y; v17=t.z; }
        v3=v21-v33; v6=s14+v33; v29=s15+v32; v7=v22; v24=v22+v29; v23=v21; v25=4.0+v35;
        // MIMG 0xda8.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v18=t.x; v19=t.y; v20=t.z; }
        // MIMG 0xdb0.
        { vec4 t=sampleAt(0,vec2(v23,v24));
          v33=t.x; v34=t.y; v35=t.z; }
        v4=v22+v32; v3=v21;
        // MIMG 0xdc0.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v9=t.x; v10=t.y; v11=t.z; }
        v4=v22-v32;
        // MIMG 0xdcc.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v12=t.x; v13=t.y; v14=t.z; }
        v3=v21+v6; v4=v22; v6=v21-v6;
        // MIMG 0xde0.
        { vec4 t=sampleAt(0,vec2(v3,v4));
          v3=t.x; v4=t.y; v5=t.z; }
        // MIMG 0xde8.
        { vec4 t=sampleAt(0,vec2(v6,v7));
          v6=t.x; v7=t.y; v8=t.z; }
        v22=v22-v29; v31=v17-v2; v29=v15-v0; v30=v16-v1; v32=omul(v29,v29); v32=omad(v30,v30,v32);
        v32=omad(v31,v31,v32); v31=v18-v0; v30=omul(v31,v31); v29=v19-v1; v31=v20-v2; v30=omad(v29,v29,v30);
        v30=omad(v31,v31,v30); cc=v32<v30; v15=cc?v15:v18; v18=v10-v1; v16=cc?v16:v19; v19=v9-v0; v29=omul(v19,v19);
        v19=v11-v2; v29=omad(v18,v18,v29); v29=omad(v19,v19,v29); v19=v12-v0; v19=omul(v19,v19); v17=cc?v17:v20;
        v20=v13-v1; v19=omad(v20,v20,v19); v20=v14-v2; v19=omad(v20,v20,v19); cc=v29<v19; v9=cc?v9:v12;
        v11=cc?v11:v14; v9=v9+v15; v20=v4-v1; v19=v7-v1; v13=cc?v10:v13; v13=v13+v16; v12=v11+v17; v11=v3-v0;
        v18=omul(v11,v11); v18=omad(v20,v20,v18); v20=v5-v2; v18=omad(v20,v20,v18); v20=v6-v0; v15=v26+v9;
        // MIMG 0xebc.
        { vec4 t=sampleAt(0,vec2(v21,v22));
          v9=t.x; v10=t.y; v11=t.z; }
        v21=omul(v20,v20); v20=v8-v2; v21=omad(v19,v19,v21); v21=omad(v20,v20,v21); cc=v18<v21; v3=cc?v3:v6;
        v4=cc?v4:v7; v7=v34-v1; v5=cc?v5:v8; v8=v33-v0; v18=omul(v8,v8); v8=v35-v2; v18=omad(v7,v7,v18);
        v18=omad(v8,v8,v18); v16=v27+v13; v17=v28+v12; v8=v9-v0; v7=omul(v8,v8); v6=v10-v1; v8=v11-v2;
        v7=omad(v6,v6,v7); v7=omad(v8,v8,v7); cc=v18<v7; v6=cc?v33:v9; v7=cc?v34:v10; v8=cc?v35:v11; v3=v6+v3;
        v4=v7+v4; v5=v8+v5; v3=v15+v3; v4=v16+v4; v5=v17+v5;
    }
    v6=omad(2.0,v25,1.0); v6=1.0/(v6); v0=omad(2.0,v3,v0); v1=omad(2.0,v4,v1); v2=omad(2.0,v5,v2);
    v0=omul(v6,v0); v1=omul(v6,v1); v2=omul(v6,v2);
    put(vec4(v0,v1,v2,1.0));
#elif ROLE == 11
    // Scalar loads 0x58 (s0..7), 0x5c (s16..19), 0x60 (s8).
    vec3 rgb=sampleAt(0,a0.zw).rgb; // MIMG 0x3c
    float depth=sampleAt(1,a0.xy).x; // MIMG 0x50, distinct sampler
    float v3=c[5].w,v4=c[6].y,v5,v6;
    v6=clamp(rgb.r-c[4].z,0.0,1.0); rgb.r=omad(c[4].w,v6,rgb.r);
    v6=clamp(rgb.g-c[4].z,0.0,1.0); rgb.g=omad(c[4].w,v6,rgb.g);
    v5=c[4].x+depth; v5=1.0/v5; v5=omul(c[4].y,v5);
    v4=clamp(omad(c[6].x,v5,v4),0.0,1.0); v4=omul(c[6].z,v4);
    v3=clamp(omad(c[5].z,v5,v3),0.0,1.0);
    v5=clamp(rgb.b-c[4].z,0.0,1.0); v3=max(v3,v4);
    rgb.b=omad(c[4].w,v5,rgb.b);
    rgb.r=omul(v3,rgb.r); rgb.g=omul(v3,rgb.g); rgb.b=omul(v3,rgb.b);
    if ((floatBitsToUint(c[7].x)&8u)!=0u) {
        // Conditional scalar load 0xdc (s0..3).
        v4=-1.0+c[8].z; v4=omad(v3,v4,1.0);
        rgb.r=omul(c[8].x,rgb.r); rgb.g=omul(c[8].x,rgb.g); rgb.b=omul(c[8].x,rgb.b);
        rgb.r=omul(rgb.r,v4); rgb.g=omul(rgb.g,v4); rgb.b=omul(rgb.b,v4);
        v3=omul(c[8].x,v3);
    }
    put(vec4(rgb,v3));
#elif ROLE == 12
    float v0,v1,v2,v3,v4,v5,v6,v7;
    float v8,v9,v10,v11,v12;
    float s12=c[4].x,s13=c[4].y;
    bool cc;
    v2=gl_FragCoord.x; v3=gl_FragCoord.y;
    v1=omul(s13,v3); v4=B(0x3f400000u); v0=omul(s12,v2); v3=omad(v4,-s13,v1); v4=omad(v4,-s12,v0); v7=s13+v3;
    v6=s12+v4; v5=v7; v2=v6;
    // MIMG 0x50.
    v7=sampleAt(0,vec2(v6,v7)).x;
    // MIMG 0x58.
    v8=sampleAt(1,vec2(v4,v5)).x;
    // MIMG 0x60.
    v10=sampleAt(2,vec2(v0,v1)).x; v5=v3;
    // MIMG 0x6c.
    v6=sampleAt(3,vec2(v2,v3)).x;
    // MIMG 0x74.
    v4=sampleAt(4,vec2(v4,v5)).x; v2=-8.0; v9=v7+v8; v5=min(min(v8,v10),v7); v3=v7+v6; v11=v8+v4; v3=v3-v11;
    v7=max(max(v8,v10),v7); v10=v6+v4; v12=v9+v10; v12=omul(B(0x3d000000u),v12); v8=v9-v10;
    v9=max(B(0x3c000000u),v12); v10=min(abs(v8),abs(v3)); v9=v9+v10; v9=1.0/(v9); v8=omul(v9,v8);
    v3=omul(-v3,v9); v8=min(max(v2,v8),8.0); v2=min(max(v2,v3),8.0); v9=omul(s12,v8); v10=omul(s13,v2);
    v2=omad(v9,B(0x3e2aaaacu),v0); v3=omad(v10,B(0x3e2aaaacu),v1);
    // MIMG 0x110.
    v8=sampleAt(5,vec2(v2,v3)).x; v2=omad(v9,B(0xbe2aaaaau),v0); v3=omad(v10,B(0xbe2aaaaau),v1);
    // MIMG 0x128.
    v11=sampleAt(6,vec2(v2,v3)).x; v2=omad(0.5,v9,v0); v3=omad(0.5,v10,v1);
    // MIMG 0x140.
    v12=sampleAt(7,vec2(v2,v3)).x; v2=omad(-0.5,v9,v0); v3=omad(-0.5,v10,v1);
    // MIMG 0x158.
    v1=sampleAt(8,vec2(v2,v3)).x; v0=min(min(v4,v5),v6); v2=max(max(v4,v7),v6); v4=v8+v11; v3=omul(0.5,v4);
    v1=v12+v1; v1=v4+v1; v1=omul(B(0x3e800000u),v1); cc=v1>=v0; v0=cc?1.0:0.0; cc=v1>=v2; v2=cc?1.0:0.0;
    v0=v0-v2; v3=omad(v3,-v0,v3); v3=omad(v1,v0,v3);
    put(vec4(v3,v3,v3,1.0));
#endif
}
)";

// capture-5's two haze PS dumps have identical executable bytes [0,0x78).
// In the first 0x100 bytes only the footer word at 0xa0 differs:
// 0xbce8ebd5 versus 0x9ce2dbd5. Later differences belong to following programs.
// Both are shader_names' "fx_haze_pass0.fxo +3"; select the VS per draw.
constexpr const char* kHazeVs = R"(
layout(set=0,binding=0,std430) readonly buffer HazeVertex { uint w[]; } hv[3];
layout(location=0) out vec4 a0;
layout(location=1) out vec4 a1;
layout(location=2) out vec4 a2;
uint hazeWord(int slot,uint i) {
    if (slot==0) return hv[0].w[i];
    if (slot==1) return hv[1].w[i];
    return hv[2].w[i];
}
float hazeDecode(uint raw,uint bits,uint nfmt) {
    if (nfmt==7u) return unpackHalf2x16(raw).x;
    if (nfmt==0u) return float(raw)/float((1u<<bits)-1u);
    return max(float(bitfieldExtract(int(raw),0,int(bits)))/
               float((1u<<(bits-1u))-1u),-1.0);
}
vec4 hazeFetch(int slot) {
    uvec4 f=floatBitsToUint(c[20+slot]);
    uvec4 control=floatBitsToUint(c[23+slot]);
    uint index=control.y==0u?uint(gl_VertexIndex):uint(gl_InstanceIndex);
    if (index>=f.w) return vec4(0.0);
    uint base=f.x+index*f.y, dfmt=f.z&15u, nfmt=(f.z>>4u)&7u;
    vec4 r=vec4(0.0,0.0,0.0,1.0);
    if (dfmt==4u || dfmt==11u || dfmt==13u || dfmt==14u) {
        uint n=dfmt==4u?1u:dfmt==11u?2u:dfmt==13u?3u:4u;
        for (uint j=0u;j<n;++j) r[j]=uintBitsToFloat(hazeWord(slot,base+j));
    } else {
        bool halfWidth=dfmt==2u || dfmt==5u || dfmt==12u;
        uint bits=halfWidth?16u:8u;
        uint n=(dfmt==1u || dfmt==2u)?1u:(dfmt==3u || dfmt==5u)?2u:4u;
        for (uint j=0u;j<n;++j) {
            uint raw=(hazeWord(slot,base+(j*bits)/32u)>>((j*bits)%32u))&((1u<<bits)-1u);
            r[j]=hazeDecode(raw,bits,nfmt);
        }
    }
    if (control.x==0u) return r; // implicit identity selection, as decodeFormat()
    vec4 result;
    for (uint j=0u;j<4u;++j) {
        uint s=(control.x>>(3u*j))&7u;
        result[j]=s>=4u?r[s-4u]:(s==1u?1.0:0.0);
    }
    return result;
}
float hazeAdd(float a,float b) { precise float r=a+b; return r; }
float hazeSub(float a,float b) { precise float r=a-b; return r; }
float hazeObjectRow(vec4 row,float shift,vec3 v) {
    float r=mac(row.x,v.x,hazeAdd(shift,row.w));
    r=mac(row.y,v.y,r);
    return mac(row.z,v.z,r);
}
float hazeViewRow(vec4 row,vec3 v) {
    float r=mul(row.x,v.x);
    r=mac(row.y,v.y,r);
    return mac(row.z,v.z,r);
}
float hazeProjectRow(vec4 row,vec3 v) {
    float r=mac(row.x,v.x,row.w);
    r=mac(row.y,v.y,r);
    return mac(row.z,v.z,r);
}
void main() {
    vec4 pos=hazeFetch(0), colour=hazeFetch(1), uv=hazeFetch(2);
    a1=vec4(uv.xy,0.0,0.0);
    a2=vec4(0.0);
#if HAZE_MESH
    vec3 world=vec3(hazeObjectRow(c[3],c[2].x,pos.xyz),
                    hazeObjectRow(c[4],c[2].y,pos.xyz),
                    hazeObjectRow(c[5],c[2].z,pos.xyz));
    // 0x9c branches over the camera-origin subtraction only for this flag pair.
    if ((floatBitsToUint(c[1].x)&0x180000u)!=0x80000u)
        world=vec3(hazeSub(world.x,c[6].x),hazeSub(world.y,c[6].y),hazeSub(world.z,c[6].z));
    vec3 view=vec3(hazeViewRow(c[7],world),hazeViewRow(c[8],world),hazeViewRow(c[13],world));
    gl_Position=vec4(hazeProjectRow(c[9],view),hazeProjectRow(c[10],view),
                     hazeProjectRow(c[11],view),hazeProjectRow(c[12],view));
    // Preserve the distinct multiplication order for red and alpha.
    a0.r=mul(c[16].r,mul(c[14].r,colour.r));
    a0.g=mul(mul(c[14].g,c[16].g),colour.g);
    a0.b=mul(mul(c[14].b,c[16].b),colour.b);
    a0.a=mul(mul(c[16].a,mul(c[14].a,c[15].x)),colour.a);
#else
    gl_Position=vec4(mac(c[0].xy,pos.xy,c[0].zw),pos.zw);
    a0=colour;
#endif
    if (p.flags.x!=0u) gl_Position.z=mul(gl_Position.z+gl_Position.w,0.5);
}
)";

constexpr const char* kHazeFs = R"(
layout(location=0) in vec4 a0;
layout(location=1) in vec4 a1;
layout(location=0) out vec4 outColour;
layout(set=0,binding=3) uniform sampler2D hazeImage;
void main() {
    // Pixel fres() permits contraction, as in SSAO and outlines. Do not use
    // kHeader's precise vertex helpers here. Opcode 303 currently exports
    // round-to-nearest halves in the translator, despite GCN's RTZ mnemonic.
    vec4 v=texture(hazeImage,a1.xy*c[32].xy)*a0;
    v=vec4(unpackHalf2x16(packHalf2x16(v.xy)),unpackHalf2x16(packHalf2x16(v.zw)));
    outColour=vec4(v[p.swizzle.x],v[p.swizzle.y],v[p.swizzle.z],v[p.swizzle.w]);
}
)";

bool named(const std::string& name,const char* wanted) {
    if (name==wanted) return true;
    const std::string prefix=std::string(wanted)+" +";
    if (name.rfind(prefix,0)!=0 || name.size()==prefix.size()) return false;
    return name.find_first_not_of("0123456789",prefix.size())==std::string::npos;
}

class Post final : public Pass {
public:
    const char* name() const override { return "post"; }
    int claims(const std::string& ps) const override {
        // The haze matched in Fwha's compare run (0 px on ~850 compared draws) once compare mode
        // stopped replacing its copies many times a frame (that, not the haze, flickered the
        // screen). Drawn by default; LPN_POST_HAZE=0 leaves it to the translated renderer.
        static const bool haze = [] {
            const char* e = std::getenv("LPN_POST_HAZE");
            return !e || *e != '0';
        }();
        if (ps==kRoles[kHaze].ps) return haze ? kHaze : 0;
        for (int i=1;i<=kToonEdgeFxaa;++i) if (named(ps,kRoles[i].ps)) return i;
        return 0;
    }
    bool draw(int role,const Draw& d,Gpu& gpu) override {
        if (role==kHaze) return drawHaze(d,gpu);
        // The outline chain matched in Fwha's compare run: finalize and the edge FXAA at 0 px, the
        // outline pass within 1/255 (~4% of pixels a step off). Drawn by default.
        if (role<kDown2 || role>kToonEdgeFxaa) return false;
        // The SSAO chain and depth-bound draws: Codex's port, unmatched yet, drawn only when
        // LPN_NATIVE_COMPARE=post compares it (the glow chain matched within 1/255).
        static const bool comparing = [] {
            const char* e = std::getenv("LPN_NATIVE_COMPARE");
            return e && std::string(e) == "post";
        }();
        (void)comparing;
        // The SSAO term (role 6) is still off on scattered pixels: compare copies only. The temporal
        // filter, blurs and upsampling matched exactly in three compare runs (0 px, ~200 draws a
        // run each) and are drawn; depth or stencil in use stays to compare copies for the glow roles.
        if (!d.compareCopy && role == kSsao) return decline(role, "not yet matched (drawn into compare copies only)");
        // The SSAO port is wrong (its upsample flashed the screen white indoors when compare mode
        // still drew unmatched roles into the frame): off unless LPN_POST_SSAO=1 asks for it.
        static const bool ssao = [] {
            const char* e = std::getenv("LPN_POST_SSAO");
            return e && *e && *e != '0';
        }();
        // Compare runs with these roles and the depth-bound draws flashed the screen (title,
        // indoors, glowing objects): only into compare copies, only with LPN_POST_SSAO=1, and only
        // the roles LPN_POST_ROLES lists (comma separated; all of 6-9 by default), to find which
        // one disturbs the frame with the flash guard's log (hle_videoout.cpp) in a headless run.
        static const uint32_t roles = [] {
            const char* e = std::getenv("LPN_POST_ROLES");
            if (!e || !*e) return (1u << kSsao) | (1u << kTemporal) | (1u << kSsaoBlur) | (1u << kUpsample) | 1u;
            uint32_t m = 0;
            for (const char* p = e; *p;) {
                char* end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                if (v >= 0 && v < 32) m |= 1u << v;
                p = *end ? end + 1 : end;
            }
            return m; // bit 0: the depth-bound glow draws
        }();
        if (role >= kSsao && role <= kUpsample && !(roles >> role & 1)) return decline(role, "the SSAO port is off (LPN_POST_ROLES)");
        if (role == kSsao && !ssao) return decline(role, "the SSAO term is off (LPN_POST_SSAO=1 compares it)");
        // As the merged glow pass: a depth buffer bound but neither tested nor written is drawn
        // (the 2x4 downsample always has one); depth or stencil in use only into compare copies.
        const bool usesDepth = d.depthTest || d.depthWrite || d.stencil || d.depthStencil.depthTestEnable ||
                               d.depthStencil.depthWriteEnable || d.depthStencil.stencilTestEnable ||
                               d.depthStencil.depthBoundsTestEnable;
        if (role < kSsao && usesDepth && (!ssao || !d.compareCopy || !(roles & 1))) return decline(role, "depth or stencil");
        if (role==kSsaoBlur || role==kUpsample || role==kToonEdgeFxaa) {
            // nativeBuild uses shaderNameAt for an unindexed VS: it returns "?".
            // These claimed PSs accept the unnamed vs_fullscreen_53c2a1600.
            if (d.vsName && !d.vsName->empty() && *d.vsName!="?")
                return decline(role,"another vertex shader");
        } else if (!d.vsName || ((role==kLumi || role==kSsao || role==kHandDraw2) ?
                   *d.vsName!=kRoles[role].vs : !named(*d.vsName,kRoles[role].vs)))
            return decline(role,"another vertex shader");
        if (d.colorCount!=1 || !d.color[0].view || d.color[0].format==VK_FORMAT_UNDEFINED ||
            !d.color[0].width || !d.color[0].height || !d.color[0].comps || d.color[0].comps>4 ||
            (role<kSsao && d.color[0].comps!=4))
            return decline(role,"not one supported colour target");
        if (role<kSsao && d.color[0].format!=VK_FORMAT_R8G8B8A8_UNORM)
            return decline(role,"unexpected colour format (expected RGBA8 UNORM)");
        if ((role==kHandDraw2 || role==kToonEdgeFxaa) &&
            (d.color[0].format!=VK_FORMAT_R8G8B8A8_UNORM || d.color[0].comps!=4))
            return decline(role,"unexpected outline colour format (expected RGBA8 UNORM)");
        if ((d.writeMask&~0xfu) || !(d.writeMask&0xfu))
            return decline(role,"unexpected target write mask");
        const auto& blend=d.blendState[0];
        const bool additive=blend.blendEnable;
        // Capture-5's composition adds RGB and alpha (0x61010101); other roles replace.
        if (additive && role<kSsao) {
            if (role!=kCompose || blend.srcColorBlendFactor!=VK_BLEND_FACTOR_ONE ||
                blend.dstColorBlendFactor!=VK_BLEND_FACTOR_ONE || blend.colorBlendOp!=VK_BLEND_OP_ADD ||
                blend.srcAlphaBlendFactor!=VK_BLEND_FACTOR_ONE || blend.dstAlphaBlendFactor!=VK_BLEND_FACTOR_ONE ||
                blend.alphaBlendOp!=VK_BLEND_OP_ADD) return decline(role,"unexpected blending");
        } else if (!additive && d.blend) return decline(role,"inconsistent blend state");
        const auto& z=d.depthStencil;
        if ((d.depthTest || d.depthWrite || d.stencil || z.depthTestEnable ||
             z.depthWriteEnable || z.stencilTestEnable || z.depthBoundsTestEnable) && !d.depth.view)
            return decline(role,"depth/stencil state without an attachment");
        if (d.pixelPositions || d.depthBias || d.cull!=VK_CULL_MODE_NONE)
            return decline(role,"pixel positions, polygon offset or culling");
        if (d.primitive!=17 || d.count!=3 || d.instances!=1 || d.indexed)
            return decline(role,"not one nonindexed rectangle");
        if (d.scissor[2]<0 || d.scissor[3]<0 || !(d.vkViewport.width>0) || d.vkViewport.height==0)
            return decline(role,"invalid viewport or scissor");
        if (gpu.maxPushConstants()<sizeof(Push)) return decline(role,"push constant limit");

        Push push{};
        push.clipZHalf=d.clipZHalf;
        swizzle(d.color[0],push.swizzle);
        std::array<VkDescriptorBufferInfo,3> buffers{};
        const VertexBuffer* slots[2]={};
        for (const VertexBuffer& v:d.vertex) {
            if (!v.components && v.slot>=0) continue; // unused third slot in vs_glow_pass1
            if (v.slot<0 || v.slot>1) return decline(role,"unexpected vertex buffer");
            if (slots[v.slot]) return decline(role,"duplicate fetch slot");
            slots[v.slot]=&v;
        }
        const int fetches=(role==kHandDraw2 || role==kHandDrawFinalize)?2:(role>=kSsaoBlur?1:2);
        for (int i=0;i<fetches;++i) {
            const VertexBuffer* v=slots[i];
            // Capture-5 uses float3 position with DST_SEL_W=1 and float2 UV.
            if (!v || !v->data.buffer || v->nfmt!=7 || v->indexVgpr ||
                v->components<(i?2u:4u) || v->components>4 ||
                (v->dfmt!=11 && v->dfmt!=13 && v->dfmt!=14))
                return decline(role,"unsupported float32 quad fetch");
            const uint32_t n=v->dfmt==11?2:v->dfmt==13?3:4;
            // The VS consumes four position lanes and two UV lanes; decode 0/1 selectors too.
            if (!v->dstSel) return decline(role,"implicit fetch selection");
            for (uint32_t j=0;j<(i?2u:4u);++j)
                if (((v->dstSel>>(3*j))&7)>1 && ((v->dstSel>>(3*j))&7)<4)
                    return decline(role,"reserved fetch selection");
            const uint64_t first=uint64_t(v->data.byteOffset)+v->offset;
            const uint64_t end=first+uint64_t(2)*v->stride+n*4;
            if (v->records<3 || first%4 || v->stride%4 || end/4>UINT32_MAX ||
                !v->data.range || v->data.range==VK_WHOLE_SIZE || end>v->data.range)
                return decline(role,"quad fetch outside buffer or unaligned");
            push.fetch[i][0]=uint32_t(first/4);
            push.fetch[i][1]=v->stride/4;
            push.fetch[i][2]=n;
            push.fetch[i][3]=v->dstSel;
            buffers[i]={v->data.buffer,v->data.offset,v->data.range};
        }

        if (fetches==1) buffers[1]=buffers[0]; // unused descriptor remains valid

        Constants constants{};
        auto load=[&](bool vs,uint32_t pc,uint32_t n,int at) {
            uint32_t words[16]{};
            const auto& fn=vs?d.vsLoad:d.load;
            if (!fn || !fn(pc,words,n)) return false;
            std::memcpy(reinterpret_cast<unsigned char*>(constants.data())+at*16,words,n*4);
            return true;
        };
        const uint32_t positionPc=role==kHandDrawFinalize?0xc:role==kTemporal?0x14:(role==kLumi || role>=kSsao?0x10:0xc);
        if (!load(true,positionPc,4,0)) return decline(role,"missing VS position scalar load");
        uint32_t wantedTextures=0;
        if (role>=kHandDraw2) {
            if (role==kHandDraw2) {
                if (!load(false,0xc,4,4)) return decline(role,"missing hand-draw scalar load");
                const Texture* first=nullptr;
                for (uint32_t pc:kHandDrawSamples) {
                    const Texture* t=nullptr;
                    for (const Texture& x:d.textures) if (x.pc==pc) {
                        if (t) return decline(role,"duplicate hand-draw MIMG PC");
                        t=&x;
                    }
                    if (!t || !t->view || !t->sampler)
                        return decline(role,"missing hand-draw texture or sampler at MIMG PC");
                    if (first && (t->view!=first->view || t->sampler!=first->sampler ||
                        t->address!=first->address || std::memcmp(t->scale,first->scale,sizeof t->scale)))
                        return decline(role,"hand-draw MIMG bindings differ");
                    first=t;
                }
            } else if (role==kHandDrawFinalize) {
                if (!load(true,0x10,2,1) || !load(false,0x58,8,4) ||
                    !load(false,0x5c,4,6) || !load(false,0x60,1,7))
                    return decline(role,"missing hand-draw finalize scalar load");
                if ((constants[7][0]&8) && !load(false,0xdc,4,8))
                    return decline(role,"missing hand-draw finalize conditional scalar load");
            } else if (!load(false,0x8,2,4))
                return decline(role,"missing toon-edge FXAA scalar load");
        } else if (role>=kSsao) {
            if (role==kSsao && !load(true,0x14,8,1))
                return decline(role,"missing SSAO VS ray/UV load");
            if (role==kTemporal && !load(true,0x10,4,1))
                return decline(role,"missing temporal VS ray load");
            struct Scalar { uint32_t pc,n; int at; };
            // PCs in the drawn programs, not the other programs in the FXO archives.
            static constexpr Scalar term[]={{0x2c,2,13},{0x58,16,4},{0xa0,16,8},{0xa8,2,12}};
            static constexpr Scalar temporal[]={{0x40,8,4},{0x50,8,6},{0x68,4,8},{0x9c,4,9},
                {0xb4,8,10},{0xec,8,12},{0x118,4,14},{0x14c,8,15},{0x154,4,17},{0x158,4,18}};
            static constexpr Scalar blur[]={{0x24,2,4},{0x28,8,5}};
            static constexpr Scalar upsample[]={{0x2c,2,4},{0x158,4,5}};
            const std::span<const Scalar> loads=role==kSsao?std::span<const Scalar>(term):
                role==kTemporal?std::span<const Scalar>(temporal):
                role==kSsaoBlur?std::span<const Scalar>(blur):std::span<const Scalar>(upsample);
            for (const Scalar& s:loads)
                if (!load(false,s.pc,s.n,s.at)) return decline(role,"missing SSAO scalar load");
        } else if (role==kLumi) {
            if (!load(false,0x40,1,7) || !load(false,0x44,4,4))
                return decline(role,"missing luminance flags or scale load");
            wantedTextures=(constants[7][0]&8)?2:1;
            if (wantedTextures==2 && !load(false,0x90,8,5))
                return decline(role,"missing luminance threshold/colour load at 0x90");
        } else if (role==kCompose) {
            if (!load(true,0x14,8,1) || !load(true,0x10,4,3) || !load(false,0xd4,8,4))
                return decline(role,"missing composition scalar load");
        } else {
            if (!load(true,0x10,role==kDown2?1:2,1)) return decline(role,"missing VS UV load at 0x10");
            if (role==kBlur && (!load(false,0x10,16,4) || !load(false,0x8,8,8)))
                return decline(role,"missing blur scalar load at 0x10 or 0x8");
        }

        std::array<VkDescriptorImageInfo,18> images{};
        uint32_t n=0;
        for (uint32_t pc:kRoles[role].samples) {
            if (!pc || (wantedTextures && n==wantedTextures)) break;
            const Texture* t=nullptr;
            for (const Texture& x:d.textures) if (x.pc==pc) {
                if (t) return decline(role,"duplicate MIMG PC");
                t=&x;
            }
            if (!t || !t->view || !t->sampler) return decline(role,"missing texture or sampler at MIMG PC");
            if (t->address==d.color[0].address) return decline(role,"texture aliases colour target");
            images[n]={t->sampler,t->view,VK_IMAGE_LAYOUT_GENERAL};
            std::memcpy(constants[32+n].data(),t->scale,sizeof t->scale);
            ++n;
        }
        VkPipeline pipe=pipeline(gpu,role,d,n);
        if (!pipe) return decline(role,"pipeline creation failed");
        Buffer cb;
        if (!gpu.upload(constants.data(),sizeof constants,cb)) return decline(role,"constant upload failed");
        buffers[2]={cb.buffer,cb.offset,cb.range};
        // Gpu::upload currently returns byteOffset 0; GLSL c[] starts at the descriptor base.
        if (cb.byteOffset) return decline(role,"nonzero constant upload byte offset");
        if (kCompare) {
            char b[256];
            std::snprintf(b,sizeof b,"vs %s; target %ux%u format %d mask 0x%x; %u MIMG samples; clipZHalf %u",
                          d.vsName?d.vsName->c_str():"?",d.color[0].width,d.color[0].height,int(d.color[0].format),
                          d.writeMask,n,push.clipZHalf);
            info_=b;
            for (int i=0;i<(role>=kSsao?20:10);++i) {
                float f[4]; std::memcpy(f,constants[i].data(),sizeof f);
                std::snprintf(b,sizeof b,"\n    c%d: %g %g %g %g",i,f[0],f[1],f[2],f[3]);
                info_+=b;
            }
        }
        const VkRect2D sc{{d.scissor[0],d.scissor[1]},{uint32_t(d.scissor[2]),uint32_t(d.scissor[3])}};
        std::array<uint32_t,6> st{};
        for (int i=0;i<2;++i)
            st[i*3]=d.stencilRef[i],st[i*3+1]=d.stencilCompare[i],st[i*3+2]=d.stencilWrite[i];
        gpu.beginRendering(d.color,1,d.depth,d.hasStencil);
        gpu.record([pipe,layout=layouts_[role],push,buffers,images,n,vp=d.vkViewport,sc,
                    st,bc=std::to_array(d.blendConstants),bounds=z.depthBoundsTestEnable,
                    minBounds=z.minDepthBounds,maxBounds=z.maxDepthBounds,
                    pushSet=gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            VkWriteDescriptorSet w[4]{};
            for (uint32_t i=0;i<3;++i) {
                w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding=i; w[i].descriptorCount=1;
                w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo=&buffers[i];
            }
            w[3].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[3].dstBinding=3; w[3].descriptorCount=n;
            w[3].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[3].pImageInfo=images.data();
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipe);
            vkCmdSetBlendConstants(cmd,bc.data());
            for (int i=0;i<2;++i) {
                const VkStencilFaceFlags face=i?VK_STENCIL_FACE_BACK_BIT:VK_STENCIL_FACE_FRONT_BIT;
                vkCmdSetStencilReference(cmd,face,st[i*3]);
                vkCmdSetStencilCompareMask(cmd,face,st[i*3+1]);
                vkCmdSetStencilWriteMask(cmd,face,st[i*3+2]);
            }
            if (bounds) vkCmdSetDepthBounds(cmd,minBounds,maxBounds);
            vkCmdSetViewport(cmd,0,1,&vp);
            vkCmdSetScissor(cmd,0,1,&sc);
            pushSet(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,4,w);
            vkCmdPushConstants(cmd,layout,kStages,0,sizeof push,&push);
            vkCmdDraw(cmd,3,1,0,0);
        });
        return true;
    }

    std::string lastDrawInfo() const override { return info_; }

private:
    bool drawHaze(const Draw& d,Gpu& gpu) {
        const int role=kHaze;
        const bool mesh=d.vsName && *d.vsName=="p_dvtx_modulate.fxo";
        if (!mesh && d.vsName && *d.vsName!="?")
            return decline(role,"another haze vertex shader");
        if (d.colorCount!=1 || !d.color[0].view || d.color[0].format==VK_FORMAT_UNDEFINED ||
            !d.color[0].width || !d.color[0].height || !d.color[0].comps || d.color[0].comps>4)
            return decline(role,"not one supported colour target");
        if ((d.writeMask&~0xfu) || !(d.writeMask&0xfu))
            return decline(role,"unexpected target write mask");
        const auto& z=d.depthStencil;
        if ((d.depthTest || d.depthWrite || d.stencil || z.depthTestEnable ||
             z.depthWriteEnable || z.stencilTestEnable || z.depthBoundsTestEnable) && !d.depth.view)
            return decline(role,"depth/stencil state without an attachment");
        if (d.pixelPositions || d.depthBias || d.cull!=VK_CULL_MODE_NONE)
            return decline(role,"pixel positions, polygon offset or culling");
        if (d.instances!=1 || (mesh ? (d.primitive!=4 || !d.indexed || !d.count || d.count%3) :
                                     (d.primitive!=17 || d.indexed || d.count!=3)))
            return decline(role,"unsupported haze primitive");
        if (d.scissor[2]<0 || d.scissor[3]<0 || !(d.vkViewport.width>0) || d.vkViewport.height==0)
            return decline(role,"invalid viewport or scissor");
        if (gpu.maxPushConstants()<sizeof(Push)) return decline(role,"push constant limit");
        if (mesh) {
            const uint32_t bytes=d.indexType==VK_INDEX_TYPE_UINT16?2:
                                 d.indexType==VK_INDEX_TYPE_UINT32?4:0;
            const uint64_t end=uint64_t(d.index.byteOffset)+(uint64_t(d.firstIndex)+d.count)*bytes;
            if (!bytes || !d.index.buffer || d.index.byteOffset%bytes ||
                !d.index.range || d.index.range==VK_WHOLE_SIZE || end>d.index.range)
                return decline(role,"unsupported haze index buffer");
        }

        Constants constants{};
        Push push{};
        push.clipZHalf=d.clipZHalf;
        swizzle(d.color[0],push.swizzle);
        std::array<VkDescriptorBufferInfo,5> buffers{};
        const VertexBuffer* slots[3]{};
        for (const VertexBuffer& v:d.vertex) {
            if (!v.components) continue;
            if (v.slot<0 || v.slot>2) return decline(role,"unexpected haze vertex buffer");
            if (slots[v.slot]) return decline(role,"duplicate haze fetch slot");
            slots[v.slot]=&v;
        }
        // v4: position, v8: colour, v12: UV. Decode float32, half, UNORM
        // and SNORM with DST_SEL, including the quad's constant position W.
        const uint32_t used[3]={mesh?3u:4u,4,2};
        for (uint32_t i=0;i<3;++i) {
            const VertexBuffer* v=slots[i];
            if (!v || !v->data.buffer || v->components<used[i] || v->components>4 ||
                v->indexVgpr>3 || !v->stride || v->stride%4)
                return decline(role,"unsupported haze fetch input");
            uint32_t bytes=0;
            bool wide=false,half=false;
            switch (v->dfmt) {
            case 1: bytes=1; break;
            case 2: bytes=2; half=true; break;
            case 3: bytes=2; break;
            case 4: bytes=4; wide=true; break;
            case 5: bytes=4; half=true; break;
            case 10: bytes=4; break;
            case 11: bytes=8; wide=true; break;
            case 12: bytes=8; half=true; break;
            case 13: bytes=12; wide=true; break;
            case 14: bytes=16; wide=true; break;
            default: return decline(role,"unsupported haze fetch format");
            }
            if (wide ? v->nfmt!=7 : !(v->nfmt==0 || v->nfmt==1 || (half && v->nfmt==7)))
                return decline(role,"unsupported haze fetch number format");
            for (uint32_t j=0;j<used[i];++j) {
                const uint32_t s=(v->dstSel>>(3*j))&7;
                if (s==2 || s==3) return decline(role,"reserved haze fetch selection");
            }
            const uint64_t first=uint64_t(v->data.byteOffset)+v->offset;
            const uint64_t end=first+(v->records?uint64_t(v->records-1)*v->stride+((bytes+3u)&~3u):0);
            if (!v->records || (!mesh && !v->indexVgpr && v->records<3) || first%4 ||
                end>uint64_t(UINT32_MAX)*4 || !v->data.range ||
                v->data.range==VK_WHOLE_SIZE || end>v->data.range)
                return decline(role,"haze fetch outside buffer or unaligned");
            constants[20+i]={uint32_t(first/4),v->stride/4,v->dfmt|(v->nfmt<<4),v->records};
            constants[23+i]={v->dstSel,v->indexVgpr,0,0};
            buffers[i]={v->data.buffer,v->data.offset,v->data.range};
        }
        buffers[3]=buffers[0]; // unused binding 1 remains valid
        auto load=[&](uint32_t pc,uint32_t n,int at) {
            uint32_t words[16]{};
            if (!d.vsLoad || !d.vsLoad(pc,words,n)) return false;
            std::memcpy(constants[at].data(),words,n*4);
            return true;
        };
        if (mesh) {
            // Load PCs of drawn_vs_p_dvtx_modulate.s, not offsets in the FXO.
            if (!load(0x1c,1,1) || !load(0x20,4,2) || !load(0x24,8,3) ||
                !load(0x28,4,5) || !load(0xc8,1,15) || !load(0xcc,4,16) ||
                !load(0xd4,8,7) || !load(0xd8,16,9) || !load(0xdc,4,13) ||
                !load(0xe0,4,14))
                return decline(role,"missing haze mesh VS scalar load");
            if ((constants[1][0]&0x180000)!=0x80000 && !load(0xa8,4,6))
                return decline(role,"missing haze camera-origin load");
        } else if (!load(0x10,4,0)) {
            return decline(role,"missing haze rectangle transform");
        }
        const Texture* texture=nullptr;
        for (const Texture& t:d.textures) if (t.pc==kRoles[role].samples[0]) {
            if (texture) return decline(role,"duplicate haze MIMG PC");
            texture=&t;
        }
        if (!texture || !texture->view || !texture->sampler)
            return decline(role,"missing haze texture or sampler at MIMG PC");
        if (texture->address==d.color[0].address)
            return decline(role,"texture aliases colour target");
        std::memcpy(constants[32].data(),texture->scale,sizeof texture->scale);
        const VkDescriptorImageInfo image{texture->sampler,texture->view,VK_IMAGE_LAYOUT_GENERAL};
        const VkPipeline pipe=hazePipeline(gpu,d,mesh);
        if (!pipe) return decline(role,"haze pipeline creation failed");
        Buffer cb;
        if (!gpu.upload(constants.data(),sizeof constants,cb) || cb.byteOffset)
            return decline(role,"haze constant upload failed or nonzero byte offset");
        buffers[4]={cb.buffer,cb.offset,cb.range};
        if (kCompare) {
            char b[256];
            std::snprintf(b,sizeof b,"haze %s; target %ux%u format %d mask 0x%x; %u indices; blend %u; clipZHalf %u",
                          mesh?"p_dvtx_modulate.fxo":"?",d.color[0].width,d.color[0].height,
                          int(d.color[0].format),d.writeMask,mesh?d.count:0,
                          d.blendState[0].blendEnable,push.clipZHalf);
            info_=b;
        }
        const VkRect2D sc{{d.scissor[0],d.scissor[1]},{uint32_t(d.scissor[2]),uint32_t(d.scissor[3])}};
        std::array<uint32_t,6> st{};
        for (int i=0;i<2;++i)
            st[i*3]=d.stencilRef[i],st[i*3+1]=d.stencilCompare[i],st[i*3+2]=d.stencilWrite[i];
        gpu.beginRendering(d.color,1,d.depth,d.hasStencil);
        gpu.record([pipe,layout=hazeLayouts_[mesh],push,buffers,image,mesh,
                    count=d.count,index=d.index,indexType=d.indexType,firstIndex=d.firstIndex,
                    vp=d.vkViewport,sc,st,bc=std::to_array(d.blendConstants),
                    bounds=z.depthBoundsTestEnable,minBounds=z.minDepthBounds,maxBounds=z.maxDepthBounds,
                    pushSet=gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            VkWriteDescriptorSet w[4]{};
            for (uint32_t i=0;i<3;++i) {
                w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding=i; w[i].descriptorCount=i==0?3:1;
                w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo=&buffers[i==0?0:i+2];
            }
            w[3].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[3].dstBinding=3; w[3].descriptorCount=1;
            w[3].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[3].pImageInfo=&image;
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipe);
            vkCmdSetBlendConstants(cmd,bc.data());
            for (int i=0;i<2;++i) {
                const VkStencilFaceFlags face=i?VK_STENCIL_FACE_BACK_BIT:VK_STENCIL_FACE_FRONT_BIT;
                vkCmdSetStencilReference(cmd,face,st[i*3]);
                vkCmdSetStencilCompareMask(cmd,face,st[i*3+1]);
                vkCmdSetStencilWriteMask(cmd,face,st[i*3+2]);
            }
            if (bounds) vkCmdSetDepthBounds(cmd,minBounds,maxBounds);
            vkCmdSetViewport(cmd,0,1,&vp);
            vkCmdSetScissor(cmd,0,1,&sc);
            pushSet(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,4,w);
            vkCmdPushConstants(cmd,layout,kStages,0,sizeof push,&push);
            if (mesh) {
                vkCmdBindIndexBuffer(cmd,index.buffer,index.offset+index.byteOffset,indexType);
                vkCmdDrawIndexed(cmd,count,1,firstIndex,0,0);
            } else vkCmdDraw(cmd,3,1,0,0);
        });
        return true;
    }

    bool decline(int role,const char* why) {
        std::lock_guard lock(m_);
        if (declined_[role].insert(why).second)
            std::fprintf(stderr,"[native] post: %s left to the translated renderer (%s)\n",kRoles[role].ps,why);
        return false;
    }
    VkPipeline pipeline(Gpu& gpu,int role,const Draw& d,uint32_t textures) {
        std::lock_guard lock(m_);
        const VkFormat format=d.color[0].format;
        const uint32_t mask=d.writeMask&0xf;
        const auto& z=d.depthStencil;
        const auto& b=d.blendState[0];
        const uint64_t blendKey=uint64_t(b.blendEnable) | uint64_t(b.srcColorBlendFactor)<<1 |
            uint64_t(b.dstColorBlendFactor)<<6 | uint64_t(b.colorBlendOp)<<11 |
            uint64_t(b.srcAlphaBlendFactor)<<14 | uint64_t(b.dstAlphaBlendFactor)<<19 |
            uint64_t(b.alphaBlendOp)<<24;
        const bool stencil=d.stencil && d.hasStencil && z.stencilTestEnable;
        uint64_t stencilKey=uint64_t(stencil)<<40;
        if (stencil)
            for (int i=0;i<2;++i) {
                const VkStencilOpState& o=i?z.back:z.front;
                stencilKey|=uint64_t(o.failOp | o.passOp<<3 | o.depthFailOp<<6 | o.compareOp<<9)<<(16*i);
            }
        const auto key=std::make_tuple(role,int(format),mask,textures,blendKey,
            int(d.depth.view?d.depth.format:VK_FORMAT_UNDEFINED),bool(d.hasStencil),
            z.depthTestEnable,z.depthWriteEnable,int(z.depthCompareOp),z.depthBoundsTestEnable,int(d.frontFace),stencilKey);
        if (auto it=pipes_.find(key);it!=pipes_.end()) return it->second;
        VkPipeline& out=pipes_[key];
        if (!layouts_[role]) {
            VkDescriptorSetLayoutBinding bindings[4]{};
            for (uint32_t i=0;i<3;++i)
                bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,i==2?kStages:VkShaderStageFlags(VK_SHADER_STAGE_VERTEX_BIT),nullptr};
            // Luminance can toggle its second sample between draws; keep one compatible layout.
            bindings[3]={3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,role==kLumi?2u:textures,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount=4; dl.pBindings=bindings;
            if (vkCreateDescriptorSetLayout(gpu.device(),&dl,nullptr,&sets_[role])!=VK_SUCCESS) return out;
            const VkPushConstantRange range{kStages,0,sizeof(Push)};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount=1; pl.pSetLayouts=&sets_[role];
            pl.pushConstantRangeCount=1; pl.pPushConstantRanges=&range;
            if (vkCreatePipelineLayout(gpu.device(),&pl,nullptr,&layouts_[role])!=VK_SUCCESS) return out;
        }
        const std::string header=std::string(kHeader)+"\n#define ROLE "+std::to_string(role)+
                                 "\n#define TEXTURES "+std::to_string(textures)+"\n";
        const VkShaderModule vs=gpu.shader(header+(role>=kHandDraw2?kOutlineVs:kVs),Gpu::Stage::Vertex);
        std::string fsHeader=header;
        if (role==kSsao)
            fsHeader.insert(fsHeader.find('\n')+1,
                "#extension GL_KHR_shader_subgroup_basic : require\n"
                "#extension GL_KHR_shader_subgroup_quad : require\n");
        const VkShaderModule fs=gpu.shader(fsHeader+(role>=kHandDraw2?kOutlineFs:role>=kSsao?kSsaoFs:kFs),Gpu::Stage::Fragment);
        const VkShaderModule gs=gpu.shader(kGs,Gpu::Stage::Geometry);
        if (!vs || !fs || !gs) return out;
        VkPipelineShaderStageCreateInfo stages[3]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vs,"main",nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,fs,"main",nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_GEOMETRY_BIT,gs,"main",nullptr},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount=vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.frontFace=d.frontFace; rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds=z;
        ds.pNext=nullptr;
        ds.stencilTestEnable=stencil;
        VkPipelineColorBlendAttachmentState attachment=b;
        attachment.colorWriteMask=mask;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount=1; cb.pAttachments=&attachment;
        const VkDynamicState states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS,VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,VK_DYNAMIC_STATE_DEPTH_BOUNDS};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount=std::size(states); dy.pDynamicStates=states;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount=1; rendering.pColorAttachmentFormats=&format;
        if (d.depth.view) {
            rendering.depthAttachmentFormat=d.depth.format;
            if (d.hasStencil) rendering.stencilAttachmentFormat=d.depth.format;
        }
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.pNext=&rendering; pi.stageCount=3; pi.pStages=stages;
        pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia; pi.pViewportState=&vp;
        pi.pRasterizationState=&rs; pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb; pi.pDynamicState=&dy; pi.layout=layouts_[role];
        if (vkCreateGraphicsPipelines(gpu.device(),VK_NULL_HANDLE,1,&pi,nullptr,&out)!=VK_SUCCESS) out=VK_NULL_HANDLE;
        return out;
    }
    VkPipeline hazePipeline(Gpu& gpu,const Draw& d,bool mesh) {
        const int role=mesh?1:0;
        const uint32_t textures=1;
        std::lock_guard lock(m_);
        const VkFormat format=d.color[0].format;
        const uint32_t mask=d.writeMask&0xf;
        const auto& z=d.depthStencil;
        const auto& b=d.blendState[0];
        const uint64_t blendKey=uint64_t(b.blendEnable) | uint64_t(b.srcColorBlendFactor)<<1 |
            uint64_t(b.dstColorBlendFactor)<<6 | uint64_t(b.colorBlendOp)<<11 |
            uint64_t(b.srcAlphaBlendFactor)<<14 | uint64_t(b.dstAlphaBlendFactor)<<19 |
            uint64_t(b.alphaBlendOp)<<24;
        const bool stencil=d.stencil && d.hasStencil && z.stencilTestEnable;
        uint64_t stencilKey=uint64_t(stencil)<<40;
        if (stencil)
            for (int i=0;i<2;++i) {
                const VkStencilOpState& o=i?z.back:z.front;
                stencilKey|=uint64_t(o.failOp | o.passOp<<3 | o.depthFailOp<<6 | o.compareOp<<9)<<(16*i);
            }
        const auto key=std::make_tuple(role,int(format),mask,textures,blendKey,
            int(d.depth.view?d.depth.format:VK_FORMAT_UNDEFINED),bool(d.hasStencil),
            z.depthTestEnable,z.depthWriteEnable,int(z.depthCompareOp),z.depthBoundsTestEnable,int(d.frontFace),stencilKey);
        if (auto it=hazePipes_.find(key);it!=hazePipes_.end()) return it->second;
        VkPipeline& out=hazePipes_[key];
        if (!hazeLayouts_[role]) {
            VkDescriptorSetLayoutBinding bindings[4]{};
            for (uint32_t i=0;i<3;++i)
                bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,i==2?kStages:VkShaderStageFlags(VK_SHADER_STAGE_VERTEX_BIT),nullptr};
            bindings[0].descriptorCount=3;
            bindings[3]={3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount=4; dl.pBindings=bindings;
            if (vkCreateDescriptorSetLayout(gpu.device(),&dl,nullptr,&hazeSets_[role])!=VK_SUCCESS) return out;
            const VkPushConstantRange range{kStages,0,sizeof(Push)};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount=1; pl.pSetLayouts=&hazeSets_[role];
            pl.pushConstantRangeCount=1; pl.pPushConstantRanges=&range;
            if (vkCreatePipelineLayout(gpu.device(),&pl,nullptr,&hazeLayouts_[role])!=VK_SUCCESS) return out;
        }
        const std::string header=std::string(kHeader)+"\n#define HAZE_MESH "+std::to_string(mesh?1:0)+"\n";
        const VkShaderModule vs=gpu.shader(header+kHazeVs,Gpu::Stage::Vertex);
        const VkShaderModule fs=gpu.shader(header+kHazeFs,Gpu::Stage::Fragment);
        const VkShaderModule gs=mesh?VK_NULL_HANDLE:gpu.shader(kGs,Gpu::Stage::Geometry);
        if (!vs || !fs || (!mesh && !gs)) return out;
        VkPipelineShaderStageCreateInfo stages[3]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vs,"main",nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,fs,"main",nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_GEOMETRY_BIT,gs,"main",nullptr},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount=vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.frontFace=d.frontFace; rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds=z;
        ds.pNext=nullptr;
        ds.stencilTestEnable=stencil;
        VkPipelineColorBlendAttachmentState attachment=b;
        attachment.colorWriteMask=mask;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount=1; cb.pAttachments=&attachment;
        const VkDynamicState states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS,VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,VK_DYNAMIC_STATE_DEPTH_BOUNDS};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount=std::size(states); dy.pDynamicStates=states;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount=1; rendering.pColorAttachmentFormats=&format;
        if (d.depth.view) {
            rendering.depthAttachmentFormat=d.depth.format;
            if (d.hasStencil) rendering.stencilAttachmentFormat=d.depth.format;
        }
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.pNext=&rendering; pi.stageCount=mesh?2:3; pi.pStages=stages;
        pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia; pi.pViewportState=&vp;
        pi.pRasterizationState=&rs; pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb; pi.pDynamicState=&dy; pi.layout=hazeLayouts_[role];
        if (vkCreateGraphicsPipelines(gpu.device(),VK_NULL_HANDLE,1,&pi,nullptr,&out)!=VK_SUCCESS) out=VK_NULL_HANDLE;
        return out;
    }
    std::mutex m_;
    using PipelineKey=std::tuple<int,int,uint32_t,uint32_t,uint64_t,int,bool,
                                 VkBool32,VkBool32,int,VkBool32,int,uint64_t>;
    std::map<PipelineKey,VkPipeline> pipes_;
    VkDescriptorSetLayout sets_[kToonEdgeFxaa+1]{};
    VkPipelineLayout layouts_[kToonEdgeFxaa+1]{};
    std::map<PipelineKey,VkPipeline> hazePipes_;
    VkDescriptorSetLayout hazeSets_[2]{};
    VkPipelineLayout hazeLayouts_[2]{};
    std::set<std::string> declined_[kHaze+1];
    static inline const bool kCompare=std::getenv("LPN_NATIVE_COMPARE")!=nullptr;
    static inline thread_local std::string info_;
};

} // namespace

Pass* postPass() {
    static Post pass;
    return &pass;
}

} // namespace host::native
