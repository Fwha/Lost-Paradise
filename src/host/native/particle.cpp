// Particle billboards. The five archive VSs share this GLSL implementation.
// Vertex-texture animation and the atlas lane-mask path deliberately fall back.
// native.cpp: declare Pass* particlePass(); and add particlePass() to passes.
// CMake: add host/native/particle.cpp.
#include "host/native/native.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace host::native {
namespace {

// Original constant-buffer images, in vec4s. Different descriptor slots and
// shader stages never alias. The LV shader changes the meaning of V30..V70.
enum : uint32_t {
    cV10 = 0,
    cV20 = 4,
    cV30 = 96,
    cV40 = 120,
    cV50 = 144,
    cV60 = 152,
    cV70 = 160,
    cP0 = 164,
    cP10 = 172,
    cP20 = 176,
    cP30 = 212,
    cOut = 213,
    cTexScale = 214,
    cFetch = 215,       // eleven input descriptions
    cPalette = 226,     // four independent instruction descriptions
    cVsTexScale = 230, // ten independent sample coordinate scales
    kConstants = 240,
    kVsTextures = 10,
    kFetches = 11,
    kPalettes = 4,
};

struct Load {
    bool vs;
    uint32_t pc, n, index, at;
};

#define V(pc, n, b, o) {true, pc, n, b + (o) / 16, ((o) / 4) & 3}
#define P(pc, n, b, o) {false, pc, n, b + (o) / 16, ((o) / 4) & 3}

constexpr Load kVsXX[] = {
    V(0x054,4,cV30,0xa0),
    V(0x0b4,1,cV60,0x00),
    V(0x0e4,1,cV30,0x00),
    V(0x0e8,2,cV30,0x90),
    V(0x1d0,8,cV30,0x40),
    V(0x224,2,cV30,0x90),
    V(0x2a4,1,cV30,0x80),
    V(0x3d0,2,cV30,0x90),
    V(0x4ec,4,cV30,0x60),
    V(0x51c,1,cV30,0x04),
    V(0x520,1,cV30,0x78),
    V(0x6c4,8,cV40,0x00),
    V(0x6c8,4,cV40,0x20),
    V(0x73c,1,cV30,0xfc),
    V(0x740,8,cV30,0xb0),
    V(0x744,16,cV50,0x40),
    V(0x748,1,cV30,0x4c),
    V(0x74c,16,cV50,0x00),
    V(0x894,16,cV30,0x00),
    V(0x898,2,cV30,0x70),
    V(0xca4,2,cV30,0x90),
    V(0xd34,4,cV30,0x120),
    V(0xd6c,1,cV30,0xec),
    V(0xd70,4,cV20,0x590),
    V(0xd9c,2,cV10,0x28),
    V(0xdb8,1,cV30,0x11c),
};
constexpr Load kVsXN[] = {
    V(0x054,4,cV30,0xa0),
    V(0x0b4,1,cV60,0x00),
    V(0x0e4,1,cV30,0x00),
    V(0x0e8,2,cV30,0x90),
    V(0x1d0,8,cV30,0x40),
    V(0x224,2,cV30,0x90),
    V(0x2a4,1,cV30,0x80),
    V(0x3d0,2,cV30,0x90),
    V(0x4ec,4,cV30,0x60),
    V(0x51c,1,cV30,0x04),
    V(0x520,1,cV30,0x78),
    V(0x6cc,8,cV40,0x00),
    V(0x6d0,4,cV40,0x20),
    V(0x744,1,cV30,0xfc),
    V(0x748,8,cV30,0xb0),
    V(0x74c,16,cV50,0x40),
    V(0x750,1,cV30,0x4c),
    V(0x754,16,cV50,0x00),
    V(0x89c,16,cV30,0x00),
    V(0x8a0,2,cV30,0x70),
    V(0xcac,2,cV30,0x90),
    V(0xd3c,4,cV30,0x120),
    V(0xd78,4,cV30,0x110),
    V(0xd7c,8,cV30,0xd0),
    V(0xd80,4,cV20,0x590),
    V(0xdd8,2,cV10,0x28),
};
constexpr Load kVsFN[] = {
    V(0x054,4,cV30,0xa0),
    V(0x0b4,1,cV60,0x00),
    V(0x0e4,1,cV30,0x00),
    V(0x0e8,2,cV30,0x90),
    V(0x1dc,8,cV30,0x40),
    V(0x230,2,cV30,0x90),
    V(0x2b0,1,cV30,0x80),
    V(0x4a0,2,cV30,0x90),
    V(0x5bc,4,cV30,0x60),
    V(0x5ec,1,cV30,0x04),
    V(0x5f0,1,cV30,0x78),
    V(0x824,8,cV40,0x00),
    V(0x828,4,cV40,0x20),
    V(0x8c4,8,cV30,0xb0),
    V(0x8c8,16,cV50,0x40),
    V(0x8cc,1,cV30,0xfc),
    V(0x8d0,1,cV30,0x4c),
    V(0x8d4,16,cV50,0x00),
    V(0xaf0,16,cV30,0x00),
    V(0xaf4,2,cV30,0x70),
    V(0xf00,2,cV30,0x90),
    V(0xf90,4,cV30,0x120),
    V(0xfcc,4,cV30,0x110),
    V(0xfd0,8,cV30,0xd0),
    V(0xfd4,4,cV20,0x590),
    V(0x102c,2,cV10,0x28),
};
constexpr Load kVsFX[] = {
    V(0x054,4,cV30,0xa0),
    V(0x0b4,1,cV60,0x00),
    V(0x0e4,1,cV30,0x00),
    V(0x0e8,2,cV30,0x90),
    V(0x1dc,8,cV30,0x40),
    V(0x230,2,cV30,0x90),
    V(0x2b0,1,cV30,0x80),
    V(0x4a0,2,cV30,0x90),
    V(0x5bc,4,cV30,0x60),
    V(0x5ec,1,cV30,0x04),
    V(0x5f0,1,cV30,0x78),
    V(0x824,8,cV40,0x00),
    V(0x828,4,cV40,0x20),
    V(0x8bc,8,cV30,0xb0),
    V(0x8c0,16,cV50,0x40),
    V(0x8c4,1,cV30,0xfc),
    V(0x8c8,1,cV30,0x4c),
    V(0x8cc,16,cV50,0x00),
    V(0xae0,16,cV30,0x00),
    V(0xae4,2,cV30,0x70),
    V(0xef0,2,cV30,0x90),
    V(0xf80,4,cV30,0x120),
    V(0xfb8,1,cV30,0xec),
    V(0xfbc,4,cV20,0x590),
    V(0xfe8,2,cV10,0x28),
    V(0x1004,1,cV30,0x11c),
};
constexpr Load kVsLV[] = {
    V(0x054,4,cV40,0xa0),
    V(0x0b4,1,cV70,0x00),
    V(0x0e4,1,cV40,0x00),
    V(0x0e8,2,cV40,0x90),
    V(0x1dc,8,cV40,0x40),
    V(0x230,2,cV40,0x90),
    V(0x2b0,1,cV40,0x80),
    V(0x3f8,2,cV40,0x90),
    V(0x514,4,cV40,0x60),
    V(0x544,1,cV40,0x04),
    V(0x548,1,cV40,0x78),
    V(0x724,8,cV50,0x00),
    V(0x728,4,cV50,0x20),
    V(0x7cc,1,cV40,0xfc),
    V(0x7d0,8,cV40,0xb0),
    V(0x7d4,16,cV60,0x40),
    V(0x7d8,1,cV40,0x4c),
    V(0x7dc,16,cV60,0x00),
    V(0x934,16,cV40,0x00),
    V(0x938,2,cV40,0x70),
    V(0xd44,2,cV40,0x90),
    V(0xdd4,4,cV40,0x120),
    V(0xe50,8,cV40,0x100),
    V(0xe54,4,cV30,0x80),
    V(0xe58,8,cV20,0x580),
    V(0xe60,8,cV40,0xd0),
    V(0xe64,1,cV40,0x7c),
    V(0xf48,2,cV10,0x28),
};
constexpr Load kPs[] = {
    P(0x058,1,cP30,0x00),
    P(0x09c,4,cP20,0x230),
    P(0x120,4,cP10,0x10),
    P(0x170,1,cP0,0x5c),
};

#undef V
#undef P

struct Variant {
    const char* vs;
    const char* defines;
    std::span<const Load> loads;
    std::array<uint32_t, kPalettes> palettePc;
    uint32_t material, flags;
    std::array<uint32_t, kVsTextures> texturePc;
};

constexpr Variant kVariants[] = {
    {"vs_ptc_bill_xx_xx_x_x_x_x.vso", "#define PTC_XX\n",
     kVsXX, {0x56c,0x574,0x57c,0x58c}, cV30,cV60, {0x128,0x130,0x140,0x150,0x24c,0x254,0x3f8,0x400,0xccc,0xcd4}},
    {"vs_ptc_bill_xx_xx_x_x_x_n.vso", "#define PTC_XN\n#define DISTANCE_ALPHA\n",
     kVsXN, {0x56c,0x574,0x57c,0x58c}, cV30,cV60, {0x128,0x130,0x140,0x150,0x24c,0x254,0x3f8,0x400,0xcd4,0xcdc}},
    {"vs_ptc_bill_xx_xx_f_x_x_n.vso",
     "#define PTC_FN\n#define VELOCITY_ALIGNED\n#define DISTANCE_ALPHA\n",
     kVsFN, {0x63c,0x644,0x64c,0x65c}, cV30,cV60, {0x128,0x130,0x140,0x150,0x258,0x260,0x4c8,0x4d0,0xf28,0xf30}},
    {"vs_ptc_bill_xx_xx_f_x_x_x.vso", "#define PTC_FX\n#define VELOCITY_ALIGNED\n",
     kVsFX, {0x63c,0x644,0x64c,0x65c}, cV30,cV60, {0x128,0x130,0x140,0x150,0x258,0x260,0x4c8,0x4d0,0xf18,0xf20}},
    {"vs_ptc_bill_xx_lv_x_x_x_n.vso",
     "#define PTC_LV\n#define VERTEX_LIGHTING\n#define DISTANCE_ALPHA\n",
     kVsLV, {0x594,0x59c,0x5a4,0x5b4}, cV40,cV70, {0x128,0x130,0x140,0x150,0x258,0x260,0x420,0x428,0xd6c,0xd74}},
};

constexpr const char* kHeader = R"(#version 450
layout(set=0,binding=2,std430) readonly buffer Constants { uvec4 c[]; } cb;
uint U(int b,int o) { return cb.c[b+o/16][(o/4)&3]; }
float F(int b,int o) { return uintBitsToFloat(U(b,o)); }
vec3 F3(int b,int o) { return vec3(F(b,o),F(b,o+4),F(b,o+8)); }
vec4 F4(int b,int o) { return vec4(F(b,o),F(b,o+4),F(b,o+8),F(b,o+12)); }
float B(uint u) { return uintBitsToFloat(u); }
bool has(uint f,uint m) { return (f&m)!=0u; }

#define ARITH(T) \
T add(T a,T b) { precise T r=a+b; return r; } \
T sub(T a,T b) { precise T r=a-b; return r; } \
T mul(T a,T b) { precise T r=a*b; return r; } \
T mac(T a,T b,T c) { precise T r=a*b+c; return r; } \
T sat(T a) { return clamp(a,T(0.0),T(1.0)); }
ARITH(float)
ARITH(vec2)
ARITH(vec3)
ARITH(vec4)
#undef ARITH
#define MIXED(T) \
T mul(T a,float b) { precise T r=a*b; return r; } \
T mul(float a,T b) { precise T r=a*b; return r; } \
T mac(T a,float b,T c) { precise T r=a*b+c; return r; } \
T mac(float a,T b,T c) { precise T r=a*b+c; return r; }
MIXED(vec2)
MIXED(vec3)
MIXED(vec4)
#undef MIXED
float D3(vec3 a,vec3 b) {
    float r=mul(a.x,b.x);
    r=mac(a.y,b.y,r);
    return mac(a.z,b.z,r);
}
float D4(vec4 a,vec4 b) {
    float r=mul(a.x,b.x);
    r=mac(a.y,b.y,r);
    r=mac(a.z,b.z,r);
    return mac(a.w,b.w,r);
}
)";

constexpr const char* kVs = R"(
layout(set=0,binding=0,std430) readonly buffer FetchBuffer {
    uint w[];
} vertexData[11];
layout(set=0,binding=1,std430) readonly buffer PaletteBuffer {
    uint w[];
} paletteData[4];
layout(set=0,binding=4) uniform sampler2D animationImage[10];

layout(location=0) out vec4 p0;
layout(location=1) out vec4 p1;
layout(location=2) out vec4 p2;
layout(location=3) out vec4 p3;

#ifdef VERTEX_LIGHTING
const int M=120, T=144, Camera=152, Flags=160;
#else
const int M=96, T=120, Camera=144, Flags=152;
#endif
const int V10=0, V20=4, Light=96;

uint word(uint b,uint i) {
    switch (b) {
    case 0u: return vertexData[0].w[i];
    case 1u: return vertexData[1].w[i];
    case 2u: return vertexData[2].w[i];
    case 3u: return vertexData[3].w[i];
    case 4u: return vertexData[4].w[i];
    case 5u: return vertexData[5].w[i];
    case 6u: return vertexData[6].w[i];
    case 7u: return vertexData[7].w[i];
    case 8u: return vertexData[8].w[i];
    case 9u: return vertexData[9].w[i];
    default: return vertexData[10].w[i];
    }
}
uint paletteWord(uint b,uint i) {
    switch (b) {
    case 0u: return paletteData[0].w[i];
    case 1u: return paletteData[1].w[i];
    case 2u: return paletteData[2].w[i];
    default: return paletteData[3].w[i];
    }
}
uint conv(uint raw,uint bits,uint nfmt) {
    int sx=bitfieldExtract(int(raw),0,int(bits));
    if (nfmt==7u)
        return floatBitsToUint(unpackHalf2x16(raw).x);
    if (nfmt==0u)
        return floatBitsToUint(float(raw)/float((1u<<bits)-1u));
    return floatBitsToUint(max(float(sx)/float((1u<<(bits-1u))-1u),-1.0));
}
vec4 fetch(uint b) {
    uvec4 s=cb.c[215+int(b)];
    // A slot the fetch shader indexes by v1-v3 reads the instance (as the translator sets them).
    uint index=(s.z&0x100u)!=0u?uint(gl_InstanceIndex):uint(gl_VertexIndex);
    if (index>=s.w) return vec4(0.0);
    uint i=s.x+index*s.y;
    uint dfmt=s.z&15u, nfmt=(s.z>>4)&7u;
    uvec4 e=uvec4(0u,0u,0u,floatBitsToUint(1.0));
    if (dfmt==4u || dfmt==11u || dfmt==13u || dfmt==14u) {
        uint n=dfmt==4u?1u:dfmt==11u?2u:dfmt==13u?3u:4u;
        for (uint c=0u;c<n;++c) e[c]=word(b,i+c);
    } else if (dfmt==2u || dfmt==5u || dfmt==12u) {
        uint n=dfmt==2u?1u:dfmt==5u?2u:4u;
        for (uint c=0u;c<n;++c)
            e[c]=conv((word(b,i+c/2u)>>(16u*(c&1u)))&0xffffu,16u,nfmt);
    } else {
        uint n=dfmt==1u?1u:dfmt==3u?2u:4u;
        uint x=word(b,i);
        for (uint c=0u;c<n;++c)
            e[c]=conv((x>>(8u*c))&0xffu,8u,nfmt);
    }
    return uintBitsToFloat(e);
}
vec4 palette(uint b,uint index) {
    uvec4 s=cb.c[226+int(b)];
    if (index>=s.z) return vec4(0.0);
    uint i=s.x+index*s.y;
    return uintBitsToFloat(uvec4(paletteWord(b,i),paletteWord(b,i+1u),
                                paletteWord(b,i+2u),paletteWord(b,i+3u)));
}

// Each image_sample_lz has its own PC, sampler and normalized-coordinate scale.
vec4 animationSample(uint i,vec2 coordinate) {
    vec2 scale=uintBitsToFloat(cb.c[230+int(i)].xy);
    coordinate=mul(coordinate,scale);
    switch (i) {
    case 0u: return textureLod(animationImage[0],coordinate,0.0);
    case 1u: return textureLod(animationImage[1],coordinate,0.0);
    case 2u: return textureLod(animationImage[2],coordinate,0.0);
    case 3u: return textureLod(animationImage[3],coordinate,0.0);
    case 4u: return textureLod(animationImage[4],coordinate,0.0);
    case 5u: return textureLod(animationImage[5],coordinate,0.0);
    case 6u: return textureLod(animationImage[6],coordinate,0.0);
    case 7u: return textureLod(animationImage[7],coordinate,0.0);
    case 8u: return textureLod(animationImage[8],coordinate,0.0);
    default: return textureLod(animationImage[9],coordinate,0.0);
    }
}
void animationFrame(float phase,out float next,out float previous,out float blend) {
    float frames=F(M,0x94);
    float frame=floor(mul(frames,phase));
    float inverseWidth=1.0/F(M,0x90);
    next=mul(inverseWidth,add(1.5,frame));
    previous=mul(inverseWidth,add(0.5,frame));
    blend=mac(frames,phase,-frame);
}
uint atlasUint(float x) {
    if (isnan(x) || x<=0.0) return 0u;
    if (x>=4294967296.0) return 0xffffffffu;
    return uint(x);
}
uint atlasInt(float x) {
    if (isnan(x)) return 0u;
    if (x>=2147483648.0) return 0x7fffffffu;
    if (x<=-2147483648.0) return 0x80000000u;
    return uint(int(x));
}
// The reciprocal/mul-hi/carry sequences implement unsigned div/rem.
// Compare masks are consumed in the same lane; no cross-lane data is read.
// The zero-divisor quotient is all ones, and subtraction leaves the dividend.
uint atlasDivide(uint a,uint b) {
    if (b==0u) return 0xffffffffu;
    return a/b;
}
uint atlasRemainder(uint a,uint b) {
    return a-atlasDivide(a,b)*b;
}
vec4 atlasCoordinates(vec4 atlas,vec2 uv,uint age,float t) {
    uint x=atlasUint(atlas.x);
    uint y=atlasUint(atlas.y);
    uint z=atlasUint(atlas.z);
    uint w=atlasUint(atlas.w);
    x+=atlasDivide(age,atlasInt(F(M,0x08)))*atlasInt(F(M,0x70));
    x=atlasRemainder(x,atlasInt(F(M,0x10)));
    y+=x;
    uint q0=atlasDivide(y,atlasInt(F(M,0x18)));
    uint r0=y-q0*atlasInt(F(M,0x18));
    z+=atlasDivide(age,atlasInt(F(M,0x0c)))*atlasInt(F(M,0x74));
    z=atlasRemainder(z,atlasInt(F(M,0x14)));
    w+=z;
    uint q1=atlasDivide(w,atlasInt(F(M,0x1c)));
    uint r1=w-q1*atlasInt(F(M,0x1c));
    vec4 base=vec4(add(uv.x,fract(atlas.x)),add(fract(atlas.y),uv.y),
                   add(uv.x,fract(atlas.z)),add(fract(atlas.w),uv.y));
    base.x=mac(F(M,0x30),t,base.x);
    base.y=mac(t,F(M,0x34),base.y);
    base.z=mac(F(M,0x38),t,base.z);
    base.w=mac(F(M,0x3c),t,base.w);
    return vec4(mac(F(M,0x20),float(r0),base.x),
                mac(F(M,0x24),float(q0),base.y),
                mac(float(r1),F(M,0x28),base.z),
                mac(float(q1),F(M,0x2c),base.w));
}

// q*p*conjugate(q), retaining the instruction boundaries and accumulation
// order. The input quaternion is not implicitly normalized.
vec3 rotatePoint(vec4 q,vec3 p,vec3 origin) {
    float x=mul(q.w,p.x);
    float y=mac(q.w,p.y,mul(q.z,p.x));
    float z=mul(-q.y,p.x);
    float w=mul(q.x,p.x);
    x=mac(p.y,-q.z,x);
    z=mac(p.y,q.x,z);
    x=mac(p.z,q.y,x);
    w=mac(p.y,q.y,w);
    z=mac(p.z,q.w,z);
    y=mac(p.z,-q.x,y);
    w=mac(q.z,p.z,w);
    vec3 r;
    r.x=mac(x,q.w,origin.x);
    r.y=mac(y,q.w,origin.y);
    r.z=mac(z,q.w,origin.z);
    r.x=mac(w,q.x,r.x);
    r.y=mac(q.x,-z,r.y);
    r.z=mac(y,q.x,r.z);
    r.x=mac(z,q.y,r.x);
    r.y=mac(w,q.y,r.y);
    r.z=mac(q.y,-x,r.z);
    r.x=mac(q.z,-y,r.x);
    r.y=mac(x,q.z,r.y);
    r.z=mac(w,q.z,r.z);
    return r;
}
vec3 rotateAnimation(vec4 q,vec3 p) {
    float x=mul(p.x,q.w);
    float z=mul(-q.y,p.x);
    float y=mac(-q.z,p.y,x);
    float w=mul(p.x,q.z);
    float scalar=mul(p.x,q.x);
    z=mac(q.x,p.y,z);
    w=mac(q.w,p.y,w);
    y=mac(q.y,p.z,y);
    scalar=mac(q.y,p.y,scalar);
    z=mac(q.w,p.z,z);
    w=mac(p.z,-q.x,w);
    float rx=mul(q.w,y);
    float ry=mul(q.w,w);
    float rz=mul(q.w,z);
    scalar=mac(q.z,p.z,scalar);
    rx=mac(q.x,scalar,rx);
    ry=mac(q.x,-z,ry);
    rz=mac(w,q.x,rz);
    rx=mac(z,q.y,rx);
    ry=mac(scalar,q.y,ry);
    rz=mac(q.y,-y,rz);
    return vec3(mac(-w,q.z,rx),mac(q.z,y,ry),mac(q.z,scalar,rz));
}
vec3 rotateVector(vec4 q,vec3 p) {
    float x=mul(q.w,p.x);
    float y=mac(-q.z,p.y,mul(q.w,p.x));
    // Use distinct temporaries: y above is the first quaternion component.
    x=y;
    y=mac(q.x,p.y,mul(-q.y,p.x));
    float z=mac(q.w,p.y,mul(q.z,p.x));
    float w=mac(q.y,p.y,mul(q.x,p.x));
    x=mac(p.z,q.y,x);
    z=mac(p.z,-q.x,z);
    y=mac(p.z,q.w,y);
    w=mac(p.z,q.z,w);
    vec3 r;
    r.x=mul(q.w,x);
    r.y=mul(q.w,z);
    r.z=mul(q.w,y);
    r.x=mac(q.x,w,r.x);
    r.y=mac(q.x,-y,r.y);
    r.z=mac(z,q.x,r.z);
    r.x=mac(y,q.y,r.x);
    r.y=mac(w,q.y,r.y);
    r.z=mac(-x,q.y,r.z);
    r.x=mac(-z,q.z,r.x);
    r.y=mac(x,q.z,r.y);
    r.z=mac(w,q.z,r.z);
    return r;
}
float pointRow(vec4 row,vec3 p) {
    float r=mac(row.x,p.x,row.w);
    r=mac(row.y,p.y,r);
    return mac(row.z,p.z,r);
}
float vectorRow(vec4 row,vec3 p) {
    float r=mul(row.x,p.x);
    r=mac(row.y,p.y,r);
    return mac(row.z,p.z,r);
}
vec3 pointRows(vec4 a,vec4 b,vec4 c,vec3 p) {
    return vec3(pointRow(a,p),pointRow(b,p),pointRow(c,p));
}
vec3 vectorRows(vec4 a,vec4 b,vec4 c,vec3 p) {
    return vec3(vectorRow(a,p),vectorRow(b,p),vectorRow(c,p));
}

// The LV palette path explicitly constructs these three affine rows.
// They are also retained for its lighting-position calculation.
void quaternionRows(vec4 q,vec3 p,out vec4 a,out vec4 b,out vec4 c) {
    float yy=mul(q.y,q.y);
    float yz=mac(q.z,q.z,yy);
    float zw=mul(q.z,q.w);
    float yw=mul(q.y,q.w);
    float xx=mul(q.x,q.x);
    float xz=mac(q.z,q.z,xx);
    float xw=mul(q.x,q.w);
    float yx=mac(q.x,q.x,yy);
    a.x=mac(-2.0,yz,1.0);
    b.x=mul(mac(q.x,q.y,zw),2.0);
    c.x=mul(mac(-1.0,yw,mul(q.x,q.z)),2.0);
    a.y=mul(mac(q.x,q.y,-zw),2.0);
    b.y=mac(-2.0,xz,1.0);
    c.y=mul(mac(q.y,q.z,xw),2.0);
    a.z=mul(mac(q.x,q.z,yw),2.0);
    b.z=mul(mac(q.y,q.z,-xw),2.0);
    c.z=mac(-2.0,yx,1.0);
    a.w=p.x; b.w=p.y; c.w=p.z;
}

void main() {
    vec4 corner=fetch(0u);
    vec4 uv=fetch(1u);
    vec4 start=fetch(2u);
    vec3 velocity=fetch(3u).xyz;
    vec4 colour=fetch(4u);
    vec4 accelerationLife=fetch(5u);
    float initialSpeed=fetch(6u).w;
    float finalSpeed=fetch(7u).w;
    vec4 sizeAngle=fetch(9u);
    vec3 variation=fetch(10u).xyz;

    vec4 atlas=fetch(8u);

    // The atlas input at slot 8 is unused in the accepted flag configuration.
    // Keep the fetch contract identical across all five variants.
    p0=p1=p2=p3=vec4(0.0);
    gl_Position=vec4(0.0);

    int born=int(start.w);
    int life=int(accelerationLife.w);
    int now=int(F(M,0xa8));
    uint wrap=uint(int(F(M,0xac)))*uint(now<born?1:0);
    int age=int(uint(now)-uint(born)+wrap);
    if (!(age>=0 && age<life)) return;

    uint flags=U(Flags,0);
    float t=float(age);
    float normalizedAge=mul(t,1.0/float(life));
    vec3 animationVariation=vec3(0.0);
    float halfTime=mul(mac(t,t,t),0.5);
    int ramp=int(F(M,0x80));
    float rampF=float(ramp);
    float inverseRamp=1.0/rampF;
    float k=mul(initialSpeed,inverseRamp);
    float correction=mul(-rampF,k);
    correction=mul(correction,add(1.0,rampF));
    float after=mul(rampF,finalSpeed);
    after=mac(0.5,correction,after);
    float speedDifference=sub(finalSpeed,initialSpeed);
    float early=mul(t,finalSpeed);
    early=mac(k,-halfTime,early);
    float lateAge=float(int(uint(age)-uint(ramp)));
    after=mac(lateAge,speedDifference,after);
    float earlyMask=age<ramp?1.0:0.0;
    float lateMask=age<ramp?0.0:1.0;
    float travel=mul(early,earlyMask);
    travel=mac(after,lateMask,travel);

    vec3 local=mac(velocity,travel,start.xyz);
    vec3 sourceOrigin=F3(M,0x40);
    vec4 sourceRotation=F4(M,0x50);
    if (has(flags,0x20u)) {
        float phase=float(int(uint(int(F(M,0xa4)))-uint(age)));
        phase=mac(phase,1.0/float(int(F(M,0x00))),B(0x461c4000u));
        phase=fract(phase);
        float next,previous,blend;
        animationFrame(phase,next,previous,blend);
        vec4 qa=animationSample(0u,vec2(next,0.25));
        vec4 qb=animationSample(1u,vec2(previous,0.25));
        vec3 pa=animationSample(2u,vec2(next,B(0x3daaaaabu))).xyz;
        vec3 pb=animationSample(3u,vec2(previous,B(0x3daaaaabu))).xyz;
        sourceRotation=mac(sub(qa,qb),blend,qb);
        sourceRotation=mul(inversesqrt(D4(sourceRotation,sourceRotation)),sourceRotation);
        sourceOrigin=mac(sub(pa,pb),blend,pb);
    }
    if (has(flags,0x40u)) {
        float next,previous,blend;
        animationFrame(normalizedAge,next,previous,blend);
        vec3 a=animationSample(4u,vec2(next,B(0x3ed55555u))).xyz;
        vec3 b=animationSample(5u,vec2(previous,B(0x3ed55555u))).xyz;
        animationVariation=mac(blend,sub(a,b),b);
    }
    vec3 moved=rotatePoint(sourceRotation,local,sourceOrigin);
    moved=mac(halfTime,accelerationLife.xyz,moved);

#ifdef VELOCITY_ALIGNED
    float rate=mac(-t,k,finalSpeed);
    rate=mul(rate,earlyMask);
    rate=mac(lateMask,speedDifference,rate);
    vec3 derivative=rotateVector(sourceRotation,mul(velocity,rate));
    derivative=mac(t,accelerationLife.xyz,derivative);
#endif

    if (has(flags,0x200u)) {
        float next,previous,blend;
        animationFrame(normalizedAge,next,previous,blend);
        vec4 a=animationSample(6u,vec2(next,B(0x3f6aaaabu)));
        vec4 b=animationSample(7u,vec2(previous,B(0x3f6aaaabu)));
        vec4 q=mac(sub(a,b),blend,b);
        q=mul(inversesqrt(D4(q,q)),q);
        moved=rotateAnimation(q,moved);
    }

    vec3 scale=F3(M,0x60);
    vec3 scaled=mul(scale,moved);
    vec4 row0=F4(T,0x00), row1=F4(T,0x10), row2=F4(T,0x20);
    vec3 world;
#ifdef VELOCITY_ALIGNED
    vec3 worldDerivative;
#endif

    if (has(flags,16u)) {
        float phase=mac(F(M,0x78),t,float(born));
        phase=mul(1.0/float(int(F(M,0x04))),phase);
        phase=fract(phase);
        float f=mul(B(0x41f80000u),phase);
        float negativeFrame=floor(-f);
        uint frame=uint(int(-negativeFrame));
        uint twice=frame<<1u;
        vec4 qa=palette(0u,twice-1u);
        vec4 qb=palette(1u,twice+1u);
        vec3 pa=palette(2u,twice).xyz;
        vec3 pb=palette(3u,twice-2u).xyz;
        float blend=mac(B(0x41f80000u),-phase,-negativeFrame);
        float hemisphere=D4(qa,qb);
        if (0.0>hemisphere) qb=-qb;
        vec4 q=mac(sub(qa,qb),blend,qb);
        q=mul(inversesqrt(D4(q,q)),q);
        vec3 position=mac(sub(pb,pa),blend,pa);
#ifdef VERTEX_LIGHTING
        quaternionRows(q,position,row0,row1,row2);
        world=pointRows(row0,row1,row2,scaled);
#else
        world=rotatePoint(q,scaled,position);
#endif
#ifdef VELOCITY_ALIGNED
        worldDerivative=rotateVector(q,derivative);
#endif
    } else {
        world=pointRows(row0,row1,row2,scaled);
#ifdef VELOCITY_ALIGNED
        worldDerivative=vectorRows(row0,row1,row2,derivative);
#endif
    }

    // Animation-texture variations are zero in the accepted configuration.

    sizeAngle.x=mac(animationVariation.x,variation.x,sizeAngle.x);
    sizeAngle.y=mac(animationVariation.y,variation.y,sizeAngle.y);
    float cornerX=mul(corner.x,sizeAngle.x);
    float cornerY=mul(corner.y,sizeAngle.y);

    vec3 emitterMotion=mul(F3(M,0xb0),halfTime);
    emitterMotion=mac(F3(M,0xc0),t,emitterMotion);
    world=mac(scale,emitterMotion,world);

    vec4 view0=F4(Camera,0x40);
    vec4 view1=F4(Camera,0x50);
    vec4 view2=F4(Camera,0x60);
    vec4 view3=F4(Camera,0x70);

    float angle=add(F(M,0xfc),sizeAngle.z);
#ifdef PTC_XX
    angle=mac(animationVariation.z,variation.z,angle);
#else
    angle=mac(variation.z,animationVariation.z,angle);
#endif
    angle=mac(sizeAngle.w,t,angle);
    angle=mul(B(0x3e22f983u),angle);
    angle=fract(angle);
    float sn=sin(mul(angle,6.28318530717958647692));
    float cs=cos(mul(angle,6.28318530717958647692));

    vec4 view;
    view.x=pointRow(view0,world);
    view.y=pointRow(view1,world);
    view.z=pointRow(view2,world);
    view.w=pointRow(view3,world);
    float offsetX,offsetY;

#ifdef VELOCITY_ALIGNED
    vec3 heading=add(F3(M,0xc0),worldDerivative);
    heading=mac(F3(M,0xb0),t,heading);
    vec3 ahead=mac(heading,B(0x447a0000u),world);
    float aheadX=add(vectorRow(view0,ahead),view0.w);
    float aheadY=add(vectorRow(view1,ahead),view1.w);
    float aheadZ=pointRow(view2,ahead);
    float inverseZ=1.0/view.z;
    float inverseAheadZ=1.0/aheadZ;
    float dx=mul(inverseZ,view.x);
    dx=mac(aheadX,-inverseAheadZ,dx);
    float dy=mul(inverseZ,view.y);
    dy=mac(aheadY,-inverseAheadZ,dy);
    float dx2=mul(dx,dx);
    float length2=mac(dy,dy,dx2);
    dy=add(length2==0.0?1.0:0.0,dy);
    float inverseLength=inversesqrt(mac(dy,dy,dx2));
    dx=mul(inverseLength,dx);
    dy=mul(inverseLength,dy);
    float alignedSin=mul(sn,dx);
    alignedSin=mac(dy,cs,alignedSin);
    float alignedCos=mul(cs,dx);
    alignedCos=mac(dy,-sn,alignedCos);
    offsetX=mul(cornerX,alignedSin);
    offsetX=mac(alignedCos,cornerY,offsetX);
    offsetY=mul(-cornerX,alignedCos);
    offsetY=mac(alignedSin,cornerY,offsetY);
#else
    offsetX=mul(cs,cornerX);
    offsetX=mac(cornerY,-sn,offsetX);
    offsetY=mul(sn,cornerX);
    offsetY=mac(cornerY,cs,offsetY);
#endif
    view.x=mac(F(M,0x4c),offsetX,view.x);
    view.y=mac(F(M,0x6c),offsetY,view.y);
    view.z=add(view.z,corner.z);

    vec4 clip;
    for (int i=0;i<4;++i) {
        vec4 row=F4(Camera,16*i);
        float v=mul(row.x,view.x);
        v=mac(row.y,view.y,v);
        v=mac(row.z,view.z,v);
        clip[i]=mac(row.w,view.w,v);
    }

    vec4 atlasUv=vec4(uv.xy,uv.xy);
    if (has(flags,0x100u))
        atlasUv=atlasCoordinates(atlas,uv.xy,uint(age),t);

    if (has(flags,0x80u)) {
        float next,previous,blend;
        animationFrame(normalizedAge,next,previous,blend);
        vec4 a=animationSample(8u,vec2(next,B(0x3f400000u)));
        vec4 b=animationSample(9u,vec2(previous,B(0x3f400000u)));
        colour=mac(blend,sub(a,b),add(b,colour));
    }

    vec4 overrideColour=F4(M,0x120);
    if (overrideColour.w>0.0) colour.rgb=overrideColour.rgb;

    vec3 rgb;
    float alpha;
#ifdef VERTEX_LIGHTING
    vec3 offsetWorld;
    offsetWorld.x=mac(row0.x,F(M,0x4c)*offsetX,world.x);
    offsetWorld.x=mac(row0.y,F(M,0x6c)*offsetY,offsetWorld.x);
    offsetWorld.x=mac(row0.z,corner.z,offsetWorld.x);
    offsetWorld.y=mac(row1.x,F(M,0x4c)*offsetX,world.y);
    offsetWorld.y=mac(row1.y,F(M,0x6c)*offsetY,offsetWorld.y);
    offsetWorld.y=mac(row1.z,corner.z,offsetWorld.y);
    offsetWorld.z=mac(row2.x,F(M,0x4c)*offsetX,world.z);
    offsetWorld.z=mac(row2.y,F(M,0x6c)*offsetY,offsetWorld.z);
    offsetWorld.z=mac(row2.z,corner.z,offsetWorld.z);
    vec3 source=add(F3(M,0x100),sourceOrigin);
    vec3 originWorld=pointRows(row0,row1,row2,source);
    vec3 lightingDelta=sub(offsetWorld,originWorld);
    float direction=mul(F(Light,0x80),lightingDelta.x);
    direction=mac(F(Light,0x84),lightingDelta.y,direction);
    direction=mac(F(Light,0x88),lightingDelta.z,direction);
    float facing=mul(-inversesqrt(D3(lightingDelta,lightingDelta)),direction);
    float lighting=mac(F(M,0x7c),facing,1.0);
    float red=mul(F(V20,0x590),F(V20,0x580));
    red=mul(red,colour.r);
    float green=mul(F(V20,0x584),colour.g);
    green=mul(F(V20,0x594),green);
    float blue=mul(F(V20,0x588),colour.b);
    blue=mul(F(V20,0x598),blue);
    rgb=vec3(mul(red,lighting),mul(green,lighting),mul(blue,lighting));
    alpha=mul(F(M,0xec),colour.a);
    alpha=mul(F(V20,0x58c),alpha);
    alpha=mul(F(V20,0x59c),alpha);
#else
    rgb=mul(F3(V20,0x590),colour.rgb);
    alpha=mul(F(M,0xec),colour.a);
    alpha=mul(F(V20,0x59c),alpha);
#endif
#ifdef DISTANCE_ALPHA
    // s0 retains the initial 0xa0.x load while atlas mode is disabled.
    float inverseDistanceRange=1.0/F(M,0xa0);
    vec3 distanceDelta=sub(world,F3(M,0x110));
    float distanceToCamera=sqrt(D3(distanceDelta,distanceDelta));
    float fade=mul(distanceToCamera,inverseDistanceRange);
    fade=sat(mac(-F(M,0xdc),inverseDistanceRange,fade));
    alpha=mul(alpha,fade);
#endif

    float fog=0.0;
    if (has(flags,0x40000u))
        fog=sat(mac(F(V10,0x28),clip.w,F(V10,0x2c)));
    float biasedZ=mac(F(M,0x11c),clip.w,clip.z);
    gl_Position=vec4(clip.xy,biasedZ,clip.w);
    if (cb.c[213].z!=0u)
        gl_Position.z=mul(add(gl_Position.z,gl_Position.w),0.5);
    p0=vec4(rgb,alpha);
    p1=atlasUv;
    p2=vec4(0.0,0.0,clip.z,clip.w);
    p3=vec4(fog,0.0,0.0,0.0);
}
)";

// The translated pipeline's rectangle expansion (renderer.cpp expandShader for outputs 0-3), word
// for word: the corner whose edges are perpendicular, then the parallelogram's fourth vertex.
constexpr const char* kRectGs = R"(#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;
in gl_PerVertex { vec4 gl_Position; } gl_in[];
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) in vec4 iP0[];
layout(location = 0) out vec4 oP0;
layout(location = 1) in vec4 iP1[];
layout(location = 1) out vec4 oP1;
layout(location = 2) in vec4 iP2[];
layout(location = 2) out vec4 oP2;
layout(location = 3) in vec4 iP3[];
layout(location = 3) out vec4 oP3;
void emit(int i) {
    gl_Position = gl_in[i].gl_Position;
    oP0 = iP0[i];
    oP1 = iP1[i];
    oP2 = iP2[i];
    oP3 = iP3[i];
    EmitVertex();
}
void main() {
    vec2 p0 = gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;
    vec2 p1 = gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;
    vec2 p2 = gl_in[2].gl_Position.xy / gl_in[2].gl_Position.w;
    float d0 = abs(dot(p1 - p0, p2 - p0)), d1 = abs(dot(p0 - p1, p2 - p1)), d2 = abs(dot(p0 - p2, p1 - p2));
    int c = (d0 <= d1 && d0 <= d2) ? 0 : (d1 <= d2 ? 1 : 2);
    int a = (c + 1) % 3, b = (c + 2) % 3;
    emit(c); emit(a); emit(b);
    gl_Position = gl_in[a].gl_Position + gl_in[b].gl_Position - gl_in[c].gl_Position;
    oP0 = iP0[a] + iP0[b] - iP0[c];
    oP1 = iP1[a] + iP1[b] - iP1[c];
    oP2 = iP2[a] + iP2[b] - iP2[c];
    oP3 = iP3[a] + iP3[b] - iP3[c];
    EmitVertex();
    EndPrimitive();
}
)";

// The translated pipeline's quad expansion (expandShader with quads): corners 0, 1, 3, 2 as a strip.
constexpr const char* kQuadGs = R"(#version 450
layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;
in gl_PerVertex { vec4 gl_Position; } gl_in[];
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) in vec4 iP0[];
layout(location = 0) out vec4 oP0;
layout(location = 1) in vec4 iP1[];
layout(location = 1) out vec4 oP1;
layout(location = 2) in vec4 iP2[];
layout(location = 2) out vec4 oP2;
layout(location = 3) in vec4 iP3[];
layout(location = 3) out vec4 oP3;
void emit(int i) {
    gl_Position = gl_in[i].gl_Position;
    oP0 = iP0[i];
    oP1 = iP1[i];
    oP2 = iP2[i];
    oP3 = iP3[i];
    EmitVertex();
}
void main() {
    emit(0); emit(1); emit(3); emit(2);
    EndPrimitive();
}
)";

constexpr const char* kFs = R"(
layout(set=0,binding=3) uniform sampler2D image0;
layout(location=0) in vec4 a0;
layout(location=1) in vec4 a1;
// The PS's third interpolator consumes the VS's fog semantic, param3.
layout(location=3) in vec4 fogInput;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;

vec4 swz(vec4 v,uint s) {
    return vec4(v[s&3u],v[(s>>2)&3u],v[(s>>4)&3u],v[(s>>6)&3u]);
}
float halfRTZ(float x) {
    uint h=packHalf2x16(vec2(x,0.0))&0xffffu;
    float y=unpackHalf2x16(h).x;
    if (!isnan(x) && abs(y)>abs(x)) h-=1u;
    return unpackHalf2x16(h).x;
}
vec4 half4(vec4 v) {
    return vec4(halfRTZ(v.x),halfRTZ(v.y),halfRTZ(v.z),halfRTZ(v.w));
}
void main() {
    vec2 coordinate=mul(a1.xy,uintBitsToFloat(cb.c[214].xy));
    vec4 texel=texture(image0,coordinate);
    float test=mac(a0.w,texel.w,B(0xbb808081u));
    if (0.0>test) demote;

    uint flags=U(212,0);
    vec3 rgb=vec3(mul(texel.r,a0.r),mul(a0.g,texel.g),mul(a0.b,texel.b));
    float alpha=mul(a0.a,texel.a);
    if (has(flags,0x800000u)) {
        vec4 tint=F4(176,0x230);
        rgb=mac(tint.w,rgb,tint.rgb);
        rgb=mac(-tint.rgb,tint.w,rgb);
        alpha=mul(tint.w,alpha);
    }
    vec3 result=rgb;
    if (has(flags,0x400000u)) {
        float f=fogInput.x;
        float remaining=sub(1.0,f);
        result=mul(remaining,rgb);
        if (!has(flags,0x1000000u)) {
            vec3 fog=F3(172,0x10);
            result=vec3(mac(fog.r,f,rgb.r),
                        mac(f,fog.g,rgb.g),
                        mac(f,fog.b,rgb.b));
            result=mac(rgb,-f,result);
        }
    }
    float luminance=mul(B(0x3e991687u),result.r);
    luminance=mac(result.g,B(0x3f1645a2u),luminance);
    luminance=mac(result.b,B(0x3de978d5u),luminance);
    float coverage=mul(F(164,0x5c),alpha);
    coverage=mul(coverage,luminance);
    coverage=min(1.0,coverage);
    o0=swz(half4(vec4(result,alpha)),cb.c[213].x);
    o1=swz(half4(vec4(0.0,0.0,0.0,coverage)),cb.c[213].y);
}
)";

bool named(const std::string& name, const char* wanted) {
    if (name == wanted) return true;
    const std::string prefix = std::string(wanted) + " +";
    if (name.rfind(prefix, 0) != 0 || name.size() == prefix.size()) return false;
    for (size_t i = prefix.size(); i < name.size(); ++i)
        if (name[i] < '0' || name[i] > '9') return false;
    return true;
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
uint32_t formatBytes(uint32_t dfmt) {
    switch (dfmt) {
    case 1: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 10: return 4;
    case 11: case 12: return 8;
    case 13: return 12;
    case 14: return 16;
    default: return 0;
    }
}

bool decodable(const VertexBuffer& v) {
    if (!v.data.buffer || !v.stride || v.stride % 4 ||
        (uint64_t(v.data.byteOffset) + v.offset) % 4)
        return false;
    if (!formatComponents(v.dfmt)) return false;
    const bool wide = v.dfmt == 4 || v.dfmt == 11 || v.dfmt == 13 || v.dfmt == 14;
    if (wide) return v.nfmt == 7;
    const bool half = v.dfmt == 2 || v.dfmt == 5 || v.dfmt == 12;
    return v.nfmt == 0 || v.nfmt == 1 || (half && v.nfmt == 7);
}

// The decoder reads complete dwords, including packed formats. Require those
// reads to remain inside the actual descriptor range for every declared record.
bool bufferExtent(const VertexBuffer& v, uint32_t bytes) {
    if (!v.data.range || v.data.range == VK_WHOLE_SIZE) return false;
    const uint64_t first = uint64_t(v.data.byteOffset) + v.offset;
    const uint64_t extent = v.records
        ? first + uint64_t(v.records - 1) * v.stride + ((bytes + 3u) & ~3u)
        : first;
    return first <= UINT32_MAX && extent <= v.data.range &&
           extent <= uint64_t(UINT32_MAX) * 4;
}

class Particle final : public Pass {
public:
    const char* name() const override { return "particle"; }

    int claims(const std::string& psName) const override {
        return named(psName, "ps_ptc_xxx_x_x_x_x_x.pso") ? 1 : 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        if (role != 1) return false;
        int variant = -1;
        if (d.vsName)
            for (int i = 0; i < 5; ++i)
                if (named(*d.vsName, kVariants[i].vs)) variant = i;
        if (variant < 0) return decline("another vertex shader");
        const Variant& var = kVariants[variant];

        if (d.upscaler) return decline("an upscaler is on");
        if (d.pixelPositions) return decline("positions in pixels");
        if (d.depthBias) return decline("a polygon offset");
        // Rectangle lists (17: three corners each) as the translated pipeline draws them: its
        // geometry shader completes each rectangle (renderer.cpp's expandShader).
        // As the translated pipeline maps them: 1 points, 2 lines, 3 line strip, 4 triangles, 5 fan,
        // 6 strip, 17 rectangles (3 corners) and 19 quads (4 corners) through a geometry shader.
        switch (d.primitive) {
        case 1: case 2: case 3: case 4: case 5: case 6: case 17: case 19: break;
        default: {
            char why[48];
            std::snprintf(why, sizeof why, "primitive type %u", d.primitive);
            return decline(why);
        }
        }
        if (!d.count || ((d.primitive == 4 || d.primitive == 17) && d.count % 3) || (d.primitive == 19 && d.count % 4))
            return decline("an incomplete primitive list");
        if (d.colorCount < 1 || d.colorCount > 2)
            return decline("not one or two colour targets");
        for (uint32_t i = 0; i < d.colorCount; ++i)
            if (!d.color[i].view || d.color[i].format == VK_FORMAT_UNDEFINED)
                return decline("a missing colour attachment");
        if (d.depthStencil.depthBoundsTestEnable)
            return decline("depth bounds enabled");
        if (!d.depth.view &&
            (d.depthStencil.depthTestEnable || d.depthStencil.depthWriteEnable ||
             d.stencil || d.hasStencil))
            return decline("depth or stencil state without an attachment");
        if (d.depth.view && d.depth.format == VK_FORMAT_UNDEFINED)
            return decline("an undefined depth format");
        if (d.scissor[2] < 0 || d.scissor[3] < 0)
            return decline("a negative scissor extent");

        if (d.indexed) {
            uint32_t bytes = d.indexType == VK_INDEX_TYPE_UINT16 ? 2 :
                             d.indexType == VK_INDEX_TYPE_UINT32 ? 4 : 0;
            if (!bytes || !d.index.buffer ||
                (d.index.offset + d.index.byteOffset) % bytes)
                return decline("an unsupported index buffer");
            const uint64_t end = uint64_t(d.index.byteOffset) +
                                 (uint64_t(d.firstIndex) + d.count) * bytes;
            if (!d.index.range || d.index.range == VK_WHOLE_SIZE || end > d.index.range)
                return decline("an index range outside its buffer");
        }

        uint32_t c[kConstants][4] = {};
        uint64_t loaded[(kConstants * 4 + 63) / 64] = {};
        const auto readLoads = [&](std::span<const Load> loads) {
            for (const Load& l : loads) {
                const uint32_t* data = l.vs ? d.vsData : d.psData;
                const size_t size = l.vs ? d.vsSize : d.psSize;
                const auto& offset = l.vs ? d.vsOffset : d.psOffset;
                const auto& load = l.vs ? d.vsLoad : d.load;
                const int64_t at = offset ? offset(l.pc) : -1;
                std::array<uint32_t, 16> temporary{};
                const uint32_t* source = nullptr;
                if (data && at >= 0 && uint64_t(at) <= size &&
                    l.n <= size - size_t(at))
                    source = data + size_t(at);
                else if (load && load(l.pc, temporary.data(), l.n))
                    source = temporary.data();
                if (!source) return false;
                for (uint32_t j = 0; j < l.n; ++j) {
                    const uint32_t to = l.index * 4 + l.at + j;
                    const uint32_t value = source[j];
                    // Captures of untaken branches contain zeros. A later
                    // inactive copy must not erase a live overlapping load.
                    if (value || !(loaded[to / 64] >> (to % 64) & 1))
                        c[to / 4][to % 4] = value;
                    loaded[to / 64] |= uint64_t(1) << (to % 64);
                }
            }
            return true;
        };
        if (!readLoads(var.loads) || !readLoads(kPs))
            return decline("a scalar load window is missing");

        const uint32_t flags = c[var.flags][0];

        const VertexBuffer* slots[kFetches] = {};
        const VertexBuffer* palettes[kPalettes] = {};
        for (const VertexBuffer& v : d.vertex) {
            if (v.slot >= 0 && uint32_t(v.slot) < kFetches && v.components) {
                if (slots[v.slot]) return decline("duplicate fetch input");
                slots[v.slot] = &v;
            }
            if (v.slot < 0)
                for (uint32_t i = 0; i < kPalettes; ++i)
                    if (v.pc == var.palettePc[i]) {
                        if (palettes[i]) return decline("duplicate palette instruction input");
                        palettes[i] = &v;
                    }
        }
        // v4.xyz, v8.xy, v12.xyzw, v16.xyz, v20.xyzw,
        // v24.xyzw, v28.w, v32.w, v36.xyzw, v40.xyzw, v44.xyz.
        // Slot 8 is required to keep the supplied fetch layout unambiguous.
        constexpr uint32_t used[kFetches] = {3,2,4,3,4,4,4,4,4,4,3};
        for (uint32_t i = 0; i < kFetches; ++i) {
            const VertexBuffer* v = slots[i];
            if (!v || !decodable(*v) || v->components < used[i] ||
                formatComponents(v->dfmt) < used[i])
                return decline("an unsupported fetch input");
            for (uint32_t j = 0; j < used[i]; ++j)
                if (((v->dstSel >> (3 * j)) & 7) != 4 + j)
                    return decline("a fetch component selection it does not decode");
            if (!bufferExtent(*v, formatBytes(v->dfmt)))
                return decline("a fetch address layout outside its range");
            c[cFetch + i][0] = uint32_t((uint64_t(v->data.byteOffset) + v->offset) / 4);
            c[cFetch + i][1] = v->stride / 4;
            c[cFetch + i][2] = v->dfmt | v->nfmt << 4 | (v->indexVgpr ? 0x100u : 0u);
            c[cFetch + i][3] = v->records;
        }

        const bool paletteEnabled = (flags & 16u) != 0;
        for (uint32_t i = 0; i < kPalettes; ++i) {
            if (!paletteEnabled) {
                palettes[i] = slots[0]; // valid unused descriptor
                continue;
            }
            const VertexBuffer* v = palettes[i];
            if (!v || !v->data.buffer || v->stride != 16 || v->offset ||
                v->data.byteOffset % 4 || v->dfmt != 14 || v->nfmt != 7 ||
                v->dstSel != 0xfac || !bufferExtent(*v, 16))
                return decline("a palette layout it does not read exactly");
            c[cPalette + i][0] = v->data.byteOffset / 4;
            c[cPalette + i][1] = 4;
            c[cPalette + i][2] = v->records;
        }

        const Texture* texture = nullptr;
        for (const Texture& t : d.textures)
            if (t.pc == 0x24) {
                if (texture) return decline("duplicate pixel sample PC");
                texture = &t;
            }
        if (!texture || !texture->view || !texture->sampler)
            return decline("no pixel texture and sampler at PC 0x24");
        VkDescriptorImageInfo image{
            texture->sampler, texture->view, VK_IMAGE_LAYOUT_GENERAL};
        std::memcpy(c[cTexScale], texture->scale, sizeof texture->scale);

        std::array<VkDescriptorImageInfo, kVsTextures> vsImages{};
        constexpr uint32_t sampleFlags[kVsTextures] = {
            0x20,0x20,0x20,0x20,0x40,0x40,0x200,0x200,0x80,0x80};
        for (uint32_t i = 0; i < kVsTextures; ++i) {
            vsImages[i] = image; // valid unused descriptor
            if (!(flags & sampleFlags[i])) continue;
            const Texture* sample = nullptr;
            for (const Texture& t : d.vsTextures)
                if (t.pc == var.texturePc[i]) {
                    if (sample) return decline("duplicate vertex sample PC");
                    sample = &t;
                }
            if (!sample || !sample->view || !sample->sampler)
                return decline("a vertex texture or sampler is missing at its sample PC");
            vsImages[i] = {
                sample->sampler, sample->view, VK_IMAGE_LAYOUT_GENERAL};
            std::memcpy(c[cVsTexScale + i], sample->scale, sizeof sample->scale);
        }

        for (uint32_t i = 0; i < d.colorCount; ++i) {
            int32_t sw[4];
            swizzle(d.color[i], sw);
            for (int j = 0; j < 4; ++j)
                if (sw[j] < 0 || sw[j] > 3)
                    return decline("an unsupported colour export swizzle");
            c[cOut][i] = uint32_t(sw[0] | sw[1] << 2 | sw[2] << 4 | sw[3] << 6);
        }
        c[cOut][2] = d.clipZHalf ? 1 : 0;

        const VkPipeline pipe = pipeline(gpu, variant, d);
        if (!pipe) return decline("no pipeline");
        Buffer constants;
        if (!gpu.upload(c, sizeof c, constants)) return decline("no room for constants");

        std::array<VkDescriptorBufferInfo, kFetches + kPalettes + 1> buffers{};
        for (uint32_t i = 0; i < kFetches; ++i)
            buffers[i] = {slots[i]->data.buffer, slots[i]->data.offset, slots[i]->data.range};
        for (uint32_t i = 0; i < kPalettes; ++i)
            buffers[kFetches + i] = {
                palettes[i]->data.buffer, palettes[i]->data.offset, palettes[i]->data.range};
        buffers.back() = {constants.buffer, constants.offset, constants.range};

        const VkViewport viewport = d.vkViewport;
        const VkRect2D scissor{{d.scissor[0], d.scissor[1]},
                              {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        std::array<uint32_t, 6> stencil{};
        for (int i = 0; i < 2; ++i) {
            stencil[i * 3] = d.stencilRef[i];
            stencil[i * 3 + 1] = d.stencilCompare[i];
            stencil[i * 3 + 2] = d.stencilWrite[i];
        }
        char description[256];
        std::snprintf(description, sizeof description,
                      "vs %s; flags 0x%x; primitive %u; %s %u; targets %u; "
                      "swizzles 0x%x 0x%x; clipZHalf %u; palette %u",
                      var.vs, flags, d.primitive, d.indexed ? "indices" : "vertices",
                      d.count, d.colorCount, c[cOut][0], c[cOut][1], c[cOut][2],
                      unsigned(paletteEnabled));
        info_ = description;
        if (!logged_[variant].exchange(true))
            std::fprintf(stderr, "[native] particle: drawing %s natively\n", var.vs);

        gpu.beginRendering(d.color, d.colorCount, d.depth, d.hasStencil);
        gpu.record([pipe, layout = layout_, buffers, image, vsImages, viewport, scissor, stencil,
                    bc = std::to_array(d.blendConstants), indexed = d.indexed,
                    index = d.index, indexType = d.indexType, firstIndex = d.firstIndex,
                    count = d.count, instances = d.instances, pushSet = gpu.pushDescriptorSet()](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdSetBlendConstants(cmd, bc.data());
            for (int i = 0; i < 2; ++i) {
                const VkStencilFaceFlags face =
                    i ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
                vkCmdSetStencilReference(cmd, face, stencil[i * 3]);
                vkCmdSetStencilCompareMask(cmd, face, stencil[i * 3 + 1]);
                vkCmdSetStencilWriteMask(cmd, face, stencil[i * 3 + 2]);
            }
            VkWriteDescriptorSet writes[5]{};
            for (uint32_t i = 0; i < 5; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            }
            writes[0].descriptorCount = kFetches;
            writes[0].pBufferInfo = buffers.data();
            writes[1].descriptorCount = kPalettes;
            writes[1].pBufferInfo = buffers.data() + kFetches;
            writes[2].pBufferInfo = &buffers.back();
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].pImageInfo = &image;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[4].descriptorCount = kVsTextures;
            writes[4].pImageInfo = vsImages.data();
            pushSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 5, writes);
            if (indexed) {
                vkCmdBindIndexBuffer(cmd, index.buffer, index.offset + index.byteOffset, indexType);
                vkCmdDrawIndexed(cmd, count, instances, firstIndex, 0, 0);
            } else {
                vkCmdDraw(cmd, count, instances, 0, 0);
            }
        });
        return true;
    }

    std::string lastDrawInfo() const override { return info_; }

private:
    bool decline(const char* why) {
        std::lock_guard lock(m_);
        if (declined_.insert(why).second && declined_.size() <= 32)
            std::fprintf(stderr,
                         "[native] particle: a draw left to the translated renderer (%s)\n", why);
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int variant, const Draw& d) {
        const auto& z = d.depthStencil;
        const bool stencil = d.stencil && d.hasStencil && z.stencilTestEnable;
        // Keep complete enumerants in the key, rather than packing blend
        // factors into fields that could overlap as Vulkan adds enumerants.
        std::vector<uint32_t> key{
            uint32_t(variant), d.colorCount, d.writeMask,
            uint32_t(d.depth.view ? d.depth.format : VK_FORMAT_UNDEFINED),
            uint32_t(d.hasStencil), z.depthTestEnable, z.depthWriteEnable,
            uint32_t(z.depthCompareOp), uint32_t(d.cull), uint32_t(d.frontFace),
            uint32_t(stencil)};
        for (uint32_t i = 0; i < d.colorCount; ++i) {
            const auto& b = d.blendState[i];
            key.insert(key.end(), {
                uint32_t(d.color[i].format), b.blendEnable,
                uint32_t(b.srcColorBlendFactor), uint32_t(b.dstColorBlendFactor),
                uint32_t(b.colorBlendOp), uint32_t(b.srcAlphaBlendFactor),
                uint32_t(b.dstAlphaBlendFactor), uint32_t(b.alphaBlendOp)});
        }
        if (stencil)
            for (int i = 0; i < 2; ++i) {
                const auto& s = i ? z.back : z.front;
                key.insert(key.end(), {
                    uint32_t(s.failOp), uint32_t(s.passOp),
                    uint32_t(s.depthFailOp), uint32_t(s.compareOp)});
            }

        key.push_back(d.primitive);
        std::lock_guard lock(m_);
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice device = gpu.device();
        if (!setLayout_) {
            const VkDescriptorSetLayoutBinding bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFetches, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kPalettes, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kVsTextures,
                 VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo dl{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = 5;
            dl.pBindings = bindings;
            if (vkCreateDescriptorSetLayout(device, &dl, nullptr, &setLayout_) != VK_SUCCESS)
                return out = VK_NULL_HANDLE;
        }
        if (!layout_) {
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayout_;
            if (vkCreatePipelineLayout(device, &pl, nullptr, &layout_) != VK_SUCCESS)
                return out = VK_NULL_HANDLE;
        }

        const std::string header = kHeader;
        const VkShaderModule vs = gpu.shader(
            header + kVariants[variant].defines + kVs, Gpu::Stage::Vertex);
        const std::string fragmentHeader =
            "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\n" +
            header.substr(header.find('\n') + 1);
        const VkShaderModule fs = gpu.shader(fragmentHeader + kFs, Gpu::Stage::Fragment);
        if (!vs || !fs) return out = VK_NULL_HANDLE;
        const bool rects = d.primitive == 17 || d.primitive == 19;
        const VkShaderModule gs = rects ? gpu.shader(d.primitive == 19 ? kQuadGs : kRectGs, Gpu::Stage::Geometry) : VK_NULL_HANDLE;
        if (rects && !gs) return out = VK_NULL_HANDLE;

        VkPipelineShaderStageCreateInfo stages[3]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo vi{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        switch (d.primitive) {
        case 1: ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case 2: ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 3: ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case 5: ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case 6: ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 19: ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; break;
        default: ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        }
        VkPipelineViewportStateCreateInfo vp{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = d.cull;
        rs.frontFace = d.frontFace;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = z.depthTestEnable;
        ds.depthWriteEnable = z.depthWriteEnable;
        ds.depthCompareOp = z.depthCompareOp;
        if (stencil) {
            ds.stencilTestEnable = VK_TRUE;
            ds.front = z.front;
            ds.back = z.back;
        }
        std::array<VkPipelineColorBlendAttachmentState, 2> attachments{};
        std::array<VkFormat, 2> formats{};
        for (uint32_t i = 0; i < d.colorCount; ++i) {
            attachments[i] = d.blendState[i];
            attachments[i].colorWriteMask = (d.writeMask >> (4 * i)) & 0xf;
            formats[i] = d.color[i].format;
        }
        VkPipelineColorBlendStateCreateInfo blend{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = d.colorCount;
        blend.pAttachments = attachments.data();
        const VkDynamicState states[] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
        VkPipelineDynamicStateCreateInfo dynamic{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 6;
        dynamic.pDynamicStates = states;
        VkPipelineRenderingCreateInfo rendering{
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = d.colorCount;
        rendering.pColorAttachmentFormats = formats.data();
        rendering.depthAttachmentFormat =
            d.depth.view ? d.depth.format : VK_FORMAT_UNDEFINED;
        rendering.stencilAttachmentFormat =
            d.depth.view && d.hasStencil ? d.depth.format : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.pNext = &rendering;
        pi.stageCount = 2;
        if (rects) {
            stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_GEOMETRY_BIT, gs, "main", nullptr};
            pi.stageCount = 3;
        }
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &blend;
        pi.pDynamicState = &dynamic;
        pi.layout = layout_;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &out) != VK_SUCCESS)
            out = VK_NULL_HANDLE;
        return out;
    }

    std::mutex m_;
    std::set<std::string> declined_;
    std::atomic<bool> logged_[5]{};
    std::map<std::vector<uint32_t>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    static inline thread_local std::string info_;
};

} // namespace

Pass* particlePass() {
    static Particle pass;
    return &pass;
}

} // namespace host::native
