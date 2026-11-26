// Bloom/downsampling, from dis/post's shipped GCN programs. Fetch the quad rather than
// assuming its UVs cover [0,1]: glow_pass1 also remaps each colour into an atlas.
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

enum Role { kDown2 = 1, kDown4, kLumi, kBlur, kCompose };
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
};

// c0: VS position transform; c1..3: VS UV constants; c4..9: PS loads;
// c10..27: texture scales, one per MIMG instruction, even for the same T#.
// Luminance uses c4=0x44, c5..6=0x90 (when enabled), c7.x=0x40 flags.
using Constants = std::array<std::array<uint32_t, 4>, 28>;
static_assert(sizeof(Constants) == 28*16);
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
    vec2 uv=fetch(1).xy;
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
#define sampleAt(i,uv) texture(tex[i],(uv)*c[10+(i)].xy)
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
        for (int i=1;i<=kCompose;++i) if (named(ps,kRoles[i].ps)) return i;
        return 0;
    }
    bool draw(int role,const Draw& d,Gpu& gpu) override {
        if (role<kDown2 || role>kCompose) return false;
        // Codex's port, unmatched yet: drawn only when LPN_NATIVE_COMPARE=post compares it.
        static const bool compare = [] {
            const char* e = std::getenv("LPN_NATIVE_COMPARE");
            return e && std::string(e) == "post";
        }();
        if (!compare) return decline(role,"not yet matched (drawn in compare mode only)");
        if (!d.vsName || (role==kLumi ? *d.vsName!=kRoles[role].vs : !named(*d.vsName,kRoles[role].vs)))
            return decline(role,"another vertex shader");
        if (d.colorCount!=1 || !d.color[0].view || d.color[0].format==VK_FORMAT_UNDEFINED ||
            !d.color[0].width || !d.color[0].height || d.color[0].comps!=4)
            return decline(role,"not one four-component colour target");
        if (d.color[0].format!=VK_FORMAT_R8G8B8A8_UNORM)
            return decline(role,"unexpected colour format (expected RGBA8 UNORM)");
        if ((d.writeMask&~0xfu) || !(d.writeMask&0xfu))
            return decline(role,"unexpected target write mask");
        const auto& blend=d.blendState[0];
        const bool additive=blend.blendEnable;
        // Capture-5's composition adds RGB and alpha (0x61010101); other roles replace.
        if (additive) {
            if (role!=kCompose || blend.srcColorBlendFactor!=VK_BLEND_FACTOR_ONE ||
                blend.dstColorBlendFactor!=VK_BLEND_FACTOR_ONE || blend.colorBlendOp!=VK_BLEND_OP_ADD ||
                blend.srcAlphaBlendFactor!=VK_BLEND_FACTOR_ONE || blend.dstAlphaBlendFactor!=VK_BLEND_FACTOR_ONE ||
                blend.alphaBlendOp!=VK_BLEND_OP_ADD) return decline(role,"unexpected blending");
        } else if (d.blend) return decline(role,"inconsistent blend state");
        if (d.depthTest || d.depthWrite || d.stencil || d.depthStencil.depthTestEnable ||
            d.depthStencil.depthWriteEnable || d.depthStencil.stencilTestEnable ||
            d.depthStencil.depthBoundsTestEnable) return decline(role,"depth or stencil");
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
        for (int i=0;i<2;++i) {
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

        Constants constants{};
        auto load=[&](bool vs,uint32_t pc,uint32_t n,int at) {
            uint32_t words[16]{};
            const auto& fn=vs?d.vsLoad:d.load;
            if (!fn || !fn(pc,words,n)) return false;
            std::memcpy(reinterpret_cast<unsigned char*>(constants.data())+at*16,words,n*4);
            return true;
        };
        if (!load(true,role==kLumi?0x10:0xc,4,0)) return decline(role,"missing VS position scalar load");
        uint32_t wantedTextures=0;
        if (role==kLumi) {
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
            std::memcpy(constants[10+n].data(),t->scale,sizeof t->scale);
            ++n;
        }
        VkPipeline pipe=pipeline(gpu,role,d.color[0].format,d.writeMask&0xf,n,additive);
        if (!pipe) return decline(role,"pipeline creation failed");
        Buffer cb;
        if (!gpu.upload(constants.data(),sizeof constants,cb)) return decline(role,"constant upload failed");
        buffers[2]={cb.buffer,cb.offset,cb.range};
        // Gpu::upload currently returns byteOffset 0; GLSL c[] starts at the descriptor base.
        if (cb.byteOffset) return decline(role,"nonzero constant upload byte offset");
        if (kCompare) {
            char b[256];
            std::snprintf(b,sizeof b,"vs %s; target %ux%u format %d mask 0x%x; %u MIMG samples; clipZHalf %u",
                          d.vsName->c_str(),d.color[0].width,d.color[0].height,int(d.color[0].format),
                          d.writeMask,n,push.clipZHalf);
            info_=b;
            for (int i=0;i<10;++i) {
                float f[4]; std::memcpy(f,constants[i].data(),sizeof f);
                std::snprintf(b,sizeof b,"\n    c%d: %g %g %g %g",i,f[0],f[1],f[2],f[3]);
                info_+=b;
            }
        }
        const VkRect2D sc{{d.scissor[0],d.scissor[1]},{uint32_t(d.scissor[2]),uint32_t(d.scissor[3])}};
        gpu.beginRendering(d.color,1,Target{},false);
        gpu.record([pipe,layout=layouts_[role],push,buffers,images,n,vp=d.vkViewport,sc,
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
    bool decline(int role,const char* why) {
        std::lock_guard lock(m_);
        if (declined_[role].insert(why).second)
            std::fprintf(stderr,"[native] post: %s left to the translated renderer (%s)\n",kRoles[role].ps,why);
        return false;
    }
    VkPipeline pipeline(Gpu& gpu,int role,VkFormat format,uint32_t mask,uint32_t textures,bool additive) {
        std::lock_guard lock(m_);
        const auto key=std::make_tuple(role,int(format),mask,textures,additive);
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
        const VkShaderModule vs=gpu.shader(header+kVs,Gpu::Stage::Vertex);
        const VkShaderModule fs=gpu.shader(header+kFs,Gpu::Stage::Fragment);
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
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask=mask;
        attachment.blendEnable=additive;
        attachment.srcColorBlendFactor=attachment.dstColorBlendFactor=VK_BLEND_FACTOR_ONE;
        attachment.srcAlphaBlendFactor=attachment.dstAlphaBlendFactor=VK_BLEND_FACTOR_ONE;
        attachment.colorBlendOp=attachment.alphaBlendOp=VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount=1; cb.pAttachments=&attachment;
        const VkDynamicState states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount=2; dy.pDynamicStates=states;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount=1; rendering.pColorAttachmentFormats=&format;
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.pNext=&rendering; pi.stageCount=3; pi.pStages=stages;
        pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia; pi.pViewportState=&vp;
        pi.pRasterizationState=&rs; pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb; pi.pDynamicState=&dy; pi.layout=layouts_[role];
        if (vkCreateGraphicsPipelines(gpu.device(),VK_NULL_HANDLE,1,&pi,nullptr,&out)!=VK_SUCCESS) out=VK_NULL_HANDLE;
        return out;
    }
    std::mutex m_;
    std::map<std::tuple<int,int,uint32_t,uint32_t,bool>,VkPipeline> pipes_;
    VkDescriptorSetLayout sets_[6]{};
    VkPipelineLayout layouts_[6]{};
    std::set<std::string> declined_[6];
    static inline const bool kCompare=std::getenv("LPN_NATIVE_COMPARE")!=nullptr;
    static inline thread_local std::string info_;
};

} // namespace

Pass* postPass() {
    static Post pass;
    return &pass;
}

} // namespace host::native
