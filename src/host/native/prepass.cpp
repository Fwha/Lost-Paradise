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

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace host::native {

namespace {

enum Role { kOpaque = 1, kAlphaTest = 2 };

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
layout(location = 0) out vec3 n;
#ifdef ALPHA_TEST
layout(location = 1) out vec2 t;
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
    m = vec3(mac(2.0078431, m.x, -1.0), mac(2.0078431, m.y, -1.0), mac(2.0078431, m.z, -1.0));
    vec3 o = vec3(dot3(p.world[0], m), dot3(p.world[1], m), dot3(p.world[2], m));
    n = vec3(dot3(p.rot[0], o), dot3(p.rot[1], o), dot3(p.rot[2], o));
#ifdef ALPHA_TEST
    t = fetch(2u, p.uv).xy;
#endif
}
)";

constexpr const char* kFs = R"(
layout(location = 0) in vec3 n;
#ifdef ALPHA_TEST
layout(location = 1) in vec2 t;
layout(set = 0, binding = 3) uniform sampler2D mask0;
layout(set = 0, binding = 4) uniform sampler2D mask1;
#endif
layout(location = 0) out vec4 color;
void main() {
#ifdef ALPHA_TEST
    if (0.0 > texture(mask1, t).w * texture(mask0, t).z - p.off.w) discard;
#endif
    const float k = 0.49803922;
    vec3 e = n * k + k;
    // v_cvt_pkrtz_f16_f32 before the compressed export, as the translator rounds it.
    vec4 v = vec4(unpackHalf2x16(packHalf2x16(e.xy)), unpackHalf2x16(packHalf2x16(vec2(e.z, 0.0))));
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
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        const bool alpha = role == kAlphaTest;
        const char* vs = alpha ? "vs_normal_rigid_pt.vso" : "vs_normal_rigid.vso";
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
        if (!logged_[role]) {
            logged_[role] = true;
            std::fprintf(stderr, "[native] prepass: drawing %s natively (target 0x%" PRIx64 " %ux%u, depth 0x%" PRIx64 ")\n",
                         alpha ? "ps_normal_pt.pso" : "ps_normal.pso", d.color[0].address, d.color[0].width, d.color[0].height,
                         d.depth.address);
        }
        gpu.beginRendering(d.color, 1, d.depth, d.hasStencil);
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Buffer index = d.index;
        gpu.record([pipe, layout = layouts_[role], push, bufs, images, alpha, vp, sc, index, indexed = d.indexed,
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
                vkCmdDrawIndexed(c, count, 1, 0, 0, 0);
            } else {
                vkCmdDraw(c, count, 1, 0, 0);
            }
        });
        return true;
    }

private:
    // Logs each reason once.
    bool decline(int role, const char* why) {
        if (declined_[role].insert(why).second && declined_[role].size() <= 16)
            std::fprintf(stderr, "[native] prepass: a %s draw left to the translated renderer (%s)\n",
                         role == kAlphaTest ? "ps_normal_pt" : "ps_normal", why);
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        const VkPipelineDepthStencilStateCreateInfo& z = d.depthStencil;
        const auto key = std::make_tuple(role, int(d.color[0].format), int(d.writeMask & 0xF), int(d.depth.format),
                                         bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable, int(z.depthCompareOp),
                                         int(d.cull), int(d.frontFace));
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        const bool alpha = role == kAlphaTest;
        if (!layouts_[role]) {
            const VkDescriptorSetLayoutBinding b[5] = {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = alpha ? 5 : 2;
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
        const std::string defs = alpha ? "#define ALPHA_TEST\n" : "";
        const VkShaderModule vsm = gpu.shader(std::string(kHeader) + defs + kVs, Gpu::Stage::Vertex);
        const VkShaderModule fsm = gpu.shader(std::string(kHeader) + defs + kFs, Gpu::Stage::Fragment);
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
    VkDescriptorSetLayout setLayouts_[3] = {};
    VkPipelineLayout layouts_[3] = {};
    bool logged_[3] = {};
    std::set<std::string> declined_[3];
};

} // namespace

Pass* prepassPass() {
    static Prepass pass;
    return &pass;
}

} // namespace host::native
