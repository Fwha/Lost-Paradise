// The shadow maps' casters (vs_shadowmap_normal[_pt].vso with ps_shadowmap[_pt][fp32][bias].pso:
// about 500 draws a frame into the sun's three 1024x1024 cascades and the near maps), rebuilt
// from the disassembly of the four shaders.
//
// Vertex: the fetch shader gives the position (and, for _pt, the texture coordinates from a
// second V#). The position goes through the object's 3x4 world matrix (constants at 0x0 of the
// buffer in s[8:11]) plus an offset (0x170 of s[4:7]), minus the camera position (0x220 of the
// view's buffer at 0x0 of the table in s[12:13]) unless the object's flags (0x0 of the table's
// 0x10 buffer) have 0x80000 without 0x100000, then the view rotation (rows at 0x40, 0x50, 0x60
// of the view's buffer) and the light's projection (rows at 0x0). Clip z is clamped at 0;
// z and w go to the pixel shader. All of it is folded into one matrix here, on the CPU.
//
// Pixel: depth = z (z/w for _pt) plus a slope bias, the larger of the quad's x and y differences
// times a scale (no scale for _pt), capped (constants at 0x300 of the pixel shader's buffer;
// 0.01 for _pt), clamped to [0, 1]. _pt first drops pixels whose texture alpha is below 0.5.
#include "host/native/native.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>

namespace host::native {

namespace {

enum Role { kOpaque = 1, kAlphaTest = 2 };

struct Push {
    float m[16];      // clip = m * (position, 1), column major
    uint32_t pos[4];  // first word, stride in words, dfmt
    uint32_t uv[4];
    float bias[2];    // scale, cap
    float uvScale[2]; // the texture's coordinate scale (native::Texture::scale)
};
static_assert(sizeof(Push) == 112);

constexpr const char* kHeader = R"(#version 450
layout(push_constant) uniform P {
    mat4 m;
    uvec4 pos;
    uvec4 uv;
    vec4 k; // bias scale, bias cap, texture scale
} p;
)";

constexpr const char* kVs = R"(
layout(set = 0, binding = 0, std430) readonly buffer V0 { uint w[]; } v0;
#ifdef ALPHA_TEST
layout(set = 0, binding = 1, std430) readonly buffer V1 { uint w[]; } v1;
#endif
layout(location = 0) out vec4 o; // z, w, u, v
vec3 position(uint s) {
    if (p.pos.z == 12u) return vec3(unpackHalf2x16(v0.w[s]), unpackHalf2x16(v0.w[s + 1u]).x);
    return vec3(uintBitsToFloat(v0.w[s]), uintBitsToFloat(v0.w[s + 1u]), uintBitsToFloat(v0.w[s + 2u]));
}
void main() {
    vec4 c = p.m * vec4(position(p.pos.x + uint(gl_VertexIndex) * p.pos.y), 1.0);
    gl_Position = vec4(c.x, c.y, max(0.0, c.z), c.w);
    vec2 uv = vec2(0.0);
#ifdef ALPHA_TEST
    uint s = p.uv.x + uint(gl_VertexIndex) * p.uv.y;
    uv = p.uv.z == 5u ? unpackHalf2x16(v1.w[s]) : vec2(uintBitsToFloat(v1.w[s]), uintBitsToFloat(v1.w[s + 1u]));
#endif
    o = vec4(c.z, c.w, uv);
}
)";

constexpr const char* kFs = R"(
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_EXT_demote_to_helper_invocation : require
layout(location = 0) in vec4 i;
#ifdef ALPHA_TEST
layout(set = 0, binding = 2) uniform sampler2D alpha;
#endif
// The larger of the differences to the quad's pixel 1 (x) and 2 (y) from its pixel 0.
float slope(float z) {
    float q0 = subgroupQuadBroadcast(z, 0u);
    return max(abs(subgroupQuadBroadcast(z, 1u) - q0), abs(subgroupQuadBroadcast(z, 2u) - q0));
}
void main() {
#ifdef ALPHA_TEST
    if (0.5 > texture(alpha, i.zw * p.k.zw).w) demote; // kept for the quad's differences
    float rw = 1.0 / i.y;
    gl_FragDepth = clamp(min(p.k.y, slope(i.x * rw)) + rw * i.x, 0.0, 1.0);
#else
    gl_FragDepth = clamp(i.x + min(p.k.y, p.k.x * slope(i.x)), 0.0, 1.0);
#endif
}
)";

// Where the shaders' scalar loads are (offsets from the vertex shader's start).
struct Loads {
    uint32_t offset, world, world2, flags, camera, rotation, rotation2, projection;
};
constexpr Loads kOpaqueLoads{0x10, 0x14, 0x18, 0x20, 0xa4, 0xc4, 0xcc, 0xc8};
constexpr Loads kAlphaLoads{0x10, 0x14, 0x18, 0x20, 0x9c, 0xb8, 0xbc, 0xc0};

uint32_t components(uint32_t dfmt) {
    switch (dfmt) {
    case 5: case 11: return 2;  // 16_16, 32_32
    case 13: return 3;          // 32_32_32
    case 12: case 14: return 4; // 16_16_16_16, 32_32_32_32
    default: return 0;
    }
}

class Shadow final : public Pass {
public:
    const char* name() const override { return "shadow"; }

    int claims(const std::string& psName) const override {
        if (psName == "ps_shadowmap[fp32][bias].pso") return kOpaque;
        if (psName == "ps_shadowmap_pt[fp32][bias].pso") return kAlphaTest;
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        const bool alpha = role == kAlphaTest;
        const char* vs = alpha ? "vs_shadowmap_normal_pt.vso" : "vs_shadowmap_normal.vso";
        if (!d.vsName || *d.vsName != vs) return decline(role, "another vertex shader");
        if (d.colorCount || !d.depth.view) return decline(role, "colour targets or no depth buffer");
        if (d.blend || d.stencil) return decline(role, "blending or stencil");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        // The vertex buffers: the position (3 or 4 components) and, for _pt, the coordinates (2).
        // The fetch slots: the position into v4 (slot 0) and, for _pt, the coordinates into v8.
        const VertexBuffer *pos = nullptr, *uv = nullptr;
        for (const VertexBuffer& v : d.vertex) {
            if (v.slot != 0 && v.slot != 1) continue;
            if (!v.components) continue;
            const uint32_t n = components(v.dfmt);
            if (v.nfmt != 7 || !n || v.stride % 4 || (v.data.byteOffset + v.offset) % 4 || v.dstSel != 0xFAC || v.indexVgpr != 0)
                return decline(role, "a vertex format it does not read");
            (v.slot == 0 ? pos : uv) = &v;
        }
        if (pos && components(pos->dfmt) < 3) return decline(role, "a position of fewer than 3 components");
        if (!pos || (alpha && !uv)) return decline(role, "no position or coordinates");
        Push push{};
        if (!matrix(alpha ? kAlphaLoads : kOpaqueLoads, d, push.m)) return decline(role, "a constant is missing");
        push.pos[0] = (pos->data.byteOffset + pos->offset) / 4, push.pos[1] = pos->stride / 4, push.pos[2] = pos->dfmt;
        VkDescriptorBufferInfo bufs[2] = {{pos->data.buffer, pos->data.offset, pos->data.range}, {}};
        VkDescriptorImageInfo image{};
        if (alpha) {
            push.uv[0] = (uv->data.byteOffset + uv->offset) / 4, push.uv[1] = uv->stride / 4, push.uv[2] = uv->dfmt;
            bufs[1] = {uv->data.buffer, uv->data.offset, uv->data.range};
            const Texture* t = nullptr;
            for (const Texture& x : d.textures)
                if (x.pc == 0x24) t = &x;
            if (!t || !t->view || !t->sampler) return decline(role, "no alpha texture");
            image = {t->sampler, t->view, VK_IMAGE_LAYOUT_GENERAL};
            push.uvScale[0] = t->scale[0], push.uvScale[1] = t->scale[1];
            push.bias[0] = 1, push.bias[1] = 0.01f;
        } else {
            uint32_t w[2];
            if (!d.load(0xc, w, 2)) return decline(role, "no bias constants");
            std::memcpy(push.bias, w, 8);
        }
        VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role]) {
            logged_[role] = true;
            std::fprintf(stderr, "[native] shadow: drawing %s natively (depth 0x%" PRIx64 " %ux%u)\n",
                         alpha ? "ps_shadowmap_pt[fp32][bias].pso" : "ps_shadowmap[fp32][bias].pso", d.depth.address,
                         d.depth.width, d.depth.height);
        }
        gpu.beginRendering(nullptr, 0, d.depth, d.hasStencil);
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Buffer index = d.index;
        gpu.record([pipe, layout = layouts_[role], push, bufs, image, alpha, vp, sc, index, indexed = d.indexed,
                    indexType = d.indexType, count = d.count, pushSet = gpu.pushDescriptorSet()](VkCommandBuffer c) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(c, 0, 1, &vp);
            vkCmdSetScissor(c, 0, 1, &sc);
            VkWriteDescriptorSet w[3]{};
            const uint32_t n = alpha ? 3 : 1;
            for (uint32_t i = 0; i < n; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                if (i < 2) w[i].pBufferInfo = &bufs[i];
                else w[i].pImageInfo = &image;
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
    bool decline(int role, const char* why) {
        if (!declined_[role]) {
            declined_[role] = true;
            std::fprintf(stderr, "[native] shadow: a %s draw left to the translated renderer (%s)\n",
                         role == kAlphaTest ? "ps_shadowmap_pt" : "ps_shadowmap", why);
        }
        return false;
    }

    // projection * rotation * (world + offset - camera), from the vertex shader's loads.
    static bool matrix(const Loads& l, const Draw& d, float out[16]) {
        float off[4], world[12], rot[12], proj[16];
        uint32_t flags;
        auto get = [&](uint32_t pc, float* v, uint32_t n) { return d.vsLoad(pc, reinterpret_cast<uint32_t*>(v), n); };
        if (!get(l.offset, off, 4) || !get(l.world, world, 8) || !get(l.world2, world + 8, 4) || !d.vsLoad(l.flags, &flags, 1) ||
            !get(l.rotation, rot, 8) || !get(l.rotation2, rot + 8, 4) || !get(l.projection, proj, 16))
            return false;
        float cam[4] = {};
        const bool keepCamera = (flags & 0x80000) && !(flags & 0x100000);
        if (!keepCamera && !get(l.camera, cam, 4)) return false;
        // a: the object to the view's axes (3x4, rows), then p * [a; 0 0 0 1].
        double a[3][4];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) {
                double s = 0;
                for (int k = 0; k < 3; ++k) {
                    const double w = c < 3 ? world[k * 4 + c] : double(world[k * 4 + 3]) + off[k] - cam[k];
                    s += double(rot[r * 4 + k]) * w;
                }
                a[r][c] = s;
            }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                double s = c == 3 ? proj[r * 4 + 3] : 0;
                for (int k = 0; k < 3; ++k) s += double(proj[r * 4 + k]) * a[k][c];
                out[c * 4 + r] = float(s);
            }
        return true;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        const VkPipelineDepthStencilStateCreateInfo& z = d.depthStencil;
        const auto key = std::make_tuple(role, int(d.depth.format), bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable,
                                         int(z.depthCompareOp), int(d.cull), int(d.frontFace));
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        const bool alpha = role == kAlphaTest;
        if (!layouts_[role]) {
            VkDescriptorSetLayoutBinding b[3] = {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = alpha ? 3 : 1;
            dl.pBindings = b;
            if (vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
            const VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayouts_[role];
            pl.pushConstantRangeCount = 1;
            pl.pPushConstantRanges = &pr;
            if (vkCreatePipelineLayout(dev, &pl, nullptr, &layouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
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
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
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

    std::map<std::tuple<int, int, bool, VkBool32, VkBool32, int, int, int>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayouts_[3] = {};
    VkPipelineLayout layouts_[3] = {};
    bool logged_[3] = {}, declined_[3] = {};
};

} // namespace

Pass* shadowPass() {
    static Shadow pass;
    return &pass;
}

} // namespace host::native
