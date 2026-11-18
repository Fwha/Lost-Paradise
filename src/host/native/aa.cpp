// The game's anti-aliasing (MLAA, in the shader archive as ps_aa_filter_edge2, _scan_edge2 and
// ps_aa_filter2), rebuilt from the disassembly of its three pixel shaders:
//
// 1. Edges (ps_aa_filter_edge2.pso): a pixel's colour against its four neighbours'. An edge is
//    where the largest channel difference reaches the smaller of the two pixels' thresholds
//    (max(abs, rel * brightest channel), constants at 0x0 of the draw's first constant buffer).
//    Out: left, up, right, down as 0 or 1; a pixel with none is discarded.
// 2. Blend weights (ps_aa_filter_scan_edge2.pso): along each edge, up to 4 pixels each way to
//    where it ends, the kind of crossing edge at each end read from the edge texture between two
//    rows with a bilinear sampler (4 * value + 0.5, floored: 0 to 4), and the coverage from the
//    game's area texture at (9 * crossing + distance) / 64 for both ends.
// 3. Blend (ps_aa_filter2.pso): the colour fetched half-way towards the neighbour across the
//    pixel's strongest edge, with the bilinear sampler, or as it is without one.
//
// Every pass reads the target's texel size at 0x218 of a constant buffer and the pixel's centre
// (gl_FragCoord): none needs the game's vertex shader.
#include "host/native/native.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>

namespace host::native {

namespace {

enum Role { kEdges = 1, kWeights = 2, kBlend = 3 };

// The constants every shader gets: the texel size, the edge thresholds (abs, rel) and the
// coordinate scale of up to three textures.
struct Push {
    float texel[2], threshold[2];
    float scale[3][2];
    float pad[2];
};
static_assert(sizeof(Push) == 48);

constexpr const char* kHeader = R"(#version 450
layout(push_constant) uniform P {
    vec4 a; // texel size, thresholds
    vec4 b; // scales of textures 0 and 1
    vec4 c; // scale of texture 2
} p;
layout(location = 0) out vec4 o;
vec2 texel() { return p.a.xy; }
)";

constexpr const char* kEdgesFs = R"(
layout(set = 0, binding = 0) uniform sampler2D colour;
vec3 at(vec2 uv) { return texture(colour, uv * p.b.xy).rgb; }
float max3(vec3 v) { return max(max(v.x, v.y), v.z); }
float threshold(vec3 c) { return max(p.a.z, p.a.w * max3(c)); }
// 1 where c and n differ by at least the smaller of their thresholds.
float edge(vec3 c, float tc, vec3 n) {
    return min(tc, threshold(n)) <= max3(abs(c - n)) ? 1.0 : 0.0;
}
void main() {
    vec2 t = texel();
    vec2 uv = gl_FragCoord.xy * t;
    vec3 c = at(uv);
    float tc = threshold(c);
    float r = edge(c, tc, at(vec2(uv.x + t.x, uv.y)));
    float d = edge(c, tc, at(vec2(uv.x, uv.y + t.y)));
    float u = edge(c, tc, at(vec2(uv.x, uv.y - t.y)));
    float l = edge(c, tc, at(vec2(uv.x - t.x, uv.y)));
    if (r + d + l + u < 0.5) discard;
    o = vec4(l, u, r, d);
}
)";

constexpr const char* kWeightsFs = R"(
layout(set = 0, binding = 0) uniform sampler2D edges;  // point
layout(set = 0, binding = 1) uniform sampler2D cross;  // the edges again, bilinear
layout(set = 0, binding = 2) uniform sampler2D area;
vec2 e(float x, float y) { return texture(edges, vec2(x, y) * p.b.xy).xy; }
float crossing(float v) { return floor(4.0 * v + 0.5); }
vec2 coverage(float a, float b) { return texture(area, vec2(a, b) / 64.0 * p.c.xy).xw; }
void main() {
    vec2 t = texel();
    vec2 uv = gl_FragCoord.xy * t;
    vec2 c = e(uv.x, uv.y);
    vec2 h = vec2(0.0), v = vec2(0.0);
    if (c.y != 0.0) { // an edge above: along it to the left and right
        vec2 l1 = e(uv.x - t.x, uv.y), l2 = e(uv.x - 2.0 * t.x, uv.y), l3 = e(uv.x - 3.0 * t.x, uv.y);
        float l4 = e(uv.x - 4.0 * t.x, uv.y).y;
        vec2 r1 = e(uv.x + t.x, uv.y), r2 = e(uv.x + 2.0 * t.x, uv.y), r3 = e(uv.x + 3.0 * t.x, uv.y),
             r4 = e(uv.x + 4.0 * t.x, uv.y);
        float a1 = c.x != 0.0 ? 0.0 : 1.0;
        a1 = l1.y == 0.0 ? 0.0 : a1;
        float a2 = l1.x != 0.0 ? 0.0 : a1;
        a2 = l2.y == 0.0 ? 0.0 : a2;
        float a3 = l2.x != 0.0 ? 0.0 : a2;
        a3 = l3.y == 0.0 ? 0.0 : a3;
        float a4 = l3.x != 0.0 ? 0.0 : a3;
        a4 = l4 == 0.0 ? 0.0 : a4;
        float left = a1 + a2 + a3 + a4;
        float b1 = (r1.y == 0.0 || r1.x != 0.0) ? 0.0 : 1.0;
        float b2 = (r2.y == 0.0 || r2.x != 0.0) ? 0.0 : b1;
        float b3 = (r3.y == 0.0 || r3.x != 0.0) ? 0.0 : b2;
        float b4 = (r4.y == 0.0 || r4.x != 0.0) ? 0.0 : b3;
        float right = b2 + b3 + b4 + b1;
        float y = uv.y - 0.25 * t.y;
        float cl = crossing(texture(cross, vec2(uv.x - left * t.x, y) * p.b.zw).x);
        float cr = crossing(texture(cross, vec2(uv.x + t.x + right * t.x, y) * p.b.zw).x);
        h = coverage(cl * 9.0 + left, cr * 9.0 + right);
    }
    if (c.x != 0.0) { // an edge on the left: along it up and down
        vec2 u1 = e(uv.x, uv.y - t.y), u2 = e(uv.x, uv.y - 2.0 * t.y), u3 = e(uv.x, uv.y - 3.0 * t.y);
        float u4 = e(uv.x, uv.y - 4.0 * t.y).x;
        vec2 d1 = e(uv.x, uv.y + t.y), d2 = e(uv.x, uv.y + 2.0 * t.y), d3 = e(uv.x, uv.y + 3.0 * t.y),
             d4 = e(uv.x, uv.y + 4.0 * t.y);
        float a1 = c.y != 0.0 ? 0.0 : 1.0;
        a1 = u1.x == 0.0 ? 0.0 : a1;
        float a2 = u1.y != 0.0 ? 0.0 : a1;
        a2 = u2.x == 0.0 ? 0.0 : a2;
        float a3 = u2.y != 0.0 ? 0.0 : a2;
        a3 = u3.x == 0.0 ? 0.0 : a3;
        float a4 = u3.y != 0.0 ? 0.0 : a3;
        a4 = u4 == 0.0 ? 0.0 : a4;
        float up = a1 + a2 + a3 + a4;
        float b1 = (d1.x == 0.0 || d1.y != 0.0) ? 0.0 : 1.0;
        float b2 = (d2.x == 0.0 || d2.y != 0.0) ? 0.0 : b1;
        float b3 = (d3.x == 0.0 || d3.y != 0.0) ? 0.0 : b2;
        float b4 = (d4.x == 0.0 || d4.y != 0.0) ? 0.0 : b3;
        float down = b2 + b3 + b4 + b1;
        float x = uv.x - 0.25 * t.x;
        float cu = crossing(texture(cross, vec2(x, uv.y - up * t.y) * p.b.zw).y);
        float cd = crossing(texture(cross, vec2(x, uv.y + t.y + down * t.y) * p.b.zw).y);
        v = coverage(cu * 9.0 + up, cd * 9.0 + down);
    }
    o = clamp(vec4(h, v), 0.0, 1.0);
}
)";

constexpr const char* kBlendFs = R"(
layout(set = 0, binding = 0) uniform sampler2D weights;
layout(set = 0, binding = 1) uniform sampler2D colour; // bilinear
void main() {
    vec2 t = texel();
    vec2 uv = gl_FragCoord.xy * t;
    vec4 w = texture(weights, uv * p.b.xy);
    float below = texture(weights, vec2(uv.x, uv.y + t.y) * p.b.xy).y;
    float right = texture(weights, vec2(uv.x + t.x, uv.y) * p.b.xy).w;
    float sum = w.x + below + w.z + right;
    if (1e-5 > sum) {
        o = texture(colour, uv * p.b.zw);
    } else {
        float x = right > w.z ? right : -w.z;
        float y = below > w.x ? below : -w.x;
        if (abs(x) > abs(y)) y = 0.0;
        else x = 0.0;
        o = texture(colour, vec2(uv.x + t.x * x, uv.y + t.y * y) * p.b.zw);
    }
}
)";

struct RoleInfo {
    const char* shader; // archive entry
    const char* fs;
    uint32_t texelPc;   // the scalar load of the texel size
    uint32_t thresholdPc; // the edge thresholds (0: none)
    uint32_t texturePcs[3]; // the first sample of each texture (0: none)
};
constexpr RoleInfo kRoles[4] = {
    {},
    {"ps_aa_filter_edge2.pso", kEdgesFs, 0x18, 0x88, {0x48, 0, 0}},
    {"ps_aa_filter_scan_edge2.pso", kWeightsFs, 0x14, 0, {0x28, 0x1f8, 0x248}},
    {"ps_aa_filter2.pso", kBlendFs, 0x18, 0, {0x48, 0x8c, 0}},
};

class Aa final : public Pass {
public:
    const char* name() const override { return "aa"; }

    int claims(const std::string& psName) const override {
        for (int r = kEdges; r <= kBlend; ++r)
            if (psName == kRoles[r].shader || psName.rfind(std::string(kRoles[r].shader) + " +", 0) == 0) return r;
        return 0;
    }

    bool draw(int role, const Draw& d, Gpu& gpu) override {
        const RoleInfo& info = kRoles[role];
        if (d.colorCount != 1 || !d.color[0].view) return decline(role, "not one render target");
        if (d.blend) return decline(role, "blending");
        if (d.depthTest || d.depthWrite || d.stencil) return decline(role, "depth or stencil");
        Push push{};
        uint32_t words[2];
        if (!d.load(info.texelPc, words, 2)) return decline(role, "no texel size");
        std::memcpy(push.texel, words, 8);
        if (info.thresholdPc) {
            if (!d.load(info.thresholdPc, words, 2)) return decline(role, "no thresholds");
            std::memcpy(push.threshold, words, 8);
        }
        VkDescriptorImageInfo images[3]{};
        uint32_t n = 0;
        for (uint32_t pc : info.texturePcs) {
            if (!pc) break;
            const Texture* t = nullptr;
            for (const Texture& x : d.textures)
                if (x.pc == pc) t = &x;
            if (!t || !t->view || !t->sampler) return decline(role, "a texture is missing");
            images[n] = {t->sampler, t->view, VK_IMAGE_LAYOUT_GENERAL};
            push.scale[n][0] = t->scale[0], push.scale[n][1] = t->scale[1];
            ++n;
        }
        const uint32_t mask = d.writeMask & 0xF;
        VkPipeline pipe = pipeline(gpu, role, d.color[0].format, mask);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role]) {
            logged_[role] = true;
            std::fprintf(stderr, "[native] aa: drawing %s natively (target 0x%" PRIx64 " %ux%u)\n", info.shader,
                         d.color[0].address, d.color[0].width, d.color[0].height);
        }
        const VkViewport vp{d.viewport[0], d.viewport[1], d.viewport[2], d.viewport[3], 0, 1};
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Target target = d.color[0];
        gpu.record([pipe, layout = layouts_[role], push, images, n, vp, sc, target,
                    pushSet = gpu.pushDescriptorSet()](VkCommandBuffer c) {
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView = target.view;
            att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {target.width, target.height}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;
            vkCmdBeginRendering(c, &ri);
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(c, 0, 1, &vp);
            vkCmdSetScissor(c, 0, 1, &sc);
            VkWriteDescriptorSet w[3]{};
            for (uint32_t i = 0; i < n; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[i].pImageInfo = &images[i];
            }
            pushSet(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, n, w);
            vkCmdPushConstants(c, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, &push);
            vkCmdDraw(c, 3, 1, 0, 0);
            vkCmdEndRendering(c);
        });
        ++drawn_;
        return true;
    }

    void frameEnd() override { drawn_ = 0; }

private:
    bool decline(int role, const char* why) {
        if (!declined_[role]) {
            declined_[role] = true;
            std::fprintf(stderr, "[native] aa: %s left to the translated renderer (%s)\n", kRoles[role].shader, why);
        }
        return false;
    }

    VkPipeline pipeline(Gpu& gpu, int role, VkFormat format, uint32_t mask) {
        const auto key = std::make_tuple(role, int(format), mask);
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        if (!layouts_[role]) {
            VkDescriptorSetLayoutBinding b[3]{};
            uint32_t n = 0;
            for (uint32_t pc : kRoles[role].texturePcs)
                if (pc) b[n] = {n, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, ++n;
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = n;
            dl.pBindings = b;
            if (vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
            const VkPushConstantRange pr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayouts_[role];
            pl.pushConstantRangeCount = 1;
            pl.pPushConstantRanges = &pr;
            if (vkCreatePipelineLayout(dev, &pl, nullptr, &layouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
        }
        const VkShaderModule vs = gpu.shader(kFullscreenVs, Gpu::Stage::Vertex);
        const VkShaderModule fs = gpu.shader(std::string(kHeader) + kRoles[role].fs, Gpu::Stage::Fragment);
        if (!vs || !fs) return out = VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr};
        st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = mask;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rinfo.colorAttachmentCount = 1;
        rinfo.pColorAttachmentFormats = &format;
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

    std::map<std::tuple<int, int, uint32_t>, VkPipeline> pipes_;
    VkDescriptorSetLayout setLayouts_[4] = {};
    VkPipelineLayout layouts_[4] = {};
    bool logged_[4] = {}, declined_[4] = {};
    uint32_t drawn_ = 0;
};

} // namespace

Pass* aaPass() {
    static Aa pass;
    return &pass;
}

} // namespace host::native
