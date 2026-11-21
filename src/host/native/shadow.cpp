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
#include "host/native/engine_capture.h"

#include <bit>
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
// Its own pipelines for skinned casters (vs_shadowmap_skin with the opaque pixel shader).
constexpr int kSkin = 3;

struct Push {
    float m[16];      // clip = m * (position, 1), column major
    uint32_t pos[4];  // first word, stride in words, dfmt
    uint32_t uv[4];
    float bias[2];    // scale, cap
    float uvScale[2]; // the texture's coordinate scale (native::Texture::scale)
    float origin[4];  // subtracted from the position first (the camera, for world-space casters)
};
static_assert(sizeof(Push) == 128);

constexpr const char* kHeader = R"(#version 450
layout(push_constant) uniform P {
    mat4 m;
    uvec4 pos;
    uvec4 uv;
    vec4 k; // bias scale, bias cap, texture scale
    vec4 origin;
} p;
#define K p.k
)";

// Skinned casters: 128 bytes of constants (the least a device offers).
struct SkinPush {
    float m[16];     // clip = m * (world, 1): projection * rotation
    float off[4];    // the offset added to the blended translation (0x170 of s[4:7]) minus the camera; w: bias scale
    uint32_t pos[4];  // first word, stride in words, format (dfmt | nfmt << 4); w: the palette's first word
    uint32_t wgt[4];  // the weights' slot; w: the palette's stride in words
    uint32_t idx[4];  // the bone indices' slot; w: the bias cap (float bits)
};
static_assert(sizeof(SkinPush) == 128);

constexpr const char* kSkinHeader = R"(#version 450
layout(push_constant) uniform P {
    mat4 m;
    vec4 off;
    uvec4 pos;
    uvec4 wgt;
    uvec4 idx;
} p;
#define K vec4(p.off.w, uintBitsToFloat(p.idx.w), 0.0, 0.0)
)";

// vs_shadowmap_skin.vso: four bones from a palette of 3x4 rows (three 16-byte elements a bone in
// the buffer s[8:11]), weights from the second fetch slot's y, z, w (the first bone's is 1 minus
// their sum), indices from the third. The translation is blended relative to the floor of the
// first bone's, as the game keeps large world positions precise.
constexpr const char* kSkinVs = R"(
layout(set = 0, binding = 0, std430) readonly buffer V0 { uint w[]; } v0;
layout(set = 0, binding = 1, std430) readonly buffer V1 { uint w[]; } v1;
layout(set = 0, binding = 2, std430) readonly buffer V2 { uint w[]; } v2;
layout(set = 0, binding = 3, std430) readonly buffer Pal { uint w[]; } pal;
layout(location = 0) out vec4 o;
uint word(uint b, uint i) { return b == 0u ? v0.w[i] : b == 1u ? v1.w[i] : v2.w[i]; }
// An element as the GCN fetch decodes it, as raw bits (floats as their bits), by data format
// (bits 0-3) and number format (bits 4-6): 32-bit, 16-bit and 8-bit channels.
uint conv(uint raw, uint bits, uint nfmt) {
    float maxv = float((1u << bits) - 1u), half_ = float((1u << (bits - 1u)) - 1u);
    int sx = bitfieldExtract(int(raw), 0, int(bits));
    if (nfmt == 0u) return floatBitsToUint(float(raw) / maxv);
    if (nfmt == 1u) return floatBitsToUint(max(float(sx) / half_, -1.0));
    if (nfmt == 7u && bits == 16u) return floatBitsToUint(unpackHalf2x16(raw).x);
    if (nfmt == 5u) return uint(sx);
    return raw;
}
uvec4 fetch(uint b, uvec4 s) {
    uint i = s.x + uint(gl_VertexIndex) * s.y, dfmt = s.z & 15u, nfmt = (s.z >> 4) & 7u;
    // Missing channels read 0, and w 1 (an integer 1 for integer formats).
    uvec4 e = uvec4(0u, 0u, 0u, nfmt == 4u || nfmt == 5u ? 1u : floatBitsToUint(1.0));
    if (dfmt == 4u || dfmt == 11u || dfmt == 13u || dfmt == 14u) { // 32-bit channels as they are
        uint n = dfmt == 4u ? 1u : dfmt == 11u ? 2u : dfmt == 13u ? 3u : 4u;
        for (uint c = 0u; c < n; ++c) e[c] = word(b, i + c);
    } else if (dfmt == 2u || dfmt == 5u || dfmt == 12u) {
        uint n = dfmt == 2u ? 1u : dfmt == 5u ? 2u : 4u;
        for (uint c = 0u; c < n; ++c) e[c] = conv((word(b, i + c / 2u) >> (16u * (c & 1u))) & 0xffffu, 16u, nfmt);
    } else {
        uint n = dfmt == 1u ? 1u : dfmt == 3u ? 2u : 4u, x = word(b, i);
        for (uint c = 0u; c < n; ++c) e[c] = conv((x >> (8u * c)) & 0xffu, 8u, nfmt);
    }
    return e;
}
vec4 row(int e) { uint i = p.pos.w + uint(e) * p.wgt.w; return uintBitsToFloat(uvec4(pal.w[i], pal.w[i + 1u], pal.w[i + 2u], pal.w[i + 3u])); }
void main() {
    vec3 x = uintBitsToFloat(fetch(0u, p.pos).xyz);
    vec4 wv = uintBitsToFloat(fetch(1u, p.wgt));
    uvec4 bi = fetch(2u, p.idx);
    float w[4] = float[4](1.0 - ((wv.y + wv.z) + wv.w), wv.y, wv.z, wv.w);
    vec4 r[4][3];
    for (int k = 0; k < 4; ++k) {
        int b = (int(bi[k]) * 48) >> 4;
        r[k][0] = row(b), r[k][1] = row(b + 1), r[k][2] = row(b + 2);
    }
    precise vec3 world;
    for (int a = 0; a < 3; ++a) {
        float f = floor(r[0][a].w);
        precise float t = p.off[a] + f;
        for (int k = 0; k < 4; ++k) t += w[k] * (r[k][a].w - f);
        precise vec3 m = r[0][a].xyz * w[0];
        for (int k = 1; k < 4; ++k) m += w[k] * r[k][a].xyz;
        world[a] = t + m.x * x.x + m.y * x.y + m.z * x.z;
    }
    vec4 c = p.m * vec4(world, 1.0);
    gl_Position = vec4(c.x, c.y, max(0.0, c.z), c.w);
    o = vec4(c.z, c.w, 0.0, 0.0);
}
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
    precise vec3 x = position(p.pos.x + uint(gl_VertexIndex) * p.pos.y) - p.origin.xyz;
    vec4 c = p.m * vec4(x, 1.0);
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
    if (0.5 > texture(alpha, i.zw * K.zw).w) demote; // kept for the quad's differences
    float rw = 1.0 / i.y;
    gl_FragDepth = clamp(min(K.y, slope(i.x * rw)) + rw * i.x, 0.0, 1.0);
#else
    gl_FragDepth = clamp(i.x + min(K.y, K.x * slope(i.x)), 0.0, 1.0);
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
        if (role == kOpaque && d.vsName && *d.vsName == "vs_shadowmap_skin.vso") return drawSkin(d, gpu);
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
            // DST_SEL of the components it uses: x, y (and z for the position) as stored. The
            // rest (w is often the constant 1 for a 3-component position) is not read.
            const uint32_t used = v.slot == 0 ? 3 : 2;
            bool plain = true;
            for (uint32_t c = 0; c < used; ++c) plain = plain && ((v.dstSel >> (3 * c)) & 7) == 4 + c;
            if (v.nfmt != 7 || !n || v.stride % 4 || (v.data.byteOffset + v.offset) % 4 || !plain || v.indexVgpr != 0) {
                char why[96];
                std::snprintf(why, sizeof why, "vertex format dfmt %u nfmt %u stride %u offset %u sel 0x%x index v%u", v.dfmt,
                              v.nfmt, v.stride, v.data.byteOffset + v.offset, v.dstSel, v.indexVgpr);
                return decline(role, why);
            }
            (v.slot == 0 ? pos : uv) = &v;
        }
        if (pos && (components(pos->dfmt) < 3 || pos->components < 3)) return decline(role, "a position of fewer than 3 components");
        if (uv && (uv->components < 2 || (uv->dfmt != 5 && uv->dfmt != 11 && uv->dfmt != 14 && uv->dfmt != 13)))
            return decline(role, "coordinates in a format it does not read");
        if (!pos || (alpha && !uv)) return decline(role, "no position or coordinates");
        Push push{};
        if (const char* missing = matrix(alpha ? kAlphaLoads : kOpaqueLoads, d, push.m)) {
            char why[64];
            std::snprintf(why, sizeof why, "the vertex shader's %s constants are missing", missing);
            return decline(role, why);
        }
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
            // For the casters drawn from the engine's captures, which do not read their pixel
            // constants: every opaque caster has had the same (1, 0.01) so far; a change is logged.
            // (Recorded on several recorder threads, the order between these and the engine's
            // casters is not kept: a change would reach some casters a little early or late.)
            if (std::memcmp(lastBias_, w, 8) != 0 && biasChanges_++ < 8)
                std::fprintf(stderr, "[native] shadow: caster bias constants %g %g\n", std::bit_cast<float>(w[0]), std::bit_cast<float>(w[1]));
            std::memcpy(lastBias_, w, 8);
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

    // A stage caster from the engine's capture: world-space positions (the stage's casters have
    // an identity world matrix, no offset, and subtract the camera: checked with LPN_ENGINE_CHECK).
    bool drawEngine(const EngineCaster& c, const Draw& d, Gpu& gpu) {
        const int role = c.alpha ? kAlphaTest : kOpaque;
        if (d.colorCount || !d.depth.view || d.blend || d.stencil || d.vertex.empty() || !d.indexed) return false;
        const VertexBuffer& pos = d.vertex[0];
        if (pos.stride % 4 || pos.data.byteOffset % 4) return false;
        if (c.alpha && (d.vertex.size() < 2 || d.textures.empty() || (d.vertex[1].data.byteOffset + d.vertex[1].offset) % 4)) return false;
        Push push{};
        for (int r = 0; r < 4; ++r) // projection * rotation
            for (int col = 0; col < 4; ++col) {
                double s = col == 3 ? c.projection[r * 4 + 3] : 0;
                if (col < 3)
                    for (int k = 0; k < 3; ++k) s += double(c.projection[r * 4 + k]) * c.rotation[k * 4 + col];
                push.m[col * 4 + r] = float(s);
            }
        for (int a = 0; a < 3; ++a) push.origin[a] = c.camera[a];
        push.pos[0] = pos.data.byteOffset / 4, push.pos[1] = pos.stride / 4, push.pos[2] = 13;
        push.bias[0] = lastBias_[0], push.bias[1] = lastBias_[1];
        VkDescriptorImageInfo image{};
        VkDescriptorBufferInfo uvBuf{};
        if (c.alpha) { // as the alpha-tested casters: bias without scale, capped at 0.01
            const VertexBuffer& uv = d.vertex[1];
            push.uv[0] = (uv.data.byteOffset + uv.offset) / 4, push.uv[1] = uv.stride / 4, push.uv[2] = uv.dfmt;
            uvBuf = {uv.data.buffer, uv.data.offset, uv.data.range};
            const Texture& t = d.textures[0];
            image = {t.sampler, t.view, VK_IMAGE_LAYOUT_GENERAL};
            push.uvScale[0] = t.scale[0], push.uvScale[1] = t.scale[1];
            push.bias[0] = 1, push.bias[1] = 0.01f;
        }
        VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return false;
        if (!engineLogged_[c.alpha]) {
            engineLogged_[c.alpha] = true;
            std::fprintf(stderr, "[native] shadow: drawing %s stage casters from the engine's captures (depth 0x%" PRIx64 ")\n",
                         c.alpha ? "alpha-tested" : "opaque", d.depth.address);
        }
        gpu.beginRendering(nullptr, 0, d.depth, d.hasStencil);
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const VkDescriptorBufferInfo buf{pos.data.buffer, pos.data.offset, pos.data.range};
        const Buffer index = d.index;
        gpu.record([pipe, layout = layouts_[role], push, buf, uvBuf, image, alpha = c.alpha, vp, sc, index, indexType = d.indexType,
                    count = d.count, pushSet = gpu.pushDescriptorSet()](VkCommandBuffer cb) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);
            VkWriteDescriptorSet w[3]{};
            for (uint32_t i = 0; i < 3; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            }
            w[0].pBufferInfo = &buf, w[1].pBufferInfo = &uvBuf, w[2].pImageInfo = &image;
            pushSet(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, alpha ? 3 : 1, w);
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, &push);
            vkCmdBindIndexBuffer(cb, index.buffer, index.offset + index.byteOffset, indexType);
            vkCmdDrawIndexed(cb, count, 1, 0, 0, 0);
        });
        return true;
    }

private:
    float lastBias_[2] = {1.0f, 0.01f}; // the opaque casters' bias constants, as last seen
    uint32_t biasChanges_ = 0;
    bool engineLogged_[2] = {};

    // A fetch slot in a format the skinned shader decodes (32-, 16- or 8-bit channels).
    static bool decodable(const VertexBuffer& v) {
        static constexpr uint32_t kDfmts[] = {1, 2, 3, 4, 5, 10, 11, 12, 13, 14};
        bool dfmt = false;
        for (uint32_t f : kDfmts) dfmt = dfmt || v.dfmt == f;
        const bool nfmt = v.nfmt == 0 || v.nfmt == 1 || v.nfmt == 4 || v.nfmt == 5 || v.nfmt == 7;
        return dfmt && nfmt && v.stride % 4 == 0 && (v.data.byteOffset + v.offset) % 4 == 0 && v.indexVgpr == 0;
    }

    bool drawSkin(const Draw& d, Gpu& gpu) {
        const int role = kSkin;
        if (d.colorCount || !d.depth.view) return decline(role, "colour targets or no depth buffer");
        if (d.blend || d.stencil) return decline(role, "blending or stencil");
        if (d.instances != 1) return decline(role, "instanced");
        if (d.primitive != 4) return decline(role, "not a triangle list");
        // Fetch slots: position (v4), weights (v8: y, z, w used), bone indices (v12); the palette
        // is the buffer of the first bone row load (pc 0x38).
        const VertexBuffer* slot[3] = {};
        const VertexBuffer* palette = nullptr;
        for (const VertexBuffer& v : d.vertex) {
            if (v.slot >= 0 && v.slot < 3 && v.components) slot[v.slot] = &v;
            if (v.slot < 0 && v.pc == 0x38) palette = &v;
        }
        if (!slot[0] || !slot[1] || !slot[2] || !palette) return decline(role, "no position, weights, bone indices or palette");
        if (slot[0]->components < 3 || slot[1]->components < 4 || slot[2]->components < 4)
            return decline(role, "fewer components than the shader reads");
        for (const VertexBuffer* v : slot) {
            // As stored, for every component used.
            const uint32_t used = v == slot[0] ? 3 : 4;
            bool plain = true;
            for (uint32_t c = 0; c < used; ++c) plain = plain && ((v->dstSel >> (3 * c)) & 7) == 4 + c;
            if (!decodable(*v) || !plain) {
                char why[96];
                std::snprintf(why, sizeof why, "vertex format dfmt %u nfmt %u stride %u offset %u sel 0x%x index v%u", v->dfmt,
                              v->nfmt, v->stride, v->data.byteOffset + v->offset, v->dstSel, v->indexVgpr);
                return decline(role, why);
            }
        }
        if (palette->stride % 4 || palette->data.byteOffset % 4 || palette->stride < 16) return decline(role, "a palette layout it does not read");
        SkinPush push{};
        float off[4], rot[12], proj[16], cam[4] = {};
        uint32_t flags;
        auto get = [&](uint32_t pc, float* v, uint32_t n) { return d.vsLoad(pc, reinterpret_cast<uint32_t*>(v), n); };
        if (!get(0x104, off, 4) || !d.vsLoad(0x118, &flags, 1) || !get(0x2c8, rot, 8) || !get(0x2d0, rot + 8, 4) ||
            !get(0x2cc, proj, 16))
            return decline(role, "the vertex shader's constants are missing");
        if (!((flags & 0x80000) && !(flags & 0x100000)) && !get(0x2a8, cam, 4)) return decline(role, "no camera constants");
        // projection * rotation; the camera goes into the offset, so the shader subtracts it from
        // the blended translation before rotating (a translation of -rotation * camera in the
        // matrix lost the precision of positions far from the origin).
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                double s = c == 3 ? proj[r * 4 + 3] : 0;
                if (c < 3)
                    for (int k = 0; k < 3; ++k) s += double(proj[r * 4 + k]) * rot[k * 4 + c];
                push.m[c * 4 + r] = float(s);
            }
        uint32_t w[2];
        if (!d.load(0xc, w, 2)) return decline(role, "no bias constants");
        for (int a = 0; a < 3; ++a) push.off[a] = float(double(off[a]) - cam[a]);
        std::memcpy(&push.off[3], &w[0], 4);
        push.idx[3] = w[1];
        auto put = [](const VertexBuffer& v, uint32_t* s) {
            s[0] = (v.data.byteOffset + v.offset) / 4, s[1] = v.stride / 4, s[2] = v.dfmt | v.nfmt << 4;
        };
        put(*slot[0], push.pos), put(*slot[1], push.wgt), put(*slot[2], push.idx);
        push.pos[3] = palette->data.byteOffset / 4, push.wgt[3] = palette->stride / 4;
        VkDescriptorBufferInfo bufs[4];
        for (int i = 0; i < 3; ++i) bufs[i] = {slot[i]->data.buffer, slot[i]->data.offset, slot[i]->data.range};
        bufs[3] = {palette->data.buffer, palette->data.offset, palette->data.range};
        VkPipeline pipe = pipeline(gpu, role, d);
        if (!pipe) return decline(role, "no pipeline");
        if (!logged_[role]) {
            logged_[role] = true;
            std::fprintf(stderr, "[native] shadow: drawing vs_shadowmap_skin.vso natively (depth 0x%" PRIx64 " %ux%u)\n",
                         d.depth.address, d.depth.width, d.depth.height);
        }
        gpu.beginRendering(nullptr, 0, d.depth, d.hasStencil);
        const VkViewport vp = d.vkViewport;
        const VkRect2D sc{{d.scissor[0], d.scissor[1]}, {uint32_t(d.scissor[2]), uint32_t(d.scissor[3])}};
        const Buffer index = d.index;
        gpu.record([pipe, layout = layouts_[role], push, bufs, vp, sc, index, indexed = d.indexed, indexType = d.indexType,
                    count = d.count, pushSet = gpu.pushDescriptorSet()](VkCommandBuffer c) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdSetViewport(c, 0, 1, &vp);
            vkCmdSetScissor(c, 0, 1, &sc);
            VkWriteDescriptorSet wr[4]{};
            for (uint32_t i = 0; i < 4; ++i) {
                wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstBinding = i;
                wr[i].descriptorCount = 1;
                wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wr[i].pBufferInfo = &bufs[i];
            }
            pushSet(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 4, wr);
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

    // Logs each reason once.
    bool decline(int role, const char* why) {
        if (declined_[role].insert(why).second && declined_[role].size() <= 16)
            std::fprintf(stderr, "[native] shadow: a %s draw left to the translated renderer (%s)\n",
                         role == kAlphaTest ? "ps_shadowmap_pt" : role == kSkin ? "vs_shadowmap_skin" : "ps_shadowmap", why);
        return false;
    }

    // projection * rotation * (world + offset - camera), from the vertex shader's loads.
    // Null, or the constants that are missing.
    static const char* matrix(const Loads& l, const Draw& d, float out[16]) {
        float off[4], world[12], rot[12], proj[16];
        uint32_t flags;
        auto get = [&](uint32_t pc, float* v, uint32_t n) { return d.vsLoad(pc, reinterpret_cast<uint32_t*>(v), n); };
        if (!get(l.offset, off, 4)) return "offset";
        if (!get(l.world, world, 8) || !get(l.world2, world + 8, 4)) return "world matrix";
        if (!d.vsLoad(l.flags, &flags, 1)) return "flags";
        if (!get(l.rotation, rot, 8) || !get(l.rotation2, rot + 8, 4)) return "view rotation";
        if (!get(l.projection, proj, 16)) return "projection";
        float cam[4] = {};
        const bool keepCamera = (flags & 0x80000) && !(flags & 0x100000);
        if (!keepCamera && !get(l.camera, cam, 4)) return "camera";
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
        return nullptr;
    }

    VkPipeline pipeline(Gpu& gpu, int role, const Draw& d) {
        const VkPipelineDepthStencilStateCreateInfo& z = d.depthStencil;
        const auto key = std::make_tuple(role, int(d.depth.format), bool(d.hasStencil), z.depthTestEnable, z.depthWriteEnable,
                                         int(z.depthCompareOp), int(d.cull), int(d.frontFace));
        if (auto it = pipes_.find(key); it != pipes_.end()) return it->second;
        VkPipeline& out = pipes_[key];
        const VkDevice dev = gpu.device();
        const bool alpha = role == kAlphaTest, skin = role == kSkin;
        if (!layouts_[role]) {
            VkDescriptorSetLayoutBinding b[4] = {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            };
            if (skin) b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dl.bindingCount = skin ? 4 : alpha ? 3 : 1;
            dl.pBindings = b;
            if (vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
            const VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                         uint32_t(skin ? sizeof(SkinPush) : sizeof(Push))};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &setLayouts_[role];
            pl.pushConstantRangeCount = 1;
            pl.pPushConstantRanges = &pr;
            if (vkCreatePipelineLayout(dev, &pl, nullptr, &layouts_[role]) != VK_SUCCESS) return out = VK_NULL_HANDLE;
        }
        const std::string defs = alpha ? "#define ALPHA_TEST\n" : "";
        const char* header = skin ? kSkinHeader : kHeader;
        const VkShaderModule vsm = gpu.shader(std::string(header) + defs + (skin ? kSkinVs : kVs), Gpu::Stage::Vertex);
        const VkShaderModule fsm = gpu.shader(std::string(header) + defs + kFs, Gpu::Stage::Fragment);
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
    VkDescriptorSetLayout setLayouts_[4] = {};
    VkPipelineLayout layouts_[4] = {};
    bool logged_[4] = {};
    std::set<std::string> declined_[4];
};

Shadow& shadowInstance() {
    static Shadow pass;
    return pass;
}

} // namespace

Pass* shadowPass() { return &shadowInstance(); }

bool drawStageCaster(const EngineCaster& c, const Draw& d, Gpu& gpu) { return shadowInstance().drawEngine(c, d, gpu); }

} // namespace host::native
