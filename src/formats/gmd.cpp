#include "formats/gmd.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

namespace formats {

namespace {

// Header: "GSGM", then (offset, count) pairs for each table starting at 0x30.
enum Table {
    kNodes, kObjects, kMeshes, kAttributes, kMaterials, kMatrices, kVertexBuffers,
    kVertexData, kTextures, kShaders, kNodeNames, kIndexData, kDrawLists, kMatrixLists,
    kTableCount
};
constexpr size_t kTableStart = 0x30;
constexpr size_t kNodeSize = 0x80;
constexpr size_t kMeshSize = 0x40;
constexpr size_t kAttributeSize = 0x80;
constexpr size_t kVertexBufferSize = 0x20;
constexpr size_t kNameSize = 0x20; // u16 checksum + 30 chars

struct Table_ {
    uint32_t offset = 0;
    uint32_t count = 0;
};

class Reader {
public:
    explicit Reader(std::span<const uint8_t> d) : d_(d) {}
    bool has(uint64_t off, uint64_t size) const { return off + size <= d_.size(); }
    uint32_t u32(size_t off) const {
        uint32_t v;
        std::memcpy(&v, d_.data() + off, 4);
        return v;
    }
    int32_t i32(size_t off) const { return int32_t(u32(off)); }
    uint16_t u16(size_t off) const { return uint16_t(d_[off] | d_[off + 1] << 8); }
    uint8_t u8(size_t off) const { return d_[off]; }
    float f32(size_t off) const {
        float v;
        std::memcpy(&v, d_.data() + off, 4);
        return v;
    }
    float f16(size_t off) const {
        const uint16_t h = u16(off);
        const int exp = (h >> 10) & 0x1F;
        const int mant = h & 0x3FF;
        float v = exp == 0    ? std::ldexp(float(mant), -24)
                  : exp == 31 ? (mant ? NAN : INFINITY)
                              : std::ldexp(float(mant | 0x400), exp - 25);
        return (h & 0x8000) ? -v : v;
    }
    std::string name(size_t off) const {
        const char* p = reinterpret_cast<const char*>(d_.data() + off + 2);
        return std::string(p, strnlen(p, kNameSize - 2));
    }

private:
    std::span<const uint8_t> d_;
};

float unorm8ToSigned(uint8_t b) { return b / 255.0f * 2.0f - 1.0f; }

} // namespace

bool GmdModel::isGmd(std::span<const uint8_t> data) {
    return data.size() >= kTableStart + kTableCount * 8 && std::memcmp(data.data(), "GSGM", 4) == 0;
}

std::optional<GmdVertexLayout> gmdVertexLayout(uint64_t flags, std::string* error) {
    // Bit groups inferred from Lost Paradise character models:
    //   0-2   position (3 = float32 xyz)
    //   6-7   bone weights (u8 x4) + bone indices (u8 x4)
    //   8-12  normal (u8 x4)
    //   13-15 tangent (u8 x4)
    //   21-23 extra u8 x4 (unknown, possibly vertex colour)
    //   24-26 extra u8 x4 (unknown)
    //   27-28 uv0 (float16 x2)
    constexpr uint64_t kKnown = 0x7 | 0xC0 | 0x1F00 | 0xE000 | 0x7E00000 | 0x18000000;
    auto fail = [&](const std::string& msg) -> std::optional<GmdVertexLayout> {
        if (error) *error = msg;
        return std::nullopt;
    };
    const uint32_t lo = uint32_t(flags);
    if ((lo & 0x7) != 3) return fail("unsupported vertex position format");
    if (lo & ~kKnown) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "unknown vertex flags 0x%08x", unsigned(lo & ~kKnown));
        return fail(buf);
    }

    GmdVertexLayout l;
    l.stride = 12;
    if (lo & 0xC0) {
        l.weights = int(l.stride);
        l.bones = int(l.stride + 4);
        l.stride += 8;
    }
    if (lo & 0x1F00) l.normal = int(std::exchange(l.stride, l.stride + 4));
    if (lo & 0xE000) l.tangent = int(std::exchange(l.stride, l.stride + 4));
    if (lo & 0xE00000) l.stride += 4;
    if (lo & 0x7000000) l.stride += 4;
    if (lo & 0x18000000) l.uv = int(std::exchange(l.stride, l.stride + 4));
    return l;
}

std::optional<GmdModel> GmdModel::parse(std::span<const uint8_t> data, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<GmdModel> {
        if (error) *error = msg;
        return std::nullopt;
    };
    if (!isGmd(data)) return fail("not a GSGM model");
    Reader r(data);

    std::array<Table_, kTableCount> t;
    for (int i = 0; i < kTableCount; ++i)
        t[i] = {r.u32(kTableStart + i * 8), r.u32(kTableStart + i * 8 + 4)};
    auto tableFits = [&](Table e, size_t entrySize) {
        return r.has(t[e].offset, uint64_t(t[e].count) * entrySize);
    };
    if (!tableFits(kNodes, kNodeSize) || !tableFits(kMeshes, kMeshSize) ||
        !tableFits(kAttributes, kAttributeSize) || !tableFits(kVertexBuffers, kVertexBufferSize) ||
        !tableFits(kTextures, kNameSize) || !tableFits(kShaders, kNameSize) ||
        !tableFits(kNodeNames, kNameSize) || !tableFits(kVertexData, 1) ||
        !tableFits(kIndexData, 2) || !tableFits(kMatrixLists, 1))
        return fail("GSGM tables run past the end of the file");

    GmdModel m;
    m.name_ = r.name(0x10);
    for (uint32_t i = 0; i < t[kTextures].count; ++i)
        m.textures_.push_back(r.name(t[kTextures].offset + i * kNameSize));
    for (uint32_t i = 0; i < t[kShaders].count; ++i)
        m.shaders_.push_back(r.name(t[kShaders].offset + i * kNameSize));

    // Nodes: index, first child, next sibling, object, matrix, stack op, name index, type,
    // then position, rotation, scale, world position (vec4 each).
    for (uint32_t i = 0; i < t[kNodes].count; ++i) {
        const size_t e = t[kNodes].offset + i * kNodeSize;
        GmdNode n;
        n.firstChild = r.i32(e + 4);
        n.nextSibling = r.i32(e + 8);
        n.objectIndex = r.i32(e + 12);
        n.matrixIndex = r.i32(e + 16);
        const uint32_t nameIndex = r.u32(e + 24);
        n.nodeType = r.u32(e + 28);
        if (nameIndex < t[kNodeNames].count)
            n.name = r.name(t[kNodeNames].offset + nameIndex * kNameSize);
        for (int k = 0; k < 3; ++k) {
            n.position[k] = r.f32(e + 0x20 + k * 4);
            n.scale[k] = r.f32(e + 0x40 + k * 4);
            n.worldPosition[k] = r.f32(e + 0x50 + k * 4);
        }
        for (int k = 0; k < 4; ++k) n.rotation[k] = r.f32(e + 0x30 + k * 4);
        m.nodes_.push_back(std::move(n));
    }
    // Parents come from the child/sibling links.
    for (size_t i = 0; i < m.nodes_.size(); ++i) {
        int32_t c = m.nodes_[i].firstChild;
        for (size_t guard = 0; c >= 0 && size_t(c) < m.nodes_.size() && guard < m.nodes_.size();
             ++guard) {
            m.nodes_[c].parent = int32_t(i);
            c = m.nodes_[c].nextSibling;
        }
    }

    // Attributes: index, material, shader, ?, ?, flags, ?, ?, then 8 texture indices.
    for (uint32_t i = 0; i < t[kAttributes].count; ++i) {
        const size_t e = t[kAttributes].offset + i * kAttributeSize;
        GmdAttribute a;
        a.materialIndex = r.u32(e + 4);
        a.shaderIndex = r.u32(e + 8);
        a.flags = r.u32(e + 20);
        for (int k = 0; k < kGmdTextureSlots; ++k) {
            const uint32_t tex = r.u32(e + 0x20 + k * 4);
            a.textures[k] = tex < t[kTextures].count ? int32_t(tex) : -1;
        }
        m.attributes_.push_back(a);
    }

    // Vertex buffers: index, vertex count, u64 flags, data offset, data size, stride, pad.
    struct Buffer {
        uint32_t count, offset, size, stride;
        GmdVertexLayout layout;
    };
    std::vector<Buffer> buffers;
    for (uint32_t i = 0; i < t[kVertexBuffers].count; ++i) {
        const size_t e = t[kVertexBuffers].offset + i * kVertexBufferSize;
        Buffer b{r.u32(e + 4), r.u32(e + 16), r.u32(e + 20), r.u32(e + 24), {}};
        const uint64_t flags = r.u32(e + 8) | uint64_t(r.u32(e + 12)) << 32;
        auto layout = gmdVertexLayout(flags, error);
        if (!layout) return std::nullopt;
        if (layout->stride != b.stride)
            return fail("vertex buffer " + std::to_string(i) + ": layout stride " +
                        std::to_string(layout->stride) + " != " + std::to_string(b.stride));
        if (uint64_t(b.offset) + b.size > t[kVertexData].count ||
            uint64_t(b.count) * b.stride > b.size)
            return fail("vertex buffer " + std::to_string(i) + " runs past the vertex data");
        b.layout = *layout;
        buffers.push_back(b);
    }

    // Meshes: index, attribute, vertex buffer, vertex count, triangle list (count, start),
    // two triangle-strip variants (count, start each), matrix list (count, offset),
    // node, object, ?, first vertex. Indices are absolute within the vertex buffer.
    for (uint32_t i = 0; i < t[kMeshes].count; ++i) {
        const size_t e = t[kMeshes].offset + i * kMeshSize;
        GmdMesh mesh;
        mesh.attributeIndex = r.u32(e + 4);
        mesh.vertexBufferIndex = r.u32(e + 8);
        const uint32_t vertexCount = r.u32(e + 12);
        const uint32_t triCount = r.u32(e + 16);
        const uint32_t triStart = r.u32(e + 20);
        const uint32_t matrixCount = r.u32(e + 40);
        const uint32_t matrixOffset = r.u32(e + 44);
        mesh.nodeIndex = r.u32(e + 48);
        mesh.objectIndex = r.u32(e + 52);
        const uint32_t firstVertex = r.u32(e + 60);
        const std::string where = "mesh " + std::to_string(i);

        if (mesh.vertexBufferIndex >= buffers.size()) return fail(where + ": bad vertex buffer");
        const Buffer& b = buffers[mesh.vertexBufferIndex];
        if (uint64_t(firstVertex) + vertexCount > b.count) return fail(where + ": bad vertex range");
        if (triCount % 3 || uint64_t(triStart) + triCount > t[kIndexData].count)
            return fail(where + ": bad triangle list");

        const size_t base = t[kVertexData].offset + b.offset;
        mesh.vertices.resize(vertexCount);
        for (uint32_t v = 0; v < vertexCount; ++v) {
            const size_t p = base + size_t(firstVertex + v) * b.stride;
            GmdVertex& out = mesh.vertices[v];
            for (int k = 0; k < 3; ++k) out.position[k] = r.f32(p + k * 4);
            if (b.layout.weights >= 0)
                for (int k = 0; k < 4; ++k) {
                    out.weights[k] = r.u8(p + b.layout.weights + k) / 255.0f;
                    out.bones[k] = r.u8(p + b.layout.bones + k);
                }
            if (b.layout.normal >= 0)
                for (int k = 0; k < 3; ++k) out.normal[k] = unorm8ToSigned(r.u8(p + b.layout.normal + k));
            if (b.layout.tangent >= 0)
                for (int k = 0; k < 4; ++k)
                    out.tangent[k] = unorm8ToSigned(r.u8(p + b.layout.tangent + k));
            if (b.layout.uv >= 0)
                for (int k = 0; k < 2; ++k) out.uv[k] = r.f16(p + b.layout.uv + k * 2);
        }

        mesh.indices.reserve(triCount);
        for (uint32_t k = 0; k < triCount; ++k) {
            const uint32_t idx = r.u16(t[kIndexData].offset + (size_t(triStart) + k) * 2);
            if (idx < firstVertex || idx - firstVertex >= vertexCount)
                return fail(where + ": index out of range");
            mesh.indices.push_back(idx - firstVertex);
        }

        // Matrix list: u8 count followed by that many u8 matrix indices.
        if (matrixCount) {
            if (uint64_t(matrixOffset) + 1 + matrixCount > t[kMatrixLists].count ||
                r.u8(t[kMatrixLists].offset + matrixOffset) != matrixCount)
                return fail(where + ": bad matrix list");
            for (uint32_t k = 0; k < matrixCount; ++k)
                mesh.boneNodes.push_back(r.u8(t[kMatrixLists].offset + matrixOffset + 1 + k));
        }
        m.meshes_.push_back(std::move(mesh));
    }
    return m;
}

} // namespace formats
