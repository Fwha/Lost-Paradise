// GMD (GSGM) models: meshes, skeletons and material bindings.
//
// Lost Paradise uses the little-endian "GSGM" layout shared with RGG Studio's
// Yakuza 0 / Kiwami era engine. See docs/formats.md for the byte layout; the vertex
// layout flags in particular are inferred from samples and may not cover every model.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace formats {

// Row-major 4x4 matrix for column vectors: p' = M * p, translation in m[3], m[7], m[11].
using GmdMatrix = std::array<float, 16>;

struct GmdNode {
    std::string name;
    int32_t parent = -1;
    int32_t firstChild = -1;
    int32_t nextSibling = -1;
    int32_t objectIndex = -1;
    int32_t matrixIndex = -1; // into GmdModel::inverseBindMatrices when a skinning bone
    uint32_t nodeType = 0;
    std::array<float, 3> position{};
    std::array<float, 4> rotation{0, 0, 0, 1}; // quaternion xyzw
    std::array<float, 3> scale{1, 1, 1};
    std::array<float, 3> worldPosition{};
};

// Texture slots used by attributes. -1 means unused.
enum GmdTextureSlot {
    kGmdDiffuse = 0, // _di
    kGmdMulti = 1,   // _am
    kGmdMaterial = 2, // _mt
    kGmdNormal = 5,  // _tn
    kGmdTextureSlots = 8,
};

struct GmdAttribute {
    uint32_t materialIndex = 0;
    uint32_t shaderIndex = 0;
    uint32_t flags = 0;
    std::array<int32_t, kGmdTextureSlots> textures{};
};

struct GmdVertex {
    std::array<float, 3> position{};
    std::array<float, 3> normal{};
    std::array<float, 4> tangent{};
    std::array<float, 2> uv{};
    std::array<uint8_t, 4> bones{};   // indices into the mesh's matrix list
    std::array<float, 4> weights{};
};

struct GmdMesh {
    uint32_t attributeIndex = 0;
    uint32_t vertexBufferIndex = 0;
    uint32_t objectIndex = 0;
    uint32_t nodeIndex = 0;
    std::vector<GmdVertex> vertices;
    std::vector<uint32_t> indices;   // triangle list, relative to `vertices`
    std::vector<uint32_t> boneMatrices; // vertex bone slot -> inverse bind matrix index
};

class GmdModel {
public:
    static std::optional<GmdModel> parse(std::span<const uint8_t> data, std::string* error);
    static bool isGmd(std::span<const uint8_t> data);

    const std::string& name() const { return name_; }
    const std::vector<GmdNode>& nodes() const { return nodes_; }
    const std::vector<GmdMesh>& meshes() const { return meshes_; }
    const std::vector<GmdAttribute>& attributes() const { return attributes_; }
    const std::vector<std::string>& textures() const { return textures_; }
    const std::vector<std::string>& shaders() const { return shaders_; }
    const std::vector<GmdMatrix>& inverseBindMatrices() const { return inverseBind_; }

private:
    std::string name_;
    std::vector<GmdNode> nodes_;
    std::vector<GmdMesh> meshes_;
    std::vector<GmdAttribute> attributes_;
    std::vector<std::string> textures_;
    std::vector<std::string> shaders_;
    std::vector<GmdMatrix> inverseBind_;
};

// Byte layout of one vertex, derived from a vertex buffer's packing flags.
struct GmdVertexLayout {
    uint32_t stride = 0;
    int weights = -1, bones = -1, normal = -1, tangent = -1, uv = -1; // byte offsets
};
std::optional<GmdVertexLayout> gmdVertexLayout(uint64_t flags, std::string* error);

} // namespace formats
