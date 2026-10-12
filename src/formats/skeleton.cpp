#include "formats/skeleton.h"

#include <functional>

namespace formats {

GmdMatrix matIdentity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }

GmdMatrix matMultiply(const GmdMatrix& a, const GmdMatrix& b) {
    GmdMatrix m{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k) m[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    return m;
}

GmdMatrix matFromTrs(const Vec3& t, const Quat& q, const Vec3& s) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
    GmdMatrix m = matIdentity();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) m[row * 4 + col] = r[row * 3 + col] * s[col];
        m[row * 4 + 3] = t[row];
    }
    return m;
}

Vec3 matTransformPoint(const GmdMatrix& m, const Vec3& p) {
    Vec3 out;
    for (int r = 0; r < 3; ++r)
        out[r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
    return out;
}

std::vector<GmdMatrix> poseNodes(const GmdModel& model, const GmtAnimation* anim, float frame) {
    const auto& nodes = model.nodes();
    std::vector<GmdMatrix> local(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GmdNode& n = nodes[i];
        Vec3 pos = n.position;
        Quat rot = n.rotation;
        if (const GmtBone* b = anim ? anim->find(n.name) : nullptr) {
            if (!b->location.empty()) pos = sampleTrack(b->location, frame);
            for (int axis = 0; axis < 3; ++axis)
                if (!b->locationAxis[axis].empty()) pos[axis] = sampleTrack(b->locationAxis[axis], frame);
            if (!b->rotation.empty()) rot = sampleTrack(b->rotation, frame);
        }
        local[i] = matFromTrs(pos, rot, n.scale);
    }

    std::vector<GmdMatrix> world(nodes.size());
    std::vector<int> state(nodes.size(), 0); // 0 = todo, 1 = in progress, 2 = done
    std::function<const GmdMatrix&(size_t)> resolve = [&](size_t i) -> const GmdMatrix& {
        if (state[i] == 2) return world[i];
        const int p = nodes[i].parent;
        // A cycle in the hierarchy would be malformed data; treat the node as a root.
        if (state[i] == 1 || p < 0 || size_t(p) >= nodes.size()) {
            world[i] = local[i];
        } else {
            state[i] = 1;
            world[i] = matMultiply(resolve(size_t(p)), local[i]);
        }
        state[i] = 2;
        return world[i];
    };
    for (size_t i = 0; i < nodes.size(); ++i) resolve(i);
    return world;
}

std::vector<Vec3> skinMesh(const GmdModel& model, const GmdMesh& mesh,
                           const std::vector<GmdMatrix>& world) {
    std::vector<Vec3> out(mesh.vertices.size());

    if (mesh.boneMatrices.empty()) {
        const GmdMatrix m = mesh.nodeIndex < world.size() ? world[mesh.nodeIndex] : matIdentity();
        for (size_t v = 0; v < out.size(); ++v) out[v] = matTransformPoint(m, mesh.vertices[v].position);
        return out;
    }

    // Skinning matrix per bone slot: node world * inverse bind.
    const auto& inverseBind = model.inverseBindMatrices();
    std::vector<int> nodeOfMatrix(inverseBind.size(), -1);
    for (size_t i = 0; i < model.nodes().size(); ++i) {
        const int mi = model.nodes()[i].matrixIndex;
        if (mi >= 0 && size_t(mi) < inverseBind.size()) nodeOfMatrix[mi] = int(i);
    }
    std::vector<GmdMatrix> skin;
    for (uint32_t mi : mesh.boneMatrices) {
        const int node = nodeOfMatrix[mi];
        skin.push_back(node >= 0 ? matMultiply(world[node], inverseBind[mi]) : matIdentity());
    }

    for (size_t v = 0; v < out.size(); ++v) {
        const GmdVertex& vx = mesh.vertices[v];
        Vec3 acc{};
        for (int k = 0; k < 4; ++k) {
            if (vx.weights[k] == 0 || vx.bones[k] >= skin.size()) continue;
            const Vec3 p = matTransformPoint(skin[vx.bones[k]], vx.position);
            for (int c = 0; c < 3; ++c) acc[c] += vx.weights[k] * p[c];
        }
        out[v] = acc;
    }
    return out;
}

} // namespace formats
