// lpn-gmd: inspect GMD models and export them for viewing.
//
//   lpn-gmd info model.gmd
//   lpn-gmd obj  model.gmd out.obj [--anim anim.gmt --frame N]
//
// The OBJ export writes one group per mesh and an .mtl next to it whose map_Kd points at
// the diffuse texture's .dds name, so extracting the model's .par into the same folder
// lets Blender pick the textures up. Only full-detail ([l0]) meshes are exported.
// With --anim, the model is posed by the animation's first clip at frame N (default 0).

#include "formats/gmd.h"
#include "formats/gmt.h"
#include "formats/skeleton.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Mesh objects are root nodes named "[lN]name", N being the level of detail.
int lodOf(const std::string& nodeName) {
    if (nodeName.size() < 4 || nodeName[0] != '[' || nodeName[1] != 'l' || nodeName[3] != ']')
        return -1;
    return nodeName[2] - '0';
}

bool isLod0(const formats::GmdModel& m, const formats::GmdMesh& mesh) {
    return mesh.nodeIndex >= m.nodes().size() || lodOf(m.nodes()[mesh.nodeIndex].name) <= 0;
}

int usage() {
    std::fprintf(stderr, "usage: lpn-gmd info model.gmd\n"
                         "       lpn-gmd obj model.gmd out.obj [--anim anim.gmt --frame N]\n");
    return 2;
}

void info(const formats::GmdModel& m) {
    size_t verts = 0, tris = 0, lod0 = 0;
    for (const auto& mesh : m.meshes()) {
        verts += mesh.vertices.size();
        tris += mesh.indices.size() / 3;
        lod0 += isLod0(m, mesh);
    }
    std::printf("model %s: %zu nodes, %zu meshes, %zu attributes, %zu textures, %zu shaders\n",
                m.name().c_str(), m.nodes().size(), m.meshes().size(), m.attributes().size(),
                m.textures().size(), m.shaders().size());
    std::printf("%zu vertices, %zu triangles, %zu meshes at full detail\n", verts, tris, lod0);
    std::printf("shaders:\n");
    for (const auto& s : m.shaders()) std::printf("  %s\n", s.c_str());
    std::printf("attributes (diffuse / normal texture, shader):\n");
    for (size_t i = 0; i < m.attributes().size(); ++i) {
        const auto& a = m.attributes()[i];
        auto tex = [&](int slot) {
            const int t = a.textures[slot];
            return t >= 0 ? m.textures()[t].c_str() : "-";
        };
        std::printf("  %2zu  %-28s %-28s %s\n", i, tex(formats::kGmdDiffuse),
                    tex(formats::kGmdNormal),
                    a.shaderIndex < m.shaders().size() ? m.shaders()[a.shaderIndex].c_str() : "?");
    }
    std::printf("skeleton roots:");
    for (const auto& n : m.nodes())
        if (n.parent < 0 && lodOf(n.name) < 0) std::printf(" %s", n.name.c_str());
    std::printf("\n");
}

bool readFile(const char* path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), {});
    return true;
}

bool writeObj(const formats::GmdModel& m, const fs::path& out, const formats::GmtAnimation* anim,
              float frame) {
    const auto world = formats::poseNodes(m, anim, frame);
    fs::path mtlPath = out;
    mtlPath.replace_extension(".mtl");
    std::ofstream obj(out), mtl(mtlPath);
    if (!obj || !mtl) return false;

    for (size_t i = 0; i < m.attributes().size(); ++i) {
        const int diffuse = m.attributes()[i].textures[formats::kGmdDiffuse];
        mtl << "newmtl attr" << i << "\nKd 1 1 1\n";
        if (diffuse >= 0) mtl << "map_Kd " << m.textures()[diffuse] << ".dds\n";
        mtl << '\n';
    }

    obj << "# " << m.name() << "\nmtllib " << mtlPath.filename().string() << '\n';
    size_t base = 1;
    for (size_t i = 0; i < m.meshes().size(); ++i) {
        const auto& mesh = m.meshes()[i];
        if (!isLod0(m, mesh)) continue; // lower-detail copies would overlap the model
        obj << "g mesh" << i << "\nusemtl attr" << mesh.attributeIndex << '\n';
        const auto positions = anim ? formats::skinMesh(m, mesh, world) : std::vector<formats::Vec3>{};
        for (size_t vi = 0; vi < mesh.vertices.size(); ++vi) {
            const auto& v = mesh.vertices[vi];
            const auto& p = anim ? positions[vi] : v.position;
            obj << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
            // DDS rows run top-down; OBJ texture space runs bottom-up.
            obj << "vt " << v.uv[0] << ' ' << 1.0f - v.uv[1] << '\n';
            obj << "vn " << v.normal[0] << ' ' << v.normal[1] << ' ' << v.normal[2] << '\n';
        }
        for (size_t k = 0; k + 2 < mesh.indices.size(); k += 3) {
            obj << 'f';
            for (int c = 0; c < 3; ++c) {
                const size_t idx = base + mesh.indices[k + c];
                obj << ' ' << idx << '/' << idx << '/' << idx;
            }
            obj << '\n';
        }
        base += mesh.vertices.size();
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string cmd = argv[1];
    std::vector<uint8_t> data;
    if (!readFile(argv[2], data)) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }

    std::string err;
    auto model = formats::GmdModel::parse(data, &err);
    if (!model) {
        std::fprintf(stderr, "%s: %s\n", argv[2], err.c_str());
        return 1;
    }
    if (cmd == "info" && argc == 3) {
        info(*model);
        return 0;
    }
    if (cmd == "obj" && argc >= 4) {
        const char* animPath = nullptr;
        float frame = 0;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--anim" && i + 1 < argc) animPath = argv[++i];
            else if (a == "--frame" && i + 1 < argc) frame = std::stof(argv[++i]);
            else return usage();
        }
        std::optional<formats::GmtFile> gmt;
        if (animPath) {
            std::vector<uint8_t> animData;
            if (!readFile(animPath, animData)) {
                std::fprintf(stderr, "cannot read %s\n", animPath);
                return 1;
            }
            gmt = formats::GmtFile::parse(animData, &err);
            if (!gmt || gmt->animations().empty()) {
                std::fprintf(stderr, "%s: %s\n", animPath, gmt ? "no animations" : err.c_str());
                return 1;
            }
        }
        if (!writeObj(*model, argv[3], gmt ? &gmt->animations()[0] : nullptr, frame)) {
            std::fprintf(stderr, "cannot write %s\n", argv[3]);
            return 1;
        }
        return 0;
    }
    return usage();
}
