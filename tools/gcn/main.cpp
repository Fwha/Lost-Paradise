// lpn-gcn: disassemble PS4 shaders dumped by lpn-host --dump-frame, or summarise which
// instructions a set of shaders uses (what the shader recompiler has to cover).
#include "gcn/decoder.h"
#include "gcn/shader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> readShader(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    // Code ends where the "OrbShdr" binary info block begins.
    size_t end = bytes.size();
    for (size_t i = 0; i + 7 <= bytes.size(); i += 4)
        if (std::memcmp(&bytes[i], "OrbShdr", 7) == 0) {
            end = i;
            break;
        }
    std::vector<uint32_t> words(end / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

// Translate dumped shaders to GLSL and compile them. Descriptors and fetch shaders are not
// in the dump, so this checks the instruction coverage and that the output compiles.
int glsl(int argc, char** argv) {
    const bool print = !std::strcmp(argv[1], "--glsl");
    int failures = 0;
    for (int a = 2; a < argc; ++a) {
        const auto code = readShader(argv[a]);
        constexpr uint64_t kBase = 0x100000000ull;
        gcn::LaunchState l;
        const std::string name = std::strrchr(argv[a], '/') ? std::strrchr(argv[a], '/') + 1 : argv[a];
        l.stage = name.rfind("ps", 0) == 0 ? gcn::Stage::Pixel : name.rfind("cs", 0) == 0 ? gcn::Stage::Compute
                                                                                           : gcn::Stage::Vertex;
        l.entry = kBase;
        l.rsrc2 = 16 << 1;
        l.psInputEna = 0x302; // perspective center, position x/y
        l.colorFormat = 0x4;
        l.localSize[0] = 64;
        if (l.stage == gcn::Stage::Compute) l.rsrc2 |= (1 << 7) | (1 << 8) | (2 << 11);
        for (int i = 0; i < 32; ++i) l.psInputCntl[i] = i;
        auto read = [&](uint64_t addr, void* dst, size_t n) {
            if (addr < kBase || addr + n > kBase + code.size() * 4) return false;
            std::memcpy(dst, reinterpret_cast<const char*>(code.data()) + (addr - kBase), n);
            return true;
        };
        const gcn::Program p = gcn::analyze(l, read);
        const gcn::Translation t = gcn::translate(p, 0);
        if (print) std::printf("%s", t.glsl.c_str());
        std::string error;
        const gcn::GlslStage st = l.stage == gcn::Stage::Pixel ? gcn::GlslStage::Fragment
                                  : l.stage == gcn::Stage::Compute ? gcn::GlslStage::Compute : gcn::GlslStage::Vertex;
        const auto spirv = gcn::compileGlsl(t.glsl, st, &error);
        std::printf("%s: %zu instructions, %zu SPIR-V words%s\n", name.c_str(), p.code.size(), spirv.size(),
                    spirv.empty() ? " FAILED" : "");
        for (const auto& w : t.warnings) std::printf("  warning: %s\n", w.c_str());
        if (spirv.empty()) {
            std::printf("%s\n", error.c_str());
            ++failures;
        }
    }
    return failures ? 1 : 0;
}

} // namespace

// How the shader cache's saved programs (<name>.gcn beside each SPIR-V) differ between
// translations of the same shader: for each input a translation depends on, how many shaders
// have more than one value of it. What a draw-independent translation has to make dynamic first.
int variants(const char* dir) {
    struct Group {
        std::vector<gcn::Program> progs;
    };
    std::map<std::string, Group> groups; // stage + entry-relative code bytes of the entry region
    size_t files = 0, bad = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.path().extension() != ".gcn") continue;
        std::ifstream f(e.path(), std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(f)), {});
        gcn::Program p;
        if (bytes.size() < 4 || !gcn::deserialize(bytes.substr(4), p)) {
            ++bad;
            continue;
        }
        ++files;
        // Same shader: same stage and same instructions at the same offsets from the entry, up to
        // the end of the code the walk found (other walks of it find more or less).
        std::string key(1, char(p.launch.stage));
        int n = 0;
        for (auto it = p.code.find(p.launch.entry); it != p.code.end() && n < 24; ++it, ++n)
            key.append(reinterpret_cast<const char*>(it->second.words), it->second.size);
        key += std::to_string(p.launch.rsrc1) + "/" + std::to_string(p.launch.rsrc2);
        groups[key].progs.push_back(std::move(p));
    }
    std::map<std::string, size_t> differs; // input -> shaders with more than one value
    std::map<std::string, size_t> extra;   // input -> translations beyond one per shader it explains
    size_t multi = 0, translations = 0;
    for (auto& [k, g] : groups) {
        translations += g.progs.size();
        if (g.progs.size() < 2) continue;
        ++multi;
        auto count = [&](const char* name, auto&& field) {
            std::set<std::string> vals;
            for (const auto& p : g.progs) vals.insert(field(p));
            if (vals.size() > 1) ++differs[name], extra[name] += vals.size() - 1;
        };
        auto bytesOf = [](const void* d, size_t n) { return std::string(static_cast<const char*>(d), n); };
        count("code walked (scalar branches)", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& [pc, in] : p.code) s += std::to_string(pc - p.launch.entry) + ",";
            return s;
        });
        count("fetch shader calls", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& [site, callee] : p.calls) s += std::to_string(callee) + ",";
            return s;
        });
        count("PS input mapping", [&](const gcn::Program& p) { return bytesOf(p.launch.psInputCntl, sizeof p.launch.psInputCntl); });
        count("PS input enable", [&](const gcn::Program& p) { return std::to_string(p.launch.psInputEna); });
        count("MRT export formats", [&](const gcn::Program& p) { return std::to_string(p.launch.colorFormat); });
        count("MRT int/uint", [&](const gcn::Program& p) { return std::to_string(p.launch.colorUint) + "/" + std::to_string(p.launch.colorSint); });
        count("MRT swap/components", [&](const gcn::Program& p) {
            return bytesOf(p.launch.colorSwap, sizeof p.launch.colorSwap) + bytesOf(p.launch.colorComps, sizeof p.launch.colorComps);
        });
        count("depth export", [&](const gcn::Program& p) { return std::to_string(p.launch.depthExport); });
        count("clip z", [&](const gcn::Program& p) { return std::to_string(p.launch.glClipZ); });
        count("image dimension/shadow", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& i : p.images) s += std::to_string(i.type()) + (i.compare ? "c" : "") + (i.storage ? "s" : "") + ",";
            return s;
        });
        count("image number format (float/int/uint)", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& i : p.images) s += std::to_string(i.numFormat() == 4 ? 4 : i.numFormat() == 5 ? 5 : 0) + ",";
            return s;
        });
        count("image data format", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& i : p.images) s += std::to_string(i.dataFormat()) + ",";
            return s;
        });
        count("image swizzle", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& i : p.images) s += std::to_string(i.dstSel()) + ",";
            return s;
        });
        count("buffer formats", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& b : p.buffers) s += std::to_string(b.dfmt) + "/" + std::to_string(b.nfmt) + ",";
            return s;
        });
        count("buffer swizzle", [&](const gcn::Program& p) {
            std::string s;
            for (const auto& b : p.buffers) s += std::to_string(b.sharp[3] & 0xFFF) + ",";
            return s;
        });
    }
    std::printf("%zu saved programs (%zu unreadable), %zu distinct shaders, %zu with more than one translation\n",
                files, bad, groups.size(), multi);
    std::printf("%-40s %8s %12s\n", "input that differs", "shaders", "extra transl.");
    std::vector<std::pair<size_t, std::string>> order;
    for (const auto& [name, n] : differs) order.push_back({n, name});
    std::sort(order.rbegin(), order.rend());
    for (const auto& [n, name] : order) std::printf("%-40s %8zu %12zu\n", name.c_str(), n, extra[name]);
    std::map<size_t, size_t> hist;
    for (const auto& [k, g] : groups) ++hist[g.progs.size()];
    std::printf("translations per shader:");
    for (const auto& [n, c] : hist) std::printf(" %zux%zu", c, n);
    std::printf("\n");
    return 0;
}

// Stage 1b survey: across the cache's saved programs, what each image, buffer and colour output
// was at draw time, and whether the code alone predicts it (an image's dimension from the MIMG
// DA bit and v_cube* instructions, its int-ness from the instruction's data use is not visible,
// so float is assumed). What the boot-time compile (docs/shader-precompile.md stage 4) can know.
int resources(const char* dir) {
    std::map<std::string, size_t> imgs, bufs, mrt, psin;
    size_t images = 0, predicted = 0, programs = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.path().extension() != ".gcn") continue;
        std::ifstream f(e.path(), std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(f)), {});
        gcn::Program p;
        if (bytes.size() < 4 || !gcn::deserialize(bytes.substr(4), p)) continue;
        ++programs;
        bool cubeOps = false;
        for (const auto& [pc, in] : p.code)
            if (in.format == gcn::Format::VOP3 && in.op >= 324 && in.op <= 327) cubeOps = true;
        const char* stage = p.launch.stage == gcn::Stage::Pixel ? "ps" : p.launch.stage == gcn::Stage::Compute ? "cs" : "vs";
        for (const auto& im : p.images) {
            if (!im.t[0] && !im.t[1] && !im.t[3]) continue; // unresolved
            ++images;
            auto it = p.code.find(im.pc);
            const bool da = it != p.code.end() && ((it->second.words[0] >> 14) & 1);
            const uint32_t t = im.type();
            const char* real = t == 8 ? "1D" : t == 10 ? "3D" : t == 11 ? "Cube" : t == 12 ? "1DArray" : t == 13 ? "2DArray" : "2D";
            const char* guess = da ? (cubeOps ? "Cube" : "2DArray") : "2D";
            const uint32_t nf = im.numFormat();
            const char* kind = nf == 4 ? "uint" : nf == 5 ? "sint" : "float";
            const bool ok = !std::strcmp(real, guess) && !std::strcmp(kind, "float");
            predicted += ok;
            ++imgs[std::string(stage) + " " + real + " " + kind + (im.storage ? " storage" : "") + (im.compare ? " shadow" : "") +
                   " (code says " + guess + ")" + (ok ? "" : "  <- mismatch")];
        }
        for (const auto& b : p.buffers) {
            auto it = p.code.find(b.pc);
            if (it == p.code.end()) continue;
            const bool mtbuf = it->second.format == gcn::Format::MTBUF;
            const bool formatted = mtbuf || it->second.op <= 7; // buffer_load/store_format_*
            if (!formatted) continue;
            ++bufs[std::string(stage) + (mtbuf ? " tbuffer (format in code)" : " buffer_*_format dfmt ") +
                   (mtbuf ? "" : std::to_string(b.dfmt) + "/" + std::to_string(b.nfmt) + " sel " + std::to_string(b.sharp[3] & 0xFFF)) +
                   (p.calls.empty() ? "" : " [fetch]")];
        }
        if (p.launch.stage == gcn::Stage::Pixel) {
            for (int m = 0; m < 8; ++m) {
                const uint32_t fmt = (p.launch.colorFormat >> (m * 4)) & 0xF;
                if (!fmt) continue;
                ++mrt["export fmt " + std::to_string(fmt) + " comps " + std::to_string(p.launch.colorComps[m]) + " swap " +
                      std::to_string(p.launch.colorSwap[m]) + ((p.launch.colorUint >> m) & 1 ? " uint" : "") +
                      ((p.launch.colorSint >> m) & 1 ? " sint" : "")];
            }
            bool identity = true;
            for (uint32_t i = 0; i < 32; ++i)
                if (p.launch.psInputCntl[i] && (p.launch.psInputCntl[i] & 0x3F) != i) identity = false;
            ++psin[identity ? "PS_INPUT_CNTL: attribute i reads VS output i" : "PS_INPUT_CNTL: remapped"];
        }
    }
    auto print = [](const char* title, const std::map<std::string, size_t>& m) {
        std::printf("%s\n", title);
        std::vector<std::pair<size_t, std::string>> v;
        for (const auto& [k, n] : m) v.push_back({n, k});
        std::sort(v.rbegin(), v.rend());
        for (const auto& [n, k] : v) std::printf("  %6zu  %s\n", n, k.c_str());
    };
    std::printf("%zu programs; %zu resolved images, %zu predicted from the code alone\n", programs, images, predicted);
    print("images (stage, dimension, type):", imgs);
    print("formatted buffer accesses:", bufs);
    print("pixel shader colour outputs:", mrt);
    print("pixel shader inputs:", psin);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lpn-gcn <shader.bin>          disassemble\n"
                             "       lpn-gcn --stats <shader.bin>...  instruction usage across shaders\n"
                             "       lpn-gcn --glsl <shader.bin>...   translate to GLSL and compile\n"
                             "       lpn-gcn --check <shader.bin>...  as --glsl, printing only results\n"
                             "       lpn-gcn --variants <cache dir>   how saved translations of a shader differ\n"
                             "       lpn-gcn --resources <cache dir>  what draws bound, and what the code alone predicts\n");
        return 2;
    }
    if (!std::strcmp(argv[1], "--variants") && argc == 3) return variants(argv[2]);
    if (!std::strcmp(argv[1], "--resources") && argc == 3) return resources(argv[2]);
    if (!std::strcmp(argv[1], "--glsl") || !std::strcmp(argv[1], "--check")) return glsl(argc, argv);
    const bool stats = !std::strcmp(argv[1], "--stats");
    std::map<std::string, std::pair<unsigned, unsigned>> uses; // mnemonic -> (count, shaders)
    for (int a = stats ? 2 : 1; a < argc; ++a) {
        const auto code = readShader(argv[a]);
        std::map<std::string, unsigned> local;
        for (size_t i = 0; i < code.size();) {
            const gcn::Instruction in = gcn::decode(&code[i], code.size() - i);
            if (stats) ++local[gcn::mnemonic(in)];
            else std::printf("%6zx: %s\n", i * 4, gcn::disassemble(in, uint32_t(i * 4)).c_str());
            i += in.size / 4;
            if (gcn::isEndProgram(in)) break;
        }
        for (const auto& [m, n] : local) {
            uses[m].first += n;
            uses[m].second++;
        }
    }
    if (stats) {
        std::vector<std::pair<std::string, std::pair<unsigned, unsigned>>> v(uses.begin(), uses.end());
        std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
        std::printf("%zu distinct instructions\n", v.size());
        for (const auto& [m, n] : v) std::printf("%8u  %4u shaders  %s\n", n.first, n.second, m.c_str());
    }
    return 0;
}
