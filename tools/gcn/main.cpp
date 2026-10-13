// lpn-gcn: disassemble PS4 shaders dumped by lpn-host --dump-frame, or summarise which
// instructions a set of shaders uses (what the shader recompiler has to cover).
#include "gcn/decoder.h"
#include "gcn/shader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lpn-gcn <shader.bin>          disassemble\n"
                             "       lpn-gcn --stats <shader.bin>...  instruction usage across shaders\n"
                             "       lpn-gcn --glsl <shader.bin>...   translate to GLSL and compile\n"
                             "       lpn-gcn --check <shader.bin>...  as --glsl, printing only results\n");
        return 2;
    }
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
