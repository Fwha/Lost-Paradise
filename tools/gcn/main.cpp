// lpn-gcn: disassemble PS4 shaders dumped by lpn-host --dump-frame, or summarise which
// instructions a set of shaders uses (what the shader recompiler has to cover).
#include "gcn/decoder.h"

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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lpn-gcn <shader.bin>          disassemble\n"
                             "       lpn-gcn --stats <shader.bin>...  instruction usage across shaders\n");
        return 2;
    }
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
