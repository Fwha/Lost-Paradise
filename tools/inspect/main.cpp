// lpn-inspect: summarise a decrypted PS4 executable and export analysis data for Ghidra.
//
//   lpn-inspect eboot.bin [--nids aerolib.csv] [--out DIR]
//
// Writes DIR/functions.csv, DIR/imports.csv and DIR/segments.csv, which
// ghidra/ImportLpnAnalysis.py applies to a Ghidra project.

#include "ps4elf/dynlib.h"
#include "ps4elf/ehframe.h"
#include "ps4elf/elf.h"
#include "ps4elf/nid.h"

#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>

namespace fs = std::filesystem;

static int usage() {
    std::fprintf(stderr, "usage: lpn-inspect eboot.bin [--nids aerolib.csv] [--out DIR]\n");
    return 2;
}

int main(int argc, char** argv) {
    std::string input, nidPath, outDir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--nids" && i + 1 < argc) nidPath = argv[++i];
        else if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (!a.empty() && a[0] != '-' && input.empty()) input = a;
        else return usage();
    }
    if (input.empty()) return usage();

    std::string err;
    auto elf = ps4elf::Elf::load(input, &err);
    if (!elf) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    auto dyn = ps4elf::Dynlib::parse(*elf, &err);
    if (!dyn) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    ps4elf::NidDb nids;
    if (!nidPath.empty() && !nids.load(nidPath, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    auto funcs = ps4elf::functionsFromEhFrame(*elf, &err);
    if (funcs.empty()) std::fprintf(stderr, "warning: no functions from .eh_frame: %s\n", err.c_str());
    const auto imports = dyn->imports();

    std::printf("type 0x%04x  entry 0x%" PRIx64 "  image 0x%" PRIx64 "\n", elf->type(),
                elf->entry(), elf->imageSize());
    for (const auto& s : elf->segments())
        std::printf("  seg type 0x%08x  vaddr 0x%08" PRIx64 "  filesz 0x%08" PRIx64
                    "  memsz 0x%08" PRIx64 "  flags %c%c%c\n",
                    s.type, s.vaddr, s.filesz, s.memsz, s.flags & 4 ? 'R' : '-',
                    s.flags & 2 ? 'W' : '-', s.flags & 1 ? 'X' : '-');
    std::printf("needed modules: %zu  import libraries: %zu\n", dyn->modules().size(),
                dyn->libraries().size());
    std::printf("relocations: %zu  import relocations: %zu  functions (eh_frame): %zu\n",
                dyn->relocations().size(), imports.size(), funcs.size());

    std::map<std::string, std::pair<int, int>> perLib; // unique functions, named
    std::set<std::pair<std::string, std::string>> seen;
    for (const auto& imp : imports) {
        if (!seen.insert({imp.library, imp.nid}).second) continue;
        auto& [total, named] = perLib[imp.library];
        ++total;
        named += !nids.lookup(imp.nid).empty();
    }
    for (const auto& [lib, n] : perLib)
        std::printf("  %-28s %4d imports%s\n", lib.c_str(), n.first,
                    nidPath.empty() ? "" : (" (" + std::to_string(n.second) + " named)").c_str());

    if (outDir.empty()) return 0;
    fs::create_directories(outDir);

    std::ofstream seg(fs::path(outDir) / "segments.csv");
    seg << "type,offset,vaddr,filesz,memsz,flags\n";
    for (const auto& s : elf->segments()) {
        char line[128];
        std::snprintf(line, sizeof line, "0x%x,0x%" PRIx64 ",0x%" PRIx64 ",0x%" PRIx64 ",0x%" PRIx64 ",%u\n",
                      s.type, s.offset, s.vaddr, s.filesz, s.memsz, s.flags);
        seg << line;
    }

    std::ofstream fn(fs::path(outDir) / "functions.csv");
    fn << "start,size\n";
    for (const auto& f : funcs) {
        char line[64];
        std::snprintf(line, sizeof line, "0x%" PRIx64 ",%" PRIu64 "\n", f.start, f.size);
        fn << line;
    }

    std::ofstream im(fs::path(outDir) / "imports.csv");
    im << "slot,reloc,library,module,nid,name\n";
    for (const auto& imp : imports) {
        char slot[32];
        std::snprintf(slot, sizeof slot, "0x%" PRIx64, imp.slot);
        im << slot << ',' << imp.relocType << ',' << imp.library << ',' << imp.module << ','
           << imp.nid << ',' << nids.lookup(imp.nid) << '\n';
    }

    std::printf("wrote %s/{segments,functions,imports}.csv\n", outDir.c_str());
    return 0;
}
