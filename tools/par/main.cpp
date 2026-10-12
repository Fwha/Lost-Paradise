// lpn-par: list or extract .par archives.
//
//   lpn-par list    archive.par [--recursive]
//   lpn-par extract archive.par OUTDIR [--recursive]
//
// --recursive also descends into .par files stored inside the archive. When extracting,
// a nested archive "foo.par" is unpacked into a directory "foo.par/".

#include "formats/par.h"

#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool recursive = false;
int failures = 0;

bool readFile(const fs::path& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), {});
    return true;
}

void process(const std::vector<uint8_t>& data, const std::string& label, const fs::path* outDir,
             const std::string& indent) {
    std::string err;
    auto ar = formats::ParArchive::parse(data, &err);
    if (!ar) {
        std::fprintf(stderr, "%s: %s\n", label.c_str(), err.c_str());
        ++failures;
        return;
    }
    for (const auto& f : ar->files()) {
        if (!outDir)
            std::printf("%s%-48s %10u %10u%s\n", indent.c_str(), f.path.c_str(), f.size,
                        f.storedSize, f.compressed ? "  sllz" : "");

        std::vector<uint8_t> bytes;
        if (!ar->read(f, bytes, &err)) {
            std::fprintf(stderr, "%s/%s: %s\n", label.c_str(), f.path.c_str(), err.c_str());
            ++failures;
            continue;
        }
        const bool nested = recursive && formats::ParArchive::isPar(bytes);
        if (outDir) {
            const fs::path target = *outDir / f.path;
            if (nested) {
                fs::create_directories(target);
                process(bytes, label + "/" + f.path, &target, indent);
                continue;
            }
            fs::create_directories(target.parent_path());
            std::ofstream(target, std::ios::binary)
                .write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        } else if (nested) {
            process(bytes, label + "/" + f.path, nullptr, indent + "  ");
        }
    }
}

int usage() {
    std::fprintf(stderr, "usage: lpn-par list archive.par [--recursive]\n"
                         "       lpn-par extract archive.par OUTDIR [--recursive]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--recursive") recursive = true;
        else args.emplace_back(argv[i]);
    }
    if (args.size() < 2) return usage();

    std::vector<uint8_t> data;
    if (!readFile(args[1], data)) {
        std::fprintf(stderr, "cannot read %s\n", args[1].c_str());
        return 1;
    }
    if (args[0] == "list" && args.size() == 2) {
        process(data, args[1], nullptr, "");
    } else if (args[0] == "extract" && args.size() == 3) {
        const fs::path out = args[2];
        process(data, args[1], &out, "");
    } else {
        return usage();
    }
    return failures ? 1 : 0;
}
