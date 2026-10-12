// lpn-gmt: inspect GMT animations.
//
//   lpn-gmt info anim.gmt

#include "formats/gmt.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "info") {
        std::fprintf(stderr, "usage: lpn-gmt info anim.gmt\n");
        return 2;
    }
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    std::vector<uint8_t> data(std::istreambuf_iterator<char>(in), {});
    std::string err;
    auto file = formats::GmtFile::parse(data, &err);
    if (!file) {
        std::fprintf(stderr, "%s: %s\n", argv[2], err.c_str());
        return 1;
    }
    for (const auto& a : file->animations()) {
        std::printf("animation %s: frames %u-%u at %.0f fps, %zu bones\n", a.name.c_str(),
                    a.startFrame, a.endFrame, a.frameRate, a.bones.size());
        for (const auto& b : a.bones) {
            size_t axisKeys = 0;
            for (const auto& t : b.locationAxis) axisKeys += t.frames.size();
            std::printf("  %-24s loc %4zu  rot %4zu%s\n", b.name.c_str(), b.location.frames.size(),
                        b.rotation.frames.size(),
                        axisKeys ? ("  axis-loc " + std::to_string(axisKeys)).c_str() : "");
        }
    }
    if (!file->skippedFormats().empty()) {
        std::printf("curve formats not decoded:");
        for (uint32_t f : file->skippedFormats()) std::printf(" %u", f);
        std::printf("\n");
    }
    return 0;
}
