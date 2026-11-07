// lpn-pkg: what a PS4 .pkg holds, or its files extracted (the launcher's Tools panel does the same).
//   lpn-pkg info <file.pkg>
//   lpn-pkg extract <file.pkg> [game folder]   into <game folder>/<title ID>, game/ by default
#include "pkg/pkg.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 3 || (std::strcmp(argv[1], "info") && std::strcmp(argv[1], "extract"))) {
        std::fprintf(stderr, "usage: lpn-pkg info <file.pkg>\n       lpn-pkg extract <file.pkg> [game folder]\n");
        return 2;
    }
    pkg::Info info;
    std::string err;
    if (!pkg::readInfo(argv[2], info, err)) {
        std::fprintf(stderr, "lpn-pkg: %s\n", err.c_str());
        return 1;
    }
    std::printf("content ID %s\ntitle ID   %s\ntitle      %s\ncategory   %s\nkeys       %s\n", info.contentId.c_str(),
                info.titleId.c_str(), info.title.c_str(), info.category.c_str(),
                info.fake ? "fake PKG (can be extracted)" : "retail PKG (encrypted with a PS4's keys)");
    if (!std::strcmp(argv[1], "info")) return 0;
    const std::filesystem::path out = std::filesystem::path(argc > 3 ? argv[3] : "game") / pkg::folderName(info);
    pkg::Progress pr;
    std::atomic<bool> finished{false};
    bool ok = false;
    std::thread t([&] {
        ok = pkg::extract(argv[2], out.string(), &pr, err);
        finished = true;
    });
    while (!finished) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (const uint64_t total = pr.total; total)
            std::fprintf(stderr, "\r%3d%%", int(pr.done * 100 / total));
    }
    t.join();
    std::fprintf(stderr, "\r");
    if (!ok) {
        std::fprintf(stderr, "lpn-pkg: %s\n", err.c_str());
        return 1;
    }
    std::printf("extracted %llu bytes to %s\n", (unsigned long long)pr.done.load(), out.c_str());
    return 0;
}
