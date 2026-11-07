// lpn-pkg: the launcher's Tools from a terminal.
//   lpn-pkg info <file.pkg>                     what a package is
//   lpn-pkg extract <file.pkg> [out folder]     its files, into <out folder>/<title ID> (game/ by default)
//   lpn-pkg install <file.pkg> [game folder]    a game unpacked, an update or DLC applied (pkg/install.h)
//   lpn-pkg apply <folder> [game folder]        an unpacked update or DLC folder applied
#include "pkg/install.h"
#include "pkg/pkg.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>

namespace {

// Runs job on a thread, printing its progress.
bool withProgress(pkg::Progress& pr, const std::function<bool()>& job) {
    std::atomic<bool> finished{false};
    bool ok = false;
    std::thread t([&] {
        ok = job();
        finished = true;
    });
    while (!finished) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (const uint64_t total = pr.total; total) std::fprintf(stderr, "\r%3d%%", int(pr.done * 100 / total));
    }
    t.join();
    std::fprintf(stderr, "\r    \r");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    if (argc < 3 || (cmd != "info" && cmd != "extract" && cmd != "install" && cmd != "apply")) {
        std::fprintf(stderr, "usage: lpn-pkg info <file.pkg>\n"
                             "       lpn-pkg extract <file.pkg> [out folder]\n"
                             "       lpn-pkg install <file.pkg> [game folder]\n"
                             "       lpn-pkg apply <update or DLC folder> [game folder]\n");
        return 2;
    }
    const std::string games = argc > 3 ? argv[3] : "game";
    std::string err, message;
    pkg::Progress pr;
    if (cmd == "apply") {
        if (!withProgress(pr, [&] { return pkg::applyFolder(argv[2], games, false, &pr, message, err); })) {
            std::fprintf(stderr, "lpn-pkg: %s\n", err.c_str());
            return 1;
        }
        std::printf("%s\n", message.c_str());
        return 0;
    }
    pkg::Info info;
    if (!pkg::readInfo(argv[2], info, err)) {
        std::fprintf(stderr, "lpn-pkg: %s\n", err.c_str());
        return 1;
    }
    std::printf("content ID %s\ntitle ID   %s\ntitle      %s\ncategory   %s\nkeys       %s\n", info.contentId.c_str(),
                info.titleId.c_str(), info.title.c_str(), info.category.c_str(),
                info.fake ? "fake PKG (can be extracted)" : "retail PKG (encrypted with a PS4's keys)");
    if (cmd == "info") return 0;
    bool ok;
    if (cmd == "install") {
        ok = withProgress(pr, [&] { return pkg::installPkg(argv[2], games, &pr, message, err); });
    } else {
        const std::filesystem::path out = std::filesystem::path(games) / pkg::folderName(info);
        ok = withProgress(pr, [&] { return pkg::extract(argv[2], out.string(), &pr, err); });
        message = "extracted " + std::to_string(pr.done.load()) + " bytes to " + out.string();
    }
    if (!ok) {
        std::fprintf(stderr, "lpn-pkg: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", message.c_str());
    return 0;
}
