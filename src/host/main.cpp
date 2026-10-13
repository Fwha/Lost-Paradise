// lpn-host: runs the original Lost Paradise eboot natively on Linux.
//
// The eboot and the game's bundled libc/Fios2 modules are mapped into this process and
// linked against each other; everything they import from the PS4 system libraries is
// bound to our C++ implementations, or to a logging stub until one exists.
#include "host/crash.h"
#include "host/hle.h"
#include "host/kernel.h"
#include "host/linker.h"
#include "host/module.h"
#include "host/tls.h"
#include "host/videoout.h"
#include "host/window.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: lpn-host <CUSA12776 dir> --nids <aerolib.csv> [options]\n"
                 "  --headless     no window\n"
                 "  --trace-files  log every file the game opens\n"
                 "  --vblank HZ    display refresh rate (default 59.94)\n"
                 "  --data DIR     host folder for saves and downloads (default ./lpn-data)\n"
                 "  --strict       stop at the first call to an unimplemented import\n"
                 "  --log-limit N  log the first N calls to each unimplemented import (default 3)\n"
                 "  --list-unimplemented  print every import still bound to a logging stub\n");
    return 2;
}

struct EntryParams {
    int argc;
    uint32_t pad;
    const char* argv[3];
};

void onGuestExit() { std::fprintf(stderr, "[host] guest exit handler called\n"); }

} // namespace

int main(int argc, char** argv) {
    std::string gameDir, nidPath, dataDir = "lpn-data";
    bool listUnimplemented = false, headless = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--nids") && i + 1 < argc) nidPath = argv[++i];
        else if (!std::strcmp(argv[i], "--data") && i + 1 < argc) dataDir = argv[++i];
        else if (!std::strcmp(argv[i], "--strict")) host::hleOptions().abortOnUnimplemented = true;
        else if (!std::strcmp(argv[i], "--headless")) headless = true;
        else if (!std::strcmp(argv[i], "--trace-files")) host::process().traceFiles = true;
        else if (!std::strcmp(argv[i], "--vblank") && i + 1 < argc) host::setVblankRate(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--list-unimplemented")) listUnimplemented = true;
        else if (!std::strcmp(argv[i], "--log-limit") && i + 1 < argc)
            host::hleOptions().logLimitPerImport = std::strtoull(argv[++i], nullptr, 0);
        else if (argv[i][0] != '-' && gameDir.empty()) gameDir = argv[i];
        else return usage();
    }
    if (gameDir.empty() || nidPath.empty()) return usage();

    std::string error;
    ps4elf::NidDb nids;
    if (!nids.load(nidPath, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    // The eboot comes first: it gets TLS module id 1, as on the console.
    std::vector<std::unique_ptr<host::Module>> owned;
    std::vector<fs::path> paths{fs::path(gameDir) / "eboot.bin"};
    for (const char* prx : {"libc.prx", "libSceFios2.prx"}) {
        const fs::path p = fs::path(gameDir) / "sce_module" / prx;
        if (fs::exists(p)) paths.push_back(p);
    }
    for (const auto& p : paths) {
        auto m = host::Module::load(p.string(), &error);
        if (!m) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        std::fprintf(stderr, "[host] loaded %-16s at %p (0x%zx bytes)\n", m->name().c_str(),
                     static_cast<void*>(m->base()), m->size());
        owned.push_back(std::move(m));
    }
    std::vector<host::Module*> modules;
    for (auto& m : owned) modules.push_back(m.get());

    host::process().eboot = modules[0];
    host::process().gameDir = fs::absolute(gameDir).string();
    host::process().dataDir = fs::absolute(dataDir).string();
    host::installCrashHandler();
    host::registerBuiltinHle();
    host::layoutStaticTls(modules);

    const int32_t fsOffset = host::guestThreadPointerFsOffset();
    for (auto* m : modules) {
        const size_t n = m->patchThreadPointerReads(fsOffset);
        std::fprintf(stderr, "[host] %s: %zu thread pointer reads redirected\n", m->name().c_str(), n);
    }

    host::LinkStats stats;
    if (!host::linkModules(modules, nids, &stats, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr,
                 "[host] %zu relocations applied; imports: %zu to game modules, %zu to HLE, "
                 "%zu unimplemented\n",
                 stats.relocations, stats.toModules, stats.toHle, stats.unimplemented);
    if (listUnimplemented) {
        std::sort(stats.unimplementedNames.begin(), stats.unimplementedNames.end());
        for (const auto& n : stats.unimplementedNames) std::printf("%s\n", n.c_str());
        return 0;
    }

    for (auto* m : modules)
        if (!m->protect(&error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
    host::createGuestTls();
    host::startPresenter(headless);
    std::atexit(host::printUnimplementedSummary);

    // Start the bundled libraries the way the PS4 loader does: module_start(0, nullptr).
    for (size_t i = 1; i < modules.size(); ++i) {
        std::fprintf(stderr, "[host] starting %s\n", modules[i]->name().c_str());
        auto start = reinterpret_cast<int (*)(size_t, const void*, void*)>(modules[i]->entry());
        const int r = start(0, nullptr, nullptr);
        std::fprintf(stderr, "[host] %s module_start returned %d\n", modules[i]->name().c_str(), r);
    }

    std::fprintf(stderr, "[host] entering eboot at %s+0x%lx\n", modules[0]->name().c_str(),
                 modules[0]->entry() - uint64_t(modules[0]->base()));
    static EntryParams params{1, 0, {"eboot.bin", nullptr, nullptr}};
    auto entry = reinterpret_cast<void (*)(EntryParams*, void (*)())>(modules[0]->entry());
    entry(&params, onGuestExit);
    std::fprintf(stderr, "[host] eboot entry returned\n");
    return 0;
}
