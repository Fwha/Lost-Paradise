// lpn-host: runs the original Lost Paradise eboot natively on Linux.
//
// The eboot and the game's bundled libc/Fios2 modules are mapped into this process and
// linked against each other; everything they import from the PS4 system libraries is
// bound to our C++ implementations, or to a logging stub until one exists.
#include "host/guest_abi.h"
#include "common/game_dir.h"
#include "common/mods.h"
#include "host/crash.h"
#include "host/display_settings.h"
#include "host/input_history.h"
#include "host/pause_menu.h"
#include "host/title_menu.h"
#include "host/draw_sites.h"
#include "host/gpu/frame_dump.h"
#include "host/hle.h"
#include "host/kernel.h"
#include "host/linker.h"
#include "host/module.h"
#include "host/profile.h"
#include "host/tls.h"
#include "host/gpu/renderer.h"
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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#endif

namespace fs = std::filesystem;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: lpn-host [<CUSA12776 dir>] --nids <aerolib.csv> [options]\n"
                 "  the game folder defaults to game/CUSA12776 (or another dump in game/) here\n"
                 "  --headless     no window (and no sound)\n"
                 "  --no-audio     no sound\n"
                 "  --trace-files  log every file the game opens\n"
                 "  --dump-frame N write frame N's draws and shaders to ./gpu-dump\n"
                 "  --screenshot N save the Nth displayed frame as screenshot.ppm\n"
                 "  --screenshot-every N  save every Nth displayed frame as screenshot-<frame>.ppm\n"
                 "  --press LIST   scripted button presses, e.g. cross@1800,options@2400 (frame numbers)\n"
                 "  --no-gpu       do not render (no Vulkan)\n"
                 "  --compute-only run GPU compute work but skip draws (fast; the game still progresses)\n"
                 "  --vblank HZ    display refresh rate (default 59.94; uncapped: the monitor's)\n"
                 "  --fps-cap N    show at most N frames a second, at even intervals (default: the F11 menu's; 0: uncapped)\n"
                 "  --data DIR     host folder for saves and downloads (default ./lpn-data)\n"
                 "  --mods DIR     folder of mods, replacement game files (default ./mods)\n"
                 "  --no-mods      play without mods\n"
                 "  --trace-hle N  log each system call's first use, and its first use after frame N\n"
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
#ifdef _WIN32
    // 1 ms sleeps: Windows' default timer tick is 15.6 ms, too coarse for the audio grains
    // (about 5 ms each) and the frame pacing, which sleep until a deadline.
    timeBeginPeriod(1);
#endif
    host::profile::avoidRecorderCore(); // inherited by every thread started from here
    host::profile::start();
    std::string gameDir, nidPath, dataDir = "lpn-data", modsDir = host::kModsFolder;
    bool listUnimplemented = false, headless = false, gpu = true, computeOnly = false;
    double fpsCap = -1; // -1: the F11 menu's
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--nids") && i + 1 < argc) nidPath = argv[++i];
        else if (!std::strcmp(argv[i], "--data") && i + 1 < argc) dataDir = argv[++i];
        else if (!std::strcmp(argv[i], "--mods") && i + 1 < argc) modsDir = argv[++i];
        else if (!std::strcmp(argv[i], "--no-mods")) modsDir.clear();
        else if (!std::strcmp(argv[i], "--strict")) host::hleOptions().abortOnUnimplemented = true;
        else if (!std::strcmp(argv[i], "--trace-hle") && i + 1 < argc) {
            host::hleOptions().traceHle = true;
            host::hleOptions().traceFrom = std::strtoull(argv[++i], nullptr, 0);
        }
        else if (!std::strcmp(argv[i], "--headless")) headless = true;
        else if (!std::strcmp(argv[i], "--no-audio")) host::setAudioEnabled(false);
        else if (!std::strcmp(argv[i], "--dump-frame") && i + 1 < argc)
            host::enableFrameDump(unsigned(std::strtoul(argv[++i], nullptr, 0)), "gpu-dump");
        else if (!std::strcmp(argv[i], "--press") && i + 1 < argc) {
            if (!host::addScriptedPresses(argv[++i])) {
                std::fprintf(stderr, "bad --press list: %s\n", argv[i]);
                return 1;
            }
        } else if (!std::strcmp(argv[i], "--screenshot-every") && i + 1 < argc)
            host::setScreenshotEvery(std::strtoull(argv[++i], nullptr, 0));
        else if (!std::strcmp(argv[i], "--screenshot") && i + 1 < argc)
            host::setScreenshot(std::strtoull(argv[++i], nullptr, 0), "screenshot.ppm");
        else if (!std::strcmp(argv[i], "--no-gpu")) gpu = false;
        else if (!std::strcmp(argv[i], "--compute-only")) computeOnly = true;
        else if (!std::strcmp(argv[i], "--trace-files")) host::process().traceFiles = true;
        else if (!std::strcmp(argv[i], "--vblank") && i + 1 < argc) host::setVblankRate(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--fps-cap") && i + 1 < argc) fpsCap = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--list-unimplemented")) listUnimplemented = true;
        else if (!std::strcmp(argv[i], "--log-limit") && i + 1 < argc)
            host::hleOptions().logLimitPerImport = std::strtoull(argv[++i], nullptr, 0);
        else if (argv[i][0] != '-' && gameDir.empty()) gameDir = argv[i];
        else return usage();
    }
    if (nidPath.empty()) return usage();
    if (gameDir.empty()) {
        gameDir = host::findGameDir(".");
        if (gameDir.empty()) {
            std::fprintf(stderr, "error: game not found: put the dump (the folder with eboot.bin) in %s/%s\n",
                         fs::current_path().string().c_str(), host::kGameFolder);
            return 1;
        }
        std::fprintf(stderr, "[host] game: %s\n", gameDir.c_str());
    }

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
    host::process().titleId = host::gameTitleId(gameDir);
    if (!modsDir.empty()) host::loadMods(fs::absolute(modsDir).lexically_normal().string());
    host::prepareSaveFolder();
    host::installCrashHandler();
    host::registerBuiltinHle();
    host::layoutStaticTls(modules);

    // F11 menu settings (lpn-display.ini): the render resolution has to be in place before the
    // game starts, the window mode before the window opens. --fps-cap wins over the menu's cap.
    host::loadDisplaySettings();
    host::setFpsCap(fpsCap >= 0 ? fpsCap : host::displaySettings().fps);
    if (host::prepareFrameClock(*modules[0])) host::setFrameClock(int(fpsCap >= 0 ? fpsCap : host::displaySettings().fps));
    host::patchInputHistory(*modules[0]);
    if (host::patchTitleMenu(*modules[0])) host::patchPauseMenu(*modules[0]);
    {
        uint32_t w, h; // with the Aspect Ratio's width
        host::renderSize(host::displaySettings(), &w, &h);
        host::patchRenderResolution(*modules[0], w, h);
    }
    host::patchDrawSites(*modules[0]);

    const host::ThreadPointerSlot tpSlot = host::guestThreadPointerSlot();
    for (auto* m : modules) {
        const size_t n = m->patchThreadPointerReads(tpSlot.segment, tpSlot.disp);
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
    if (headless) host::setAudioEnabled(false);
    host::startPresenter(headless);
    if (gpu) host::render::startRenderer(computeOnly);
    std::atexit(host::printUnimplementedSummary);

    // Start the bundled libraries the way the PS4 loader does: module_start(0, nullptr).
    for (size_t i = 1; i < modules.size(); ++i) {
        std::fprintf(stderr, "[host] starting %s\n", modules[i]->name().c_str());
        auto start = reinterpret_cast<int (LPN_GUEST*)(size_t, const void*, void*)>(modules[i]->entry());
        const int r = start(0, nullptr, nullptr);
        std::fprintf(stderr, "[host] %s module_start returned %d\n", modules[i]->name().c_str(), r);
    }

    std::fprintf(stderr, "[host] entering eboot at %s+0x%llx\n", modules[0]->name().c_str(),
                 (unsigned long long)(modules[0]->entry() - uint64_t(modules[0]->base())));
    static EntryParams params{1, 0, {"eboot.bin", nullptr, nullptr}};
    auto entry = reinterpret_cast<void (LPN_GUEST*)(EntryParams*, void (LPN_GUEST*)())>(modules[0]->entry());
    entry(&params, reinterpret_cast<void (LPN_GUEST*)()>(host::guestEntry<&onGuestExit>()));
    std::fprintf(stderr, "[host] eboot entry returned\n");
    return 0;
}
