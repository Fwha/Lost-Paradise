#include "host/coverage.h"

#include "host/kernel.h"
#include "host/module.h"
#include "host/videoout.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

namespace host::coverage {

namespace {

struct Hit {
    uint32_t fn, caller; // eboot offsets (caller 0: not in the eboot)
    uint64_t flip;
    int phase;
};

constexpr size_t kMaxHits = 1 << 17;
Hit gHits[kMaxHits];
std::atomic<size_t> gHitCount{0};
std::atomic<int> gPhase{0}; // 0 off, 1 armed (warm-up), 2.. logged phases
uint8_t* gText = nullptr;
size_t gTextSize = 0;
std::vector<uint8_t> gOriginal; // first byte of each armed entry, by offset (0: not armed)
std::vector<uint8_t> gArmed;
struct sigaction gPrevTrap{};
uintptr_t gPage = 4096;

void onTrap(int sig, siginfo_t* info, void* ctx) {
    auto& g = static_cast<ucontext_t*>(ctx)->uc_mcontext.gregs;
    const uint64_t at = uint64_t(g[REG_RIP]) - 1;
    const uint64_t off = at - uint64_t(gText);
    if (gText && off < gTextSize && gArmed[off]) {
        // The frame clock's patches may have made the page read-only again since arming.
        uint8_t* p = gText + off;
        mprotect(reinterpret_cast<void*>(uintptr_t(p) & ~(gPage - 1)), gPage * 2, PROT_READ | PROT_WRITE | PROT_EXEC);
        *p = gOriginal[off];
        __atomic_store_n(&gArmed[off], 0, __ATOMIC_RELEASE);
        g[REG_RIP] = int64_t(at);
        const uint64_t ret = *reinterpret_cast<const uint64_t*>(g[REG_RSP]) - uint64_t(gText);
        const size_t i = gHitCount.fetch_add(1);
        if (i < kMaxHits) gHits[i] = {uint32_t(off), ret < gTextSize ? uint32_t(ret) : 0, flipCount(), gPhase.load()};
        return;
    }
    if (gText && off < gTextSize && gOriginal[off] && *(gText + off) != 0xcc) {
        g[REG_RIP] = int64_t(at); // another thread restored it first
        return;
    }
    if (gPrevTrap.sa_flags & SA_SIGINFO) gPrevTrap.sa_sigaction(sig, info, ctx);
    else _exit(128 + sig);
}

void flusher() {
    size_t shown = 0, warm = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        const size_t n = std::min(gHitCount.load(), kMaxHits);
        for (; shown < n; ++shown) {
            const Hit& h = gHits[shown];
            if (h.phase < 2) {
                ++warm;
                continue;
            }
            std::fprintf(stderr, "[cover] phase %d flip %llu fn 0x%x from 0x%x\n", h.phase,
                         (unsigned long long)h.flip, h.fn, h.caller);
        }
        static size_t lastWarm = 0;
        if (warm != lastWarm && gPhase.load() == 1) {
            static auto last = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
                std::fprintf(stderr, "[cover] warm-up: %zu functions seen\n", warm);
                last = std::chrono::steady_clock::now(), lastWarm = warm;
            }
        }
    }
}

void arm() {
    Module* m = process().eboot;
    if (!m) return;
    const char* list = std::getenv("LPN_COVERAGE");
    const char* path = list && std::strcmp(list, "1") ? list : "analysis/functions.csv";
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[cover] cannot read %s\n", path);
        return;
    }
    gText = m->base();
    gTextSize = 0x2291180; // the eboot's code segment
    gPage = uintptr_t(sysconf(_SC_PAGESIZE));
    gOriginal.assign(gTextSize, 0);
    gArmed.assign(gTextSize, 0);
    std::vector<uint32_t> entries;
    char line[128];
    while (std::fgets(line, sizeof line, f)) {
        unsigned long start = 0, size = 0;
        if (std::sscanf(line, "%lx,%lu", &start, &size) != 2 || start + 4 > gTextSize) continue;
        // Only entries that open a frame: push %rbp; mov %rsp,%rbp. Anything else in the list
        // could be data or a mid-function address, where an int3 would do harm.
        if (std::memcmp(gText + start, "\x55\x48\x89\xe5", 4) == 0) entries.push_back(uint32_t(start));
    }
    std::fclose(f);

    struct sigaction sa{};
    sa.sa_sigaction = onTrap;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGTRAP, &sa, &gPrevTrap);
    mprotect(gText, (gTextSize + gPage - 1) & ~(gPage - 1), PROT_READ | PROT_WRITE | PROT_EXEC);
    for (uint32_t e : entries) {
        gOriginal[e] = gText[e];
        gArmed[e] = 1;
    }
    gPhase = 1;
    for (uint32_t e : entries) __atomic_store_n(gText + e, uint8_t(0xcc), __ATOMIC_RELEASE);
    std::thread(flusher).detach();
    std::fprintf(stderr, "[cover] armed %zu function entries (flip %llu): warm-up, F9 again starts phase 2\n",
                 entries.size(), (unsigned long long)flipCount());
}

} // namespace

void onKey() {
    static const bool enabled = std::getenv("LPN_COVERAGE") != nullptr;
    if (!enabled) return;
    if (gPhase == 0) return arm();
    const int phase = ++gPhase;
    std::fprintf(stderr, "[cover] phase %d starts at flip %llu (%zu functions seen so far)\n", phase,
                 (unsigned long long)flipCount(), std::min(gHitCount.load(), kMaxHits));
}

} // namespace host::coverage
