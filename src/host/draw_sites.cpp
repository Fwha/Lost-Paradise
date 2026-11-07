#include "host/guest_abi.h"
#include "host/draw_sites.h"

#include "host/hooks.h"
#include "host/module.h"
#include "host/videoout.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace host {

namespace {

// The eboot's statically linked Gnm command buffer layer (0xf9fd30-0xfa6e20) wraps the
// libSceGnmDriver packet calls; every draw and dispatch goes through one of these.
constexpr uint32_t kDrawIndexAuto = 0xfa09b0; // the only caller of sceGnmDrawIndexAuto
constexpr uint32_t kDrawIndex = 0xfa0a10;     // sceGnmDrawIndex
constexpr uint32_t kDispatchDirect = 0xfa0a80; // sceGnmDispatchDirect
constexpr uint32_t kTextSize = 0x2291180;
constexpr int kDepth = 4;
constexpr int kTop = 20;

enum Kind : uint32_t { kAuto, kIndexed, kDispatch };
constexpr const char* kKindName[] = {"auto", "index", "dispatch"};

using Fn = uint64_t (LPN_GUEST*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
Fn gAuto = nullptr, gIndexed = nullptr, gDispatch = nullptr;
uint8_t* gBase = nullptr;

struct Key {
    std::array<uint32_t, kDepth> at{}; // eboot offsets of return addresses, innermost first
    uint32_t kind = 0;
    bool operator==(const Key& o) const { return at == o.at && kind == o.kind; }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        uint64_t h = k.kind;
        for (uint32_t a : k.at) h = h * 0x9e3779b97f4a7c15ull + a;
        return size_t(h ^ (h >> 29));
    }
};

std::mutex gMutex;
std::unordered_map<Key, uint64_t, KeyHash> gCounts;
uint64_t gTotal[3] = {};
uint64_t gLastFlip = 0;
std::chrono::steady_clock::time_point gLast;

uint32_t textOffset(uint64_t a) {
    const uint64_t b = reinterpret_cast<uint64_t>(gBase);
    return a >= b && a < b + kTextSize ? uint32_t(a - b) : 0;
}

void report(std::chrono::steady_clock::time_point now) {
    const uint64_t flip = flipCount();
    const uint64_t frames = flip > gLastFlip ? flip - gLastFlip : 1;
    std::vector<std::pair<uint64_t, Key>> rows;
    rows.reserve(gCounts.size());
    for (auto& [k, n] : gCounts) rows.push_back({n, k});
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "[sites] flip=%llu frames=%llu per frame: %llu draws (%llu auto, %llu indexed), %llu dispatches; %zu chains\n",
                 (unsigned long long)flip, (unsigned long long)frames, (unsigned long long)((gTotal[0] + gTotal[1]) / frames),
                 (unsigned long long)(gTotal[0] / frames), (unsigned long long)(gTotal[1] / frames),
                 (unsigned long long)(gTotal[2] / frames), rows.size());
    for (size_t i = 0; i < rows.size() && i < size_t(kTop); ++i) {
        const Key& k = rows[i].second;
        std::fprintf(stderr, "[sites]   %6llu/frame %-8s 0x%x <- 0x%x <- 0x%x <- 0x%x\n",
                     (unsigned long long)(rows[i].first / frames), kKindName[k.kind], k.at[0], k.at[1], k.at[2], k.at[3]);
    }
    gCounts.clear();
    gTotal[0] = gTotal[1] = gTotal[2] = 0;
    gLastFlip = flip;
    gLast = now;
}

// `frame` is this hook's own frame: [frame] holds the game caller's %rbp, [frame+8] the return
// address into it. The game is built with frame pointers, so its %rbp chain continues from there.
void count(Kind kind, const uint64_t* frame) {
    Key k;
    k.kind = kind;
    k.at[0] = textOffset(frame[1]);
    const uint64_t* fp = reinterpret_cast<const uint64_t*>(frame[0]);
    for (int i = 1; i < kDepth && fp; ++i) {
        // Stay on this stack: frames above ours, at most 1 MB up, 8-byte aligned.
        const uint64_t f = reinterpret_cast<uint64_t>(fp), lo = reinterpret_cast<uint64_t>(frame);
        if (f <= lo || f - lo > (1u << 20) || (f & 7)) break;
        k.at[i] = textOffset(fp[1]);
        if (!k.at[i]) break;
        fp = reinterpret_cast<const uint64_t*>(fp[0]);
    }
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(gMutex);
    ++gCounts[k];
    ++gTotal[kind];
    if (now - gLast >= std::chrono::seconds(1)) report(now);
}

uint64_t drawAuto(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
    count(kAuto, static_cast<const uint64_t*>(LPN_GUEST_FRAME()));
    return gAuto(a, b, c, d, e, f);
}
uint64_t drawIndexed(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
    count(kIndexed, static_cast<const uint64_t*>(LPN_GUEST_FRAME()));
    return gIndexed(a, b, c, d, e, f);
}
uint64_t dispatch(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
    count(kDispatch, static_cast<const uint64_t*>(LPN_GUEST_FRAME()));
    return gDispatch(a, b, c, d, e, f);
}

} // namespace

bool patchDrawSites(Module& eboot) {
    const char* e = std::getenv("LPN_DRAW_SITES");
    if (!e || !*e || *e == '0') return false;
    // First instructions of each wrapper, moved to the trampoline (none are rip-relative).
    const Hook hooks[] = {
        {kDrawIndexAuto, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xfb, 0x41, 0x89, 0xf6},
         guestHook<&drawAuto>(), reinterpret_cast<void**>(&gAuto)},
        {kDrawIndex, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x48, 0x89, 0xfb},
         guestHook<&drawIndexed>(), reinterpret_cast<void**>(&gIndexed)},
        {kDispatchDirect, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xfb},
         guestHook<&dispatch>(), reinterpret_cast<void**>(&gDispatch)},
    };
    gBase = eboot.base();
    gLast = std::chrono::steady_clock::now();
    if (!installHooks(gBase, hooks, std::size(hooks), "sites")) return false;
    std::fprintf(stderr, "[sites] counting draws by call chain (LPN_DRAW_SITES)\n");
    return true;
}

} // namespace host
