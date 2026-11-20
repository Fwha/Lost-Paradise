#include "host/native/engine_capture.h"

#include "host/guest_abi.h"
#include "host/hooks.h"
#include "host/module.h"
#include "host/native/draw_origin.h"
#include "host/display_settings.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <deque>
#include <mutex>

namespace host::native {

namespace {

constexpr uint32_t kDrawIndex = 0xeac430; // engine drawIndex(ctx, count, first, baseVertex)
constexpr uint32_t kTextSize = 0x2291180;
// The stage shadow casters' drawIndex calls in 0xa51100 (return addresses): the two classes'
// intermediate and final flushes.
constexpr uint32_t kSites[] = {0xa5177a, 0xa518fe, 0xa51b79, 0xa51d37};
constexpr size_t kMaxQueued = 1 << 16;

using Fn = uint64_t(LPN_GUEST*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
Fn gDrawIndex = nullptr;
uint8_t* gBase = nullptr;
bool gOn = false;

std::mutex gMutex;
std::deque<EngineDraw> gQueue;
bool gShadows = false; // LPN_ENGINE_SHADOWS
constexpr uint32_t kCasters = 1 << 14; // captures kept (a frame has a few hundred)
std::mutex gCasterMutex;
std::vector<EngineCaster> gCasters(kCasters);
std::vector<uint32_t> gCasterIds(kCasters, ~0u);
uint32_t gNextCaster = 0;

bool readGuestFloats(uint64_t addr, float* out, size_t n) {
    if (!addr || addr >= (1ull << 40)) return false;
    std::memcpy(out, reinterpret_cast<const void*>(addr), n * 4);
    return true;
}

bool displaySettingsNative() { return displaySettings().nativeRenderer; }

template <typename T> T at(uint64_t a) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof v);
    return v;
}

// rdi: the Gnm context, whose state block is at +0x320 (render-decomp.md); esi: indices, edx:
// first index, ecx: base vertex.
uint64_t drawIndex(uint64_t ctx, uint64_t count, uint64_t first, uint64_t baseVertex, uint64_t r8, uint64_t r9) {
    const auto* frame = static_cast<const uint64_t*>(LPN_GUEST_FRAME());
    const uint64_t base = reinterpret_cast<uint64_t>(gBase);
    auto text = [&](uint64_t a) { return a >= base && a < base + kTextSize ? uint32_t(a - base) : 0u; };
    const uint32_t site = text(frame[1]);
    // The draw's origin as the wrapper would have seen it had eac430 run as the game's code (it
    // tail-calls the wrapper; called from here, the wrapper's return address is ours).
    Origin o;
    o.at[0] = site;
    const uint64_t* fp = reinterpret_cast<const uint64_t*>(frame[0]);
    for (int i = 1; i < 3 && fp && o.at[i - 1]; ++i) {
        const uint64_t f = reinterpret_cast<uint64_t>(fp), lo = reinterpret_cast<uint64_t>(frame);
        if (f <= lo || f - lo > (1u << 20) || (f & 7)) break;
        o.at[i] = text(fp[1]);
        fp = reinterpret_cast<const uint64_t*>(fp[0]);
    }
    tForcedOrigin = o;
    if (engineSite(site)) {
        const uint64_t s = at<uint64_t>(ctx + 0x320);
        EngineDraw d;
        d.site = site;
        d.stride = (at<uint32_t>(s + 0x298) >> 4) & 0xFFFF;
        d.vertexBase = at<uint64_t>(s + 0x288) + uint64_t(uint32_t(baseVertex)) * d.stride;
        // eac430: first << (1 + bit 31 of +0x690), plus its low bits and +0x688.
        const uint32_t ib = at<uint32_t>(s + 0x690);
        // In 32 bits, as eac475-eac477 compute it, then added to the 64-bit base.
        d.indexAddr = at<uint64_t>(s + 0x688) + uint32_t((uint32_t(first) << ((ib >> 31) + 1)) + (ib & 0x7FFFFFFF));
        d.count = uint32_t(count);
        for (int i = 0; i < 4; ++i) d.constants[i] = at<uint64_t>(s + 0x6c8 + 8 * i);
        d.view = at<uint64_t>(s + 0x20c8);
        d.vertexFormat = at<uint32_t>(s + 0x298);
        static std::atomic<int> copies{0};
        if (copies.fetch_add(1) < 3) {
            d.context = std::make_shared<std::vector<uint8_t>>(0x400);
            d.state = std::make_shared<std::vector<uint8_t>>(0x4000);
            std::memcpy(d.context->data(), reinterpret_cast<const void*>(ctx), 0x400);
            std::memcpy(d.state->data(), reinterpret_cast<const void*>(s), 0x4000);
        }
        const bool opaqueClass = site == 0xa5177a || site == 0xa518fe;
        if (gShadows && opaqueClass && displaySettingsNative()) {
            EngineCaster c;
            c.draw = d;
            c.vertices = at<uint32_t>(s + 0x294);
            c.index32 = (ib >> 31) != 0;
            const uint64_t view = d.view & 0xFFFFFFFFFFull;
            if (readGuestFloats(view, c.projection, 16) && readGuestFloats(view + 0x40, c.rotation, 12) &&
                readGuestFloats(view + 0x220, c.camera, 4)) {
                std::lock_guard lock(gCasterMutex);
                const uint32_t id = gNextCaster++;
                gCasters[id % kCasters] = c;
                gCasterIds[id % kCasters] = id;
                tEngineDrawId = id + 1;
            }
        } else if (gOn) {
            std::lock_guard lock(gMutex);
            if (gQueue.size() < kMaxQueued) gQueue.push_back(d);
        }
    }
    const uint64_t r = gDrawIndex(ctx, count, first, baseVertex, r8, r9);
    tForcedOrigin = {}; // not left for another draw when this one wrote none
    tEngineDrawId = 0;
    return r;
}

} // namespace

bool engineSite(uint32_t origin) {
    for (uint32_t s : kSites)
        if (origin == s) return true;
    return false;
}

bool takeEngineDraw(EngineDraw& out) {
    std::lock_guard lock(gMutex);
    if (gQueue.empty()) return false;
    out = gQueue.front();
    gQueue.pop_front();
    return true;
}

bool engineCheckOn() { return gOn && !gShadows; }
bool engineShadowsOn() { return gShadows; }

bool engineCaster(uint32_t id, EngineCaster& out) {
    std::lock_guard lock(gCasterMutex);
    if (gCasterIds[id % kCasters] != id) return false;
    out = gCasters[id % kCasters];
    return true;
}

bool installEngineCapture(Module& eboot) {
    auto on = [](const char* name) {
        const char* e = std::getenv(name);
        return e && *e && *e != '0';
    };
    gShadows = on("LPN_ENGINE_SHADOWS");
    if (!on("LPN_ENGINE_CHECK") && !gShadows) return false;
    // push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov eax, ecx
    const Hook hooks[] = {
        {kDrawIndex, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xc8}, guestHook<&drawIndex>(),
         reinterpret_cast<void**>(&gDrawIndex)},
    };
    gBase = eboot.base();
    if (!installHooks(gBase, hooks, std::size(hooks), "engine")) return false;
    gOn = true;
    if (gShadows) std::fprintf(stderr, "[native] engine capture: the stage's opaque shadow casters are drawn from the engine (LPN_ENGINE_SHADOWS)\n");
    else std::fprintf(stderr, "[native] engine capture: checking the stage shadow casters against their draws (LPN_ENGINE_CHECK)\n");
    return true;
}

} // namespace host::native
