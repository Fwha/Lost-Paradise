#include "host/native/engine_capture.h"

#include "host/guest_abi.h"
#include "host/hooks.h"
#include "host/module.h"
#include "host/native/draw_origin.h"
#include "host/crash.h"
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
// Captures kept: a frame has a few hundred and the renderer is at most a few frames behind the
// game, so an id is drawn long before its slot comes round again (a lost one is logged).
constexpr uint32_t kCasters = 1 << 16;
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

bool readable(uint64_t a) { return a > 0x100000000ull && a < (1ull << 40); }

// What the alpha-tested casters need besides the opaque ones' (Codex's reading of eabdb0 and
// e61ae0): fetch slot 1 through the metadata at S+0x698 (table at +0x10 of 8-byte entries:
// offset/4 in bits 0-7, stream in 8-15, then V# word 3), and the texture through the pixel
// shader's slot table and S+0x8e8.
void readAlphaInputs(uint64_t ctx, uint64_t s, EngineDraw& d) {
    const uint64_t f = *reinterpret_cast<const uint64_t*>(s + 0x698);
    if (readable(f)) {
        const uint64_t t = *reinterpret_cast<const uint64_t*>(f + 0x10);
        const uint32_t n = *reinterpret_cast<const uint32_t*>(f + 0x18);
        if (readable(t) && n >= 2) {
            const uint32_t a = *reinterpret_cast<const uint32_t*>(t + 8);
            d.uvOffset = (a & 0xFF) * 4, d.uvStream = (a >> 8) & 0xFF;
            d.uvWord3 = *reinterpret_cast<const uint32_t*>(t + 12);
            d.uvKnown = !(a & 0xFF0000);
        }
    }
    const uint64_t h = *reinterpret_cast<const uint64_t*>(ctx + 0xb8);
    if (!readable(h)) return;
    for (int j = 0; j < 16; ++j) {
        const uint16_t m = *reinterpret_cast<const uint16_t*>(h + 0x1c + 2 * j);
        if (m == 0xFFFF || !(m & 0x8000) || (m & 0x3FFF) != 0) continue;
        const uint64_t p = *reinterpret_cast<const uint64_t*>(s + 0x8e8 + 8 * j);
        if (!readable(p)) return;
        std::memcpy(d.texture, reinterpret_cast<const void*>(p), 32);
        d.textureSlot = j;
        return;
    }
}

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
        readAlphaInputs(ctx, s, d);
        // Three of each class (the alpha-tested class's textures are looked for too).
        static std::atomic<int> copies[2]{};
        const bool alphaClass = site == 0xa51b79 || site == 0xa51d37;
        if (copies[alphaClass].fetch_add(1) < 3) {
            d.context = std::make_shared<std::vector<uint8_t>>(0x400);
            d.state = std::make_shared<std::vector<uint8_t>>(0x4000);
            std::memcpy(d.context->data(), reinterpret_cast<const void*>(ctx), 0x400);
            std::memcpy(d.state->data(), reinterpret_cast<const void*>(s), 0x4000);
            if (const uint64_t t = at<uint64_t>(ctx + 0x68); t > 0x100000000ull && t < (1ull << 40)) {
                d.table = std::make_shared<std::vector<uint8_t>>(0x3000);
                struct Copy {
                    void* dst;
                    uint64_t src;
                } cp{d.table->data(), t};
                // Guarded: a table shorter than this at the end of a mapping must not crash the game.
                if (!guarded([](void* c) {
                        auto* k = static_cast<Copy*>(c);
                        std::memcpy(k->dst, reinterpret_cast<const void*>(k->src), 0x3000);
                    }, &cp))
                    d.table.reset();
            }
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
    // On by default (with Native Renderer on, checked at each draw); LPN_ENGINE_SHADOWS=0 turns it off.
    // LPN_ENGINE_CHECK alone checks the game's draws instead, which then still run.
    const char* es = std::getenv("LPN_ENGINE_SHADOWS");
    const bool check = on("LPN_ENGINE_CHECK");
    gShadows = es ? *es && *es != '0' : !check;
    if (!check && !gShadows) return false;
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
