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
#include <algorithm>
#include <atomic>
#include <chrono>
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

// LPN_ENGINE_TIMING: how long the game spends in the stage caster loop (0xa51d90), in its
// per-batch state commit (0xe61ae0 called from 0xa51100) and in drawIndex for those casters,
// logged every 10 s of wall time: what skipping the engine's own state work could save.
bool gTiming = false;
Fn gCasterLoop = nullptr, gCommit = nullptr;
std::atomic<uint64_t> gLoopNs{0}, gCommitNs{0}, gDrawNs{0}, gDrawsCaptured{0}, gDrawsGame{0}, gLoops{0}, gCommits{0};
std::atomic<int64_t> gTimingStart{0};

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void timingReport(int64_t now) {
    int64_t start = gTimingStart.load();
    if (!start) {
        gTimingStart.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 10'000'000'000ll || !gTimingStart.compare_exchange_strong(start, now)) return;
    const double s = double(now - start) / 1e9;
    const uint64_t loops = gLoops.exchange(0), commits = gCommits.exchange(0);
    const uint64_t captured = gDrawsCaptured.exchange(0), game = gDrawsGame.exchange(0);
    std::fprintf(stderr,
                 "[native] engine timing over %.1f s: caster loop %.2f ms/s (%llu calls), its state commits %.2f ms/s (%llu), "
                 "its drawIndex %.2f ms/s (%llu drawn from captures, %llu by the game)\n",
                 s, double(gLoopNs.exchange(0)) / 1e6 / s, (unsigned long long)loops, double(gCommitNs.exchange(0)) / 1e6 / s,
                 (unsigned long long)commits, double(gDrawNs.exchange(0)) / 1e6 / s, (unsigned long long)captured,
                 (unsigned long long)game);
}

uint64_t casterLoop(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
    const int64_t t = nowNs();
    const uint64_t r = gCasterLoop(a, b, c, d, e, f);
    const int64_t n = nowNs();
    gLoopNs += uint64_t(n - t);
    ++gLoops;
    timingReport(n);
    return r;
}

uint64_t commit(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
    const auto* frame = static_cast<const uint64_t*>(LPN_GUEST_FRAME());
    const uint64_t ret = frame[1] - reinterpret_cast<uint64_t>(gBase);
    if (ret < 0xa51100 || ret >= 0xa51d90) return gCommit(a, b, c, d, e, f);
    const int64_t t = nowNs();
    const uint64_t r = gCommit(a, b, c, d, e, f);
    gCommitNs += uint64_t(nowNs() - t);
    ++gCommits;
    return r;
}

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
void readAlphaInputsNow(uint64_t ctx, uint64_t s, EngineDraw& d) {
    const uint64_t f = *reinterpret_cast<const uint64_t*>(s + 0x698);
    if (readable(f)) {
        const uint64_t t = *reinterpret_cast<const uint64_t*>(f + 0x10);
        const uint32_t n = *reinterpret_cast<const uint32_t*>(f + 0x18);
        if (readable(t)) {
            d.attributeCount = std::min<uint32_t>(n, 16);
            std::memcpy(d.attributes, reinterpret_cast<const void*>(t), 8 * d.attributeCount);
        }
        // The coordinates: the one attribute after the position in a two-channel float format
        // (16_16 or 32_32; the table has unused all-zero entries between: Fwha's check runs).
        int found = -1;
        for (uint32_t i = 1; i < d.attributeCount; ++i) {
            const uint32_t w3 = d.attributes[i][1], dfmt = (w3 >> 15) & 0xF, nfmt = (w3 >> 12) & 7;
            if ((dfmt == 5 || dfmt == 11) && nfmt == 7) {
                if (found >= 0) {
                    found = -2; // two of them: not known which
                    break;
                }
                found = int(i);
            }
        }
        if (found >= 0) {
            const uint32_t a = d.attributes[found][0];
            d.uvOffset = (a & 0xFF) * 4, d.uvStream = (a >> 8) & 0xFF;
            d.uvWord3 = d.attributes[found][1];
            // As stored: x and y (the shader reads them raw).
            d.uvKnown = !(a & 0xFF0000) && d.uvStream == 0 && (d.uvWord3 & 0x3F) == (4 | 5 << 3);
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

// Guarded: a pointer the engine left stale must not crash the game; the inputs are then unknown
// and the game draws the caster itself.
void readAlphaInputs(uint64_t ctx, uint64_t s, EngineDraw& d) {
    struct Args {
        uint64_t ctx, s;
        EngineDraw* d;
    } a{ctx, s, &d};
    if (!guarded([](void* p) {
            auto* k = static_cast<Args*>(p);
            readAlphaInputsNow(k->ctx, k->s, *k->d);
        }, &a)) {
        d.uvKnown = false;
        d.textureSlot = -1;
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
        // The alpha-tested class needs the coordinates and texture as well; without them the
        // game's draw stays.
        const bool alpha = site == 0xa51b79 || site == 0xa51d37;
        if (gShadows && (opaqueClass || (alpha && d.uvKnown && d.textureSlot >= 0)) && displaySettingsNative()) {
            EngineCaster c;
            c.draw = d;
            c.vertices = at<uint32_t>(s + 0x294);
            c.index32 = (ib >> 31) != 0;
            c.alpha = alpha;
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
    const bool timed = gTiming && engineSite(site);
    const bool captured = tEngineDrawId != 0;
    const int64_t t = timed ? nowNs() : 0;
    const uint64_t r = gDrawIndex(ctx, count, first, baseVertex, r8, r9);
    if (timed) {
        gDrawNs += uint64_t(nowNs() - t);
        ++(captured ? gDrawsCaptured : gDrawsGame);
    }
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
    if (on("LPN_ENGINE_TIMING")) {
        // push rbp; mov rbp, rsp; push r15..rbx (13 bytes: the 12-byte jump fits, the trampoline slot too)
        const Hook timing[] = {
            {0xa51d90, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53},
             guestHook<&casterLoop>(), reinterpret_cast<void**>(&gCasterLoop)},
            {0xe61ae0,
             {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53},
             guestHook<&commit>(), reinterpret_cast<void**>(&gCommit)},
        };
        gTiming = installHooks(gBase, timing, std::size(timing), "engine timing");
        if (gTiming) std::fprintf(stderr, "[native] engine timing: the stage caster loop is timed (LPN_ENGINE_TIMING)\n");
    }
    if (gShadows) std::fprintf(stderr, "[native] engine capture: the stage's opaque shadow casters are drawn from the engine (LPN_ENGINE_SHADOWS)\n");
    else std::fprintf(stderr, "[native] engine capture: checking the stage shadow casters against their draws (LPN_ENGINE_CHECK)\n");
    return true;
}

} // namespace host::native
