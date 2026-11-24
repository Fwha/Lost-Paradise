#include "host/native/engine_capture.h"

#include "host/guest_abi.h"
#include "host/hooks.h"
#include "host/module.h"
#include "host/native/draw_origin.h"
#include "host/crash.h"
#include "host/display_settings.h"
#include "host/native/shader_names.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <string>

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

// LPN_ENGINE_PROBE: for the scene's draw-command-list draws (0xea3f40 tail-calls drawIndex, so
// they come back to 0x3700), which shaders the engine state points at, and through which field:
// candidates from docs/engine/draw-state.md (S+0xf0 = the PS object's +0x58, S+0x100 = the VS
// object's +0x38, G+0xc0, G+0xf8, G+0x100) read as Gnm stage registers (PGM_LO, PGM_HI) and as
// code pointers, named by the shader index. Logged every 10 s.
bool gProbe = false;
std::mutex gProbeMutex;
std::map<std::string, uint64_t> gProbeCounts;
std::map<std::string, std::string> gProbeSample;
int64_t gProbeStart = 0;

// Guarded: a candidate field can hold anything (a fault is caught, as renderer.cpp's readGuest does).
bool probeRead(uint64_t a, void* d, size_t n) {
    if (!readable(a) || !readable(a + n)) return false;
    struct Copy {
        void* dst;
        uint64_t addr;
        size_t size;
    } c{d, a, n};
    return host::guarded([](void* p) {
        auto& c = *static_cast<Copy*>(p);
        std::memcpy(c.dst, reinterpret_cast<const void*>(c.addr), c.size);
    }, &c);
}

// The names a pointer field can lead to: as stage registers (lo << 8 | (hi & 0xff) << 40) and as
// a pointer to code.
std::string probeNames(uint64_t p) {
    uint32_t w[2];
    if (!probeRead(p, w, 8)) return "-";
    std::string out;
    const uint64_t regs = (uint64_t(w[0]) << 8) | (uint64_t(w[1] & 0xff) << 40);
    if (readable(regs)) out += "regs:" + shaderNameAt(regs, probeRead);
    const uint64_t ptr = uint64_t(w[0]) | uint64_t(w[1]) << 32;
    if (readable(ptr)) out += " ptr:" + shaderNameAt(ptr, probeRead);
    out += " self:" + shaderNameAt(p, probeRead);
    return out;
}

void probeScene(uint64_t g) {
    const uint64_t s = at<uint64_t>(g + 0x320);
    std::string key;
    char b[96];
    const std::pair<const char*, uint64_t> fields[] = {{"S+0xf0", at<uint64_t>(s + 0xf0)}, {"S+0x100", at<uint64_t>(s + 0x100)},
                                                       {"G+0xc0", at<uint64_t>(g + 0xc0)}, {"G+0xf8", at<uint64_t>(g + 0xf8)},
                                                       {"G+0x100", at<uint64_t>(g + 0x100)}};
    for (const auto& [name, p] : fields) key += std::string(name) + " " + probeNames(p) + "; ";
    std::snprintf(b, sizeof b, "masks: cb ps 0x%x vs 0x%x, res ps 0x%llx vs 0x%llx", at<uint32_t>(s + 0x190), at<uint32_t>(s + 0x1b0),
                  (unsigned long long)at<uint64_t>(s + 0x180), (unsigned long long)at<uint64_t>(s + 0x1a0));
    const int64_t now = nowNs();
    std::lock_guard lock(gProbeMutex);
    if (++gProbeCounts[key] == 1) gProbeSample[key] = b;
    if (!gProbeStart) gProbeStart = now;
    if (now - gProbeStart < 10'000'000'000ll) return;
    gProbeStart = now;
    std::vector<std::pair<uint64_t, std::string>> v;
    for (const auto& [k, n] : gProbeCounts) v.push_back({n, k});
    std::sort(v.rbegin(), v.rend());
    std::fprintf(stderr, "[native] engine probe: %zu kinds of list draws in 10 s\n", v.size());
    for (size_t i = 0; i < v.size() && i < 25; ++i)
        std::fprintf(stderr, "[native]   %llu x %s(%s)\n", (unsigned long long)v[i].first, v[i].second.c_str(), gProbeSample[v[i].second].c_str());
    gProbeCounts.clear();
}


// LPN_ENGINE_SCENE_CHECK (engine_capture.h, docs/engine/user-data.md).
bool gSceneCheck = false;
constexpr uint32_t kScenes = 1 << 15;
std::mutex gSceneMutex;
std::vector<EngineSceneDraw> gScenes(kScenes);
std::vector<uint32_t> gSceneIds(kScenes, ~0u);
uint32_t gNextScene = 0;

template <typename T> bool rd(uint64_t a, T& v) { return probeRead(a, &v, sizeof v); }

// 0xeac01d-0xeac0a2: the physical binding bank of a logical stage.
uint32_t physicalBank(uint32_t l, uint32_t active) {
    const bool s3 = active & (1u << 3), s5 = active & (1u << 5);
    if (!s5 && s3 && l == 4) return 2;
    if (s5 && l == 6) return 2;
    if (s5 && !s3 && l == 2) return 4;
    return l;
}

// e584d0 / e58680 (Codex's reading): an ordinary resource whose format reads 0x6d is given format
// 0x10 before it is placed (words 1 and 3 change); `d` holds the V# (4 words) or T# (8 words).
void ordinaryResourceFormat(uint32_t* d, bool texture) {
    const uint32_t w1 = d[1], w3 = d[3];
    constexpr uint32_t to = 0x10;
    if (texture) { // fa6d10, fa6cd0
        const uint32_t f = ((w1 >> 6) & 3) | ((w1 >> 28) & 0xC) | ((w3 & 0x04000000) ? 0x10 : 0x60);
        if (f != 0x6D) return;
        d[1] = (w1 & 0x3FFFFF3F) | ((to & 3) << 6) | ((to & 0xFFFFFFFCu) << 28);
        d[3] = (w3 & 0xFBFFFFFF) | ((to & 0x10) << 22);
    } else { // f9e030, f9dcb0
        const uint32_t f = ((w1 >> 14) & 3) | ((w1 >> 7) & 0x60) | ((w3 >> 25) & 0x1C);
        if (f != 0x6D) return;
        d[1] = (w1 & 0xFFFF0FFF) | ((to & 3) << 14) | ((to & 0x60) << 7);
        d[3] = (w3 & 0xC7FFFFFF) | ((to & 0x1C) << 25);
    }
}

// One stage (l: 1 PS, 2 VS) as the engine is about to set it up for this draw.
void decodeStage(uint64_t g, uint64_t s, uint32_t l, uint64_t record, EngineStage& st) {
    uint64_t code = 0, prog = 0;
    uint32_t variant = 0;
    if (!rd(record, code) || !rd(record + 0x10, prog) || !rd(record + 0x1c, variant)) return;
    st.code = code;
    const uint64_t m = prog + 4 * ((variant >> 4) & 0xFFFF);
    uint16_t n = 0;
    uint64_t staging = 0;
    uint32_t active = 0;
    if (!rd(m, n) || !rd(g + 0x68, staging) || !rd(s + 0xdc, active)) return;
    const uint32_t b = physicalBank(l, active);
    st.table.assign(n, 0);
    st.tableKnown.assign(n, 0);
    if (n) probeRead(staging + 0x1800 * l, st.table.data(), 4u * n);
    auto place = [&](uint16_t map, const uint32_t* d, uint32_t words) {
        if (map == 0xFFFF) return;
        const uint32_t at = map & 0x3FFF;
        for (uint32_t k = 0; k < words; ++k) {
            if (map & 0x8000) {
                if (at + k < 16) st.sgpr[at + k] = d[k], st.sgprKnown |= uint16_t(1u << (at + k));
            } else if (at + k < n) {
                st.table[at + k] = d[k], st.tableKnown[at + k] = 1;
            }
        }
    };
    auto bits = [](uint64_t mask, auto&& fn) {
        for (uint32_t i = 0; i < 64; ++i)
            if (mask >> i & 1) fn(i);
    };
    uint32_t cbUse = 0, sampUse = 0, extraUse = 0, extraType = 0;
    uint64_t resUse = 0, resType = 0;
    rd(s + 0x170 + 0x20 * l, cbUse), rd(s + 0x160 + 0x20 * l, resUse), rd(s + 0x16c + 0x20 * l, sampUse), rd(s + 0x168 + 0x20 * l, extraUse);
    rd(s + 0x6a8 + 0x220 * b, resType), rd(s + 0x16c8 + 0x110 * b, extraType);
    uint32_t d[8];
    uint16_t map;
    bits(cbUse, [&](uint32_t i) {
        if (probeRead(s + 0x1d38 + 0x210 * b + 0x20 * i, d, 16) && rd(m + 0xdc + 2 * i, map)) place(map, d, 4);
    });
    bits(resUse, [&](uint32_t i) {
        uint64_t p = 0;
        const uint32_t words = (resType >> i & 1) ? 8 : 4;
        if (rd(s + 0x6c8 + 0x220 * b + 8 * i, p) && probeRead(p, d, 4 * words) && rd(m + 0x1c + 2 * i, map)) {
            ordinaryResourceFormat(d, words == 8);
            place(map, d, words);
        }
    });
    bits(sampUse, [&](uint32_t i) {
        uint64_t p = 0;
        if (rd(s + 0x1378 + 0x90 * b + 8 * i, p) && probeRead(p, d, 16) && rd(m + 0xbc + 2 * i, map)) place(map, d, 4);
    });
    bits(extraUse, [&](uint32_t i) {
        uint64_t p = 0;
        const uint32_t words = (extraType >> i & 1) ? 4 : 8;
        if (rd(s + 0x16d8 + 0x110 * b + 0x10 * i, p) && probeRead(p, d, 4 * words) && rd(m + 0x9c + 2 * i, map)) place(map, d, words);
    });
    uint16_t special = 0;
    if (rd(prog + 4, special) && special && rd(m + 0xfa, map) && probeRead(m + 0x144, d, 16)) place(map, d, 4);
    // Pointers into the table, and the scalars (e574c0).
    uint8_t hdr[16];
    uint16_t offs[5];
    if (!probeRead(m, hdr, 16) || !probeRead(m + 0x12, offs, 10)) return;
    auto pointer = [&](uint8_t sgpr, uint32_t dword) {
        if (sgpr != 0xFF && st.pointerCount < 8) st.pointers[st.pointerCount++] = {sgpr, dword};
    };
    auto value = [&](uint8_t sgpr, uint64_t v, uint32_t words) {
        if (sgpr == 0xFF) return;
        for (uint32_t k = 0; k < words && sgpr + k < 16; ++k) st.sgpr[sgpr + k] = uint32_t(v >> (32 * k)), st.sgprKnown |= uint16_t(1u << (sgpr + k));
    };
    if (hdr[3] & 2) {
        pointer(hdr[7], offs[0]);   // M+0x12
        pointer(hdr[8], offs[2]);   // M+0x16
        pointer(hdr[9], offs[3]);   // M+0x18
        pointer(hdr[0xa], offs[4]); // M+0x1a
        pointer(hdr[6], 0);
        uint64_t v70 = 0;
        if (rd(g + 0x70, v70)) value(hdr[0xb], v70, 2);
        uint32_t v94 = 0;
        if (rd(g + 0x94 + 0x40 * l, v94)) value(hdr[0xd], v94, 1);
    }
    if (l == 2) {
        if (hdr[3] & 4) pointer(hdr[5], offs[1]); // M+0x14 (M+4: the fetch shader, installed inside drawIndex)
        uint16_t a = 0, c = 0;
        if (hdr[0xe] != 0xFF && rd(g + 0x2c8, a) && rd(g + 0x2ca, c)) value(hdr[0xe], uint32_t(a) * c, 1);
    }
    st.ok = true;
}

void captureScene(uint64_t g) {
    const uint64_t s = at<uint64_t>(g + 0x320);
    EngineSceneDraw e;
    decodeStage(g, s, 2, at<uint64_t>(s + 0x100), e.vs);
    decodeStage(g, s, 1, at<uint64_t>(s + 0xf0), e.ps);
    std::lock_guard lock(gSceneMutex);
    const uint32_t id = gNextScene++;
    gScenes[id % kScenes] = std::move(e);
    gSceneIds[id % kScenes] = id;
    tEngineCheckId = id + 1;
}

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
    if (gProbe && site == 0x3700) probeScene(ctx);
    if (gSceneCheck && site == 0x3700) captureScene(ctx);
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
bool engineSceneCheckOn() { return gSceneCheck; }
bool engineScene(uint32_t id, EngineSceneDraw& out) {
    std::lock_guard lock(gSceneMutex);
    if (gSceneIds[id % kScenes] != id) return false;
    out = gScenes[id % kScenes];
    return true;
}
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
    gProbe = on("LPN_ENGINE_PROBE");
    gSceneCheck = on("LPN_ENGINE_SCENE_CHECK");
    if (!check && !gShadows && !gProbe && !gSceneCheck) return false;
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
