#include "host/native/engine_capture.h"

#include "host/guest_abi.h"
#include "host/hooks.h"
#include "host/module.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template <typename T> T at(uint64_t a) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof v);
    return v;
}

// rdi: the Gnm context, whose state block is at +0x320 (render-decomp.md); esi: indices, edx:
// first index, ecx: base vertex.
uint64_t drawIndex(uint64_t ctx, uint64_t count, uint64_t first, uint64_t baseVertex, uint64_t r8, uint64_t r9) {
    const auto* frame = static_cast<const uint64_t*>(LPN_GUEST_FRAME());
    const uint64_t ret = frame[1], base = reinterpret_cast<uint64_t>(gBase);
    const uint32_t site = ret >= base && ret < base + kTextSize ? uint32_t(ret - base) : 0;
    if (engineSite(site)) {
        const uint64_t s = at<uint64_t>(ctx + 0x320);
        EngineDraw d;
        d.site = site;
        d.stride = (at<uint32_t>(s + 0x298) >> 4) & 0xFFFF;
        d.vertexBase = at<uint64_t>(s + 0x288) + uint64_t(uint32_t(baseVertex)) * d.stride;
        // eac430: first << (1 + bit 31 of +0x690), plus its low bits and +0x688.
        const uint32_t ib = at<uint32_t>(s + 0x690);
        d.indexAddr = at<uint64_t>(s + 0x688) + (ib & 0x7FFFFFFF) + (uint64_t(uint32_t(first)) << ((ib >> 31) + 1));
        d.count = uint32_t(count);
        for (int i = 0; i < 4; ++i) d.constants[i] = at<uint64_t>(s + 0x6c8 + 8 * i);
        std::lock_guard lock(gMutex);
        if (gQueue.size() < kMaxQueued) gQueue.push_back(d);
    }
    return gDrawIndex(ctx, count, first, baseVertex, r8, r9);
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

bool engineCheckOn() { return gOn; }

bool installEngineCapture(Module& eboot) {
    const char* e = std::getenv("LPN_ENGINE_CHECK");
    if (!e || !*e || *e == '0') return false;
    // push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov eax, ecx
    const Hook hooks[] = {
        {kDrawIndex, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x89, 0xc8}, guestHook<&drawIndex>(),
         reinterpret_cast<void**>(&gDrawIndex)},
    };
    gBase = eboot.base();
    if (!installHooks(gBase, hooks, std::size(hooks), "engine")) return false;
    gOn = true;
    std::fprintf(stderr, "[native] engine capture: checking the stage shadow casters against their draws (LPN_ENGINE_CHECK)\n");
    return true;
}

} // namespace host::native
