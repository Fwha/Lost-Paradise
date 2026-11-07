
#ifdef _WIN32
// Debugging tool built on SIGTRAP breakpoints: Linux only for now.
#include "host/combo_trace.h"

#include <cstdio>

namespace host::combo_trace {
void onKey() { std::fprintf(stderr, "[combo] not available on Windows\n"); }
} // namespace host::combo_trace
#else
#include "host/combo_trace.h"

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

#include <signal.h>
#include "host/mman.h"
#include <ucontext.h>
#include <unistd.h>

namespace host::combo_trace {

namespace {

// Each hook replaces the first byte of one instruction with int3; the handler does what that
// instruction did (only the few forms below) and logs.
enum class Op { PushRbp, MovR14dToEax, LoadR15FromRbp };
enum Kind { kResolve = 1, kCandidate, kCandidateResult, kCondition, kExecute };
struct Hook {
    uint32_t at;
    Op op;
    Kind kind;
    uint8_t was[7];
    int len;
};
Hook gHooks[] = {
    {0x3f77b0, Op::PushRbp, kResolve, {0x55, 0x48, 0x89, 0xe5}, 4},
    {0x3eb7d0, Op::PushRbp, kCandidate, {0x55, 0x48, 0x89, 0xe5}, 4},
    {0x3eb851, Op::MovR14dToEax, kCandidateResult, {0x44, 0x89, 0xf0}, 3},
    {0x3eb178, Op::LoadR15FromRbp, kCondition, {0x4c, 0x8b, 0xbd, 0xf8, 0xfd, 0xff, 0xff}, 7},
    {0x3f7a20, Op::PushRbp, kExecute, {0x55, 0x48, 0x89, 0xe5}, 4},
};

struct Event {
    uint64_t flip;
    uint32_t kind, a, b, c;
    uint32_t tid;
};
constexpr size_t kRing = 1 << 18;
Event gRing[kRing];
std::atomic<uint64_t> gWritten{0};
std::atomic<bool> gOn{false};
uint8_t* gText = nullptr;
struct sigaction gPrev{};

uint32_t u32(uint64_t p) {
    uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(p), 4);
    return v;
}
uint16_t u16(uint64_t p) {
    uint16_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(p), 2);
    return v;
}

void log(Kind k, uint32_t a, uint32_t b, uint32_t c) {
    if (!gOn.load(std::memory_order_relaxed)) return;
    const uint64_t i = gWritten.fetch_add(1);
    gRing[i % kRing] = {flipCount(), uint32_t(k), a, b, c, uint32_t(gettid())};
}

void onTrap(int sig, siginfo_t* info, void* ctx) {
    auto& g = static_cast<ucontext_t*>(ctx)->uc_mcontext.gregs;
    const uint64_t at = uint64_t(g[REG_RIP]) - 1;
    for (const Hook& h : gHooks) {
        if (!gText || at != uint64_t(gText) + h.at) continue;
        switch (h.op) {
        case Op::PushRbp:
            g[REG_RSP] -= 8;
            *reinterpret_cast<uint64_t*>(g[REG_RSP]) = uint64_t(g[REG_RBP]);
            g[REG_RIP] = int64_t(at + 1);
            break;
        case Op::MovR14dToEax:
            g[REG_RAX] = int64_t(uint32_t(g[REG_R14]));
            g[REG_RIP] = int64_t(at + 3);
            break;
        case Op::LoadR15FromRbp:
            g[REG_R15] = *reinterpret_cast<const int64_t*>(g[REG_RBP] - 0x208);
            g[REG_RIP] = int64_t(at + 7);
            break;
        }
        switch (h.kind) {
        case kResolve: // (resolver, current move id)
            log(kResolve, uint32_t(g[REG_RSI]), uint32_t(g[REG_RDI]), 0);
            break;
        case kCandidate: { // (candidate record, actor, context, byte 7 of the current move)
            const uint64_t rec = uint64_t(g[REG_RDI]);
            log(kCandidate, u16(rec + 2), uint32_t(g[REG_RSI]), uint32_t(rec));
            break;
        }
        case kCandidateResult:
            log(kCandidateResult, uint32_t(g[REG_R14]), 0, 0);
            break;
        case kCondition: { // %rbx: the condition {type, parameter}; %r12d: passed
            const uint64_t c = uint64_t(g[REG_RBX]);
            log(kCondition, u32(c), u32(c + 4), uint32_t(g[REG_R12]));
            break;
        }
        case kExecute:
            log(kExecute, uint32_t(g[REG_RSI]), uint32_t(g[REG_RDI]), 0);
            break;
        }
        return;
    }
    if (gPrev.sa_flags & SA_SIGINFO) gPrev.sa_sigaction(sig, info, ctx);
    else _exit(128 + sig);
}

void flusher() {
    uint64_t shown = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const uint64_t n = gWritten.load();
        if (n - shown > kRing) {
            std::fprintf(stderr, "[combo] %llu events lost\n", (unsigned long long)(n - shown - kRing));
            shown = n - kRing;
        }
        for (; shown < n; ++shown) {
            const Event& e = gRing[shown % kRing];
            static const char* names[] = {"", "resolve", "cand", "cand=", "cond", "EXEC"};
            std::fprintf(stderr, "[combo] %llu t%u %s %x %x %x\n", (unsigned long long)e.flip, e.tid,
                         names[e.kind], e.a, e.b, e.c);
        }
    }
}

void install() {
    Module* m = process().eboot;
    if (!m) return;
    uint8_t* text = m->base();
    for (const Hook& h : gHooks)
        if (std::memcmp(text + h.at, h.was, h.len) != 0) {
            std::fprintf(stderr, "[combo] unexpected eboot code at 0x%x: no trace\n", h.at);
            return;
        }
    gText = text;
    struct sigaction sa{};
    sa.sa_sigaction = onTrap;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGTRAP, &sa, &gPrev);
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    for (const Hook& h : gHooks) {
        uint8_t* p = text + h.at;
        mprotect(reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1)), page * 2, PROT_READ | PROT_WRITE | PROT_EXEC);
        __atomic_store_n(p, uint8_t(0xcc), __ATOMIC_RELEASE);
    }
    std::thread(flusher).detach();
}

} // namespace

void onKey() {
    static const bool enabled = std::getenv("LPN_COMBO_TRACE") != nullptr;
    if (!enabled) return;
    static bool installed = false;
    if (!installed) install(), installed = true;
    const bool on = !gOn.load();
    gOn = on;
    std::fprintf(stderr, "[combo] trace %s at flip %llu\n", on ? "on" : "off", (unsigned long long)flipCount());
}

} // namespace host::combo_trace
#endif
