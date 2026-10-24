// High-level emulation: our C++ implementations of PS4 system library functions.
//
// The PS4 uses the same System V x86-64 calling convention as Linux, so an HLE function is
// an ordinary C++ function with the guest's signature, bound directly into the import slot.
#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <chrono>

#include <cstdint>
#include <string>

namespace host {

// LPN_PROFILE=1: what each game thread is blocked in, sampled by profile.cpp ([wait] lines).
// seq is odd while the thread is inside a wait; stack is the guest stack at the HLE call, return
// address first, so the sampler can name the game code that is waiting.
struct WaitSlot {
    std::atomic<uint32_t> seq{0};
    std::atomic<const char*> what{nullptr};
    std::atomic<const uint64_t*> stack{nullptr};
    int tid = 0;
};
WaitSlot& waitSlot(); // this thread's, registered with the sampler on first use
bool waitSamplingOn();

// LPN_WAITLOG=1 (debugging): a blocking call that waited over 100 ms (or LPN_WAITLOG=<ms>) logs
// its name and timeout. frame is __builtin_frame_address(0) of the HLE function the game called.
struct WaitLog {
    const char* what;
    int64_t timeoutUs;
    WaitSlot* slot = nullptr;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    WaitLog(const char* w, int64_t timeout, const void* frame) : what(w), timeoutUs(timeout) {
        if (!waitSamplingOn()) return;
        slot = &waitSlot();
        slot->what.store(w, std::memory_order_relaxed);
        slot->stack.store(frame ? static_cast<const uint64_t*>(frame) + 1 : nullptr, std::memory_order_relaxed);
        slot->seq.fetch_add(1, std::memory_order_release);
    }
    WaitLog(const WaitLog&) = delete;
    ~WaitLog() {
        if (slot) slot->seq.fetch_add(1, std::memory_order_release);
        static const long limit = std::getenv("LPN_WAITLOG") ? std::max(1L, std::atol(std::getenv("LPN_WAITLOG"))) : 0;
        if (!limit) return;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (ms > (limit == 1 ? 100 : limit)) std::fprintf(stderr, "[wait] %s %lld ms (timeout %lld us)\n", what, (long long)ms, (long long)timeoutUs);
    }
};

// Everything the trampoline for an unimplemented import needs to report it.
struct ImportInfo {
    std::string library;
    std::string nid;
    std::string name; // empty when the NID isn't in the database
    uint64_t calls = 0;
    void* target = nullptr;      // traced imports: the implementation to run
    uint64_t lastTracedFlip = 0; // traced imports: flip of the last logged call
};

// Register an implementation by library and symbol name. Call before linking.
void registerHle(const std::string& library, const std::string& name, void* fn);

// The registered implementation, or nullptr.
void* findHle(const std::string& library, const std::string& name);

// Install the built-in implementations (see hle_*.cpp).
void registerBuiltinHle();

// Logging switches, set from the command line.
struct HleOptions {
    bool abortOnUnimplemented = false;
    uint64_t logLimitPerImport = 3; // further calls to the same import are counted, not logged
    bool traceHle = false;          // route implemented imports through a logging trampoline
    uint64_t traceFrom = 0;         // with traceHle: also log each import's first call after this flip
};
HleOptions& hleOptions();

// Summary of unimplemented imports hit so far, most-called first.
void printUnimplementedSummary();

} // namespace host
