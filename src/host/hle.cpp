#include "host/hle.h"

#include "host/guest_abi.h"
#include "host/videoout.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

namespace host {

namespace {

std::map<std::pair<std::string, std::string>, void*>& registry() {
    static std::map<std::pair<std::string, std::string>, void*> r;
    return r;
}

std::mutex& logMutex() {
    static std::mutex m;
    return m;
}

std::vector<ImportInfo*>& hitImports() {
    static std::vector<ImportInfo*> v;
    return v;
}

} // namespace

HleOptions& hleOptions() {
    static HleOptions o;
    return o;
}

void registerHle(const std::string& library, const std::string& name, void* fn) {
    registry()[{library, name}] = fn;
}

void* findHle(const std::string& library, const std::string& name) {
    auto it = registry().find({library, name});
    return it != registry().end() ? it->second : nullptr;
}

void printUnimplementedSummary() {
    std::lock_guard lock(logMutex());
    auto v = hitImports();
    std::sort(v.begin(), v.end(), [](auto* a, auto* b) { return a->calls > b->calls; });
    std::fprintf(stderr, "[hle] %zu unimplemented imports were called:\n", v.size());
    for (auto* i : v)
        std::fprintf(stderr, "  %8" PRIu64 "  %s::%s\n", i->calls, i->library.c_str(),
                     i->name.empty() ? i->nid.c_str() : i->name.c_str());
}

} // namespace host

// Called from lpn_unimplemented_stub (trampoline.S) with the guest's integer arguments.
namespace {
uint64_t onUnimplemented(host::ImportInfo* info, const uint64_t* regs) {
    using namespace host;
    {
        std::lock_guard lock(logMutex());
        if (info->calls++ == 0) hitImports().push_back(info);
        if (info->calls <= hleOptions().logLimitPerImport)
            std::fprintf(stderr,
                         "[hle] unimplemented %s::%s(0x%" PRIx64 ", 0x%" PRIx64 ", 0x%" PRIx64
                         ", 0x%" PRIx64 ") -> 0\n",
                         info->library.c_str(), info->name.empty() ? info->nid.c_str() : info->name.c_str(),
                         regs[0], regs[1], regs[2], regs[3]);
    }
    if (hleOptions().abortOnUnimplemented) {
        std::fprintf(stderr, "[hle] stopping at first unimplemented call (--strict)\n");
        std::abort();
    }
    return 0;
}

} // namespace

// Called from lpn_trace_stub (trampoline.S): logs the first call to each import, and the first
// after --trace-hle's flip, then returns the implementation to run.
namespace {
void* onTraced(host::ImportInfo* info) {
    using namespace host;
    const uint64_t flip = flipCount();
    std::lock_guard lock(logMutex());
    const uint64_t from = hleOptions().traceFrom;
    const bool first = info->calls++ == 0;
    const bool firstAfter = from && flip >= from && info->lastTracedFlip < from;
    if (first || firstAfter) {
        info->lastTracedFlip = flip;
        std::fprintf(stderr, "[trace] %s::%s at flip %" PRIu64 "%s\n", info->library.c_str(),
                     info->name.empty() ? info->nid.c_str() : info->name.c_str(), flip, first ? "" : " (again)");
    }
    return info->target;
}
} // namespace

// The stubs in trampoline.S call these the System V way.
extern "C" LPN_GUEST uint64_t lpn_on_unimplemented(host::ImportInfo* info, const uint64_t* regs) {
    uint64_t (*volatile f)(host::ImportInfo*, const uint64_t*) = onUnimplemented;
    return f(info, regs);
}
extern "C" LPN_GUEST void* lpn_on_traced(host::ImportInfo* info) {
    void* (*volatile f)(host::ImportInfo*) = onTraced;
    return f(info);
}
