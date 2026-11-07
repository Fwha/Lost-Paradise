// libkernel: process info, errno, time, sleeping, console output and other small calls.
#include "host/equeue.h"
#include "host/hle.h"
#include "host/videoout.h"
#include "host/kernel.h"
#include "host/module.h"
#include "host/tls.h"

#include <sched.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace host {

Process& process() {
    static Process p;
    return p;
}

int toPs4Errno(int e) {
    switch (e) {
    case EDEADLK: return ps4errno::kEDEADLK;
    case EAGAIN: return ps4errno::kEAGAIN;
    case ETIMEDOUT: return ps4errno::kETIMEDOUT;
    case ENAMETOOLONG: return ps4errno::kENAMETOOLONG;
    case ENOTEMPTY: return ps4errno::kENOTEMPTY;
    case ENOSYS: return ps4errno::kENOSYS;
    case EOPNOTSUPP: return 45;
    case ELOOP: return 62;
    default: return e < 35 ? e : ps4errno::kEINVAL; // 1..34 match between Linux and FreeBSD
    }
}

int* guestErrno() {
    static thread_local int e = 0;
    return &e;
}

namespace {

using Clock = std::chrono::steady_clock;
const Clock::time_point gStart = Clock::now();

// Data imports: the linker binds these slots to the variables' addresses.
uint64_t stack_chk_guard = 0x2d3a5f7c91b4e608ull;
const char* progname = "eboot.bin";

struct Timespec {
    int64_t sec, nsec;
};
struct Timeval {
    int64_t sec, usec;
};

int* error_() { return guestErrno(); }

void stack_chk_fail() {
    std::fprintf(stderr, "[kernel] guest stack smashing detected\n");
    std::abort();
}

void exit_(int code) {
    std::fprintf(stderr, "[kernel] guest _exit(%d)\n", code);
    printUnimplementedSummary();
    std::_Exit(code);
}

int64_t write_(int fd, const void* buf, size_t n) {
    if (fd == 1 || fd == 2) {
        std::fwrite(buf, 1, n, fd == 1 ? stdout : stderr);
        std::fflush(fd == 1 ? stdout : stderr);
        return int64_t(n);
    }
    const ssize_t r = ::write(fd, buf, n);
    if (r < 0) *guestErrno() = toPs4Errno(errno);
    return r;
}

int64_t read_(int fd, void* buf, size_t n) {
    const ssize_t r = ::read(fd, buf, n);
    if (r < 0) *guestErrno() = toPs4Errno(errno);
    return r;
}

void* sceKernelGetProcParam() {
    const Module* m = process().eboot;
    const auto* seg = m->elf().findSegment(ps4elf::PT_SCE_PROCPARAM);
    return seg ? m->base() + seg->vaddr : nullptr;
}

int32_t sceKernelIsNeoMode() { return 0; } // base PS4 for now; Pro mode later
int32_t sceKernelGetCpumode() { return 0; }
int32_t sceKernelIsAddressSanitizerEnabled() { return 0; }

void sceKernelDebugRaiseException(uint32_t code, uint32_t arg) {
    std::fprintf(stderr, "[kernel] guest raised exception 0x%08x (arg 0x%x)\n", code, arg);
}
void sceKernelDebugRaiseExceptionOnReleaseMode(uint32_t code, uint32_t arg) {
    sceKernelDebugRaiseException(code, arg);
}

int32_t _sceKernelRtldSetApplicationHeapAPI(void*) { return 0; }
void __pthread_cxa_finalize(void*) {}

clockid_t hostClock(int32_t id) {
    switch (id) {
    case 0: return CLOCK_REALTIME;
    case 15: return CLOCK_PROCESS_CPUTIME_ID;
    case 14: return CLOCK_THREAD_CPUTIME_ID;
    default: return CLOCK_MONOTONIC; // 4 (MONOTONIC) and the PS4's precise/fast variants
    }
}

int32_t clock_gettime_(int32_t id, Timespec* ts) {
    timespec t;
    clock_gettime(hostClock(id), &t);
    ts->sec = t.tv_sec;
    ts->nsec = t.tv_nsec;
    return 0;
}
int32_t sceKernelClockGettime(int32_t id, Timespec* ts) {
    if (!ts) return sceError(ps4errno::kEFAULT);
    return clock_gettime_(id, ts);
}

int32_t gettimeofday_(Timeval* tv, void*) {
    timeval t;
    gettimeofday(&t, nullptr);
    if (tv) *tv = {t.tv_sec, t.tv_usec};
    return 0;
}
int32_t sceKernelGettimeofday(Timeval* tv) {
    if (!tv) return sceError(ps4errno::kEFAULT);
    return gettimeofday_(tv, nullptr);
}

uint64_t sceKernelGetProcessTime() { return gameClockNs() / 1000; }
uint64_t sceKernelGetProcessTimeCounter() { return gameClockNs(); }
uint64_t sceKernelGetProcessTimeCounterFrequency() { return 1000000000ull; }

int32_t sceKernelUsleep(uint32_t us) {
    WaitLog w{"usleep", us, __builtin_frame_address(0)};
    usleep(us);
    return 0;
}
int32_t nanosleep_(const Timespec* req, Timespec* rem) {
    WaitLog w{"nanosleep", int64_t(req->sec) * 1000000 + req->nsec / 1000, __builtin_frame_address(0)};
    timespec r{req->sec, req->nsec}, left{};
    const int ret = ::nanosleep(&r, &left);
    if (rem) *rem = {left.tv_sec, left.tv_nsec};
    if (ret < 0) *guestErrno() = toPs4Errno(errno);
    return ret;
}
int32_t sched_yield_() { return ::sched_yield(); }
int32_t getpagesize_() { return 0x4000; }

// Signals are the host's; the guest's handlers are not installed.
int32_t sigprocmask_(int32_t, const void*, void* old) {
    if (old) std::memset(old, 0, 16);
    return 0;
}
int32_t sigfillset_(void* set) {
    std::memset(set, 0xFF, 16);
    return 0;
}
void* signal_(int32_t, void*) { return nullptr; }

// libSceLibcInternalExt: libc.prx asks the system for its heap bookkeeping tables at startup.
struct HeapTraceInfo {
    uint64_t size;
    uint32_t flags;
    uint32_t getSegmentInfo;
    uint64_t* mspaceAtomicIdMask;
    uint64_t* mstateTable;
};
int32_t sceLibcHeapGetTraceInfo(HeapTraceInfo* info) {
    static uint64_t idMask = 0;
    static uint64_t mstateTable[64] = {};
    info->getSegmentInfo = 0;
    info->mspaceAtomicIdMask = &idMask;
    info->mstateTable = mstateTable;
    return 0;
}

} // namespace

double gameTimeScale() {
    static const double scale = [] {
        const char* v = std::getenv("LPN_TIME_SCALE");
        const double f = v ? std::atof(v) : 1.0;
        if (f > 0 && f != 1.0) std::fprintf(stderr, "[clock] LPN_TIME_SCALE: the game's clock runs at %.3gx\n", f);
        return f > 0 ? f : 1.0;
    }();
    return scale;
}

uint64_t gameClockNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - gStart).count();
    return uint64_t(double(ns) * gameTimeScale());
}

void registerKernelHle() {
    HLE("libSceLibcInternalExt", sceLibcHeapGetTraceInfo);
    const char* k = "libkernel";
    registerHle(k, "__tls_get_addr", guestEntry<&guestTlsGetAddr>());
    HLE_NAMED(k, "__stack_chk_guard", stack_chk_guard);
    HLE_NAMED(k, "__progname", progname);
    HLE_NAMED(k, "__stack_chk_fail", stack_chk_fail);
    HLE_NAMED(k, "__error", error_);
    HLE_NAMED(k, "_exit", exit_);
    HLE_NAMED(k, "_write", write_);
    HLE_NAMED(k, "_read", read_);
    HLE(k, sceKernelGetProcParam);
    HLE(k, sceKernelIsNeoMode);
    HLE(k, sceKernelGetCpumode);
    HLE(k, sceKernelIsAddressSanitizerEnabled);
    HLE(k, sceKernelDebugRaiseException);
    HLE(k, sceKernelDebugRaiseExceptionOnReleaseMode);
    HLE(k, _sceKernelRtldSetApplicationHeapAPI);
    HLE(k, __pthread_cxa_finalize);
    HLE_NAMED(k, "clock_gettime", clock_gettime_);
    HLE(k, sceKernelClockGettime);
    HLE_NAMED(k, "gettimeofday", gettimeofday_);
    HLE(k, sceKernelGettimeofday);
    HLE(k, sceKernelGetProcessTime);
    HLE(k, sceKernelGetProcessTimeCounter);
    HLE(k, sceKernelGetProcessTimeCounterFrequency);
    HLE(k, sceKernelUsleep);
    HLE_NAMED(k, "nanosleep", nanosleep_);
    HLE_NAMED(k, "_nanosleep", nanosleep_);
    HLE_NAMED(k, "sched_yield", sched_yield_);
    HLE_NAMED(k, "getpagesize", getpagesize_);
    HLE_NAMED(k, "sigprocmask", sigprocmask_);
    HLE_NAMED(k, "_sigprocmask", sigprocmask_);
    HLE_NAMED(k, "sigfillset", sigfillset_);
    HLE_NAMED(k, "signal", signal_);
}

void registerBuiltinHle() {
    registerKernelHle();
    registerMemoryHle();
    registerPthreadHle();
    registerFileHle();
    registerEqueueHle();
    registerVideoOutHle();
    registerGnmHle();
    registerServicesHle();
}

} // namespace host
