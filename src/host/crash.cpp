#include "host/crash.h"

#include "host/hle.h"
#include "host/linker.h"
#include "host/module.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#endif

namespace host {

namespace {

void describe(const char* what, uint64_t addr) {
    if (Module* m = findModule(addr))
        std::fprintf(stderr, "  %-4s 0x%016llx  %s+0x%llx\n", what, (unsigned long long)addr, m->name().c_str(),
                     (unsigned long long)(addr - uint64_t(m->base())));
    else
        std::fprintf(stderr, "  %-4s 0x%016llx\n", what, (unsigned long long)addr);
}

void describeStack(const uint64_t* sp) {
    // Return addresses near the top of the stack hint at the call chain.
    std::fprintf(stderr, "  stack (guest code only):\n");
    for (int i = 0, shown = 0; i < 512 && shown < 12; ++i)
        if (findModule(sp[i])) {
            describe("", sp[i]);
            ++shown;
        }
}

#ifdef _WIN32

// The plain register jump buffer of __builtin_setjmp: the exception handler sends the faulting
// thread to guardResume, which jumps back into guarded().
thread_local void** tGuard = nullptr;

[[noreturn]] __attribute__((noinline)) void guardResume() { __builtin_longjmp(tGuard, 1); }

void report(EXCEPTION_POINTERS* e) {
    const CONTEXT& c = *e->ContextRecord;
    const EXCEPTION_RECORD& r = *e->ExceptionRecord;
    std::fprintf(stderr, "\n[crash] exception 0x%08lx at %p", r.ExceptionCode, r.ExceptionAddress);
    if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2)
        std::fprintf(stderr, ", %s address 0x%llx", r.ExceptionInformation[0] == 1 ? "writing" : "reading",
                     (unsigned long long)r.ExceptionInformation[1]);
    std::fprintf(stderr, "\n");
    describe("rip", c.Rip);
    describe("rsp", c.Rsp);
    const struct { const char* name; DWORD64 v; } regs[] = {
        {"rax", c.Rax}, {"rbx", c.Rbx}, {"rcx", c.Rcx}, {"rdx", c.Rdx}, {"rsi", c.Rsi}, {"rdi", c.Rdi}, {"rbp", c.Rbp},
        {"r8", c.R8},   {"r9", c.R9},   {"r12", c.R12}, {"r13", c.R13}, {"r14", c.R14}, {"r15", c.R15},
    };
    for (const auto& g : regs) describe(g.name, g.v);
    describeStack(reinterpret_cast<const uint64_t*>(c.Rsp));
    std::fflush(stderr);
}

bool isFault(DWORD code) {
    return code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION ||
           code == EXCEPTION_PRIV_INSTRUCTION || code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
           code == EXCEPTION_STACK_OVERFLOW || code == EXCEPTION_IN_PAGE_ERROR;
}

LONG CALLBACK onException(EXCEPTION_POINTERS* e) {
    const DWORD code = e->ExceptionRecord->ExceptionCode;
    if ((code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR) && tGuard) {
        CONTEXT& c = *e->ContextRecord;
        c.Rsp = (c.Rsp & ~uint64_t(15)) - 8; // as if guardResume had been called
        c.Rip = reinterpret_cast<DWORD64>(&guardResume);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    // A fault in the game's code: nothing on the Windows side could handle it (game frames
    // have no unwind data), so report it here. Others go on to the host's handlers.
    if (isFault(code) && findModule(e->ContextRecord->Rip)) {
        report(e);
        _exit(3);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI onUnhandled(EXCEPTION_POINTERS* e) {
    report(e);
    _exit(3);
}

BOOL WINAPI onConsoleCtrl(DWORD type) {
    if (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT) return FALSE;
    printUnimplementedSummary();
    _exit(130);
}

#else

thread_local sigjmp_buf* tGuard = nullptr; // set while guarded() runs on this thread

void onFatal(int sig, siginfo_t* info, void* ctx) {
    if ((sig == SIGSEGV || sig == SIGBUS) && tGuard) {
        // Entered with sig blocked; sigsetjmp(.., 0) does not restore the mask, so unblock it here.
        sigset_t s;
        sigemptyset(&s);
        sigaddset(&s, sig);
        pthread_sigmask(SIG_UNBLOCK, &s, nullptr);
        siglongjmp(*tGuard, 1);
    }
    const auto& g = static_cast<ucontext_t*>(ctx)->uc_mcontext.gregs;
    std::fprintf(stderr, "\n[crash] signal %d (%s), fault address %p\n", sig, strsignal(sig), info->si_addr);
    describe("rip", g[REG_RIP]);
    describe("rsp", g[REG_RSP]);
    static const struct { const char* name; int reg; } regs[] = {
        {"rax", REG_RAX}, {"rbx", REG_RBX}, {"rcx", REG_RCX}, {"rdx", REG_RDX}, {"rsi", REG_RSI},
        {"rdi", REG_RDI}, {"rbp", REG_RBP}, {"r8", REG_R8},   {"r9", REG_R9},   {"r12", REG_R12},
        {"r13", REG_R13}, {"r14", REG_R14}, {"r15", REG_R15},
    };
    for (const auto& r : regs) describe(r.name, g[r.reg]);
    describeStack(reinterpret_cast<const uint64_t*>(g[REG_RSP]));
    _exit(128 + sig);
}

void onInterrupt(int) {
    printUnimplementedSummary();
    _exit(130);
}

#endif

} // namespace

#ifdef _WIN32
bool guarded(void (*fn)(void*), void* ctx) {
    void* jb[5];
    void** const prev = tGuard;
    if (__builtin_setjmp(jb)) {
        tGuard = prev;
        return false;
    }
    tGuard = jb;
    fn(ctx);
    tGuard = prev;
    return true;
}

void installCrashHandler() {
    AddVectoredExceptionHandler(1, onException);
    SetUnhandledExceptionFilter(onUnhandled);
    SetConsoleCtrlHandler(onConsoleCtrl, TRUE); // Ctrl+C: show which stubs were hit before leaving
}
#else
bool guarded(void (*fn)(void*), void* ctx) {
    sigjmp_buf jb;
    sigjmp_buf* const prev = tGuard;
    if (sigsetjmp(jb, 0)) { // no signal mask saved: that would be a syscall per call
        tGuard = prev;
        return false;
    }
    tGuard = &jb;
    fn(ctx);
    tGuard = prev;
    return true;
}

void installCrashHandler() {
    static char altStack[1 << 16];
    stack_t ss{};
    ss.ss_sp = altStack;
    ss.ss_size = sizeof(altStack);
    sigaltstack(&ss, nullptr);

    struct sigaction sa{};
    sa.sa_sigaction = onFatal;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP}) sigaction(sig, &sa, nullptr);
    signal(SIGINT, onInterrupt); // Ctrl+C: show which stubs were hit before leaving
}
#endif

} // namespace host
