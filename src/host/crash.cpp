#include "host/crash.h"

#include "host/hle.h"
#include "host/linker.h"
#include "host/module.h"

#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace host {

namespace {

thread_local sigjmp_buf* tGuard = nullptr; // set while guarded() runs on this thread

void describe(const char* what, uint64_t addr) {
    if (Module* m = findModule(addr))
        std::fprintf(stderr, "  %-4s 0x%016lx  %s+0x%lx\n", what, addr, m->name().c_str(),
                     addr - uint64_t(m->base()));
    else
        std::fprintf(stderr, "  %-4s 0x%016lx\n", what, addr);
}

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
    // Return addresses near the top of the stack hint at the call chain.
    const auto* sp = reinterpret_cast<const uint64_t*>(g[REG_RSP]);
    std::fprintf(stderr, "  stack (guest code only):\n");
    for (int i = 0, shown = 0; i < 512 && shown < 12; ++i)
        if (findModule(sp[i])) {
            describe("", sp[i]);
            ++shown;
        }
    _exit(128 + sig);
}

void onInterrupt(int) {
    printUnimplementedSummary();
    _exit(130);
}

} // namespace

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

} // namespace host
