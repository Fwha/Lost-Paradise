#include "host/cpu_compat.h"

#include "common/x86_emu.h"
#include "common/compat.h"
#include "host/linker.h"
#include "host/mman.h"
#include "host/module.h"
#include "ps4elf/ehframe.h"

#include <algorithm>
#include <atomic>
#include <cpuid.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <signal.h>
#include <ucontext.h>
#endif

namespace host {

namespace {

struct Features {
    bool avx = false, bmi1 = false, lzcnt = false, movbe = false, f16c = false;
};
Features gCpu;
bool gForce = false; // LPN_CPU_EMULATE=1

// Patched instructions: the first byte became 0x06 (push es, invalid in 64-bit mode, so it
// traps); this keeps the byte it was. Sorted by address, read-only once the game runs.
struct Patch {
    uint64_t at;
    uint8_t was;
};
std::vector<Patch> gPatches;

// Instructions found by scanMissingInstructions, on the code as loaded.
struct Site {
    uint8_t* at;
    uint8_t len;
    x86::Op op;
    uint8_t bytes[15];
};
struct Scan {
    const Module* module;
    std::vector<Site> sites;
    size_t skipped, functions;
};
std::vector<Scan> gScans;
uint32_t gOnly = 0; // LPN_CPU_EMULATE=<names>: only these (bit per x86::Op); 0 for all

std::atomic<uint64_t> gCount[size_t(x86::Op::Count)];
std::atomic<bool> gSeen[size_t(x86::Op::Count)];

uint64_t xgetbv0() {
    uint32_t lo, hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0)); // xgetbv
    return (uint64_t(hi) << 32) | lo;
}

Features readFeatures() {
    Features f;
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        const bool osxsave = c & (1u << 27);
        f.avx = (c & (1u << 28)) && osxsave && (xgetbv0() & 6) == 6; // the OS saves xmm and ymm
        f.movbe = c & (1u << 22);
        f.f16c = c & (1u << 29);
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) f.bmi1 = b & (1u << 3);
    if (__get_cpuid(0x80000001, &a, &b, &c, &d)) f.lzcnt = c & (1u << 5);
    return f;
}

bool wanted(x86::Op op) {
    if (op == x86::Op::None) return false;
    if (gForce) return !gOnly || (gOnly >> unsigned(op) & 1);
    // Only the ones that would not trap by themselves: older CPUs read F3 0F BD as bsr and
    // F3 0F BC as bsf. The others raise #UD on their own.
    if (op == x86::Op::Lzcnt) return !gCpu.lzcnt;
    if (op == x86::Op::Tzcnt) return !gCpu.bmi1;
    return false;
}

// A line on stderr, safe in a signal handler.
void say(const char* s) {
#ifdef _WIN32
    std::fputs(s, stderr);
    std::fflush(stderr);
#else
    for (size_t n = std::strlen(s); n;) {
        const ssize_t w = write(2, s, n);
        if (w <= 0) break;
        s += w;
        n -= size_t(w);
    }
#endif
}

void noteFirst(x86::Op op, uint64_t rip) {
    if (gSeen[size_t(op)].exchange(true, std::memory_order_relaxed)) return;
    char line[160];
    const Module* m = findModule(rip);
    std::snprintf(line, sizeof line, "[cpu] emulating %s (first at %s+0x%llx)\n", x86::opName(op),
                  m ? m->name().c_str() : "?", (unsigned long long)(m ? rip - uint64_t(m->base()) : rip));
    say(line);
}

// The instruction at rip with its patched first byte put back. Returns false when rip is not in
// the game's code or holds a 0x06 this file did not write.
bool fetch(uint64_t rip, x86::Insn* in) {
    const Module* m = findModule(rip);
    if (!m) return false;
    uint8_t bytes[15];
    const size_t avail = std::min<uint64_t>(sizeof bytes, uint64_t(m->base()) + m->size() - rip);
    std::memcpy(bytes, reinterpret_cast<const void*>(rip), avail);
    if (bytes[0] == 0x06) {
        const auto it = std::lower_bound(gPatches.begin(), gPatches.end(), rip,
                                         [](const Patch& p, uint64_t a) { return p.at < a; });
        if (it == gPatches.end() || it->at != rip) return false;
        bytes[0] = it->was;
    }
    return x86::decode(bytes, avail, in);
}

x86::Op run(const x86::Insn& in, x86::State& s) {
    const x86::Op op = x86::emulate(in, s);
    if (op != x86::Op::None) {
        gCount[size_t(op)].fetch_add(1, std::memory_order_relaxed);
        noteFirst(op, s.rip - in.len);
    }
    return op;
}

} // namespace

bool checkCpu() {
    gCpu = readFeatures();
    const char* env = std::getenv("LPN_CPU_EMULATE");
    gForce = env && *env && std::strcmp(env, "0") != 0;
    // A list of names (lzcnt,movbe,...) narrows it, to find which one misbehaves.
    if (gForce && std::strcmp(env, "1") != 0) {
        const std::string list = std::string(",") + env + ",";
        for (unsigned i = 1; i < unsigned(x86::Op::Count); ++i)
            if (list.find(std::string(",") + x86::opName(x86::Op(i)) + ",") != std::string::npos) gOnly |= 1u << i;
    }
    if (!gCpu.avx) {
        std::fprintf(stderr, "error: this CPU (or OS) has no AVX, which the game's code needs. "
                             "AVX CPUs start with Intel Sandy Bridge (2011) and AMD Bulldozer.\n");
        return false;
    }
    std::string missing;
    if (!gCpu.bmi1) missing += " BMI1";
    if (!gCpu.lzcnt) missing += " LZCNT";
    if (!gCpu.movbe) missing += " MOVBE";
    if (!gCpu.f16c) missing += " F16C";
    if (!missing.empty())
        std::fprintf(stderr, "[cpu] missing%s: the game's uses of them run in software\n", missing.c_str());
    if (gForce)
        std::fprintf(stderr, "[cpu] LPN_CPU_EMULATE: emulating %s\n", gOnly ? env : "all of BMI1, LZCNT, MOVBE and F16C");
    return true;
}

void scanMissingInstructions(Module& m) {
    if (!gForce && gCpu.lzcnt && gCpu.bmi1) return;
    std::string error;
    const auto functions = ps4elf::functionsFromEhFrame(m.elf(), &error);
    if (functions.empty()) {
        std::fprintf(stderr, "[cpu] %s: no function list (%s), nothing patched\n", m.name().c_str(), error.c_str());
        return;
    }
    // Decode each function from its start, so only real instruction boundaries are touched. A
    // function that does not decode to its exact end is left alone.
    size_t skipped = 0;
    std::vector<Site> sites;
    for (const auto& f : functions) {
        uint8_t* const start = m.base() + f.start;
        if (f.start + f.size > m.size()) continue;
        size_t at = 0;
        std::vector<Site> here;
        bool ok = true;
        while (at < f.size) {
            x86::Insn in;
            if (!x86::decode(start + at, f.size - at, &in)) {
                ok = false;
                break;
            }
            if (wanted(x86::classify(in))) {
                Site site{start + at, in.len, x86::classify(in), {}};
                std::memcpy(site.bytes, start + at, in.len);
                here.push_back(site);
            }
            at += in.len;
        }
        if (!ok || at != f.size) {
            ++skipped;
            continue;
        }
        sites.insert(sites.end(), here.begin(), here.end());
    }
    std::sort(sites.begin(), sites.end(), [](const Site& a, const Site& b) { return a.at < b.at; });
    sites.erase(std::unique(sites.begin(), sites.end(), [](const Site& a, const Site& b) { return a.at == b.at; }),
                sites.end());
    gScans.push_back({&m, std::move(sites), skipped, functions.size()});
}

size_t patchMissingInstructions(Module& m) {
    const Scan* scan = nullptr;
    for (const Scan& sc : gScans)
        if (sc.module == &m) scan = &sc;
    if (!scan) return 0;
    size_t counts[size_t(x86::Op::Count)] = {};
    size_t patched = 0, changed = 0;
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    for (const Site& site : scan->sites) {
        uint8_t* p = site.at;
        // Another patch (title menu, draw sites...) rewrote this code since the scan: leave it.
        if (std::memcmp(p, site.bytes, site.len) != 0) {
            ++changed;
            continue;
        }
        // Other patches (display settings) may already have set this page back to read+exec;
        // Module::protect sets the final protection afterwards.
        mprotect(reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1)), page, PROT_READ | PROT_WRITE | PROT_EXEC);
        ++counts[size_t(site.op)];
        gPatches.push_back({uint64_t(p), *p});
        *p = 0x06;
        ++patched;
    }
    std::sort(gPatches.begin(), gPatches.end(), [](const Patch& a, const Patch& b) { return a.at < b.at; });

    std::string list;
    for (size_t i = 1; i < size_t(x86::Op::Count); ++i)
        if (counts[i]) list += " " + std::to_string(counts[i]) + " " + x86::opName(x86::Op(i));
    std::fprintf(stderr,
                 "[cpu] %s: %zu instructions set to trap for emulation%s%s (%zu left as other patches changed them, "
                 "%zu of %zu functions not decoded)\n",
                 m.name().c_str(), patched, list.empty() ? "" : ":", list.c_str(), changed, scan->skipped,
                 scan->functions);
    return patched;
}

#ifdef _WIN32

bool emulateIllegalInstruction(void* ctx) {
    CONTEXT& c = *static_cast<CONTEXT*>(ctx);
    x86::Insn in;
    if (!fetch(c.Rip, &in) || x86::classify(in) == x86::Op::None) return false;
    DWORD64* const regs[16] = {&c.Rax, &c.Rcx, &c.Rdx, &c.Rbx, &c.Rsp, &c.Rbp, &c.Rsi, &c.Rdi,
                               &c.R8,  &c.R9,  &c.R10, &c.R11, &c.R12, &c.R13, &c.R14, &c.R15};
    x86::State s;
    for (int i = 0; i < 16; ++i) s.gpr[i] = *regs[i];
    s.rip = c.Rip;
    s.rflags = c.EFlags;
    s.mxcsr = c.MxCsr;
    s.xmm = reinterpret_cast<uint8_t(*)[16]>(c.FltSave.XmmRegisters);
    DWORD64 mask = 0;
    DWORD length = 0;
    void* ymm = nullptr;
    if ((c.ContextFlags & CONTEXT_XSTATE) == CONTEXT_XSTATE && GetXStateFeaturesMask(&c, &mask))
        ymm = LocateXStateFeature(&c, XSTATE_AVX, &length);
    if (ymm && length >= 256) {
        if (!(mask & XSTATE_MASK_AVX)) std::memset(ymm, 0, 256); // the upper halves are all zero
        s.ymmHigh = static_cast<uint8_t(*)[16]>(ymm);
    }
    if (run(in, s) == x86::Op::None) return false;
    for (int i = 0; i < 16; ++i) *regs[i] = s.gpr[i];
    c.Rip = s.rip;
    c.EFlags = DWORD(s.rflags);
    if (s.ymmHighWritten && s.ymmHigh) SetXStateFeaturesMask(&c, mask | XSTATE_MASK_AVX);
    return true;
}

#else

bool emulateIllegalInstruction(void* ctx) {
    ucontext_t& uc = *static_cast<ucontext_t*>(ctx);
    greg_t* const g = uc.uc_mcontext.gregs;
    x86::Insn in;
    if (!fetch(uint64_t(g[REG_RIP]), &in) || x86::classify(in) == x86::Op::None) return false;
    static const int kRegs[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                  REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
    x86::State s;
    for (int i = 0; i < 16; ++i) s.gpr[i] = uint64_t(g[kRegs[i]]);
    s.rip = uint64_t(g[REG_RIP]);
    s.rflags = uint64_t(g[REG_EFL]);
    uint8_t* const fp = reinterpret_cast<uint8_t*>(uc.uc_mcontext.fpregs);
    if (!fp) return false;
    std::memcpy(&s.mxcsr, fp + 24, 4);
    s.xmm = reinterpret_cast<uint8_t(*)[16]>(fp + 160);
    // The frame is an FXSAVE area; with XSAVE state, software-reserved bytes 464+ say so and the
    // XSAVE header (512) and the ymm upper halves (576, standard format) follow.
    uint32_t magic, xstateSize;
    uint64_t features, present;
    std::memcpy(&magic, fp + 464, 4);
    std::memcpy(&features, fp + 472, 8);
    std::memcpy(&xstateSize, fp + 480, 4);
    const bool xsave = magic == 0x46505853u && (features & 4) && xstateSize >= 576 + 256;
    if (xsave) {
        std::memcpy(&present, fp + 512, 8);
        if (!(present & 4)) std::memset(fp + 576, 0, 256); // not saved: the upper halves are zero
        s.ymmHigh = reinterpret_cast<uint8_t(*)[16]>(fp + 576);
    }
    if (run(in, s) == x86::Op::None) return false;
    for (int i = 0; i < 16; ++i) g[kRegs[i]] = greg_t(s.gpr[i]);
    g[REG_RIP] = greg_t(s.rip);
    g[REG_EFL] = greg_t(s.rflags);
    if (s.ymmHighWritten && xsave) {
        present |= 4;
        std::memcpy(fp + 512, &present, 8);
    }
    return true;
}

#endif

void printCpuCompatSummary() {
    std::string list;
    for (size_t i = 1; i < size_t(x86::Op::Count); ++i)
        if (const uint64_t n = gCount[i].load())
            list += " " + std::string(x86::opName(x86::Op(i))) + " x" + std::to_string(n);
    if (!list.empty()) std::fprintf(stderr, "[cpu] emulated:%s\n", list.c_str());
}

} // namespace host
