#include "host/cpu_compat.h"

#include "common/x86_emu.h"
#include "common/compat.h"
#include "host/linker.h"
#include "host/mman.h"
#include "host/module.h"
#include "ps4elf/ehframe.h"

#include <algorithm>
#include <atomic>
#include <climits>
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
#include <sys/mman.h>
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

// Instructions found by scanMissingInstructions, on the code as loaded. With a stub (Windows),
// the instruction becomes a jump to it; one shorter than the 5-byte jump takes the instruction
// after it along (moved into the stub), when that one is safe to move.
struct Site {
    uint8_t* at;
    uint8_t len;
    x86::Op op;
    uint8_t bytes[15];
    uint8_t movedLen = 0; // the next instruction, moved into the stub
    uint8_t moved[15] = {};
    int8_t movedDispAt = -1; // its rip-relative displacement, to fix for the new place
    bool stubOk = false;
};

// Windows delivers exceptions on the faulting thread's stack right below rsp, over the 128-byte
// red zone that System V code (the game's) keeps locals in without moving rsp. So there each
// instruction jumps to a stub that moves rsp past the red zone before it traps:
//   lea rsp, [rsp-128]; 06 (traps); lea rsp, [rsp+128]; [moved instruction]; jmp back
// Linux leaves the red zone alone when it delivers a signal, so it traps in place.
#ifdef _WIN32
bool gStubs = true;
#else
bool gStubs = false; // LPN_CPU_STUBS=1 to try the Windows way
#endif
struct Stub {
    uint64_t trap; // address of the 06 in the stub
    uint64_t orig; // where the instruction was
    uint8_t len;
    uint8_t bytes[15];
};
std::vector<Stub> gStubTable; // sorted by trap
uint8_t* gStubNext = nullptr;
uint8_t* gStubEnd = nullptr;
constexpr int kRedZone = 128;
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
    // Older CPUs read F3 0F BD as bsr and F3 0F BC as bsf, so those have to be made to trap. The
    // others raise #UD on their own, which is enough where traps keep off the red zone (Linux).
    switch (op) {
    case x86::Op::Lzcnt: return !gCpu.lzcnt;
    case x86::Op::Tzcnt: return !gCpu.bmi1;
    case x86::Op::Andn: case x86::Op::Bextr: case x86::Op::Blsi: case x86::Op::Blsmsk: case x86::Op::Blsr:
        return gStubs && !gCpu.bmi1;
    case x86::Op::Movbe: return gStubs && !gCpu.movbe;
    case x86::Op::Cvtph2ps: case x86::Op::Cvtps2ph: return gStubs && !gCpu.f16c;
    default: return false;
    }
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

// The instruction to emulate for a trap at rip: in place (with its patched first byte put back)
// or in a stub. Returns false when rip is neither.
struct Fetched {
    x86::Insn in;
    uint64_t orig;      // the instruction's own address (rip-relative operands are from there)
    bool stub = false;  // in a stub: rsp is 128 lower than the game's, and execution goes on at rip+1
};
bool fetch(uint64_t rip, Fetched* f) {
    if (!gStubTable.empty() && rip >= gStubTable.front().trap && rip <= gStubTable.back().trap) {
        const auto it = std::lower_bound(gStubTable.begin(), gStubTable.end(), rip,
                                         [](const Stub& s, uint64_t a) { return s.trap < a; });
        if (it != gStubTable.end() && it->trap == rip) {
            f->orig = it->orig;
            f->stub = true;
            return x86::decode(it->bytes, it->len, &f->in);
        }
    }
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
    f->orig = rip;
    f->stub = false;
    return x86::decode(bytes, avail, &f->in);
}

// Runs the fetched instruction on s (holding the trap's registers) and leaves s as execution
// should go on.
x86::Op runFetched(const Fetched& f, x86::State& s) {
    const uint64_t trap = s.rip;
    if (f.stub) {
        s.gpr[4] += kRedZone;
        s.rip = f.orig;
    }
    const x86::Op op = x86::emulate(f.in, s);
    if (op == x86::Op::None) {
        if (f.stub) {
            s.gpr[4] -= kRedZone;
            s.rip = trap;
        }
        return op;
    }
    gCount[size_t(op)].fetch_add(1, std::memory_order_relaxed);
    noteFirst(op, f.orig);
    if (f.stub) {
        s.gpr[4] -= kRedZone;
        s.rip = trap + 1;
    }
    return op;
}

// Instructions that cannot be moved into a stub: branches, calls, returns and the like.
bool isControlFlow(const x86::Insn& in) {
    if (in.vex) return false;
    if (in.map == 0) {
        const uint8_t o = in.opcode;
        if ((o >= 0x70 && o <= 0x7F) || (o >= 0xE0 && o <= 0xE3) || o == 0xE8 || o == 0xE9 || o == 0xEB) return true;
        if (o == 0xC2 || o == 0xC3 || o == 0xCA || o == 0xCB || (o >= 0xCC && o <= 0xCF)) return true;
        if (o == 0xFF && (in.reg & 7) >= 2 && (in.reg & 7) <= 5) return true;
        return false;
    }
    if (in.map == 1) return (in.opcode >= 0x80 && in.opcode <= 0x8F) || in.opcode == 0x05 || in.opcode == 0x0B;
    return false;
}

// The target of a direct jump or call, relative to the function, or -1.
int64_t branchTarget(const x86::Insn& in, uint64_t at) {
    if (in.vex) return -1;
    const bool rel8 = in.map == 0 && ((in.opcode >= 0x70 && in.opcode <= 0x7F) || (in.opcode >= 0xE0 && in.opcode <= 0xE3) ||
                                      in.opcode == 0xEB);
    const bool rel32 = (in.map == 0 && (in.opcode == 0xE8 || in.opcode == 0xE9)) ||
                       (in.map == 1 && in.opcode >= 0x80 && in.opcode <= 0x8F);
    if (rel8) return int64_t(at + in.len) + int8_t(in.imm);
    if (rel32) return int64_t(at + in.len) + int32_t(uint32_t(in.imm));
    return -1;
}

// Decodes one function and adds its instructions to patch to out. Returns false when it does
// not decode to its exact end (it is then left alone).
bool planFunction(uint8_t* start, size_t size, std::vector<Site>& out) {
    struct Item {
        uint32_t at;
        x86::Insn in;
    };
    std::vector<Item> items;
    std::vector<int64_t> targets;
    size_t at = 0;
    while (at < size) {
        x86::Insn in;
        if (!x86::decode(start + at, size - at, &in)) return false;
        items.push_back({uint32_t(at), in});
        if (const int64_t t = branchTarget(in, at); t >= 0) targets.push_back(t);
        at += in.len;
    }
    if (at != size) return false;
    std::sort(targets.begin(), targets.end());
    for (size_t i = 0; i < items.size(); ++i) {
        const x86::Insn& in = items[i].in;
        const x86::Op op = x86::classify(in);
        if (!wanted(op)) continue;
        Site site{start + items[i].at, in.len, op, {}};
        std::memcpy(site.bytes, site.at, in.len);
        if (in.len >= 5) site.stubOk = true;
        else if (i + 1 < items.size()) {
            // Take the next instruction along, unless something jumps to it or it is itself one
            // to patch or cannot run elsewhere.
            const Item& next = items[i + 1];
            const bool target = std::binary_search(targets.begin(), targets.end(), int64_t(next.at));
            if (!target && !isControlFlow(next.in) && !wanted(x86::classify(next.in)) && next.in.len + in.len >= 5) {
                site.movedLen = next.in.len;
                std::memcpy(site.moved, start + next.at, next.in.len);
                if (next.in.ripRelative) site.movedDispAt = int8_t(next.in.len - next.in.immSize - 4);
                site.stubOk = true;
            }
        }
        out.push_back(site);
    }
    return true;
}

// Room for stubs within a 32-bit jump of the game's code: just below it (the guest range starts
// at the eboot, and the host keeps its own allocations low).
bool stubSpace(const Module& m, size_t need) {
    if (size_t(gStubEnd - gStubNext) >= need) return true;
    const size_t size = std::max<size_t>(need, 1 << 22);
    for (uint64_t at = (uint64_t(m.base()) - size) & ~uint64_t(0xFFFF); at > uint64_t(m.base()) - (1ull << 30);
         at -= 1 << 22) {
#ifdef _WIN32
        void* p = VirtualAlloc(reinterpret_cast<void*>(at), size, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#else
        void* p = mmap(reinterpret_cast<void*>(at), size, PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) p = nullptr;
        else if (uint64_t(p) != at) {
            munmap(p, size);
            p = nullptr;
        }
#endif
        if (p) {
            gStubNext = static_cast<uint8_t*>(p);
            gStubEnd = gStubNext + size;
            return true;
        }
    }
    return false;
}

bool fitsRel32(int64_t d) { return d >= INT32_MIN && d <= INT32_MAX; }

// Sends the instruction at site through a new stub. Returns false (nothing changed) when there is
// no room within reach.
bool writeStub(const Module& m, const Site& site) {
    constexpr size_t kMax = 5 + 1 + 8 + 15 + 5;
    if (!stubSpace(m, kMax)) return false;
    uint8_t* const stub = gStubNext;
    const uint64_t back = uint64_t(site.at) + site.len + site.movedLen;
    if (!fitsRel32(int64_t(uint64_t(stub)) - int64_t(uint64_t(site.at) + 5))) return false;
    uint8_t* p = stub;
    static const uint8_t kDown[] = {0x48, 0x8D, 0x64, 0x24, 0x80};             // lea rsp, [rsp-128]
    static const uint8_t kUp[] = {0x48, 0x8D, 0xA4, 0x24, 0x80, 0x00, 0x00, 0x00}; // lea rsp, [rsp+128]
    std::memcpy(p, kDown, sizeof kDown);
    p += sizeof kDown;
    const uint64_t trap = uint64_t(p);
    *p++ = 0x06;
    std::memcpy(p, kUp, sizeof kUp);
    p += sizeof kUp;
    if (site.movedLen) {
        std::memcpy(p, site.moved, site.movedLen);
        if (site.movedDispAt >= 0) {
            int32_t disp;
            std::memcpy(&disp, p + site.movedDispAt, 4);
            const int64_t target = int64_t(uint64_t(site.at) + site.len + site.movedLen) + disp;
            const int64_t now = target - int64_t(uint64_t(p) + site.movedLen);
            if (!fitsRel32(now)) return false;
            disp = int32_t(now);
            std::memcpy(p + site.movedDispAt, &disp, 4);
        }
        p += site.movedLen;
    }
    const int64_t rel = int64_t(back) - int64_t(uint64_t(p) + 5);
    if (!fitsRel32(rel)) return false;
    *p++ = 0xE9;
    const int32_t rel32 = int32_t(rel);
    std::memcpy(p, &rel32, 4);
    p += 4;
    gStubNext = p;

    Stub entry{trap, uint64_t(site.at), site.len, {}};
    std::memcpy(entry.bytes, site.bytes, site.len);
    gStubTable.push_back(entry);

    // The instruction (and the one taken along) becomes a jump to the stub, then int3 padding.
    const int32_t to = int32_t(int64_t(uint64_t(stub)) - int64_t(uint64_t(site.at) + 5));
    site.at[0] = 0xE9;
    std::memcpy(site.at + 1, &to, 4);
    for (size_t i = 5; i < size_t(site.len) + site.movedLen; ++i) site.at[i] = 0xCC;
    return true;
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
    if (const char* stubs = std::getenv("LPN_CPU_STUBS")) gStubs = std::strcmp(stubs, "0") != 0;
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
    bool any = false;
    for (unsigned i = 1; i < unsigned(x86::Op::Count); ++i) any |= wanted(x86::Op(i));
    if (!any) return;
    std::string error;
    const auto functions = ps4elf::functionsFromEhFrame(m.elf(), &error);
    if (functions.empty()) {
        std::fprintf(stderr, "[cpu] %s: no function list (%s), nothing patched\n", m.name().c_str(), error.c_str());
        return;
    }
    // Decode each function from its start, so only real instruction boundaries are touched.
    size_t skipped = 0;
    std::vector<Site> sites;
    for (const auto& f : functions) {
        if (f.start + f.size > m.size()) continue;
        std::vector<Site> here;
        if (planFunction(m.base() + f.start, f.size, here)) sites.insert(sites.end(), here.begin(), here.end());
        else ++skipped;
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
    size_t patched = 0, changed = 0, inPlace = 0;
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    for (const Site& site : scan->sites) {
        uint8_t* p = site.at;
        // Another patch (title menu, draw sites...) rewrote this code since the scan: leave it.
        if (std::memcmp(p, site.bytes, site.len) != 0 ||
            (site.movedLen && std::memcmp(p + site.len, site.moved, site.movedLen) != 0)) {
            ++changed;
            continue;
        }
        // Other patches (display settings) may already have set this page back to read+exec;
        // Module::protect sets the final protection afterwards.
        const uintptr_t first = uintptr_t(p) & ~(page - 1);
        mprotect(reinterpret_cast<void*>(first), uintptr_t(p) + 15 - first, PROT_READ | PROT_WRITE | PROT_EXEC);
        ++counts[size_t(site.op)];
        ++patched;
        if (gStubs && site.stubOk && writeStub(m, site)) continue;
        if (gStubs) ++inPlace;
        gPatches.push_back({uint64_t(p), *p});
        *p = 0x06;
    }
    std::sort(gPatches.begin(), gPatches.end(), [](const Patch& a, const Patch& b) { return a.at < b.at; });
    std::sort(gStubTable.begin(), gStubTable.end(), [](const Stub& a, const Stub& b) { return a.trap < b.trap; });
    // A trap in place on Windows can overwrite the game's red zone: count them.
    if (inPlace)
        std::fprintf(stderr, "[cpu] %s: %zu of them trap in place (no stub possible), which may misbehave on Windows\n",
                     m.name().c_str(), inPlace);

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
    Fetched f;
    if (!fetch(c.Rip, &f) || x86::classify(f.in) == x86::Op::None) return false;
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
    if (runFetched(f, s) == x86::Op::None) return false;
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
    Fetched f;
    if (!fetch(uint64_t(g[REG_RIP]), &f) || x86::classify(f.in) == x86::Op::None) return false;
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
    if (runFetched(f, s) == x86::Op::None) return false;
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
