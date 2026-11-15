// common/x86_emu: the software BMI1, LZCNT, MOVBE and F16C against the CPU running the same
// bytes (skipped where the CPU lacks them), and the length decoder on known instructions.
#include "common/x86_emu.h"

#include <cpuid.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

int gFailures = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c);                                 \
            ++gFailures;                                                                                               \
        }                                                                                                              \
    } while (0)

// The registers the native runner loads and stores: gpr in encoding order (rsp unused), flags.
struct Regs {
    uint64_t gpr[16];
    uint64_t flags;
};

extern "C" void lpnRunNative(Regs* r, void* code);
extern "C" Regs* lpnRunRegs;
extern "C" void* lpnRunCode;
Regs* lpnRunRegs;
void* lpnRunCode;

// Windows passes the arguments in rcx and rdx, and rdi and rsi are callee-saved there.
#ifdef _WIN32
#define LPN_WIN_ENTRY "push %rdi\n push %rsi\n mov %rcx, %rdi\n mov %rdx, %rsi\n"
#define LPN_WIN_EXIT "pop %rsi\n pop %rdi\n"
#else
#define LPN_WIN_ENTRY ""
#define LPN_WIN_EXIT ""
#endif

// Loads every register but rsp from *r, calls code (the instruction then ret), stores them back.
__asm__(R"(
    .text
    .globl lpnRunNative
lpnRunNative:
    push %rbx
    push %rbp
    push %r12
    push %r13
    push %r14
    push %r15
)" LPN_WIN_ENTRY R"(
    mov %rdi, lpnRunRegs(%rip)
    mov %rsi, lpnRunCode(%rip)
    pushq 128(%rdi)
    popfq
    mov 0(%rdi), %rax
    mov 8(%rdi), %rcx
    mov 16(%rdi), %rdx
    mov 24(%rdi), %rbx
    mov 40(%rdi), %rbp
    mov 48(%rdi), %rsi
    mov 64(%rdi), %r8
    mov 72(%rdi), %r9
    mov 80(%rdi), %r10
    mov 88(%rdi), %r11
    mov 96(%rdi), %r12
    mov 104(%rdi), %r13
    mov 112(%rdi), %r14
    mov 120(%rdi), %r15
    mov 56(%rdi), %rdi
    call *lpnRunCode(%rip)
    pushfq
    push %rdi
    mov lpnRunRegs(%rip), %rdi
    popq 56(%rdi)
    popq 128(%rdi)
    mov %rax, 0(%rdi)
    mov %rcx, 8(%rdi)
    mov %rdx, 16(%rdi)
    mov %rbx, 24(%rdi)
    mov %rbp, 40(%rdi)
    mov %rsi, 48(%rdi)
    mov %r8, 64(%rdi)
    mov %r9, 72(%rdi)
    mov %r10, 80(%rdi)
    mov %r11, 88(%rdi)
    mov %r12, 96(%rdi)
    mov %r13, 104(%rdi)
    mov %r14, 112(%rdi)
    mov %r15, 120(%rdi)
)" LPN_WIN_EXIT R"(
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    pop %rbx
    ret
)");

uint8_t* gCode = nullptr;

void runNative(const std::vector<uint8_t>& insn, Regs& r) {
    std::memcpy(gCode, insn.data(), insn.size());
    gCode[insn.size()] = 0xC3;
    __builtin___clear_cache(reinterpret_cast<char*>(gCode), reinterpret_cast<char*>(gCode + 32));
    lpnRunNative(&r, gCode);
}

struct Cpu {
    bool bmi1, lzcnt, movbe, f16c;
};
Cpu readCpu() {
    unsigned a, b, c, d;
    Cpu f{};
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        f.movbe = c & (1u << 22);
        f.f16c = (c & (1u << 29)) && (c & (1u << 28));
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) f.bmi1 = b & (1u << 3);
    if (__get_cpuid(0x80000001, &a, &b, &c, &d)) f.lzcnt = c & (1u << 5);
    return f;
}

constexpr uint64_t kCF = 1, kZF = 1 << 6, kSF = 1 << 7, kOF = 1 << 11;

std::mt19937_64 gRng(79);

uint64_t interesting() {
    switch (gRng() % 8) {
    case 0: return 0;
    case 1: return ~uint64_t(0);
    case 2: return uint64_t(1) << (gRng() % 64);
    case 3: return gRng() & 0xFFFFFFFF;
    case 4: return gRng() >> (gRng() % 64);
    case 5: return uint64_t(0x80000000);
    default: return gRng();
    }
}

// Runs insn natively and emulated from the same random registers, then compares the general
// registers and the flags in flagMask. memReg, when not -1, points at a 16-byte buffer (filled
// randomly, compared too).
void compare(const char* what, const std::vector<uint8_t>& insn, uint64_t flagMask, int memReg = -1) {
    Regs native{};
    for (int i = 0; i < 16; ++i) native.gpr[i] = interesting();
    native.gpr[4] = 0;
    native.flags = 0x202 | (gRng() & (kCF | kZF | kSF | kOF | 4 | 16));
    alignas(16) uint8_t memNative[32], memEmu[32];
    for (auto& b : memNative) b = uint8_t(gRng());
    std::memcpy(memEmu, memNative, sizeof memEmu);
    Regs emu = native;
    if (memReg >= 0) {
        native.gpr[memReg] = uint64_t(memNative);
        emu.gpr[memReg] = uint64_t(memEmu);
    }

    x86::Insn in;
    CHECK(x86::decode(insn.data(), insn.size(), &in));
    CHECK(in.len == insn.size());
    x86::State s;
    std::memcpy(s.gpr, emu.gpr, sizeof s.gpr);
    s.rflags = emu.flags;
    s.rip = 0x1000;
    const x86::Op op = x86::emulate(in, s);
    CHECK(op != x86::Op::None);
    CHECK(s.rip == 0x1000 + insn.size());

    runNative(insn, native);
    bool same = (native.flags & flagMask) == (s.rflags & flagMask);
    for (int i = 0; i < 16; ++i)
        if (i != 4 && i != memReg && native.gpr[i] != s.gpr[i]) same = false;
    if (std::memcmp(memNative, memEmu, sizeof memNative) != 0) same = false;
    if (!same) {
        std::fprintf(stderr, "%s: emulation differs from the CPU\n", what);
        ++gFailures;
    }
}

void testGpr(const Cpu& cpu) {
    struct Case {
        const char* name;
        std::vector<uint8_t> bytes;
        uint64_t flags;
        bool* has;
        int memReg;
    };
    bool bmi1 = cpu.bmi1, lzcnt = cpu.lzcnt, movbe = cpu.movbe;
    const uint64_t blsFlags = kCF | kZF | kSF | kOF;
    const Case cases[] = {
        {"andn eax, ecx, edx", {0xC4, 0xE2, 0x70, 0xF2, 0xC2}, kCF | kZF | kSF | kOF, &bmi1, -1},
        {"andn rax, rcx, rdx", {0xC4, 0xE2, 0xF0, 0xF2, 0xC2}, kCF | kZF | kSF | kOF, &bmi1, -1},
        {"andn r9, r10, [rbx+8]", {0xC4, 0x62, 0xA8, 0xF2, 0x4B, 0x08}, kCF | kZF | kSF | kOF, &bmi1, 3},
        {"bextr eax, ecx, edx", {0xC4, 0xE2, 0x68, 0xF7, 0xC1}, kCF | kZF | kOF, &bmi1, -1},
        {"bextr rax, rcx, rdx", {0xC4, 0xE2, 0xE8, 0xF7, 0xC1}, kCF | kZF | kOF, &bmi1, -1},
        {"blsr eax, ecx", {0xC4, 0xE2, 0x78, 0xF3, 0xC9}, blsFlags, &bmi1, -1},
        {"blsr r11, r12", {0xC4, 0xC2, 0xA0, 0xF3, 0xCC}, blsFlags, &bmi1, -1},
        {"blsmsk rax, rcx", {0xC4, 0xE2, 0xF8, 0xF3, 0xD1}, blsFlags, &bmi1, -1},
        {"blsi eax, [rsi]", {0xC4, 0xE2, 0x78, 0xF3, 0x1E}, blsFlags, &bmi1, 6},
        {"blsi rax, rcx", {0xC4, 0xE2, 0xF8, 0xF3, 0xD9}, blsFlags, &bmi1, -1},
        {"tzcnt eax, ecx", {0xF3, 0x0F, 0xBC, 0xC1}, kCF | kZF, &bmi1, -1},
        {"tzcnt rax, rcx", {0xF3, 0x48, 0x0F, 0xBC, 0xC1}, kCF | kZF, &bmi1, -1},
        {"tzcnt ax, cx", {0x66, 0xF3, 0x0F, 0xBC, 0xC1}, kCF | kZF, &bmi1, -1},
        {"tzcnt r8, [rdi+4]", {0xF3, 0x4C, 0x0F, 0xBC, 0x47, 0x04}, kCF | kZF, &bmi1, 7},
        {"lzcnt eax, ecx", {0xF3, 0x0F, 0xBD, 0xC1}, kCF | kZF, &lzcnt, -1},
        {"lzcnt rax, rcx", {0xF3, 0x48, 0x0F, 0xBD, 0xC1}, kCF | kZF, &lzcnt, -1},
        {"lzcnt ax, cx", {0x66, 0xF3, 0x0F, 0xBD, 0xC1}, kCF | kZF, &lzcnt, -1},
        {"lzcnt r13d, [rdx]", {0xF3, 0x44, 0x0F, 0xBD, 0x2A}, kCF | kZF, &lzcnt, 2},
        {"movbe eax, [rsi]", {0x0F, 0x38, 0xF0, 0x06}, 0, &movbe, 6},
        {"movbe rax, [rsi+8]", {0x48, 0x0F, 0x38, 0xF0, 0x46, 0x08}, 0, &movbe, 6},
        {"movbe ax, [rsi]", {0x66, 0x0F, 0x38, 0xF0, 0x06}, 0, &movbe, 6},
        {"movbe [rsi], ecx", {0x0F, 0x38, 0xF1, 0x0E}, 0, &movbe, 6},
        {"movbe [rsi+4], r9", {0x4C, 0x0F, 0x38, 0xF1, 0x4E, 0x04}, 0, &movbe, 6},
        {"movbe [rsi], cx", {0x66, 0x0F, 0x38, 0xF1, 0x0E}, 0, &movbe, 6},
    };
    for (const Case& c : cases) {
        if (!*c.has) {
            std::fprintf(stderr, "skipped %s: not on this CPU\n", c.name);
            continue;
        }
        for (int i = 0; i < 2000; ++i) compare(c.name, c.bytes, c.flags, c.memReg);
    }
}

#define F16C __attribute__((target("f16c,avx")))

F16C uint16_t nativeToHalf(float f, int rc) {
    const __m128 v = _mm_set_ss(f);
    __m128i h;
    switch (rc) {
    case 0: h = _mm_cvtps_ph(v, 0); break;
    case 1: h = _mm_cvtps_ph(v, 1); break;
    case 2: h = _mm_cvtps_ph(v, 2); break;
    default: h = _mm_cvtps_ph(v, 3); break;
    }
    return uint16_t(_mm_extract_epi16(h, 0));
}

F16C float nativeToFloat(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

uint32_t bitsOf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

void testHalfConversions(const Cpu& cpu) {
    if (!cpu.f16c) {
        std::fprintf(stderr, "skipped F16C: not on this CPU\n");
        return;
    }
    for (uint32_t h = 0; h < 0x10000; ++h)
        if (bitsOf(x86::halfToFloat(uint16_t(h))) != bitsOf(nativeToFloat(uint16_t(h)))) {
            std::fprintf(stderr, "halfToFloat(0x%04x) differs\n", h);
            ++gFailures;
            break;
        }
    auto check = [](uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, 4);
        for (int rc = 0; rc < 4; ++rc)
            if (x86::floatToHalf(f, rc) != nativeToHalf(f, rc)) {
                std::fprintf(stderr, "floatToHalf(0x%08x, rc %d) = 0x%04x, CPU 0x%04x\n", bits, rc,
                             x86::floatToHalf(f, rc), nativeToHalf(f, rc));
                ++gFailures;
                return false;
            }
        return true;
    };
    // Every float with the exponents around the half range, and random ones elsewhere.
    for (uint32_t e = 96; e <= 145; ++e)
        for (uint32_t m = 0; m < (1u << 23); m += 7)
            if (!check((e << 23) | m) || !check(0x80000000u | (e << 23) | m)) return;
    for (int i = 0; i < 2000000; ++i)
        if (!check(uint32_t(gRng()))) return;
    for (uint32_t b : {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F800001u, 0xFFBFFFFFu, 1u})
        check(b);
}

// vcvtph2ps / vcvtps2ph as instructions: operands, ymm halves, memory forms.
F16C void testF16cInstructions(const Cpu& cpu) {
    if (!cpu.f16c) return;
    alignas(32) uint8_t xmm[16][16], ymmh[16][16];
    for (int round = 0; round < 500; ++round) {
        for (auto& r : xmm)
            for (auto& b : r) b = uint8_t(gRng());
        for (auto& r : ymmh)
            for (auto& b : r) b = uint8_t(gRng());
        float src[8];
        for (float& f : src) f = float(int64_t(gRng() % 200000) - 100000) / float(1 + gRng() % 1000);
        alignas(16) uint16_t halves[8];
        for (uint16_t& h : halves) h = uint16_t(gRng());

        x86::State s;
        s.xmm = xmm;
        s.ymmHigh = ymmh;
        s.mxcsr = 0x1F80 | (uint32_t(gRng() % 4) << 13);
        x86::Insn in;

        // vcvtph2ps ymm3, xmm9: C4 C2 7D 13 D9
        std::memcpy(xmm[9], halves, 16);
        const uint8_t a[] = {0xC4, 0xC2, 0x7D, 0x13, 0xD9};
        CHECK(x86::decode(a, sizeof a, &in) && in.len == sizeof a);
        CHECK(x86::emulate(in, s) == x86::Op::Cvtph2ps);
        alignas(32) float want8[8];
        _mm256_store_ps(want8, _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(halves))));
        CHECK(std::memcmp(xmm[3], want8, 16) == 0 && std::memcmp(ymmh[3], want8 + 4, 16) == 0);

        // vcvtph2ps xmm2, qword [rax]: C4 E2 79 13 10; clears the upper half of ymm2
        s.gpr[0] = uint64_t(halves);
        const uint8_t b[] = {0xC4, 0xE2, 0x79, 0x13, 0x10};
        CHECK(x86::decode(b, sizeof b, &in) && in.len == sizeof b);
        CHECK(x86::emulate(in, s) == x86::Op::Cvtph2ps);
        alignas(16) float want4[4];
        _mm_store_ps(want4, _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(halves))));
        static const uint8_t zero[16] = {};
        CHECK(std::memcmp(xmm[2], want4, 16) == 0 && std::memcmp(ymmh[2], zero, 16) == 0);

        // vcvtps2ph xmm12, ymm5, imm: C4 C3 7D 1D EC imm (rm = 12 dest, reg = 5 source)
        std::memcpy(xmm[5], src, 16);
        std::memcpy(ymmh[5], src + 4, 16);
        const int imm = int(gRng() % 8);
        const uint8_t c[] = {0xC4, 0xC3, 0x7D, 0x1D, 0xEC, uint8_t(imm)};
        CHECK(x86::decode(c, sizeof c, &in) && in.len == sizeof c);
        CHECK(x86::emulate(in, s) == x86::Op::Cvtps2ph);
        const unsigned savedCsr = _mm_getcsr();
        _mm_setcsr(s.mxcsr);
        const __m256 v = _mm256_loadu_ps(src);
        __m128i h;
        switch (imm) {
        case 0: h = _mm256_cvtps_ph(v, 0); break;
        case 1: h = _mm256_cvtps_ph(v, 1); break;
        case 2: h = _mm256_cvtps_ph(v, 2); break;
        case 3: h = _mm256_cvtps_ph(v, 3); break;
        case 4: h = _mm256_cvtps_ph(v, 4); break;
        case 5: h = _mm256_cvtps_ph(v, 5); break;
        case 6: h = _mm256_cvtps_ph(v, 6); break;
        default: h = _mm256_cvtps_ph(v, 7); break;
        }
        // vcvtps2ph [rcx], xmm5 (four), imm 4: C4 E3 79 1D 29 04
        const __m128i h4 = _mm_cvtps_ph(_mm_loadu_ps(src), 4);
        _mm_setcsr(savedCsr);
        alignas(16) uint8_t want[16];
        _mm_store_si128(reinterpret_cast<__m128i*>(want), h);
        CHECK(std::memcmp(xmm[12], want, 16) == 0 && std::memcmp(ymmh[12], zero, 16) == 0);

        alignas(16) uint8_t out[16];
        std::memset(out, 0xAB, sizeof out);
        s.gpr[1] = uint64_t(out);
        const uint8_t d[] = {0xC4, 0xE3, 0x79, 0x1D, 0x29, 0x04};
        CHECK(x86::decode(d, sizeof d, &in) && in.len == sizeof d);
        CHECK(x86::emulate(in, s) == x86::Op::Cvtps2ph);
        _mm_store_si128(reinterpret_cast<__m128i*>(want), h4);
        CHECK(std::memcmp(out, want, 8) == 0 && out[8] == 0xAB);
    }
}

void testDecoderLengths() {
    struct Case {
        std::vector<uint8_t> bytes;
    };
    const Case cases[] = {
        {{0x90}},                                                       // nop
        {{0x48, 0x8B, 0x05, 1, 2, 3, 4}},                               // mov rax, [rip+x]
        {{0x48, 0xB8, 1, 2, 3, 4, 5, 6, 7, 8}},                         // mov rax, imm64
        {{0x66, 0xB8, 1, 2}},                                           // mov ax, imm16
        {{0x48, 0x81, 0xEC, 0x00, 0x01, 0x00, 0x00}},                   // sub rsp, 0x100
        {{0x66, 0x81, 0x7C, 0x24, 0x08, 0x34, 0x12}},                   // cmp word [rsp+8], 0x1234
        {{0xF6, 0x40, 0x10, 0x01}},                                     // test byte [rax+16], 1
        {{0xF7, 0xC1, 1, 0, 0, 0}},                                     // test ecx, 1
        {{0xF7, 0xD9}},                                                 // neg ecx
        {{0xE8, 1, 2, 3, 4}},                                           // call rel32
        {{0x0F, 0x84, 1, 2, 3, 4}},                                     // je rel32
        {{0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00}},       // nopw [rax+rax+0]
        {{0xC5, 0xF8, 0x77}},                                           // vzeroupper
        {{0xC5, 0xFC, 0x28, 0x44, 0x24, 0x20}},                         // vmovaps ymm0, [rsp+32]
        {{0xC4, 0xE3, 0x7D, 0x18, 0xC1, 0x01}},                         // vinsertf128 ymm0, ymm0, xmm1, 1
        {{0xC5, 0xF9, 0x70, 0xC0, 0x1B}},                               // vpshufd xmm0, xmm0, 0x1b
        {{0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x08}},                         // palignr xmm0, xmm1, 8
        {{0xF2, 0x0F, 0x38, 0xF1, 0xC1}},                               // crc32 eax, ecx
        {{0xC8, 0x10, 0x00, 0x00}},                                     // enter 16, 0
        {{0xA1, 1, 2, 3, 4, 5, 6, 7, 8}},                               // mov eax, [moffs64]
        {{0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0}},                   // mov rax, fs:[0]
        {{0xF0, 0x48, 0x0F, 0xB1, 0x0F}},                               // lock cmpxchg [rdi], rcx
        {{0x0F, 0xBA, 0xE0, 0x05}},                                     // bt eax, 5
        {{0x42, 0x8B, 0x44, 0x88, 0x10}},                               // mov eax, [rax+r9*4+16]
        {{0x41, 0x8B, 0x45, 0x00}},                                     // mov eax, [r13+0]
        {{0x8B, 0x04, 0x25, 1, 2, 3, 4}},                               // mov eax, [abs32]
    };
    for (const Case& c : cases) {
        x86::Insn in;
        if (!x86::decode(c.bytes.data(), c.bytes.size(), &in) || in.len != c.bytes.size()) {
            std::fprintf(stderr, "decode length wrong for %02x %02x %02x...\n", c.bytes[0],
                         c.bytes.size() > 1 ? c.bytes[1] : 0, c.bytes.size() > 2 ? c.bytes[2] : 0);
            ++gFailures;
        }
    }
    x86::Insn in;
    const uint8_t evex[] = {0x62, 0xF1, 0x7C, 0x48, 0x28, 0xC1};
    CHECK(!x86::decode(evex, sizeof evex, &in));
    const uint8_t pushEs[] = {0x06};
    CHECK(!x86::decode(pushEs, sizeof pushEs, &in));
    const uint8_t crc[] = {0xF2, 0x0F, 0x38, 0xF0, 0x06};
    CHECK(x86::decode(crc, sizeof crc, &in) && x86::classify(in) == x86::Op::None);
    const uint8_t bsr[] = {0x0F, 0xBD, 0xC1};
    CHECK(x86::decode(bsr, sizeof bsr, &in) && x86::classify(in) == x86::Op::None);
}

} // namespace

int main() {
#ifdef _WIN32
    gCode = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
#else
    gCode = static_cast<uint8_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
#endif
    const Cpu cpu = readCpu();
    testDecoderLengths();
    testGpr(cpu);
    testHalfConversions(cpu);
    testF16cInstructions(cpu);
    if (gFailures) {
        std::fprintf(stderr, "%d failures\n", gFailures);
        return 1;
    }
    std::printf("x86_emu_test: ok\n");
    return 0;
}
