// Software versions of the x86 instructions the PS4's CPU (AMD Jaguar) has that older AVX CPUs
// lack: BMI1 (andn, bextr, blsi, blsmsk, blsr, tzcnt), LZCNT, MOVBE and F16C. The host runs the
// game's code natively, so on a CPU without them the game would stop on an illegal instruction
// or, for lzcnt and tzcnt (which older CPUs run as bsr and bsf), quietly compute wrong values.
// host/cpu_compat.cpp uses this from its illegal-instruction handler.
#pragma once

#include <cstddef>
#include <cstdint>

namespace x86 {

// One decoded instruction: its length and the fields the emulation below needs.
struct Insn {
    uint8_t len = 0;
    uint8_t opcodeAt = 0; // offset of the opcode byte (after prefixes, REX or VEX)
    uint8_t map = 0;      // 0 one-byte, 1 0F, 2 0F 38, 3 0F 3A
    uint8_t opcode = 0;
    bool opsize = false, addr32 = false, rep = false, repne = false, lock = false;
    uint8_t segment = 0; // 0x64 (fs) or 0x65 (gs) when overridden, else 0
    uint8_t rex = 0;     // REX byte (0x40-0x4F) or 0
    bool vex = false;
    uint8_t vexPP = 0, vexL = 0, vexW = 0, vexV = 0; // vexV: register number, already un-inverted
    bool r = false, x = false, b = false;            // REX/VEX register extensions
    bool w = false;                                  // REX.W or VEX.W
    bool hasModrm = false;
    uint8_t mod = 0, reg = 0, rm = 0; // reg and rm include the R and B extensions
    bool hasSib = false;
    uint8_t scale = 0, index = 0, base = 0; // index and base include X and B; index 4 (no X) is none
    int32_t disp = 0;
    bool ripRelative = false, noBase = false;
    uint8_t immSize = 0;
    uint64_t imm = 0;

    bool memoryOperand() const { return hasModrm && mod != 3; }
};

// Decodes the instruction at p (at most avail bytes are read). Returns false for bytes that are
// not a valid x86-64 instruction this decoder knows (EVEX, XOP, 3DNow!, invalid opcodes).
bool decode(const uint8_t* p, size_t avail, Insn* out);

// A thread's registers as the emulation sees them. gpr is in encoding order (rax, rcx, rdx, rbx,
// rsp, rbp, rsi, rdi, r8-r15). xmm points at 16 registers of 16 bytes; ymmHigh at their upper
// halves (bits 255:128), 16 x 16 bytes, or is null when the context has no AVX state.
struct State {
    uint64_t gpr[16] = {};
    uint64_t rip = 0;
    uint64_t rflags = 0;
    uint32_t mxcsr = 0x1f80;
    uint8_t (*xmm)[16] = nullptr;
    uint8_t (*ymmHigh)[16] = nullptr;
    bool ymmHighWritten = false; // set when an emulated instruction changed ymmHigh
};

enum class Op : uint8_t { None, Andn, Bextr, Blsi, Blsmsk, Blsr, Tzcnt, Lzcnt, Movbe, Cvtph2ps, Cvtps2ph, Count };
const char* opName(Op op);

// The instruction emulate() would run for in, or Op::None.
Op classify(const Insn& in);

// Runs in (decoded from the bytes at s.rip) on s, memory operands read and written in place,
// and moves s.rip past it. Returns the instruction run, or Op::None (s untouched) when in is
// not one of these or uses a segment override.
Op emulate(const Insn& in, State& s);

// IEEE half <-> single as F16C converts them. rc: 0 nearest even, 1 down, 2 up, 3 toward zero.
float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f, int rc);

} // namespace x86
