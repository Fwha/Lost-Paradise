// GCN (Sea Islands / "Liverpool", as in the PS4 GPU) instruction decoding.
//
// Decodes encodings, operands and lengths for every instruction format; op names cover the
// instructions shaders actually use, others print as <format>_op<N>.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

enum class Format {
    SOP2, SOPK, SOP1, SOPC, SOPP, SMRD, VOP2, VOP1, VOPC, VOP3, VINTRP, DS, MUBUF, MTBUF, MIMG, EXP, FLAT,
    Unknown,
};
const char* formatName(Format f);

struct Instruction {
    Format format = Format::Unknown;
    uint32_t op = 0;       // opcode within the format (VOP3 uses the unified 9-bit space)
    uint32_t words[3] = {}; // raw dwords: instruction (1 or 2) plus literal
    uint32_t size = 4;      // bytes including any literal
    bool hasLiteral = false;
    uint32_t literal = 0;
};

// Decode the instruction at code[0]; `avail` is the number of dwords readable.
Instruction decode(const uint32_t* code, size_t avail);

// Mnemonic, e.g. "v_mul_f32" or "vop2_op45" for unnamed ops.
std::string mnemonic(const Instruction& in);

// Full disassembly line (mnemonic and operands).
std::string disassemble(const Instruction& in, uint32_t pc);

// True for s_endpgm.
inline bool isEndProgram(const Instruction& in) { return in.format == Format::SOPP && in.op == 1; }

} // namespace gcn
