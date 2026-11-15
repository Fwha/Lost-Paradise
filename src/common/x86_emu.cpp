#include "common/x86_emu.h"

#include <cstring>

namespace x86 {

namespace {

// One-byte opcodes in 64-bit mode. Bits: has ModRM, then the immediate kind.
enum : uint8_t {
    M = 0x01,   // ModRM
    I8 = 0x02,  // imm8
    IZ = 0x04,  // imm16/32 (16 with 66)
    I16 = 0x08, // imm16
    IV = 0x10,  // imm16/32/64 (mov r, imm)
    MO = 0x20,  // moffs (8 bytes, 4 with 67)
    BAD = 0x80, // invalid in 64-bit mode (or a prefix, handled before the table)
};

// clang-format off
const uint8_t kOneByte[256] = {
    // 0x00
    M, M, M, M, I8, IZ, BAD, BAD, M, M, M, M, I8, IZ, BAD, BAD,
    // 0x10
    M, M, M, M, I8, IZ, BAD, BAD, M, M, M, M, I8, IZ, BAD, BAD,
    // 0x20
    M, M, M, M, I8, IZ, BAD, BAD, M, M, M, M, I8, IZ, BAD, BAD,
    // 0x30
    M, M, M, M, I8, IZ, BAD, BAD, M, M, M, M, I8, IZ, BAD, BAD,
    // 0x40 (REX)
    BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD,
    // 0x50
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0x60
    BAD, BAD, BAD, M, BAD, BAD, BAD, BAD, IZ, M | IZ, I8, M | I8, 0, 0, 0, 0,
    // 0x70
    I8, I8, I8, I8, I8, I8, I8, I8, I8, I8, I8, I8, I8, I8, I8, I8,
    // 0x80
    M | I8, M | IZ, BAD, M | I8, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0x90
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, BAD, 0, 0, 0, 0, 0,
    // 0xA0
    MO, MO, MO, MO, 0, 0, 0, 0, I8, IZ, 0, 0, 0, 0, 0, 0,
    // 0xB0
    I8, I8, I8, I8, I8, I8, I8, I8, IV, IV, IV, IV, IV, IV, IV, IV,
    // 0xC0
    M | I8, M | I8, I16, 0, BAD, BAD, M | I8, M | IZ, I16 | I8, 0, I16, 0, 0, I8, BAD, 0,
    // 0xD0
    M, M, M, M, BAD, BAD, BAD, 0, M, M, M, M, M, M, M, M,
    // 0xE0
    I8, I8, I8, I8, I8, I8, I8, I8, IZ, IZ, BAD, I8, 0, 0, 0, 0,
    // 0xF0
    BAD, 0, BAD, BAD, 0, 0, M, M, 0, 0, 0, 0, 0, 0, M, M,
};

// 0F xx.
const uint8_t kTwoByte[256] = {
    // 0x00
    M, M, M, M, BAD, 0, 0, 0, 0, 0, BAD, 0, BAD, M, 0, BAD,
    // 0x10
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0x20
    M, M, M, M, BAD, BAD, BAD, BAD, M, M, M, M, M, M, M, M,
    // 0x30
    0, 0, 0, 0, 0, 0, BAD, 0, BAD, BAD, BAD, BAD, BAD, BAD, BAD, BAD,
    // 0x40
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0x50
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0x60
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0x70
    M | I8, M | I8, M | I8, M | I8, M, M, M, 0, M, M, BAD, BAD, M, M, M, M,
    // 0x80
    IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ, IZ,
    // 0x90
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0xA0
    0, 0, 0, M, M | I8, M, BAD, BAD, 0, 0, 0, M, M | I8, M, M, M,
    // 0xB0
    M, M, M, M, M, M, M, M, M, M, M | I8, M, M, M, M, M,
    // 0xC0
    M, M, M | I8, M, M | I8, M | I8, M | I8, M, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0xD0
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0xE0
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
    // 0xF0
    M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
};
// clang-format on

bool isPrefix(uint8_t c) {
    switch (c) {
    case 0x26: case 0x2E: case 0x36: case 0x3E: case 0x64: case 0x65:
    case 0x66: case 0x67: case 0xF0: case 0xF2: case 0xF3:
        return true;
    default:
        return false;
    }
}

// Reads ModRM, SIB and displacement at p[i]. Returns false when past avail.
bool modrm(const uint8_t* p, size_t avail, size_t& i, Insn& in) {
    if (i >= avail) return false;
    const uint8_t m = p[i++];
    in.hasModrm = true;
    in.mod = m >> 6;
    in.reg = uint8_t(((m >> 3) & 7) | (in.r ? 8 : 0));
    const uint8_t rm = m & 7;
    in.rm = uint8_t(rm | (in.b ? 8 : 0));
    if (in.mod == 3) return true;
    int dispSize = in.mod == 1 ? 1 : in.mod == 2 ? 4 : 0;
    if (rm == 4) {
        if (i >= avail) return false;
        const uint8_t s = p[i++];
        in.hasSib = true;
        in.scale = s >> 6;
        in.index = uint8_t(((s >> 3) & 7) | (in.x ? 8 : 0));
        in.base = uint8_t((s & 7) | (in.b ? 8 : 0));
        if ((s & 7) == 5 && in.mod == 0) {
            in.noBase = true;
            dispSize = 4;
        }
    } else if (rm == 5 && in.mod == 0) {
        in.ripRelative = true;
        dispSize = 4;
    }
    if (i + size_t(dispSize) > avail) return false;
    if (dispSize == 1) in.disp = int8_t(p[i]);
    else if (dispSize == 4) std::memcpy(&in.disp, p + i, 4);
    i += size_t(dispSize);
    return true;
}

bool immediate(const uint8_t* p, size_t avail, size_t& i, Insn& in, int size) {
    if (i + size_t(size) > avail) return false;
    uint64_t v = 0;
    std::memcpy(&v, p + i, size_t(size));
    in.imm = v;
    in.immSize = uint8_t(size);
    i += size_t(size);
    return true;
}

} // namespace

bool decode(const uint8_t* p, size_t avail, Insn* out) {
    Insn in;
    if (avail > 15) avail = 15;
    size_t i = 0;
    for (; i < avail && isPrefix(p[i]); ++i) {
        switch (p[i]) {
        case 0x66: in.opsize = true; break;
        case 0x67: in.addr32 = true; break;
        case 0xF0: in.lock = true; break;
        case 0xF2: in.repne = true; in.rep = false; break;
        case 0xF3: in.rep = true; in.repne = false; break;
        case 0x64: case 0x65: in.segment = p[i]; break;
        default: break; // es/cs/ss/ds: no effect in 64-bit mode
        }
    }
    if (i >= avail) return false;
    if ((p[i] & 0xF0) == 0x40) {
        in.rex = p[i++];
        in.w = in.rex & 8;
        in.r = in.rex & 4;
        in.x = in.rex & 2;
        in.b = in.rex & 1;
        if (i >= avail) return false;
    }

    if (p[i] == 0xC4 || p[i] == 0xC5) {
        // VEX: no REX, 66, F2, F3 or lock before it.
        if (in.rex || in.opsize || in.rep || in.repne || in.lock) return false;
        in.vex = true;
        uint8_t mapSelect = 1;
        if (p[i] == 0xC5) {
            if (i + 1 >= avail) return false;
            const uint8_t v = p[i + 1];
            in.r = !(v & 0x80);
            in.vexV = uint8_t((~v >> 3) & 15);
            in.vexL = (v >> 2) & 1;
            in.vexPP = v & 3;
            i += 2;
        } else {
            if (i + 2 >= avail) return false;
            const uint8_t a = p[i + 1], c = p[i + 2];
            in.r = !(a & 0x80);
            in.x = !(a & 0x40);
            in.b = !(a & 0x20);
            mapSelect = a & 0x1F;
            in.w = c & 0x80;
            in.vexW = in.w;
            in.vexV = uint8_t((~c >> 3) & 15);
            in.vexL = (c >> 2) & 1;
            in.vexPP = c & 3;
            i += 3;
        }
        if (mapSelect < 1 || mapSelect > 3) return false;
        in.map = mapSelect;
        if (i >= avail) return false;
        in.opcodeAt = uint8_t(i);
        in.opcode = p[i++];
        // Every VEX instruction has ModRM except vzeroupper/vzeroall (0F 77).
        if (!(in.map == 1 && in.opcode == 0x77) && !modrm(p, avail, i, in)) return false;
        bool imm8 = in.map == 3;
        if (in.map == 1 && ((in.opcode >= 0x70 && in.opcode <= 0x73) || in.opcode == 0xC2 ||
                            (in.opcode >= 0xC4 && in.opcode <= 0xC6)))
            imm8 = true;
        if (imm8 && !immediate(p, avail, i, in, 1)) return false;
        in.len = uint8_t(i);
        *out = in;
        return true;
    }

    uint8_t flags;
    in.opcodeAt = uint8_t(i);
    if (p[i] == 0x0F) {
        if (++i >= avail) return false;
        if (p[i] == 0x38 || p[i] == 0x3A) {
            in.map = p[i] == 0x38 ? 2 : 3;
            if (++i >= avail) return false;
            flags = in.map == 2 ? M : M | I8;
        } else {
            in.map = 1;
            flags = kTwoByte[p[i]];
            if (p[i] == 0x0F) return false; // 3DNow!
        }
        in.opcodeAt = uint8_t(i);
    } else {
        flags = kOneByte[p[i]];
        if (p[i] == 0x8F && i + 1 < avail && (p[i + 1] & 0x38)) return false; // XOP
    }
    in.opcode = p[i++];
    if (flags & BAD) return false;
    if ((flags & M) && !modrm(p, avail, i, in)) return false;
    if (in.map == 0) {
        // Immediates that depend on ModRM.reg: test in groups F6 and F7.
        if (in.opcode == 0xF6 && in.reg % 8 < 2) flags |= I8;
        if (in.opcode == 0xF7 && in.reg % 8 < 2) flags |= IZ;
    }
    if ((flags & I16) && !immediate(p, avail, i, in, 2)) return false;
    if (flags & I8) {
        // enter (C8) has imm16 then imm8; keep the imm16 in imm.
        Insn tmp = in;
        if (!immediate(p, avail, i, tmp, 1)) return false;
        if (!(flags & I16)) in = tmp;
        else in.immSize = 3;
    }
    if (flags & IZ) {
        // Near branches (E8, E9, 0F 8x) take rel32 whatever the operand size.
        const bool branch = (in.map == 0 && (in.opcode == 0xE8 || in.opcode == 0xE9)) || in.map == 1;
        if (!immediate(p, avail, i, in, in.opsize && !branch ? 2 : 4)) return false;
    }
    if ((flags & IV) && !immediate(p, avail, i, in, in.w ? 8 : in.opsize ? 2 : 4)) return false;
    if ((flags & MO) && !immediate(p, avail, i, in, in.addr32 ? 4 : 8)) return false;
    in.len = uint8_t(i);
    *out = in;
    return true;
}

const char* opName(Op op) {
    static const char* const names[] = {"none",  "andn",  "bextr", "blsi",      "blsmsk",   "blsr",
                                        "tzcnt", "lzcnt", "movbe", "vcvtph2ps", "vcvtps2ph"};
    return unsigned(op) < unsigned(Op::Count) ? names[unsigned(op)] : "?";
}

Op classify(const Insn& in) {
    if (!in.hasModrm) return Op::None;
    if (in.vex) {
        if (in.map == 2 && in.vexPP == 0 && in.vexL == 0) {
            if (in.opcode == 0xF2) return Op::Andn;
            if (in.opcode == 0xF7) return Op::Bextr;
            if (in.opcode == 0xF3) {
                switch (in.reg & 7) {
                case 1: return Op::Blsr;
                case 2: return Op::Blsmsk;
                case 3: return Op::Blsi;
                default: return Op::None;
                }
            }
        }
        if (in.vexPP == 1 && !in.vexW && in.vexV == 0) {
            if (in.map == 2 && in.opcode == 0x13) return Op::Cvtph2ps;
            if (in.map == 3 && in.opcode == 0x1D) return Op::Cvtps2ph;
        }
        return Op::None;
    }
    if (in.lock) return Op::None;
    if (in.map == 1 && in.rep && (in.opcode == 0xBC || in.opcode == 0xBD))
        return in.opcode == 0xBC ? Op::Tzcnt : Op::Lzcnt;
    // 0F 38 F0/F1 is movbe without F2 (crc32 with it); memory operand only.
    if (in.map == 2 && (in.opcode == 0xF0 || in.opcode == 0xF1) && !in.repne && !in.rep && in.mod != 3)
        return Op::Movbe;
    return Op::None;
}

namespace {

constexpr uint64_t CF = 1 << 0, PF = 1 << 2, AF = 1 << 4, ZF = 1 << 6, SF = 1 << 7, OF = 1 << 11;

uint64_t mask(int bits) { return bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1; }

uint64_t effectiveAddress(const Insn& in, const State& s) {
    uint64_t a;
    if (in.ripRelative) a = s.rip + in.len + uint64_t(int64_t(in.disp));
    else {
        a = uint64_t(int64_t(in.disp));
        if (in.hasSib) {
            if (!in.noBase) a += s.gpr[in.base];
            if (in.index != 4) a += s.gpr[in.index] << in.scale;
        } else
            a += s.gpr[in.rm];
    }
    return in.addr32 ? uint32_t(a) : a;
}

uint64_t loadInt(uint64_t addr, int bits) {
    uint64_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), size_t(bits / 8));
    return v;
}

void storeInt(uint64_t addr, uint64_t v, int bits) { std::memcpy(reinterpret_cast<void*>(addr), &v, size_t(bits / 8)); }

// r/m as an integer operand of the given width.
uint64_t readRm(const Insn& in, const State& s, int bits) {
    return in.mod == 3 ? s.gpr[in.rm] & mask(bits) : loadInt(effectiveAddress(in, s), bits);
}

// Writes a general register as x86 does: 32-bit results clear the upper half, 16-bit ones keep it.
void writeGpr(State& s, int reg, uint64_t v, int bits) {
    if (bits == 16) s.gpr[reg] = (s.gpr[reg] & ~uint64_t(0xFFFF)) | (v & 0xFFFF);
    else s.gpr[reg] = v & mask(bits);
}

void setFlags(State& s, uint64_t set, uint64_t affected) { s.rflags = (s.rflags & ~affected) | (set & affected); }

uint64_t zeroSign(uint64_t r, int bits) { return (r == 0 ? ZF : 0) | ((r >> (bits - 1)) & 1 ? SF : 0); }

uint64_t bswap(uint64_t v, int bits) {
    switch (bits) {
    case 16: return __builtin_bswap16(uint16_t(v));
    case 32: return __builtin_bswap32(uint32_t(v));
    default: return __builtin_bswap64(v);
    }
}

uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

float bitsFloat(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

} // namespace

float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    if (exp == 31) return bitsFloat(sign | 0x7F800000 | (man ? 0x400000 | (man << 13) : 0));
    if (exp != 0) return bitsFloat(sign | ((exp + 112) << 23) | (man << 13));
    if (man == 0) return bitsFloat(sign);
    int e = -14; // denormal: man * 2^-24, normalised into a float
    uint32_t m = man;
    while (!(m & 0x400)) {
        m <<= 1;
        --e;
    }
    return bitsFloat(sign | uint32_t(e + 127) << 23 | ((m & 0x3FF) << 13));
}

uint16_t floatToHalf(float f, int rc) {
    const uint32_t u = floatBits(f);
    const uint16_t sign = uint16_t((u >> 16) & 0x8000);
    const int exp = int((u >> 23) & 0xFF);
    const uint32_t man = u & 0x7FFFFF;
    if (exp == 0xFF) return uint16_t(sign | 0x7C00 | (man ? 0x200 | (man >> 13) : 0));
    if (exp == 0 && man == 0) return sign;
    // value = mant * 2^e
    const uint64_t mant = exp ? (man | 0x800000) : man;
    const int e = exp ? exp - 150 : -149;
    const int top = 63 - __builtin_clzll(mant); // value's exponent is e + top
    const int t = e + top;
    const int quantum = t >= -14 ? t - 10 : -24; // exponent of the half's last bit
    int shift = quantum - e;
    uint64_t q;
    bool guard = false, sticky = false;
    if (shift <= 0) q = mant << -shift;
    else {
        if (shift > 40) shift = 40;
        q = mant >> shift;
        guard = (mant >> (shift - 1)) & 1;
        sticky = (mant & ((uint64_t(1) << (shift - 1)) - 1)) != 0;
    }
    const bool inexact = guard || sticky;
    bool up = false;
    switch (rc & 3) {
    case 0: up = guard && (sticky || (q & 1)); break;
    case 1: up = inexact && sign; break;
    case 2: up = inexact && !sign; break;
    default: break;
    }
    q += up;
    int hexp;
    if (t >= -14) {
        int tt = t;
        if (q >> 11) { // rounded up to the next power of two
            q >>= 1;
            ++tt;
        }
        hexp = tt + 15;
        if (hexp >= 31) {
            // Overflow: infinity, or the largest finite half when rounding toward zero from it.
            const bool toInf = (rc & 3) == 0 || ((rc & 3) == 1 && sign) || ((rc & 3) == 2 && !sign);
            return uint16_t(sign | (toInf ? 0x7C00 : 0x7BFF));
        }
        return uint16_t(sign | (hexp << 10) | (q & 0x3FF));
    }
    return uint16_t(sign | q); // denormal, or 0x400 when it rounded up to the smallest normal
}

Op emulate(const Insn& in, State& s) {
    const Op op = classify(in);
    if (op == Op::None) return op;
    if (in.memoryOperand() && in.segment) return Op::None;

    switch (op) {
    case Op::Andn: case Op::Bextr: case Op::Blsi: case Op::Blsmsk: case Op::Blsr: {
        const int bits = in.vexW ? 64 : 32;
        const uint64_t src = readRm(in, s, bits);
        const uint64_t v = s.gpr[in.vexV] & mask(bits);
        uint64_t r, f;
        int dest = in.reg;
        if (op == Op::Andn) {
            r = ~v & src & mask(bits);
            f = zeroSign(r, bits);
        } else if (op == Op::Bextr) {
            const unsigned start = v & 0xFF, len = (v >> 8) & 0xFF;
            r = start < unsigned(bits) ? src >> start : 0;
            if (len < unsigned(bits)) r &= mask(int(len));
            f = r == 0 ? ZF : 0;
        } else {
            dest = in.vexV;
            if (op == Op::Blsi) r = (0 - src) & src;
            else if (op == Op::Blsmsk) r = src ^ (src - 1);
            else r = src & (src - 1);
            r &= mask(bits);
            f = zeroSign(r, bits);
            if (op == Op::Blsi ? src != 0 : src == 0) f |= CF;
        }
        writeGpr(s, dest, r, bits);
        setFlags(s, f, CF | ZF | SF | OF | (op == Op::Bextr ? AF | PF : 0));
        break;
    }
    case Op::Tzcnt: case Op::Lzcnt: {
        const int bits = in.w ? 64 : in.opsize ? 16 : 32;
        const uint64_t src = readRm(in, s, bits);
        uint64_t r;
        if (src == 0) r = uint64_t(bits);
        else if (op == Op::Tzcnt) r = uint64_t(__builtin_ctzll(src));
        else r = uint64_t(__builtin_clzll(src) - (64 - bits));
        writeGpr(s, in.reg, r, bits);
        setFlags(s, (src == 0 ? CF : 0) | (r == 0 ? ZF : 0), CF | ZF);
        break;
    }
    case Op::Movbe: {
        const int bits = in.w ? 64 : in.opsize ? 16 : 32;
        const uint64_t addr = effectiveAddress(in, s);
        if (in.opcode == 0xF0) writeGpr(s, in.reg, bswap(loadInt(addr, bits), bits), bits);
        else storeInt(addr, bswap(s.gpr[in.reg] & mask(bits), bits), bits);
        break;
    }
    case Op::Cvtph2ps: {
        if (in.vexL && !s.ymmHigh) return Op::None;
        const int n = in.vexL ? 8 : 4;
        uint16_t h[8];
        if (in.mod == 3) std::memcpy(h, s.xmm[in.rm], size_t(n) * 2);
        else std::memcpy(h, reinterpret_cast<const void*>(effectiveAddress(in, s)), size_t(n) * 2);
        float f[8];
        for (int k = 0; k < n; ++k) f[k] = halfToFloat(h[k]);
        std::memcpy(s.xmm[in.reg], f, 16);
        if (s.ymmHigh) {
            if (in.vexL) std::memcpy(s.ymmHigh[in.reg], f + 4, 16);
            else std::memset(s.ymmHigh[in.reg], 0, 16);
            s.ymmHighWritten = true;
        }
        break;
    }
    case Op::Cvtps2ph: {
        if (in.vexL && !s.ymmHigh) return Op::None;
        const int n = in.vexL ? 8 : 4;
        const int imm = int(in.imm);
        const int rc = (imm & 4) ? int((s.mxcsr >> 13) & 3) : imm & 3;
        float f[8];
        std::memcpy(f, s.xmm[in.reg], 16);
        if (in.vexL) std::memcpy(f + 4, s.ymmHigh[in.reg], 16);
        uint16_t h[8] = {};
        for (int k = 0; k < n; ++k) h[k] = floatToHalf(f[k], rc);
        if (in.mod == 3) {
            std::memcpy(s.xmm[in.rm], h, 16); // the halves, then zeros
            if (s.ymmHigh) {
                std::memset(s.ymmHigh[in.rm], 0, 16);
                s.ymmHighWritten = true;
            }
        } else
            std::memcpy(reinterpret_cast<void*>(effectiveAddress(in, s)), h, size_t(n) * 2);
        break;
    }
    default:
        return Op::None;
    }
    s.rip += in.len;
    return op;
}

} // namespace x86
