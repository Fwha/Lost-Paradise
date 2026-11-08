// GCN (Liverpool) register offsets used by the command processor and the GNM helpers.
// Offsets are relative to each SET_*_REG window, as they appear in PM4 packets.
#pragma once

#include <cstdint>

namespace gpu {

// PM4 type-3 opcodes.
enum Op : uint32_t {
    kOpNop = 0x10,
    kOpSetBase = 0x11,
    kOpIndexBufferSize = 0x13,
    kOpDispatchDirect = 0x15,
    kOpDispatchIndirect = 0x16,
    kOpDrawIndirect = 0x24,
    kOpDrawIndexIndirect = 0x25,
    kOpIndexBase = 0x26,
    kOpDrawIndex2 = 0x27,
    kOpIndexType = 0x2A,
    kOpDrawIndexAuto = 0x2D,
    kOpNumInstances = 0x2F,
    kOpIndirectBufferConst = 0x33,
    kOpDrawIndexOffset2 = 0x35,
    kOpWriteData = 0x37,
    kOpWaitRegMem = 0x3C,
    kOpIndirectBuffer = 0x3F,
    kOpEventWrite = 0x46,
    kOpEventWriteEop = 0x47,
    kOpEventWriteEos = 0x48,
    kOpReleaseMem = 0x49,
    kOpDmaData = 0x50,
    kOpAcquireMem = 0x58,
    kOpSetContextReg = 0x69,
    kOpSetShReg = 0x76,
    kOpSetUconfigReg = 0x79,
};

// Register file sizes per window.
constexpr uint32_t kContextRegs = 0x400; // 0xA000..
constexpr uint32_t kShRegs = 0x400;      // 0x2C00..
constexpr uint32_t kUconfigRegs = 0x4000; // 0xC000..

namespace sh {
constexpr uint32_t kSpiShaderPgmLoPs = 0x08, kSpiShaderPgmRsrc1Ps = 0x0A, kSpiShaderUserDataPs = 0x0C;
constexpr uint32_t kSpiShaderPgmLoVs = 0x48, kSpiShaderPgmRsrc1Vs = 0x4A, kSpiShaderUserDataVs = 0x4C;
constexpr uint32_t kSpiShaderPgmLoGs = 0x88, kSpiShaderPgmLoEs = 0xC8, kSpiShaderPgmLoHs = 0x108,
                   kSpiShaderPgmLoLs = 0x148;
constexpr uint32_t kSpiShaderUserDataGs = 0x8C, kSpiShaderUserDataEs = 0xCC;
constexpr uint32_t kComputeNumThreadX = 0x207, kComputePgmLo = 0x20C, kComputePgmRsrc1 = 0x212,
                   kComputeUserData = 0x240;
} // namespace sh

namespace ctx {
constexpr uint32_t kDbZInfo = 0x010, kDbZReadBase = 0x012, kDbZWriteBase = 0x014, kDbDepthSize = 0x016;
constexpr uint32_t kCbShaderMask = 0x08F;
constexpr uint32_t kPaScScreenScissorTl = 0x00C;
constexpr uint32_t kPaClVportXscale = 0x10F; // 6 regs per viewport
constexpr uint32_t kSpiPsInputCntl0 = 0x191;
constexpr uint32_t kSpiVsOutConfig = 0x1B1, kSpiPsInputEna = 0x1B3, kSpiPsInputAddr = 0x1B4,
                   kSpiPsInControl = 0x1B6, kSpiBarycCntl = 0x1B8;
constexpr uint32_t kSpiShaderPosFormat = 0x1C3, kSpiShaderZFormat = 0x1C4, kSpiShaderColFormat = 0x1C5;
constexpr uint32_t kCbBlend0Control = 0x1E0;
constexpr uint32_t kDbDepthControl = 0x200, kCbColorControl = 0x202, kDbShaderControl = 0x203;
constexpr uint32_t kPaSuScModeCntl = 0x205, kPaClVsOutCntl = 0x207;
constexpr uint32_t kVgtShaderStagesEn = 0x2D5;
// Geometry shader pipeline.
constexpr uint32_t kVgtGsMode = 0x290, kVgtGsvsRingOffset1 = 0x298, kVgtGsOutPrimType = 0x29B,
                   kVgtEsgsRingItemsize = 0x2AB, kVgtGsvsRingItemsize = 0x2AC, kVgtGsMaxVertOut = 0x2CE,
                   kVgtGsVertItemsize = 0x2D7, kVgtGsInstanceCnt = 0x2E4, kVgtStrmoutConfig = 0x2E5;
constexpr uint32_t kIaMultiVgtParam = 0x2AA;
constexpr uint32_t kCbColor0Base = 0x318; // 15 regs per render target, 8 targets
constexpr uint32_t kCbColorStride = 0xF;
constexpr uint32_t kCbColorPitch = 1, kCbColorSlice = 2, kCbColorView = 3, kCbColorInfo = 4, kCbColorAttrib = 5;
} // namespace ctx

namespace uconfig {
constexpr uint32_t kVgtPrimitiveType = 0x242, kVgtIndexType = 0x243, kVgtNumIndices = 0x24C,
                   kVgtNumInstances = 0x24D;
} // namespace uconfig

} // namespace gpu
