// The sceGnm* packet helpers: each writes PM4 into the caller's buffer as (cmd, numDwords, ...)
// and pads whatever it doesn't use with a NOP so the buffer keeps the exact size the game
// reserved. Formats follow what the PS4 driver emits for the same calls.
#include "host/gpu/registers.h"
#include "host/hle.h"
#include "host/kernel.h"
#include "host/pm4.h"

#include <cstdio>
#include <initializer_list>

namespace host {

namespace {

using namespace gpu;

class Writer {
public:
    Writer(uint32_t* cmd, uint32_t size) : p_(cmd), end_(cmd ? cmd + size : nullptr) {}

    bool packet(uint32_t op, std::initializer_list<uint32_t> body, bool compute = false) {
        const uint32_t n = uint32_t(body.size());
        if (!p_ || p_ + n + 1 > end_) return ok_ = false;
        *p_++ = (3u << 30) | ((n - 1) << 16) | (op << 8) | (compute ? 2u : 0u);
        for (uint32_t v : body) *p_++ = v;
        return true;
    }
    template <typename... V> void ctx(uint32_t reg, V... v) { packet(kOpSetContextReg, {reg, uint32_t(v)...}); }
    template <typename... V> void sh(uint32_t reg, V... v) { packet(kOpSetShReg, {reg, uint32_t(v)...}); }
    template <typename... V> void shCompute(uint32_t reg, V... v) { packet(kOpSetShReg, {reg, uint32_t(v)...}, true); }

    // Pad the rest of the reserved space. Returns 0 (s_ok) like the driver, or -1 on overflow.
    int32_t finish(const char* fn) {
        if (!ok_) {
            std::fprintf(stderr, "[gnm] %s: reserved space too small\n", fn);
            return -1;
        }
        if (p_ && end_ > p_) writePm4Nop(p_, uint32_t(end_ - p_));
        return 0;
    }

private:
    uint32_t* p_;
    uint32_t* end_;
    bool ok_ = true;
};

// Draw initiator: SOURCE_SELECT (bits 1:0) = 0 DMA (indexed), 2 auto-index; USE_OPAQUE bit 3.
constexpr uint32_t kSourceDma = 0, kSourceAuto = 2, kUseOpaque = 0x8;

int32_t sceGnmDrawIndex(uint32_t* cmd, uint32_t size, uint32_t count, uint64_t indexAddr, uint32_t, uint32_t) {
    Writer w(cmd, size);
    w.packet(kOpDrawIndex2, {0x7FFFFFFF, uint32_t(indexAddr), uint32_t(indexAddr >> 32), count, kSourceDma});
    return w.finish(__func__);
}
int32_t sceGnmDrawIndexAuto(uint32_t* cmd, uint32_t size, uint32_t count, uint32_t) {
    Writer w(cmd, size);
    w.packet(kOpDrawIndexAuto, {count, kSourceAuto});
    return w.finish(__func__);
}
int32_t sceGnmDrawIndexOffset(uint32_t* cmd, uint32_t size, uint32_t offset, uint32_t count, uint32_t) {
    Writer w(cmd, size);
    w.packet(kOpDrawIndexOffset2, {0x7FFFFFFF, offset, count, kSourceDma});
    return w.finish(__func__);
}
int32_t sceGnmDrawOpaqueAuto(uint32_t* cmd, uint32_t size, uint32_t) {
    Writer w(cmd, size);
    w.packet(kOpDrawIndexAuto, {0, kSourceAuto | kUseOpaque});
    return w.finish(__func__);
}
// Indirect draws read their arguments from a buffer bound with SET_BASE; the user-data
// slots receive the base vertex / instance. Body: offset, base-vertex reg, instance reg, initiator.
int32_t sceGnmDrawIndexIndirect(uint32_t* cmd, uint32_t size, uint32_t dataOffset, uint32_t stage,
                                uint32_t vertexSgpr, uint32_t instanceSgpr, uint32_t) {
    (void)stage; // the game only draws with a plain VS pipeline (no LS/ES), whose user data is VS
    const uint32_t userData = sh::kSpiShaderUserDataVs;
    Writer w(cmd, size);
    w.packet(kOpDrawIndexIndirect, {dataOffset, userData + vertexSgpr, userData + instanceSgpr, kSourceDma});
    return w.finish(__func__);
}
int32_t sceGnmDrawIndirect(uint32_t* cmd, uint32_t size, uint32_t dataOffset, uint32_t stage, uint32_t vertexSgpr,
                           uint32_t instanceSgpr, uint32_t) {
    (void)stage; // the game only draws with a plain VS pipeline (no LS/ES), whose user data is VS
    const uint32_t userData = sh::kSpiShaderUserDataVs;
    Writer w(cmd, size);
    w.packet(kOpDrawIndirect, {dataOffset, userData + vertexSgpr, userData + instanceSgpr, kSourceAuto});
    return w.finish(__func__);
}

int32_t sceGnmDispatchDirect(uint32_t* cmd, uint32_t size, uint32_t x, uint32_t y, uint32_t z, uint32_t mode) {
    Writer w(cmd, size);
    w.packet(kOpDispatchDirect, {x, y, z, mode | 1 /*COMPUTE_SHADER_EN*/}, true);
    return w.finish(__func__);
}
int32_t sceGnmDispatchIndirect(uint32_t* cmd, uint32_t size, uint32_t dataOffset, uint32_t mode) {
    Writer w(cmd, size);
    w.packet(kOpDispatchIndirect, {dataOffset, mode | 1}, true);
    return w.finish(__func__);
}

// --- shader stages ----------------------------------------------------------------------

struct VsStageRegisters {
    uint32_t pgmLo, pgmHi, rsrc1, rsrc2, vsOutConfig, posFormat, clVsOutCntl;
};
struct PsStageRegisters {
    uint32_t pgmLo, pgmHi, rsrc1, rsrc2, zFormat, colFormat, inputEna, inputAddr, inControl, barycCntl,
        dbShaderControl, cbShaderMask;
};
struct CsStageRegisters {
    uint32_t pgmLo, pgmHi, rsrc1, rsrc2, numThreadX, numThreadY, numThreadZ;
};

// A fetch-shader modifier replaces the VGPR/SGPR counts in RSRC1.
uint32_t applyModifier(uint32_t rsrc1, uint32_t modifier) {
    return modifier ? (rsrc1 & 0xFCFFFC00u) | modifier : rsrc1;
}

int32_t sceGnmSetVsShader(uint32_t* cmd, uint32_t size, const VsStageRegisters* vs, uint32_t modifier) {
    Writer w(cmd, size);
    if (vs) {
        w.sh(sh::kSpiShaderPgmLoVs, vs->pgmLo, vs->pgmHi);
        w.sh(sh::kSpiShaderPgmRsrc1Vs, applyModifier(vs->rsrc1, modifier), vs->rsrc2);
        w.ctx(ctx::kSpiVsOutConfig, vs->vsOutConfig);
        w.ctx(ctx::kSpiShaderPosFormat, vs->posFormat);
        w.ctx(ctx::kPaClVsOutCntl, vs->clVsOutCntl);
    }
    return w.finish(__func__);
}
int32_t sceGnmUpdateVsShader(uint32_t* cmd, uint32_t size, const VsStageRegisters* vs, uint32_t modifier) {
    return sceGnmSetVsShader(cmd, size, vs, modifier);
}

int32_t sceGnmSetPsShader350(uint32_t* cmd, uint32_t size, const PsStageRegisters* ps) {
    Writer w(cmd, size);
    if (!ps) { // pixel shader disabled
        w.sh(sh::kSpiShaderPgmLoPs, 0u, 0u);
        w.ctx(ctx::kSpiShaderZFormat, 0u, 0u);
        w.ctx(ctx::kCbShaderMask, 0u);
    } else {
        w.sh(sh::kSpiShaderPgmLoPs, ps->pgmLo, ps->pgmHi);
        w.sh(sh::kSpiShaderPgmRsrc1Ps, ps->rsrc1, ps->rsrc2);
        w.ctx(ctx::kSpiShaderZFormat, ps->zFormat, ps->colFormat);
        w.ctx(ctx::kSpiPsInputEna, ps->inputEna, ps->inputAddr);
        w.ctx(ctx::kSpiPsInControl, ps->inControl);
        w.ctx(ctx::kSpiBarycCntl, ps->barycCntl);
        w.ctx(ctx::kDbShaderControl, ps->dbShaderControl);
        w.ctx(ctx::kCbShaderMask, ps->cbShaderMask);
    }
    return w.finish(__func__);
}
int32_t sceGnmUpdatePsShader350(uint32_t* cmd, uint32_t size, const PsStageRegisters* ps) {
    return sceGnmSetPsShader350(cmd, size, ps);
}

int32_t sceGnmSetCsShaderWithModifier(uint32_t* cmd, uint32_t size, const CsStageRegisters* cs, uint32_t modifier) {
    Writer w(cmd, size);
    if (cs) {
        w.shCompute(sh::kComputePgmLo, cs->pgmLo, cs->pgmHi);
        w.shCompute(sh::kComputePgmRsrc1, applyModifier(cs->rsrc1, modifier), cs->rsrc2);
        w.shCompute(sh::kComputeNumThreadX, cs->numThreadX, cs->numThreadY, cs->numThreadZ);
    }
    return w.finish(__func__);
}

int32_t sceGnmSetVgtControl(uint32_t* cmd, uint32_t size, uint32_t primGroupSizeMinusOne, uint32_t partialVsWave,
                            uint32_t wdSwitchOnlyOnEop) {
    Writer w(cmd, size);
    w.ctx(ctx::kIaMultiVgtParam,
          (primGroupSizeMinusOne & 0xFFFF) | ((partialVsWave & 1) << 16) | ((wdSwitchOnlyOnEop & 1) << 20));
    return w.finish(__func__);
}
int32_t sceGnmResetVgtControl(uint32_t* cmd, uint32_t size) {
    Writer w(cmd, size);
    w.ctx(ctx::kIaMultiVgtParam, 0xFFu);
    return w.finish(__func__);
}

} // namespace

void registerGnmPacketHle() {
    const char* g = "libSceGnmDriver";
    HLE(g, sceGnmDrawIndex);
    HLE(g, sceGnmDrawIndexAuto);
    HLE(g, sceGnmDrawIndexOffset);
    HLE(g, sceGnmDrawOpaqueAuto);
    HLE(g, sceGnmDrawIndexIndirect);
    HLE(g, sceGnmDrawIndirect);
    HLE(g, sceGnmDispatchDirect);
    HLE(g, sceGnmDispatchIndirect);
    HLE(g, sceGnmSetVsShader);
    HLE(g, sceGnmUpdateVsShader);
    HLE(g, sceGnmSetPsShader350);
    HLE(g, sceGnmUpdatePsShader350);
    HLE(g, sceGnmSetCsShaderWithModifier);
    HLE(g, sceGnmSetVgtControl);
    HLE(g, sceGnmResetVgtControl);
}

} // namespace host
