// GPU command processor (PM4, as on AMD GCN).
//
// Executes what the CPU can observe today: register writes into a register file, fences
// (WRITE_DATA, EVENT_WRITE_EOP/EOS, RELEASE_MEM), CP DMA and interrupts. Draws and
// dispatches are handed to a listener with the register state at that moment; the
// renderer will be one such listener.
#pragma once

#include "host/gpu/registers.h"

#include <cstdint>

namespace host {

// Event id the GNM end-of-pipe interrupt is delivered under (sceGnmAddEqEvent).
constexpr uint64_t kEqEventGfxEop = 0x40;

struct Pm4Stats {
    uint64_t packets = 0, fences = 0, interrupts = 0, skippedWaits = 0, draws = 0, dispatches = 0;
    uint64_t opcodes[256] = {}; // type-3 packets by opcode
};

struct GpuRegs {
    uint32_t ctx[gpu::kContextRegs] = {};
    uint32_t sh[gpu::kShRegs] = {};
    uint32_t uconfig[gpu::kUconfigRegs] = {};
    uint64_t indexBase = 0;
    uint32_t indexBufferSize = 0; // in indices
    uint32_t indexType = 0;       // 0: 16-bit, 1: 32-bit
    uint32_t numInstances = 1;
    uint64_t indirectBase = 0;    // SET_BASE for draw/dispatch indirect arguments
};

struct DrawCall {
    enum Kind { kIndexed, kAuto, kIndexedIndirect, kIndirect, kDispatch, kDispatchIndirect } kind;
    uint32_t count = 0;              // indices or vertices
    uint32_t indexOffset = 0;        // first index for DRAW_INDEX_OFFSET_2
    uint64_t indexAddr = 0;          // index buffer for kIndexed
    uint32_t groups[3] = {};         // thread groups for kDispatch
    uint32_t indirectOffset = 0;     // into GpuRegs::indirectBase
};

class DrawListener {
public:
    virtual ~DrawListener() = default;
    virtual void onDraw(const DrawCall& draw, const GpuRegs& regs) = 0;
    virtual void onSubmitEnd() {} // a submission's command buffers have all been processed
    virtual void onFrameEnd() {}
};
void addDrawListener(DrawListener* l);
void notifySubmitEnd();
void notifyFrameEnd();

void executePm4(const uint32_t* cmds, uint32_t numDwords, Pm4Stats* stats);

// A PM4 NOP occupying exactly numDwords (used where GNM would emit real packets).
void writePm4Nop(uint32_t* cmds, uint32_t numDwords);

} // namespace host
