// GPU command processor (PM4, as on AMD GCN).
//
// Executes what the CPU can observe today: register writes into a register file, fences
// (WRITE_DATA, EVENT_WRITE_EOP/EOS, RELEASE_MEM), CP DMA and interrupts. Draws and
// dispatches are handed to a listener with the register state at that moment; the
// renderer will be one such listener.
#pragma once

#include "host/gpu/registers.h"

#include <cstdint>
#include <functional>

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
    // Runs `fn` once the GPU work handed over so far has finished, for fences and interrupts
    // that must not fire before it. False when the listener runs work synchronously (the
    // caller then runs `fn` itself).
    virtual bool afterGpuWork(std::function<void()>&) { return false; }
    // The command processor wrote guest memory (DMA_DATA, WRITE_DATA) between draws.
    virtual void onMemoryWrite(uint64_t /*addr*/, uint64_t /*size*/) {}
    // The command processor is about to wait (WAIT_REG_MEM): hand the GPU what was recorded,
    // since the value waited for may come from its fences.
    virtual void onWait() {}
    // The analysis thread's view: the same commands, in the same order, are executed there first
    // with a register file of its own, so a listener can prepare a draw (analyse its shaders)
    // ahead of onDraw. Nothing in these may touch the GPU. Without the command processor thread,
    // each is called just before its counterpart.
    virtual void onAnalyzeDraw(const DrawCall&, const GpuRegs&) {}
    virtual void onAnalyzeDma(uint64_t /*dst*/, uint64_t /*src*/, uint64_t /*bytes*/, bool /*fill*/, uint32_t /*value*/) {}
    virtual void onAnalyzeWrite(uint64_t /*addr*/, const uint32_t* /*values*/, uint32_t /*count*/) {}
    virtual void onAnalyzeSubmitEnd() {}
    virtual void onAnalyzeFrameEnd() {}
    // The analysis of the draws so far must not wait for more: their recording is about to be queued.
    virtual void onAnalyzeFlush() {}
    // WRITE_DATA wrote memory (after onMemoryWrite): the counterpart of onAnalyzeWrite.
    virtual void onCpWrite(uint64_t /*addr*/, uint64_t /*size*/) {}
    // CP DMA from memory (or a fill) to memory: true when the listener runs it on the GPU timeline,
    // in order with the draws around it; otherwise the command processor copies it now.
    virtual bool onDmaCopy(uint64_t /*dst*/, uint64_t /*src*/, uint64_t /*bytes*/, bool /*fill*/, uint32_t /*value*/) {
        return false;
    }
};
void addDrawListener(DrawListener* l);
void notifySubmitEnd();
void notifyFrameEnd();

void executePm4(const uint32_t* cmds, uint32_t numDwords, Pm4Stats* stats);

// The command processor runs on a thread of its own, as on the console: submissions return at
// once and their packets are executed there, in order. Queues `fn` to run on that thread after
// everything queued before it. LPN_SYNC_SUBMIT=1 runs it on the caller's thread instead.
void queueCommandProcessor(std::function<void()> fn);
// From the command processor thread: runs `fn` once the commands handed over so far have been
// recorded (draws go to the listeners on a recorder thread of their own).
void afterCommandsRecorded(std::function<void()> fn);

// LPN_TIMELINE=<seconds>: logs what happens on the GPU side for 100 ms from then, with times.
bool timelineOn();
// Turns the timeline on for `seconds` from now (F10 with LPN_PROFILE=1).
void startTimeline(double seconds);
void timeline(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// A PM4 NOP occupying exactly numDwords (used where GNM would emit real packets).
void writePm4Nop(uint32_t* cmds, uint32_t numDwords);

} // namespace host
