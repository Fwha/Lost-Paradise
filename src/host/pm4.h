// GPU command buffers (PM4, as on AMD GCN).
//
// For now only the packets that touch memory or signal the CPU are executed: fences
// (WRITE_DATA, EVENT_WRITE_EOP/EOS, RELEASE_MEM) and their interrupts. Drawing and
// state packets are skipped until the renderer lands.
#pragma once

#include <cstdint>

namespace host {

// Event id the GNM end-of-pipe interrupt is delivered under (sceGnmAddEqEvent).
constexpr uint64_t kEqEventGfxEop = 0x40;

struct Pm4Stats {
    uint64_t packets = 0, fences = 0, interrupts = 0, skippedWaits = 0;
};

void executePm4(const uint32_t* cmds, uint32_t numDwords, Pm4Stats* stats);

// A PM4 NOP occupying exactly numDwords (used where GNM would emit real packets).
void writePm4Nop(uint32_t* cmds, uint32_t numDwords);

} // namespace host
