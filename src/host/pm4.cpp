#include "host/pm4.h"

#include "host/equeue.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace host {

namespace {

enum Opcode : uint32_t {
    kNop = 0x10,
    kIndirectBufferConst = 0x33,
    kWriteData = 0x37,
    kWaitRegMem = 0x3C,
    kIndirectBuffer = 0x3F,
    kEventWriteEop = 0x47,
    kEventWriteEos = 0x48,
    kReleaseMem = 0x49,
};

uint64_t gpuClock() {
    // The GPU counter runs at 100 MHz on the PS4.
    using namespace std::chrono;
    return uint64_t(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 10);
}

uint64_t address(uint32_t lo, uint32_t hi) { return uint64_t(lo) | (uint64_t(hi & 0xFFFF) << 32); }

// Shared tail of EVENT_WRITE_EOP and RELEASE_MEM.
void endOfPipe(uint64_t addr, uint32_t dataSel, uint32_t intSel, uint32_t dataLo, uint32_t dataHi, Pm4Stats* st) {
    if (addr) switch (dataSel) {
        case 1: *reinterpret_cast<volatile uint32_t*>(addr) = dataLo; break;
        case 2: *reinterpret_cast<volatile uint64_t*>(addr) = uint64_t(dataLo) | (uint64_t(dataHi) << 32); break;
        case 3:
        case 4: *reinterpret_cast<volatile uint64_t*>(addr) = gpuClock(); break;
        default: break;
        }
    if (dataSel) ++st->fences;
    if (intSel) {
        ++st->interrupts;
        triggerEvent(kEqEventGfxEop, kFilterGraphicsCore, 0);
    }
}

void execute(const uint32_t* p, const uint32_t* end, Pm4Stats* st, int depth) {
    while (p < end) {
        const uint32_t h = *p;
        const uint32_t type = h >> 30;
        if (type == 2) { // filler
            ++p;
            continue;
        }
        const uint32_t count = (h >> 16) & 0x3FFF;
        const uint32_t len = count + 2;
        if (type == 1 || p + len > end) {
            std::fprintf(stderr, "[pm4] malformed packet 0x%08x, skipping rest of buffer\n", h);
            return;
        }
        ++st->packets;
        if (type == 3) {
            const uint32_t op = (h >> 8) & 0xFF;
            const uint32_t* b = p + 1;
            switch (op) {
            case kIndirectBuffer:
            case kIndirectBufferConst:
                if (depth < 4) {
                    const auto* ib = reinterpret_cast<const uint32_t*>(address(b[0] & ~3u, b[1]));
                    execute(ib, ib + (b[2] & 0xFFFFF), st, depth + 1);
                }
                break;
            case kWriteData: {
                const uint32_t dstSel = (b[0] >> 8) & 0xF;
                if (dstSel == 2 || dstSel == 5) {
                    auto* dst = reinterpret_cast<volatile uint32_t*>(address(b[1], b[2]));
                    for (uint32_t i = 3; i < count + 1; ++i) dst[i - 3] = b[i];
                    ++st->fences;
                }
                break;
            }
            case kEventWriteEop:
                endOfPipe(address(b[1], b[2]), b[2] >> 29, (b[2] >> 24) & 3, b[3], b[4], st);
                break;
            case kReleaseMem:
                endOfPipe(address(b[2], b[3]), b[1] >> 29, (b[1] >> 24) & 3, b[4], b[5], st);
                break;
            case kEventWriteEos:
                if ((b[2] >> 29) == 2) {
                    *reinterpret_cast<volatile uint32_t*>(address(b[1], b[2])) = b[3];
                    ++st->fences;
                }
                break;
            case kWaitRegMem:
                // Commands run synchronously at submit, so whatever this waits for has either
                // happened already or comes from the CPU later; don't block on it.
                ++st->skippedWaits;
                break;
            default: break;
            }
        }
        p += len;
    }
}

} // namespace

void executePm4(const uint32_t* cmds, uint32_t numDwords, Pm4Stats* stats) {
    Pm4Stats local;
    execute(cmds, cmds + numDwords, stats ? stats : &local, 0);
}

void writePm4Nop(uint32_t* cmds, uint32_t numDwords) {
    if (!cmds || numDwords == 0) return;
    if (numDwords == 1) {
        cmds[0] = 0x80000000u; // type-2 filler
        return;
    }
    cmds[0] = (3u << 30) | ((numDwords - 2) << 16) | (kNop << 8);
    std::memset(cmds + 1, 0, (numDwords - 1) * 4);
}

} // namespace host
