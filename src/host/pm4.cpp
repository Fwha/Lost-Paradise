#include "host/pm4.h"

#include "host/equeue.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace host {

namespace {

using namespace gpu;

GpuRegs gRegs; // the graphics ring's state; submissions are serialised by the caller
DrawListener* gListener = nullptr;

void draw(const DrawCall& d, Pm4Stats* st) {
    (d.kind >= DrawCall::kDispatch ? st->dispatches : st->draws)++;
    if (gListener) gListener->onDraw(d, gRegs);
}

// CP DMA: copies or fills memory on the GPU timeline, which is now.
void dmaData(const uint32_t* b) {
    const uint32_t control = b[0], bytes = b[5] & 0x1FFFFF;
    const uint32_t srcSel = (control >> 29) & 3, dstSel = (control >> 20) & 3;
    const uint64_t src = uint64_t(b[1]) | (uint64_t(b[2]) << 32), dst = uint64_t(b[3]) | (uint64_t(b[4]) << 32);
    if (dstSel != 0 || !dst || !bytes) return; // only memory destinations matter to the CPU
    if (srcSel == 0 && src) std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
    else if (srcSel == 2) { // fill with the immediate in src_lo
        auto* d = reinterpret_cast<uint32_t*>(dst);
        for (uint32_t i = 0; i < bytes / 4; ++i) d[i] = b[1];
    }
}

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
            ++st->opcodes[op];
            const uint32_t* b = p + 1;
            switch (op) {
            case kOpSetContextReg:
            case kOpSetShReg:
            case kOpSetUconfigReg: {
                // body: register offset (low 16 bits; the top holds an index field), then values
                const uint32_t reg = b[0] & 0xFFFF;
                uint32_t* file = op == kOpSetContextReg ? gRegs.ctx : op == kOpSetShReg ? gRegs.sh : gRegs.uconfig;
                const uint32_t size = op == kOpSetContextReg ? kContextRegs : op == kOpSetShReg ? kShRegs : kUconfigRegs;
                if (reg + count <= size) std::memcpy(&file[reg], b + 1, count * 4);
                break;
            }
            case kOpIndexBase: gRegs.indexBase = address(b[0], b[1]); break;
            case kOpIndexBufferSize: gRegs.indexBufferSize = b[0]; break;
            case kOpIndexType: gRegs.indexType = b[0] & 3; break;
            case kOpNumInstances: gRegs.numInstances = b[0] ? b[0] : 1; break;
            case kOpSetBase: gRegs.indirectBase = address(b[1], b[2]); break;
            case kOpDrawIndex2: {
                DrawCall d{DrawCall::kIndexed};
                d.indexAddr = address(b[1], b[2]);
                d.count = b[3];
                draw(d, st);
                break;
            }
            case kOpDrawIndexOffset2: {
                DrawCall d{DrawCall::kIndexed};
                d.indexAddr = gRegs.indexBase;
                d.indexOffset = b[1];
                d.count = b[2];
                draw(d, st);
                break;
            }
            case kOpDrawIndexAuto: {
                DrawCall d{DrawCall::kAuto};
                d.count = b[0];
                draw(d, st);
                break;
            }
            case kOpDrawIndexIndirect:
            case kOpDrawIndirect: {
                DrawCall d{op == kOpDrawIndirect ? DrawCall::kIndirect : DrawCall::kIndexedIndirect};
                d.indirectOffset = b[0];
                d.indexAddr = gRegs.indexBase;
                draw(d, st);
                break;
            }
            case kOpDispatchDirect: {
                DrawCall d{DrawCall::kDispatch};
                d.groups[0] = b[0], d.groups[1] = b[1], d.groups[2] = b[2];
                draw(d, st);
                break;
            }
            case kOpDispatchIndirect: {
                DrawCall d{DrawCall::kDispatchIndirect};
                d.indirectOffset = b[0];
                draw(d, st);
                break;
            }
            case kOpDmaData: dmaData(b); break;
            case kOpIndirectBuffer:
            case kOpIndirectBufferConst:
                if (depth < 4) {
                    const auto* ib = reinterpret_cast<const uint32_t*>(address(b[0] & ~3u, b[1]));
                    execute(ib, ib + (b[2] & 0xFFFFF), st, depth + 1);
                }
                break;
            case kOpWriteData: {
                const uint32_t dstSel = (b[0] >> 8) & 0xF;
                if (dstSel == 2 || dstSel == 5) {
                    auto* dst = reinterpret_cast<volatile uint32_t*>(address(b[1], b[2]));
                    for (uint32_t i = 3; i < count + 1; ++i) dst[i - 3] = b[i];
                    ++st->fences;
                }
                break;
            }
            case kOpEventWriteEop:
                endOfPipe(address(b[1], b[2]), b[2] >> 29, (b[2] >> 24) & 3, b[3], b[4], st);
                break;
            case kOpReleaseMem:
                endOfPipe(address(b[2], b[3]), b[1] >> 29, (b[1] >> 24) & 3, b[4], b[5], st);
                break;
            case kOpEventWriteEos:
                if ((b[2] >> 29) == 2) {
                    *reinterpret_cast<volatile uint32_t*>(address(b[1], b[2])) = b[3];
                    ++st->fences;
                }
                break;
            case kOpWaitRegMem:
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

void setDrawListener(DrawListener* l) { gListener = l; }
DrawListener* drawListener() { return gListener; }

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
    cmds[0] = (3u << 30) | ((numDwords - 2) << 16) | (kOpNop << 8);
    std::memset(cmds + 1, 0, (numDwords - 1) * 4);
}

} // namespace host
