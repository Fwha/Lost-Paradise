#include "host/pm4.h"

#include "host/equeue.h"
#include "host/profile.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace host {

namespace {

using namespace gpu;

// The graphics ring's state. The recorder thread (or, without the command processor thread,
// the submitting thread) and the analysis thread each keep their own, from the same packets.
GpuRegs gRecordRegs, gAnalysisRegs;
thread_local GpuRegs* tRegs = &gRecordRegs;
thread_local bool tDry = false; // the analysis thread: no side effects, analysis hooks only
bool commandProcessorIsAsync();
// Analysis hooks run inline, just before their counterparts: without the command processor
// thread, or with LPN_NO_ANALYSIS_THREAD=1 (for comparison).
bool inlineAnalysis() {
    static const bool v = !commandProcessorIsAsync() || getenv("LPN_NO_ANALYSIS_THREAD") != nullptr;
    return v;
}
std::atomic<uint64_t> gFrame{0};
const uint64_t gTraceFrame = getenv("LPN_PM4_TRACE") ? std::strtoull(getenv("LPN_PM4_TRACE"), nullptr, 0) : ~0ull;
std::vector<DrawListener*> gListeners;

void draw(const DrawCall& d, Pm4Stats* st) {
    if (tDry) {
        for (DrawListener* l : gListeners) l->onAnalyzeDraw(d, *tRegs);
        return;
    }
    (d.kind >= DrawCall::kDispatch ? st->dispatches : st->draws)++;
    for (DrawListener* l : gListeners) {
        if (inlineAnalysis()) {
            l->onAnalyzeDraw(d, *tRegs);
            l->onAnalyzeFlush();
        }
        l->onDraw(d, *tRegs);
    }
}

// CP DMA: copies or fills memory on the GPU timeline: by the renderer, in order with the draws,
// or here and now without one.
void dmaData(const uint32_t* b) {
    const uint32_t control = b[0], bytes = b[5] & 0x1FFFFF;
    const uint32_t srcSel = (control >> 29) & 3, dstSel = (control >> 20) & 3;
    const uint64_t src = uint64_t(b[1]) | (uint64_t(b[2]) << 32), dst = uint64_t(b[3]) | (uint64_t(b[4]) << 32);
    if (dstSel != 0 || !dst || !bytes) return; // only memory destinations matter to the CPU
    if (!((srcSel == 0 && src) || srcSel == 2)) return;
    if (tDry || inlineAnalysis())
        for (DrawListener* l : gListeners) l->onAnalyzeDma(dst, srcSel == 0 ? src : 0, bytes, srcSel == 2, b[1]);
    if (tDry) return;
    for (DrawListener* l : gListeners)
        if (l->onDmaCopy(dst, srcSel == 0 ? src : 0, bytes, srcSel == 2, b[1])) return;
    if (srcSel == 0 && src) std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
    else if (srcSel == 2) { // fill with the immediate in src_lo
        auto* d = reinterpret_cast<uint32_t*>(dst);
        for (uint32_t i = 0; i < bytes / 4; ++i) d[i] = b[1];
    } else {
        return;
    }
    for (DrawListener* l : gListeners) l->onMemoryWrite(dst, bytes);
}

uint64_t gpuClock() {
    // The GPU counter runs at 100 MHz on the PS4.
    using namespace std::chrono;
    return uint64_t(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 10);
}

uint64_t address(uint32_t lo, uint32_t hi) { return uint64_t(lo) | (uint64_t(hi & 0xFFFF) << 32); }

void queueRecorder(std::function<void()> fn);
void queueChunk(std::vector<uint32_t> chunk, Pm4Stats* st);

bool commandProcessorIsAsync() {
    static const bool async = getenv("LPN_SYNC_SUBMIT") == nullptr;
    return async;
}

// WAIT_REG_MEM on memory: the game makes the GPU wait for a value, often one the CPU writes
// once it has filled the data the next draws read. On the command processor thread this
// waits as the GPU would (up to a limit, in case nothing ever writes it); run synchronously
// at submit, the CPU could not write it, so the wait is skipped.
// Values the command stream has the GPU write to memory (labels, WRITE_DATA), by address: the
// command processor thread notes them as it passes them.
std::unordered_map<uint64_t, uint32_t> gGpuWillWrite;

void noteGpuWrite(uint64_t addr, uint32_t value) {
    if (gGpuWillWrite.size() > 4096) gGpuWillWrite.clear();
    gGpuWillWrite[addr] = value;
}

// selfOk: a value the stream has already asked the GPU to write counts as written.
void waitRegMem(const uint32_t* b, Pm4Stats* st, bool selfOk = true) {
    const uint32_t control = b[0], func = control & 7;
    if (!(control & 0x10) || func == 0) return; // registers, or "always"
    const auto* mem = reinterpret_cast<volatile uint32_t*>(address(b[1] & ~3u, b[2]));
    const uint32_t ref = b[3], mask = b[4];
    auto test = [&](uint32_t v) {
        v &= mask;
        switch (func) {
        case 1: return v < ref;
        case 2: return v <= ref;
        case 3: return v == ref;
        case 4: return v != ref;
        case 5: return v >= ref;
        case 6: return v > ref;
        default: return true;
        }
    };
    auto met = [&] { return test(*mem); };
    if (met()) return;
    // Mostly the GPU waits for its own end-of-pipe label (a pipeline drain between passes). The
    // renderer runs everything in order behind full barriers, so a value the stream has already
    // asked the GPU to write is as good as written: only values from elsewhere (the CPU) are
    // waited for.
    if (commandProcessorIsAsync() && selfOk) {
        auto it = gGpuWillWrite.find(uint64_t(reinterpret_cast<uintptr_t>(mem)));
        if (it != gGpuWillWrite.end() && test(it->second)) return;
    }
    if (!commandProcessorIsAsync()) {
        ++st->skippedWaits;
        return;
    }
    queueRecorder([] {
        for (DrawListener* l : gListeners) l->onWait();
    });
    const auto start = std::chrono::steady_clock::now();
    struct WaitStats {
        uint64_t waits = 0, ns = 0, maxNs = 0, maxAddr = 0;
    };
    static WaitStats ws;
    static const bool profile = getenv("LPN_PROFILE") != nullptr;
    struct Done {
        decltype(start) t0;
        const volatile uint32_t* mem;
        ~Done() {
            if (!profile) return;
            const uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            ++ws.waits, ws.ns += ns;
            if (ns > ws.maxNs) ws.maxNs = ns, ws.maxAddr = uint64_t(reinterpret_cast<uintptr_t>(mem));
            static uint64_t reportedFrame = 0;
            static double reportedAt = 0;
            if (gFrame > reportedFrame && profile::seconds() >= reportedAt + 1) {
                std::fprintf(stderr, "[pm4] t=%.1f frames to %" PRIu64 ": %" PRIu64 " waits, %.2f ms per frame, longest %.2f ms on 0x%" PRIx64 "\n",
                             profile::seconds(), gFrame.load(), ws.waits, ws.ns / 1e6 / double(gFrame - reportedFrame), ws.maxNs / 1e6, ws.maxAddr);
                reportedFrame = gFrame;
                reportedAt = profile::seconds();
                ws = {};
            }
        }
    } done{start, mem};
    const bool tl = timelineOn();
    if (tl) timeline("cp: wait for 0x%" PRIx64 " (now 0x%x)", uint64_t(reinterpret_cast<uintptr_t>(mem)), *mem);
    const auto deadline = start + std::chrono::seconds(2);
    for (uint32_t spins = 0; !met(); ++spins) {
        if (std::chrono::steady_clock::now() > deadline) {
            static uint32_t warned = 0;
            if (warned++ < 5)
                std::fprintf(stderr, "[pm4] gave up waiting for 0x%" PRIx64 " (func %u ref 0x%x, now 0x%x)\n",
                             uint64_t(reinterpret_cast<uintptr_t>(mem)), func, ref, *mem);
            ++st->skippedWaits;
            return;
        }
        if (spins < 64) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (tl) timeline("cp: wait for 0x%" PRIx64 " done", uint64_t(reinterpret_cast<uintptr_t>(mem)));
}

// Runs fn when the GPU work submitted before it is done (now, without an asynchronous renderer).
void afterGpuWork(std::function<void()> fn) {
    for (DrawListener* l : gListeners)
        if (l->afterGpuWork(fn)) return;
    fn();
}

// Shared tail of EVENT_WRITE_EOP and RELEASE_MEM: the fence and interrupt fire once the
// draws before them have run, so the game never sees a label for unfinished work.
void endOfPipe(uint64_t addr, uint32_t dataSel, uint32_t intSel, uint32_t dataLo, uint32_t dataHi, Pm4Stats* st) {
    if (dataSel) ++st->fences;
    if (intSel) ++st->interrupts;
    if (!(addr && dataSel) && !intSel) return;
    afterGpuWork([=] {
        if (timelineOn()) timeline("gpu: end-of-pipe label 0x%" PRIx64 " = 0x%x%s", addr, dataLo, intSel ? " + interrupt" : "");
        if (addr) switch (dataSel) {
            case 1: *reinterpret_cast<volatile uint32_t*>(addr) = dataLo; break;
            case 2: *reinterpret_cast<volatile uint64_t*>(addr) = uint64_t(dataLo) | (uint64_t(dataHi) << 32); break;
            case 3:
            case 4: *reinterpret_cast<volatile uint64_t*>(addr) = gpuClock(); break;
            default: break;
            }
        if (intSel) triggerEvent(kEqEventGfxEop, kFilterGraphicsCore, 0);
    });
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
            if (tDry && (op == kOpEventWriteEop || op == kOpReleaseMem || op == kOpEventWriteEos || op == kOpWaitRegMem)) {
                p += len;
                continue;
            }
            switch (op) {
            case kOpSetContextReg:
            case kOpSetShReg:
            case kOpSetUconfigReg: {
                // body: register offset (low 16 bits; the top holds an index field), then values
                const uint32_t reg = b[0] & 0xFFFF;
                uint32_t* file = op == kOpSetContextReg ? tRegs->ctx : op == kOpSetShReg ? tRegs->sh : tRegs->uconfig;
                const uint32_t size = op == kOpSetContextReg ? kContextRegs : op == kOpSetShReg ? kShRegs : kUconfigRegs;
                if (reg + count <= size) std::memcpy(&file[reg], b + 1, count * 4);
                if (op == kOpSetContextReg) ++tRegs->ctxSerial;
                break;
            }
            case kOpIndexBase: tRegs->indexBase = address(b[0], b[1]); break;
            case kOpIndexBufferSize: tRegs->indexBufferSize = b[0]; break;
            case kOpIndexType: tRegs->indexType = b[0] & 3; break;
            case kOpNumInstances: tRegs->numInstances = b[0] ? b[0] : 1; break;
            case kOpSetBase: tRegs->indirectBase = address(b[1], b[2]); break;
            case kOpDrawIndex2: {
                DrawCall d{DrawCall::kIndexed};
                d.indexAddr = address(b[1], b[2]);
                d.count = b[3];
                draw(d, st);
                break;
            }
            case kOpDrawIndexOffset2: {
                DrawCall d{DrawCall::kIndexed};
                d.indexAddr = tRegs->indexBase;
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
                d.indexAddr = tRegs->indexBase;
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
            case kOpDmaData:
                if (gFrame == gTraceFrame)
                    std::fprintf(stderr, "[pm4] dma control 0x%x src 0x%x%08x dst 0x%x%08x bytes 0x%x\n", b[0], b[2], b[1], b[4],
                                 b[3], b[5]);
                dmaData(b);
                break;
            case kOpEventWrite:
                if (gFrame == gTraceFrame)
                    std::fprintf(stderr, "[pm4] event 0x%x%s 0x%x%08x\n", b[0] & 0x3F, count >= 2 ? " addr" : "",
                                 count >= 2 ? b[2] : 0, count >= 2 ? b[1] : 0);
                break;
            case kOpIndirectBuffer:
            case kOpIndirectBufferConst:
                if (depth < 4) {
                    const auto* ib = reinterpret_cast<const uint32_t*>(address(b[0] & ~3u, b[1]));
                    execute(ib, ib + (b[2] & 0xFFFFF), st, depth + 1);
                }
                break;
            case kOpWriteData: {
                const uint32_t dstSel = (b[0] >> 8) & 0xF;
                if (gFrame == gTraceFrame)
                    std::fprintf(stderr, "[pm4] write data control 0x%x addr 0x%" PRIx64 " dwords %u first 0x%x (draws so far %" PRIu64 ")\n",
                                 b[0], address(b[1], b[2]), count - 2, count >= 3 ? b[3] : 0, st->draws);
                if (dstSel == 2 || dstSel == 5) {
                    if (tDry || inlineAnalysis())
                        for (DrawListener* l : gListeners) l->onAnalyzeWrite(address(b[1], b[2]), b + 3, count - 2);
                    if (tDry) break;
                    auto* dst = reinterpret_cast<volatile uint32_t*>(address(b[1], b[2]));
                    for (uint32_t i = 3; i < count + 1; ++i) dst[i - 3] = b[i];
                    for (DrawListener* l : gListeners) l->onMemoryWrite(address(b[1], b[2]), uint64_t(count - 2) * 4);
                    for (DrawListener* l : gListeners) l->onCpWrite(address(b[1], b[2]), uint64_t(count - 2) * 4);
                    ++st->fences;
                }
                break;
            }
            case kOpEventWriteEop:
                if (gFrame == gTraceFrame)
                    std::fprintf(stderr, "[pm4] eop addr 0x%" PRIx64 " data sel %u int %u value 0x%x (draws so far %" PRIu64 ")\n",
                                 address(b[1], b[2]), b[2] >> 29, (b[2] >> 24) & 3, b[3], st->draws);
                endOfPipe(address(b[1], b[2]), b[2] >> 29, (b[2] >> 24) & 3, b[3], b[4], st);
                break;
            case kOpReleaseMem:
                if (gFrame == gTraceFrame)
                    std::fprintf(stderr, "[pm4] release mem addr 0x%" PRIx64 " data sel %u int %u value 0x%x (draws so far %" PRIu64 ")\n",
                                 address(b[2], b[3]), b[1] >> 29, (b[1] >> 24) & 3, b[4], st->draws);
                endOfPipe(address(b[2], b[3]), b[1] >> 29, (b[1] >> 24) & 3, b[4], b[5], st);
                break;
            case kOpEventWriteEos:
                if ((b[2] >> 29) == 2) {
                    const uint64_t addr = address(b[1], b[2]);
                    const uint32_t value = b[3];
                    afterGpuWork([=] { *reinterpret_cast<volatile uint32_t*>(addr) = value; });
                    ++st->fences;
                }
                break;
            case kOpWaitRegMem: waitRegMem(b, st); break; // only without the command processor thread
            default: break;
            }
        }
        p += len;
    }
}

// The command processor thread's half: walks the packets (into indirect buffers), waits where
// the game told the GPU to wait, and hands everything else, flattened, to the recorder thread,
// which executes it as before. Recording thus carries on while the command processor waits.
void scan(const uint32_t* p, const uint32_t* end, std::vector<uint32_t>& out, Pm4Stats* st, int depth) {
    const uint32_t* drain = nullptr; // the last wait skipped as self-satisfied, until a draw
    while (p < end) {
        const uint32_t h = *p;
        const uint32_t type = h >> 30;
        if (type == 2) {
            ++p;
            continue;
        }
        const uint32_t count = (h >> 16) & 0x3FFF;
        const uint32_t len = count + 2;
        if (type == 1 || p + len > end) {
            std::fprintf(stderr, "[pm4] malformed packet 0x%08x, skipping rest of buffer\n", h);
            return;
        }
        const uint32_t op = type == 3 ? (h >> 8) & 0xFF : 0;
        const uint32_t* b = p + 1;
        if (type == 3 && (op == kOpIndirectBuffer || op == kOpIndirectBufferConst)) {
            if (depth < 4) {
                const auto* ib = reinterpret_cast<const uint32_t*>(address(b[0] & ~3u, b[1]));
                scan(ib, ib + (b[2] & 0xFFFFF), out, st, depth + 1);
            }
        } else if (type == 3 && op == kOpWaitRegMem) {
            if (gFrame == gTraceFrame)
                std::fprintf(stderr, "[pm4] wait reg mem control 0x%x addr 0x%" PRIx64 " ref 0x%x mask 0x%x now 0x%x\n", b[0],
                             address(b[1], b[2]), b[3], b[4],
                             (b[0] & 0x10) ? *reinterpret_cast<volatile uint32_t*>(address(b[1] & ~3u, b[2])) : 0);
            if (!out.empty()) // what comes before is recorded meanwhile
                queueChunk(std::move(out), st);
            out.clear();
            waitRegMem(b, st);
            drain = b;
        } else {
            // A wait the GPU satisfies itself (a drain) is skipped, but memory the command
            // processor writes after it, before the next draw, is the game's signal that the GPU
            // got past the drain: the CPU may then reuse what the draws before it read. Such a
            // WRITE_DATA is done once the GPU has finished the work before it. (A DMA after a
            // drain runs on the GPU timeline anyway: see onDmaCopy.)
            if (drain && type == 3 && op == kOpWriteData && (((b[0] >> 8) & 0xF) == 2 || ((b[0] >> 8) & 0xF) == 5)) {
                std::vector<uint32_t> values(b + 3, b + count + 1);
                const uint64_t addr = address(b[1], b[2]);
                noteGpuWrite(addr, values.empty() ? 0 : values[0]);
                if (!out.empty())
                    queueChunk(std::move(out), st);
                out.clear();
                queueRecorder([addr, values = std::move(values)] {
                    afterGpuWork([addr, values] {
                        auto* dst = reinterpret_cast<volatile uint32_t*>(addr);
                        for (size_t i = 0; i < values.size(); ++i) dst[i] = values[i];
                    });
                });
                ++st->fences;
                p += len;
                continue;
            }
            if (drain && type == 3) {
                if (op == kOpDrawIndex2 || op == kOpDrawIndexOffset2 || op == kOpDrawIndexAuto ||
                           op == kOpDrawIndexIndirect || op == kOpDrawIndirect || op == kOpDispatchDirect ||
                           op == kOpDispatchIndirect) {
                    drain = nullptr;
                }
            }
            if (type == 3) switch (op) { // the values this will have the GPU write
                case kOpEventWriteEop:
                    if ((b[2] >> 29) == 1 || (b[2] >> 29) == 2) noteGpuWrite(address(b[1], b[2]), b[3]);
                    break;
                case kOpReleaseMem:
                    if ((b[1] >> 29) == 1 || (b[1] >> 29) == 2) noteGpuWrite(address(b[2], b[3]), b[4]);
                    break;
                case kOpEventWriteEos:
                    if ((b[2] >> 29) == 2) noteGpuWrite(address(b[1], b[2]), b[3]);
                    break;
                case kOpWriteData:
                    if (((b[0] >> 8) & 0xF) == 2 || ((b[0] >> 8) & 0xF) == 5)
                        for (uint32_t i = 3; i < count + 1; ++i) noteGpuWrite(address(b[1], b[2]) + (i - 3) * 4, b[i]);
                    break;
                default: break;
                }
            out.insert(out.end(), p, p + len);
        }
        p += len;
    }
}

struct {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::function<void()>> work;
    bool started = false;
} gRec;

constexpr size_t kRecQueueLimit = 1024;

void recorderThread() {
    profile::nameThread("lpn-record");
    for (;;) {
        std::function<void()> fn;
        {
            std::unique_lock lock(gRec.m);
            gRec.cv.wait(lock, [] { return !gRec.work.empty(); });
            fn = std::move(gRec.work.front());
            gRec.work.pop_front();
        }
        gRec.cv.notify_all();
        fn();
    }
}

void queueRecord(std::function<void()> fn) {
    std::unique_lock lock(gRec.m);
    if (!gRec.started) {
        gRec.started = true;
        std::thread(recorderThread).detach();
    }
    gRec.cv.wait(lock, [] { return gRec.work.size() < kRecQueueLimit; });
    gRec.work.push_back(std::move(fn));
    gRec.cv.notify_all();
}

// The analysis thread sits between the command processor and the recorder: it executes each
// chunk of packets dry (analysis hooks only, its own registers) and then hands it on, so the
// recorder finds every draw's shaders analysed.
struct Stage {
    std::vector<uint32_t> chunk;
    Pm4Stats* st = nullptr;
    std::function<void()> analyze, record;
};
struct {
    std::mutex m;
    std::condition_variable cv;
    std::deque<Stage> work;
    bool started = false;
} gAna;

void analysisThread() {
    profile::nameThread("lpn-analyze");
    tDry = true;
    tRegs = &gAnalysisRegs;
    Pm4Stats dry;
    for (;;) {
        Stage w;
        {
            std::unique_lock lock(gAna.m);
            gAna.cv.wait(lock, [] { return !gAna.work.empty(); });
            w = std::move(gAna.work.front());
            gAna.work.pop_front();
        }
        gAna.cv.notify_all();
        if (!w.chunk.empty()) {
            execute(w.chunk.data(), w.chunk.data() + w.chunk.size(), &dry, 0);
            for (DrawListener* l : gListeners) l->onAnalyzeFlush();
            queueRecord([chunk = std::move(w.chunk), st = w.st] { execute(chunk.data(), chunk.data() + chunk.size(), st, 0); });
        }
        if (w.analyze) w.analyze();
        if (w.analyze || w.record)
            for (DrawListener* l : gListeners) l->onAnalyzeFlush();
        if (w.record) queueRecord(std::move(w.record));
    }
}

void queueStage(Stage w) {
    if (!commandProcessorIsAsync()) { // on the caller's thread, analysis inline
        if (!w.chunk.empty()) execute(w.chunk.data(), w.chunk.data() + w.chunk.size(), w.st, 0);
        if (w.analyze) w.analyze();
        if (w.record) w.record();
        return;
    }
    if (inlineAnalysis()) {
        if (!w.chunk.empty())
            queueRecord([chunk = std::move(w.chunk), st = w.st] { execute(chunk.data(), chunk.data() + chunk.size(), st, 0); });
        if (w.analyze || w.record)
            queueRecord([a = std::move(w.analyze), r = std::move(w.record)] {
                if (a) a();
                if (r) r();
            });
        return;
    }
    std::unique_lock lock(gAna.m);
    if (!gAna.started) {
        gAna.started = true;
        std::thread(analysisThread).detach();
    }
    gAna.cv.wait(lock, [] { return gAna.work.size() < kRecQueueLimit; });
    gAna.work.push_back(std::move(w));
    gAna.cv.notify_all();
}

// Work for the recorder thread, in order with the commands queued before it.
void queueRecorder(std::function<void()> fn) {
    Stage w;
    w.record = std::move(fn);
    queueStage(std::move(w));
}
void queueChunk(std::vector<uint32_t> chunk, Pm4Stats* st) {
    Stage w;
    w.chunk = std::move(chunk);
    w.st = st;
    queueStage(std::move(w));
}

} // namespace

namespace {
const auto gTimelineStart = std::chrono::steady_clock::now();
std::atomic<double> gTimelineAt{getenv("LPN_TIMELINE") ? std::atof(getenv("LPN_TIMELINE")) : -1};
std::atomic<double> gTimelineFor{0.1};
} // namespace

bool timelineOn() {
    const double at = gTimelineAt.load(std::memory_order_relaxed);
    if (at < 0) return false;
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - gTimelineStart).count();
    return t >= at && t < at + gTimelineFor.load(std::memory_order_relaxed);
}

void startTimeline(double seconds) {
    gTimelineFor = seconds;
    gTimelineAt = std::chrono::duration<double>(std::chrono::steady_clock::now() - gTimelineStart).count();
}

void timeline(const char* fmt, ...) {
    static const auto start = std::chrono::steady_clock::now();
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[tl] %9.3f %s\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), buf);
}

void addDrawListener(DrawListener* l) { gListeners.push_back(l); }
void notifySubmitEnd() {
    Stage w;
    w.analyze = [] {
        if (timelineOn()) timeline("analysis: submission analysed");
        for (DrawListener* l : gListeners) l->onAnalyzeSubmitEnd();
    };
    w.record = [] {
        if (timelineOn()) timeline("rec: submission recorded");
        for (DrawListener* l : gListeners) l->onSubmitEnd();
    };
    queueStage(std::move(w));
}
void notifyFrameEnd() {
    Stage w;
    w.analyze = [] {
        if (timelineOn()) timeline("analysis: frame analysed");
        for (DrawListener* l : gListeners) l->onAnalyzeFrameEnd();
    };
    w.record = [] {
        if (timelineOn()) timeline("rec: frame %" PRIu64 " recorded", gFrame.load() + 1);
        ++gFrame;
        for (DrawListener* l : gListeners) l->onFrameEnd();
    };
    queueStage(std::move(w));
}
void afterCommandsRecorded(std::function<void()> fn) { queueRecorder(std::move(fn)); }

void executePm4(const uint32_t* cmds, uint32_t numDwords, Pm4Stats* stats) {
    static Pm4Stats local;
    if (!stats) stats = &local;
    if (!commandProcessorIsAsync()) {
        execute(cmds, cmds + numDwords, stats, 0);
        return;
    }
    std::vector<uint32_t> out;
    scan(cmds, cmds + numDwords, out, stats, 0);
    if (!out.empty())
        queueChunk(std::move(out), stats);
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

namespace {

struct {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::function<void()>> work;
    bool started = false;
} gCp;

// How far the game may get ahead of the command processor; on the console the ring fills up.
constexpr size_t kCpQueueLimit = 64;

void commandProcessorThread() {
    profile::nameThread("lpn-cp");
    for (;;) {
        std::function<void()> fn;
        {
            std::unique_lock lock(gCp.m);
            gCp.cv.wait(lock, [] { return !gCp.work.empty(); });
            fn = std::move(gCp.work.front());
        }
        fn();
        {
            std::lock_guard lock(gCp.m);
            gCp.work.pop_front(); // popped after running, so the queue's length counts it
        }
        gCp.cv.notify_all();
    }
}

} // namespace

void queueCommandProcessor(std::function<void()> fn) {
    if (!commandProcessorIsAsync()) {
        static std::mutex m;
        std::lock_guard lock(m);
        fn();
        return;
    }
    std::unique_lock lock(gCp.m);
    if (!gCp.started) {
        gCp.started = true;
        std::thread(commandProcessorThread).detach();
    }
    gCp.cv.wait(lock, [] { return gCp.work.size() < kCpQueueLimit; });
    gCp.work.push_back(std::move(fn));
    gCp.cv.notify_all();
}

} // namespace host
