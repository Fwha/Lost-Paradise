// libSceGnmDriver: command buffer submission, GPU events and the packet-writing helpers.
//
// The sceGnmSet*/Draw*/Dispatch* helpers write PM4 into the caller's buffer as
// (cmd, numDwords, ...). Until the renderer exists they write a NOP of that size, so the
// buffer stays walkable and the game's own fences still run.
#include "host/equeue.h"
#include "host/hle.h"
#include "host/kernel.h"
#include "host/memory.h"
#include "host/pm4.h"
#include "host/videoout.h"

#include "host/mman.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace host {

void registerGnmPacketHle();

namespace {

std::mutex gSubmitMutex; // guards gComputeQueues
Pm4Stats gStats; // the command processor thread's
uint64_t gSubmits = 0;

// LPN_PROFILE=1: time spent inside submissions (PM4 processing plus the renderer).
const bool gProfile = getenv("LPN_PROFILE") != nullptr;
uint64_t gSubmitNs = 0;

// A submission's command buffers in execution order. The game keeps them intact until the GPU
// signals it is done with them, so only the list is copied.
struct CommandBuffers {
    std::vector<std::pair<const uint32_t*, uint32_t>> list; // address, dwords

    CommandBuffers(uint32_t count, void* const* dcb, const uint32_t* dcbBytes, void* const* ccb,
                   const uint32_t* ccbBytes) {
        for (uint32_t i = 0; i < count; ++i) {
            if (ccb && ccb[i] && ccbBytes[i]) list.push_back({static_cast<const uint32_t*>(ccb[i]), ccbBytes[i] / 4});
            if (dcb && dcb[i] && dcbBytes[i]) list.push_back({static_cast<const uint32_t*>(dcb[i]), dcbBytes[i] / 4});
        }
    }
};

// Runs on the command processor thread.
void execute(const CommandBuffers& cbs) {
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& [cmds, dwords] : cbs.list) executePm4(cmds, dwords, &gStats);
    notifySubmitEnd();
    if (gProfile) {
        gSubmitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        if ((gSubmits + 1) % 900 == 0) {
            std::fprintf(stderr, "[prof] submissions: %.2f ms each\n", gSubmitNs / 1e6 / 900);
            gSubmitNs = 0;
        }
    }
    // LPN_GPU_STATS=N: PM4 opcode counts over the 300 submissions ending at submission N (default 300).
    static const uint64_t statsAt = getenv("LPN_GPU_STATS") ? std::max(300ul, std::strtoul(getenv("LPN_GPU_STATS"), nullptr, 0)) : 0;
    ++gSubmits;
    if (statsAt && gSubmits == statsAt - 300) gStats = Pm4Stats{};
    if (statsAt && gSubmits == statsAt) {
        std::fprintf(stderr, "[gnm] PM4 opcodes over submissions %llu-%llu:\n", (unsigned long long)(statsAt - 300),
                     (unsigned long long)statsAt);
        for (int op = 0; op < 256; ++op)
            if (gStats.opcodes[op]) std::fprintf(stderr, "  0x%02x %10llu\n", op, (unsigned long long)gStats.opcodes[op]);
    }
    if (gSubmits == 1 || gSubmits % 600 == 0)
        std::fprintf(stderr, "[gnm] %llu submits: %llu packets, %llu fences, %llu interrupts\n",
                     (unsigned long long)gSubmits, (unsigned long long)gStats.packets,
                     (unsigned long long)gStats.fences, (unsigned long long)gStats.interrupts);
}

int32_t sceGnmSubmitCommandBuffers(uint32_t count, void* const* dcb, const uint32_t* dcbBytes, void* const* ccb,
                                   const uint32_t* ccbBytes) {
    if (timelineOn()) timeline("game: submit");
    queueCommandProcessor([cbs = CommandBuffers(count, dcb, dcbBytes, ccb, ccbBytes)] {
        if (timelineOn()) timeline("cp: submission");
        execute(cbs);
    });
    return 0;
}

int32_t sceGnmSubmitAndFlipCommandBuffers(uint32_t count, void* const* dcb, const uint32_t* dcbBytes,
                                          void* const* ccb, const uint32_t* ccbBytes, int32_t videoOut,
                                          int32_t bufferIndex, int32_t flipMode, int64_t flipArg) {
    if (timelineOn()) timeline("game: submit and flip");
    queueCommandProcessor([cbs = CommandBuffers(count, dcb, dcbBytes, ccb, ccbBytes)] {
        if (timelineOn()) timeline("cp: submission and flip");
        execute(cbs);
        notifyFrameEnd();
    });
    return videoOutSubmitFlip(videoOut, bufferIndex, flipMode, flipArg); // shown once that work is done
}

int32_t sceGnmSubmitCommandBuffersForWorkload(uint64_t, uint32_t count, void* const* dcb, const uint32_t* dcbBytes,
                                              void* const* ccb, const uint32_t* ccbBytes) {
    return sceGnmSubmitCommandBuffers(count, dcb, dcbBytes, ccb, ccbBytes);
}

int32_t sceGnmSubmitAndFlipCommandBuffersForWorkload(uint64_t, uint32_t count, void* const* dcb,
                                                     const uint32_t* dcbBytes, void* const* ccb,
                                                     const uint32_t* ccbBytes, int32_t videoOut, int32_t bufferIndex,
                                                     int32_t flipMode, int64_t flipArg) {
    return sceGnmSubmitAndFlipCommandBuffers(count, dcb, dcbBytes, ccb, ccbBytes, videoOut, bufferIndex, flipMode,
                                             flipArg);
}

int32_t sceGnmSubmitDone() { return 0; }
int32_t sceGnmAreSubmitsAllowed() { return 1; }

int32_t sceGnmAddEqEvent(EventQueue* q, uint64_t id, void* udata) {
    if (!q) return sceError(ps4errno::kEBADF);
    q->add(id, kFilterGraphicsCore, udata);
    return 0;
}
int32_t sceGnmDeleteEqEvent(EventQueue* q, uint64_t id) {
    if (q) q->remove(id, kFilterGraphicsCore);
    return 0;
}
int32_t sceGnmGetEqEventType(const KernelEvent* ev) { return int32_t(ev->ident); }
uint64_t sceGnmGetEqTimeStamp(const KernelEvent* ev) { return uint64_t(ev->data); }

// Compute queues: rings the game writes packets into, kicked with DingDong.
struct ComputeQueue {
    const uint32_t* ring;
    uint32_t sizeDw;
    uint32_t readDw;
};
std::vector<ComputeQueue> gComputeQueues;

int32_t sceGnmMapComputeQueue(uint32_t, uint32_t, void* ring, uint32_t sizeDw, void*) {
    std::lock_guard lock(gSubmitMutex);
    gComputeQueues.push_back({static_cast<const uint32_t*>(ring), sizeDw, 0});
    return int32_t(gComputeQueues.size()); // vqueue id, 1-based
}
int32_t sceGnmMapComputeQueueWithPriority(uint32_t pipe, uint32_t queue, void* ring, uint32_t sizeDw, void* rptr,
                                          uint32_t) {
    return sceGnmMapComputeQueue(pipe, queue, ring, sizeDw, rptr);
}
void sceGnmUnmapComputeQueue(uint32_t) {}

void sceGnmDingDong(uint32_t vqid, uint32_t nextDw) {
    std::lock_guard lock(gSubmitMutex);
    if (vqid == 0 || vqid > gComputeQueues.size()) return;
    auto& q = gComputeQueues[vqid - 1];
    // The packets are copied now: the game may refill the ring once it is kicked. A wrapped
    // range is stitched so packets crossing the end stay whole.
    std::vector<uint32_t> packets;
    if (nextDw >= q.readDw) {
        packets.assign(q.ring + q.readDw, q.ring + nextDw);
    } else {
        packets.assign(q.ring + q.readDw, q.ring + q.sizeDw);
        packets.insert(packets.end(), q.ring, q.ring + nextDw);
    }
    q.readDw = nextDw % q.sizeDw;
    queueCommandProcessor([packets = std::move(packets)] {
        executePm4(packets.data(), uint32_t(packets.size()), &gStats);
        notifySubmitEnd();
    });
}
void sceGnmDingDongForWorkload(uint32_t vqid, uint32_t nextDw, uint64_t) { sceGnmDingDong(vqid, nextDw); }

// Workloads are profiling scopes.
int32_t sceGnmCreateWorkloadStream(const char*, uint32_t* stream) {
    if (stream) *stream = 1;
    return 0;
}
int32_t sceGnmBeginWorkload(uint32_t, uint64_t* workload) {
    static std::atomic<uint64_t> next{1};
    if (workload) *workload = next++;
    return 0;
}
int32_t sceGnmEndWorkload(uint64_t) { return 0; }
int32_t sceGnmDestroyWorkloadStream(uint32_t) { return 0; }

void* sceGnmGetTheTessellationFactorRingBufferBaseAddress() {
    static void* ring = guestMap(nullptr, 0x20000, 0x10000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return ring;
}
uint32_t sceGnmGetOffChipTessellationBufferSize() { return 0x800000; }

int32_t returnsZero() { return 0; }
void doesNothing() {}

int32_t writesNop(uint32_t* cmd, uint32_t numDwords) {
    if (cmd && numDwords > 0 && numDwords < 0x4000) writePm4Nop(cmd, numDwords);
    return 0;
}

} // namespace

void registerGnmHle() {
    const char* g = "libSceGnmDriver";
    registerGnmPacketHle();
    HLE(g, sceGnmSubmitCommandBuffers);
    HLE(g, sceGnmSubmitAndFlipCommandBuffers);
    HLE(g, sceGnmSubmitCommandBuffersForWorkload);
    HLE(g, sceGnmSubmitAndFlipCommandBuffersForWorkload);
    HLE(g, sceGnmSubmitDone);
    HLE(g, sceGnmAreSubmitsAllowed);
    HLE(g, sceGnmAddEqEvent);
    HLE(g, sceGnmDeleteEqEvent);
    HLE(g, sceGnmGetEqEventType);
    HLE(g, sceGnmGetEqTimeStamp);
    HLE(g, sceGnmMapComputeQueue);
    HLE(g, sceGnmMapComputeQueueWithPriority);
    HLE(g, sceGnmUnmapComputeQueue);
    HLE(g, sceGnmDingDong);
    HLE(g, sceGnmDingDongForWorkload);
    HLE(g, sceGnmCreateWorkloadStream);
    HLE(g, sceGnmBeginWorkload);
    HLE(g, sceGnmEndWorkload);
    HLE(g, sceGnmDestroyWorkloadStream);
    HLE(g, sceGnmGetTheTessellationFactorRingBufferBaseAddress);
    HLE(g, sceGnmGetOffChipTessellationBufferSize);

    // Packet writers not translated yet (other shader stages, markers, waits): NOP-filled.
    for (const char* name : {"sceGnmDrawIndexIndirectCountMulti", "sceGnmDrawIndexMultiInstanced", "sceGnmDrawIndirectCountMulti", "sceGnmDispatchIndirectOnMec", "sceGnmSetEmbeddedPsShader", "sceGnmSetEmbeddedVsShader", "sceGnmSetHsShader", "sceGnmSetLsShader", "sceGnmUpdateHsShader", "sceGnmInsertPushMarker", "sceGnmInsertPopMarker", "sceGnmInsertSetMarker", "sceGnmInsertPushColorMarker", "sceGnmInsertSetColorMarker", "sceGnmInsertThreadTraceMarker", "sceGnmInsertDingDongMarker", "sceGnmInsertWaitFlipDone", "sceGnmComputeWaitOnAddress", "sceGnmComputeWaitSemaphore"})
        registerHle(g, name, guestEntry<&writesNop>());
    // Default-state writers return how many dwords they wrote; writing none is valid.
    for (const char* name : {"sceGnmDrawInitDefaultHardwareState350", "sceGnmDrawInitToDefaultContextState400",
                             "sceGnmDispatchInitDefaultHardwareState", "sceGnmIsUserPaEnabled",
                             "sceGnmDriverCaptureInProgress", "sceGnmDriverTraceInProgress", "sceRazorIsLoaded",
                             "sceGnmSetGsRingSizes", "sceGnmSetWaveLimitMultipliers"})
        registerHle(g, name, guestEntry<&returnsZero>());
    registerHle(g, "sceGnmFlushGarlic", guestEntry<&doesNothing>());
}

} // namespace host
