// Vulkan renderer. Every draw and dispatch is recorded as it arrives, with full barriers
// between them, and a batch is submitted when the game's own submission ends. Submission
// does not wait: fences and interrupts the game waits on fire once the GPU has finished
// (see afterGpuWork), with up to kBatches batches in flight.
//
// Guest memory is shared with the GPU by importing it (VK_EXT_external_memory_host), so
// shader buffer reads and writes go straight to the game's memory. Render targets and depth
// buffers live in Vulkan images keyed by their guest address; a texture whose address
// matches one samples that image, any other texture is uploaded from guest memory.
#include "host/gpu/renderer.h"

#include "gcn/shader.h"
#include "host/gpu/formats.h"
#include "host/gpu/registers.h"
#include "host/gpu/tiling.h"
#include "host/gpu/vk.h"
#include "host/pm4.h"
#include "host/profile.h"

#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <thread>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

namespace host::render {

namespace {

// ---- register offsets used here (context window unless noted) --------------------------------
namespace reg {
constexpr uint32_t kDbRenderControl = 0x000, kDbHtileDataBase = 0x005, kDbStencilClear = 0x00A,
                   kDbDepthClear = 0x00B, kDbZInfo = 0x010, kDbStencilInfo = 0x011, kDbZReadBase = 0x012,
                   kDbZWriteBase = 0x014, kDbDepthSize = 0x016;
constexpr uint32_t kScreenScissorTl = 0x00C, kScreenScissorBr = 0x00D, kWindowOffset = 0x080,
                   kWindowScissorTl = 0x081, kWindowScissorBr = 0x082, kGenericScissorTl = 0x090,
                   kGenericScissorBr = 0x091, kVportScissorTl = 0x094, kVportScissorBr = 0x095;
constexpr uint32_t kCbTargetMask = 0x08E, kCbShaderMask = 0x08F, kCbBlendRed = 0x105;
constexpr uint32_t kDbStencilControl = 0x10B, kDbStencilRefMask = 0x10C, kDbStencilRefMaskBf = 0x10D;
constexpr uint32_t kVportXScale = 0x10F;
constexpr uint32_t kSpiPsInputCntl = 0x191, kSpiPsInputEna = 0x1B3, kSpiPsInControl = 0x1B6,
                   kSpiShaderColFormat = 0x1C5;
constexpr uint32_t kCbBlendControl = 0x1E0, kDbDepthControl = 0x200, kCbColorControl = 0x202,
                   kDbShaderControl = 0x203, kPaClClipCntl = 0x204, kPaSuScModeCntl = 0x205, kPaClVteCntl = 0x206;
// PA_SU_POLY_OFFSET_CLAMP / FRONT_SCALE / FRONT_OFFSET (depth bias; BACK_* follow).
constexpr uint32_t kPaSuPolyOffsetClamp = 0x2df, kPaSuPolyOffsetFrontScale = 0x2e0, kPaSuPolyOffsetFrontOffset = 0x2e1;
constexpr uint32_t kCbColorBase = 0x318, kCbColorStride = 0xF;
constexpr uint32_t kCbColorCmask = 7, kCbColorClearWord0 = 11, kCbColorClearWord1 = 12; // within a target's block
// sh window
constexpr uint32_t kPgmLoPs = 0x08, kRsrc1Ps = 0x0A, kRsrc2Ps = 0x0B, kUserDataPs = 0x0C;
constexpr uint32_t kPgmLoVs = 0x48, kRsrc1Vs = 0x4A, kRsrc2Vs = 0x4B, kUserDataVs = 0x4C;
constexpr uint32_t kNumThreadX = 0x207, kPgmLoCs = 0x20C, kRsrc1Cs = 0x212, kRsrc2Cs = 0x213, kUserDataCs = 0x240;
// uconfig window
constexpr uint32_t kPrimitiveType = 0x242;
} // namespace reg

constexpr uint32_t kPsBindingBase = 32;
constexpr VkDeviceSize kRingSize = 64ull << 20;
// Batches in flight. A long submission is cut into several (see kFlushDraws), so the GPU
// starts on a frame while the rest is still being recorded.
constexpr uint32_t kBatches = 4;
constexpr uint64_t kImportGranule = 2ull << 20;

bool readGuest(uint64_t addr, void* dst, size_t size) {
    if (!addr || addr >= (1ull << 40)) return false;
    iovec local{dst, size}, remote{reinterpret_cast<void*>(addr), size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t(size);
}

uint64_t shaderAddress(const GpuRegs& r, uint32_t lo) { return ((uint64_t(r.sh[lo + 1]) << 32) | r.sh[lo]) << 8; }

uint64_t hashBytes(const void* p, size_t n) {
    // FNV-1a over 8-byte words in four independent lanes (a single multiply chain is the
    // bottleneck on large textures); good enough to notice a changed texture.
    const auto* w = static_cast<const uint64_t*>(p);
    const size_t words = n / 8;
    uint64_t h[4] = {0xcbf29ce484222325ull, 0x84222325cbf29ce4ull, 0x9ce484222325cbf2ull, 0x2325cbf29ce48422ull};
    size_t i = 0;
    for (; i + 4 <= words; i += 4)
        for (int l = 0; l < 4; ++l) h[l] = (h[l] ^ w[i + l]) * 0x100000001b3ull;
    for (; i < words; ++i) h[0] = (h[0] ^ w[i]) * 0x100000001b3ull;
    return ((h[0] * 31 + h[1]) * 31 + h[2]) * 31 + h[3];
}

// Hashes guest memory for texture checks: 128 KiB chunks, read into a cache-sized buffer and
// hashed on worker threads, so the textures a frame samples (~100 MB in gameplay) are neither
// copied nor hashed on the renderer thread alone.
// A cheap fingerprint of [addr, addr + size): 64 samples of 64 bytes spread over it (read
// in one syscall). It catches a texture rewritten as a whole (a video frame) between full hashes.
// Hash of guest memory, 0 when it is not readable.
uint64_t memoryHash(uint64_t addr, size_t size) {
    thread_local std::vector<uint8_t> buf;
    buf.resize(size);
    iovec local{buf.data(), size}, remote{reinterpret_cast<void*>(addr), size};
    if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != ssize_t(size)) return 0;
    return hashBytes(buf.data(), size) | 1;
}

uint64_t sampleHash(uint64_t addr, size_t size) {
    constexpr size_t kSample = 64, kMax = 32;
    const size_t stride = std::max<size_t>(kSample, (size / kMax + kSample - 1) & ~(kSample - 1));
    thread_local std::vector<iovec> remote;
    thread_local std::vector<uint8_t> buf(kSample * kMax);
    remote.clear();
    for (size_t off = 0; off + kSample <= size && remote.size() < kMax; off += stride)
        remote.push_back({reinterpret_cast<void*>(addr + off), kSample});
    if (remote.empty()) return 0;
    iovec local{buf.data(), remote.size() * kSample};
    if (process_vm_readv(getpid(), &local, 1, remote.data(), remote.size(), 0) != ssize_t(local.iov_len)) return 0;
    return hashBytes(buf.data(), local.iov_len) | 1;
}

class GuestHasher {
public:
    GuestHasher() {
        const unsigned n = std::clamp(std::thread::hardware_concurrency() / 3, 1u, 6u);
        for (unsigned i = 0; i < n; ++i) std::thread([this] { worker(); }).detach();
    }

    // Hash of [addr, addr + size), or 0 when part of it is unreadable.
    uint64_t hash(uint64_t addr, size_t size) {
        const size_t chunks = (size + kChunk - 1) / kChunk;
        {
            std::unique_lock l(m_);
            idle_.wait(l, [&] { return active_ == 0; }); // no worker is still in the last job
            addr_ = addr;
            size_ = size;
            chunks_ = chunks;
            results_.assign(chunks, 0);
            next_ = 0;
            done_ = 0;
            failed_ = false;
            if (chunks > 2) {
                ++job_;
                wake_.notify_all();
            }
        }
        work();
        std::unique_lock l(m_);
        idle_.wait(l, [&] { return done_ == chunks_; });
        if (failed_) return 0;
        uint64_t h = 0xcbf29ce484222325ull ^ size;
        for (uint64_t r : results_) h = (h ^ r) * 0x100000001b3ull;
        return h | 1;
    }

private:
    static constexpr size_t kChunk = 128 << 10;

    void worker() {
        profile::nameThread("lpn-texhash");
        uint64_t seen = 0;
        std::unique_lock l(m_);
        for (;;) {
            wake_.wait(l, [&] { return job_ != seen; });
            seen = job_;
            ++active_;
            l.unlock();
            work();
            l.lock();
            if (--active_ == 0) idle_.notify_all();
        }
    }

    void work() {
        thread_local std::vector<uint8_t> buf(kChunk);
        for (size_t i; (i = next_.fetch_add(1)) < chunks_;) {
            const size_t off = i * kChunk, n = std::min(kChunk, size_ - off);
            if (readGuest(addr_ + off, buf.data(), n)) results_[i] = hashBytes(buf.data(), n);
            else failed_ = true;
            if (done_.fetch_add(1) + 1 == chunks_) {
                std::lock_guard l(m_);
                idle_.notify_all();
            }
        }
    }

    std::mutex m_;
    std::condition_variable wake_, idle_;
    uint64_t job_ = 0;
    unsigned active_ = 0;
    uint64_t addr_ = 0;
    size_t size_ = 0, chunks_ = 0;
    std::vector<uint64_t> results_;
    std::atomic<size_t> next_{0}, done_{0};
    std::atomic<bool> failed_{false};
};

// GuestHasher::hash on the calling thread alone: the same chunks, the same result.
uint64_t hashGuestSerial(uint64_t addr, size_t size) {
    constexpr size_t kChunk = 128 << 10;
    thread_local std::vector<uint8_t> buf(kChunk);
    uint64_t h = 0xcbf29ce484222325ull ^ size;
    for (size_t off = 0; off < size; off += kChunk) {
        const size_t n = std::min(kChunk, size - off);
        if (!readGuest(addr + off, buf.data(), n)) return 0;
        h = (h ^ hashBytes(buf.data(), n)) * 0x100000001b3ull;
    }
    return h | 1;
}

// Guest reads for shader analysis, a page at a time. Analysis reads instruction by
// instruction, and a syscall each made it the renderer's main cost. Memory a submission
// references cannot change until the GPU is done with it, except where the command processor
// writes it between draws (onMemoryWrite drops those pages), so pages stay valid until the
// submission ends.
// Bytes a GPU copy recorded but not yet run will write: reads see them already, as they would
// on the console where the copy runs before the draws recorded after it. Shared by the page
// caches of all analysis workers.
struct Overlays {
    struct Overlay {
        uint64_t addr;
        std::vector<uint8_t> bytes;
        uint64_t id;
    };
    std::mutex m;
    std::vector<Overlay> list;
    std::atomic<uint64_t> version{0}; // bumped when list changes
    void add(uint64_t addr, std::vector<uint8_t> bytes, uint64_t id) {
        std::lock_guard lock(m);
        list.push_back({addr, std::move(bytes), id});
        version.fetch_add(1, std::memory_order_release);
    }
    void drop(uint64_t id) {
        std::lock_guard lock(m);
        std::erase_if(list, [&](const Overlay& o) { return o.id == id; });
        version.fetch_add(1, std::memory_order_release);
    }
};

class PageCache {
public:
    explicit PageCache(Overlays& o) : overlays_(o) {}
    bool read(uint64_t addr, void* dst, size_t size) {
        if (!readPages(addr, dst, size)) return false;
        // Overlays are few but live across many draws: their ranges are checked without the
        // lock, from a copy taken when they change.
        if (const uint64_t v = overlays_.version.load(std::memory_order_acquire); v != seenVersion_) {
            std::lock_guard lock(overlays_.m);
            ranges_.clear();
            for (const auto& o : overlays_.list) ranges_.emplace_back(o.addr, o.addr + o.bytes.size());
            seenVersion_ = overlays_.version.load(std::memory_order_relaxed);
        }
        for (const auto& [lo, hi] : ranges_)
            if (addr < hi && lo < addr + size) {
                std::lock_guard lock(overlays_.m);
                for (const auto& o : overlays_.list) { // oldest first
                    const uint64_t a = std::max(addr, o.addr), b = std::min(addr + size, o.addr + o.bytes.size());
                    if (a < b) std::memcpy(static_cast<uint8_t*>(dst) + (a - addr), o.bytes.data() + (a - o.addr), b - a);
                }
                break;
            }
        return true;
    }
    void clear() { // after every submission: the page buffers are kept for the next one
        for (auto& [page, data] : pages_)
            if (data) spare_.push_back(std::move(data));
        pages_.clear();
        for (Slot& s : slots_) s = Slot{};
    }
    // Forgets the pages of [addr, addr + size), which changed.
    void invalidate(uint64_t addr, uint64_t size) {
        for (uint64_t page = addr & ~uint64_t(kPage - 1); page < addr + size; page += kPage) {
            if (auto it = pages_.find(page); it != pages_.end()) {
                if (it->second) spare_.push_back(std::move(it->second));
                pages_.erase(it);
            }
            if (Slot& s = slot(page); s.page == page) s = Slot{};
        }
    }

private:
    bool readPages(uint64_t addr, void* dst, size_t size) {
        auto* out = static_cast<uint8_t*>(dst);
        while (size) {
            const uint64_t page = addr & ~uint64_t(kPage - 1);
            const size_t off = size_t(addr - page), n = std::min(size, kPage - off);
            const uint8_t* src = lookup(page);
            if (!src) return readGuest(addr, out, size); // unmapped page or mapping edge: exact read
            std::memcpy(out, src + off, n);
            out += n;
            addr += n;
            size -= n;
        }
        return true;
    }
    Overlays& overlays_;
    uint64_t seenVersion_ = ~0ull; // the reader's copy of the ranges is from this version
    std::vector<std::pair<uint64_t, uint64_t>> ranges_;

    static constexpr size_t kPage = 4096;
    // Recently used pages, direct-mapped in front of the map: a draw's loads hop between a few
    // pages (descriptors, constants), and a map lookup each cost more than the copy.
    struct Slot {
        uint64_t page = ~0ull;
        const uint8_t* data = nullptr;
    };
    static constexpr size_t kSlots = 256;
    Slot& slot(uint64_t page) { return slots_[(page / kPage) & (kSlots - 1)]; }
    const uint8_t* lookup(uint64_t page) {
        Slot& s = slot(page);
        if (s.page == page) return s.data;
        auto it = pages_.find(page);
        if (it == pages_.end()) {
            std::unique_ptr<std::array<uint8_t, kPage>> data;
            if (!spare_.empty()) {
                data = std::move(spare_.back());
                spare_.pop_back();
            } else {
                data = std::make_unique<std::array<uint8_t, kPage>>();
            }
            if (!readGuest(page, data->data(), kPage)) spare_.push_back(std::move(data));
            it = pages_.emplace(page, std::move(data)).first;
        }
        s.page = page;
        return s.data = it->second ? it->second->data() : nullptr;
    }
    std::unordered_map<uint64_t, std::unique_ptr<std::array<uint8_t, kPage>>> pages_; // null: unreadable
    std::vector<std::unique_ptr<std::array<uint8_t, kPage>>> spare_; // page buffers to reuse (a few thousand a frame)
    std::array<Slot, kSlots> slots_{};
};

// Identifies a shader launch: the launch state analysis or translation reads, except user data
// (an analysis is re-targeted to other user data by gcn::refresh). Built in a reused buffer: it
// runs for every launch.
const std::string& launchKey(const gcn::LaunchState& l, uint32_t bindingBase) {
    thread_local std::string k;
    k.clear();
    auto put = [&](const void* p, size_t n) { k.append(static_cast<const char*>(p), n); };
    put(&bindingBase, 4);
    put(&l.stage, 1);
    put(&l.entry, 8);
    put(&l.rsrc1, 4);
    put(&l.rsrc2, 4);
    put(&l.glClipZ, 1);
    if (l.stage == gcn::Stage::Pixel) {
        put(&l.psInputEna, 4);
        put(l.psInputCntl, sizeof(l.psInputCntl));
        put(&l.numInterp, 4);
        put(&l.colorFormat, 4);
        put(&l.colorUint, 4);
        put(&l.colorSint, 4);
        put(l.colorSwap, sizeof(l.colorSwap));
        put(l.colorComps, sizeof(l.colorComps));
        put(&l.depthExport, 1);
    }
    if (l.stage == gcn::Stage::Compute) put(l.localSize, sizeof(l.localSize));
    return k;
}

constexpr VkWriteDescriptorSet descriptorWrite(VkDescriptorType type) {
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.descriptorCount = 1;
    w.descriptorType = type;
    return w;
}

// Hash map keys of a few words, looked up per draw without building a string.
struct WordsHash {
    template <size_t N> size_t operator()(const std::array<uint32_t, N>& a) const {
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t w : a) h = (h ^ w) * 0x100000001b3ull;
        return size_t(h ^ (h >> 32));
    }
};

#define VK_CHECK(x)                                                                                     \
    do {                                                                                                \
        VkResult r_ = (x);                                                                              \
        if (r_ != VK_SUCCESS) std::fprintf(stderr, "[gpu] %s failed: %s\n", #x, vkResultName(r_));       \
    } while (0)

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0, layers = 1, levels = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    std::map<std::array<uint32_t, 7>, VkImageView> views;
};

// A render target or depth buffer at a guest address.
struct Surface {
    Image img;
    uint32_t elementBytes = 4;
    bool depth = false;
    // Memory a render target lived in can later hold a texture the CPU writes (a pool reused
    // for a video frame): such a surface is sampled from memory instead (see resolveImage).
    uint64_t writtenFrame = 0; // last frame the GPU rendered into it
    uint64_t memSample = 0;    // sampleHash of its memory, while the GPU leaves it alone
    bool cpuOwned = false;     // its memory changed meanwhile: the CPU's contents win
    uint64_t checkedFrame = ~0ull;
};

// A texture's full hash, computed off the recorder (see Renderer::guestTexture).
struct AsyncHash {
    std::atomic<int> state{0}; // 0 idle, 1 queued or running, 2 result ready
    uint64_t result = 0;       // 0: unreadable
};

struct GuestTexture {
    Image img;
    std::shared_ptr<AsyncHash> check;
    uint64_t hash = 0;
    uint64_t checkedFrame = ~0ull;
    uint64_t changedFrame = 0; // the contents were last (re)uploaded in this frame
    uint64_t sample = 0;       // sampleHash at the last full check
    uint32_t uses = 0;         // frames it was used in while unchanged
    uint32_t phase = 0;        // which of every 8 uses gets the full check (spread across textures)
};

struct Import {
    uint64_t start = 0, end = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct ShaderModule {
    VkShaderModule module = VK_NULL_HANDLE;
    gcn::Translation translation;
    uint32_t id = 0;
    std::string name; // its SPIR-V cache name: a hash of the GLSL, the same in every run
};

struct BufferRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, range = 0;
    uint32_t byteOffset = 0; // guest base relative to `offset`
};

// An analysed program, reused while its code is unchanged and gcn::refresh can re-target it to
// the launch. The renderer fills resource fields into prog.data per draw.
struct CachedProgram {
    gcn::Program prog;
    std::vector<std::pair<uint64_t, uint64_t>> code; // [start, end) of the code analysed
    uint64_t codeHash = 0;
    uint64_t checkedSubmit = ~0ull; // the submission codeHash was last verified in
    ShaderModule* module = nullptr;
    bool valid = false;
};

// The analyses of one shader launch key, most recently used first: user data or memory can send
// the code down other branches (walks), and a walk can meet resources of other formats (shapes,
// e.g. one material shader with BC1 or BC3 textures), each its own translation. A launch replays
// one trace per walk and then only compares formats.
struct ProgramWalk {
    std::vector<std::unique_ptr<CachedProgram>> shapes; // all with the same trace
};
struct ProgramVariants {
    std::vector<ProgramWalk> walks;
};
constexpr size_t kMaxWalks = 8, kMaxShapes = 32;

// LPN_PROFILE=1: where the renderer's time goes, printed once a second (see host/profile.h).
enum ProfSlot { kProfRenderer, kProfAnalyze, kProfShaderLookup, kProfTexRead, kProfTexUpload, kProfPipeline, kProfGpuWait, kProfBind, kProfLaunch, kProfAnalysisWait, kProfCount };
constexpr const char* kProfNames[kProfCount] = {"renderer", "analyze", "shader-lookup", "tex-read", "tex-upload", "pipeline", "gpu-wait", "bind", "launch", "analysis-wait"};

struct Profiler {
    bool on = false;
    uint64_t ns[kProfCount] = {};
    uint64_t passWhy[4] = {}; // why a new render pass started, by PassWhy
    uint64_t feedbackOn[2] = {}; // a draw sampled the open pass's depth / color attachment
    uint64_t missWhy[4] = {}; // a launch not repeating the stage's last: shader, submission, user data, other
    uint64_t missUser[16] = {}; // the user data words that differed
    uint64_t fastMiss[8] = {}; // a draw not taking fastDraw, by FastMiss
    std::unordered_map<uint32_t, uint64_t> fastCtx; // the context registers that differed
    uint64_t barriers = 0, calls = 0, fastDraws = 0, analysisHits = 0, repeats = 0, analyses = 0, passes = 0, texChecks = 0, texUploads = 0, texBytes = 0, flushes = 0, dmas = 0, dmaBytes = 0, pipelinesMade = 0, pipelineWaits = 0, vramCopies = 0, vramCopyBytes = 0;
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
};

// Why a draw starts a new render pass: other targets, it samples one of the pass's
// attachments, the draw before wrote memory from a shader, or the pass was ended by something
// else (a dispatch, copy, flush or clear).
enum PassWhy { kPassTargets, kPassFeedback, kPassStore, kPassOther };

// Why a draw was recorded in full rather than by fastDraw: the draw before could not be repeated
// (no open pass, a depth clear, it sampled what it renders to), something was recorded in
// between, a new frame or images, another draw kind, shader or primitive, other context
// registers, other resources, or no index buffer.
enum FastMiss { kFastNotKept, kFastBetween, kFastFrame, kFastKind, kFastCtx, kFastResources, kFastIndex, kFastMissCount };
constexpr const char* kFastMissNames[kFastMissCount] = {"not repeatable", "recorded between", "new frame/images/pass",
                                                        "kind/shader/prim", "context regs", "resources", "index buffer"};

class ProfScope {
public:
    ProfScope(Profiler& p, ProfSlot s) : p_(p.on ? &p : nullptr), s_(s) {
        if (p_) t0_ = std::chrono::steady_clock::now();
    }
    ~ProfScope() {
        if (p_)
            p_->ns[s_] += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0_).count());
    }

private:
    Profiler* p_;
    ProfSlot s_;
    std::chrono::steady_clock::time_point t0_;
};

// Vulkan commands recorded on a thread of their own. The recorder thread decides every draw's
// state, resources and pipeline, and queues the Vulkan calls themselves (about a quarter of its
// time went into the driver) as closures in chunks; the command thread runs them in order into
// the batch's command buffer, ends it and submits it. Arrays a call points to are copied into
// the chunk. LPN_SYNC_RECORD=1 runs each call on the recorder as it is made.
class CmdChunk {
public:
    template <typename F> void push(F&& f) {
        using T = std::decay_t<F>;
        void* p = alloc(sizeof(T), alignof(T));
        new (p) T(std::forward<F>(f));
        ops_.push_back({[](void* o) {
                            T& t = *static_cast<T*>(o);
                            t();
                            t.~T();
                        },
                        p});
    }
    template <typename T> T* keep(const T* src, size_t n) {
        if (!n) return nullptr;
        T* d = static_cast<T*>(alloc(sizeof(T) * n, alignof(T)));
        std::memcpy(static_cast<void*>(d), src, sizeof(T) * n);
        return d;
    }
    bool empty() const { return ops_.empty(); }
    void run() {
        for (const Op& o : ops_) o.call(o.obj);
        ops_.clear();
        block_ = 0;
        used_ = 0;
    }

private:
    static constexpr size_t kBlock = 256 << 10;
    void* alloc(size_t size, size_t align) {
        for (;;) {
            if (block_ < blocks_.size()) {
                const size_t at = (used_ + align - 1) & ~(align - 1);
                if (at + size <= blocks_[block_].second) {
                    used_ = at + size;
                    return blocks_[block_].first.get() + at;
                }
                ++block_;
                used_ = 0;
                continue;
            }
            const size_t n = std::max(kBlock, size + align);
            blocks_.emplace_back(std::make_unique<std::byte[]>(n), n);
        }
    }
    struct Op {
        void (*call)(void*);
        void* obj;
    };
    std::vector<Op> ops_;
    std::vector<std::pair<std::unique_ptr<std::byte[]>, size_t>> blocks_;
    size_t block_ = 0, used_ = 0;
};

class Renderer final : public DrawListener {
public:
    bool init() {
        std::string error;
        if (!vk_.init(&error)) {
            std::fprintf(stderr, "[gpu] no renderer: %s\n", error.c_str());
            return false;
        }
        if (!vk_.hostImport) {
            std::fprintf(stderr, "[gpu] no renderer: VK_EXT_external_memory_host is required\n");
            return false;
        }
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = vk_.family;
        VK_CHECK(vkCreateCommandPool(vk_.device, &pci, nullptr, &pool_));
        for (Batch& b : batches_) {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool_;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(vk_.device, &cai, &b.cmd));
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VK_CHECK(vkCreateFence(vk_.device, &fci, nullptr, &b.fence));
            b.ring = createHostBuffer(kRingSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      &b.ringMemory, &b.ringPtr);
            if (!b.ring) return false;
        }
        if (getenv("LPN_PROFILE")) { // GPU time per batch, from timestamps at its start and end
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(vk_.phys, &props);
            gpuTickNs_ = props.limits.timestampPeriod;
            VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qci.queryCount = 2 * kBatches;
            if (vkCreateQueryPool(vk_.device, &qci, nullptr, &timestamps_) != VK_SUCCESS) timestamps_ = VK_NULL_HANDLE;
        }
        useBatch(0);
        dummyBuffer_ = createHostBuffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dummyMemory_, nullptr);
        if (!dummyBuffer_) return false;
        sync_ = getenv("LPN_GPU_SYNC") != nullptr;
        std::thread([this] { completeBatches(); }).detach();
        if (!syncRecord_) std::thread([this] { commandThread(); }).detach();
        if (asyncTexChecks_)
            for (int i = 0; i < 3; ++i) std::thread([this] { texHashWorker(); }).detach();
        storageAlign_ = std::max<VkDeviceSize>(vk_.props.limits.minStorageBufferOffsetAlignment, 16);
        if (const char* t = getenv("LPN_GPU_TRACE")) traceFrame_ = std::strtoull(t, nullptr, 0);
        if (const char* sn = getenv("LPN_GPU_SNAP")) {
            // "1": every draw; otherwise a list of draw numbers and ranges ("2752,2860-2870").
            snap_ = true;
            if (const char* sc = getenv("LPN_GPU_SNAP_SCALE")) snapScale_ = std::max(1, std::atoi(sc));
            if (std::strcmp(sn, "1") != 0) {
                for (const char* q = sn; *q;) {
                    char* end;
                    const uint64_t a = std::strtoull(q, &end, 10);
                    uint64_t b = a;
                    if (*end == '-') b = std::strtoull(end + 1, &end, 10);
                    snapDraws_.emplace_back(a, b);
                    q = *end ? end + 1 : end;
                }
            }
        }
        prof_.on = aprof_.on = getenv("LPN_PROFILE") != nullptr;
        {
            const char* w = getenv("LPN_ANALYSIS_WORKERS");
            analysisWorkers_ = w ? unsigned(std::atoi(w)) : std::thread::hardware_concurrency() >= 8 ? 3 : 1;
            analysisWorkers_ = std::min(analysisWorkers_, 16u);
            for (unsigned i = 0; i < std::max(analysisWorkers_, 1u); ++i) {
                actx_.push_back(std::make_unique<AnalysisCtx>(overlays_));
                actx_.back()->prof.on = prof_.on;
            }
            for (unsigned i = 0; i < analysisWorkers_; ++i)
                std::thread([this, c = actx_[i].get()] { analysisWorker(*c); }).detach();
            pending_.reserve(kAnalysisBatch);
        }
        openPipelineCache();
        startPipelineWorkers();
        dumpGlsl_ = getenv("LPN_GPU_GLSL") != nullptr;
        return true;
    }

    // ---- the analysis thread (see pm4.h): shader analysis for each draw, ahead of recording ----
    // It reads each draw's launches from the registers and hands them, in batches of consecutive
    // draws, to analysis workers (LPN_ANALYSIS_WORKERS, default 3; 0 analyses on this thread).
    // One thread replaying every launch capped crowded views: looking up the steps after New
    // Game, ~6,350 draws kept it 94% busy at ~59 fps. Each worker keeps its own analyses, page
    // cache and memo (a cached program is re-targeted in place); shader modules, pipelines and
    // the overlays are shared. Memory changes only between batches: the thread waits for the
    // workers before it applies one (see analysisBarrier). The recorder takes each draw's
    // DrawAnalysis from analyses_ in order, once a worker has filled it in.

    void onAnalyzeDraw(const DrawCall& d, const GpuRegs& r) override {
        DrawAnalysis* a;
        {
            // A spare one keeps its vectors' storage: no allocations per draw here, or frees on the recorder.
            std::lock_guard lock(analysesMutex_);
            if (!spareAnalyses_.empty()) {
                analyses_.push_back(std::move(spareAnalyses_.back()));
                spareAnalyses_.pop_back();
            } else {
                analyses_.emplace_back();
            }
            a = &analyses_.back(); // stays put while later ones are added (a deque)
            a->ready = false;
        }
        a->ok = a->repeat = false;
        a->vsm = a->psm = nullptr;
        a->vsc = a->psc = nullptr;
        pending_.emplace_back();
        DrawJob& j = pending_.back();
        j.out = a;
        j.kind = DrawJob::kNone;
        if (d.kind == DrawCall::kDispatch) prepareDispatch(r, j);
        else if (!computeOnly_ && (d.kind == DrawCall::kAuto || d.kind == DrawCall::kIndexed)) prepareDraw(r, j);
        if (pending_.size() >= kAnalysisBatch) submitBatch();
    }
    void onAnalyzeFlush() override { submitBatch(); }

    // Memory the command processor writes reaches guest memory only when the recorder (or the
    // GPU) gets there: until then analysis reads it through an overlay on its page caches.
    void onAnalyzeDma(uint64_t dst, uint64_t src, uint64_t bytes, bool fill, uint32_t value) override {
        analysisBarrier();
        std::vector<uint8_t> data(bytes);
        bool ok = true;
        if (fill) for (uint64_t i = 0; i + 4 <= bytes; i += 4) std::memcpy(&data[i], &value, 4);
        else ok = actx_[0]->pages.read(src, data.data(), bytes);
        for (auto& c : actx_) c->pages.invalidate(dst, bytes);
        ++submit_; // and re-verify shader code before reuse
        uint64_t id = 0;
        if (ok) overlays_.add(dst, std::move(data), id = ++overlayIds_);
        std::lock_guard lock(analysesMutex_);
        dmaOverlays_.push_back(id);
    }
    void onAnalyzeWrite(uint64_t addr, const uint32_t* values, uint32_t count) override {
        analysisBarrier();
        std::vector<uint8_t> data(size_t(count) * 4);
        std::memcpy(data.data(), values, data.size());
        for (auto& c : actx_) c->pages.invalidate(addr, data.size());
        ++submit_;
        const uint64_t id = ++overlayIds_;
        overlays_.add(addr, std::move(data), id);
        std::lock_guard lock(analysesMutex_);
        writeOverlays_.push_back(id);
    }
    void onAnalyzeSubmitEnd() override {
        analysisBarrier();
        for (auto& c : actx_) c->pages.clear();
        ++submit_;
    }
    void onAnalyzeFrameEnd() override {
        analysisBarrier();
        ++submit_;
        for (auto& c : actx_) {
            c->pages.clear();
            if (c->programCount > 20000) {
                c->programs.clear();
                c->programCount = 0;
                for (LastLaunch& l : c->last) l = LastLaunch{};
            }
        }
    }

    // Hands the pending draws to a worker (or analyses them here without workers).
    void submitBatch() {
        if (pending_.empty()) return;
        if (analysisWorkers_ == 0) {
            runBatch(*actx_[0], pending_);
            pending_.clear();
            return;
        }
        {
            std::lock_guard lock(batchMutex_);
            analysisBatches_.push_back(std::move(pending_));
            if (!spareBatches_.empty()) {
                pending_ = std::move(spareBatches_.back());
                spareBatches_.pop_back();
            } else {
                pending_ = {};
            }
        }
        batchCv_.notify_one();
        pending_.reserve(kAnalysisBatch);
    }
    // Waits until the workers have analysed every draw so far: memory is about to change.
    void analysisBarrier() {
        submitBatch();
        if (analysisWorkers_ == 0) return;
        std::unique_lock lock(batchMutex_);
        batchIdleCv_.wait(lock, [&] { return analysisBatches_.empty() && batchesBusy_ == 0; });
    }
    void onDraw(const DrawCall& d, const GpuRegs& r) override {
        DrawAnalysis a;
        {
            std::unique_lock lock(analysesMutex_);
            if (analyses_.empty()) {
                warnOnce("a draw arrived without its analysis");
                return;
            }
            if (!analyses_.front().ready) {
                ProfScope pw(prof_, kProfAnalysisWait);
                analysesCv_.wait(lock, [&] { return analyses_.front().ready; });
            }
            a = std::move(analyses_.front());
            analyses_.pop_front();
        }
        struct Recycle { // hands `a` back to the analysis thread however the draw ends
            Renderer& r;
            DrawAnalysis& a;
            ~Recycle() {
                std::lock_guard lock(r.analysesMutex_);
                // As many as can be in flight: analysis runs up to a frame ahead, and at ~6,000
                // draws a frame a smaller pool left most draws allocating a fresh one (12% of the
                // analysis thread in malloc) and the recorder freeing it (6% in ~Program).
                if (r.spareAnalyses_.size() < 16384) r.spareAnalyses_.push_back(std::move(a));
            }
        } recycle{*this, a};
        std::lock_guard lock(m_);
        ProfScope ps(prof_, kProfRenderer);
        ++prof_.calls;
        ++drawIndex_;
        trace_ = frame_ == traceFrame_;
        if (trace_)
            std::fprintf(traceOut_, "[trace] #%" PRIu64 " %s vs 0x%" PRIx64 " ps 0x%" PRIx64 " cs 0x%" PRIx64 "\n", drawIndex_,
                         d.kind == DrawCall::kDispatch ? "dispatch" : "draw", shaderAddress(r, reg::kPgmLoVs),
                         shaderAddress(r, reg::kPgmLoPs), shaderAddress(r, reg::kPgmLoCs));
        if (raceCheck_) raceShader_ = shaderAddress(r, d.kind == DrawCall::kDispatch ? reg::kPgmLoCs : reg::kPgmLoPs);
        const uint64_t passesBefore = prof_.passes;
        if (d.kind == DrawCall::kDispatch) dispatch(d, r, a);
        else if (computeOnly_) return;
        else if (d.kind == DrawCall::kAuto || d.kind == DrawCall::kIndexed) draw(d, r, a);
        if (keepDraw_) { // the next draw may repeat this one (see fastDraw)
            keepDraw_ = false;
            std::swap(a, prevDraw_);
        }
        else warnOnce("indirect draws and dispatches are not rendered yet");
        if (frame_ == timingFrame_ && timing_ && recording_ && timingMeta_.size() + 1 < kTimingQueries) {
            rec([c = cmd_, q = timing_, i = uint32_t(timingMeta_.size() + 1)] {
                vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q, i);
            });
            timingMeta_.push_back({drawIndex_, shaderAddress(r, d.kind == DrawCall::kDispatch ? reg::kPgmLoCs : reg::kPgmLoPs),
                                   d.kind == DrawCall::kDispatch, prof_.passes != passesBefore, d.count});
        }
        if (++drawsSinceHandOff_ >= 32) handOff(); // the command thread trails by a few draws
        static const uint32_t flushDraws = getenv("LPN_FLUSH_DRAWS") ? std::atoi(getenv("LPN_FLUSH_DRAWS")) : 400;
        if (flushDraws && recording_ && ++drawsSinceFlush_ >= flushDraws) flush();
    }

    // The command processor copies descriptors and constants between draws (DMA_DATA): draws
    // after it must not see the page cache's older copy.
    void onMemoryWrite(uint64_t addr, uint64_t size) override {
        std::lock_guard lock(m_);
        fast_.valid = false;
        if (raceCheck_) noteWrite(addr, size);
        cpWrites(addr, size);
    }
    void onCpWrite(uint64_t, uint64_t) override { // written now: the analysis overlay is done
        uint64_t id;
        {
            std::lock_guard lock(analysesMutex_);
            if (writeOverlays_.empty()) return;
            id = writeOverlays_.front();
            writeOverlays_.pop_front();
        }
        overlays_.drop(id);
    }

    bool onDmaCopy(uint64_t dst, uint64_t src, uint64_t bytes, bool fill, uint32_t value) override {
        fast_.valid = false; // memory changes (recorder thread, like fastDraw)
        uint64_t id = 0; // the analysis thread's overlay for this copy
        {
            std::lock_guard lock(analysesMutex_);
            if (!dmaOverlays_.empty()) {
                id = dmaOverlays_.front();
                dmaOverlays_.pop_front();
            }
        }
        // Copied by the CPU right after a false return: the overlay can go at once.
        auto cpu = [&] {
            if (id) overlays_.drop(id);
            return false;
        };
        static const bool cpuDma = getenv("LPN_CPU_DMA") != nullptr; // the old way, for comparison
        if (cpuDma) return cpu();
        std::lock_guard lock(m_);
        VkBuffer db, sb = VK_NULL_HANDLE;
        VkDeviceSize doff, soff = 0;
        if (!guestBuffer(dst, bytes, db, doff)) return cpu();
        if (fill) {
            if (dst % 4 || bytes % 4) return cpu();
        } else {
            if (src < dst + bytes && dst < src + bytes) return cpu(); // overlapping: copied by the CPU
            if (!guestBuffer(src, bytes, sb, soff)) return cpu();
        }
        begin();
        fullBarrier();
        if (fill) {
            rec([c = cmd_, db, doff, bytes, value] { vkCmdFillBuffer(c, db, doff, bytes, value); });
        } else {
            const VkBufferCopy bc{soff, doff, bytes};
            rec([c = cmd_, sb, db, bc] { vkCmdCopyBuffer(c, sb, db, 1, &bc); });
        }
        if (raceCheck_) noteWrite(dst, bytes);
        gpuWrites(dst, bytes);
        ++prof_.dmas, prof_.dmaBytes += bytes;
        if (id) pendingDone_.push_back([this, id] { overlays_.drop(id); }); // once the GPU has copied
        return true;
    }

    void onWait() override {
        std::lock_guard lock(m_);
        flush();
    }

    void onSubmitEnd() override {
        std::lock_guard lock(m_);
        ProfScope ps(prof_, kProfRenderer);
        flush();
    }

    void onFrameEnd() override {
        std::lock_guard lock(m_);
        {
            ProfScope ps(prof_, kProfRenderer);
            flush();
        }
        if (frame_ == traceFrame_) {
            dumpSurfaces();
            if (traceOut_ != stderr) {
                std::fclose(traceOut_);
                traceOut_ = stderr;
                std::fprintf(stderr, "[capture] saved %s\n", traceDir_.c_str());
            }
        }
        if (prof_.on) reportSlowFrame();
        savePipelineCache();
        ++frame_;
        if (!pendingCapture_.empty()) { // F12: trace this frame into the capture folder
            traceDir_ = pendingCapture_ + "/";
            captured_.clear();
            pendingCapture_.clear();
            traceFrame_ = frame_;
            timingFrame_ = frame_ + 1; // and times the one after it (see reportTiming)
            traceOut_ = std::fopen((traceDir_ + "trace.txt").c_str(), "w");
            if (!traceOut_) traceOut_ = stderr;
        }
        if (prof_.on && std::chrono::steady_clock::now() - prof_.windowStart >= std::chrono::seconds(1)) reportProfile();
        if (frame_ % 300 == 0) std::erase_if(vramCandidates_, [&](const auto& c) { return c.second + 60 < frame_; });
        if (frame_ == timingFrame_ + 1) reportTiming();
        if (frame_ == timingFrame_) startTiming();
        drawIndex_ = 0;
    }

    // A frame over 50 ms (a visible hitch): where its time went.
    void reportSlowFrame() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - lastFrameEnd_).count();
        if (frame_ > 0) ++windowFrames_, worstFrameMs_ = std::max(worstFrameMs_, ms), slowFrames_ += ms > 17;
        if (ms > 50 && frame_ > 0) {
            std::fprintf(stderr, "[prof] slow frame %" PRIu64 ": %.1f ms;", frame_, ms);
            for (int i = 0; i < kProfCount; ++i) std::fprintf(stderr, " %s %.1f", kProfNames[i], (prof_.ns[i] - frameNs_[i]) / 1e6);
            std::fprintf(stderr, "; %" PRIu64 " analyses, %" PRIu64 " pipelines made\n", prof_.analyses - frameAnalyses_,
                         prof_.pipelinesMade - framePipelines_);
        }
        lastFrameEnd_ = now;
        std::copy(std::begin(prof_.ns), std::end(prof_.ns), frameNs_);
        frameAnalyses_ = prof_.analyses;
        framePipelines_ = prof_.pipelinesMade;
    }

    // LPN_GPU_TIMING=<frame>, or the frame after an F12 capture: GPU time of every draw and
    // dispatch in that frame (a timestamp after each), summed by shader, printed to the log.
    static constexpr uint32_t kTimingQueries = 16384;
    struct TimingMeta {
        uint64_t draw, shader;
        bool dispatch, newPass;
        uint32_t count;
        bool batchStart = false; // not a draw: the start of a batch, after the GPU idled for it
    };
    uint64_t timingFrame_ = getenv("LPN_GPU_TIMING") ? std::strtoull(getenv("LPN_GPU_TIMING"), nullptr, 0) : ~0ull;
    VkQueryPool timing_ = VK_NULL_HANDLE;
    std::vector<TimingMeta> timingMeta_;
    void startTiming() {
        begin(); // before the pool exists, so begin() writes no timestamp into it yet
        endPass();
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kTimingQueries;
        if (vkCreateQueryPool(vk_.device, &qci, nullptr, &timing_) != VK_SUCCESS) return;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(vk_.phys, &props);
        gpuTickNs_ = props.limits.timestampPeriod;
        rec([c = cmd_, q = timing_] {
            vkCmdResetQueryPool(c, q, 0, kTimingQueries);
            vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q, 0);
        });
        timingMeta_.clear();
    }
    void reportTiming() {
        if (!timing_) return;
        waitIdle();
        const uint32_t n = uint32_t(timingMeta_.size()) + 1;
        std::vector<uint64_t> t(n);
        if (vkGetQueryPoolResults(vk_.device, timing_, 0, n, n * 8, t.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
            return;
        struct Row { double ms; size_t i; };
        std::vector<Row> rows;
        std::map<uint64_t, std::pair<double, uint32_t>> byShader;
        double total = 0, passMs = 0, idle = 0;
        uint32_t passes = 0, batches = 0;
        for (size_t i = 0; i < timingMeta_.size(); ++i) {
            const double ms = t[i + 1] >= t[i] ? double(t[i + 1] - t[i]) * gpuTickNs_ / 1e6 : 0;
            if (timingMeta_[i].batchStart) { // the gap since the previous batch ended
                idle += ms, ++batches;
                continue;
            }
            total += ms;
            if (timingMeta_[i].newPass) passMs += ms, ++passes;
            rows.push_back({ms, i});
            auto& b = byShader[timingMeta_[i].shader];
            b.first += ms, ++b.second;
        }
        std::fprintf(stderr, "[timing] frame %" PRIu64 ": %zu draws/dispatches, %.2f ms of GPU work; draws that start a pass (%u): %.2f ms;"
                             " GPU idle between %u batches: %.2f ms\n",
                     timingFrame_, timingMeta_.size() - batches, total, passes, passMs, batches, idle);
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ms > b.ms; });
        for (size_t k = 0; k < std::min<size_t>(25, rows.size()); ++k) {
            const TimingMeta& m = timingMeta_[rows[k].i];
            std::fprintf(stderr, "[timing]   %.3f ms  #%" PRIu64 " %s 0x%" PRIx64 " count %u%s\n", rows[k].ms, m.draw,
                         m.dispatch ? "cs" : "ps", m.shader, m.count, m.newPass ? " (new pass)" : "");
        }
        std::vector<std::pair<double, uint64_t>> shaders;
        for (auto& [sh, v] : byShader) shaders.push_back({v.first, sh});
        std::sort(shaders.rbegin(), shaders.rend());
        for (size_t k = 0; k < std::min<size_t>(15, shaders.size()); ++k)
            std::fprintf(stderr, "[timing]   shader 0x%" PRIx64 ": %.2f ms in %u uses\n", shaders[k].second, shaders[k].first,
                         byShader[shaders[k].second].second);
        vkDestroyQueryPool(vk_.device, timing_, nullptr);
        timing_ = VK_NULL_HANDLE;
    }

    void captureNextFrame(const std::string& dir) {
        std::lock_guard lock(m_);
        pendingCapture_ = dir;
    }

    // The analysis thread's slots and counts join the recorder's (a little racy, for a report).
    // Analysis time sums the workers' (CPU time, not wall time, with several).
    void mergeAnalysisProfile() {
        auto merge = [&](Profiler& a) {
            for (int i : {kProfAnalyze, kProfShaderLookup, kProfLaunch}) prof_.ns[i] += std::exchange(a.ns[i], 0);
            prof_.analysisHits += std::exchange(a.analysisHits, 0);
            prof_.repeats += std::exchange(a.repeats, 0);
            for (int i = 0; i < 4; ++i) prof_.missWhy[i] += std::exchange(a.missWhy[i], 0);
            for (int i = 0; i < 16; ++i) prof_.missUser[i] += std::exchange(a.missUser[i], 0);
            prof_.analyses += std::exchange(a.analyses, 0);
            prof_.pipelinesMade += std::exchange(a.pipelinesMade, 0);
        };
        merge(aprof_);
        for (auto& c : actx_) merge(c->prof);
    }

    void reportProfile() {
        mergeAnalysisProfile();
        const auto now = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - prof_.windowStart).count();
        const double n = double(std::max<uint64_t>(windowFrames_, 1));
        std::fprintf(stderr, "[prof] t=%.1f frames %" PRIu64 "-%" PRIu64 ": %.1f fps, %.1f ms/frame, worst %.1f ms, %" PRIu64 " over 17 ms;",
                     profile::seconds(), frame_ - windowFrames_, frame_, n / wall, wall * 1000 / n, worstFrameMs_, slowFrames_);
        for (int i = 0; i < kProfCount; ++i) std::fprintf(stderr, " %s %.2f", kProfNames[i], prof_.ns[i] / 1e6 / n);
        std::fprintf(stderr, " passes for targets %.0f feedback %.0f (depth %.0f color %.0f) store %.0f other %.0f;",
                     prof_.passWhy[0] / n, prof_.passWhy[1] / n, prof_.feedbackOn[0] / n,
                     prof_.feedbackOn[1] / n, prof_.passWhy[2] / n, prof_.passWhy[3] / n);
        std::fprintf(stderr, " gpu-busy %.2f dma %.0f (%.2f MB)", gpuBusyNs_.exchange(0) / 1e6 / n, prof_.dmas / n,
                     prof_.dmaBytes / n / 1e6);
        if (raceCheck_) std::fprintf(stderr, " races %" PRIu64 "/%" PRIu64, raceChanged_.load(), raceChecked_.exchange(0));
        std::fprintf(stderr, " (ms/frame); %.0f draws (%.0f fast), %.0f analysis hits (%.0f repeats), %.0f analyses, %.0f tex checks, %.1f uploads (%.1f MB), %.1f flushes, %.0f render passes, %.0f barriers per frame, %" PRIu64 " pipelines made, %" PRIu64 " blended draws skipped for one; VRAM buffers %.0f MB, %.1f copies (%.2f MB) per frame\n",
                     prof_.calls / n, prof_.fastDraws / n, prof_.analysisHits / n, prof_.repeats / n, prof_.analyses / n, prof_.texChecks / n, prof_.texUploads / n, prof_.texBytes / n / 1e6, prof_.flushes / n, prof_.passes / n,
                     prof_.barriers / n, prof_.pipelinesMade, prof_.pipelineWaits, vramBytes_ / 1e6, prof_.vramCopies / n,
                     prof_.vramCopyBytes / n / 1e6);
        std::fprintf(stderr, "[prof] t=%.1f launches not repeating the last: other shader %.0f, new submission %.0f, user data %.0f, other %.0f per frame; user data words differing:",
                     profile::seconds(), prof_.missWhy[0] / n, prof_.missWhy[1] / n, prof_.missWhy[2] / n, prof_.missWhy[3] / n);
        for (int i = 0; i < 16; ++i)
            if (prof_.missUser[i]) std::fprintf(stderr, " s%d %.0f", i, prof_.missUser[i] / n);
        std::fprintf(stderr, "\n[prof] t=%.1f draws recorded in full:", profile::seconds());
        for (int i = 0; i < kFastMissCount; ++i) std::fprintf(stderr, " %s %.0f", kFastMissNames[i], prof_.fastMiss[i] / n);
        if (!prof_.fastCtx.empty()) {
            std::vector<std::pair<uint64_t, uint32_t>> regs;
            for (auto [reg, count] : prof_.fastCtx) regs.push_back({count, reg});
            std::sort(regs.rbegin(), regs.rend());
            std::fprintf(stderr, "; context regs differing:");
            for (size_t i = 0; i < regs.size() && i < 8; ++i)
                std::fprintf(stderr, " 0x%x %.0f", 0xA000 + regs[i].second, regs[i].first / n);
        }
        std::fprintf(stderr, " per frame\n");
        windowFrames_ = slowFrames_ = 0;
        worstFrameMs_ = 0;
        const bool on = prof_.on;
        std::fill(std::begin(frameNs_), std::end(frameNs_), 0);
        frameAnalyses_ = framePipelines_ = 0;
        prof_ = Profiler{};
        prof_.on = on;
        prof_.windowStart = now;
    }

    // Called from the display thread. Only recording the copy holds the renderer; waiting for
    // the GPU does not, so the game keeps submitting meanwhile.
    // Records a copy of the image at display buffer addr into readback slot c.slot and submits it.
    // The copy sits in GPU order with the game's work, so the caller can wait for it elsewhere.
    bool queueDisplay(uint64_t addr, uint32_t width, uint32_t height, DisplayCopy& c) {
        std::lock_guard lock(m_);
        auto it = colors_.find(addr);
        if (it == colors_.end()) {
            warnOnce("display buffer 0x" + hex(addr) + " was not rendered to");
            return false;
        }
        Surface& s = it->second;
        if (s.elementBytes != 4 || c.slot < 0 || c.slot >= kDisplaySlots) return false;
        c.w = std::min(width, s.img.width), c.h = std::min(height, s.img.height);
        DisplaySlot& d = displays_[c.slot];
        const VkDeviceSize bytes = VkDeviceSize(c.w) * c.h * 4;
        if (bytes > d.size) {
            if (d.buf) {
                waitIdle(); // an older copy may still be writing it
                vkDestroyBuffer(vk_.device, d.buf, nullptr);
                vkFreeMemory(vk_.device, d.mem, nullptr);
            }
            d.buf = createHostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &d.mem, &d.ptr, true);
            d.size = d.buf ? bytes : 0;
            if (!d.buf) return false;
        }
        // Mid-submission, the copy rides in the batch being recorded: flushing it here would split
        // the game's frame and leave the recorder waiting on its own first half.
        const bool riding = recording_;
        begin();
        fullBarrier();
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {c.w, c.h, 1};
        rec([c = cmd_, im = s.img.image, buf = d.buf, r] { vkCmdCopyImageToBuffer(c, im, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &r); });
        if (!riding) flush();
        std::lock_guard dl(doneMutex_);
        c.seq = riding ? submitted_ + 1 : submitted_;
        return true;
    }

    // The batch holding everything recorded so far, and whether the GPU has finished it.
    uint64_t gpuMark() {
        std::lock_guard lock(m_);
        std::lock_guard dl(doneMutex_);
        return recording_ ? submitted_ + 1 : submitted_;
    }
    bool gpuDone(uint64_t mark) {
        std::lock_guard dl(doneMutex_);
        return completed_ >= mark;
    }

    // Waits for a queued copy and returns its pixels as linear rows of `width`.
    void fetchDisplay(const DisplayCopy& c, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels) {
        {
            std::unique_lock lock(doneMutex_);
            doneCv_.wait(lock, [&] { return completed_ >= c.seq; });
        }
        const auto* src = static_cast<const uint8_t*>(displays_[c.slot].ptr);
        pixels.assign(size_t(width) * height, 0);
        for (uint32_t y = 0; y < c.h; ++y) std::memcpy(&pixels[size_t(y) * width], src + size_t(y) * c.w * 4, c.w * 4);
    }

private:
    // Debugging: a render target as it is after the current draw.
    void snapshot(uint64_t addr) {
        auto it = colors_.find(addr);
        if (it == colors_.end() || it->second.elementBytes != 4) return;
        std::vector<uint32_t> px;
        const uint32_t w = it->second.img.width, h = it->second.img.height;
        if (!readSurface(addr, w, h, px)) return;
        const std::string name = traceDir_ + "gpu-draw-" + std::to_string(drawIndex_) + ".ppm";
        if (FILE* f = std::fopen(name.c_str(), "wb")) {
            const uint32_t s = snapScale_; // LPN_GPU_SNAP_SCALE: keep every s-th pixel and row
            std::fprintf(f, "P6\n%u %u\n255\n", w / s, h / s);
            for (uint32_t y = 0; y + s <= h; y += s)
                for (uint32_t x = 0; x + s <= w; x += s) {
                    const uint32_t p = px[size_t(y) * w + x];
                    const uint8_t rgb[3] = {uint8_t(p), uint8_t(p >> 8), uint8_t(p >> 16)};
                    std::fwrite(rgb, 1, 3, f);
                }
            std::fclose(f);
        }
    }

    // Debugging: every 32-bit render target as gpu-rt-<address>.ppm (memory order BGRA assumed).
    // F12 captures: each traced draw's shaders (GLSL), the data its shaders get (D[]) and the raw
    // bytes of its small textures and buffers, so a draw can be worked through offline.
    std::set<std::string> captured_;
    void captureProgram(const gcn::Program& p, const ShaderModule* m, const char* stage) {
        if (traceDir_.empty() || !m) return;
        auto save = [&](const std::string& name, const void* data, size_t size) {
            if (!captured_.insert(name).second) return;
            if (FILE* f = std::fopen((traceDir_ + name).c_str(), "wb")) {
                std::fwrite(data, 1, size, f);
                std::fclose(f);
            }
        };
        const std::string& g = m->translation.glsl;
        save(std::string(stage) + "-" + hex(p.launch.entry) + "-" + std::to_string(m->id) + ".glsl", g.data(), g.size());
        { // the machine code (lpn-gcn disassembles it; it stops at the OrbShdr block)
            std::vector<uint8_t> code(64 << 10);
            size_t n = code.size();
            while (n >= 4096 && !readGuest(p.launch.entry, code.data(), n)) n /= 2;
            if (n >= 4096) save(std::string(stage) + "-" + hex(p.launch.entry) + ".bin", code.data(), n);
        }
        save("draw-" + std::to_string(drawIndex_) + "-" + stage + "-data.bin", p.data.data(), p.data.size() * 4);
        std::vector<uint8_t> bytes;
        for (const auto& b : p.buffers)
            std::fprintf(traceOut_, "[trace]   %s buffer 0x%" PRIx64 " stride %u records %u pc 0x%" PRIx64 "%s%s\n", stage, b.base(),
                         b.stride(), b.numRecords(), b.pc, b.store ? " (written)" : "", b.known ? "" : " (unknown)");
        for (const auto& b : p.buffers) {
            const uint64_t size = b.stride() ? uint64_t(b.stride()) * b.numRecords() : b.numRecords();
            if (!b.known || !size) continue;
            if (size > (4u << 20)) { // a big buffer (a ring, say): the 256 KB the draw's base points at
                bytes.resize(256u << 10);
                if (readGuest(b.base(), bytes.data(), bytes.size()))
                    save("bufwin-" + hex(b.base()) + "-stride" + std::to_string(b.stride()) + ".bin", bytes.data(), bytes.size());
                continue;
            }
            bytes.resize(size);
            if (readGuest(b.base(), bytes.data(), size))
                save("buf-" + hex(b.base()) + "-stride" + std::to_string(b.stride()) + "-n" + std::to_string(b.numRecords()) + ".bin",
                     bytes.data(), size);
        }
        for (const auto& im : p.images) {
            const uint32_t bpe = bytesPerElement(im.dataFormat());
            const size_t size = size_t((im.pitch() + 7) & ~7u) * ((im.height() + 7) & ~7u) * std::max(bpe, 1u);
            if (!bpe || size > (1u << 20)) continue;
            bytes.resize(size);
            if (readGuest(im.base(), bytes.data(), size))
                save("tex-" + hex(im.base()) + "-" + std::to_string(im.width()) + "x" + std::to_string(im.height()) + "-pitch" +
                         std::to_string(im.pitch()) + "-fmt" + std::to_string(im.dataFormat()) + "." + std::to_string(im.numFormat()) +
                         "-tile" + std::to_string(im.tileMode()) + "-sel" + std::to_string(im.dstSel()) + ".bin",
                     bytes.data(), size);
        }
    }

    void dumpSurfaces() {
        for (const auto& [addr, s] : colors_) {
            if (s.elementBytes != 4) continue;
            std::vector<uint32_t> px;
            if (!readSurface(addr, s.img.width, s.img.height, px)) continue;
            if (s.img.format == VK_FORMAT_R32_SFLOAT) {
                float lo = INFINITY, hi = -INFINITY, first;
                double sum = 0;
                std::memcpy(&first, px.data(), 4);
                for (uint32_t p : px) {
                    float f;
                    std::memcpy(&f, &p, 4);
                    lo = std::min(lo, f), hi = std::max(hi, f), sum += f;
                }
                std::fprintf(stderr, "[gpu] r32f 0x%" PRIx64 " first %g min %g max %g mean %g\n", addr, first, lo, hi,
                             sum / px.size());
            }
            const std::string name = traceDir_ + "gpu-rt-" + hex(addr) + ".ppm";
            FILE* f = std::fopen(name.c_str(), "wb");
            if (!f) continue;
            std::fprintf(f, "P6\n%u %u\n255\n", s.img.width, s.img.height);
            for (uint32_t p : px) {
                const uint8_t rgb[3] = {uint8_t(p), uint8_t(p >> 8), uint8_t(p >> 16)};
                std::fwrite(rgb, 1, 3, f);
            }
            std::fclose(f);
        }
        std::fprintf(stderr, "[gpu] dumped %zu render targets\n", colors_.size());
    }

    bool readSurface(uint64_t addr, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels) {
        auto it = colors_.find(addr);
        if (it == colors_.end()) {
            warnOnce("display buffer 0x" + hex(addr) + " was not rendered to");
            return false;
        }
        Surface& s = it->second;
        if (s.elementBytes != 4) return false;
        const uint32_t w = std::min(width, s.img.width), h = std::min(height, s.img.height);
        const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
        if (bytes > kRingSize) return false;
        flush();
        begin();
        fullBarrier();
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {w, h, 1};
        const VkDeviceSize at = ringAlloc(bytes, 16);
        c.bufferOffset = at;
        rec([cb = cmd_, im = s.img.image, buf = ring_, c] { vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &c); });
        const auto* src = static_cast<const uint8_t*>(ringPtr_) + at;
        waitIdle();
        pixels.assign(size_t(width) * height, 0);
        for (uint32_t y = 0; y < h; ++y) std::memcpy(&pixels[size_t(y) * width], src + size_t(y) * w * 4, w * 4);
        return true;
    }

    // ---- recording -----------------------------------------------------------------------------

    // Work is recorded into one of kBatches batches (command buffer, fence, upload ring). A flush
    // submits the current one without waiting and moves to the next, which first has to
    // finish its previous submission; a completion thread waits on the fences and then runs
    // what was deferred until the GPU is done (fences the game polls, interrupts, deletions).
    // LPN_GPU_SYNC=1 waits after every submission instead.
    // Queues a Vulkan call (a closure over values, given the batch's command buffer) for the
    // command thread.
    template <typename F> void rec(F&& f) {
        ++recCount_;
        if (syncRecord_) f();
        else chunk_->push(std::forward<F>(f));
    }
    // Storage for an array a queued call points to, valid until the call has run.
    template <typename T> const T* keep(const T* p, size_t n) { return syncRecord_ ? p : chunk_->keep(p, n); }
    void handOff() {
        drawsSinceHandOff_ = 0;
        if (syncRecord_ || chunk_->empty()) return;
        std::unique_ptr<CmdChunk> next;
        {
            std::lock_guard lock(chunkMutex_);
            chunks_.push_back(std::move(chunk_));
            if (!spareChunks_.empty()) {
                next = std::move(spareChunks_.front());
                spareChunks_.pop_front();
            }
        }
        chunkCv_.notify_one();
        chunk_ = next ? std::move(next) : std::make_unique<CmdChunk>();
    }
    void commandThread() {
        profile::nameThread("lpn-vkcmd");
        for (;;) {
            std::unique_ptr<CmdChunk> c;
            {
                std::unique_lock lock(chunkMutex_);
                chunkCv_.wait(lock, [&] { return !chunks_.empty(); });
                c = std::move(chunks_.front());
                chunks_.pop_front();
            }
            c->run();
            std::lock_guard lock(chunkMutex_);
            spareChunks_.push_back(std::move(c));
        }
    }
    void pushDescriptors(VkPipelineBindPoint point, VkPipelineLayout layout, std::vector<VkWriteDescriptorSet>& writes,
                         std::vector<VkDescriptorBufferInfo>& bufs, std::vector<VkDescriptorImageInfo>& imgs) {
        if (syncRecord_) {
            linkWrites(writes, bufs, imgs);
            vk_.cmdPushDescriptorSet(cmd_, point, layout, 0, uint32_t(writes.size()), writes.data());
            return;
        }
        auto* w = const_cast<VkWriteDescriptorSet*>(keep(writes.data(), writes.size()));
        const VkDescriptorBufferInfo* b = keep(bufs.data(), bufs.size());
        const VkDescriptorImageInfo* im = keep(imgs.data(), imgs.size());
        size_t nb = 0, ni = 0;
        for (size_t k = 0; k < writes.size(); ++k) {
            if (w[k].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) w[k].pBufferInfo = &b[nb++];
            else w[k].pImageInfo = &im[ni++];
        }
        const uint32_t n = uint32_t(writes.size());
        rec([this, c = cmd_, point, layout, n, w] { vk_.cmdPushDescriptorSet(c, point, layout, 0, n, w); });
    }

    void useBatch(uint32_t i) {
        cur_ = i;
        cmd_ = batches_[i].cmd;
        ring_ = batches_[i].ring;
        ringPtr_ = batches_[i].ringPtr;
        ringUsed_ = 0;
    }

    void begin() {
        if (recording_) return;
        Batch& b = batches_[cur_];
        {
            ProfScope ps(prof_, kProfGpuWait);
            std::unique_lock lock(doneMutex_);
            doneCv_.wait(lock, [&] { return !b.busy; });
        }
        rec([this, c = cmd_, fence = b.fence] {
            VK_CHECK(vkResetFences(vk_.device, 1, &fence));
            VK_CHECK(vkResetCommandBuffer(c, 0));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(c, &bi));
        });
        recording_ = true;
        bound_ = Bound{};
        needBarrier_ = true; // earlier batches' writes
        if (frame_ == timingFrame_ && timing_ && timingMeta_.size() + 1 < kTimingQueries) {
            rec([c = cmd_, q = timing_, i = uint32_t(timingMeta_.size() + 1)] {
                vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, q, i);
            });
            timingMeta_.push_back({0, 0, false, false, 0, true});
        }
        if (timestamps_)
            rec([c = cmd_, q = timestamps_, i = 2 * cur_] {
                vkCmdResetQueryPool(c, q, i, 2);
                vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, q, i);
            });
    }

    void flush() {
        if (!recording_) {
            if (!pendingDone_.empty()) {
                // Nothing new to submit: due when the last submission finishes. Decided on the
                // command thread, which has submitted everything queued before.
                rec([this, fns = std::move(pendingDone_)]() mutable {
                    {
                        std::lock_guard lock(doneMutex_);
                        if (!inFlight_.empty()) {
                            auto& done = batches_[inFlight_.back()].done;
                            for (auto& f : fns) done.push_back(std::move(f));
                            return;
                        }
                    }
                    for (auto& f : fns) f();
                });
                pendingDone_.clear();
                handOff();
            }
            if (sync_) waitSubmitted();
            return;
        }
        ++prof_.flushes;
        endPass();
        if (raceCheck_ && !raceReads_.empty()) {
            auto reads = std::make_shared<std::unordered_map<uint64_t, RaceRead>>(std::move(raceReads_));
            raceReads_.clear();
            afterFlush_.insert(afterFlush_.begin(), [this, reads] { checkReads(*reads); });
        }
        Batch& b = batches_[cur_];
        if (timestamps_)
            rec([c = cmd_, q = timestamps_, i = 2 * cur_ + 1] { vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q, i); });
        std::vector<std::function<void()>> fns = std::move(afterFlush_);
        for (auto& f : pendingDone_) fns.push_back(std::move(f));
        afterFlush_.clear();
        pendingDone_.clear();
        {
            std::lock_guard lock(doneMutex_);
            b.busy = true; // its ring and command buffer are not reused until the GPU is done
            ++submitted_;
        }
        // The completion thread learns of the batch once it is submitted (inFlight_).
        rec([this, i = cur_, c = cmd_, fns = std::move(fns)]() mutable {
            Batch& b = batches_[i];
            VK_CHECK(vkEndCommandBuffer(c));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &c;
            VK_CHECK(vkQueueSubmit(vk_.queue, 1, &si, b.fence));
            if (timelineOn()) timeline("rec: batch submitted");
            {
                std::lock_guard lock(doneMutex_);
                for (auto& f : fns) b.done.push_back(std::move(f));
                inFlight_.push_back(i);
            }
            doneCv_.notify_all();
        });
        handOff();
        recording_ = false;
        useBatch((cur_ + 1) % kBatches);
        drawsSinceFlush_ = 0;
        if (sync_) waitSubmitted();
    }

    // Submits what is recorded and waits until the GPU (and the deferred work) is done.
    void waitIdle() {
        flush();
        waitSubmitted();
    }
    void waitSubmitted() {
        ProfScope ps(prof_, kProfGpuWait);
        std::unique_lock lock(doneMutex_);
        doneCv_.wait(lock, [&] { return completed_ >= submitted_ && !completing_; });
    }

    void completeBatches() {
        profile::nameThread("lpn-gpudone");
        for (;;) {
            uint32_t i;
            {
                std::unique_lock lock(doneMutex_);
                doneCv_.wait(lock, [&] { return !inFlight_.empty(); });
                i = inFlight_.front();
            }
            Batch& b = batches_[i];
            VK_CHECK(vkWaitForFences(vk_.device, 1, &b.fence, VK_TRUE, UINT64_MAX));
            if (timelineOn()) timeline("gpu: batch %" PRIu64 " done", completed_ + 1);
            if (timestamps_) {
                uint64_t t[2];
                if (vkGetQueryPoolResults(vk_.device, timestamps_, 2 * i, 2, sizeof(t), t, sizeof(uint64_t),
                                          VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0])
                    gpuBusyNs_ += uint64_t(double(t[1] - t[0]) * gpuTickNs_);
            }
            std::vector<std::function<void()>> done;
            {
                std::lock_guard lock(doneMutex_);
                done.swap(b.done);
                b.busy = false;
                inFlight_.pop_front();
                ++completed_;
                completing_ = true;
            }
            for (auto& f : done) f();
            {
                std::lock_guard lock(doneMutex_);
                completing_ = false;
            }
            doneCv_.notify_all();
        }
    }

public:
    bool afterGpuWork(std::function<void()>& fn) override {
        std::lock_guard lock(m_);
        pendingDone_.push_back(std::move(fn));
        return true;
    }

private:

    // Before a copy, clear, dispatch or upload: everything before it is done and visible. What it
    // writes is then made visible by the barrier the next draw's pass starts with (needBarrier_).
    void fullBarrier() {
        endPass();
        barrier();
        needBarrier_ = true;
    }
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        rec([c = cmd_, mb] {
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0,
                                 nullptr);
        });
        ++prof_.barriers;
        needBarrier_ = false;
        drawWritten_.clear();
        drawSampled_.clear();
    }
    static bool overlaps(const std::vector<VkImage>& a, const std::vector<VkImage>& b) {
        for (VkImage x : a)
            if (std::find(b.begin(), b.end(), x) != b.end()) return true;
        return false;
    }
    static void addOnce(std::vector<VkImage>& v, VkImage x) {
        if (std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
    }

    // Space in the upload ring; flushes first if the ring is full.
    VkDeviceSize ringAlloc(VkDeviceSize size, VkDeviceSize align) {
        VkDeviceSize at = (ringUsed_ + align - 1) & ~(align - 1);
        if (at + size > kRingSize) {
            flush();
            begin();
            at = 0;
        }
        ringUsed_ = at + size;
        return at;
    }
    bool ringFits(VkDeviceSize size) const { return ringUsed_ + size + 4096 <= kRingSize; }

    void warnOnce(const std::string& w) {
        std::lock_guard lock(warnMutex_); // the analysis and recorder threads both warn
        if (warned_.insert(w).second) std::fprintf(stderr, "[gpu] %s\n", w.c_str());
    }
    std::mutex warnMutex_;

    // ---- memory --------------------------------------------------------------------------------

    VkBuffer createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory* mem, void** map,
                              bool cpuReads = false) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        VkBuffer b;
        if (vkCreateBuffer(vk_.device, &bci, nullptr, &b) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(vk_.device, b, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // Uncached memory is slow for the CPU to read: prefer cached memory for readbacks.
        mai.memoryTypeIndex = cpuReads ? vk_.memoryType(req.memoryTypeBits, kHost | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                                       : UINT32_MAX;
        if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = vk_.memoryType(req.memoryTypeBits, kHost);
        VK_CHECK(vkAllocateMemory(vk_.device, &mai, nullptr, mem));
        VK_CHECK(vkBindBufferMemory(vk_.device, b, *mem, 0));
        if (map) VK_CHECK(vkMapMemory(vk_.device, *mem, 0, size, 0, map));
        else {
            void* p;
            VK_CHECK(vkMapMemory(vk_.device, *mem, 0, size, 0, &p));
            std::memset(p, 0, size);
            vkUnmapMemory(vk_.device, *mem);
        }
        return b;
    }

    bool importRange(uint64_t start, uint64_t end, Import& out) {
        // Some drivers (lavapipe) accept any pointer, so make sure every page is mapped.
        std::vector<unsigned char> residency((end - start + 4095) / 4096);
        if (mincore(reinterpret_cast<void*>(start), end - start, residency.data()) != 0) return false;
        VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        void* ptr = reinterpret_cast<void*>(start);
        if (vk_.getMemoryHostPointerProperties(vk_.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr,
                                                &hp) != VK_SUCCESS)
            return false;
        VkExternalMemoryBufferCreateInfo ebci{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ebci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.pNext = &ebci;
        bci.size = end - start;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (vkCreateBuffer(vk_.device, &bci, nullptr, &out.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(vk_.device, out.buffer, &req);
        VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = ptr;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &imp;
        mai.allocationSize = end - start;
        mai.memoryTypeIndex = vk_.memoryType(req.memoryTypeBits & hp.memoryTypeBits, 0);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk_.device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
            vkDestroyBuffer(vk_.device, out.buffer, nullptr);
            return false;
        }
        VK_CHECK(vkBindBufferMemory(vk_.device, out.buffer, out.memory, 0));
        out.start = start;
        out.end = end;
        return true;
    }

    // A Vulkan buffer over guest memory [addr, addr + size).
    bool guestBuffer(uint64_t addr, uint64_t size, VkBuffer& buffer, VkDeviceSize& offset) {
        if (!addr || addr >= (1ull << 40) || !size) return false;
        const uint64_t end = addr + size;
        // The import starting closest below `addr` usually covers it (imports are aligned granules);
        // a scan of all of them is the fallback.
        if (auto it = imports_.upper_bound(addr); it != imports_.begin()) {
            --it;
            if (it->second.end >= end) {
                buffer = it->second.buffer;
                offset = addr - it->first;
                return true;
            }
        }
        for (const auto& [start, im] : imports_) {
            if (start > addr) break;
            if (im.end >= end) {
                buffer = im.buffer;
                offset = addr - start;
                return true;
            }
        }
        const uint64_t align = std::max<uint64_t>(vk_.hostPointerAlignment, 4096);
        Import im;
        const uint64_t gs = addr & ~(kImportGranule - 1), ge = (end + kImportGranule - 1) & ~(kImportGranule - 1);
        const uint64_t ps = addr & ~(align - 1), pe = (end + align - 1) & ~(align - 1);
        if (!importRange(gs, ge, im) && !importRange(ps, pe, im)) {
            warnOnce("cannot import guest memory at 0x" + std::to_string(addr));
            return false;
        }
        imports_[im.start] = im;
        buffer = im.buffer;
        offset = addr - im.start;
        return true;
    }

    // ---- images --------------------------------------------------------------------------------

    bool createImage(Image& img, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage, bool depth,
                     VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D, uint32_t layers = 1) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = depth ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        if (viewType == VK_IMAGE_VIEW_TYPE_CUBE) ici.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ici.imageType = viewType == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {w, h, viewType == VK_IMAGE_VIEW_TYPE_3D ? layers : 1};
        ici.mipLevels = 1;
        ici.arrayLayers = viewType == VK_IMAGE_VIEW_TYPE_3D ? 1 : layers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ++imageGen_; // remembered image bindings may name an image this replaces
        if (vkCreateImage(vk_.device, &ici, nullptr, &img.image) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(vk_.device, img.image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = vk_.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = vk_.memoryType(req.memoryTypeBits, 0);
        VK_CHECK(vkAllocateMemory(vk_.device, &mai, nullptr, &img.memory));
        VK_CHECK(vkBindImageMemory(vk_.device, img.image, img.memory, 0));
        img.format = format;
        img.width = w;
        img.height = h;
        img.layers = layers;
        img.aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        if (format == VK_FORMAT_D32_SFLOAT_S8_UINT) img.aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

        // Everything stays in GENERAL; start it there.
        begin();
        endPass();
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = img.image;
        ib.subresourceRange = {img.aspect, 0, 1, 0, ici.arrayLayers};
        rec([c = cmd_, ib] {
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &ib);
        });
        // Clear so never-written surfaces read as black rather than garbage.
        const VkImageSubresourceRange range{img.aspect, 0, 1, 0, ici.arrayLayers};
        if (depth) {
            if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                rec([c = cmd_, im = img.image, range] {
                    const VkClearDepthStencilValue v{1.0f, 0};
                    vkCmdClearDepthStencilImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range);
                });
        } else if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
            rec([c = cmd_, im = img.image, range] {
                const VkClearColorValue v{};
                vkCmdClearColorImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range);
            });
        }
        return true;
    }

    void destroyImage(Image& img) {
        if (!img.image) return;
        // The command buffer may still reference it: release after the next flush.
        Image copy = img;
        afterFlush_.push_back([this, copy] {
            for (auto& [k, v] : copy.views) vkDestroyImageView(vk_.device, v, nullptr);
            vkDestroyImage(vk_.device, copy.image, nullptr);
            vkFreeMemory(vk_.device, copy.memory, nullptr);
        });
        img = Image{};
    }

    VkImageView view(Image& img, VkFormat format, VkImageViewType type, VkComponentMapping swz,
                     VkImageAspectFlags aspect = 0) {
        if (!aspect) aspect = img.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT;
        if (!aspect) aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        const std::array<uint32_t, 7> k{uint32_t(format), uint32_t(type), uint32_t(swz.r), uint32_t(swz.g),
                                        uint32_t(swz.b), uint32_t(swz.a), aspect};
        auto it = img.views.find(k);
        if (it != img.views.end()) return it->second;
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = type;
        vci.format = format;
        vci.components = swz;
        vci.subresourceRange = {aspect, 0, 1, 0,
                                type == VK_IMAGE_VIEW_TYPE_3D ? 1 : (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D) ? 1 : img.layers};
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        VkImageView v = VK_NULL_HANDLE;
        if (vkCreateImageView(vk_.device, &vci, nullptr, &v) != VK_SUCCESS) {
            (void)usage;
            warnOnce("image view creation failed");
            return VK_NULL_HANDLE;
        }
        img.views[k] = v;
        return v;
    }

    static constexpr VkComponentMapping kIdentity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                  VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

    Surface* colorSurface(uint64_t addr, uint32_t w, uint32_t h, VkFormat format, uint32_t bytes) {
        auto it = colors_.find(addr);
        if (it != colors_.end()) {
            Surface& s = it->second;
            if (s.img.width == w && s.img.height == h && s.elementBytes == bytes) return &s;
            destroyImage(s.img);
            colors_.erase(it);
        }
        Surface s;
        s.elementBytes = bytes;
        if (!createImage(s.img, format,
                         w, h,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         false)) {
            warnOnce("cannot create a render target");
            return nullptr;
        }
        std::fprintf(stderr, "[gpu] render target 0x%" PRIx64 " %ux%u format %d\n", addr, w, h, int(format));
        return &(colors_[addr] = std::move(s));
    }

    Surface* depthSurface(uint64_t addr, uint32_t w, uint32_t h, VkFormat format) {
        auto it = depths_.find(addr);
        if (it != depths_.end()) {
            Surface& s = it->second;
            if (s.img.width == w && s.img.height == h && s.img.format == format) return &s;
            destroyImage(s.img);
            depths_.erase(it);
        }
        Surface s;
        s.depth = true;
        s.elementBytes = format == VK_FORMAT_D16_UNORM ? 2 : 4;
        if (!createImage(s.img, format, w, h,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         true)) {
            warnOnce("cannot create a depth buffer");
            return nullptr;
        }
        std::fprintf(stderr, "[gpu] depth buffer 0x%" PRIx64 " %ux%u\n", addr, w, h);
        return &(depths_[addr] = std::move(s));
    }

    // ---- shaders and pipelines -----------------------------------------------------------------

    ShaderModule* shader(const gcn::Program& p, uint32_t bindingBase) {
        std::string key = p.key();
        key.append(reinterpret_cast<const char*>(&bindingBase), 4);
        auto it = shaders_.find(key);
        if (it != shaders_.end()) return it->second.get();
        auto m = std::make_unique<ShaderModule>();
        m->translation = gcn::translate(p, bindingBase);
        for (const auto& w : p.warnings) warnOnce("shader 0x" + hex(p.launch.entry) + ": " + w);
        for (const auto& w : m->translation.warnings) warnOnce("shader 0x" + hex(p.launch.entry) + ": " + w);
        const gcn::GlslStage stage = p.launch.stage == gcn::Stage::Pixel     ? gcn::GlslStage::Fragment
                                     : p.launch.stage == gcn::Stage::Compute ? gcn::GlslStage::Compute
                                                                             : gcn::GlslStage::Vertex;
        // Debugging: LPN_DBG_PS=<entry>[,<entry>...] rewrites those pixel shaders' color output to
        // LPN_DBG_OUT (a GLSL vec4 argument list). dbgT<n> holds image n's last sample, and
        // LPN_DBG_CAP="<line text>|<expr>" stores vec4(expr) in dbgC after the first matching line.
        if (const char* dp = getenv("LPN_DBG_PS"); dp && *dp && getenv("LPN_DBG_OUT") && stage == gcn::GlslStage::Fragment &&
                                                   std::strstr(dp, hex(p.launch.entry).c_str())) {
            std::string& g = m->translation.glsl;
            for (int i = 15; i >= 0; --i) {
                const std::string pat = "vec4 t_ = vec4(texture(img" + std::to_string(i) + ",";
                for (size_t q = g.find(pat); q != std::string::npos; q = g.find(pat, q + 1)) {
                    const size_t e = g.find(";\n", q);
                    g.insert(e + 1, " dbgT" + std::to_string(i) + " = t_;");
                }
            }
            if (const char* cap = getenv("LPN_DBG_CAP"); cap && std::strchr(cap, '|')) {
                const std::string c = cap;
                const size_t bar = c.find('|');
                const size_t q = g.find(c.substr(0, bar));
                if (q != std::string::npos) g.insert(g.find('\n', q) + 1, "dbgC = vec4(" + c.substr(bar + 1) + ");\n");
            }
            std::string decl = "vec4 dbgC = vec4(0.0);\n";
            for (int i = 0; i < 16; ++i) decl += "vec4 dbgT" + std::to_string(i) + " = vec4(0.0);\n";
            g.insert(g.find("void main()"), decl);
            const size_t at = g.find("\noC0 = ");
            if (at != std::string::npos) {
                const size_t end = g.find('\n', at + 1);
                g.replace(at + 1, end - at - 1, std::string("oC0 = vec4(") + getenv("LPN_DBG_OUT") + ");");
            }
        }
        m->module = module(m->translation.glsl, stage);
        m->name = spirvName(m->translation.glsl, stage);
        m->id = uint32_t(shaders_.size() + 1);
        if (dumpGlsl_) { // debugging: every translation as gpu-glsl-<stage>-<entry>-<id>.glsl
            const char* st = stage == gcn::GlslStage::Fragment ? "ps" : stage == gcn::GlslStage::Compute ? "cs" : "vs";
            const std::string name = std::string("gpu-glsl-") + st + "-" + hex(p.launch.entry) + "-" + std::to_string(m->id) + ".glsl";
            if (FILE* f = std::fopen(name.c_str(), "wb")) {
                std::fwrite(m->translation.glsl.data(), 1, m->translation.glsl.size(), f);
                std::fclose(f);
            }
        }
        return (shaders_[key] = std::move(m)).get();
    }

    VkShaderModule module(const std::string& glsl, gcn::GlslStage stage) {
        const std::string cached = spirvCachePath(glsl, stage);
        std::vector<uint32_t> spirv = loadSpirv(cached);
        if (spirv.empty()) {
            std::string error;
            spirv = gcn::compileGlsl(glsl, stage, &error);
            if (spirv.empty()) {
                std::fprintf(stderr, "[gpu] shader compile failed:\n%s\n--- source ---\n%s\n", error.c_str(), glsl.c_str());
                return VK_NULL_HANDLE;
            }
            saveSpirv(cached, spirv);
        }
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spirv.size() * 4;
        ci.pCode = spirv.data();
        VkShaderModule m = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(vk_.device, &ci, nullptr, &m));
        return m;
    }

    // Compiling GLSL takes ~10 ms a shader, and a new effect brings dozens: SPIR-V is kept on disk
    // in $LPN_SHADER_CACHE (default ~/.cache/lpn-host/spirv; "off" disables), named by a hash of
    // the source.
    std::string spirvCachePath(const std::string& glsl, gcn::GlslStage stage) {
        cacheDir();
        if (spirvDir_ == "-") return {};
        return spirvDir_ + "/" + spirvName(glsl, stage) + ".spv";
    }
    static std::string spirvName(const std::string& glsl, gcn::GlslStage stage) {
        std::string text = glsl;
        text += char('0' + int(stage));
        text.resize((text.size() + 7) & ~size_t(7), '\0');
        const uint64_t a = hashBytes(text.data(), text.size());
        for (char& c : text) c = char(c ^ 0x5a); // a second, independent hash
        const uint64_t b = hashBytes(text.data(), text.size());
        char name[40];
        std::snprintf(name, sizeof name, "%016" PRIx64 "%016" PRIx64, a, b);
        return name;
    }
    // The on-disk caches' folder: $LPN_SHADER_CACHE (default ~/.cache/lpn-host/spirv; "off": none).
    void cacheDir() {
        if (!spirvDir_.empty()) return;
        const char* env = getenv("LPN_SHADER_CACHE");
        if (env && !std::strcmp(env, "off")) spirvDir_ = "-";
        else if (env && *env) spirvDir_ = env;
        else if (const char* x = getenv("XDG_CACHE_HOME"); x && *x) spirvDir_ = std::string(x) + "/lpn-host/spirv";
        else if (const char* h = getenv("HOME"); h && *h) spirvDir_ = std::string(h) + "/.cache/lpn-host/spirv";
        else spirvDir_ = "-";
        if (spirvDir_ != "-") {
            std::error_code ec;
            std::filesystem::create_directories(spirvDir_, ec);
            if (ec) spirvDir_ = "-";
        }
    }

    // Pipelines (the driver compiling SPIR-V for the GPU, ~50 ms a frame when an area appears)
    // are kept in pipelines.bin next to the SPIR-V; the driver ignores data from another GPU or
    // driver version.
    void openPipelineCache() {
        cacheDir();
        std::vector<char> data;
        if (spirvDir_ != "-") {
            pipelineCachePath_ = spirvDir_ + "/pipelines.bin";
            if (FILE* f = std::fopen(pipelineCachePath_.c_str(), "rb")) {
                std::fseek(f, 0, SEEK_END);
                const long n = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (n > 0) {
                    data.resize(size_t(n));
                    if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
                }
                std::fclose(f);
            }
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = data.size();
        ci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(vk_.device, &ci, nullptr, &pipelineCache_) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            VK_CHECK(vkCreatePipelineCache(vk_.device, &ci, nullptr, &pipelineCache_));
        }
    }
    // Called at frame end: writes the cache once some new pipelines have been made. Writing it
    // (tens of MB) stalled the frame ~190 ms every 5 s in crowds, where pipelines keep coming, so
    // it is written on its own thread (pipeline caches are internally synchronized).
    void savePipelineCache() {
        if (pipelineCachePath_.empty() || !newPipelines_ || pipelineSaving_.load()) return;
        const auto now = std::chrono::steady_clock::now();
        if (now - lastPipelineSave_ < std::chrono::seconds(20)) return;
        newPipelines_ = 0;
        lastPipelineSave_ = now;
        pipelineSaving_ = true;
        std::thread([this] { writePipelineCache(); }).detach();
    }
public:
    // At exit: pipelines made since the last save would otherwise be compiled again next run.
    void savePipelineCacheNow() {
        if (pipelineCachePath_.empty()) return;
        while (pipelineSaving_.exchange(true)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (newPipelines_) writePipelineCache();
        else pipelineSaving_ = false;
    }
private:
    void writePipelineCache() {
        {
            size_t n = 0;
            std::vector<char> data;
            if (vkGetPipelineCacheData(vk_.device, pipelineCache_, &n, nullptr) == VK_SUCCESS && n) {
                data.resize(n);
                if (vkGetPipelineCacheData(vk_.device, pipelineCache_, &n, data.data()) != VK_SUCCESS) n = 0;
            }
            const std::string tmp = pipelineCachePath_ + ".tmp" + std::to_string(getpid());
            if (n)
                if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
                    const bool ok = std::fwrite(data.data(), 1, n, f) == n;
                    if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), pipelineCachePath_.c_str());
                    else std::remove(tmp.c_str());
                }
            pipelineSaving_ = false;
        }
    }

    static std::vector<uint32_t> loadSpirv(const std::string& path) {
        std::vector<uint32_t> words;
        if (path.empty()) return words;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return words;
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 20 && n % 4 == 0) {
            words.resize(size_t(n) / 4);
            if (std::fread(words.data(), 4, words.size(), f) != words.size() || words[0] != 0x07230203) words.clear();
        }
        std::fclose(f);
        return words;
    }
    static void saveSpirv(const std::string& path, const std::vector<uint32_t>& words) {
        if (path.empty()) return;
        const std::string tmp = path + ".tmp" + std::to_string(getpid());
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const bool ok = std::fwrite(words.data(), 4, words.size(), f) == words.size();
        if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), path.c_str()); // never a partial file
        else std::remove(tmp.c_str());
    }

    static std::string hex(uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "%" PRIx64, v);
        return b;
    }

    // Geometry shader turning each primitive of a RECTLIST (3 corners) or QUADLIST into a strip.
    VkShaderModule expandShader(bool quads, uint32_t outputs) {
        std::lock_guard lock(layoutMutex_);
        const uint64_t key = (uint64_t(quads) << 32) | outputs;
        auto it = expanders_.find(key);
        if (it != expanders_.end()) return it->second;
        std::string g = "#version 450\n";
        g += quads ? "layout(lines_adjacency) in;\n" : "layout(triangles) in;\n";
        g += "layout(triangle_strip, max_vertices = 4) out;\n"
             "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
             "out gl_PerVertex { vec4 gl_Position; };\n";
        for (uint32_t i = 0; i < 32; ++i)
            if ((outputs >> i) & 1) {
                g += "layout(location = " + std::to_string(i) + ") in vec4 iP" + std::to_string(i) + "[];\n";
                g += "layout(location = " + std::to_string(i) + ") out vec4 oP" + std::to_string(i) + ";\n";
            }
        g += "void emit(int i) {\n    gl_Position = gl_in[i].gl_Position;\n";
        for (uint32_t i = 0; i < 32; ++i)
            if ((outputs >> i) & 1) g += "    oP" + std::to_string(i) + " = iP" + std::to_string(i) + "[i];\n";
        g += "    EmitVertex();\n}\n";
        g += "void main() {\n";
        if (quads) {
            g += "    emit(0); emit(1); emit(3); emit(2);\n";
        } else {
            // The corner is the vertex whose two edges are perpendicular; the fourth vertex
            // completes the parallelogram opposite it.
            g += "    vec2 p0 = gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;\n"
                 "    vec2 p1 = gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;\n"
                 "    vec2 p2 = gl_in[2].gl_Position.xy / gl_in[2].gl_Position.w;\n"
                 "    float d0 = abs(dot(p1 - p0, p2 - p0)), d1 = abs(dot(p0 - p1, p2 - p1)), d2 = abs(dot(p0 - p2, p1 - p2));\n"
                 "    int c = (d0 <= d1 && d0 <= d2) ? 0 : (d1 <= d2 ? 1 : 2);\n"
                 "    int a = (c + 1) % 3, b = (c + 2) % 3;\n"
                 "    emit(c); emit(a); emit(b);\n"
                 "    gl_Position = gl_in[a].gl_Position + gl_in[b].gl_Position - gl_in[c].gl_Position;\n";
            for (uint32_t i = 0; i < 32; ++i)
                if ((outputs >> i) & 1) {
                    const std::string n = std::to_string(i);
                    g += "    oP" + n + " = iP" + n + "[a] + iP" + n + "[b] - iP" + n + "[c];\n";
                }
            g += "    EmitVertex();\n";
        }
        g += "    EndPrimitive();\n}\n";
        VkShaderModule m = module(g, gcn::GlslStage::Geometry);
        expanders_[key] = m;
        return m;
    }

    struct Layout {
        VkDescriptorSetLayout set = VK_NULL_HANDLE;
        VkPipelineLayout pipeline = VK_NULL_HANDLE;
    };

    void addBindings(std::vector<VkDescriptorSetLayoutBinding>& b, const gcn::Program& p, uint32_t base,
                     VkShaderStageFlags stage) {
        b.push_back({base, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stage, nullptr});
        uint32_t n = base + 1;
        for (size_t i = 0; i < p.buffers.size(); ++i) b.push_back({n++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stage, nullptr});
        for (const auto& im : p.images)
            b.push_back({n++, im.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                         stage, nullptr});
    }

    Layout& layout(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        std::lock_guard lock(layoutMutex_);
        thread_local std::string key; // per draw: built in a reused buffer
        key.clear();
        for (const auto& b : bindings) {
            const uint32_t k[] = {b.binding, uint32_t(b.descriptorType), b.stageFlags};
            key.append(reinterpret_cast<const char*>(k), sizeof(k));
        }
        auto it = layouts_.find(key);
        if (it != layouts_.end()) return it->second;
        Layout l;
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dci.bindingCount = uint32_t(bindings.size());
        dci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(vk_.device, &dci, nullptr, &l.set));
        VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = 1;
        pci.pSetLayouts = &l.set;
        VK_CHECK(vkCreatePipelineLayout(vk_.device, &pci, nullptr, &l.pipeline));
        return layouts_[key] = l;
    }

    // Fixed-function state that selects a graphics pipeline.
    struct GraphicsState {
        uint32_t vs = 0, ps = 0;
        uint32_t topology = 0; // VGT primitive type
        uint32_t modeCntl = 0, depthControl = 0, stencilControl = 0, stencilMasks = 0, stencilMasksBf = 0;
        uint32_t depthFormat = 0, stencil = 0, depthWriteOff = 0;
        uint32_t colorFormats[8] = {};
        uint32_t writeMasks = 0;
        uint32_t blend[8] = {};
        uint32_t vsOutputs = 0;
        // Stencil reference/masks and the depth bias are dynamic state: they vary per character
        // and per shadow cascade, and as pipeline state each value made a new pipeline (a frame
        // building a dozen of them hitched 0.2-0.4 s in crowds). Not part of the key; draw() sets them.
    };
    struct DynamicState {
        uint32_t stencilMasks = 0, stencilMasksBf = 0; // DB_STENCILREFMASK(_BF)
        uint32_t polyOffset[3] = {}; // clamp, scale, offset (float bits) while polygon offset is on
    };

    static VkBlendFactor blendFactor(uint32_t f) {
        switch (f) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 2: return VK_BLEND_FACTOR_SRC_COLOR;
        case 3: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 4: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 5: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 6: return VK_BLEND_FACTOR_DST_ALPHA;
        case 7: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_DST_COLOR;
        case 9: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 10: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 13: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 14: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 15: return VK_BLEND_FACTOR_SRC1_COLOR;
        case 16: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case 17: return VK_BLEND_FACTOR_SRC1_ALPHA;
        case 18: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 19: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 20: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
        }
    }
    static VkBlendOp blendOp(uint32_t f) {
        switch (f) {
        case 1: return VK_BLEND_OP_SUBTRACT;
        case 2: return VK_BLEND_OP_MIN;
        case 3: return VK_BLEND_OP_MAX;
        case 4: return VK_BLEND_OP_REVERSE_SUBTRACT;
        default: return VK_BLEND_OP_ADD;
        }
    }
    static VkStencilOp stencilOp(uint32_t o) {
        switch (o) {
        case 1: return VK_STENCIL_OP_ZERO;
        case 2: return VK_STENCIL_OP_REPLACE; // ONES: approximated
        case 3: case 4: return VK_STENCIL_OP_REPLACE;
        case 5: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 6: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 7: return VK_STENCIL_OP_INVERT;
        case 8: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 9: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
        }
    }

    // ---- pipeline builds off the recorder ----
    // The driver takes 10-100 ms to build a pipeline it has not cached, and a new area or effect
    // needs dozens at once: built in line, they froze the game for up to seconds. Pipelines are
    // built by worker threads instead. A blended draw that writes no depth (effects, particles,
    // UI) whose pipeline is still being built is skipped: it can be missing for a frame or two.
    // Other draws wait for theirs, as skipping one let the ground vanish for a moment and could
    // leave depth that later draws test against. Every pipeline is also recorded in
    // pipelines.list beside the SPIR-V (state, shader names, bindings), and each run builds the
    // recorded ones at startup, so places seen before have them ready.
    // LPN_SYNC_PIPELINES=1 builds in line as before (no skipped draws); LPN_PREBUILD=0 skips the startup builds.
    struct BuiltPipeline {
        std::atomic<bool> ready{false};
        std::atomic<bool> started{false}; // a worker has it
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    struct PipelineJob {
        GraphicsState state; // vs and ps are 0: the modules below
        VkShaderModule vs = VK_NULL_HANDLE, ps = VK_NULL_HANDLE, geom = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::shared_ptr<BuiltPipeline> out;
        std::string record; // appended to pipelines.list once built, unless it came from there
        bool prebuild = false; // from pipelines.list: vs and ps are its own, destroyed after
    };
    static constexpr uint32_t kRecordVersion = 1;

    // What identifies a pipeline across runs (its key in built_ and its pipelines.list entry).
    static std::string pipelineRecord(const GraphicsState& s, const std::string& vs, const std::string& ps,
                                      const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        std::string r;
        auto put = [&](const void* p, size_t n) { r.append(static_cast<const char*>(p), n); };
        const uint32_t head[] = {kRecordVersion, uint32_t(sizeof(GraphicsState)), uint32_t(vs.size()), uint32_t(ps.size()),
                                 uint32_t(bindings.size())};
        put(head, sizeof head);
        put(&s, sizeof s);
        r += vs;
        r += ps;
        for (const auto& b : bindings) {
            const uint32_t k[] = {b.binding, uint32_t(b.descriptorType), b.stageFlags};
            put(k, sizeof k);
        }
        return r;
    }
    static bool parseRecord(const std::string& r, GraphicsState& s, std::string& vs, std::string& ps,
                            std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        uint32_t head[5];
        if (r.size() < sizeof head) return false;
        std::memcpy(head, r.data(), sizeof head);
        if (head[0] != kRecordVersion || head[1] != sizeof(GraphicsState) ||
            r.size() != sizeof head + sizeof s + head[2] + head[3] + size_t(head[4]) * 12)
            return false;
        size_t at = sizeof head;
        std::memcpy(&s, r.data() + at, sizeof s);
        at += sizeof s;
        vs = r.substr(at, head[2]);
        ps = r.substr(at + head[2], head[3]);
        at += head[2] + head[3];
        bindings.resize(head[4]);
        for (auto& b : bindings) {
            uint32_t k[3];
            std::memcpy(k, r.data() + at, sizeof k);
            at += sizeof k;
            b = {k[0], VkDescriptorType(k[1]), 1, k[2], nullptr};
        }
        return true;
    }

    VkShaderModule geometryFor(const GraphicsState& s) {
        if (s.topology == 17) return expandShader(false, s.vsOutputs);
        if (s.topology == 19) return expandShader(true, s.vsOutputs);
        return VK_NULL_HANDLE;
    }

    VkPipeline graphicsPipeline(const GraphicsState& s, ShaderModule* vs, ShaderModule* ps, VkPipelineLayout layout,
                                const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        thread_local std::string key; // per draw: built in a reused buffer
        key.assign(reinterpret_cast<const char*>(&s), sizeof(s));
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) return it->second;
        if (!vs->module || (ps && !ps->module)) return pipelines_[key] = VK_NULL_HANDLE;
        GraphicsState c = s;
        c.vs = c.ps = 0;
        std::string record = pipelineRecord(c, vs->name, ps ? ps->name : std::string(), bindings);
        std::shared_ptr<BuiltPipeline> b;
        bool fresh = false;
        {
            std::lock_guard lock(buildMutex_);
            auto& slot = built_[record];
            if (!slot) slot = std::make_shared<BuiltPipeline>(), fresh = true;
            b = slot;
        }
        if (fresh) {
            ++prof_.pipelinesMade;
            PipelineJob j{c, vs->module, ps ? ps->module : VK_NULL_HANDLE, geometryFor(s), layout, b, std::move(record), false};
            if (syncPipelines_) runJob(j);
            else {
                queueJob(std::move(j), true);
                hurry(b); // the recorder needs this one now
            }
        }
        if (!b->ready.load(std::memory_order_acquire)) {
            if (!b->started.load()) hurry(b);
            if (skippable(c)) {
                ++prof_.pipelineWaits;
                return VK_NULL_HANDLE; // still being built: the draw is skipped
            }
            ProfScope pw(prof_, kProfGpuWait);
            b->ready.wait(false, std::memory_order_acquire);
        }
        return pipelines_[key] = b->pipeline;
    }

    // Blended into the color targets and writing no depth: nothing later depends on it.
    static bool skippable(const GraphicsState& s) {
        if (s.depthFormat && ((s.depthControl >> 2) & 1) && !s.depthWriteOff) return false;
        bool any = false;
        for (uint32_t i = 0; i < 8; ++i)
            if (s.colorFormats[i] && ((s.writeMasks >> (i * 4)) & 0xF)) {
                if (!((s.blend[i] >> 30) & 1)) return false;
                any = true;
            }
        return any;
    }

    // A startup build a draw is now waiting for goes to the front of the queue.
    void hurry(const std::shared_ptr<BuiltPipeline>& b) {
        std::lock_guard lock(buildMutex_);
        for (auto* q : {&jobs_, &backgroundJobs_})
            for (auto it = q->begin(); it != q->end(); ++it)
                if (it->out == b) {
                    PipelineJob j = std::move(*it);
                    q->erase(it);
                    jobs_.push_front(std::move(j));
                    return;
                }
    }

    void startPipelineWorkers() {
        syncPipelines_ = getenv("LPN_SYNC_PIPELINES") != nullptr;
        if (spirvDir_ != "-") {
            pipelineListPath_ = spirvDir_ + "/pipelines.list";
            pipelineList_ = std::fopen(pipelineListPath_.c_str(), "ab");
        }
        if (syncPipelines_) return;
        const unsigned n = std::clamp(std::thread::hardware_concurrency() >= 8 ? std::thread::hardware_concurrency() - 4 : 2u, 2u, 16u);
        for (unsigned i = 0; i < n; ++i)
            std::thread([this] {
                profile::nameThread("lpn-pipeline");
                for (;;) {
                    PipelineJob j;
                    {
                        std::unique_lock lock(buildMutex_);
                        buildCv_.wait(lock, [&] { return !jobs_.empty() || !backgroundJobs_.empty(); });
                        auto& q = jobs_.empty() ? backgroundJobs_ : jobs_;
                        j = std::move(q.front());
                        q.pop_front();
                        j.out->started = true;
                    }
                    runJob(j);
                }
            }).detach();
        const char* pb = getenv("LPN_PREBUILD");
        if (!pipelineListPath_.empty() && !(pb && !std::strcmp(pb, "0"))) std::thread([this] { prebuildPipelines(); }).detach();
    }

    // Draws' pipelines go ahead of the startup builds.
    void queueJob(PipelineJob&& j, bool urgent) {
        {
            std::lock_guard lock(buildMutex_);
            // A draw's pipelines in the order they were asked for (a waiting draw's goes first, see
            // hurry()); the startup builds after all of them.
            (urgent ? jobs_ : backgroundJobs_).push_back(std::move(j));
        }
        buildCv_.notify_one();
    }

    void runJob(PipelineJob& j) {
        j.out->pipeline = buildGraphics(j.state, j.vs, j.ps, j.geom, j.layout);
        j.out->started = true;
        j.out->ready.store(true, std::memory_order_release);
        j.out->ready.notify_all();
        ++newPipelines_;
        if (j.prebuild) {
            vkDestroyShaderModule(vk_.device, j.vs, nullptr);
            if (j.ps) vkDestroyShaderModule(vk_.device, j.ps, nullptr);
        } else if (j.out->pipeline && pipelineList_) {
            std::lock_guard lock(listMutex_);
            const uint32_t n = uint32_t(j.record.size());
            std::fwrite(&n, 4, 1, pipelineList_);
            std::fwrite(j.record.data(), 1, n, pipelineList_);
            std::fflush(pipelineList_);
        }
    }

    VkShaderModule loadModule(const std::string& name) {
        if (name.empty()) return VK_NULL_HANDLE;
        const std::vector<uint32_t> spirv = loadSpirv(spirvDir_ + "/" + name + ".spv");
        if (spirv.empty()) return VK_NULL_HANDLE;
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spirv.size() * 4;
        ci.pCode = spirv.data();
        VkShaderModule m = VK_NULL_HANDLE;
        return vkCreateShaderModule(vk_.device, &ci, nullptr, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
    }

    // Queues every pipeline in pipelines.list (newest first) for the workers.
    void prebuildPipelines() {
        profile::nameThread("lpn-prebuild");
        std::vector<std::string> records;
        if (FILE* f = std::fopen(pipelineListPath_.c_str(), "rb")) {
            uint32_t n;
            std::string r;
            while (std::fread(&n, 4, 1, f) == 1 && n < (1u << 20)) {
                r.resize(n);
                if (std::fread(r.data(), 1, n, f) != n) break;
                records.push_back(r);
            }
            std::fclose(f);
        }
        std::unordered_set<std::string> seen;
        size_t queued = 0;
        for (auto r = records.rbegin(); r != records.rend(); ++r) {
            if (!seen.insert(*r).second) continue;
            GraphicsState s;
            std::string vsName, psName;
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            if (!parseRecord(*r, s, vsName, psName, bindings)) continue;
            PipelineJob j;
            j.state = s;
            j.prebuild = true;
            j.vs = loadModule(vsName);
            j.ps = loadModule(psName);
            if (!j.vs || (!psName.empty() && !j.ps)) {
                if (j.vs) vkDestroyShaderModule(vk_.device, j.vs, nullptr);
                if (j.ps) vkDestroyShaderModule(vk_.device, j.ps, nullptr);
                continue;
            }
            j.layout = layout(bindings).pipeline;
            j.geom = geometryFor(s);
            {
                std::lock_guard lock(buildMutex_);
                auto& slot = built_[*r];
                if (slot) { // a draw got there first
                    vkDestroyShaderModule(vk_.device, j.vs, nullptr);
                    if (j.ps) vkDestroyShaderModule(vk_.device, j.ps, nullptr);
                    continue;
                }
                slot = j.out = std::make_shared<BuiltPipeline>();
            }
            queueJob(std::move(j), false);
            ++queued;
        }
        std::fprintf(stderr, "[gpu] building %zu pipelines from earlier runs\n", queued);
        // Keep the list from growing without bound: rewrite it without duplicates.
        if (records.size() > seen.size() + 1024) {
            std::lock_guard lock(listMutex_);
            const std::string tmp = pipelineListPath_ + ".tmp" + std::to_string(getpid());
            if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
                std::unordered_set<std::string> written;
                for (const auto& r : records)
                    if (written.insert(r).second) {
                        const uint32_t n = uint32_t(r.size());
                        std::fwrite(&n, 4, 1, f);
                        std::fwrite(r.data(), 1, n, f);
                    }
                if (std::fclose(f) == 0) {
                    std::rename(tmp.c_str(), pipelineListPath_.c_str());
                    if (pipelineList_) std::fclose(pipelineList_);
                    pipelineList_ = std::fopen(pipelineListPath_.c_str(), "ab");
                }
            }
        }
    }

    // Thread-safe: only the device and the pipeline cache (internally synchronized).
    VkPipeline buildGraphics(const GraphicsState& s, VkShaderModule vsModule, VkShaderModule psModule, VkShaderModule geom,
                             VkPipelineLayout layout) {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        auto stage = [&](VkShaderStageFlagBits st, VkShaderModule m) {
            VkPipelineShaderStageCreateInfo i{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            i.stage = st;
            i.module = m;
            i.pName = "main";
            stages.push_back(i);
        };
        stage(VK_SHADER_STAGE_VERTEX_BIT, vsModule);
        VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        switch (s.topology) {
        case 1: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case 2: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case 5: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case 6: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 17: stage(VK_SHADER_STAGE_GEOMETRY_BIT, geom); break;
        case 19:
            topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
            stage(VK_SHADER_STAGE_GEOMETRY_BIT, geom);
            break;
        default: break;
        }
        if (psModule) stage(VK_SHADER_STAGE_FRAGMENT_BIT, psModule);

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topo;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        const bool rect = s.topology == 17 || s.topology == 19;
        rs.cullMode = VK_CULL_MODE_NONE;
        if (!rect && (s.modeCntl & 1)) rs.cullMode |= VK_CULL_MODE_FRONT_BIT;
        if (!rect && (s.modeCntl & 2)) rs.cullMode |= VK_CULL_MODE_BACK_BIT;
        rs.frontFace = (s.modeCntl & 4) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        if (s.modeCntl & (3u << 11)) { // POLY_OFFSET_FRONT/BACK_ENABLE
            // Scale is in 1/16ths of the slope; the offset is in the depth format's units, as Vulkan's.
            rs.depthBiasEnable = VK_TRUE; // the factors are dynamic (setDynamicState)
        }
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (s.depthFormat) {
            ds.depthTestEnable = (s.depthControl >> 1) & 1;
            ds.depthWriteEnable = ((s.depthControl >> 2) & 1) && !s.depthWriteOff;
            ds.depthCompareOp = VkCompareOp((s.depthControl >> 4) & 7);
            if (s.stencil) {
                ds.stencilTestEnable = s.depthControl & 1;
                auto face = [&](uint32_t func, uint32_t ops) { // masks and reference are dynamic
                    VkStencilOpState f{};
                    f.compareOp = VkCompareOp(func);
                    f.failOp = stencilOp(ops & 0xF);
                    f.passOp = stencilOp((ops >> 4) & 0xF);
                    f.depthFailOp = stencilOp((ops >> 8) & 0xF);
                    return f;
                };
                ds.front = face((s.depthControl >> 8) & 7, s.stencilControl);
                ds.back = (s.depthControl >> 7) & 1 ? face((s.depthControl >> 20) & 7, s.stencilControl >> 12) : ds.front;
            }
        }
        VkPipelineColorBlendAttachmentState att[8]{};
        uint32_t colorCount = 0;
        VkFormat formats[8]{};
        for (uint32_t i = 0; i < 8; ++i) {
            formats[i] = VkFormat(s.colorFormats[i]);
            if (formats[i]) colorCount = i + 1;
            const uint32_t b = s.blend[i];
            att[i].colorWriteMask = (s.writeMasks >> (i * 4)) & 0xF;
            att[i].blendEnable = (b >> 30) & 1;
            att[i].srcColorBlendFactor = blendFactor(b & 0x1F);
            att[i].colorBlendOp = blendOp((b >> 5) & 7);
            att[i].dstColorBlendFactor = blendFactor((b >> 8) & 0x1F);
            if ((b >> 29) & 1) {
                att[i].srcAlphaBlendFactor = blendFactor((b >> 16) & 0x1F);
                att[i].alphaBlendOp = blendOp((b >> 21) & 7);
                att[i].dstAlphaBlendFactor = blendFactor((b >> 24) & 0x1F);
            } else {
                att[i].srcAlphaBlendFactor = att[i].srcColorBlendFactor;
                att[i].alphaBlendOp = att[i].colorBlendOp;
                att[i].dstAlphaBlendFactor = att[i].dstColorBlendFactor;
            }
            // Integer targets cannot blend.
            if (formats[i] && isIntegerFormat(formats[i])) att[i].blendEnable = VK_FALSE;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = colorCount;
        cb.pAttachments = att;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT,          VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_BLEND_CONSTANTS,   VK_DYNAMIC_STATE_DEPTH_BIAS,
                                      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                      VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = uint32_t(std::size(dyn));
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        ri.colorAttachmentCount = colorCount;
        ri.pColorAttachmentFormats = formats;
        ri.depthAttachmentFormat = VkFormat(s.depthFormat);
        if (s.stencil) ri.stencilAttachmentFormat = VkFormat(s.depthFormat);

        VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gci.pNext = &ri;
        gci.stageCount = uint32_t(stages.size());
        gci.pStages = stages.data();
        gci.pVertexInputState = &vi;
        gci.pInputAssemblyState = &ia;
        gci.pViewportState = &vp;
        gci.pRasterizationState = &rs;
        gci.pMultisampleState = &ms;
        gci.pDepthStencilState = &ds;
        gci.pColorBlendState = &cb;
        gci.pDynamicState = &dy;
        gci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        const VkResult r = vkCreateGraphicsPipelines(vk_.device, pipelineCache_, 1, &gci, nullptr, &p);
        if (r != VK_SUCCESS) std::fprintf(stderr, "[gpu] graphics pipeline creation failed: %s\n", vkResultName(r));
        return p;
    }

    void setDynamicState(const DynamicState& d, uint32_t depthControl) {
        // Scale is in 1/16ths of the slope; the offset is in the depth format's units, as Vulkan's.
        if (!bound_.state || std::memcmp(bound_.polyOffset, d.polyOffset, sizeof d.polyOffset) != 0) {
            std::memcpy(bound_.polyOffset, d.polyOffset, sizeof d.polyOffset);
            rec([c = cmd_, k = std::bit_cast<float>(d.polyOffset[2]), cl = std::bit_cast<float>(d.polyOffset[0]),
                 sl = std::bit_cast<float>(d.polyOffset[1]) / 16.0f] { vkCmdSetDepthBias(c, k, cl, sl); });
        }
        const uint32_t back = (depthControl >> 7) & 1 ? d.stencilMasksBf : d.stencilMasks;
        if (bound_.state && bound_.stencil[0] == d.stencilMasks && bound_.stencil[1] == back) return;
        bound_.stencil[0] = d.stencilMasks, bound_.stencil[1] = back;
        rec([c = cmd_, front = d.stencilMasks, back] {
            for (auto [face, m] : {std::pair{VK_STENCIL_FACE_FRONT_BIT, front}, std::pair{VK_STENCIL_FACE_BACK_BIT, back}}) {
                vkCmdSetStencilCompareMask(c, face, (m >> 8) & 0xFF);
                vkCmdSetStencilWriteMask(c, face, (m >> 16) & 0xFF);
                vkCmdSetStencilReference(c, face, m & 0xFF);
            }
        });
    }
    // What the command buffer being recorded has bound and set, so unchanged state is not set
    // again for every draw (most of a crowd's draws share it). Reset by begin().
    struct Bound {
        VkPipeline pipeline = VK_NULL_HANDLE;
        bool state = false; // the dynamic state below has been set
        uint32_t blend[4] = {}, polyOffset[3] = {}, stencil[2] = {};
        VkViewport viewport{};
        VkRect2D scissor{};
        VkBuffer index = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    } bound_;

    static bool isIntegerFormat(VkFormat f) {
        switch (f) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return true;
        default: return false;
        }
    }

    VkPipeline computePipeline(ShaderModule* cs, VkPipelineLayout layout) {
        auto it = computePipelines_.find(cs->id);
        if (it != computePipelines_.end()) return it->second;
        VkPipeline p = VK_NULL_HANDLE;
        if (cs->module) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = cs->module;
            ci.stage.pName = "main";
            ci.layout = layout;
            VK_CHECK(vkCreateComputePipelines(vk_.device, pipelineCache_, 1, &ci, nullptr, &p));
            ++newPipelines_;
        }
        return computePipelines_[cs->id] = p;
    }

    // ---- resources -----------------------------------------------------------------------------

    VkSampler sampler(const uint32_t* s, bool compare) {
        std::array<uint32_t, 5> key{s[0], s[1], s[2], s[3], compare};
        auto it = samplers_.find(key);
        if (it != samplers_.end()) return it->second;
        auto wrap = [](uint32_t m) {
            switch (m) {
            case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case 2: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case 3: case 5: case 7: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            }
        };
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.addressModeU = wrap(s[0] & 7);
        ci.addressModeV = wrap((s[0] >> 3) & 7);
        ci.addressModeW = wrap((s[0] >> 6) & 7);
        ci.magFilter = ((s[2] >> 20) & 3) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.minFilter = ((s[2] >> 22) & 3) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.mipmapMode = ((s[2] >> 26) & 3) == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.minLod = float(s[1] & 0xFFF) / 256.0f;
        ci.maxLod = float((s[1] >> 12) & 0xFFF) / 256.0f;
        ci.compareEnable = compare;
        ci.compareOp = VkCompareOp((s[0] >> 12) & 7);
        ci.borderColor = compare ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        VkSampler smp = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSampler(vk_.device, &ci, nullptr, &smp));
        return samplers_[key] = smp;
    }

    static VkImageViewType viewType(uint32_t type) {
        switch (type) {
        case 10: return VK_IMAGE_VIEW_TYPE_3D;
        case 11: return VK_IMAGE_VIEW_TYPE_CUBE;
        case 12: case 13: return VK_IMAGE_VIEW_TYPE_2D_ARRAY; // 1D images are 2D, one row (as in the shaders)
        default: return VK_IMAGE_VIEW_TYPE_2D;
        }
    }

    static VkComponentMapping swizzle(uint32_t dstSel) {
        auto one = [](uint32_t s) {
            switch (s) {
            case 0: return VK_COMPONENT_SWIZZLE_ZERO;
            case 1: return VK_COMPONENT_SWIZZLE_ONE;
            case 4: return VK_COMPONENT_SWIZZLE_R;
            case 5: return VK_COMPONENT_SWIZZLE_G;
            case 6: return VK_COMPONENT_SWIZZLE_B;
            case 7: return VK_COMPONENT_SWIZZLE_A;
            default: return VK_COMPONENT_SWIZZLE_ZERO;
            }
        };
        return {one(dstSel & 7), one((dstSel >> 3) & 7), one((dstSel >> 6) & 7), one((dstSel >> 9) & 7)};
    }

    // A placeholder image for resources that cannot be resolved.
    VkImageView dummyView(VkImageViewType type, char kind, bool storage) {
        const uint32_t key = uint32_t(type) * 16 + uint32_t(kind == 'u') * 2 + uint32_t(kind == 'i') * 4 + storage;
        auto it = dummies_.find(key);
        if (it != dummies_.end()) return it->second;
        const VkFormat f = kind == 'u' ? VK_FORMAT_R32G32B32A32_UINT : kind == 'i' ? VK_FORMAT_R32G32B32A32_SINT
                                                                                    : VK_FORMAT_R8G8B8A8_UNORM;
        auto img = std::make_unique<Image>();
        const uint32_t layers = (type == VK_IMAGE_VIEW_TYPE_CUBE) ? 6 : 1;
        createImage(*img, f, 1, 1,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false,
                    type, layers);
        VkImageView v = view(*img, f, type, kIdentity);
        dummyImages_.push_back(std::move(img));
        return dummies_[key] = v;
    }

    VkImageView dummyDepthView() {
        if (dummyDepth_) return dummyDepth_;
        auto img = std::make_unique<Image>();
        createImage(*img, VK_FORMAT_D32_SFLOAT, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, true);
        dummyDepth_ = view(*img, VK_FORMAT_D32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D, kIdentity);
        dummyImages_.push_back(std::move(img));
        return dummyDepth_;
    }

    // Uploads a texture from guest memory (linear layouts only for now).
    GuestTexture* guestTexture(const gcn::ImageResource& t, VkFormat format) {
        const uint32_t df = t.dataFormat();
        const bool bc = isBlockCompressed(df);
        const uint32_t bpe = bytesPerElement(df);
        // Sizes in elements (4x4 blocks for compressed formats).
        const uint32_t w = t.width(), h = t.height();
        const uint32_t ew = bc ? (w + 3) / 4 : w, eh = bc ? (h + 3) / 4 : h;
        const uint32_t pitchPx = std::max(t.pitch(), w);
        const uint32_t ep = bc ? (pitchPx + 3) / 4 : pitchPx;
        const uint32_t layers = t.type() == 11 ? 6 : (t.type() == 13 || t.type() == 10) ? t.lastArray() + 1 : 1;
        const uint32_t slices = t.type() == 10 ? t.depth() : layers;
        GuestTexture& tex = textures_[{t.t[0], t.t[1], t.t[2], t.t[3] & ~0xFFFu, t.t[4], t.t[5]}];
        if (tex.img.image && tex.checkedFrame == frame_) return &tex;
        // Re-reading every sampled texture each frame cost ~8 ms in fights and ~15 ms in crowded
        // areas, though most never change. One unchanged for 60 frames is fully hashed on every
        // 8th frame it is used and fingerprinted on the others, which catches whole rewrites at
        // once; a small partial rewrite is seen up to 8 uses late. Counting uses rather than
        // frames matters: the title video alternates two textures frame by frame, and a schedule
        // by frame number never checked one of them. Another T# at the same address is another entry.
        // Each texture has its own phase: textures loaded together otherwise all came due on the
        // same frame, a ~20 ms spike every 8th frame in gameplay.
        const size_t span = size_t(ew) * eh * bpe * slices;
        // A full hash finished on a hasher thread is taken at the very next use, due or not: a
        // partial rewrite (new glyphs in a font atlas) then shows one use late, not eight.
        bool asyncChanged = false;
        if (tex.check && tex.check->state.load(std::memory_order_acquire) == 2) {
            tex.check->state.store(0, std::memory_order_relaxed);
            asyncChanged = tex.check->result != tex.hash;
        }
        if (!asyncChanged && tex.img.image && frame_ - tex.changedFrame >= 60 && (++tex.uses + tex.phase) % 8 != 0) {
            tex.checkedFrame = frame_;
            // Unchanged for 10 s: fingerprinted on every other use only. A syscall each, they
            // took ~16% of the recorder looking up the steps after New Game (hundreds of textures).
            if (frame_ - tex.changedFrame >= 600 && (tex.uses + tex.phase) % 2 != 0) return &tex;
            if (sampleHash(t.base(), span) == tex.sample) return &tex;
        }
        uint32_t mode = t.tileMode();
        size_t slice = tiledSliceSize(mode, ep, eh, bpe);
        if (!slice) {
            warnOnce("texture tile mode " + std::to_string(mode) + " is read as linear");
            mode = 8;
            slice = tiledSliceSize(mode, ep, eh, bpe);
        }
        const size_t tiled = slice * slices;
        const VkDeviceSize bytes = VkDeviceSize(ew) * eh * bpe * slices;
        if (!bytes || bytes > kRingSize / 2 || tiled > (256u << 20)) return nullptr;
        // The full check of a texture already uploaded runs on a hasher thread, its result taken
        // at the next use (~2.7 ms a frame off the recorder in crowds). The fingerprint is still
        // taken now, so a texture rewritten as a whole (a video frame) is uploaded at once; a
        // partial rewrite shows one use later than before. LPN_SYNC_TEX_CHECKS=1: all in line.
        if (tex.img.image && asyncTexChecks_ && !asyncChanged) {
            tex.checkedFrame = frame_;
            if (sampleHash(t.base(), span) == tex.sample) {
                if (!tex.check) tex.check = std::make_shared<AsyncHash>();
                AsyncHash& c = *tex.check;
                const int st = c.state.load(std::memory_order_acquire);
                if (st == 2) {
                    c.state.store(0, std::memory_order_relaxed);
                    if (c.result == tex.hash) return &tex;
                    // changed (or unreadable): checked again in line below
                } else {
                    if (st == 0) queueTexHash(tex.check, t.base(), tiled);
                    return &tex;
                }
            }
        }
        uint64_t h64;
        {
            ProfScope ps(prof_, kProfTexRead);
            ++prof_.texChecks;
            h64 = hasher_.hash(t.base(), tiled);
            if (!h64) {
                warnOnce("texture at 0x" + hex(t.base()) + " is not readable");
                return nullptr;
            }
        }
        tex.checkedFrame = frame_;
        tex.sample = sampleHash(t.base(), span);
        if (!tex.img.image || tex.hash != h64) tex.changedFrame = frame_;
        if (tex.img.image && tex.hash == h64) return &tex;
        ProfScope pu(prof_, kProfTexUpload);
        scratch_.resize(tiled);
        if (!readGuest(t.base(), scratch_.data(), tiled)) return nullptr;
        // Debugging: LPN_DUMP_TEX_BPE=<bytes per element> writes small textures of that element size
        // as uploaded (still tiled) to tex-<address>-<w>x<h>-mode<m>.bin.
        if (static const uint32_t dumpBpe = getenv("LPN_DUMP_TEX_BPE") ? std::atoi(getenv("LPN_DUMP_TEX_BPE")) : 0;
            dumpBpe == bpe && ew * eh <= 1024)
            if (FILE* f = std::fopen(("tex-" + hex(t.base()) + "-" + std::to_string(ew) + "x" + std::to_string(eh) + "-mode" +
                                      std::to_string(mode) + ".bin").c_str(), "wb")) {
                std::fwrite(scratch_.data(), 1, tiled, f);
                std::fclose(f);
            }
        ++prof_.texUploads;
        prof_.texBytes += bytes;
        if (!ringFits(bytes)) {
            flush();
            begin();
        }
        const VkDeviceSize at = ringAlloc(bytes, 16);
        uint8_t* dst = static_cast<uint8_t*>(ringPtr_) + at;
        for (uint32_t s = 0; s < slices; ++s)
            detile(mode, scratch_.data() + s * slice, dst + size_t(s) * ew * eh * bpe, ew, eh, ep, bpe);
        if (!tex.img.image) {
            tex.phase = textureCount_++ % 8;
            const VkImageViewType vt = viewType(t.type());
            if (!createImage(tex.img, format, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false,
                             vt, slices))
                return nullptr;
        }
        tex.hash = h64;
        fullBarrier();
        VkBufferImageCopy c{};
        c.bufferOffset = at;
        c.bufferRowLength = bc ? ew * 4 : ew;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, t.type() == 10 ? 1 : layers};
        c.imageExtent = {w, h, t.type() == 10 ? t.depth() : 1};
        rec([cb = cmd_, buf = ring_, im = tex.img.image, c] { vkCmdCopyBufferToImage(cb, buf, im, VK_IMAGE_LAYOUT_GENERAL, 1, &c); });
        return &tex;
    }

    void queueTexHash(const std::shared_ptr<AsyncHash>& c, uint64_t addr, size_t size) {
        c->state.store(1, std::memory_order_relaxed);
        {
            std::lock_guard lock(texHashMutex_);
            texHashJobs_.push_back({c, addr, size});
        }
        texHashCv_.notify_one();
    }
    void texHashWorker() {
        profile::nameThread("lpn-texcheck");
        for (;;) {
            TexHashJob j;
            {
                std::unique_lock lock(texHashMutex_);
                texHashCv_.wait(lock, [&] { return !texHashJobs_.empty(); });
                j = std::move(texHashJobs_.front());
                texHashJobs_.pop_front();
            }
            j.check->result = hashGuestSerial(j.addr, j.size);
            j.check->state.store(2, std::memory_order_release);
        }
    }
    struct TexHashJob {
        std::shared_ptr<AsyncHash> check;
        uint64_t addr = 0;
        size_t size = 0;
    };
    const bool asyncTexChecks_ = getenv("LPN_SYNC_TEX_CHECKS") == nullptr;
    std::mutex texHashMutex_;
    std::condition_variable texHashCv_;
    std::deque<TexHashJob> texHashJobs_;

    struct ImageBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        float scale[2] = {1, 1};
    };

    void gpuWrote(Surface& s) {
        s.writtenFrame = frame_;
        s.memSample = 0;
        if (s.cpuOwned) ++imageGen_; // sampled as a render target again, not from memory
        s.cpuOwned = false;
    }
    // Whether a render target's memory has been rewritten (by the CPU) since the GPU last
    // rendered into it two or more frames ago; it is then sampled from memory like a texture.
    bool cpuWritten(Surface& s, const gcn::ImageResource& t) {
        if (t.storage) { // a shader writes it
            gpuWrote(s);
            return false;
        }
        if (s.cpuOwned) return true;
        if (frame_ < s.writtenFrame + 2 || s.checkedFrame == frame_) return false; // once a frame
        s.checkedFrame = frame_;
        const uint64_t sample = sampleHash(t.base(), size_t(t.pitch()) * t.height() * s.elementBytes);
        if (!s.memSample) s.memSample = sample;
        else if (sample && sample != s.memSample) s.cpuOwned = true, ++imageGen_;
        return s.cpuOwned;
    }

    // Draws in a frame resolve the same few hundred T#/S# pairs over and over (a crowd repeats its
    // materials): the first resolution in a frame does the checks, later ones reuse its result
    // until an image is created or a render target goes back to the GPU (imageGen_).
    struct ImageMemo {
        uint32_t t[8], s[4];
        bool compare;
        uint64_t frame = ~0ull, gen = 0;
        ImageBinding b;
        VkImage sampled; // the render target it samples, if any (for feedback checks)
    };
    static constexpr size_t kImageMemos = 1024;
    ImageBinding resolveImage(const gcn::ImageResource& t) {
        if (t.storage || trace_) return resolveImageUncached(t); // storage: marks the target written
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t w : t.t) h = (h ^ w) * 0x100000001b3ull;
        for (uint32_t w : t.s) h = (h ^ w) * 0x100000001b3ull;
        ImageMemo& m = imageMemos_[(h ^ (h >> 29) ^ t.compare) & (kImageMemos - 1)];
        if (m.frame == frame_ && m.gen == imageGen_ && m.compare == t.compare && std::equal(t.t, t.t + 8, m.t) &&
            std::equal(t.s, t.s + 4, m.s)) {
            if (m.sampled) sampledSurfaces_.push_back(m.sampled);
            return m.b;
        }
        const size_t before = sampledSurfaces_.size();
        const ImageBinding b = resolveImageUncached(t);
        std::copy(t.t, t.t + 8, m.t);
        std::copy(t.s, t.s + 4, m.s);
        m.compare = t.compare;
        m.frame = frame_;
        m.gen = imageGen_;
        m.b = b;
        m.sampled = sampledSurfaces_.size() > before ? sampledSurfaces_.back() : VK_NULL_HANDLE;
        return b;
    }
    ImageBinding resolveImageUncached(const gcn::ImageResource& t) {
        ImageBinding b;
        const VkImageViewType vt = viewType(t.type());
        const uint32_t nf = t.numFormat();
        const char kind = nf == 4 ? 'u' : nf == 5 ? 'i' : 'f';
        if (!t.storage) b.sampler = sampler(t.s, t.compare);
        const uint64_t addr = t.base();
        VkFormat format = textureFormat(t.dataFormat(), nf);
        const VkComponentMapping swz = t.storage ? kIdentity : swizzle(t.dstSel());

        if (auto it = depths_.find(addr); it != depths_.end()) {
            Image& img = it->second.img;
            sampledSurfaces_.push_back(img.image);
            b.view = view(img, img.format == VK_FORMAT_D32_SFLOAT_S8_UINT ? img.format : img.format, vt,
                          kIdentity, VK_IMAGE_ASPECT_DEPTH_BIT);
            b.scale[0] = float(t.width()) / float(img.width);
            b.scale[1] = float(t.height()) / float(img.height);
            return b;
        }
        if (t.compare) {
            b.view = dummyDepthView();
            warnOnce("depth-compare texture without a depth buffer at 0x" + hex(addr));
            return b;
        }
        // A texture of another element size (a compressed one, say) at a render target's address
        // is not that render target: the memory was reused, so it is read from memory.
        if (auto it = colors_.find(addr); it != colors_.end() && format != VK_FORMAT_UNDEFINED &&
                                          bytesPerElement(t.dataFormat()) == it->second.elementBytes &&
                                          !cpuWritten(it->second, t)) {
            Surface& s = it->second;
            sampledSurfaces_.push_back(s.img.image);
            b.view = view(s.img, format, vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
                          swz);
            b.scale[0] = float(t.width()) / float(s.img.width);
            b.scale[1] = float(t.height()) / float(s.img.height);
            return b;
        }
        if (t.storage) {
            // A compute shader writing an image no draw has rendered yet: make it a surface.
            if (format != VK_FORMAT_UNDEFINED) {
                Surface* s = colorSurface(addr, t.width(), t.height(), format, bytesPerElement(t.dataFormat()));
                if (s) gpuWrote(*s);
                if (s) b.view = view(s->img, format, VK_IMAGE_VIEW_TYPE_2D, kIdentity);
            }
            if (!b.view) b.view = dummyView(vt, kind, true);
            return b;
        }
        if (format == VK_FORMAT_UNDEFINED) {
            warnOnce("texture format " + std::to_string(t.dataFormat()) + "/" + std::to_string(nf) + " unsupported");
        } else if (GuestTexture* g = guestTexture(t, format)) {
            b.view = view(g->img, format, vt, swz);
        }
        if (!b.view) b.view = dummyView(vt, kind, false);
        return b;
    }

    // Binds a stage's resources: draw data, buffers and images.
    // LPN_RACE_CHECK=1: guest buffers the GPU reads straight from memory are hashed when a draw
    // is recorded and again once the GPU has finished, before the batch's fences fire. A
    // change means the CPU rewrote them while the GPU could still be reading them.
    struct RaceRead {
        uint64_t size, hash, frame, draw, shader;
        char kind; // 'b' buffer, 'i' index buffer
        uint64_t batch;
    };
    struct GpuWrite {
        uint64_t addr, size, batch;
    };
    std::mutex raceMutex_;
    std::deque<GpuWrite> gpuWrites_; // memory the GPU (shaders, the command processor) wrote lately
    uint64_t raceShader_ = 0;        // the draw's pixel or compute shader
    void noteWrite(uint64_t addr, uint64_t size) {
        std::lock_guard lock(raceMutex_);
        gpuWrites_.push_back({addr, size, submitted_ + 1});
        while (gpuWrites_.size() > 65536) gpuWrites_.pop_front();
    }
    const bool raceCheck_ = getenv("LPN_RACE_CHECK") != nullptr;
    std::unordered_map<uint64_t, RaceRead> raceReads_; // by address, in the batch being recorded
    std::atomic<uint64_t> raceChanged_{0}, raceChecked_{0};
    void noteRead(uint64_t addr, uint64_t size, char kind) {
        if (size > (1u << 20)) return;
        auto [it, fresh] = raceReads_.try_emplace(addr);
        if (!fresh && it->second.size >= size) return;
        it->second = {size, memoryHash(addr, size), frame_, drawIndex_, raceShader_, kind, submitted_ + 1};
    }
    void checkReads(std::unordered_map<uint64_t, RaceRead>& reads) {
        for (auto& [addr, r] : reads) {
            ++raceChecked_;
            if (!r.hash || memoryHash(addr, r.size) == r.hash) continue;
            bool gpu = false; // written by the GPU itself since
            {
                std::lock_guard lock(raceMutex_);
                for (const GpuWrite& w : gpuWrites_)
                    if (w.batch >= r.batch && w.addr < addr + r.size && addr < w.addr + w.size) {
                        gpu = true;
                        break;
                    }
            }
            if (gpu) continue;
            if (raceChanged_++ < 60)
                std::fprintf(stderr, "[race] %s 0x%" PRIx64 " (%" PRIu64 " bytes) read by draw %" PRIu64 " of frame %" PRIu64
                                     " (shader 0x%" PRIx64 ") changed before the GPU finished\n",
                             r.kind == 'i' ? "index buffer" : "buffer", addr, r.size, r.draw, r.frame, r.shader);
        }
    }

    // ---- read-only buffers in VRAM ----------------------------------------------------------------
    // Guest memory is imported host memory, so every vertex, index and constant fetch crossed
    // PCIe (a third of the GPU's time in a crowd). A buffer of 64 KiB or more that shaders only
    // read gets a copy in VRAM, made by the GPU in order with the draws. It is checked like a
    // texture: fingerprinted on its first use in each frame and fully hashed (on a hasher thread)
    // on every 8th; a change copies it again. Memory the command processor writes is copied again
    // at its next use; memory a shader, a GPU copy or a fill writes is never shadowed, as the copy
    // would go stale. A buffer that keeps changing is left in host memory for a while.
    // LPN_NO_VRAM_BUFFERS=1 reads every buffer from host memory as before.
    struct VramBuffer {
        uint64_t base = 0, size = 0;
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        uint64_t sample = 0, hash = 0;
        uint64_t checkedFrame = ~0ull, changedFrame = 0, lastUse = 0, hostUntil = 0;
        uint32_t uses = 0, phase = 0, changes = 0;
        bool gpuWritten = false, dirty = false;
        std::shared_ptr<AsyncHash> check;
    };
    struct PairHash {
        size_t operator()(const std::pair<uint64_t, uint64_t>& k) const { return std::hash<uint64_t>()(k.first * 0x9E3779B97F4A7C15ull ^ k.second); }
    };
    static constexpr uint64_t kVramMin = 64 << 10, kVramMax = 64ull << 20, kVramBudget = 1536ull << 20;
    static constexpr size_t kVramEntries = 1500; // each is a memory allocation of its own
    const bool vramOn_ = getenv("LPN_NO_VRAM_BUFFERS") == nullptr;
    std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<VramBuffer>, PairHash> vram_;
    std::multimap<uint64_t, VramBuffer*> vramByBase_; // for overlap queries
    std::map<uint64_t, uint64_t> gpuWrittenRanges_;  // [start, end), merged
    uint64_t vramBytes_ = 0, vramCount_ = 0;
    std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t, PairHash> vramCandidates_; // first seen in frame

    // Each VRAM buffer overlapping [lo, hi), via fn.
    template <typename F> void forVramOverlapping(uint64_t lo, uint64_t hi, F&& fn) {
        for (auto it = vramByBase_.lower_bound(lo >= kVramMax ? lo - kVramMax : 0); it != vramByBase_.end() && it->first < hi; ++it)
            if (it->second->base + it->second->size > lo) fn(*it->second);
    }
    // The GPU writes [addr, addr + size): copies of it would go stale.
    void gpuWrites(uint64_t addr, uint64_t size) {
        if (!vramOn_ || !size) return;
        const uint64_t end = addr + size;
        auto it = gpuWrittenRanges_.upper_bound(addr);
        if (it != gpuWrittenRanges_.begin() && std::prev(it)->second >= end) return; // known
        uint64_t lo = addr, hi = end;
        if (it != gpuWrittenRanges_.begin() && std::prev(it)->second >= lo) --it;
        while (it != gpuWrittenRanges_.end() && it->first <= hi) {
            lo = std::min(lo, it->first);
            hi = std::max(hi, it->second);
            it = gpuWrittenRanges_.erase(it);
        }
        gpuWrittenRanges_[lo] = hi;
        forVramOverlapping(addr, end, [](VramBuffer& v) { v.gpuWritten = true; });
    }
    bool gpuWritten(uint64_t addr, uint64_t size) const {
        auto it = gpuWrittenRanges_.upper_bound(addr + size - 1);
        return it != gpuWrittenRanges_.begin() && std::prev(it)->second > addr;
    }
    void cpWrites(uint64_t addr, uint64_t size) {
        if (vramOn_) forVramOverlapping(addr, addr + size, [](VramBuffer& v) { v.dirty = true; });
    }

    // A fingerprint of imported guest memory, read directly: 64 samples of 16 bytes. Memory the
    // GPU reads through an import has to stay mapped anyway, so no syscall guards the reads.
    static uint64_t importedSample(uint64_t base, uint64_t size) {
        uint64_t h = 0xcbf29ce484222325ull;
        const uint64_t stride = std::max<uint64_t>(16, (size / 64) & ~15ull);
        for (uint64_t off = 0; off + 16 <= size; off += stride) {
            uint64_t w[2];
            std::memcpy(w, reinterpret_cast<const void*>(base + off), 16);
            h = ((h ^ w[0]) * 0x100000001b3ull ^ w[1]) * 0x100000001b3ull;
        }
        return h | 1;
    }

    // A VRAM copy of [base, base + size) for reading, or false to read host memory.
    bool vramBuffer(uint64_t base, uint64_t size, VkBuffer& out, VkDeviceSize& off) {
        if (!vramOn_ || size < kVramMin || size > kVramMax || base % 4 || size % 4) return false;
        auto found = vram_.find({base, size});
        if (found == vram_.end()) {
            // Only a buffer used again in a later frame gets a copy: per-frame data in rings sits at
            // a new address each frame, and copying it would cost more than it saves.
            if (gpuWritten(base, size)) return false;
            auto [cand, fresh] = vramCandidates_.try_emplace({base, size}, frame_);
            if (fresh || cand->second == frame_) return false;
            vramCandidates_.erase(cand);
            VkBuffer imported;
            VkDeviceSize at;
            if (!guestBuffer(base, size, imported, at)) return false; // mapped (importedSample reads it directly)
            if (vram_.size() >= kVramEntries) evictVram();
            found = vram_.emplace(std::make_pair(base, size), std::make_unique<VramBuffer>()).first;
        }
        auto& slot = found->second;
        if (!slot->size) {
            slot->base = base, slot->size = size, slot->phase = uint32_t(vramCount_++ % 8);
            slot->gpuWritten = gpuWritten(base, size);
            vramByBase_.emplace(base, slot.get());
        }
        VramBuffer& v = *slot;
        if (v.gpuWritten || frame_ < v.hostUntil) return false;
        v.lastUse = frame_;
        if (v.checkedFrame != frame_ || v.dirty) {
            v.checkedFrame = frame_;
            const uint64_t smp = importedSample(base, size);
            bool changed = v.dirty || (v.buf && smp != v.sample);
            v.dirty = false;
            if (v.buf && !changed) {
                if (!v.check) v.check = std::make_shared<AsyncHash>();
                const int st = v.check->state.load(std::memory_order_acquire);
                if (st == 2) {
                    v.check->state.store(0, std::memory_order_relaxed);
                    changed = v.check->result != v.hash;
                } else if (st == 0 && (++v.uses + v.phase) % 8 == 0) {
                    queueTexHash(v.check, base, size);
                }
            }
            if (changed) {
                if (frame_ - v.changedFrame < 30 && ++v.changes >= 2) { // keeps changing: host memory for a while
                    v.hostUntil = frame_ + 600;
                    v.changes = 0;
                    return false;
                }
                v.changedFrame = frame_;
            }
            v.sample = smp;
            if (!v.buf || changed) {
                VkBuffer src;
                VkDeviceSize soff;
                if (!guestBuffer(base, size, src, soff)) return false;
                if (!v.buf && !allocVram(v)) return false;
                v.hash = hasher_.hash(base, size); // what the copy holds (the game does not write memory the GPU is about to read)
                if (v.check) v.check->state.store(0, std::memory_order_relaxed); // a result in flight is about the old contents
                fullBarrier();
                const VkBufferCopy bc{soff, 0, size};
                rec([c = cmd_, src, dst = v.buf, bc] { vkCmdCopyBuffer(c, src, dst, 1, &bc); });
                ++prof_.vramCopies, prof_.vramCopyBytes += size;
            }
        }
        out = v.buf;
        off = 0;
        return true;
    }
    bool allocVram(VramBuffer& v) {
        if (vramBytes_ + v.size > kVramBudget) evictVram();
        if (vramBytes_ + v.size > kVramBudget) return false;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = v.size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(vk_.device, &bci, nullptr, &v.buf) != VK_SUCCESS) return v.buf = VK_NULL_HANDLE, false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(vk_.device, v.buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = vk_.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk_.device, &mai, nullptr, &v.mem) != VK_SUCCESS) {
            vkDestroyBuffer(vk_.device, v.buf, nullptr);
            v.buf = VK_NULL_HANDLE;
            return false;
        }
        VK_CHECK(vkBindBufferMemory(vk_.device, v.buf, v.mem, 0));
        vramBytes_ += v.size;
        return true;
    }
    // Frees the copies used longest ago (a quarter of them), once the GPU is done with them.
    void evictVram() {
        std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> byUse;
        for (auto& [k, v] : vram_)
            if (v->lastUse + 2 < frame_) byUse.push_back({v->lastUse, k});
        std::sort(byUse.begin(), byUse.end());
        for (size_t i = 0; i < std::max<size_t>(byUse.size() / 4, std::min<size_t>(byUse.size(), 16)); ++i) {
            auto it = vram_.find(byUse[i].second);
            VramBuffer* v = it->second.get();
            for (auto b = vramByBase_.lower_bound(v->base); b != vramByBase_.end() && b->first == v->base; ++b)
                if (b->second == v) {
                    vramByBase_.erase(b);
                    break;
                }
            if (v->buf) {
                vramBytes_ -= v->size;
                afterFlush_.push_back([this, buf = v->buf, mem = v->mem] {
                    vkDestroyBuffer(vk_.device, buf, nullptr);
                    vkFreeMemory(vk_.device, mem, nullptr);
                });
            }
            vram_.erase(it);
        }
    }

    void bindResources(gcn::Program& p, uint32_t base, VkShaderStageFlags, std::vector<VkWriteDescriptorSet>& writes,
                       std::vector<VkDescriptorBufferInfo>& bufInfos, std::vector<VkDescriptorImageInfo>& imgInfos) {
        ProfScope pb(prof_, kProfBind);
        uint32_t n = base + 1;
        for (auto& b : p.buffers) {
            uint64_t size = b.stride() ? uint64_t(b.stride()) * b.numRecords() : b.numRecords();
            size = std::min<uint64_t>(size ? size : 16, std::min<uint64_t>(vk_.props.limits.maxStorageBufferRange, 64ull << 20));
            BufferRange r;
            VkDeviceSize off = 0;
            if (b.store && b.known) gpuWrites(b.base(), size);
            if (b.known && ((!b.store && vramBuffer(b.base(), size, r.buffer, off)) || guestBuffer(b.base(), size, r.buffer, off))) {
                if (raceCheck_) b.store ? noteWrite(b.base(), size) : noteRead(b.base(), size, 'b');
                r.offset = off & ~(storageAlign_ - 1);
                r.byteOffset = uint32_t(off - r.offset);
                r.range = std::min<VkDeviceSize>(size + r.byteOffset, vk_.props.limits.maxStorageBufferRange);
                p.data[b.data] = r.byteOffset;
                p.data[b.data + 1] = b.stride();
                p.data[b.data + 2] = b.numRecords();
                p.data[b.data + 3] = uint32_t(r.range);
            } else {
                r.buffer = dummyBuffer_;
                r.offset = 0;
                r.range = 4096;
                p.data[b.data] = p.data[b.data + 1] = p.data[b.data + 2] = p.data[b.data + 3] = 0;
            }
            bufInfos.push_back({r.buffer, r.offset, r.range});
            writes.push_back(kBufferWrite); // copied from a constant: built on the stack it stalled on store forwarding
            writes.back().dstBinding = n++;
        }
        for (auto& im : p.images) {
            const ImageBinding ib = resolveImage(im);
            if (trace_)
                std::fprintf(traceOut_, "[trace]   tex 0x%" PRIx64 " %ux%u fmt %u/%u type %u tile %u sel %03o pc 0x%" PRIx64 "%s\n",
                             im.base(), im.width(), im.height(), im.dataFormat(), im.numFormat(), im.type(), im.tileMode(),
                             im.dstSel(), im.pc,
                             colors_.count(im.base()) ? " (render target)" : depths_.count(im.base()) ? " (depth)" : "");
            float scale[2] = {ib.scale[0], ib.scale[1]};
            std::memcpy(&p.data[im.data], scale, 8);
            imgInfos.push_back({ib.sampler, ib.view, VK_IMAGE_LAYOUT_GENERAL});
            writes.push_back(im.storage ? kStorageImageWrite : kSampledImageWrite);
            writes.back().dstBinding = n++;
        }
        // Draw data last: resource resolution above fills parts of it.
        const VkDeviceSize bytes = p.data.size() * 4;
        const VkDeviceSize at = ringAlloc(bytes, storageAlign_);
        std::memcpy(static_cast<uint8_t*>(ringPtr_) + at, p.data.data(), bytes);
        bufInfos.push_back({ring_, at, bytes});
        writes.push_back(kBufferWrite);
        writes.back().dstBinding = base;
    }
    static constexpr VkWriteDescriptorSet kBufferWrite = descriptorWrite(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    static constexpr VkWriteDescriptorSet kStorageImageWrite = descriptorWrite(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    static constexpr VkWriteDescriptorSet kSampledImageWrite = descriptorWrite(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    // Points each write at its info entry (the vectors are complete by now).
    static void linkWrites(std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorBufferInfo>& bufs,
                           std::vector<VkDescriptorImageInfo>& imgs) {
        size_t b = 0, i = 0;
        for (auto& w : writes) {
            if (w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) w.pBufferInfo = &bufs[b++];
            else w.pImageInfo = &imgs[i++];
        }
    }

    // ---- draws ---------------------------------------------------------------------------------

    gcn::LaunchState launch(const GpuRegs& r, gcn::Stage stage) {
        ProfScope pl(aprof_, kProfLaunch);
        gcn::LaunchState l;
        l.stage = stage;
        switch (stage) {
        case gcn::Stage::Vertex:
            l.entry = shaderAddress(r, reg::kPgmLoVs);
            std::memcpy(l.userData, &r.sh[reg::kUserDataVs], 64);
            l.rsrc1 = r.sh[reg::kRsrc1Vs];
            l.rsrc2 = r.sh[reg::kRsrc2Vs];
            l.glClipZ = !((r.ctx[reg::kPaClClipCntl] >> 19) & 1);
            break;
        case gcn::Stage::Pixel:
            l.entry = shaderAddress(r, reg::kPgmLoPs);
            std::memcpy(l.userData, &r.sh[reg::kUserDataPs], 64);
            l.rsrc1 = r.sh[reg::kRsrc1Ps];
            l.rsrc2 = r.sh[reg::kRsrc2Ps];
            l.psInputEna = r.ctx[::gpu::ctx::kSpiPsInputEna];
            std::memcpy(l.psInputCntl, &r.ctx[reg::kSpiPsInputCntl], sizeof(l.psInputCntl));
            l.numInterp = r.ctx[reg::kSpiPsInControl] & 0x3F;
            l.colorFormat = r.ctx[reg::kSpiShaderColFormat];
            l.depthExport = r.ctx[reg::kDbShaderControl] & 1;
            break;
        case gcn::Stage::Compute:
            l.entry = shaderAddress(r, reg::kPgmLoCs);
            std::memcpy(l.userData, &r.sh[reg::kUserDataCs], 64);
            l.rsrc1 = r.sh[reg::kRsrc1Cs];
            l.rsrc2 = r.sh[reg::kRsrc2Cs];
            for (int i = 0; i < 3; ++i) l.localSize[i] = std::max<uint32_t>(r.sh[reg::kNumThreadX + i] & 0xFFFF, 1);
            break;
        }
        return l;
    }

    // The stage's last launch in a worker, and its program (see analyzed()).
    struct LastLaunch {
        gcn::LaunchState launch;
        uint32_t bindingBase = 0;
        uint64_t submit = 0;
        CachedProgram* program = nullptr;
    };
    // An analysis worker's own state (see onAnalyzeDraw).
    struct AnalysisCtx {
        explicit AnalysisCtx(Overlays& o) : pages(o) {}
        PageCache pages;
        const gcn::ReadMemory read = [this](uint64_t a, void* d, size_t n) { return pages.read(a, d, n); };
        std::unordered_map<std::string, ProgramVariants> programs;
        size_t programCount = 0;
        gcn::Program replayScratch; // its storage cycles through adopt() (no allocations per launch)
        std::vector<uint8_t> codeBytes;
        LastLaunch last[3]; // per stage
        bool memoHit = false; // the last analyzed() call repeated the launch before it
        uint64_t draws = 0;
        struct {
            uint64_t draw = 0;
            ShaderModule *vsm = nullptr, *psm = nullptr;
            GraphicsState gs{};
        } lastPrefetch;
        std::unordered_set<uint64_t> prefetched; // pipelines prefetchPipeline() queued, by state and programs
        Profiler prof;
    };
    // A draw's analysed programs, handed from an analysis worker to the recorder: copies of the
    // per-draw parts (the cached programs are re-targeted by the next draws meanwhile).
    struct DrawAnalysis {
        bool ready = false; // filled in (analysesMutex_)
        bool ok = false;
        bool repeat = false; // both programs repeat the worker's previous draw's (analysis only)
        const void *vsc = nullptr, *psc = nullptr; // the cached programs (analysis only)
        ShaderModule *vsm = nullptr, *psm = nullptr; // a dispatch's module is vsm
        gcn::Program vs, ps;
    };
    // A draw's launches as the registers give them, for a worker to analyse.
    struct DrawJob {
        enum Kind : uint8_t { kNone, kDraw, kDispatch } kind = kNone;
        bool hasPs = false;
        bool prefetch = false; // gs holds the pipeline state (but vsOutputs)
        gcn::LaunchState vs, ps; // a dispatch's launch is vs
        GraphicsState gs{};
        DrawAnalysis* out = nullptr;
    };
    static constexpr size_t kAnalysisBatch = 32;

    void analysisWorker(AnalysisCtx& c) {
        profile::nameThread("lpn-analyze-w");
        for (;;) {
            std::vector<DrawJob> b;
            {
                std::unique_lock lock(batchMutex_);
                batchCv_.wait(lock, [&] { return !analysisBatches_.empty(); });
                b = std::move(analysisBatches_.front());
                analysisBatches_.pop_front();
                ++batchesBusy_;
            }
            runBatch(c, b);
            {
                std::lock_guard lock(batchMutex_);
                --batchesBusy_;
                b.clear();
                spareBatches_.push_back(std::move(b));
            }
            batchIdleCv_.notify_all();
        }
    }
    void runBatch(AnalysisCtx& c, std::vector<DrawJob>& b) {
        for (DrawJob& j : b) runJob(c, j);
        {
            std::lock_guard lock(analysesMutex_);
            for (DrawJob& j : b) j.out->ready = true;
        }
        analysesCv_.notify_all();
    }

    // The program a launch runs, analysed (or reused) with its shader module; the returned
    // program's data is fresh for this draw. Null when there is no shader.
    CachedProgram* analyzed(AnalysisCtx& c, const gcn::LaunchState& l, uint32_t bindingBase) {
        if (!l.entry) return nullptr;
        ProfScope ps(c.prof, kProfAnalyze);
        // Most draws repeat the previous one's launch with only another index range (72% of the
        // 6,350 draws looking up the steps after New Game): with memory unchanged since (same
        // submit_), its program is already re-targeted to this launch.
        LastLaunch& last = c.last[size_t(l.stage)];
        if (last.program && last.submit == submit_ && last.bindingBase == bindingBase && sameLaunch(last.launch, l)) {
            ++c.prof.analysisHits, ++c.prof.repeats;
            c.memoHit = true;
            return last.program;
        }
        c.memoHit = false;
        if (c.prof.on && last.program) { // LPN_PROFILE: why the launch did not repeat
            if (last.launch.entry != l.entry) ++c.prof.missWhy[0];
            else if (last.submit != submit_) ++c.prof.missWhy[1];
            else {
                bool user = false;
                for (int i = 0; i < 16; ++i)
                    if (last.launch.userData[i] != l.userData[i]) ++c.prof.missUser[i], user = true;
                ++c.prof.missWhy[user ? 2 : 3];
            }
        }
        last.program = analyzedUncached(c, l, bindingBase);
        last.launch = l;
        last.bindingBase = bindingBase;
        last.submit = submit_;
        return last.program;
    }
    static bool sameLaunch(const gcn::LaunchState& a, const gcn::LaunchState& b) {
        return a.stage == b.stage && a.entry == b.entry && !std::memcmp(a.userData, b.userData, sizeof(a.userData)) &&
               a.rsrc1 == b.rsrc1 && a.rsrc2 == b.rsrc2 && a.glClipZ == b.glClipZ && a.psInputEna == b.psInputEna &&
               !std::memcmp(a.psInputCntl, b.psInputCntl, sizeof(a.psInputCntl)) && a.numInterp == b.numInterp &&
               a.colorFormat == b.colorFormat && a.colorUint == b.colorUint && a.colorSint == b.colorSint &&
               !std::memcmp(a.colorSwap, b.colorSwap, sizeof(a.colorSwap)) &&
               !std::memcmp(a.colorComps, b.colorComps, sizeof(a.colorComps)) && a.depthExport == b.depthExport &&
               !std::memcmp(a.localSize, b.localSize, sizeof(a.localSize));
    }
    CachedProgram* analyzedUncached(AnalysisCtx& c, const gcn::LaunchState& l, uint32_t bindingBase) {
        auto& walks = c.programs[launchKey(l, bindingBase)].walks;
        gcn::Program& r = c.replayScratch;
        for (size_t i = 0; i < walks.size(); ++i) {
            if (!codeUnchanged(c, *walks[i].shapes.front())) {
                c.programCount -= walks[i].shapes.size();
                walks.erase(walks.begin() + ptrdiff_t(i--));
                continue;
            }
            if (!gcn::replay(walks[i].shapes.front()->prog, l, c.read, r)) continue;
            if (i) std::rotate(walks.begin(), walks.begin() + ptrdiff_t(i), walks.begin() + ptrdiff_t(i) + 1);
            auto& shapes = walks.front().shapes;
            for (size_t j = 0; j < shapes.size(); ++j)
                if (gcn::adopt(shapes[j]->prog, r)) {
                    ++c.prof.analysisHits;
                    if (j) std::rotate(shapes.begin(), shapes.begin() + ptrdiff_t(j), shapes.begin() + ptrdiff_t(j) + 1);
                    return shapes.front().get();
                }
            // The same branches with other resource formats (a material with other textures).
            if (shapes.size() >= kMaxShapes) {
                shapes.pop_back();
                --c.programCount;
            }
            shapes.insert(shapes.begin(), std::make_unique<CachedProgram>());
            ++c.programCount;
            return analyze(c, *shapes.front(), l, bindingBase);
        }
        if (walks.size() >= kMaxWalks) {
            c.programCount -= walks.back().shapes.size();
            walks.pop_back();
        }
        walks.insert(walks.begin(), ProgramWalk{});
        walks.front().shapes.push_back(std::make_unique<CachedProgram>());
        ++c.programCount;
        return analyze(c, *walks.front().shapes.front(), l, bindingBase);
    }

    // Only the bytes of the instructions count: a shader's last page often holds other data that
    // changes. Memory cannot change within a submission (see PageCache), so one check per
    // submission is enough.
    bool codeUnchanged(AnalysisCtx& c, CachedProgram& p) {
        if (!p.valid) return false;
        if (p.checkedSubmit == submit_) return true;
        if (hashCode(c, p.code) != p.codeHash) return false;
        p.checkedSubmit = submit_;
        return true;
    }
    static uint64_t hashCode(AnalysisCtx& c, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
        uint64_t h = 0xcbf29ce484222325ull;
        for (const auto& [start, end] : ranges) {
            c.codeBytes.assign((end - start + 7) & ~7ull, 0);
            if (!c.pages.read(start, c.codeBytes.data(), end - start)) return 0;
            h = (h ^ hashBytes(c.codeBytes.data(), c.codeBytes.size())) * 0x100000001b3ull;
        }
        return h | 1;
    }

    CachedProgram* analyze(AnalysisCtx& c, CachedProgram& p, const gcn::LaunchState& l, uint32_t bindingBase) {
        ++c.prof.analyses;
        p.prog = gcn::analyze(l, c.read);
        p.code.clear();
        for (const auto& [pc, in] : p.prog.code) {
            if (!p.code.empty() && p.code.back().second == pc) p.code.back().second = pc + in.size;
            else p.code.emplace_back(pc, pc + in.size);
        }
        p.codeHash = hashCode(c, p.code);
        p.checkedSubmit = submit_;
        p.valid = p.codeHash != 0;
        ProfScope pl(c.prof, kProfShaderLookup);
        if (p.prog.launch.entry) {
            std::lock_guard lock(shadersMutex_); // the workers share the modules
            p.module = shader(p.prog, bindingBase);
        } else {
            p.module = nullptr;
        }
        return p.prog.launch.entry ? &p : nullptr;
    }

    static void takeDrawPart(gcn::Program& to, const gcn::Program& from) {
        to.launch = from.launch;
        to.data = from.data;
        to.buffers = from.buffers;
        to.images = from.images;
        to.warnings = from.warnings;
    }

    void prepareDispatch(const GpuRegs& r, DrawJob& j) {
        static const uint64_t skipCs = getenv("LPN_SKIP_CS") ? std::strtoull(getenv("LPN_SKIP_CS"), nullptr, 0) : 0;
        if (skipCs && shaderAddress(r, reg::kPgmLoCs) == skipCs) return; // debugging: what a shader costs
        j.vs = launch(r, gcn::Stage::Compute);
        j.kind = DrawJob::kDispatch;
    }

    // The register-only part of draw(): which draws render, the pixel shader's launch as the
    // render targets' formats shape it, and the pipeline state for prefetchPipeline().
    void prepareDraw(const GpuRegs& r, DrawJob& j) {
        const uint32_t cbMode = (r.ctx[reg::kCbColorControl] >> 4) & 7;
        const uint32_t dbRender = r.ctx[reg::kDbRenderControl];
        if (cbMode > 1 || (dbRender & 0x1C)) return;
        const uint32_t prim = r.uconfig[reg::kPrimitiveType];
        if (prim == 0 || prim == 8 || prim == 13) return;
        j.vs = launch(r, gcn::Stage::Vertex);
        j.kind = DrawJob::kDraw;
        j.hasPs = shaderAddress(r, reg::kPgmLoPs) != 0;
        const uint32_t targetMask = r.ctx[reg::kCbTargetMask] & r.ctx[reg::kCbShaderMask];
        if (j.hasPs) {
            j.ps = launch(r, gcn::Stage::Pixel);
            for (uint32_t i = 0; i < 8 && cbMode == 1; ++i) {
                const uint32_t* cb = &r.ctx[reg::kCbColorBase + i * reg::kCbColorStride];
                const uint32_t info = cb[::gpu::ctx::kCbColorInfo];
                const uint32_t fmt = (info >> 2) & 0x1F, num = (info >> 8) & 7;
                if (!fmt || !((targetMask >> (i * 4)) & 0xF) || colorFormat(fmt, num) == VK_FORMAT_UNDEFINED) continue;
                psTargetLaunch(j.ps, i, fmt, num, info);
            }
        }
        GraphicsState& gs = j.gs;
        gs = GraphicsState{};
        bool any = false;
        for (uint32_t i = 0; i < 8 && cbMode == 1; ++i) {
            const uint32_t* cb = &r.ctx[reg::kCbColorBase + i * reg::kCbColorStride];
            const uint32_t info = cb[::gpu::ctx::kCbColorInfo];
            const uint32_t fmt = (info >> 2) & 0x1F, num = (info >> 8) & 7;
            if (!fmt || !((targetMask >> (i * 4)) & 0xF)) continue;
            const VkFormat vf = colorFormat(fmt, num);
            if (vf == VK_FORMAT_UNDEFINED) continue;
            gs.colorFormats[i] = vf;
            gs.writeMasks |= ((targetMask >> (i * 4)) & 0xF) << (i * 4);
            gs.blend[i] = r.ctx[reg::kCbBlendControl + i];
            any = true;
        }
        const uint32_t zInfo = r.ctx[reg::kDbZInfo];
        if (zInfo & 3) {
            const bool stencil = r.ctx[reg::kDbStencilInfo] & 1;
            gs.depthFormat = stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : (zInfo & 3) == 1 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
            gs.stencil = stencil;
            gs.depthControl = r.ctx[reg::kDbDepthControl];
            gs.stencilControl = r.ctx[reg::kDbStencilControl];
            gs.depthWriteOff = r.ctx[reg::kDbRenderControl] & 1;
            any = true;
        }
        gs.topology = prim;
        gs.modeCntl = r.ctx[reg::kPaSuScModeCntl];
        j.prefetch = any;
    }

    // On a worker: analyses one draw or dispatch into its DrawAnalysis.
    void runJob(AnalysisCtx& c, DrawJob& j) {
        DrawAnalysis& a = *j.out;
        if (j.kind == DrawJob::kDispatch) {
            CachedProgram* csc = analyzed(c, j.vs, 0);
            if (!csc) return;
            a.ok = true;
            a.vsm = csc->module;
            takeDrawPart(a.vs, csc->prog);
            return;
        }
        if (j.kind != DrawJob::kDraw) return;
        ++c.draws;
        CachedProgram* vsc = analyzed(c, j.vs, 0);
        if (!vsc) return;
        a.repeat = c.memoHit;
        a.vsc = vsc;
        a.ok = true;
        a.vsm = vsc->module;
        takeDrawPart(a.vs, vsc->prog);
        if (j.hasPs)
            if (CachedProgram* psc = analyzed(c, j.ps, kPsBindingBase)) {
                a.psm = psc->module;
                takeDrawPart(a.ps, psc->prog);
                a.repeat = a.repeat && c.memoHit;
                a.psc = psc;
            }
        prefetchPipeline(c, j, a);
    }

    // The analysis runs ahead of the recorder (by up to a frame): it works out each draw's
    // pipeline from the registers as draw() will, and queues any not built yet. A frame that
    // brings dozens of new pipelines then has them built in parallel by all the workers, rather
    // than one after another as the recorder reaches each draw. A wrong guess only costs a build.
    void prefetchPipeline(AnalysisCtx& c, const DrawJob& j, const DrawAnalysis& a) {
        static const bool off = getenv("LPN_NO_PIPELINE_PREFETCH") != nullptr;
        if (off || syncPipelines_ || !j.prefetch || !a.vsm || !a.vsm->module || (a.psm && !a.psm->module)) return;
        GraphicsState gs = j.gs;
        gs.vsOutputs = a.vsm->translation.outputs;
        // The previous draw queued this pipeline already when its programs and state repeat.
        const bool repeat = a.repeat && c.lastPrefetch.draw + 1 == c.draws && c.lastPrefetch.vsm == a.vsm &&
                            c.lastPrefetch.psm == a.psm && !std::memcmp(&c.lastPrefetch.gs, &gs, sizeof(gs));
        c.lastPrefetch = {c.draws, a.vsm, a.psm, gs};
        if (repeat) return;
        // Seen before with the same programs (their bindings follow from them): already queued.
        // Building the record below (strings, a locked map) cost ~5% of the analysis thread. A
        // program freed and its address reused can only skip a prefetch: the recorder builds it.
        uint64_t seen = 0xcbf29ce484222325ull;
        auto mix = [&](const void* p, size_t n) {
            for (size_t i = 0; i < n; ++i) seen = (seen ^ static_cast<const uint8_t*>(p)[i]) * 0x100000001b3ull;
        };
        mix(&gs, sizeof(gs));
        mix(&a.vsc, sizeof(a.vsc));
        mix(&a.psc, sizeof(a.psc));
        if (c.prefetched.size() > 200000) c.prefetched.clear();
        if (!c.prefetched.insert(seen).second) return;
        thread_local std::vector<VkDescriptorSetLayoutBinding> lb;
        lb.clear();
        addBindings(lb, a.vs, 0, VK_SHADER_STAGE_VERTEX_BIT);
        if (a.psm) addBindings(lb, a.ps, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT);
        std::string record = pipelineRecord(gs, a.vsm->name, a.psm ? a.psm->name : std::string(), lb);
        std::shared_ptr<BuiltPipeline> b;
        {
            std::lock_guard lock(buildMutex_);
            auto& slot = built_[record];
            if (slot) return;
            slot = b = std::make_shared<BuiltPipeline>();
        }
        ++c.prof.pipelinesMade;
        queueJob(PipelineJob{gs, a.vsm->module, a.psm ? a.psm->module : VK_NULL_HANDLE, geometryFor(gs), layout(lb).pipeline,
                             std::move(b), std::move(record), false},
                 true);
    }

    static void psTargetLaunch(gcn::LaunchState& l, uint32_t i, uint32_t fmt, uint32_t num, uint32_t info) {
        if (num == 4) l.colorUint |= 1u << i;
        if (num == 5) l.colorSint |= 1u << i;
        l.colorSwap[i] = (info >> 11) & 3;
        static const uint8_t kComps[32] = {0, 1, 1, 2, 1, 2, 3, 3, 4, 4, 4, 2, 4, 3, 4, 0,
                                           3, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        l.colorComps[i] = kComps[fmt];
    }

    void dispatch(const DrawCall& d, const GpuRegs&, DrawAnalysis& a) {
        if (!a.ok) return;
        gcn::Program& cs = a.vs;
        if (!ringFits(1 << 20)) flush();
        begin();
        if (fillDispatch(d, cs)) return;
        ShaderModule* m = a.vsm;
        std::vector<VkDescriptorSetLayoutBinding> lb;
        addBindings(lb, cs, 0, VK_SHADER_STAGE_COMPUTE_BIT);
        Layout& lay = layout(lb);
        VkPipeline pipe = computePipeline(m, lay.pipeline);
        if (!pipe) return;
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> bufs;
        std::vector<VkDescriptorImageInfo> imgs;
        bufs.reserve(64);
        imgs.reserve(64);
        bindResources(cs, 0, VK_SHADER_STAGE_COMPUTE_BIT, writes, bufs, imgs);
        linkWrites(writes, bufs, imgs);
        fullBarrier();
        rec([c = cmd_, pipe] { vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); });
        pushDescriptors(VK_PIPELINE_BIND_POINT_COMPUTE, lay.pipeline, writes, bufs, imgs);
        if (trace_) captureProgram(cs, m, "cs");
        if (trace_)
            for (const auto& b : cs.buffers) {
                uint32_t first = 0;
                readGuest(b.base(), &first, 4);
                std::fprintf(traceOut_, "[trace]   buffer 0x%" PRIx64 " stride %u records %u first 0x%x%s\n", b.base(),
                             b.stride(), b.numRecords(), first, b.store ? " (written)" : "");
            }
        rec([c = cmd_, x = d.groups[0], y = d.groups[1], z = d.groups[2]] { vkCmdDispatch(c, x, y, z); });
        ++stats_.dispatches;
        for (const auto& im : cs.images)
            if (im.storage && im.tileMode() == kTileLinearAligned) writeBack(im);
        for (const auto& b : cs.buffers)
            if (b.store) fastClear(b.base());
    }

    // The game clears buffers with a small compute shader that stores one 16-byte value from its
    // constants into records [0, groups x 64): two stores per work item, 32 per group. Guest buffers
    // live in host memory, where the GPU's scattered dword writes crawl over PCIe (~4 ms for the 4 MB
    // buffer it clears every gameplay frame at 1440p); a transfer fill writes whole lines instead.
    // Only that exact code, filling with one repeated dword, takes this path.
    bool fillDispatch(const DrawCall& d, const gcn::Program& cs) {
        static constexpr uint32_t kFillCode[] = {0xbeeb03ff, 0x0000000c, 0x936ac008, 0xc2820500, 0xd2840004, 0x01a90500,
                                                 0xbf8c007f, 0x7e000204, 0x7e020205, 0x7e040206, 0x7e060207, 0xe01c2000,
                                                 0x80000004, 0xd2ba0004, 0x04110081, 0xe01c2000, 0x80000004, 0xbf810000};
        static const bool off = getenv("LPN_NO_FILL_DISPATCH") != nullptr;
        if (off || trace_ || cs.buffers.empty() || !cs.images.empty() || cs.launch.localSize[0] != 32 ||
            cs.launch.localSize[1] != 1 || cs.launch.localSize[2] != 1)
            return false;
        uint32_t code[std::size(kFillCode)];
        if (!readGuest(cs.launch.entry, code, sizeof code) || std::memcmp(code, kFillCode, sizeof code) != 0) return false;
        // s_buffer_load_dwordx4 s[4:7], s[4:7], 0: the value is the first record of the V# in
        // user data s[4:7].
        const uint64_t constants = cs.launch.userData[4] | (uint64_t(cs.launch.userData[5] & 0xFFF) << 32);
        uint32_t value[4];
        if (cs.launch.userSgprs() < 8 || !readGuest(constants, value, sizeof value)) return false;
        if (value[1] != value[0] || value[2] != value[0] || value[3] != value[0]) return false;
        const gcn::BufferResource& b = cs.buffers[0];
        for (const auto& o : cs.buffers) // both stores go through s[0:3]
            if (!o.known || !o.store || o.base() != b.base() || o.stride() != b.stride() || o.numRecords() != b.numRecords())
                return false;
        if (b.stride() != 16) return false;
        const uint64_t records = std::min<uint64_t>(b.numRecords(), uint64_t(d.groups[0]) * 64);
        if (!records) return true;
        VkBuffer buffer;
        VkDeviceSize offset;
        if (b.base() % 4 || !guestBuffer(b.base(), records * 16, buffer, offset)) return false;
        fullBarrier();
        rec([c = cmd_, buffer, offset, bytes = records * 16, v = value[0]] { vkCmdFillBuffer(c, buffer, offset, bytes, v); });
        if (raceCheck_) noteWrite(b.base(), records * 16);
        gpuWrites(b.base(), records * 16);
        ++stats_.dispatches;
        fastClear(b.base());
        return true;
    }

    // ---- fast clears -----------------------------------------------------------------------------
    // The game clears depth buffers and render targets by filling their HTILE or CMASK metadata
    // from a compute shader; the hardware then reads untouched tiles as the clear value (and a
    // fast-clear-eliminate pass writes it to memory). Surfaces here have no metadata, so a fill
    // of a surface's HTILE/CMASK clears the surface itself.
    struct Metadata {
        uint64_t surface = 0;
        bool depth = false;
        uint32_t clear[2] = {}; // depth (float bits) and stencil, or CB_COLOR_CLEAR_WORD0/1
    };

    void noteMetadata(uint64_t meta, uint64_t surface, bool depth, uint32_t c0, uint32_t c1) {
        if (!meta || meta == surface) return;
        Metadata& m = metadata_[meta];
        m.surface = surface;
        m.depth = depth;
        m.clear[0] = c0;
        m.clear[1] = c1;
    }

    void fastClear(uint64_t meta) {
        auto it = metadata_.find(meta);
        if (it == metadata_.end()) return;
        // Zero low bits (ZMASK for HTILE, the first tile for CMASK) mean "cleared". Memory holds what
        // the last fill wrote, as nothing else writes metadata here.
        uint32_t word = 0;
        if (readGuest(meta, &word, 4) && (word & 0xF)) return;
        const Metadata& m = it->second;
        auto& surfaces = m.depth ? depths_ : colors_;
        auto s = surfaces.find(m.surface);
        if (s == surfaces.end()) return;
        Image& img = s->second.img;
        fullBarrier();
        VkImageSubresourceRange range{img.aspect, 0, 1, 0, img.layers};
        if (m.depth) {
            const VkClearDepthStencilValue v{std::bit_cast<float>(m.clear[0]), m.clear[1] & 0xFF};
            rec([c = cmd_, im = img.image, v, range] { vkCmdClearDepthStencilImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range); });
        } else {
            VkClearColorValue v;
            if (!clearColor(img.format, m.clear, v)) {
                warnOnce("fast clear of a render target in format " + std::to_string(img.format));
                return;
            }
            rec([c = cmd_, im = img.image, v, range] { vkCmdClearColorImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range); });
        }
        if (trace_)
            std::fprintf(traceOut_, "[trace]   fast clear of %s 0x%" PRIx64 " to 0x%x 0x%x\n", m.depth ? "depth" : "target",
                         m.surface, m.clear[0], m.clear[1]);
    }

    // CB_COLOR_CLEAR_WORD0/1 hold one element in memory order, which is the images' component
    // order here (shader exports apply COMP_SWAP).
    static bool clearColor(VkFormat f, const uint32_t w[2], VkClearColorValue& v) {
        v = {};
        auto byte = [&](int i) { return float((w[0] >> (8 * i)) & 0xFF) / 255.0f; };
        auto half = [&](int i) { return unpackHalf(uint16_t(w[i / 2] >> (16 * (i % 2)))); };
        switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM:
            for (int i = 0; i < 4; ++i) v.float32[i] = byte(i);
            return true;
        case VK_FORMAT_R8G8B8A8_SRGB: // the clear value is given in linear space
            for (int i = 0; i < 4; ++i) {
                const float c = byte(i);
                v.float32[i] = i == 3 ? c : c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
            }
            return true;
        case VK_FORMAT_R8G8_UNORM: v.float32[1] = byte(1); [[fallthrough]];
        case VK_FORMAT_R8_UNORM: v.float32[0] = byte(0); return true;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            for (int i = 0; i < 3; ++i) v.float32[i] = float((w[0] >> (10 * i)) & 0x3FF) / 1023.0f;
            v.float32[3] = float(w[0] >> 30) / 3.0f;
            return true;
        case VK_FORMAT_R32_SFLOAT: v.float32[0] = std::bit_cast<float>(w[0]); return true;
        case VK_FORMAT_R32G32_SFLOAT:
            v.float32[0] = std::bit_cast<float>(w[0]);
            v.float32[1] = std::bit_cast<float>(w[1]);
            return true;
        case VK_FORMAT_R16G16B16A16_SFLOAT: v.float32[2] = half(2), v.float32[3] = half(3); [[fallthrough]];
        case VK_FORMAT_R16G16_SFLOAT: v.float32[1] = half(1); [[fallthrough]];
        case VK_FORMAT_R16_SFLOAT: v.float32[0] = half(0); return true;
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: v.uint32[0] = w[0]; return true;
        case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: v.uint32[0] = w[0], v.uint32[1] = w[1]; return true;
        default: return false;
        }
    }

    static float unpackHalf(uint16_t h) {
        const uint32_t sign = uint32_t(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF;
        if (e == 0) return (sign ? -1.0f : 1.0f) * std::ldexp(float(m), -24);
        if (e == 31) return std::bit_cast<float>(sign | 0x7F800000u | (m << 13));
        return std::bit_cast<float>(sign | ((e + 112) << 23) | (m << 13));
    }

    // Images in the linear layout are how the game reads GPU results on the CPU (the measured
    // scene luminance behind auto exposure, for one): copy what a shader stored into one back to
    // guest memory. The game waits for the frame's fences before reading, which fire after the
    // GPU has finished, copy included.
    static constexpr uint32_t kTileLinearAligned = 8;
    void writeBack(const gcn::ImageResource& t) {
        auto it = colors_.find(t.base());
        if (it == colors_.end()) return;
        if (raceCheck_) noteWrite(t.base(), uint64_t(std::max(t.pitch(), t.width())) * t.height() * it->second.elementBytes);
        Surface& s = it->second;
        gpuWrote(s); // its memory changes now, from the GPU
        const uint32_t w = std::min(t.width(), s.img.width), h = std::min(t.height(), s.img.height);
        const uint32_t pitch = std::max(t.pitch(), w);
        const uint64_t bytes = (uint64_t(pitch) * (h - 1) + w) * s.elementBytes;
        VkBuffer buffer;
        VkDeviceSize offset = 0;
        if (!guestBuffer(t.base(), bytes, buffer, offset)) return;
        gpuWrites(t.base(), bytes);
        if (offset % std::max<uint32_t>(s.elementBytes, 4)) {
            warnOnce("linear image at 0x" + hex(t.base()) + " is not aligned for a copy back");
            return;
        }
        fullBarrier();
        VkBufferImageCopy c{};
        c.bufferOffset = offset;
        c.bufferRowLength = pitch;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {w, h, 1};
        rec([cb = cmd_, im = s.img.image, buffer, c] { vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &c); });
        if (trace_)
            std::fprintf(traceOut_, "[trace]   copied %ux%u back to 0x%" PRIx64 "\n", w, h, t.base());
    }

    void draw(const DrawCall& d, const GpuRegs& r, DrawAnalysis& a) {
        const uint32_t cbMode = (r.ctx[reg::kCbColorControl] >> 4) & 7;
        const uint32_t dbRender = r.ctx[reg::kDbRenderControl];
        if (trace_)
            std::fprintf(traceOut_, "[trace]   cb mode %u db render 0x%x depth control 0x%x z info 0x%x stencil info 0x%x control 0x%x"
                                 " ref/mask 0x%x z base 0x%" PRIx64 " htile 0x%" PRIx64 " clear %g prim %u count %u\n",
                         cbMode, dbRender, r.ctx[reg::kDbDepthControl], r.ctx[reg::kDbZInfo], r.ctx[reg::kDbStencilInfo],
                         r.ctx[reg::kDbStencilControl], r.ctx[reg::kDbStencilRefMask],
                         uint64_t(r.ctx[reg::kDbZReadBase]) << 8, uint64_t(r.ctx[reg::kDbHtileDataBase]) << 8,
                         std::bit_cast<float>(r.ctx[reg::kDbDepthClear]), r.uconfig[reg::kPrimitiveType], d.count);
        if (trace_ && (r.ctx[reg::kPaSuScModeCntl] & (3u << 11)))
            std::fprintf(traceOut_, "[trace]   polygon offset scale %g offset %g clamp %g (mode 0x%x)\n",
                         std::bit_cast<float>(r.ctx[reg::kPaSuPolyOffsetFrontScale]),
                         std::bit_cast<float>(r.ctx[reg::kPaSuPolyOffsetFrontOffset]),
                         std::bit_cast<float>(r.ctx[reg::kPaSuPolyOffsetClamp]), r.ctx[reg::kPaSuScModeCntl]);
        if (trace_ && cbMode > 1) {
            const uint32_t* cb = &r.ctx[reg::kCbColorBase];
            std::fprintf(traceOut_, "[trace]   eliminate 0x%" PRIx64 " info 0x%x cmask 0x%" PRIx64 " clear 0x%x 0x%x\n",
                         uint64_t(cb[0]) << 8, cb[::gpu::ctx::kCbColorInfo], uint64_t(cb[reg::kCbColorCmask]) << 8,
                         cb[reg::kCbColorClearWord0], cb[reg::kCbColorClearWord1]);
        }
        if (cbMode > 1) return;           // fast-clear eliminate, resolve, decompress: nothing to do here
        if (dbRender & 0x1C) return;      // depth/stencil copy, resummarize
        const uint32_t prim = r.uconfig[reg::kPrimitiveType];
        if (prim == 0 || prim == 8 || prim == 13) {
            warnOnce("primitive type " + std::to_string(prim) + " not rendered");
            return;
        }

        if (!a.ok) return;
        if (fastDraw(d, r, a)) return;
        fast_.valid = false;
        gcn::Program& vs = a.vs;

        // Render targets. The pass and vectors below are reused from draw to draw (no allocations).
        Pass& pass = drawPass_;
        pass.open = false;
        pass.images.clear();
        pass.written.clear();
        pass.depthImage = VK_NULL_HANDLE;
        GraphicsState gs{};
        DynamicState dyn{};
        VkRenderingAttachmentInfo colors[8]{};
        uint32_t colorCount = 0;
        uint32_t width = 16384, height = 16384;
        const uint32_t targetMask = r.ctx[reg::kCbTargetMask] & r.ctx[reg::kCbShaderMask];
        for (uint32_t i = 0; i < 8 && cbMode == 1; ++i) {
            const uint32_t* cb = &r.ctx[reg::kCbColorBase + i * reg::kCbColorStride];
            const uint32_t info = cb[::gpu::ctx::kCbColorInfo];
            const uint32_t fmt = (info >> 2) & 0x1F, num = (info >> 8) & 7;
            if (!fmt || !((targetMask >> (i * 4)) & 0xF)) continue;
            const VkFormat vf = colorFormat(fmt, num);
            if (vf == VK_FORMAT_UNDEFINED) {
                warnOnce("color format " + std::to_string(fmt) + "/" + std::to_string(num) + " unsupported");
                continue;
            }
            const uint32_t pitch = ((cb[::gpu::ctx::kCbColorPitch] & 0x7FF) + 1) * 8;
            const uint32_t slice = ((cb[::gpu::ctx::kCbColorSlice] & 0x3FFFFF) + 1) * 64;
            Surface* s = colorSurface(uint64_t(cb[0]) << 8, pitch, slice / pitch, vf, bytesPerElement(fmt));
            if (!s) continue;
            gpuWrote(*s);
            colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colors[i].imageView = view(s->img, vf, VK_IMAGE_VIEW_TYPE_2D, kIdentity);
            pass.images.push_back(s->img.image);
            colors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            colors[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorCount = i + 1;
            if (trace_)
                std::fprintf(traceOut_, "[trace]   rt%u 0x%" PRIx64 " %ux%u fmt %u/%u swap %u mask 0x%x blend 0x%x info 0x%x cmask 0x%" PRIx64 "\n", i,
                             uint64_t(cb[0]) << 8, s->img.width, s->img.height, fmt, num, (info >> 11) & 3,
                             (targetMask >> (i * 4)) & 0xF, r.ctx[reg::kCbBlendControl + i], info,
                             uint64_t(cb[reg::kCbColorCmask]) << 8);
            width = std::min(width, s->img.width);
            height = std::min(height, s->img.height);
            gs.colorFormats[i] = vf;
            gs.writeMasks |= ((targetMask >> (i * 4)) & 0xF) << (i * 4);
            gs.blend[i] = r.ctx[reg::kCbBlendControl + i];
            if ((info >> 13) & 1) // FAST_CLEAR: a CMASK fill clears it (see dispatch)
                noteMetadata(uint64_t(cb[reg::kCbColorCmask]) << 8, uint64_t(cb[0]) << 8, false,
                             cb[reg::kCbColorClearWord0], cb[reg::kCbColorClearWord1]);
        }
        for (uint32_t i = colorCount; i < 8; ++i) colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

        // Depth.
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        bool hasDepth = false, clearDepth = false;
        const uint32_t zInfo = r.ctx[reg::kDbZInfo];
        const uint32_t depthControl = r.ctx[reg::kDbDepthControl];
        if (zInfo & 3) {
            const VkFormat df = (zInfo & 3) == 1 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
            const bool stencil = r.ctx[reg::kDbStencilInfo] & 1;
            const VkFormat f = stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : df;
            const uint32_t size = r.ctx[reg::kDbDepthSize];
            const uint32_t dw = ((size & 0x7FF) + 1) * 8, dh = (((size >> 11) & 0x7FF) + 1) * 8;
            if (Surface* s = depthSurface(uint64_t(r.ctx[reg::kDbZReadBase]) << 8, dw, dh, f)) {
                hasDepth = true;
                depth.imageView = view(s->img, f, VK_IMAGE_VIEW_TYPE_2D, kIdentity, s->img.aspect);
                pass.images.push_back(s->img.image);
                pass.depthImage = s->img.image;
                depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                gs.depthFormat = f;
                gs.stencil = stencil;
                gs.depthControl = depthControl;
                gs.stencilControl = r.ctx[reg::kDbStencilControl];
                dyn.stencilMasks = r.ctx[reg::kDbStencilRefMask];
                dyn.stencilMasksBf = r.ctx[reg::kDbStencilRefMaskBf];
                width = std::min(width, dw);
                height = std::min(height, dh);
                clearDepth = dbRender & 1;
                if ((zInfo >> 29) & 1) // TILE_SURFACE_ENABLE: an HTILE fill clears it (see dispatch)
                    noteMetadata(uint64_t(r.ctx[reg::kDbHtileDataBase]) << 8, uint64_t(r.ctx[reg::kDbZReadBase]) << 8,
                                 true, r.ctx[reg::kDbDepthClear], r.ctx[reg::kDbStencilClear]);
                gs.depthWriteOff = clearDepth;
            }
        }
        if (!colorCount && !hasDepth) return;

        if (!ringFits(1 << 20)) flush();
        begin();
        ShaderModule* vsm = a.vsm;
        gcn::Program& ps = a.ps;
        ShaderModule* psm = a.psm;
        gs.vs = vsm->id;
        gs.ps = psm ? psm->id : 0;
        gs.topology = prim;
        gs.modeCntl = r.ctx[reg::kPaSuScModeCntl];
        if (gs.modeCntl & (3u << 11))
            for (int i = 0; i < 3; ++i) dyn.polyOffset[i] = r.ctx[reg::kPaSuPolyOffsetClamp + i];
        gs.vsOutputs = vsm->translation.outputs;

        std::vector<VkDescriptorSetLayoutBinding>& lb = drawBindings_;
        lb.clear();
        addBindings(lb, vs, 0, VK_SHADER_STAGE_VERTEX_BIT);
        if (psm) addBindings(lb, ps, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT);
        Layout* layp;
        VkPipeline pipe;
        {
            ProfScope pp(prof_, kProfPipeline);
            layp = &layout(lb);
            pipe = graphicsPipeline(gs, vsm, psm, layp->pipeline, lb);
        }
        if (!pipe) return;
        Layout& lay = *layp;

        std::vector<VkWriteDescriptorSet>& writes = drawWrites_;
        std::vector<VkDescriptorBufferInfo>& bufs = drawBufs_;
        std::vector<VkDescriptorImageInfo>& imgs = drawImgs_;
        writes.clear();
        bufs.clear();
        imgs.clear();
        sampledSurfaces_.clear();
        if (trace_) {
            captureProgram(vs, vsm, "vs");
            if (psm) captureProgram(ps, psm, "ps");
            for (const auto* prog : {&vs, &ps}) {
                if (prog == &vs || psm) {
                    std::fprintf(traceOut_, "[trace]   %s user:", prog == &vs ? "vs" : "ps");
                    for (uint32_t i = 0; i < prog->launch.userSgprs() && i < 16; ++i) std::fprintf(traceOut_, " %x", prog->launch.userData[i]);
                    std::fprintf(traceOut_, "\n");
                }
                if (prog->data.size() <= gcn::data::kSmrd) continue;
                std::fprintf(traceOut_, "[trace]   %s smrd:", prog == &vs ? "vs" : "ps");
                for (size_t i = gcn::data::kSmrd; i < prog->data.size() && i < gcn::data::kSmrd + 24; ++i) {
                    float f;
                    std::memcpy(&f, &prog->data[i], 4);
                    std::fprintf(traceOut_, " %g", f);
                }
                std::fprintf(traceOut_, "\n");
                for (const auto& w : prog->warnings) std::fprintf(traceOut_, "[trace]   warning: %s\n", w.c_str());
            }
        }
        if (!(r.ctx[reg::kPaClVteCntl] & 0xF)) {
            const float t[4] = {2.0f / float(width), 2.0f / float(height), -1.0f, -1.0f};
            std::memcpy(&vs.data[gcn::data::kPosTransform], t, sizeof(t));
        }
        bindResources(vs, 0, VK_SHADER_STAGE_VERTEX_BIT, writes, bufs, imgs);
        if (psm) bindResources(ps, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT, writes, bufs, imgs);
        linkWrites(writes, bufs, imgs);

        // Index buffer.
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        if (d.kind == DrawCall::kIndexed) {
            const uint32_t isz = r.indexType ? 4 : 2;
            if (!vramBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset) &&
                !guestBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset)) return;
            if (trace_ && !traceDir_.empty() && uint64_t(d.indexOffset + d.count) * isz <= (1u << 20)) { // captures: the indices
                std::vector<uint8_t> idx(size_t(d.count) * isz);
                if (readGuest(d.indexAddr + uint64_t(d.indexOffset) * isz, idx.data(), idx.size()))
                    if (FILE* f = std::fopen((traceDir_ + "draw-" + std::to_string(drawIndex_) + "-indices-u" + std::to_string(isz * 8) + ".bin").c_str(), "wb")) {
                        std::fwrite(idx.data(), 1, idx.size(), f);
                        std::fclose(f);
                    }
            }
            if (raceCheck_) noteRead(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, 'i');
        }

        // Consecutive draws into the same targets share a render pass. A new pass starts behind a
        // full barrier, as before every draw, so earlier passes, copies and dispatches are visible.
        // Inside a pass only its own attachments change, so a draw sampling one of them (or one
        // after a draw that wrote memory from a shader) needs a new pass.
        for (uint32_t i = 0; i < colorCount; ++i) pass.views[i] = colors[i].imageView;
        pass.depth = hasDepth ? depth.imageView : VK_NULL_HANDLE;
        pass.colorCount = colorCount;
        pass.stencil = hasDepth && gs.stencil;
        pass.width = width;
        pass.height = height;
        // Sampling an attachment needs a new pass (and its barrier) only once a draw in the open
        // pass has written it: decals and soft particles that read the depth buffer without
        // writing it then share one pass.
        bool feedback = false;
        for (VkImage im : sampledSurfaces_)
            if (std::find(drawWritten_.begin(), drawWritten_.end(), im) != drawWritten_.end()) {
                feedback = true;
                if (prof_.on && pass_.open && pass_.sameTargets(pass)) ++prof_.feedbackOn[im == pass_.depthImage ? 0 : 1];
            }
        if (!pass_.open || feedback || !pass_.sameTargets(pass)) {
            if (prof_.on) ++prof_.passWhy[!pass_.open ? passClosedBy_ : feedback ? kPassFeedback : kPassTargets];
            if (pass_.open) { // to other targets: no barrier unless something since the last one needs it
                rec([c = cmd_] { vkCmdEndRendering(c); });
                pass_.open = false;
            }
            // Draws since the last barrier wrote and sampled only images; another pass needs the
            // barrier when it samples what they wrote, or renders into what they wrote or sampled.
            if (needBarrier_ || feedback || overlaps(pass.images, drawWritten_) || overlaps(pass.images, drawSampled_))
                barrier();
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {width, height}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = colorCount;
            ri.pColorAttachments = keep(colors, colorCount);
            if (hasDepth) {
                ri.pDepthAttachment = keep(&depth, 1);
                if (gs.stencil) ri.pStencilAttachment = ri.pDepthAttachment;
            }
            rec([c = cmd_, ri] { vkCmdBeginRendering(c, &ri); });
            std::swap(pass_, pass); // the old pass's storage is reused by the next draw
            pass_.open = true;
            ++prof_.passes;
        }
        if (bound_.pipeline != pipe) {
            rec([c = cmd_, pipe] { vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); });
            bound_.pipeline = pipe;
        }
        pushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, lay.pipeline, writes, bufs, imgs);
        setViewportScissor(r, width, height);
        if (!bound_.state || std::memcmp(bound_.blend, &r.ctx[reg::kCbBlendRed], 16) != 0) {
            std::memcpy(bound_.blend, &r.ctx[reg::kCbBlendRed], 16);
            rec([c = cmd_, b = std::bit_cast<std::array<float, 4>>(bound_.blend)] { vkCmdSetBlendConstants(c, b.data()); });
        }
        setDynamicState(dyn, gs.depthControl);
        bound_.state = true;
        if (clearDepth) {
            VkClearAttachment ca{};
            ca.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            float z;
            std::memcpy(&z, &r.ctx[reg::kDbDepthClear], 4);
            ca.clearValue.depthStencil = {z, r.ctx[reg::kDbStencilClear] & 0xFF};
            if (gs.stencil && (dbRender & 2)) ca.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            VkClearRect cr{{{0, 0}, {width, height}}, 0, 1};
            rec([c = cmd_, ca, cr] { vkCmdClearAttachments(c, 1, &ca, 1, &cr); });
        }
        if (d.kind == DrawCall::kIndexed) {
            const VkIndexType it = r.indexType ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (bound_.index != indexBuffer || bound_.indexOffset != indexOffset || bound_.indexType != it) {
                rec([c = cmd_, indexBuffer, indexOffset, it] { vkCmdBindIndexBuffer(c, indexBuffer, indexOffset, it); });
                bound_.index = indexBuffer, bound_.indexOffset = indexOffset, bound_.indexType = it;
            }
            rec([c = cmd_, n = d.count, inst = std::max(r.numInstances, 1u), first = d.indexOffset] {
                vkCmdDrawIndexed(c, n, inst, first, 0, 0);
            });
        } else {
            rec([c = cmd_, n = d.count, inst = std::max(r.numInstances, 1u)] { vkCmdDraw(c, n, inst, 0, 0); });
        }
        auto stores = [](const gcn::Program& p) {
            for (const auto& b : p.buffers)
                if (b.store) return true;
            for (const auto& i : p.images)
                if (i.storage) return true;
            return false;
        };
        for (VkImage im : pass_.images) // what this draw wrote
            if ((im != pass_.depthImage || clearDepth || (((depthControl >> 2) & 1) && ((depthControl >> 1) & 1)) ||
                 (gs.stencil && (depthControl & 1) && ((r.ctx[reg::kDbStencilRefMask] >> 16) & 0xFF) &&
                  (r.ctx[reg::kDbStencilControl] & 0xFFFFFF))) &&
                std::find(pass_.written.begin(), pass_.written.end(), im) == pass_.written.end()) {
                pass_.written.push_back(im);
                addOnce(drawWritten_, im);
            }
        for (VkImage im : sampledSurfaces_) addOnce(drawSampled_, im);
        if (stores(vs) || (psm && stores(ps))) {
            endPass();
            passClosedBy_ = kPassStore;
        }
        ++stats_.draws;
        if (trace_ && snap_ && colorCount && snapWanted()) snapshot(uint64_t(r.ctx[reg::kCbColorBase]) << 8);
        // What a repeat of this draw can take as is (see fastDraw).
        bool repeatable = pass_.open && !clearDepth && !trace_ && !raceCheck_ && !snap_;
        for (VkImage im : sampledSurfaces_) // sampling what it renders to: a repeat needs a new pass
            if (std::find(drawWritten_.begin(), drawWritten_.end(), im) != drawWritten_.end()) repeatable = false;
        if (repeatable && !noFastDraws_) {
            FastDraw& f = fast_;
            f.valid = true;
            f.rec = recCount_;
            f.frame = frame_;
            f.imageGen = imageGen_;
            f.kind = d.kind;
            f.indexType = r.indexType;
            f.numInstances = r.numInstances;
            f.prim = prim;
            std::memcpy(f.ctx, r.ctx, sizeof f.ctx);
            f.layout = lay.pipeline;
            f.vsm = vsm;
            f.psm = psm;
            keepDraw_ = true;
        }
    }

    // A draw that repeats the one recorded just before it but for its index range and user data:
    // ~70% of the 6,350 draws looking up the steps after New Game, which change only per-draw
    // constant pointers in their user SGPRs. With nothing recorded since, the same registers,
    // memory and images, and the same resources, its targets, pass, pipeline, dynamic state and
    // resource descriptors all hold: only the draw data (user data words in it) is uploaded and
    // pushed again (push descriptors update incrementally), and the draw recorded.
    // LPN_NO_FAST_DRAWS=1 records every draw in full.
    struct FastDraw {
        bool valid = false;
        uint64_t rec = 0, frame = 0, imageGen = 0;
        DrawCall::Kind kind = DrawCall::kIndexed;
        uint32_t indexType = 0, numInstances = 0, prim = 0;
        uint32_t ctx[::gpu::kContextRegs];
        VkPipelineLayout layout = VK_NULL_HANDLE;
        ShaderModule *vsm = nullptr, *psm = nullptr;
    };
    bool fastDraw(const DrawCall& d, const GpuRegs& r, DrawAnalysis& a) {
        const FastDraw& f = fast_;
        auto miss = [&](FastMiss why) {
            if (prof_.on) ++prof_.fastMiss[why];
            return false;
        };
        if (!f.valid) return miss(kFastNotKept);
        if (f.rec != recCount_) return miss(kFastBetween);
        if (f.frame != frame_ || f.imageGen != imageGen_ || !recording_ || !pass_.open) return miss(kFastFrame);
        if (d.kind != f.kind || a.vsm != f.vsm || a.psm != f.psm || r.indexType != f.indexType ||
            r.numInstances != f.numInstances || r.uconfig[reg::kPrimitiveType] != f.prim || !ringFits(1 << 16))
            return miss(kFastKind);
        if (std::memcmp(r.ctx, f.ctx, sizeof f.ctx) != 0) {
            if (prof_.on)
                for (uint32_t i = 0; i < ::gpu::kContextRegs; ++i)
                    if (r.ctx[i] != f.ctx[i]) ++prof_.fastCtx[i];
            return miss(kFastCtx);
        }
        if (!sameButUserData(a.vs, prevDraw_.vs) || (a.psm && !sameButUserData(a.ps, prevDraw_.ps)))
            return miss(kFastResources);
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        if (d.kind == DrawCall::kIndexed) {
            const uint32_t isz = r.indexType ? 4 : 2;
            if (!vramBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset) &&
                !guestBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset))
                return miss(kFastIndex);
        }
        ProfScope pb(prof_, kProfBind);
        std::vector<VkWriteDescriptorSet>& writes = drawWrites_;
        std::vector<VkDescriptorBufferInfo>& bufs = drawBufs_;
        std::vector<VkDescriptorImageInfo>& imgs = drawImgs_;
        writes.clear();
        bufs.clear();
        imgs.clear();
        // prevDraw_ holds the user data bound now.
        auto data = [&](const gcn::Program& p, gcn::Program& bound, uint32_t base) {
            constexpr size_t kUser = gcn::data::kUserData, kWords = 16;
            if (!std::memcmp(&p.data[kUser], &bound.data[kUser], kWords * 4)) return;
            const VkDeviceSize bytes = p.data.size() * 4;
            const VkDeviceSize at = ringAlloc(bytes, storageAlign_);
            std::memcpy(static_cast<uint8_t*>(ringPtr_) + at, p.data.data(), bytes);
            bufs.push_back({ring_, at, bytes});
            writes.push_back(kBufferWrite);
            writes.back().dstBinding = base;
            std::memcpy(&bound.data[kUser], &p.data[kUser], kWords * 4);
        };
        data(a.vs, prevDraw_.vs, 0);
        if (a.psm) data(a.ps, prevDraw_.ps, kPsBindingBase);
        if (!writes.empty()) pushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, f.layout, writes, bufs, imgs);
        if (d.kind == DrawCall::kIndexed) {
            const VkIndexType it = r.indexType ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (bound_.index != indexBuffer || bound_.indexOffset != indexOffset || bound_.indexType != it) {
                rec([c = cmd_, indexBuffer, indexOffset, it] { vkCmdBindIndexBuffer(c, indexBuffer, indexOffset, it); });
                bound_.index = indexBuffer, bound_.indexOffset = indexOffset, bound_.indexType = it;
            }
            rec([c = cmd_, n = d.count, inst = std::max(r.numInstances, 1u), first = d.indexOffset] {
                vkCmdDrawIndexed(c, n, inst, first, 0, 0);
            });
        } else {
            rec([c = cmd_, n = d.count, inst = std::max(r.numInstances, 1u)] { vkCmdDraw(c, n, inst, 0, 0); });
        }
        fast_.rec = recCount_;
        ++stats_.draws;
        ++prof_.fastDraws;
        return true;
    }
    // Whether `cur` is `prev` (as bindResources and draw() left it) but for its user data words.
    // Fills in the words those fill from prev's (the same resources resolve alike).
    static bool sameButUserData(gcn::Program& cur, const gcn::Program& prev) {
        if (cur.data.size() != prev.data.size() || cur.buffers.size() != prev.buffers.size() ||
            cur.images.size() != prev.images.size() || cur.data.size() < gcn::data::kSmrd)
            return false;
        for (size_t i = 0; i < cur.buffers.size(); ++i) {
            const gcn::BufferResource &x = cur.buffers[i], &y = prev.buffers[i];
            if (std::memcmp(x.sharp, y.sharp, sizeof x.sharp) || x.known != y.known || x.store != y.store || x.data != y.data ||
                x.dfmt != y.dfmt || x.nfmt != y.nfmt)
                return false;
        }
        for (size_t i = 0; i < cur.images.size(); ++i) {
            const gcn::ImageResource &x = cur.images[i], &y = prev.images[i];
            if (std::memcmp(x.t, y.t, sizeof x.t) || std::memcmp(x.s, y.s, sizeof x.s) || x.storage != y.storage ||
                x.compare != y.compare || x.data != y.data)
                return false;
        }
        std::memcpy(&cur.data[gcn::data::kPosTransform], &prev.data[gcn::data::kPosTransform], 16);
        for (const auto& b : prev.buffers) std::memcpy(&cur.data[b.data], &prev.data[b.data], 16);
        for (const auto& im : prev.images) std::memcpy(&cur.data[im.data], &prev.data[im.data], 8);
        constexpr size_t kAfterUser = gcn::data::kUserData + 16;
        return !std::memcmp(&cur.data[kAfterUser], &prev.data[kAfterUser], (cur.data.size() - kAfterUser) * 4);
    }

    void endPass() {
        needBarrier_ = true; // something other than a draw comes next
        if (!pass_.open) return;
        rec([c = cmd_] { vkCmdEndRendering(c); });
        pass_.open = false;
        passClosedBy_ = kPassOther;
    }

    void setViewportScissor(const GpuRegs& r, uint32_t width, uint32_t height) {
        const float* v = reinterpret_cast<const float*>(&r.ctx[reg::kVportXScale]);
        float xs = v[0], xo = v[1], ys = v[2], yo = v[3], zs = v[4], zo = v[5];
        const uint32_t vte = r.ctx[reg::kPaClVteCntl];
        if (!(vte & 1)) xs = 1;
        if (!(vte & 2)) xo = 0;
        if (!(vte & 4)) ys = 1;
        if (!(vte & 8)) yo = 0;
        if (!(vte & 16)) zs = 1;
        if (!(vte & 32)) zo = 0;
        VkViewport vp{};
        vp.x = xo - xs;
        vp.width = 2 * xs;
        vp.y = yo - ys;
        vp.height = 2 * ys;
        if (!(vte & 0xF)) { // positions are already in pixels: the vertex shader maps them to the target
            vp.x = 0;
            vp.y = 0;
            vp.width = float(width);
            vp.height = float(height);
        }
        const bool dx = (r.ctx[reg::kPaClClipCntl] >> 19) & 1;
        vp.minDepth = dx ? zo : zo - zs;
        vp.maxDepth = zo + zs;
        vp.minDepth = std::clamp(vp.minDepth, 0.0f, 1.0f);
        vp.maxDepth = std::clamp(vp.maxDepth, 0.0f, 1.0f);
        if (vp.width < 0) {
            vp.x += vp.width;
            vp.width = -vp.width;
        }
        if (!bound_.state || std::memcmp(&bound_.viewport, &vp, sizeof vp) != 0) rec([c = cmd_, vp] { vkCmdSetViewport(c, 0, 1, &vp); }), bound_.viewport = vp;

        int32_t x0 = 0, y0 = 0, x1 = int32_t(width), y1 = int32_t(height);
        auto clip = [&](uint32_t tl, uint32_t br) {
            if (!br) return; // never programmed
            x0 = std::max<int32_t>(x0, tl & 0x7FFF);
            y0 = std::max<int32_t>(y0, (tl >> 16) & 0x7FFF);
            x1 = std::min<int32_t>(x1, br & 0x7FFF);
            y1 = std::min<int32_t>(y1, (br >> 16) & 0x7FFF);
        };
        clip(r.ctx[reg::kScreenScissorTl], r.ctx[reg::kScreenScissorBr]);
        clip(r.ctx[reg::kGenericScissorTl], r.ctx[reg::kGenericScissorBr]);
        clip(r.ctx[reg::kWindowScissorTl], r.ctx[reg::kWindowScissorBr]);
        if ((r.ctx[0x292] >> 1) & 1) clip(r.ctx[reg::kVportScissorTl], r.ctx[reg::kVportScissorBr]);
        VkRect2D sc{{x0, y0}, {uint32_t(std::max(x1 - x0, 0)), uint32_t(std::max(y1 - y0, 0))}};
        if (trace_)
            std::fprintf(traceOut_, "[trace]   scissors screen %x %x window %x %x generic %x %x vport %x %x mode0 %x\n",
                         r.ctx[reg::kScreenScissorTl], r.ctx[reg::kScreenScissorBr], r.ctx[reg::kWindowScissorTl],
                         r.ctx[reg::kWindowScissorBr], r.ctx[reg::kGenericScissorTl], r.ctx[reg::kGenericScissorBr],
                         r.ctx[reg::kVportScissorTl], r.ctx[reg::kVportScissorBr], r.ctx[0x292]);
        if (trace_)
            std::fprintf(traceOut_, "[trace]   viewport %.1f,%.1f %.1fx%.1f z %.2f..%.2f scissor %d,%d %ux%u vte 0x%x colfmt 0x%x\n",
                         vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth, sc.offset.x, sc.offset.y,
                         sc.extent.width, sc.extent.height, vte, r.ctx[reg::kSpiShaderColFormat]);
        if (!bound_.state || std::memcmp(&bound_.scissor, &sc, sizeof sc) != 0) rec([c = cmd_, sc] { vkCmdSetScissor(c, 0, 1, &sc); }), bound_.scissor = sc;
    }

    Vk vk_;
    std::mutex m_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    struct Batch {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkBuffer ring = VK_NULL_HANDLE;
        VkDeviceMemory ringMemory = VK_NULL_HANDLE;
        void* ringPtr = nullptr;
        bool busy = false;                     // submitted and not yet finished
        std::vector<std::function<void()>> done; // runs once it has finished
    };
    Batch batches_[kBatches];
    uint32_t cur_ = 0;
    uint32_t drawsSinceFlush_ = 0;
    PassWhy passClosedBy_ = kPassOther;
    VkQueryPool timestamps_ = VK_NULL_HANDLE; // LPN_PROFILE: two per batch
    double gpuTickNs_ = 1;
    std::atomic<uint64_t> gpuBusyNs_{0};
    uint32_t textureCount_ = 0; // textures created, for their check phases
    std::mutex doneMutex_; // guards Batch::busy and done, inFlight_, completing_, the counters
    std::condition_variable doneCv_;
    std::deque<uint32_t> inFlight_;
    bool completing_ = false;
    uint64_t submitted_ = 0, completed_ = 0; // batches
    struct DisplaySlot { // display readback: the GPU copies into it, the display thread reads it
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        void* ptr = nullptr;
        VkDeviceSize size = 0;
    };
    DisplaySlot displays_[kDisplaySlots];
    std::vector<std::function<void()>> pendingDone_; // due after the batch being recorded
    bool sync_ = false;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE; // the current batch's
    // The command thread's queue (see CmdChunk): chunks handed over in order, and spare ones.
    bool syncRecord_ = getenv("LPN_SYNC_RECORD") != nullptr;
    std::unique_ptr<CmdChunk> chunk_ = std::make_unique<CmdChunk>(); // being filled
    std::mutex chunkMutex_;
    std::condition_variable chunkCv_;
    std::deque<std::unique_ptr<CmdChunk>> chunks_, spareChunks_;
    uint32_t drawsSinceHandOff_ = 0;
    bool recording_ = false;
    VkBuffer ring_ = VK_NULL_HANDLE, dummyBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyMemory_ = VK_NULL_HANDLE;
    void* ringPtr_ = nullptr;
    VkDeviceSize ringUsed_ = 0, storageAlign_ = 16;
    std::vector<std::function<void()>> afterFlush_;
    std::vector<uint8_t> scratch_;
    GuestHasher hasher_;
    std::chrono::steady_clock::time_point lastFrameEnd_ = std::chrono::steady_clock::now();
    uint64_t windowFrames_ = 0, slowFrames_ = 0; // LPN_PROFILE: frames in the report window, and those over 17 ms
    double worstFrameMs_ = 0;
    uint64_t frameNs_[kProfCount] = {}, frameAnalyses_ = 0, framePipelines_ = 0;

    std::map<uint64_t, Import> imports_;
    std::unordered_map<uint64_t, Surface> colors_, depths_;
    std::unordered_map<uint64_t, Metadata> metadata_; // by HTILE/CMASK address
    std::unordered_map<std::array<uint32_t, 6>, GuestTexture, WordsHash> textures_;
    std::map<std::array<uint32_t, 5>, VkSampler> samplers_;
    std::unordered_map<std::string, std::unique_ptr<ShaderModule>> shaders_;
    std::unordered_map<uint64_t, VkShaderModule> expanders_;
    std::unordered_map<std::string, Layout> layouts_;
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::unordered_map<uint32_t, VkPipeline> computePipelines_;
    std::unordered_map<uint32_t, VkImageView> dummies_;
    VkImageView dummyDepth_ = VK_NULL_HANDLE;
    std::vector<std::unique_ptr<Image>> dummyImages_;
    std::set<std::string> warned_;
    uint64_t frame_ = 0, drawIndex_ = 0, traceFrame_ = ~0ull;
    FILE* traceOut_ = stderr;   // LPN_GPU_TRACE output; a capture's trace.txt
    std::string traceDir_;      // where trace dumps go ("" = working directory, else ends in /)
    std::string pendingCapture_; // folder of a requested capture (see captureNextFrame)
    bool trace_ = false, snap_ = false, dumpGlsl_ = false;
    uint32_t snapScale_ = 1;
    std::vector<std::pair<uint64_t, uint64_t>> snapDraws_;
    bool snapWanted() const {
        if (snapDraws_.empty()) return true;
        for (auto [a, b] : snapDraws_)
            if (drawIndex_ >= a && drawIndex_ <= b) return true;
        return false;
    }

public:
    bool computeOnly_ = false;

private:
    struct {
        uint64_t draws = 0, dispatches = 0;
    } stats_;
    Profiler prof_;
    Overlays overlays_;
    uint64_t overlayIds_ = 0;
    uint64_t recCount_ = 0; // calls rec()'d (see fastDraw)
    FastDraw fast_;
    DrawAnalysis prevDraw_; // the last draw recorded in full, if fast_.valid
    bool keepDraw_ = false; // onDraw keeps the draw as prevDraw_
    const bool noFastDraws_ = getenv("LPN_NO_FAST_DRAWS") != nullptr;
    Profiler aprof_; // the analysis thread's slots and counts, merged into prof_'s reports
    std::mutex analysesMutex_;
    std::condition_variable analysesCv_; // a DrawAnalysis is ready
    std::deque<DrawAnalysis> analyses_; // one per draw or dispatch, for the recorder
    std::vector<DrawAnalysis> spareAnalyses_; // used ones, back from the recorder
    std::deque<uint64_t> dmaOverlays_, writeOverlays_; // the analysis thread's overlay of each DMA / WRITE_DATA
    std::vector<std::unique_ptr<AnalysisCtx>> actx_; // one per worker (one without workers)
    unsigned analysisWorkers_ = 0;
    std::vector<DrawJob> pending_; // the analysis thread's batch being filled
    std::mutex batchMutex_;
    std::condition_variable batchCv_, batchIdleCv_;
    std::deque<std::vector<DrawJob>> analysisBatches_; // for the workers
    std::vector<std::vector<DrawJob>> spareBatches_;
    unsigned batchesBusy_ = 0;
    std::mutex shadersMutex_; // guards shaders_ (the workers create modules)
    uint64_t submit_ = 0; // counts the page caches' clears and changes; the workers read it between barriers
    // The open render pass (see draw) and the render-target surfaces the current draw samples.
    struct Pass {
        bool open = false;
        VkImageView views[8] = {}, depth = VK_NULL_HANDLE;
        uint32_t colorCount = 0, width = 0, height = 0;
        bool stencil = false;
        std::vector<VkImage> images; // attachments
        VkImage depthImage = VK_NULL_HANDLE;
        std::vector<VkImage> written; // attachments a draw in the pass has written
        bool sameTargets(const Pass& o) const {
            return colorCount == o.colorCount && depth == o.depth && stencil == o.stencil && width == o.width &&
                   height == o.height && std::equal(views, views + colorCount, o.views);
        }
    } pass_, drawPass_; // the open pass; the draw being recorded's
    std::vector<VkDescriptorSetLayoutBinding> drawBindings_;
    std::vector<VkWriteDescriptorSet> drawWrites_;
    std::vector<VkDescriptorBufferInfo> drawBufs_;
    std::vector<VkDescriptorImageInfo> drawImgs_;
    std::vector<VkImage> sampledSurfaces_;
    std::array<ImageMemo, kImageMemos> imageMemos_{};
    uint64_t imageGen_ = 1;
    // Since the last barrier: images draws wrote and sampled, and whether anything else (a copy,
    // dispatch, upload, clear, shader store or new batch) needs one before the next pass.
    std::vector<VkImage> drawWritten_, drawSampled_;
    bool needBarrier_ = true;
    std::string spirvDir_; // "-": no SPIR-V cache
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::string pipelineCachePath_;
    std::atomic<uint32_t> newPipelines_{0};
    bool syncPipelines_ = false;
    std::mutex buildMutex_, listMutex_; // built_ and jobs_; pipelines.list
    std::condition_variable buildCv_;
    std::unordered_map<std::string, std::shared_ptr<BuiltPipeline>> built_; // by pipelineRecord
    std::deque<PipelineJob> jobs_, backgroundJobs_;
    std::mutex layoutMutex_; // layouts_ and expanders_: the recorder, the analysis thread (prefetch) and startup builds
    std::string pipelineListPath_;
    FILE* pipelineList_ = nullptr;
    std::chrono::steady_clock::time_point lastPipelineSave_{};
    std::atomic<bool> pipelineSaving_{false};
};

Renderer* gRenderer = nullptr;

} // namespace

bool startRenderer(bool computeOnly) {
    auto* r = new Renderer;
    r->computeOnly_ = computeOnly;
    if (!r->init()) {
        delete r;
        return false;
    }
    gRenderer = r;
    addDrawListener(r);
    return true;
}

bool queueDisplayCopy(uint64_t address, uint32_t width, uint32_t height, DisplayCopy& c) {
    return gRenderer && gRenderer->queueDisplay(address, width, height, c);
}

uint64_t gpuMark() { return gRenderer ? gRenderer->gpuMark() : 0; }

bool gpuDone(uint64_t mark) { return !gRenderer || gRenderer->gpuDone(mark); }

void fetchDisplayCopy(const DisplayCopy& c, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels) {
    if (gRenderer) gRenderer->fetchDisplay(c, width, height, pixels);
}

void captureNextFrame(const std::string& dir) {
    if (gRenderer) gRenderer->captureNextFrame(dir);
}

void saveCaches() {
    if (gRenderer) gRenderer->savePipelineCacheNow();
}

} // namespace host::render
