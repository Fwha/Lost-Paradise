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
#include "host/gpu/dlss.h"
#include "host/gpu/formats.h"
#include "host/gpu/registers.h"
#include "host/gpu/tiling.h"
#include "host/gpu/vk.h"
#include "host/crash.h"
#include "host/display_settings.h"
#include "host/kernel.h"
#include "host/module.h"
#include "host/videoout.h"
#include "host/pm4.h"
#include "host/profile.h"

#include "common/compat.h"
#include "host/mman.h"
#ifdef _WIN32
// LPN_SYSCALL_READS has no Windows counterpart; the iovec lists are kept for the sampled reads.
struct iovec {
    void* iov_base;
    size_t iov_len;
};
#else
#include <sys/uio.h>
#endif
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
#include <tuple>
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
#include <shared_mutex>
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
// Batches in flight. A long submission is cut into several (see LPN_FLUSH_DRAWS in onDraw), so the GPU
// starts on a frame while the rest is still being recorded.
constexpr uint32_t kBatches = 4;
constexpr uint64_t kImportGranule = 2ull << 20;

// Guest memory is read in place, a fault (unmapped or PROT_NONE) caught by host::guarded.
// process_vm_readv did the same safely, but pinning each page in the kernel cost ~20% of the
// recorder and most of the texture checker outside the city gate. LPN_SYSCALL_READS=1: the old way.
#ifdef _WIN32
const bool gSyscallReads = false;
#else
const bool gSyscallReads = getenv("LPN_SYSCALL_READS") != nullptr;
#endif

bool readGuest(uint64_t addr, void* dst, size_t size) {
    if (!addr || addr >= (1ull << 40)) return false;
#ifndef _WIN32
    if (gSyscallReads) {
        iovec local{dst, size}, remote{reinterpret_cast<void*>(addr), size};
        return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t(size);
    }
#endif
    struct Copy {
        void* dst;
        uint64_t addr;
        size_t size;
    } c{dst, addr, size};
    return host::guarded([](void* p) {
        auto& c = *static_cast<Copy*>(p);
        std::memcpy(c.dst, reinterpret_cast<const void*>(c.addr), c.size);
    }, &c);
}

uint64_t shaderAddress(const GpuRegs& r, uint32_t lo) { return ((uint64_t(r.sh[lo + 1]) << 32) | r.sh[lo]) << 8; }

// What recording a draw reads of the registers: the context registers (the live ones, or a copy
// taken for a recorder thread) and the few others.
struct DrawRegs {
    const uint32_t* ctx = nullptr;
    uint32_t prim = 0, indexType = 0, numInstances = 1;
    static DrawRegs of(const GpuRegs& r, const uint32_t* ctx = nullptr) {
        return {ctx ? ctx : r.ctx, r.uconfig[reg::kPrimitiveType], r.indexType, r.numInstances};
    }
};

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
// hashBytes of guest memory read in place, without a copy; false when part is unreadable.
bool hashGuestBytes(uint64_t addr, size_t size, uint64_t& out) {
    if (!addr || addr >= (1ull << 40)) return false;
    if (gSyscallReads) {
        thread_local std::vector<uint8_t> buf;
        buf.resize(size);
        if (!readGuest(addr, buf.data(), size)) return false;
        out = hashBytes(buf.data(), size);
        return true;
    }
    struct Hash {
        uint64_t addr;
        size_t size;
        uint64_t h;
    } c{addr, size, 0};
    if (!host::guarded([](void* p) {
            auto& c = *static_cast<Hash*>(p);
            c.h = hashBytes(reinterpret_cast<const void*>(c.addr), c.size);
        }, &c))
        return false;
    out = c.h;
    return true;
}

uint64_t memoryHash(uint64_t addr, size_t size) {
    uint64_t h;
    return hashGuestBytes(addr, size, h) ? h | 1 : 0;
}

uint64_t sampleHash(uint64_t addr, size_t size) {
    constexpr size_t kSample = 64, kMax = 32;
    const size_t stride = std::max<size_t>(kSample, (size / kMax + kSample - 1) & ~(kSample - 1));
    thread_local std::vector<iovec> remote;
    thread_local std::vector<uint8_t> buf(kSample * kMax);
    remote.clear();
    for (size_t off = 0; off + kSample <= size && remote.size() < kMax; off += stride)
        remote.push_back({reinterpret_cast<void*>(addr + off), kSample});
    if (remote.empty() || !addr || addr + size > (1ull << 40)) return 0;
    iovec local{buf.data(), remote.size() * kSample};
#ifndef _WIN32
    if (gSyscallReads) {
        if (process_vm_readv(getpid(), &local, 1, remote.data(), remote.size(), 0) != ssize_t(local.iov_len)) return 0;
    } else
#endif
    if (!host::guarded([](void*) {
                   for (size_t i = 0; i < remote.size(); ++i)
                       std::memcpy(buf.data() + i * kSample, remote[i].iov_base, kSample);
               }, nullptr)) {
        return 0;
    }
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
        for (size_t i; (i = next_.fetch_add(1)) < chunks_;) {
            const size_t off = i * kChunk, n = std::min(kChunk, size_ - off);
            if (!hashGuestBytes(addr_ + off, n, results_[i])) failed_ = true;
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
    uint64_t h = 0xcbf29ce484222325ull ^ size;
    for (size_t off = 0; off < size; off += kChunk) {
        const size_t n = std::min(kChunk, size - off);
        uint64_t c;
        if (!hashGuestBytes(addr + off, n, c)) return 0;
        h = (h ^ c) * 0x100000001b3ull;
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
    std::map<std::array<uint32_t, 8>, VkImageView> views;
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
    bool writtenBack = false;  // the GPU copies it to memory (writeBack): sampled lazily instead
    uint64_t checkedFrame = ~0ull;
    uint64_t addr = 0, memBytes = 0; // the memory memSample covers
    uint64_t loggedFrame = ~0ull;    // writtenFrame of the last "[gpu] stale target" line
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
    // Unchanged for 60 frames and checked by the sweep once a frame (see sweepVram), so image
    // bindings of it stay good from frame to frame. What the sweep reads: base, span, tiled.
    bool settled = false, swept = false, sweptChanged = false;
    uint64_t base = 0;
    size_t span = 0, tiled = 0;
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
    uint8_t smaa = 0; // one of the game's SMAA pixel shaders: 1 edges, 2 weights, 3 blend (see dlssSmaa)
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
    uint64_t fastKind[3] = {}; // of kFastKind: other draw kind, other shaders, other prim/instances/index type
    uint64_t fastRes[3] = {}; // of kFastResources: resource counts, a buffer, an image
    uint64_t fastMoved = 0; // fast draws that bound a buffer again
    uint64_t fastImages = 0; // fast draws that bound other textures
    uint64_t diffsTaken = 0; // fastDraw checks an analysis worker had done
    uint64_t fastSwitches = 0; // fast draws that switched shaders (switchDraw)
    // Fast draws that differ from the draw before only in draw data (same buffers, textures and
    // index buffer), so one multi-draw could take both; and how many runs of them there were.
    uint64_t mergeable = 0, mergeRuns = 0;
    uint64_t switchMiss[6] = {}; // other shaders not switched to: no slot, slot stale, state, sampled, resources, index
    uint64_t depthJoins = 0, depthOnly = 0; // depth-only draws that joined the open pass, of all depth-only ones
    std::unordered_map<uint32_t, uint64_t> fastCtx; // the context registers that differed
    uint64_t barriers = 0, calls = 0, fastDraws = 0, analysisHits = 0, repeats = 0, analyses = 0, passes = 0, texChecks = 0, texUploads = 0, texBytes = 0, flushes = 0, dmas = 0, dmaBytes = 0, pipelinesMade = 0, pipelineWaits = 0, vramCopies = 0, vramCopyBytes = 0, imageMisses = 0, imageGens = 0, resolvedAhead = 0;
    // VRAM copies: allocations, evictions run, copies they freed, and evictions that found nothing to free.
    uint64_t vramAllocs = 0, vramEvicts = 0, vramFreed = 0, vramEvictsEmpty = 0;
    // Render passes by how many draws they hold (1-15, 16-127, 128-1023, 1024+): passes, and the
    // draws in them; and the largest pass's draws. Sizes how far recording can split by pass.
    uint64_t passBySize[4] = {}, drawsBySize[4] = {}, largestPass = 0;
    // Recorder threads: segments recorded, waits for them (and their ns), their recording time summed.
    uint64_t segments = 0, drains = 0, drainNs = 0, workerNs = 0, aheadNs = 0; // aheadNs: waits for the threads to catch up
    uint64_t sharedDraws = 0, sharedRetries = 0; // recorder threads' full draws looked up under a shared m_, and those that started over
    uint64_t drainWhy[4] = {}; // drains by cause: a sequential draw, a memory write, a CPU-copied DMA, frame end
    uint64_t locks = 0, lockWaits = 0, lockNs = 0; // recorder threads taking m_, finding it held, waiting for it
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
    // Calls and kept arrays go into one stream: a header (the call, or none for an array, and the
    // record's size) and then its storage. Chunks come back from the command thread with their
    // memory in that core's cache, so the recorder asks for the lines ahead of where it writes
    // (a separate list of calls cost a cache miss every few calls).
    template <typename F> void push(F&& f) {
        using T = std::decay_t<F>;
        static_assert(alignof(T) <= kAlign);
        void* p = alloc(sizeof(T), [](void* o) {
            T& t = *static_cast<T*>(o);
            t();
            t.~T();
        });
        new (p) T(std::forward<F>(f));
    }
    template <typename T> T* keep(const T* src, size_t n) {
        if (!n) return nullptr;
        static_assert(alignof(T) <= kAlign);
        T* d = static_cast<T*>(alloc(sizeof(T) * n, nullptr));
        std::memcpy(static_cast<void*>(d), src, sizeof(T) * n);
        return d;
    }
    bool empty() const { return !calls_; }
    void run() {
        for (size_t b = 0; b <= block_ && b < blocks_.size(); ++b) {
            std::byte* base = blocks_[b].mem.get();
            const size_t end = b == block_ ? used_ : blocks_[b].used;
            for (size_t at = 0; at < end;) {
                Header* h = reinterpret_cast<Header*>(base + at);
                if (h->call) h->call(h + 1);
                at += h->size;
            }
        }
        calls_ = 0;
        block_ = 0;
        used_ = 0;
        ahead_ = 0;
    }

private:
    static constexpr size_t kBlock = 256 << 10, kAlign = 16, kAhead = 512;
    struct alignas(kAlign) Header {
        void (*call)(void*);
        size_t size; // of the record, header included
    };
    void* alloc(size_t size, void (*call)(void*)) {
        const size_t n = (sizeof(Header) + size + kAlign - 1) & ~(kAlign - 1);
        while (block_ >= blocks_.size() || used_ + n > blocks_[block_].size) {
            if (block_ < blocks_.size()) {
                blocks_[block_].used = used_;
                ++block_;
                used_ = 0;
                ahead_ = 0;
                continue;
            }
            const size_t sz = std::max(kBlock, n);
            blocks_.push_back({std::make_unique<std::byte[]>(sz), sz, 0});
        }
        Block& b = blocks_[block_];
        std::byte* at = b.mem.get() + used_;
        used_ += n;
        if (prefetchWrites_)
            for (const size_t to = std::min(used_ + kAhead, b.size); ahead_ < to; ahead_ += 64) __builtin_prefetch(b.mem.get() + ahead_, 1);
        Header* h = reinterpret_cast<Header*>(at);
        h->call = call;
        h->size = n;
        if (call) ++calls_;
        return h + 1;
    }
    struct Block {
        std::unique_ptr<std::byte[]> mem;
        size_t size, used;
    };
    std::vector<Block> blocks_;
    size_t block_ = 0, used_ = 0, ahead_ = 0, calls_ = 0;
    // LPN_NO_CHUNK_PREFETCH=1: perf at the overlook put 4% of the recorder on these prefetches.
    static inline const bool prefetchWrites_ = getenv("LPN_NO_CHUNK_PREFETCH") == nullptr;
};

class Renderer final : public DrawListener {
    struct DrawAnalysis; // declared further down

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
        dlss::init(vk_);
        { // pipeline libraries: with generic translations (docs/shader-precompile.md stage 3)
            const char* e = getenv("LPN_PIPELINE_LIBRARIES");
            libraries_ = vk_.pipelineLibraries && gcn::genericTranslation() && !(e && !std::strcmp(e, "0"));
            if (libraries_) std::fprintf(stderr, "[gpu] pipelines linked from per-shader libraries\n");
        }
        if (const char* js = getenv("LPN_DLSS_JITTER_SIGN")) std::sscanf(js, "%f,%f", &jitterSign_[0], &jitterSign_[1]);
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
        R().prof_.on = aprof_.on = getenv("LPN_PROFILE") != nullptr;
        {
            const char* w = getenv("LPN_ANALYSIS_WORKERS");
            // 6 on 16+ threads: on an i7-12700K (20) 3 workers left mid-size scenes (~2,900 draws)
            // waiting 8-10 ms a frame for analysis at 36-59 fps; 6 cut that to ~1 ms (90-140 fps).
            const unsigned hw = std::thread::hardware_concurrency();
            analysisWorkers_ = w ? unsigned(std::atoi(w)) : hw >= 16 ? 6 : hw >= 8 ? 3 : 1;
            analysisWorkers_ = std::min(analysisWorkers_, 16u);
            for (unsigned i = 0; i < std::max(analysisWorkers_, 1u); ++i) {
                actx_.push_back(std::make_unique<AnalysisCtx>(overlays_));
                actx_.back()->prof.on = R().prof_.on;
            }
            for (unsigned i = 0; i < analysisWorkers_; ++i)
                std::thread([this, c = actx_[i].get()] { analysisWorker(*c); }).detach();
            pending_.reserve(kAnalysisBatch);
        }
        openPipelineCache();
        startPipelineWorkers();
        dumpGlsl_ = getenv("LPN_GPU_GLSL") != nullptr;
        startRecorderThreads();
        return true;
    }

    // ---- the analysis thread (see pm4.h): shader analysis for each draw, ahead of recording ----
    // It reads each draw's launches from the registers and hands them, in batches of consecutive
    // draws, to analysis workers (LPN_ANALYSIS_WORKERS, default 6 on 16+ hardware threads, else 3 on 8+,
    // else 1; 0 analyses on this thread).
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
                analyses_.push_back(std::make_unique<DrawAnalysis>());
            }
            a = analyses_.back().get();
            a->ready = false;
        }
        a->ok = a->repeat = false;
        a->vsm = a->psm = nullptr;
        a->vsc = a->psc = nullptr;
        a->diffPrev = false;
        a->resEpoch = 0;
        a->seq = ++analysisSeq_;
        // Most draws keep the last one's context registers: a compare here, a hash only on a change.
        if (std::memcmp(r.ctx, lastCtx_, sizeof lastCtx_) != 0) {
            std::memcpy(lastCtx_, r.ctx, sizeof lastCtx_);
            lastCtxHash_ = hashBytes(lastCtx_, sizeof lastCtx_);
        }
        a->ctxHash = lastCtxHash_;
        pending_.emplace_back();
        DrawJob& j = pending_.back();
        j.out = a;
        j.kind = DrawJob::kNone;
        if (d.kind == DrawCall::kDispatch) prepareDispatch(r, j);
        else if (!computeOnly_ && (d.kind == DrawCall::kAuto || d.kind == DrawCall::kIndexed)) prepareDraw(r, j);
        if (pending_.size() >= kAnalysisBatch) submitBatch();
    }
    void onAnalyzeFlush() override { submitBatch(); }
    uint64_t analysisSeq_ = 0, lastCtxHash_ = 0; // the analysis thread's
    uint32_t lastCtx_[::gpu::kContextRegs] = {};

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
    // The recorder spends most of a draw waiting on memory: the analysis a worker wrote on
    // another core, then the VRAM copy, import and image memo each resource looks up, one
    // dependent miss after another. The draws after this one in the run are known, so their
    // memory is requested ahead in three steps (each reading only what the step before
    // fetched), and the waits overlap. LPN_NO_PREFETCH=1 turns it off.
    const bool prefetch_ = getenv("LPN_NO_PREFETCH") == nullptr;
    static void prefetchRange(const void* p, size_t bytes) {
        const char* c = static_cast<const char*>(p);
        for (size_t o = 0; o < std::min<size_t>(bytes, 1024); o += 64) __builtin_prefetch(c + o);
    }
    // Only the lines a draw reads: prefetching whole analyses and draw data (~400 lines a draw)
    // filled the core's miss buffers and stalled on the prefetches themselves. Buffers resolved
    // ahead (resolveAhead) need no lookups here.
    void prefetchAhead() {
        const size_t n = taken_.size();
        if (takenNext_ + 6 < n) { // the resource list headers
            const DrawAnalysis& x = *taken_[takenNext_ + 6];
            for (const gcn::Program* p : {&x.vs, &x.ps}) {
                __builtin_prefetch(&p->buffers);
                __builtin_prefetch(&p->images);
            }
            __builtin_prefetch(&x.vsRes);
            __builtin_prefetch(&x.psRes);
        }
        if (takenNext_ + 3 < n) { // the resource lists themselves, and their bindings resolved ahead
            const DrawAnalysis& x = *taken_[takenNext_ + 3];
            for (const gcn::Program* p : {&x.vs, &x.ps}) {
                prefetchRange(p->buffers.data(), p->buffers.size() * sizeof(gcn::BufferResource));
                prefetchRange(p->images.data(), p->images.size() * sizeof(gcn::ImageResource));
            }
            prefetchRange(x.vsRes.data(), x.vsRes.size() * sizeof(BufRes));
            prefetchRange(x.psRes.data(), x.psRes.size() * sizeof(BufRes));
        }
        if (takenNext_ + 1 < n) { // the image memo entries for its images
            const DrawAnalysis& x = *taken_[takenNext_ + 1];
            for (const gcn::Program* p : {&x.vs, &x.ps})
                for (const gcn::ImageResource& t : p->images) prefetchRange(&imageMemos_[imageMemoSet(t)], 2 * sizeof(ImageMemo));
        }
    }

    template <typename Run> void spare(Run& run) { // DrawAnalysis is declared further down
        std::lock_guard lock(analysesMutex_);
        // As many as can be in flight: analysis runs up to a frame ahead, and at ~6,000
        // draws a frame a smaller pool left most draws allocating a fresh one (12% of the
        // analysis thread in malloc) and the recorder freeing it (6% in ~Program).
        for (auto& x : run)
            if (spareAnalyses_.size() < 16384) spareAnalyses_.push_back(std::move(x));
        run.clear();
    }

    // Geometry shader pipeline draws (export + geometry + copy shaders) are not rendered yet. The
    // game draws some gore with them (the head bursting in some heat actions): drawn as a plain VS
    // draw they showed the previous mesh. F12 captures save what is needed to translate them:
    // the three shaders, both register files, and the memory their user data points to.
    void onGeometryShaderDraw(const DrawCall& d, const GpuRegs& r) override {
        const uint64_t es = shaderAddress(r, gpu::sh::kSpiShaderPgmLoEs), gs = shaderAddress(r, gpu::sh::kSpiShaderPgmLoGs),
                       vs = shaderAddress(r, gpu::sh::kSpiShaderPgmLoVs);
        warnOnce("geometry shader draws are not rendered yet (es 0x" + hex(es) + " gs 0x" + hex(gs) + " copy vs 0x" + hex(vs) + ")");
        std::lock_guard lock(m_);
        if (frame_ != traceFrame_ || traceDir_.empty()) return;
        const uint64_t n = ++gsDrawIndex_;
        std::fprintf(traceOut_,
                     "[trace] gs draw %" PRIu64 " (after draw #%" PRIu64 ") es 0x%" PRIx64 " gs 0x%" PRIx64 " copy vs 0x%" PRIx64
                     " ps 0x%" PRIx64 " count %u kind %d index 0x%" PRIx64 "+%u %s prim %u instances %u stages 0x%x gs mode 0x%x"
                     " out prim %u max vert out %u esgs item %u gsvs item %u vert item %u (skipped)\n",
                     n, drawIndex_, es, gs, vs, shaderAddress(r, gpu::sh::kSpiShaderPgmLoPs), d.count, int(d.kind), d.indexAddr,
                     d.indexOffset, r.indexType ? "u32" : "u16", r.uconfig[gpu::uconfig::kVgtPrimitiveType], r.numInstances,
                     r.ctx[gpu::ctx::kVgtShaderStagesEn], r.ctx[gpu::ctx::kVgtGsMode], r.ctx[gpu::ctx::kVgtGsOutPrimType],
                     r.ctx[gpu::ctx::kVgtGsMaxVertOut], r.ctx[gpu::ctx::kVgtEsgsRingItemsize], r.ctx[gpu::ctx::kVgtGsvsRingItemsize],
                     r.ctx[gpu::ctx::kVgtGsVertItemsize]);
        auto save = [&](const std::string& name, const void* data, size_t size) {
            if (!captured_.insert(name).second) return;
            if (FILE* f = std::fopen((traceDir_ + name).c_str(), "wb")) {
                std::fwrite(data, 1, size, f);
                std::fclose(f);
            }
        };
        const std::string tag = "gsdraw-" + std::to_string(n);
        save(tag + "-ctx.bin", r.ctx, sizeof(r.ctx));
        save(tag + "-sh.bin", r.sh, sizeof(r.sh));
        save(tag + "-uconfig.bin", r.uconfig, 0x400 * 4);
        std::vector<uint8_t> bytes(64 << 10);
        for (const auto& [stage, addr] : {std::pair{"es", es}, {"gs", gs}, {"vs", vs}}) {
            size_t size = bytes.size();
            while (size >= 4096 && !readGuest(addr, bytes.data(), size)) size /= 2;
            if (size >= 4096) save(std::string(stage) + "-" + hex(addr) + ".bin", bytes.data(), size);
        }
        // What the user data points to (descriptor tables, constants): 4 KB at each address.
        for (uint32_t base : {gpu::sh::kSpiShaderUserDataEs, gpu::sh::kSpiShaderUserDataGs, gpu::sh::kSpiShaderUserDataVs})
            for (uint32_t i = 0; i + 1 < 16; ++i) {
                const uint64_t a = r.sh[base + i] | uint64_t(r.sh[base + i + 1] & 0xFFFF) << 32;
                if (a < (1ull << 32) || (a & 3)) continue;
                bytes.resize(4096);
                if (readGuest(a, bytes.data(), bytes.size())) save("mem-" + hex(a) + ".bin", bytes.data(), bytes.size());
                bytes.resize(64 << 10);
            }
        if (d.kind == DrawCall::kIndexed) {
            const size_t isz = r.indexType ? 4 : 2, size = size_t(d.indexOffset + d.count) * isz;
            bytes.resize(size);
            if (size <= (1u << 20) && readGuest(d.indexAddr, bytes.data(), size)) save(tag + "-indices.bin", bytes.data(), size);
        }
    }
    uint64_t gsDrawIndex_ = 0;

    void onDraw(const DrawCall& d, const GpuRegs& r) override {
        // Analyses are taken, and handed back, a run at a time: a lock per draw was 4% of the
        // recorder at 6,000 draws a frame.
        if (takenNext_ == taken_.size()) {
            taken_.clear();
            takenNext_ = 0;
            std::unique_lock lock(analysesMutex_);
            if (analyses_.empty()) {
                warnOnce("a draw arrived without its analysis");
                return;
            }
            if (!analyses_.front()->ready) {
                ProfScope pw(R().prof_, kProfAnalysisWait);
                analysesCv_.wait(lock, [&] { return analyses_.front()->ready; });
            }
            do {
                taken_.push_back(std::move(analyses_.front()));
                analyses_.pop_front();
            } while (taken_.size() < kTakeRun && !analyses_.empty() && analyses_.front()->ready);
        }
        // Analyses move by pointer: moved by value (two Programs each), handing them along was
        // ~10% of the recorder at 8,000 draws a frame.
        std::unique_ptr<DrawAnalysis>& held = taken_[takenNext_++];
        if (recordThreads_) {
            if (segmentable(d, *held)) {
                addToSegment(d, r, held);
                return;
            }
            drainSegments(0); // recorded here, in order with everything before it
        }
        DrawAnalysis& a = *held;
        Recycle recycle{*this, held};
        std::lock_guard lock(m_);
        ProfScope ps(R().prof_, kProfRenderer);
        if (prefetch_) prefetchAhead();
        ++drawIndex_;
        trace_ = frame_ == traceFrame_;
        if (trace_)
            std::fprintf(traceOut_, "[trace] #%" PRIu64 " %s vs 0x%" PRIx64 " ps 0x%" PRIx64 " cs 0x%" PRIx64 "\n", drawIndex_,
                         d.kind == DrawCall::kDispatch ? "dispatch" : "draw", shaderAddress(r, reg::kPgmLoVs),
                         shaderAddress(r, reg::kPgmLoPs), shaderAddress(r, reg::kPgmLoCs));
        if (raceCheck_) raceShader_ = shaderAddress(r, d.kind == DrawCall::kDispatch ? reg::kPgmLoCs : reg::kPgmLoPs);
        const uint64_t passesBefore = R().prof_.passes;
        if (d.kind == DrawCall::kDispatch) {
            ++R().prof_.calls;
            dispatch(d, r, a);
            R().prevSeq_ = 0;
        } else if (computeOnly_) {
            return;
        } else {
            recordDraw(d, DrawRegs::of(r), held);
        }
        if (frame_ == timingFrame_ && timing_ && R().recording_ && timingMeta_.size() + 1 < kTimingQueries) {
            rec([c = R().cmd_, q = timing_, i = uint32_t(timingMeta_.size() + 1)] {
                vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q, i);
            });
            timingMeta_.push_back({drawIndex_, shaderAddress(r, d.kind == DrawCall::kDispatch ? reg::kPgmLoCs : reg::kPgmLoPs),
                                   d.kind == DrawCall::kDispatch, R().prof_.passes != passesBefore, d.count});
        }
        if (++R().drawsSinceHandOff_ >= 32) handOff(); // the command thread trails by a few draws
        if (flushDraws_ && R().recording_ && ++R().drawsSinceFlush_ >= flushDraws_) flush();
    }
    // Every 1,000 draws: at 400, the 6,350 draws looking up the steps after New Game made 16
    // batches a frame, and with kBatches in flight the recorder waited on the GPU (80 fps, 90 at 1,000).
    const uint32_t flushDraws_ = getenv("LPN_FLUSH_DRAWS") ? std::atoi(getenv("LPN_FLUSH_DRAWS")) : 1000;

    // Hands `a` back to the analysis thread however the draw ends.
    struct Recycle {
        Renderer& r;
        std::unique_ptr<DrawAnalysis>& a;
        ~Recycle() {
            if (!a) return;
            r.R().returned_.push_back(std::move(a));
            if (r.R().returned_.size() < kTakeRun) return;
            if (r.syncRecord_ || !r.deferUploads_ || r.R().immediate_) {
                r.spare(r.R().returned_);
                return;
            }
            // After the command thread has copied their data (see uploadData).
            std::lock_guard lock(r.m_);
            // Not rec(): a call between draws would end fastDraw's run (see FastDraw::rec).
            r.R().chunk_->push([&r = r, run = std::move(r.R().returned_)]() mutable { r.spare(run); });
            r.R().returned_.clear(); // moved from
            r.R().returned_.reserve(kTakeRun);
        }
    };

    // A draw (not a dispatch), on whichever thread records it: the main recorder under m_, or a
    // recorder thread (which takes m_ itself where the caches are touched, see WLock).
    void recordDraw(const DrawCall& d, const DrawRegs& r, std::unique_ptr<DrawAnalysis>& held) {
        DrawAnalysis& a = *held;
        ++R().prof_.calls;
        const uint64_t drawsBefore = R().stats_.draws;
        if (d.kind == DrawCall::kAuto || d.kind == DrawCall::kIndexed) draw(d, r, a);
        else warnOnce("indirect draws and dispatches are not rendered yet");
        if (R().prof_.on && R().stats_.draws != drawsBefore) {
            if (R().countedPass_ != R().passSerial_) {
                notePassSize();
                R().countedPass_ = R().passSerial_;
            }
            ++R().passDraws_;
        }
        // R().prevDraw_ holds this draw's resources if it was recorded (fast, or in full and kept).
        R().prevSeq_ = R().keepDraw_ || R().recordedSeq_ == a.seq ? a.seq : 0;
        if (R().keepDraw_) { // the next draw may repeat this one (see fastDraw)
            R().keepDraw_ = false;
            std::swap(held, R().prevDraw_);
        }
    }

    // The command processor copies descriptors and constants between draws (DMA_DATA): draws
    // after it must not see the page cache's older copy.
    void onMemoryWrite(uint64_t addr, uint64_t size) override {
        if (recordThreads_) drainSegments(1);
        std::lock_guard lock(m_);
        R().fast_.valid = false;
        ++fastEpoch_;
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

    // With recorder threads, a copy the GPU makes is queued after the segments before it without
    // waiting for them (their draws run first on the GPU, and nothing they read on the CPU changes);
    // one the CPU makes waits for them, as it changes memory they may still read.
    bool onDmaCopy(uint64_t dst, uint64_t src, uint64_t bytes, bool fill, uint32_t value) override {
        if (recordThreads_) dispatchSegment();
        R().fast_.valid = false; // memory changes (recorder thread, like fastDraw)
        ++fastEpoch_;
        uint64_t id = 0; // the analysis thread's overlay for this copy
        {
            std::lock_guard lock(analysesMutex_);
            if (!dmaOverlays_.empty()) {
                id = dmaOverlays_.front();
                dmaOverlays_.pop_front();
            }
        }
        // Copied by the CPU right after a false return: the overlay can go at once.
        std::unique_lock lock(m_, std::defer_lock);
        auto cpu = [&] {
            if (recordThreads_) {
                if (lock.owns_lock()) lock.unlock();
                drainSegments(2);
            }
            if (id) overlays_.drop(id);
            return false;
        };
        static const bool cpuDma = getenv("LPN_CPU_DMA") != nullptr; // the old way, for comparison
        if (cpuDma) return cpu();
        lock.lock();
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
            rec([c = R().cmd_, db, doff, bytes, value] { vkCmdFillBuffer(c, db, doff, bytes, value); });
        } else {
            const VkBufferCopy bc{soff, doff, bytes};
            rec([c = R().cmd_, sb, db, bc] { vkCmdCopyBuffer(c, sb, db, 1, &bc); });
        }
        if (raceCheck_) noteWrite(dst, bytes);
        gpuWrites(dst, bytes);
        ++R().prof_.dmas, R().prof_.dmaBytes += bytes;
        if (id) R().pendingDone_.push_back([this, id] { overlays_.drop(id); }); // once the GPU has copied
        return true;
    }

    void onWait() override {
        if (recordThreads_) dispatchSegment();
        std::lock_guard lock(m_);
        flush();
    }

    void onSubmitEnd() override {
        if (recordThreads_) dispatchSegment();
        std::lock_guard lock(m_);
        ProfScope ps(R().prof_, kProfRenderer);
        flush();
    }

    void notePassSize() {
        if (!R().passDraws_) return;
        const int b = R().passDraws_ < 16 ? 0 : R().passDraws_ < 128 ? 1 : R().passDraws_ < 1024 ? 2 : 3;
        ++R().prof_.passBySize[b];
        R().prof_.drawsBySize[b] += R().passDraws_;
        R().prof_.largestPass = std::max(R().prof_.largestPass, R().passDraws_);
        R().passDraws_ = 0;
    }
    void onFrameEnd() override {
        if (recordThreads_) drainSegments(3);
        std::lock_guard lock(m_);
        if (recordThreads_ && R().prof_.on) mergeRecorderProfiles();
        if (R().prof_.on) {
            notePassSize();
            R().countedPass_ = ~0ull;
        }
        {
            ProfScope ps(R().prof_, kProfRenderer);
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
        adoptImports();
        if (R().prof_.on) reportSlowFrame();
        savePipelineCache();
        dlssFrameEnd();
        sweepVram();
        trace_ = false; // recorder threads read it (a traced frame is recorded here, in order)
        // Anisotropic Filtering changed: remembered bindings hold the old samplers.
        if (const int a = displaySettings().anisotropy; a != anisoSeen_) anisoSeen_ = a, ++imageGen_;
        ++frame_;
        frameNow_.store(frame_, std::memory_order_relaxed);
        if (!pendingCapture_.empty()) { // F12: trace this frame into the capture folder
            traceDir_ = pendingCapture_ + "/";
            captured_.clear();
            pendingCapture_.clear();
            traceFrame_ = frame_;
            timingFrame_ = frame_ + 1; // and times the one after it (see reportTiming)
            traceOut_ = std::fopen((traceDir_ + "trace.txt").c_str(), "w");
            if (!traceOut_) traceOut_ = stderr;
        }
        if (R().prof_.on && std::chrono::steady_clock::now() - R().prof_.windowStart >= std::chrono::seconds(1)) reportProfile();
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
            for (int i = 0; i < kProfCount; ++i) std::fprintf(stderr, " %s %.1f", kProfNames[i], (R().prof_.ns[i] - frameNs_[i]) / 1e6);
            std::fprintf(stderr, "; %" PRIu64 " analyses, %" PRIu64 " pipelines made, %" PRIu64 " memory imports (%.1f ms)\n",
                         R().prof_.analyses - frameAnalyses_, R().prof_.pipelinesMade - framePipelines_,
                         syncImports_.load() - frameSyncImports_, (syncImportNs_.load() - frameSyncImportNs_) / 1e6);
        }
        frameSyncImports_ = syncImports_, frameSyncImportNs_ = syncImportNs_;
        lastFrameEnd_ = now;
        std::copy(std::begin(R().prof_.ns), std::end(R().prof_.ns), frameNs_);
        frameAnalyses_ = R().prof_.analyses;
        framePipelines_ = R().prof_.pipelinesMade;
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
        rec([c = R().cmd_, q = timing_] {
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
            for (int i : {kProfAnalyze, kProfShaderLookup, kProfLaunch}) R().prof_.ns[i] += std::exchange(a.ns[i], 0);
            R().prof_.analysisHits += std::exchange(a.analysisHits, 0);
            R().prof_.repeats += std::exchange(a.repeats, 0);
            for (int i = 0; i < 4; ++i) R().prof_.missWhy[i] += std::exchange(a.missWhy[i], 0);
            for (int i = 0; i < 16; ++i) R().prof_.missUser[i] += std::exchange(a.missUser[i], 0);
            R().prof_.analyses += std::exchange(a.analyses, 0);
            R().prof_.pipelinesMade += std::exchange(a.pipelinesMade, 0);
        };
        merge(aprof_);
        for (auto& c : actx_) merge(c->prof);
    }

    void reportProfile() {
        mergeAnalysisProfile();
        const auto now = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - R().prof_.windowStart).count();
        const double n = double(std::max<uint64_t>(windowFrames_, 1));
        std::fprintf(stderr, "[prof] t=%.1f frames %" PRIu64 "-%" PRIu64 ": %.1f fps, %.1f ms/frame, worst %.1f ms, %" PRIu64 " over 17 ms;",
                     profile::seconds(), frame_ - windowFrames_, frame_, n / wall, wall * 1000 / n, worstFrameMs_, slowFrames_);
        for (int i = 0; i < kProfCount; ++i) std::fprintf(stderr, " %s %.2f", kProfNames[i], R().prof_.ns[i] / 1e6 / n);
        std::fprintf(stderr, " passes for targets %.0f feedback %.0f (depth %.0f color %.0f) store %.0f other %.0f; depth-only draws %.0f (%.0f joined a pass);",
                     R().prof_.passWhy[0] / n, R().prof_.passWhy[1] / n, R().prof_.feedbackOn[0] / n,
                     R().prof_.feedbackOn[1] / n, R().prof_.passWhy[2] / n, R().prof_.passWhy[3] / n, R().prof_.depthOnly / n,
                     R().prof_.depthJoins / n);
        std::fprintf(stderr, " gpu-busy %.2f dma %.0f (%.2f MB)", gpuBusyNs_.exchange(0) / 1e6 / n, R().prof_.dmas / n,
                     R().prof_.dmaBytes / n / 1e6);
        if (raceCheck_) std::fprintf(stderr, " races %" PRIu64 "/%" PRIu64, raceChanged_.load(), raceChecked_.exchange(0));
        std::fprintf(stderr, " (ms/frame); %.0f draws (%.0f fast), %.0f analysis hits (%.0f repeats), %.0f analyses, %.0f tex checks, %.1f uploads (%.1f MB), %.1f flushes, %.0f render passes, %.0f barriers per frame, %" PRIu64 " pipelines made, %" PRIu64 " blended draws skipped for one; VRAM buffers %.0f MB, %.1f copies (%.2f MB) per frame; %.0f image lookups missed, %.1f image changes; %.0f buffers resolved ahead\n",
                     R().prof_.calls / n, R().prof_.fastDraws / n, R().prof_.analysisHits / n, R().prof_.repeats / n, R().prof_.analyses / n, R().prof_.texChecks / n, R().prof_.texUploads / n, R().prof_.texBytes / n / 1e6, R().prof_.flushes / n, R().prof_.passes / n,
                     R().prof_.barriers / n, R().prof_.pipelinesMade, R().prof_.pipelineWaits, vramBytes_ / 1e6, R().prof_.vramCopies / n,
                     R().prof_.vramCopyBytes / n / 1e6, R().prof_.imageMisses / n, R().prof_.imageGens / n, R().prof_.resolvedAhead / n);
        std::fprintf(stderr, "[prof] t=%.1f launches not repeating the last: other shader %.0f, new submission %.0f, user data %.0f, other %.0f per frame; user data words differing:",
                     profile::seconds(), R().prof_.missWhy[0] / n, R().prof_.missWhy[1] / n, R().prof_.missWhy[2] / n, R().prof_.missWhy[3] / n);
        for (int i = 0; i < 16; ++i)
            if (R().prof_.missUser[i]) std::fprintf(stderr, " s%d %.0f", i, R().prof_.missUser[i] / n);
        std::fprintf(stderr, "\n[prof] t=%.1f draws recorded in full:", profile::seconds());
        for (int i = 0; i < kFastMissCount; ++i) std::fprintf(stderr, " %s %.0f", kFastMissNames[i], R().prof_.fastMiss[i] / n);
        std::fprintf(stderr, " (kind %.0f, shaders %.0f, prim %.0f; resource counts %.0f, buffer %.0f, image %.0f); fast draws binding a buffer again %.0f, other textures %.0f, switching shaders %.0f",
                     R().prof_.fastKind[0] / n, R().prof_.fastKind[1] / n, R().prof_.fastKind[2] / n, R().prof_.fastRes[0] / n,
                     R().prof_.fastRes[1] / n, R().prof_.fastRes[2] / n, R().prof_.fastMoved / n, R().prof_.fastImages / n, R().prof_.fastSwitches / n);
        std::fprintf(stderr, "; mergeable fast draws %.0f in %.0f runs", R().prof_.mergeable / n, R().prof_.mergeRuns / n);
        std::fprintf(stderr, " (not switched: no slot %.0f, slot stale %.0f, state %.0f, sampled %.0f, resources %.0f, index %.0f); checks done ahead %.0f",
                     R().prof_.switchMiss[0] / n, R().prof_.switchMiss[1] / n, R().prof_.switchMiss[2] / n, R().prof_.switchMiss[3] / n,
                     R().prof_.switchMiss[4] / n, R().prof_.switchMiss[5] / n, R().prof_.diffsTaken / n);
        if (!R().prof_.fastCtx.empty()) {
            std::vector<std::pair<uint64_t, uint32_t>> regs;
            for (auto [reg, count] : R().prof_.fastCtx) regs.push_back({count, reg});
            std::sort(regs.rbegin(), regs.rend());
            std::fprintf(stderr, "; context regs differing:");
            for (size_t i = 0; i < regs.size() && i < 8; ++i)
                std::fprintf(stderr, " 0x%x %.0f", 0xA000 + regs[i].second, regs[i].first / n);
        }
        std::fprintf(stderr, " per frame\n");
        std::fprintf(stderr, "[prof] t=%.1f passes by draws: 1-15 %.0f (%.0f draws), 16-127 %.0f (%.0f), 128-1023 %.0f (%.0f), 1024+ %.0f (%.0f) per frame; largest %" PRIu64 " draws\n",
                     profile::seconds(), R().prof_.passBySize[0] / n, R().prof_.drawsBySize[0] / n, R().prof_.passBySize[1] / n,
                     R().prof_.drawsBySize[1] / n, R().prof_.passBySize[2] / n, R().prof_.drawsBySize[2] / n, R().prof_.passBySize[3] / n,
                     R().prof_.drawsBySize[3] / n, R().prof_.largestPass);
        if (recordThreads_)
            std::fprintf(stderr, "[prof] t=%.1f recorder threads %u: %.0f segments, threads recording %.2f ms (summed), "
                                 "%.0f waits for them (draw %.0f, write %.0f, dma %.0f, frame %.0f) %.2f ms, ahead-waits %.2f ms; m_ taken %.0f, held by another %.0f, waited %.2f ms (summed); full draws shared %.0f, started over %.0f per frame\n",
                         profile::seconds(), recordThreads_, R().prof_.segments / n, R().prof_.workerNs / 1e6 / n, R().prof_.drains / n,
                         R().prof_.drainWhy[0] / n, R().prof_.drainWhy[1] / n, R().prof_.drainWhy[2] / n, R().prof_.drainWhy[3] / n,
                         R().prof_.drainNs / 1e6 / n, R().prof_.aheadNs / 1e6 / n, R().prof_.locks / n, R().prof_.lockWaits / n,
                         R().prof_.lockNs / 1e6 / n, R().prof_.sharedDraws / n, R().prof_.sharedRetries / n);
        std::fprintf(stderr, "[prof] t=%.1f VRAM copies %zu: %.1f allocations, %.2f evictions (%.1f freed, %.2f found none) per frame\n",
                     profile::seconds(), vram_.size(), R().prof_.vramAllocs / n, R().prof_.vramEvicts / n, R().prof_.vramFreed / n,
                     R().prof_.vramEvictsEmpty / n);
        {
            const uint64_t si = syncImports_ - windowSyncImports_, sn = syncImportNs_ - windowSyncImportNs_;
            const uint64_t ai = aheadImports_ - windowAheadImports_, an = aheadImportNs_ - windowAheadImportNs_;
            windowSyncImports_ += si, windowSyncImportNs_ += sn, windowAheadImports_ += ai, windowAheadImportNs_ += an;
            std::fprintf(stderr, "[prof] t=%.1f memory imports %zu: %" PRIu64 " by draws (%.1f ms), %" PRIu64 " ahead (%.1f ms; %" PRIu64 " MB so far, %" PRIu64 " MB waiting)\n",
                         profile::seconds(), imports_.size(), si, sn / 1e6, ai, an / 1e6, aheadImportBytes_.load() >> 20,
                         aheadPending_.load() >> 20);
        }
        if (const gcn::ReplayStats rs = gcn::takeReplayStats(); rs.planned)
            std::fprintf(stderr, "[prof] t=%.1f fast replays %.0f per frame, %.0f%% of their trace steps run directly (%.0f of %.0f); %" PRIu64 " differed from a full replay, %" PRIu64 " changed memory\n",
                         profile::seconds(), rs.planned / n, 100.0 * rs.kept / std::max<uint64_t>(rs.steps, 1), rs.kept / n,
                         rs.steps / n, rs.mismatches, rs.races);
        windowFrames_ = slowFrames_ = 0;
        worstFrameMs_ = 0;
        const bool on = R().prof_.on;
        std::fill(std::begin(frameNs_), std::end(frameNs_), 0);
        frameAnalyses_ = framePipelines_ = 0;
        R().prof_ = Profiler{};
        R().prof_.on = on;
        R().prof_.windowStart = now;
    }

    // Called from the display thread. Only recording the copy holds the renderer; waiting for
    // the GPU does not, so the game keeps submitting meanwhile.
    // Records a copy of the image at display buffer addr into readback slot c.slot and submits it.
    // The copy sits in GPU order with the game's work, so the caller can wait for it elsewhere.
    bool queueDisplay(uint64_t addr, uint32_t width, uint32_t height, DisplayCopy& c) {
        noteDisplay(addr);
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
        const bool riding = R().recording_;
        begin();
        fullBarrier();
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {c.w, c.h, 1};
        rec([c = R().cmd_, im = s.img.image, buf = d.buf, r] { vkCmdCopyImageToBuffer(c, im, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &r); });
        if (!riding) flush();
        std::lock_guard dl(doneMutex_);
        c.seq = riding ? submitted_ + 1 : submitted_;
        return true;
    }

    // The batch holding everything recorded so far, and whether the GPU has finished it.
    uint64_t gpuMark() {
        std::lock_guard lock(m_);
        std::lock_guard dl(doneMutex_);
        return R().recording_ ? submitted_ + 1 : submitted_;
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
        rec([cb = R().cmd_, im = s.img.image, buf = R().ring_, c] { vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &c); });
        const auto* src = static_cast<const uint8_t*>(R().ringPtr_) + at;
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
        ++R().recCount_;
        if (syncRecord_ || R().immediate_) f();
        else R().chunk_->push(std::forward<F>(f));
    }
    // Storage for an array a queued call points to, valid until the call has run.
    template <typename T> const T* keep(const T* p, size_t n) { return syncRecord_ || R().immediate_ ? p : R().chunk_->keep(p, n); }
    void handOff() {
        R().drawsSinceHandOff_ = 0;
        if (syncRecord_ || R().chunk_->empty()) return;
        std::unique_ptr<CmdChunk> next;
        {
            std::lock_guard lock(chunkMutex_);
            chunks_.push_back(std::move(R().chunk_));
            if (!spareChunks_.empty()) {
                next = std::move(spareChunks_.front());
                spareChunks_.pop_front();
            }
        }
        chunkCv_.notify_one();
        R().chunk_ = next ? std::move(next) : std::make_unique<CmdChunk>();
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
    struct Layout {
        VkDescriptorSetLayout set = VK_NULL_HANDLE;
        VkPipelineLayout pipeline = VK_NULL_HANDLE;
        // Pipeline libraries (libraries_, graphics only): `set` holds the vertex shader's bindings
        // (pushed) and psSet the pixel shader's (a set per draw); the layouts are independent-set
        // ones, pre and frag for the stages' libraries, with their keys.
        VkDescriptorSetLayout psSet = VK_NULL_HANDLE;
        VkPipelineLayout pre = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
        std::string preKey, fragKey;
    };

    void pushDescriptors(VkPipelineBindPoint point, const Layout& lay, std::vector<VkWriteDescriptorSet>& writes,
                         std::vector<VkDescriptorBufferInfo>& bufs, std::vector<VkDescriptorImageInfo>& imgs) {
        const VkPipelineLayout layout = lay.pipeline;
        if (lay.psSet) { // pipeline libraries: the pixel shader's part goes in a set of its own
            thread_local std::vector<VkWriteDescriptorSet> vw;
            thread_local std::vector<VkDescriptorBufferInfo> vb;
            thread_local std::vector<VkDescriptorImageInfo> vi;
            pixelDescriptors(point, lay, writes, bufs, imgs, vw, vb, vi);
            if (vw.empty()) return;
            pushVertex(point, layout, vw, vb, vi);
            return;
        }
        pushVertex(point, layout, writes, bufs, imgs);
    }
    void pushVertex(VkPipelineBindPoint point, VkPipelineLayout layout, std::vector<VkWriteDescriptorSet>& writes,
                    std::vector<VkDescriptorBufferInfo>& bufs, std::vector<VkDescriptorImageInfo>& imgs) {
        if (syncRecord_ || R().immediate_) {
            linkWrites(writes, bufs, imgs);
            vk_.cmdPushDescriptorSet(R().cmd_, point, layout, 0, uint32_t(writes.size()), writes.data());
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
        rec([this, c = R().cmd_, point, layout, n, w] { vk_.cmdPushDescriptorSet(c, point, layout, 0, n, w); });
    }

    // Pipeline libraries: the pixel shader's writes go into what is bound (Bound::ps*), the rest
    // to vw/vb/vi (to be pushed); then all of the pixel shader's are written into a new set, bound
    // as set 1. Push descriptors update incrementally (a fast draw pushes only what changed), a set
    // does not.
    void pixelDescriptors(VkPipelineBindPoint point, const Layout& lay, const std::vector<VkWriteDescriptorSet>& writes,
                          const std::vector<VkDescriptorBufferInfo>& bufs, const std::vector<VkDescriptorImageInfo>& imgs,
                          std::vector<VkWriteDescriptorSet>& vw, std::vector<VkDescriptorBufferInfo>& vb,
                          std::vector<VkDescriptorImageInfo>& vi) {
        Bound& bd = R().bound_;
        if (bd.psLayout != &lay) bd.psLayout = &lay, bd.psHas = 0;
        bool any = false;
        size_t nb = 0, ni = 0;
        vw.clear(), vb.clear(), vi.clear();
        for (size_t k = 0; k < writes.size(); ++k) {
            const VkWriteDescriptorSet& w = writes[k];
            const bool buf = w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            const size_t at = buf ? nb++ : ni++;
            if (w.dstBinding >= kPsBindingBase && w.dstBinding < kPsBindingBase + 64) {
                const uint32_t i = w.dstBinding - kPsBindingBase;
                bd.psHas |= 1ull << i;
                bd.psType[i] = w.descriptorType;
                if (buf) bd.psBuf[i] = bufs[at];
                else bd.psImg[i] = imgs[at];
                any = true;
                continue;
            }
            vw.push_back(w); // the vertex shader's, pushed as before
            if (buf) vb.push_back(bufs[at]);
            else vi.push_back(imgs[at]);
        }
        linkWrites(vw, vb, vi);
        if (!any || !bd.psHas) return;
        const VkDescriptorSet set = allocPixelSet(lay.psSet);
        if (!set) return;
        thread_local std::vector<VkWriteDescriptorSet> sw;
        sw.clear();
        for (uint64_t m = bd.psHas; m; m &= m - 1) {
            const uint32_t i = uint32_t(std::countr_zero(m));
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set;
            w.dstBinding = kPsBindingBase + i;
            w.descriptorCount = 1;
            w.descriptorType = bd.psType[i];
            if (w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) w.pBufferInfo = &bd.psBuf[i];
            else w.pImageInfo = &bd.psImg[i];
            sw.push_back(w);
        }
        vkUpdateDescriptorSets(vk_.device, uint32_t(sw.size()), sw.data(), 0, nullptr);
        if (syncRecord_ || R().immediate_) {
            vkCmdBindDescriptorSets(R().cmd_, point, lay.pipeline, 1, 1, &set, 0, nullptr);
            return;
        }
        rec([c = R().cmd_, point, layout = lay.pipeline, set] { vkCmdBindDescriptorSets(c, point, layout, 1, 1, &set, 0, nullptr); });
    }
    // Pixel shader sets come from pools used in turn: a full pool is retired, and reset for reuse
    // once every batch that could hold its sets has finished (more pools are made meanwhile).
    VkDescriptorSet allocPixelSet(VkDescriptorSetLayout l) {
        std::lock_guard lock(setPoolMutex_);
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (setPool_) {
                VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                ai.descriptorPool = setPool_;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts = &l;
                VkDescriptorSet set = VK_NULL_HANDLE;
                if (vkAllocateDescriptorSets(vk_.device, &ai, &set) == VK_SUCCESS) return set;
                uint64_t mark;
                {
                    std::lock_guard dl(doneMutex_);
                    mark = submitted_ + kBatches + 1; // any batch recording now is submitted by then
                }
                retiredPools_.push_back({mark, setPool_});
                setPool_ = VK_NULL_HANDLE;
            }
            uint64_t completed;
            {
                std::lock_guard dl(doneMutex_);
                completed = completed_;
            }
            if (!retiredPools_.empty() && retiredPools_.front().first <= completed) {
                setPool_ = retiredPools_.front().second;
                retiredPools_.pop_front();
                vkResetDescriptorPool(vk_.device, setPool_, 0);
                continue;
            }
            constexpr uint32_t kSets = 2048;
            const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSets * 8},
                                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSets * 8},
                                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSets}};
            VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pci.maxSets = kSets;
            pci.poolSizeCount = uint32_t(std::size(sizes));
            pci.pPoolSizes = sizes;
            if (vkCreateDescriptorPool(vk_.device, &pci, nullptr, &setPool_) != VK_SUCCESS) {
                setPool_ = VK_NULL_HANDLE;
                std::fprintf(stderr, "[gpu] descriptor pool not created\n");
                return VK_NULL_HANDLE;
            }
        }
        return VK_NULL_HANDLE;
    }

    void useBatch(uint32_t i) {
        R().cur_ = i;
        R().cmd_ = batches_[i].cmd;
        R().ring_ = batches_[i].ring;
        R().ringPtr_ = batches_[i].ringPtr;
        R().ringUsed_ = 0;
    }

    void begin() {
        if (R().recording_) return;
        if (R().worker_) { // the uploads ahead of the segments (a segment's own Rec records from its start)
            if (&R() == &pro_) openPrologue();
            return;
        }
        Batch& b = batches_[R().cur_];
        {
            ProfScope ps(R().prof_, kProfGpuWait);
            std::unique_lock lock(doneMutex_);
            if (b.busy && recordThreads_ && !tRec_) {
                // m_ held: the batch may still execute segments whose threads need it.
                lock.unlock();
                m_.unlock();
                lock.lock();
                doneCv_.wait(lock, [&] { return !b.busy; });
                lock.unlock();
                m_.lock();
                if (R().recording_ || &b != &batches_[R().cur_]) return begin(); // another thread began meanwhile
                lock.lock();
            }
            doneCv_.wait(lock, [&] { return !b.busy; });
        }
        rec([this, c = R().cmd_, fence = b.fence] {
            VK_CHECK(vkResetFences(vk_.device, 1, &fence));
            VK_CHECK(vkResetCommandBuffer(c, 0));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(c, &bi));
        });
        R().recording_ = true;
        R().bound_ = Bound{};
        R().needBarrier_ = true; // earlier batches' writes
        if (frame_ == timingFrame_ && timing_ && timingMeta_.size() + 1 < kTimingQueries) {
            rec([c = R().cmd_, q = timing_, i = uint32_t(timingMeta_.size() + 1)] {
                vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, q, i);
            });
            timingMeta_.push_back({0, 0, false, false, 0, true});
        }
        if (timestamps_)
            rec([c = R().cmd_, q = timestamps_, i = 2 * R().cur_] {
                vkCmdResetQueryPool(c, q, i, 2);
                vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, q, i);
            });
    }

    void flush() {
        if (R().worker_) {
            if (&R() == &pro_) closePrologue(); // its ring is full: the next upload starts another
            else warnOnce("a recorder thread's upload ring is full (draw data may be overwritten)");
            return;
        }
        if (!R().recording_) {
            if (!R().pendingDone_.empty()) {
                // Nothing new to submit: due when the last submission finishes. Decided on the
                // command thread, which has submitted everything queued before.
                rec([this, fns = std::move(R().pendingDone_)]() mutable {
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
                R().pendingDone_.clear();
                handOff();
            }
            if (sync_) waitSubmitted();
            return;
        }
        ++R().prof_.flushes;
        endPass();
        if (raceCheck_ && !raceReads_.empty()) {
            auto reads = std::make_shared<std::unordered_map<uint64_t, RaceRead>>(std::move(raceReads_));
            raceReads_.clear();
            R().afterFlush_.insert(R().afterFlush_.begin(), [this, reads] { checkReads(*reads); });
        }
        Batch& b = batches_[R().cur_];
        if (timestamps_)
            rec([c = R().cmd_, q = timestamps_, i = 2 * R().cur_ + 1] { vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q, i); });
        std::vector<std::function<void()>> fns = std::move(R().afterFlush_);
        if (recordThreads_) { // what recorder threads freed: after every segment recorded by now
            std::lock_guard lock(lateMutex_);
            for (auto& f : lateFree_) fns.push_back(std::move(f));
            lateFree_.clear();
        }
        for (auto& f : R().pendingDone_) fns.push_back(std::move(f));
        R().afterFlush_.clear();
        R().pendingDone_.clear();
        {
            std::lock_guard lock(doneMutex_);
            b.busy = true; // its ring and command buffer are not reused until the GPU is done
            ++submitted_;
        }
        // The completion thread learns of the batch once it is submitted (inFlight_).
        rec([this, i = R().cur_, c = R().cmd_, fns = std::move(fns)]() mutable {
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
        R().recording_ = false;
        useBatch((R().cur_ + 1) % kBatches);
        R().drawsSinceFlush_ = 0;
        if (sync_) waitSubmitted();
    }

    // Submits what is recorded and waits until the GPU (and the deferred work) is done.
    void waitIdle() { // m_ held
        const bool paused = pauseSegments();
        flush();
        waitSubmitted();
        if (paused) resumeSegments();
    }
    // Before waiting on the GPU with m_ held: no new segments, and the dispatched ones recorded
    // (their threads need m_, and the batches waited for execute them). Releases m_ meanwhile.
    bool pauseSegments() {
        if (!recordThreads_ || tRec_) return false;
        std::unique_lock lock(segMutex_);
        ++paused_;
        if (outstanding_) {
            m_.unlock();
            segDoneCv_.wait(lock, [&] { return outstanding_ == 0; });
            lock.unlock();
            m_.lock();
        }
        return true;
    }
    void resumeSegments() {
        {
            std::lock_guard lock(segMutex_);
            --paused_;
        }
        segDoneCv_.notify_all();
    }
    void waitSubmitted() {
        ProfScope ps(R().prof_, kProfGpuWait);
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
        if (recordThreads_) dispatchSegment(); // due after the draws before it too
        std::lock_guard lock(m_);
        R().pendingDone_.push_back(std::move(fn));
        return true;
    }

private:

    // Before a copy, clear, dispatch or upload: everything before it is done and visible. What it
    // writes is then made visible by the barrier the next draw's pass starts with (R().needBarrier_).
    void fullBarrier() {
        endPass();
        barrier();
        R().needBarrier_ = true;
    }
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        rec([c = R().cmd_, mb] {
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0,
                                 nullptr);
        });
        ++R().prof_.barriers;
        R().needBarrier_ = false;
        R().drawWritten_.clear();
        R().drawSampled_.clear();
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
        VkDeviceSize at = (R().ringUsed_ + align - 1) & ~(align - 1);
        if (at + size > R().ringSize_) {
            flush();
            begin();
            at = 0;
        }
        R().ringUsed_ = at + size;
        return at;
    }
    bool ringFits(VkDeviceSize size) const { return R().ringUsed_ + size + 4096 <= R().ringSize_; }
    // A draw's data into the ring, copied by the command thread: the recorder's copies were 15% of
    // its time at the city overlook, nearly all misses on data the analysis workers had just
    // written. The data stays put until then: analyses go back to the workers through the command
    // thread too (see onDraw's Recycle), and only the draw being recorded writes its own data.
    VkDeviceSize uploadData(const std::vector<uint32_t>& data) {
        const VkDeviceSize bytes = data.size() * 4;
        const VkDeviceSize at = ringAlloc(bytes, storageAlign_);
        uint8_t* dst = static_cast<uint8_t*>(R().ringPtr_) + at;
        const bool fix = hud().on;
        if (syncRecord_ || !deferUploads_ || R().immediate_) {
            std::memcpy(dst, data.data(), bytes);
            if (fix) fixViewRays(data.data(), dst, bytes);
        } else {
            rec([dst, src = data.data(), bytes, fix] {
                std::memcpy(dst, src, bytes);
                if (fix) fixViewRays(src, dst, bytes);
            });
        }
        return at;
    }
    // Aspect Ratio: the ambient occlusion pass gets its view rays as constants, a pixel size quad
    // (s/w, -s/h, -1, 1) with s 1 or 2 and a tan quad (tan x, tan y, -1, 0), in either order. The
    // pixel size is right but tan x is worked out for 16:9 by code not found (ghost shadows at
    // 21:9). Where both quads sit side by side, tan x is made to match the pixel size's shape.
    // It reads the draw's own data and writes only the fixed values to the ring, which is slow
    // to read (uncached).
    static void fixViewRays(const uint32_t* src, uint8_t* dst, size_t bytes) {
        const float* f = reinterpret_cast<const float*>(src);
        float* out = reinterpret_cast<float*>(dst);
        const size_t n = bytes / 4;
        auto pixel = [&](size_t i) {
            return f[i] > 0 && f[i + 1] < 0 && f[i + 2] == -1.0f && f[i + 3] == 1.0f;
        };
        auto tan = [&](size_t i) {
            return f[i] > 0 && f[i + 1] > 0 && f[i + 2] == -1.0f && f[i + 3] == 0.0f &&
                   std::fabs(f[i] / f[i + 1] - 16.0f / 9) < 1e-3f;
        };
        for (size_t i = gcn::data::smrd(); i + 4 <= n; ++i) {
            if (!pixel(i)) continue;
            const float aspect = -f[i + 1] / f[i]; // (s/h) / (s/w)
            if (std::fabs(aspect - 16.0f / 9) < 0.02f || aspect < 1.0f || aspect > 4.0f) continue;
            if (i + 8 <= n && tan(i + 4)) out[i + 4] = f[i + 5] * aspect;
            if (i >= gcn::data::smrd() + 4 && tan(i - 4)) out[i - 4] = f[i - 3] * aspect;
        }
    }

    // Under the shared lock (a recorder thread's draw), a lookup that would change a shared cache
    // flags the draw instead, which then starts over holding m_ exclusively. True when it did.
    bool sharedMiss() {
        if (!R().lockShared_) return false;
        R().abort_ = true;
        return true;
    }
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

    // Importing guest memory is slow on some drivers (tens of ms a call on NVIDIA's Windows one: a
    // hitch whenever a draw first read from a new 2 MB of memory). The direct memory the game maps
    // is imported on a thread of its own instead, in kAheadChunk pieces as it is mapped, and the
    // renderer adopts what is ready; a draw only imports what that thread has not reached.
    // On by default on Windows; LPN_IMPORT_AHEAD=1/0 sets it.
    static constexpr uint64_t kAheadChunk = 16ull << 20, kAheadOverlap = 6ull << 20;
    std::mutex aheadMutex_;
    std::condition_variable aheadCv_;
    std::deque<std::pair<uint64_t, uint64_t>> aheadQueue_; // mapped ranges still to import
    // A buffer lying across two pieces is in neither, so a draw imported it (up to 4 MB ones, in
    // play): each piece runs kAheadOverlap into the next, and one more import spans a mapping's
    // start, which takes every buffer up to that size.
    // The mappings noted so far, adjacent ones merged (start -> end). Nothing is imported
    // outside them: past a mapping's end lies memory that is not the game's (the driver's own
    // mappings among it), and importing that made a later allocation fail.
    std::map<uint64_t, uint64_t> aheadMapped_;
    bool aheadMappedHas(uint64_t a, uint64_t e) { // aheadMutex_ held
        auto it = aheadMapped_.upper_bound(a);
        return it != aheadMapped_.begin() && std::prev(it)->second >= e;
    }
    std::vector<Import> aheadReady_;
    std::atomic<bool> aheadHasReady_{false};
    bool aheadStarted_ = false;
    std::atomic<uint64_t> syncImports_{0}, syncImportNs_{0}; // imports made by a draw
    std::atomic<uint64_t> aheadImports_{0}, aheadImportNs_{0}, aheadImportBytes_{0}, aheadPending_{0};
    uint64_t frameSyncImports_ = 0, frameSyncImportNs_ = 0;
    static bool importAheadOn() {
        static const bool on = [] {
            if (const char* e = getenv("LPN_IMPORT_AHEAD"); e && *e) return *e != '0';
#ifdef _WIN32
            return true;
#else
            return false;
#endif
        }();
        return on;
    }
public:
    // The game mapped [addr, addr + len) of direct memory (any thread).
    void noteMapping(uint64_t addr, uint64_t len) {
        if (!importAheadOn() || !vk_.getMemoryHostPointerProperties || !addr || addr + len > (1ull << 40)) return;
        {
            std::lock_guard lock(aheadMutex_);
            aheadQueue_.emplace_back(addr, addr + len);
            uint64_t a = addr, e = addr + len;
            for (auto it = aheadMapped_.begin(); it != aheadMapped_.end();)
                if (it->first <= e && it->second >= a) a = std::min(a, it->first), e = std::max(e, it->second), it = aheadMapped_.erase(it);
                else ++it;
            aheadMapped_[a] = e;
            aheadPending_ += len;
            if (!aheadStarted_) {
                aheadStarted_ = true;
                std::thread([this] { importAheadThread(); }).detach();
            }
        }
        aheadCv_.notify_one();
    }

private:
    void importAheadThread() {
        profile::nameThread("lpn-import");
        const uint64_t align = std::max<uint64_t>(vk_.hostPointerAlignment, 4096);
        auto import = [&](uint64_t a, uint64_t e) {
            {
                std::shared_lock lock(cacheMutex_);
                VkBuffer b;
                VkDeviceSize o;
                if (importFor(a, e - a, b, o)) return;
            }
            Import im;
            const auto t0 = std::chrono::steady_clock::now();
            if (!importRange(a, e, im)) return;
            aheadImportNs_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            ++aheadImports_;
            aheadImportBytes_ += e - a;
            std::lock_guard lock(aheadMutex_);
            aheadReady_.push_back(im);
            aheadHasReady_.store(true, std::memory_order_release);
        };
        for (;;) {
            std::pair<uint64_t, uint64_t> range;
            {
                std::unique_lock lock(aheadMutex_);
                aheadCv_.wait(lock, [&] { return !aheadQueue_.empty(); });
                range = aheadQueue_.front();
                aheadQueue_.pop_front();
            }
            const uint64_t start = (range.first + align - 1) & ~(align - 1), end = range.second & ~(align - 1);
            // How far past `e` an import may run: to the end of the game's mapped memory there.
            auto mappedTo = [&](uint64_t a, uint64_t e) {
                std::lock_guard lock(aheadMutex_);
                auto it = aheadMapped_.upper_bound(a);
                if (it == aheadMapped_.begin()) return e;
                return std::max(e, std::min(e + kAheadOverlap, std::prev(it)->second & ~(align - 1)));
            };
            for (uint64_t a = start; a < end;) {
                const uint64_t e = std::min(end, (a + kAheadChunk) & ~(kAheadChunk - 1));
                import(a, mappedTo(a, e));
                a = e;
            }
            bool before;
            {
                std::lock_guard lock(aheadMutex_);
                before = start > kAheadOverlap && aheadMappedHas(start - kAheadOverlap, start + kAheadOverlap);
            }
            if (before && end - start >= kAheadOverlap) import(start - kAheadOverlap, start + kAheadOverlap);
            aheadPending_ -= range.second - range.first;
        }
    }
    // Takes over what the import thread has ready (m_ held exclusively). Whether there was any.
    bool adoptImports() {
        if (!aheadHasReady_.load(std::memory_order_acquire)) return false;
        std::vector<Import> ready;
        {
            std::lock_guard lock(aheadMutex_);
            ready.swap(aheadReady_);
            aheadHasReady_.store(false, std::memory_order_release);
        }
        std::unique_lock lock(cacheMutex_);
        for (const Import& im : ready) {
            auto [it, fresh] = imports_.try_emplace(im.start, im);
            if (fresh) continue;
            if (it->second.end >= im.end) { // a draw imported as much there meanwhile: this one was never used
                vkDestroyBuffer(vk_.device, im.buffer, nullptr);
                vkFreeMemory(vk_.device, im.memory, nullptr);
            } else {
                it->second = im; // the smaller one stays alive: recorded commands may name its buffer
            }
        }
        return !ready.empty();
    }

    // A Vulkan buffer over guest memory [addr, addr + size).
    bool guestBuffer(uint64_t addr, uint64_t size, VkBuffer& buffer, VkDeviceSize& offset) {
        if (!addr || addr >= (1ull << 40) || !size) return false;
        const uint64_t end = addr + size;
        // The import that last covered this granule: a map walk per buffer was a cache miss per
        // level, at 8,000 draws a frame.
        const Import*& hit = R().importHits_[(addr / kImportGranule) % kImportHits];
        if (hit && hit->start <= addr && hit->end >= end) {
            buffer = hit->buffer;
            offset = addr - hit->start;
            return true;
        }
        // The import starting closest below `addr` usually covers it (imports are aligned granules);
        // a scan of all of them is the fallback.
        if (auto it = imports_.upper_bound(addr); it != imports_.begin()) {
            --it;
            if (it->second.end >= end) {
                hit = &it->second;
                buffer = it->second.buffer;
                offset = addr - it->first;
                return true;
            }
        }
        for (const auto& [start, im] : imports_) {
            if (start > addr) break;
            if (im.end >= end) {
                hit = &im;
                buffer = im.buffer;
                offset = addr - start;
                return true;
            }
        }
        if (sharedMiss()) return false; // a recorder thread's draw: recorded again with m_ held
        if (adoptImports()) return guestBuffer(addr, size, buffer, offset); // the import thread may have it by now
        const uint64_t align = std::max<uint64_t>(vk_.hostPointerAlignment, 4096);
        Import im;
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t gs = addr & ~(kImportGranule - 1), ge = (end + kImportGranule - 1) & ~(kImportGranule - 1);
        const uint64_t ps = addr & ~(align - 1), pe = (end + align - 1) & ~(align - 1);
        if (!importRange(gs, ge, im) && !importRange(ps, pe, im)) {
            warnOnce("cannot import guest memory at 0x" + std::to_string(addr));
            return false;
        }
        ++syncImports_;
        if (R().prof_.on && importAheadOn())
            std::fprintf(stderr, "[prof] memory import by a draw: 0x%" PRIx64 " +0x%" PRIx64 " (imported 0x%" PRIx64 "-0x%" PRIx64 ")\n", addr,
                         size, im.start, im.end);
        syncImportNs_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        {
            std::unique_lock lock(cacheMutex_);
            imports_[im.start] = im;
        }
        buffer = im.buffer;
        offset = addr - im.start;
        return true;
    }

    // ---- images --------------------------------------------------------------------------------

    bool createImage(Image& img, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage, bool depth,
                     VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D, uint32_t layers = 1, uint32_t levels = 1) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = depth ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        if (viewType == VK_IMAGE_VIEW_TYPE_CUBE) ici.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ici.imageType = viewType == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {w, h, viewType == VK_IMAGE_VIEW_TYPE_3D ? layers : 1};
        ici.mipLevels = levels;
        ici.arrayLayers = viewType == VK_IMAGE_VIEW_TYPE_3D ? 1 : layers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ++imageGen_; // remembered image bindings may name an image this replaces
        ++R().prof_.imageGens;
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
        img.levels = levels;
        img.aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        if (format == VK_FORMAT_D32_SFLOAT_S8_UINT) img.aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

        // Everything stays in GENERAL; start it there.
        Prologue pg(*this); // on a recorder thread: ahead of the segments
        begin();
        endPass();
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = img.image;
        ib.subresourceRange = {img.aspect, 0, levels, 0, ici.arrayLayers};
        rec([c = R().cmd_, ib] {
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &ib);
        });
        // Clear so never-written surfaces read as black rather than garbage.
        const VkImageSubresourceRange range{img.aspect, 0, levels, 0, ici.arrayLayers};
        if (depth) {
            if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                rec([c = R().cmd_, im = img.image, range] {
                    const VkClearDepthStencilValue v{1.0f, 0};
                    vkCmdClearDepthStencilImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range);
                });
        } else if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
            rec([c = R().cmd_, im = img.image, range] {
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
        R().afterFlush_.push_back([this, copy] {
            for (auto& [k, v] : copy.views) vkDestroyImageView(vk_.device, v, nullptr);
            vkDestroyImage(vk_.device, copy.image, nullptr);
            vkFreeMemory(vk_.device, copy.memory, nullptr);
        });
        img = Image{};
    }

    // A view of mip levels [baseLevel, baseLevel + levelCount) (all of the image's by default).
    VkImageView view(Image& img, VkFormat format, VkImageViewType type, VkComponentMapping swz,
                     VkImageAspectFlags aspect = 0, uint32_t baseLevel = 0, uint32_t levelCount = 0) {
        if (!aspect) aspect = img.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT;
        if (!aspect) aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        baseLevel = std::min(baseLevel, img.levels - 1);
        levelCount = levelCount ? std::min(levelCount, img.levels - baseLevel) : img.levels - baseLevel;
        const std::array<uint32_t, 8> k{uint32_t(format), uint32_t(type), uint32_t(swz.r), uint32_t(swz.g),
                                        uint32_t(swz.b), uint32_t(swz.a), aspect, baseLevel << 8 | levelCount};
        auto it = img.views.find(k);
        if (it != img.views.end()) return it->second;
        if (sharedMiss()) return VK_NULL_HANDLE;
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = type;
        vci.format = format;
        vci.components = swz;
        vci.subresourceRange = {aspect, baseLevel, levelCount, 0,
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
            if (sharedMiss()) return nullptr;
            destroyImage(s.img);
            colors_.erase(it);
        }
        if (sharedMiss()) return nullptr;
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
        s.addr = addr;
        s.memBytes = uint64_t(w) * h * bytes;
        return &(colors_[addr] = std::move(s));
    }

    Surface* depthSurface(uint64_t addr, uint32_t w, uint32_t h, VkFormat format) {
        auto it = depths_.find(addr);
        if (it != depths_.end()) {
            Surface& s = it->second;
            if (s.img.width == w && s.img.height == h && s.img.format == format) return &s;
            if (sharedMiss()) return nullptr;
            destroyImage(s.img);
            depths_.erase(it);
        }
        if (sharedMiss()) return nullptr;
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

    // Pixel shaders' resources are descriptor set 1 when pipelines are linked from libraries: each
    // stage's library has only its own set, and only one set can be pushed.
    uint32_t descriptorSet(const gcn::Program& p) const { return libraries_ && p.launch.stage == gcn::Stage::Pixel ? 1 : 0; }

    ShaderModule* shader(const gcn::Program& p, uint32_t bindingBase) {
        std::string key = p.key();
        key.append(reinterpret_cast<const char*>(&bindingBase), 4);
        auto it = shaders_.find(key);
        if (it != shaders_.end()) return it->second.get();
        auto m = std::make_unique<ShaderModule>();
        m->translation = gcn::translate(p, bindingBase, descriptorSet(p));
        for (const auto& w : p.warnings) warnOnce("shader 0x" + hex(p.launch.entry) + ": " + w);
        for (const auto& w : m->translation.warnings) warnOnce("shader 0x" + hex(p.launch.entry) + ": " + w);
        const gcn::GlslStage stage = p.launch.stage == gcn::Stage::Pixel     ? gcn::GlslStage::Fragment
                                     : p.launch.stage == gcn::Stage::Compute ? gcn::GlslStage::Compute
                                                                             : gcn::GlslStage::Vertex;
        // Debugging: LPN_DBG_PS=<entry>[,<entry>...] rewrites those pixel shaders' color output to
        // LPN_DBG_OUT (a GLSL vec4 argument list). dbgT<n> holds image n's last sample, and
        // LPN_DBG_CAP="<line text>|<expr>" stores vec4(expr) in dbgC after the first matching line.
        bool rewritten = false; // by a debugging option below: not kept for later runs
        if (const char* dp = getenv("LPN_DBG_PS"); dp && *dp && getenv("LPN_DBG_OUT") && stage == gcn::GlslStage::Fragment &&
                                                   std::strstr(dp, hex(p.launch.entry).c_str())) {
            rewritten = true;
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
        if (m->module && !rewritten) saveSource(m->name, p, bindingBase);
        if (libraries_ && !rewritten) checkBootTranslation(p, m->name);
        if (stage == gcn::GlslStage::Fragment) {
            uint8_t code[256];
            if (readGuest(p.launch.entry, code, sizeof code)) m->smaa = smaaRole(fnv(code, sizeof code));
            if (m->smaa == 3) std::fprintf(stderr, "[dlss] found the game's SMAA\n");
        }
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
    // The on-disk caches' folder: $LPN_SHADER_CACHE (default ~/.cache/lpn-host/spirv, on Windows
    // %LOCALAPPDATA%\lpn-host\spirv; "off": none).
    void cacheDir() {
        if (!spirvDir_.empty()) return;
        const char* env = getenv("LPN_SHADER_CACHE");
        if (env && !std::strcmp(env, "off")) spirvDir_ = "-";
        else if (env && *env) spirvDir_ = env;
        else if (const std::string c = host::userCacheDir(); !c.empty()) spirvDir_ = c + "/lpn-host/spirv";
        else spirvDir_ = "-";
        // Generic translations (gcn::genericTranslation) name other SPIR-V than specialised ones
        // and record other pipelines: each kind keeps its own cache, so switching back and forth
        // (LPN_GENERIC_SHADERS) never builds the other kind's pipelines at startup.
        if (spirvDir_ != "-" && gcn::genericTranslation()) spirvDir_ += "/generic3"; // generic/ held stage 1 programs, whose draw data lacks V# words; generic2/ ones lack linkage words
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
            // Workers may still be adding pipelines (closing the window during the startup compile):
            // the cache can outgrow the size just asked for, and the read then gives VK_INCOMPLETE.
            // Ask again with room to spare instead of saving nothing.
            size_t n = 0;
            std::vector<char> data;
            for (int tries = 0; tries < 8; ++tries) {
                if (vkGetPipelineCacheData(vk_.device, pipelineCache_, &n, nullptr) != VK_SUCCESS || !n) {
                    n = 0;
                    break;
                }
                n += n / 8;
                data.resize(n);
                const VkResult r = vkGetPipelineCacheData(vk_.device, pipelineCache_, &n, data.data());
                if (r == VK_SUCCESS) break;
                n = 0;
                if (r != VK_INCOMPLETE) break;
            }
            if (!n) std::fprintf(stderr, "[gpu] pipeline cache not saved: the driver gave no data\n");
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
        saveFile(path, words.data(), words.size() * 4);
    }
    static void saveFile(const std::string& path, const void* data, size_t size) {
        if (path.empty()) return;
        const std::string tmp = path + ".tmp" + std::to_string(getpid()) + "-" +
                                std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const bool ok = std::fwrite(data, 1, size, f) == size;
        if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), path.c_str()); // never a partial file
        else std::remove(tmp.c_str());
    }
    static std::string loadFile(const std::string& path) {
        std::string bytes;
        FILE* f = path.empty() ? nullptr : std::fopen(path.c_str(), "rb");
        if (!f) return bytes;
        char buf[65536];
        for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) bytes.append(buf, n);
        std::fclose(f);
        return bytes;
    }

    // Beside each <name>.spv, <name>.gcn holds the program it was translated from (the binding
    // base, then gcn::serialize), so the startup builds can translate it again once the
    // translator has changed and the old SPIR-V names no longer match (see currentShader).
    void saveSource(const std::string& name, const gcn::Program& p, uint32_t bindingBase) {
        if (spirvDir_ == "-") return;
        const std::string path = spirvDir_ + "/" + name + ".gcn";
        if (access(path.c_str(), F_OK) == 0) return;
        std::string bytes(reinterpret_cast<const char*>(&bindingBase), 4);
        bytes += gcn::serialize(p);
        saveFile(path, bytes.data(), bytes.size());
    }

    // The name the shader recorded as `name` has with today's translator: translated again from
    // its .gcn (and its SPIR-V compiled if new or lost), or `name` itself without one. Any thread:
    // the startup compile translates on every core (translateAll).
    std::unordered_map<std::string, std::string> currentNames_;
    std::mutex namesMutex_;
    std::string currentShader(const std::string& name) {
        if (name.empty() || spirvDir_ == "-") return name;
        {
            std::lock_guard lock(namesMutex_);
            if (auto it = currentNames_.find(name); it != currentNames_.end()) return it->second;
        }
        const std::string now = translateSaved(name);
        std::lock_guard lock(namesMutex_);
        return currentNames_.emplace(name, now).first->second;
    }
    std::string translateSaved(const std::string& name) {
        const std::string out = name;
        const std::string bytes = loadFile(spirvDir_ + "/" + name + ".gcn");
        gcn::Program p;
        uint32_t bindingBase;
        if (bytes.size() < 4 || !gcn::deserialize(bytes.substr(4), p)) return out;
        std::memcpy(&bindingBase, bytes.data(), 4);
        const gcn::Translation t = gcn::translate(p, bindingBase, descriptorSet(p));
        const gcn::GlslStage stage = p.launch.stage == gcn::Stage::Pixel     ? gcn::GlslStage::Fragment
                                     : p.launch.stage == gcn::Stage::Compute ? gcn::GlslStage::Compute
                                                                             : gcn::GlslStage::Vertex;
        const std::string now = spirvName(t.glsl, stage);
        const std::string path = spirvCachePath(t.glsl, stage);
        if (access(path.c_str(), F_OK) != 0 || (now != name && loadSpirv(path).empty())) { // compiled again if lost too
            std::string error;
            const std::vector<uint32_t> spirv = gcn::compileGlsl(t.glsl, stage, &error);
            if (spirv.empty()) return out;
            saveSpirv(path, spirv);
            ++compiled_;
        }
        if (now == name) return out;
        if (access((spirvDir_ + "/" + now + ".gcn").c_str(), F_OK) != 0) saveFile(spirvDir_ + "/" + now + ".gcn", bytes.data(), bytes.size());
        ++retranslated_;
        return now;
    }
    std::atomic<size_t> retranslated_{0}, compiled_{0};

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
        bool graphics = false;
        for (const auto& b : bindings) graphics |= (b.stageFlags & VK_SHADER_STAGE_VERTEX_BIT) != 0;
        if (libraries_ && graphics) {
            thread_local std::vector<VkDescriptorSetLayoutBinding> vsb, psb;
            vsb.clear(), psb.clear();
            for (const auto& b : bindings) {
                (b.binding < kPsBindingBase ? vsb : psb).push_back(b);
                const uint32_t k[] = {b.binding, uint32_t(b.descriptorType), b.stageFlags};
                (b.binding < kPsBindingBase ? l.preKey : l.fragKey).append(reinterpret_cast<const char*>(k), sizeof(k));
            }
            VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            dci.bindingCount = uint32_t(vsb.size());
            dci.pBindings = vsb.data();
            VK_CHECK(vkCreateDescriptorSetLayout(vk_.device, &dci, nullptr, &l.set));
            dci.flags = 0;
            dci.bindingCount = uint32_t(psb.size());
            dci.pBindings = psb.data();
            VK_CHECK(vkCreateDescriptorSetLayout(vk_.device, &dci, nullptr, &l.psSet));
            auto make = [&](VkDescriptorSetLayout a, VkDescriptorSetLayout b, uint32_t n) {
                const VkDescriptorSetLayout sets[2] = {a, b};
                VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                pci.flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT;
                pci.setLayoutCount = n;
                pci.pSetLayouts = sets;
                VkPipelineLayout pl;
                VK_CHECK(vkCreatePipelineLayout(vk_.device, &pci, nullptr, &pl));
                return pl;
            };
            l.pipeline = make(l.set, l.psSet, 2);
            l.pre = make(l.set, VK_NULL_HANDLE, 1);
            l.frag = make(VK_NULL_HANDLE, l.psSet, 2);
            return layouts_[key] = l;
        }
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
        // Pipeline libraries: the rasteriser and depth/stencil state the pipeline has dynamic (see
        // libraryState): cull mode, front face, depth bias enable, depth test, write, compare,
        // stencil test, then each face's fail, pass, depth fail ops and compare op.
        uint32_t lib[15] = {};
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
        bool prebuild = false; // from pipelines.list: vs and ps are shared, held by keep
        bool forDraw = false;  // queued urgent, counted in drawBuilds_
        std::shared_ptr<void> keep; // a startup build's shader modules, destroyed with the last job using them
        const Layout* lay = nullptr; // for libraries: its stages' layouts
        std::string vsName, psName;  // for libraries: what they are kept by
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

    VkPipeline graphicsPipeline(const GraphicsState& s, ShaderModule* vs, ShaderModule* ps, const Layout& lay,
                                const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        const VkPipelineLayout layout = lay.pipeline;
        thread_local std::string key; // per draw: built in a reused buffer
        key.assign(reinterpret_cast<const char*>(&s), sizeof(s));
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) return it->second;
        if (sharedMiss()) return VK_NULL_HANDLE;
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
            ++R().prof_.pipelinesMade;
            notePipeline(c, vs->name, ps ? ps->name : std::string(), R().prof_.on);
            PipelineJob j{c, vs->module, ps ? ps->module : VK_NULL_HANDLE, geometryFor(s), layout, b, std::move(record), false};
            j.lay = &lay;
            j.vsName = vs->name;
            j.psName = ps ? ps->name : std::string();
            if (syncPipelines_) runJob(j);
            else {
                queueJob(std::move(j), true);
                hurry(b); // the recorder needs this one now
            }
        }
        if (!b->ready.load(std::memory_order_acquire)) {
            if (!b->started.load()) hurry(b);
            if (skippable(c)) {
                ++R().prof_.pipelineWaits;
                return VK_NULL_HANDLE; // still being built: the draw is skipped
            }
            ProfScope pw(R().prof_, kProfGpuWait);
            b->ready.wait(false, std::memory_order_acquire);
        }
        return pipelines_[key] = b->pipeline;
    }

    // LPN_PROFILE: why a pipeline made during play was not among those built before: new shaders,
    // or which state differs from the closest one with the same shaders ([pipe] lines).
    std::unordered_map<std::string, std::vector<GraphicsState>> pipelinesByShaders_;
    std::unordered_set<std::string> knownShaders_;
    std::mutex notedMutex_;
    void notePipeline(const GraphicsState& s, const std::string& vs, const std::string& ps, bool report,
                      const char* when = "at draw") {
        std::lock_guard lock(notedMutex_);
        auto& known = pipelinesByShaders_[vs + "/" + ps];
        if (report) {
            std::string why;
            if (known.empty()) {
                why = std::string("new shader pair (vs ") + (knownShaders_.count(vs) ? "seen" : "NEW") + ", ps " +
                      (ps.empty() ? "none" : knownShaders_.count(ps) ? "seen" : "NEW") + ")";
            } else {
                const GraphicsState* best = nullptr;
                int bestDiff = 1 << 30;
                auto fields = [](const GraphicsState& x) {
                    std::vector<std::pair<std::string, uint32_t>> f = {
                        {"topology", x.topology}, {"modeCntl", x.modeCntl}, {"depthControl", x.depthControl},
                        {"stencilControl", x.stencilControl}, {"depthFormat", x.depthFormat}, {"stencil", x.stencil},
                        {"depthWriteOff", x.depthWriteOff}, {"writeMasks", x.writeMasks}, {"vsOutputs", x.vsOutputs}};
                    for (int i = 0; i < 8; ++i) {
                        f.push_back({"colorFormat" + std::to_string(i), x.colorFormats[i]});
                        f.push_back({"blend" + std::to_string(i), x.blend[i]});
                    }
                    return f;
                };
                const auto mine = fields(s);
                for (const GraphicsState& k : known) {
                    const auto theirs = fields(k);
                    int diff = 0;
                    for (size_t i = 0; i < mine.size(); ++i) diff += mine[i].second != theirs[i].second;
                    if (diff < bestDiff) bestDiff = diff, best = &k;
                }
                char buf[96];
                std::snprintf(buf, sizeof buf, "%zu known with these shaders; closest differs in", known.size());
                why = buf;
                const auto theirs = fields(*best);
                for (size_t i = 0; i < mine.size(); ++i)
                    if (mine[i].second != theirs[i].second) {
                        std::snprintf(buf, sizeof buf, " %s 0x%x->0x%x", mine[i].first.c_str(), theirs[i].second, mine[i].second);
                        why += buf;
                    }
                if (!bestDiff) why += " nothing (bindings differ)";
            }
            std::fprintf(stderr, "[pipe] t=%.1f made %s, vs %.12s ps %.12s: %s\n", profile::seconds(), when, vs.c_str(),
                         ps.c_str(), why.c_str());
        }
        known.push_back(s);
        knownShaders_.insert(vs);
        if (!ps.empty()) knownShaders_.insert(ps);
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
                    buildCv_.notify_one(); // workers that skip startup builds were not woken for it
                    return;
                }
    }

    void startPipelineWorkers() {
        syncPipelines_ = getenv("LPN_SYNC_PIPELINES") != nullptr;
        if (spirvDir_ != "-") {
            pipelineListPath_ = spirvDir_ + "/pipelines.list";
            pipelineList_ = std::fopen(pipelineListPath_.c_str(), "ab");
        }
        // Compile Shaders at Startup (lpn-display.ini precompile_shaders, LPN_PRECOMPILE=0/1 wins):
        // the game waits behind a progress screen while every saved shader is translated and every
        // recorded pipeline built, on all cores, instead of them being built under the game.
        const char* pc = getenv("LPN_PRECOMPILE");
        const char* pb = getenv("LPN_PREBUILD");
        const bool prebuild = !pipelineListPath_.empty() && !(pb && !std::strcmp(pb, "0"));
        warmupAll_ = prebuild && !syncPipelines_ && !computeOnly_ &&
                     (pc ? std::strcmp(pc, "0") != 0 : displaySettings().precompileShaders);
        if (syncPipelines_) return;
        const unsigned n = std::clamp(std::thread::hardware_concurrency() >= 8 ? std::thread::hardware_concurrency() - 4 : 2u, 2u, 16u);
        // The startup builds (thousands, every run) on every worker took the CPU from the game for
        // the first minute or so, slow even with every pipeline in the driver's cache: only a
        // quarter of the workers take them, at a lower priority. Draws' pipelines go to any worker.
        const unsigned background = std::max(1u, n / 4);
        for (unsigned i = 0; i < n; ++i)
            std::thread([this, takesBackground = i < background] {
                profile::nameThread("lpn-pipeline");
                if (takesBackground) profile::lowerThreadPriority();
                for (;;) {
                    PipelineJob j;
                    {
                        std::unique_lock lock(buildMutex_);
                        // While the game waits for the startup compile, every worker takes its builds.
                        buildCv_.wait(lock, [&] {
                            return !jobs_.empty() || ((takesBackground || warmupAll_.load()) && !backgroundJobs_.empty());
                        });
                        auto& q = jobs_.empty() ? backgroundJobs_ : jobs_;
                        j = std::move(q.front());
                        q.pop_front();
                        j.out->started = true;
                    }
                    runJob(j);
                }
            }).detach();
        if (prebuild) std::thread([this] { prebuildPipelines(); }).detach();
        // LPN_BOOT_COMPILE=0 turns the boot compile off.
        if (const char* e = getenv("LPN_BOOT_COMPILE"); libraries_ && spirvDir_ != "-" && !(e && !std::strcmp(e, "0")))
            std::thread([this] { bootCompile(); }).detach();
    }

    // Draws' pipelines go ahead of the startup builds.
    void queueJob(PipelineJob&& j, bool urgent) {
        {
            std::lock_guard lock(buildMutex_);
            // A draw's pipelines in the order they were asked for (a waiting draw's goes first, see
            // hurry()); the startup builds after all of them.
            if (urgent) j.forDraw = true, ++drawBuilds_;
            (urgent ? jobs_ : backgroundJobs_).push_back(std::move(j));
        }
        // notify_one could wake a worker that skips startup builds and leave them waiting.
        if (urgent) buildCv_.notify_one();
        else buildCv_.notify_all();
    }

    void runJob(PipelineJob& j) {
        j.out->pipeline = libraries_ && j.lay && j.lay->pre ? linkGraphics(j) : buildGraphics(j.state, j.vs, j.ps, j.geom, j.layout);
        j.out->started = true;
        j.out->ready.store(true, std::memory_order_release);
        j.out->ready.notify_all();
        if (j.forDraw) --drawBuilds_;
        ++newPipelines_;
        if (j.prebuild) {
            j.keep.reset();
            ++warmupBuilt_;
            if (prebuildLeft_.fetch_sub(1) == 1) reportPrebuildDone();
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

    // ---- Stage 4 of docs/shader-precompile.md: every shader before it is drawn ----

    // A program's pipeline libraries (see linkGraphics), built unless a draw or an earlier call
    // did: the startup compile builds them for every saved vertex and pixel shader, and the boot
    // compile for every shader the game registers, so a draw only links.
    void buildLibraries(const gcn::Program& p, const std::string& name) {
        if (!libraries_ || name.empty() || p.launch.stage == gcn::Stage::Compute) return;
        const bool vertex = p.launch.stage == gcn::Stage::Vertex;
        std::vector<VkDescriptorSetLayoutBinding> lb;
        if (vertex) {
            addBindings(lb, p, 0, VK_SHADER_STAGE_VERTEX_BIT);
        } else { // with a vertex shader's draw data, for a graphics layout; only the pixel shader's part keys the library
            lb.push_back({0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr});
            addBindings(lb, p, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT);
        }
        const Layout& lay = layout(lb);
        {
            std::lock_guard lock(libMutex_);
            if (libs_.count(vertex ? "pre" + name + "/" + lay.preKey : "frag" + name + "/" + lay.fragKey)) return;
        }
        VkShaderModule m = loadModule(name);
        if (!m) return;
        if (vertex) preLibrary(name, m, VK_NULL_HANDLE, "", lay);
        else fragLibrary(name, m, lay);
        vkDestroyShaderModule(vk_.device, m, nullptr); // a library keeps what it needs
        ++librariesBuilt_;
    }
    void warmLibraries(const std::string& name) {
        if (!libraries_ || name.empty()) return;
        const std::string bytes = loadFile(spirvDir_ + "/" + name + ".gcn");
        gcn::Program p;
        if (bytes.size() >= 4 && gcn::deserialize(bytes.substr(4), p)) buildLibraries(p, name);
    }
    std::atomic<size_t> librariesBuilt_{0};
    std::atomic<int> drawBuilds_{0}; // pipelines draws wait for, queued or building
    // Background compiles wait while a draw waits for a pipeline: the CPU is the draw's
    // (with them competing, the new game's first pipelines took 7.4 s instead of 3.1).
    void yieldToDraws() {
        const auto since = std::chrono::steady_clock::now();
        bool told = false;
        while (drawBuilds_.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (!told && std::chrono::steady_clock::now() - since > std::chrono::seconds(10)) {
                told = true;
                std::fprintf(stderr, "[gpu] boot compile waiting 10 s for %d draw pipelines\n", drawBuilds_.load());
            }
        }
    }

    // The boot compile: once the title menu has registered the game's shaders in the engine's
    // tables (see the doc), each is analysed without a draw (gcn::analyzeUnbound), translated,
    // compiled and saved like a drawn one, and its library built, in the background. A shader
    // whose translation turns out to depend on its draw after all is translated again when drawn.
    struct TableShader {
        gcn::Stage stage;
        uint64_t gnmx;
    };
    std::vector<TableShader> readShaderTables() {
        std::vector<TableShader> out;
        Module* eboot = process().eboot;
        if (!eboot) return out;
        const uint64_t base = reinterpret_cast<uint64_t>(eboot->base());
        struct Table {
            uint64_t at;
            gcn::Stage stage;
            uint32_t gnmxAt; // the Gnmx shader's pointer in the engine's object
        };
        for (const Table& t : {Table{0x2698148, gcn::Stage::Vertex, 0x48}, Table{0x2698168, gcn::Stage::Pixel, 0x68}}) {
            uint64_t range[2] = {};
            if (!readGuest(base + t.at, range, sizeof range) || range[1] < range[0]) continue;
            const uint64_t n = std::min<uint64_t>((range[1] - range[0]) / 8, 0x4000);
            for (uint64_t i = 0; i < n; ++i) {
                uint64_t obj = 0, gnmx = 0;
                if (!readGuest(range[0] + i * 8, &obj, 8) || !obj) break;
                if (readGuest(obj + t.gnmxAt, &gnmx, 8) && gnmx) out.push_back({t.stage, gnmx});
            }
        }
        return out;
    }
    // A launch from the Gnmx shader's registers (the doc's table); what a draw adds is guessed:
    // the generic translation takes it from the draw data.
    // A vertex shader's fetch slots (gcn::Program::fetchSlots) from its input semantics, 4 bytes
    // each from +0x28 (semantic, VGPR, dwords, 0), their count in the byte at +0x24: one slot per
    // four VGPRs from v4 up to the last input's. ~0u (from the code) when they do not look right.
    static bool tableLaunch(const TableShader& t, gcn::LaunchState& l, uint32_t& fetchSlots) {
        uint32_t r[12] = {}; // from +0x08: PGM_LO, PGM_HI, RSRC1, RSRC2, then the stage's own
        if (!readGuest(t.gnmx + 8, r, sizeof r)) return false;
        fetchSlots = ~0u;
        if (t.stage == gcn::Stage::Vertex) {
            const uint32_t n = r[7] & 0xFF; // +0x24
            uint8_t sem[0x18 * 4] = {};
            if (n && n <= 0x18 && readGuest(t.gnmx + 0x28, sem, n * 4)) {
                uint32_t end = 0;
                bool ok = true;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t v = sem[i * 4 + 1], size = sem[i * 4 + 2];
                    ok &= v >= 4 && size >= 1 && size <= 4 && v + size <= 256;
                    end = std::max(end, v + size);
                }
                if (ok) fetchSlots = (end - 4 + 3) / 4;
            }
        }
        l = gcn::LaunchState{};
        l.stage = t.stage;
        l.entry = (r[0] | uint64_t(r[1]) << 32) << 8;
        l.rsrc1 = r[2];
        l.rsrc2 = r[3];
        if (t.stage == gcn::Stage::Pixel) {
            l.colorFormat = r[5];
            l.psInputEna = r[6];
            l.numInterp = r[8] & 0x3F;
            l.depthExport = r[10] & 1;
            for (uint32_t m = 0; m < 8; ++m) { // integer exports, as to integer targets
                const uint32_t fmt = (l.colorFormat >> (m * 4)) & 0xF;
                if (fmt == 7) l.colorUint |= 1u << m;
                if (fmt == 8) l.colorSint |= 1u << m;
            }
        }
        return l.entry >= 0x100000 && l.entry < (1ull << 40);
    }
    void bootCompile() {
        profile::nameThread("lpn-boot-compile");
        profile::lowerThreadPriority();
        while (warmupAll_.load() && !warmupDone_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100)); // the startup compile's CPU
        // Half the cores: it waits whenever a draw waits for a pipeline (yieldToDraws), so draws keep
        // theirs, and on Windows (antivirus scanning each new file) a quarter left 3,600 files to minutes.
        const unsigned n = std::max(1u, std::thread::hardware_concurrency() / 2);
        // First the libraries of every shader an earlier run saved (most of them registered ones),
        // in the background rather than behind the progress screen (0.17 s each, uncached).
        {
            std::vector<std::string> names;
            std::error_code ec;
            for (const auto& e : std::filesystem::directory_iterator(spirvDir_, ec))
                if (e.path().extension() == ".gcn") names.push_back(e.path().stem().string());
            const auto start = std::chrono::steady_clock::now();
            std::atomic<size_t> next{0};
            std::vector<std::thread> pool;
            for (unsigned i = 0; i < n; ++i)
                pool.emplace_back([&] {
                    profile::lowerThreadPriority();
                    for (size_t k; (k = next++) < names.size();) {
                        yieldToDraws();
                        warmLibraries(currentShader(names[k]));
                    }
                });
            for (auto& t : pool) t.join();
            if (!names.empty())
                std::fprintf(stderr, "[gpu] saved shaders' libraries: %zu shaders in %.1f s (%zu built)\n", names.size(),
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), librariesBuilt_.load());
        }
        // The tables fill while the title menu loads: wait for them to stop growing.
        std::vector<TableShader> shaders;
        for (size_t last = 0, steady = 0, polls = 0; steady < 6; ++polls) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            shaders = readShaderTables();
            steady = shaders.size() && shaders.size() == last ? steady + 1 : 0;
            last = shaders.size();
            if (polls == 240) std::fprintf(stderr, "[gpu] boot compile: the shader tables hold %zu entries after 2 minutes\n", last);
        }
        std::fprintf(stderr, "[gpu] boot compile started: %zu registered shaders on %u threads\n", shaders.size(), n);
        const auto start = std::chrono::steady_clock::now();
        bootStart_ = start;
        bootTotal_ = shaders.size();
        bootRunning_ = true;
        std::atomic<size_t> next{0}, compiled{0}, failed{0};
        std::atomic<size_t>& done = bootCount_;
        const size_t before = librariesBuilt_.load();
        std::vector<std::thread> pool;
        for (unsigned i = 0; i < n; ++i)
            pool.emplace_back([&] {
                profile::lowerThreadPriority();
                for (size_t k; (k = next++) < shaders.size();) {
                    yieldToDraws();
                    gcn::LaunchState l;
                    uint32_t slots;
                    if (!tableLaunch(shaders[k], l, slots)) {
                        ++failed;
                        continue;
                    }
                    // A vertex shader takes the fetch slots its code reads, as a draw does (at
                    // least: gcn::analyze); its input semantics may declare more, which a mesh's
                    // fetch shader then loads, so that count is translated too.
                    std::vector<gcn::Program> programs{gcn::analyzeUnbound(l, readGuest)};
                    if (l.stage == gcn::Stage::Vertex && slots != ~0u && !programs[0].code.empty() &&
                        programs[0].fetchSite && slots > programs[0].fetchSlots)
                        programs.push_back(gcn::analyzeUnbound(l, readGuest, slots));
                    // An image its code takes for a cube may still be bound 2D (a face's coordinates
                    // sampled another way): that translation is made too, the T# its draw gives.
                    for (size_t n = programs.size(), i = 0; i < n; ++i) {
                        gcn::Program flat = programs[i];
                        bool cube = false;
                        for (gcn::ImageResource& im : flat.images)
                            if (!im.t[0] && !im.t[1] && !im.t[3] && gcn::imageDimension(flat, im) == 11) {
                                im.t[3] = 9u << 28;
                                cube = true;
                            }
                        if (cube) programs.push_back(std::move(flat));
                    }
                    for (const gcn::Program& p : programs) {
                        if (p.code.empty()) {
                            ++failed;
                            break;
                        }
                        const uint32_t base = l.stage == gcn::Stage::Pixel ? kPsBindingBase : 0;
                        const gcn::Translation t = gcn::translate(p, base, descriptorSet(p));
                        const gcn::GlslStage stage = l.stage == gcn::Stage::Pixel ? gcn::GlslStage::Fragment : gcn::GlslStage::Vertex;
                        const std::string name = spirvName(t.glsl, stage), path = spirvCachePath(t.glsl, stage);
                        if (path.empty()) continue;
                        if (access(path.c_str(), F_OK) != 0) {
                            std::string error;
                            const std::vector<uint32_t> spirv = gcn::compileGlsl(t.glsl, stage, &error);
                            if (spirv.empty()) {
                                ++failed;
                                continue;
                            }
                            saveSpirv(path, spirv);
                            ++compiled;
                        }
                        saveSource(name, p, base); // the startup compile builds its library in later runs
                        buildLibraries(p, name);
                        std::lock_guard lock(bootMutex_);
                        bootNames_.insert(name);
                        bootEntries_.insert(l.entry);
                        bootPrograms_.emplace(l.entry, p);
                    }
                    if (const size_t d = ++done; d % 500 == 0)
                        std::fprintf(stderr, "[gpu] boot compile: %zu of %zu after %.1f s\n", d, shaders.size(),
                                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                }
            });
        for (auto& t : pool) t.join();
        std::fprintf(stderr, "[gpu] boot compile: %zu registered shaders in %.1f s (%zu SPIR-V compiled, %zu libraries built, %zu failed)\n",
                     shaders.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), compiled.load(),
                     librariesBuilt_.load() - before, failed.load());
        bootRunning_ = false;
        bootDone_ = true;
    }
  public:
    // The boot compile's progress, for the corner of the window: shown once it has run a second
    // (a run with everything cached takes a fraction of one).
    bool bootProgress(size_t& done, size_t& total) {
        if (!bootRunning_.load() || std::chrono::steady_clock::now() - bootStart_.load() < std::chrono::seconds(1)) return false;
        done = bootCount_.load(), total = bootTotal_.load();
        return true;
    }
  private:
    std::atomic<size_t> bootCount_{0}, bootTotal_{0};
    std::atomic<bool> bootRunning_{false};
    std::atomic<std::chrono::steady_clock::time_point> bootStart_{};
    // A drawn shader the boot compile also translated, but otherwise (its translation depends on
    // the draw after all): logged once per shader, the boot compile's misses to look into.
    void checkBootTranslation(const gcn::Program& p, const std::string& name) {
        if (!bootDone_.load() || p.launch.stage == gcn::Stage::Compute) return;
        std::lock_guard lock(bootMutex_);
        if (bootNames_.count(name) || !bootEntries_.count(p.launch.entry) || !bootMissed_.insert(p.launch.entry).second) return;
        std::string why;
        auto add = [&](const std::string& w) { why += (why.empty() ? "" : ", ") + w; };
        auto it = bootPrograms_.find(p.launch.entry);
        if (it != bootPrograms_.end()) { // what differs, as lpn-gcn --boot reports it
            const gcn::Program& b = it->second;
            if (p.fetchSlots != b.fetchSlots)
                add("fetch slots " + std::to_string(p.fetchSlots) + " drawn, " + std::to_string(b.fetchSlots) + " at boot");
            if (p.code.size() != b.code.size()) add("code walked " + std::to_string(p.code.size()) + " vs " + std::to_string(b.code.size()));
            if (p.buffers.size() != b.buffers.size()) add("buffers " + std::to_string(p.buffers.size()) + " vs " + std::to_string(b.buffers.size()));
            if (p.images.size() != b.images.size()) add("images " + std::to_string(p.images.size()) + " vs " + std::to_string(b.images.size()));
            else
                for (size_t i = 0; i < p.images.size(); ++i) {
                    const uint32_t dp = gcn::imageDimension(p, p.images[i]), db = gcn::imageDimension(b, b.images[i]);
                    if (dp != db) add("image " + std::to_string(i) + " dimension " + std::to_string(dp) + " drawn, " + std::to_string(db) + " at boot");
                    if (p.images[i].numFormat() == 4 || p.images[i].numFormat() == 5) add("image " + std::to_string(i) + " integer");
                }
            if (p.smrd.size() != b.smrd.size()) add("scalar loads " + std::to_string(p.smrd.size()) + " vs " + std::to_string(b.smrd.size()));
            if (p.launch.stage == gcn::Stage::Pixel) {
                if (p.launch.colorUint != b.launch.colorUint || p.launch.colorSint != b.launch.colorSint) add("integer targets");
                if (p.launch.colorFormat != b.launch.colorFormat) add("export formats");
            }
            if (why.empty()) add("other");
        }
        std::fprintf(stderr, "[gpu] boot compile missed %s 0x%" PRIx64 ": its draw translates it otherwise (%s)\n",
                     p.launch.stage == gcn::Stage::Pixel ? "ps" : "vs", p.launch.entry, why.c_str());
    }
    std::unordered_map<uint64_t, gcn::Program> bootPrograms_; // what the boot compile translated, for the reasons above
    std::mutex bootMutex_;
    std::unordered_set<std::string> bootNames_;
    std::unordered_set<uint64_t> bootEntries_, bootMissed_;
    std::atomic<bool> bootDone_{false};

    void reportPrebuildDone() {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - prebuildStart_).count();
        std::fprintf(stderr, "[gpu] startup builds done: %zu pipelines in %.1f s\n", prebuildCount_, s);
        finishWarmup();
    }

    // ---- the startup compile (Compile Shaders at Startup) ----
    std::atomic<bool> warmupAll_{false}; // the game waits for the startup builds, which every worker takes
    std::atomic<int> warmupPhase_{0};     // 0 reading the lists, 1 translating shaders, 2 building pipelines
    std::atomic<size_t> warmupTranslated_{0}, warmupShaders_{0}, warmupBuilt_{0}, warmupQueued_{0}, warmupExpected_{0};
    std::atomic<bool> warmupDone_{false};
    std::chrono::steady_clock::time_point warmupStart_ = std::chrono::steady_clock::now();
    std::mutex warmupMutex_;
    std::condition_variable warmupCv_;

    void finishWarmup() {
        {
            std::lock_guard lock(warmupMutex_);
            warmupDone_ = true;
        }
        warmupCv_.notify_all();
    }

  public:
    int maxAnisotropy() {
        VkPhysicalDeviceFeatures f{};
        vkGetPhysicalDeviceFeatures(vk_.phys, &f);
        return f.samplerAnisotropy ? int(vk_.props.limits.maxSamplerAnisotropy) : 1;
    }

    bool warmupProgress(WarmupProgress& p) {
        if (!warmupAll_.load() || warmupDone_.load()) return false;
        p.phase = warmupPhase_.load();
        if (p.phase == 2) p.done = warmupBuilt_.load(), p.total = std::max(warmupQueued_.load(), warmupExpected_.load());
        else p.done = warmupTranslated_.load(), p.total = warmupShaders_.load();
        p.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - warmupStart_).count();
        return true;
    }
    void skipWarmup() {
        if (!warmupAll_.load() || warmupDone_.load()) return;
        std::fprintf(stderr, "[gpu] startup compile skipped: the rest builds in the background\n");
        finishWarmup();
    }
    // The game starts once the startup compile is done (or skipped); the builds left go back to
    // a quarter of the workers.
    void waitForWarmup() {
        if (!warmupAll_.load()) return;
        auto last = std::chrono::steady_clock::now();
        std::unique_lock lock(warmupMutex_);
        while (!warmupCv_.wait_for(lock, std::chrono::seconds(2), [&] { return warmupDone_.load(); })) {
            WarmupProgress p;
            lock.unlock();
            if (warmupProgress(p) && std::chrono::steady_clock::now() - last >= std::chrono::seconds(2)) {
                std::fprintf(stderr, "[gpu] compiling shaders: %s %zu/%zu (%.0f s)\n",
                             p.phase == 2 ? "pipelines" : "shaders", p.done, p.total, p.seconds);
                last = std::chrono::steady_clock::now();
            }
            lock.lock();
        }
        lock.unlock();
        std::fprintf(stderr, "[gpu] startup compile finished in %.1f s: %zu shaders (%zu compiled), %zu of %zu pipelines\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warmupStart_).count(),
                     warmupShaders_.load(), compiled_.load(), warmupBuilt_.load(), warmupQueued_.load());
        warmupAll_ = false;
    }

  private:
    std::atomic<size_t> prebuildLeft_{1}; // startup builds not done, plus 1 while they are being queued
    size_t prebuildCount_ = 0;
    std::chrono::steady_clock::time_point prebuildStart_;

    // Queues every pipeline in pipelines.list (newest first) for the workers.
    // The startup compile's first step, on every core: each shader the cache holds (the recorded
    // pipelines' and every other saved program, compute shaders among them) translated with
    // today's translator and its SPIR-V compiled if missing, so neither happens during play.
    void translateAll(const std::vector<std::string>& records) {
        warmupPhase_ = 1;
        std::vector<std::string> names;
        {
            std::unordered_set<std::string> seen;
            for (const auto& r : records) {
                GraphicsState s;
                std::string vs, ps;
                std::vector<VkDescriptorSetLayoutBinding> bindings;
                if (!parseRecord(r, s, vs, ps, bindings)) continue;
                for (auto* n : {&vs, &ps})
                    if (!n->empty() && seen.insert(*n).second) names.push_back(*n);
            }
            std::error_code ec;
            for (const auto& e : std::filesystem::directory_iterator(spirvDir_, ec))
                if (e.path().extension() == ".gcn" && seen.insert(e.path().stem().string()).second)
                    names.push_back(e.path().stem().string());
        }
        warmupShaders_ = names.size();
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < n; ++i)
            pool.emplace_back([&] {
                for (size_t k; (k = next++) < names.size() && !warmupDone_.load();) {
                    currentShader(names[k]);
                    ++warmupTranslated_;
                }
            });
        for (auto& t : pool) t.join();
        std::fprintf(stderr, "[gpu] startup compile: %zu shaders translated in %.1f s (%zu SPIR-V compiled, %zu renamed)\n",
                     names.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - prebuildStart_).count(),
                     compiled_.load(), retranslated_.load());
    }

    void prebuildPipelines() {
        profile::nameThread("lpn-prebuild");
        const bool warmup = warmupAll_.load();
        if (!warmup) profile::lowerThreadPriority();
        prebuildStart_ = std::chrono::steady_clock::now();
        // Each shader's module once, shared by its pipelines (one per pipeline read and created
        // ~16000 modules for ~8000 pipelines, 11 s into the run before the last was queued).
        std::unordered_map<std::string, std::shared_ptr<VkShaderModule_T>> modules;
        auto shared = [&](const std::string& name) -> std::shared_ptr<VkShaderModule_T> {
            if (name.empty()) return nullptr;
            auto [it, fresh] = modules.try_emplace(name);
            if (fresh)
                if (VkShaderModule m = loadModule(name))
                    it->second = std::shared_ptr<VkShaderModule_T>(m, [dev = vk_.device](VkShaderModule m) { vkDestroyShaderModule(dev, m, nullptr); });
            return it->second;
        };
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
        if (warmup) translateAll(records);
        if (warmup) { // the bar's total from the start, not growing as builds are queued
            std::unordered_set<std::string> distinct(records.begin(), records.end());
            warmupExpected_ = distinct.size();
        }
        warmupPhase_ = 2;
        std::unordered_set<std::string> seen;
        size_t queued = 0, renamed = 0;
        for (auto r = records.rbegin(); r != records.rend(); ++r) {
            GraphicsState s;
            std::string vsName, psName;
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            if (parseRecord(*r, s, vsName, psName, bindings)) { // its shaders as translated today
                const std::string vsNow = currentShader(vsName), psNow = currentShader(psName);
                if (vsNow != vsName || psNow != psName) {
                    vsName = vsNow, psName = psNow;
                    *r = pipelineRecord(s, vsName, psName, bindings);
                    ++renamed;
                }
            }
            if (!seen.insert(*r).second) continue;
            if (!parseRecord(*r, s, vsName, psName, bindings)) continue;
            notePipeline(s, vsName, psName, false);
            PipelineJob j;
            j.state = s;
            j.prebuild = true;
            auto vs = shared(vsName), ps = shared(psName);
            if (!vs || (!psName.empty() && !ps)) continue;
            j.vs = vs.get();
            j.ps = ps.get();
            j.keep = std::make_shared<std::pair<decltype(vs), decltype(ps)>>(std::move(vs), std::move(ps));
            j.lay = &layout(bindings);
            j.layout = j.lay->pipeline;
            j.vsName = vsName;
            j.psName = psName;
            j.geom = geometryFor(s);
            {
                std::lock_guard lock(buildMutex_);
                auto& slot = built_[*r];
                if (slot) continue; // a draw got there first
                slot = j.out = std::make_shared<BuiltPipeline>();
            }
            ++prebuildLeft_;
            ++warmupQueued_;
            queueJob(std::move(j), false);
            ++queued;
        }
        modules.clear(); // each module now lives as long as its last job
        warmupExpected_ = 0; // queued is the total now
        prebuildCount_ = queued;
        std::fprintf(stderr, "[gpu] building %zu pipelines from earlier runs (%zu shaders translated again, queued in %.1f s)\n",
                     queued, retranslated_.load(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - prebuildStart_).count());
        if (prebuildLeft_.fetch_sub(1) == 1) { // the workers finished them all already
            if (queued) reportPrebuildDone();
            else finishWarmup();
        }

        // Keep the list from growing without bound, and its shader names current: rewrite it
        // without duplicates, with today's names.
        if (records.size() > seen.size() + 1024 || renamed) {
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
        rasterState(s, rs);
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthState(s, ds);
        VkPipelineColorBlendAttachmentState att[8]{};
        uint32_t colorCount = 0;
        VkFormat formats[8]{};
        blendState(s, att, formats, colorCount);
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

    // The rasteriser, depth/stencil and blend state of `s`: baked into pipelines (buildGraphics),
    // or with pipeline libraries the first two set per draw (libraryState) and the last baked
    // into the fragment output library.
    static void rasterState(const GraphicsState& s, VkPipelineRasterizationStateCreateInfo& rs) {
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
    }
    static void depthState(const GraphicsState& s, VkPipelineDepthStencilStateCreateInfo& ds) {
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
    }
    static void blendState(const GraphicsState& s, VkPipelineColorBlendAttachmentState* att, VkFormat* formats, uint32_t& colorCount) {
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
    }

    // Pipeline libraries: the state libraryState sets per draw, as the pipeline would have it.
    static void libraryState(const GraphicsState& s, DynamicState& d) {
        VkPipelineRasterizationStateCreateInfo rs{};
        rasterState(s, rs);
        VkPipelineDepthStencilStateCreateInfo ds{};
        depthState(s, ds);
        const uint32_t v[15] = {rs.cullMode, uint32_t(rs.frontFace), rs.depthBiasEnable, ds.depthTestEnable, ds.depthWriteEnable,
                                uint32_t(ds.depthCompareOp), ds.stencilTestEnable,
                                uint32_t(ds.front.failOp), uint32_t(ds.front.passOp), uint32_t(ds.front.depthFailOp), uint32_t(ds.front.compareOp),
                                uint32_t(ds.back.failOp), uint32_t(ds.back.passOp), uint32_t(ds.back.depthFailOp), uint32_t(ds.back.compareOp)};
        std::memcpy(d.lib, v, sizeof v);
    }

    // ---- pipeline libraries (docs/shader-precompile.md stage 3) ----
    // With fast-linked graphics pipeline libraries (generic translation; LPN_PIPELINE_LIBRARIES=0
    // turns them off) a pipeline is linked from four parts: vertex input (per topology),
    // pre-rasterisation (per vertex shader, with the rect-list geometry shader), fragment shader
    // (per pixel shader) and fragment output (per targets' formats and blend). The shader parts are
    // compiled once per shader, whatever state it is drawn with; cull mode, depth and stencil state
    // are dynamic, set per draw from DynamicState; and a link without link-time optimisation takes
    // a fraction of a millisecond, so new state no longer compiles anything.
    struct LibSlot {
        std::once_flag once;
        VkPipeline p = VK_NULL_HANDLE;
    };
    std::mutex libMutex_;
    std::unordered_map<std::string, std::unique_ptr<LibSlot>> libs_; // by kind and what it is made of
    template <class F> VkPipeline library(const std::string& key, F&& make) {
        LibSlot* slot;
        {
            std::lock_guard lock(libMutex_);
            auto& u = libs_[key];
            if (!u) u = std::make_unique<LibSlot>();
            slot = u.get();
        }
        std::call_once(slot->once, [&] { slot->p = make(); }); // others wanting it wait for this build
        return slot->p;
    }
    VkPipeline makeLibrary(VkGraphicsPipelineCreateInfo& gci, VkGraphicsPipelineLibraryFlagsEXT flags, const char* what) {
        VkGraphicsPipelineLibraryCreateInfoEXT li{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
        li.flags = flags;
        li.pNext = gci.pNext;
        gci.pNext = &li;
        gci.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        VkPipeline p = VK_NULL_HANDLE;
        const VkResult r = vkCreateGraphicsPipelines(vk_.device, pipelineCache_, 1, &gci, nullptr, &p);
        if (r != VK_SUCCESS) std::fprintf(stderr, "[gpu] %s pipeline library not created: %s\n", what, vkResultName(r));
        else ++newPipelines_; // pipelines.bin is written again with it (savePipelineCache)
        return r == VK_SUCCESS ? p : VK_NULL_HANDLE;
    }

    // The pipeline libraries of one shader (see linkGraphics), which the startup compile also
    // builds for every shader it knows (warmLibrary). `variant` names the geometry shader's input.
    VkPipeline preLibrary(const std::string& vsName, VkShaderModule vs, VkShaderModule geom, const std::string& variant,
                          const Layout& lay) {
        VkPipelineRenderingCreateInfo none{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; // view mask 0
        return library("pre" + vsName + variant + "/" + lay.preKey, [&] {
            VkPipelineShaderStageCreateInfo st[2]{};
            for (auto& i : st) i.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, i.pName = "main";
            st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            st[0].module = vs;
            st[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
            st[1].module = geom;
            VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vp.viewportCount = 1;
            vp.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.lineWidth = 1.0f;
            const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT,  VK_DYNAMIC_STATE_SCISSOR,    VK_DYNAMIC_STATE_DEPTH_BIAS,
                                          VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE};
            VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dy.dynamicStateCount = uint32_t(std::size(dyn));
            dy.pDynamicStates = dyn;
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.pNext = &none;
            gci.stageCount = geom ? 2 : 1;
            gci.pStages = st;
            gci.pViewportState = &vp;
            gci.pRasterizationState = &rs;
            gci.pDynamicState = &dy;
            gci.layout = lay.pre;
            return makeLibrary(gci, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, "vertex shader");
        });
    }
    VkPipeline fragLibrary(const std::string& psName, VkShaderModule ps, const Layout& lay) {
        VkPipelineRenderingCreateInfo none{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; // view mask 0
        return library("frag" + psName + "/" + lay.fragKey, [&] {
            VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            st.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            st.module = ps;
            st.pName = "main";
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,       VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                                          VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
                                          VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,     VK_DYNAMIC_STATE_STENCIL_OP,
                                          VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,    VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                          VK_DYNAMIC_STATE_STENCIL_REFERENCE};
            VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dy.dynamicStateCount = uint32_t(std::size(dyn));
            dy.pDynamicStates = dyn;
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.pNext = &none;
            gci.stageCount = ps ? 1 : 0;
            gci.pStages = &st;
            gci.pMultisampleState = &ms;
            gci.pDepthStencilState = &ds;
            gci.pDynamicState = &dy;
            gci.layout = lay.frag;
            return makeLibrary(gci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, "pixel shader");
        });
    }

    VkPipeline linkGraphics(const PipelineJob& j) {
        const GraphicsState& s = j.state;
        const Layout& lay = *j.lay;
        VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        switch (s.topology) {
        case 1: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case 2: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case 5: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case 6: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 19: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; break;
        default: break;
        }

        const VkPipeline input = library("vi" + std::to_string(topo), [&] {
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = topo;
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.pVertexInputState = &vi;
            gci.pInputAssemblyState = &ia;
            return makeLibrary(gci, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, "vertex input");
        });
        const bool rect = s.topology == 17 || s.topology == 19;
        const VkPipeline pre = preLibrary(j.vsName, j.vs, j.geom, rect ? "/" + std::to_string(s.topology) + "/" + std::to_string(s.vsOutputs) : "", lay);
        const VkPipeline frag = fragLibrary(j.psName, j.ps, lay);
        std::string outKey = "out";
        outKey.append(reinterpret_cast<const char*>(s.colorFormats), sizeof s.colorFormats);
        outKey.append(reinterpret_cast<const char*>(s.blend), sizeof s.blend);
        for (uint32_t v : {s.writeMasks, s.depthFormat, s.stencil}) outKey.append(reinterpret_cast<const char*>(&v), 4);
        const VkPipeline output = library(outKey, [&] {
            VkPipelineColorBlendAttachmentState att[8]{};
            uint32_t colorCount = 0;
            VkFormat formats[8]{};
            blendState(s, att, formats, colorCount);
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount = colorCount;
            cb.pAttachments = att;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};
            VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dy.dynamicStateCount = 1;
            dy.pDynamicStates = dyn;
            VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            ri.colorAttachmentCount = colorCount;
            ri.pColorAttachmentFormats = formats;
            ri.depthAttachmentFormat = VkFormat(s.depthFormat);
            if (s.stencil) ri.stencilAttachmentFormat = VkFormat(s.depthFormat);
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.pNext = &ri;
            gci.pColorBlendState = &cb;
            gci.pMultisampleState = &ms;
            gci.pDynamicState = &dy;
            return makeLibrary(gci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, "output");
        });
        if (!input || !pre || !frag || !output) return VK_NULL_HANDLE;
        const VkPipeline parts[] = {input, pre, frag, output};
        VkPipelineLibraryCreateInfoKHR li{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
        li.libraryCount = uint32_t(std::size(parts));
        li.pLibraries = parts;
        VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gci.pNext = &li;
        gci.layout = lay.pipeline;
        VkPipeline p = VK_NULL_HANDLE;
        const VkResult r = vkCreateGraphicsPipelines(vk_.device, VK_NULL_HANDLE, 1, &gci, nullptr, &p);
        if (r != VK_SUCCESS) std::fprintf(stderr, "[gpu] pipeline link failed: %s\n", vkResultName(r));
        return r == VK_SUCCESS ? p : VK_NULL_HANDLE;
    }

    void setDynamicState(const DynamicState& d, uint32_t depthControl) {
        if (libraries_ && (!R().bound_.state || std::memcmp(R().bound_.lib, d.lib, sizeof d.lib) != 0)) {
            std::memcpy(R().bound_.lib, d.lib, sizeof d.lib);
            rec([c = R().cmd_, v = std::to_array(d.lib)] {
                vkCmdSetCullMode(c, v[0]);
                vkCmdSetFrontFace(c, VkFrontFace(v[1]));
                vkCmdSetDepthBiasEnable(c, v[2]);
                vkCmdSetDepthTestEnable(c, v[3]);
                vkCmdSetDepthWriteEnable(c, v[4]);
                vkCmdSetDepthCompareOp(c, VkCompareOp(v[5]));
                vkCmdSetDepthBoundsTestEnable(c, VK_FALSE);
                vkCmdSetStencilTestEnable(c, v[6]);
                vkCmdSetStencilOp(c, VK_STENCIL_FACE_FRONT_BIT, VkStencilOp(v[7]), VkStencilOp(v[8]), VkStencilOp(v[9]), VkCompareOp(v[10]));
                vkCmdSetStencilOp(c, VK_STENCIL_FACE_BACK_BIT, VkStencilOp(v[11]), VkStencilOp(v[12]), VkStencilOp(v[13]), VkCompareOp(v[14]));
            });
        }
        // Scale is in 1/16ths of the slope; the offset is in the depth format's units, as Vulkan's.
        if (!R().bound_.state || std::memcmp(R().bound_.polyOffset, d.polyOffset, sizeof d.polyOffset) != 0) {
            std::memcpy(R().bound_.polyOffset, d.polyOffset, sizeof d.polyOffset);
            rec([c = R().cmd_, k = std::bit_cast<float>(d.polyOffset[2]), cl = std::bit_cast<float>(d.polyOffset[0]),
                 sl = std::bit_cast<float>(d.polyOffset[1]) / 16.0f] { vkCmdSetDepthBias(c, k, cl, sl); });
        }
        const uint32_t back = (depthControl >> 7) & 1 ? d.stencilMasksBf : d.stencilMasks;
        if (R().bound_.state && R().bound_.stencil[0] == d.stencilMasks && R().bound_.stencil[1] == back) return;
        R().bound_.stencil[0] = d.stencilMasks, R().bound_.stencil[1] = back;
        rec([c = R().cmd_, front = d.stencilMasks, back] {
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
        uint32_t blend[4] = {}, polyOffset[3] = {}, stencil[2] = {}, lib[15] = {};
        // Pipeline libraries: the pixel shader's descriptors in the set bound last (pushDescriptors),
        // by binding from kPsBindingBase, for the layout `psLayout`.
        const void* psLayout = nullptr;
        uint64_t psHas = 0;
        VkDescriptorType psType[64] = {};
        VkDescriptorBufferInfo psBuf[64] = {};
        VkDescriptorImageInfo psImg[64] = {};
        VkViewport viewport{};
        VkRect2D scissor{};
        VkBuffer index = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    };

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
        // The Anisotropic Filtering setting is part of the key: a change makes new samplers.
        const uint32_t userAniso = uint32_t(std::max(displaySettings().anisotropy, 0));
        std::array<uint32_t, 6> key{s[0], s[1], s[2], s[3], compare, userAniso};
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
        // XY filters: 0 point, 1 bilinear, 2 anisotropic point, 3 anisotropic bilinear.
        const uint32_t mag = (s[2] >> 20) & 3, min = (s[2] >> 22) & 3, mip = (s[2] >> 26) & 3;
        ci.magFilter = (mag & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.minFilter = (min & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.minLod = float(s[1] & 0xFFF) / 256.0f;
        ci.maxLod = float((s[1] >> 12) & 0xFFF) / 256.0f;
        if (mip == 0) ci.maxLod = std::min(ci.maxLod, 0.25f); // no mip filter: the base level only
        ci.minLod = std::min(ci.minLod, ci.maxLod);
        // LOD_BIAS: signed, 8 fraction bits.
        const int32_t bias = int32_t(s[2] << 18) >> 18;
        ci.mipLodBias = std::clamp(float(bias) / 256.0f, -vk_.props.limits.maxSamplerLodBias,
                                   vk_.props.limits.maxSamplerLodBias);
        // MAX_ANISO_RATIO 0-4: 1x to 16x, used when a filter is anisotropic. The Anisotropic
        // Filtering setting raises every filtered (non-depth-compare) sampler to at least its Nx.
        // LPN_ANISO=N forces Nx on them instead (1 turns it off).
        static const int forced = getenv("LPN_ANISO") ? std::atoi(getenv("LPN_ANISO")) : 0;
        float aniso = (mag >= 2 || min >= 2) ? float(1u << std::min((s[0] >> 9) & 7, 4u)) : 1.0f;
        if ((min & 1) && !compare) {
            if (forced > 0) aniso = float(forced);
            else if (userAniso > 1) aniso = std::max(aniso, float(userAniso));
        }
        static const bool canAniso = [this] { // the device enables it whenever it has it (vk.cpp)
            VkPhysicalDeviceFeatures f{};
            vkGetPhysicalDeviceFeatures(vk_.phys, &f);
            return f.samplerAnisotropy == VK_TRUE;
        }();
        if (aniso > 1.0f && canAniso) {
            ci.anisotropyEnable = VK_TRUE;
            ci.maxAnisotropy = std::min(aniso, vk_.props.limits.maxSamplerAnisotropy);
        }
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
    // A texture just checked and found unchanged: settled once unchanged for a while.
    GuestTexture* settleTexture(GuestTexture& tex, uint64_t base, size_t span, size_t tiled) {
        if (resolveAhead_ && !tex.settled && frame_ - tex.changedFrame >= kSettleFrames) {
            tex.settled = true;
            tex.base = base, tex.span = span, tex.tiled = tiled;
            if (!tex.swept) {
                tex.swept = true;
                sweptTextures_.push_back(&tex);
            }
        }
        R().lastTexture_ = &tex;
        return &tex;
    }
    // Found changed (by the sweep): checked in full at its next use, which every image binding
    // remembered from earlier frames is dropped for.
    void unsettleTexture(GuestTexture& t) {
        t.settled = false;
        t.sweptChanged = true;
        ++imageGen_, ++R().prof_.imageGens;
    }
    std::vector<GuestTexture*> sweptTextures_; // settled ones (and some no longer), for the sweep

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
        if (tex.img.image && tex.checkedFrame == frame_) {
            R().lastTexture_ = &tex;
            return &tex;
        }
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
        bool asyncChanged = std::exchange(tex.sweptChanged, false);
        if (!asyncChanged && tex.check && tex.check->state.load(std::memory_order_acquire) == 2) {
            tex.check->state.store(0, std::memory_order_relaxed);
            asyncChanged = tex.check->result != tex.hash;
        }
        // The cheap checks below only follow a texture while it is in use. One left unused for a
        // few frames is fully hashed in line at its next use: the game loads each special move's
        // banner into the same memory, and an entry last used for an earlier move's banner (never
        // checked while idle, so "unchanged for 10 s") was bound unchecked or passed the
        // fingerprint, which mostly samples transparent texels. It flashed the earlier banner.
        // Settled textures are left to the sweep, which checks them every frame, used or not.
        const bool idle = tex.img.image && !tex.settled && frame_ - tex.checkedFrame > 2;
        if (!asyncChanged && !idle && !fullTexChecks_ && tex.img.image && frame_ - tex.changedFrame >= 60 &&
            (++tex.uses + tex.phase) % 8 != 0) {
            tex.checkedFrame = frame_;
            // Unchanged for 10 s: fingerprinted on every other use only. A syscall each, they
            // took ~16% of the recorder looking up the steps after New Game (hundreds of textures).
            if (frame_ - tex.changedFrame >= 600 && (tex.uses + tex.phase) % 2 != 0) return &tex;
            if (sampleHash(t.base(), span) == tex.sample) return &tex;
        }
        uint32_t mode = t.tileMode();
        if (!tiledSliceSize(mode, ep, eh, bpe)) {
            warnOnce("texture tile mode " + std::to_string(mode) + " is read as linear");
            mode = 8;
        }
        // The T#'s mip chain (levels 0 to LAST_LEVEL), uploaded whole: sampled with only level 0,
        // minified textures (hair, skin detail, fences) shimmered and broke into pixels.
        // LPN_NO_MIPS=1: level 0 only, as before.
        uint32_t levels = 1;
        if (!noMips_ && slices == 1) { // cube maps and arrays: level 0 only (their mip layout is not verified)
            uint32_t full = 1;
            while ((std::max(w, h) >> full) != 0) ++full;
            levels = std::clamp(t.lastLevel() + 1, 1u, std::min(full, kMaxMips));
        }
        MipLevel mips[kMaxMips];
        size_t tiled = mipLayout(mode, w, h, pitchPx, bc, bpe, slices, levels, t.pow2Pad(), mips);
        if (levels == 1) { // as before: level 0 exactly as the T# describes it
            mips[0] = {0, tiledSliceSize(mode, ep, eh, bpe), ep, ew, eh};
            tiled = mips[0].sliceBytes * slices;
        }
        // Linear copies of the levels, each at a 16-byte boundary.
        size_t linearAt[kMaxMips];
        VkDeviceSize bytes = 0;
        for (uint32_t m = 0; m < levels; ++m) {
            linearAt[m] = bytes;
            bytes += (VkDeviceSize(mips[m].width) * mips[m].height * bpe * slices + 15) & ~VkDeviceSize(15);
        }
        if (!bytes || bytes > kRingSize / 2 || tiled > (256u << 20)) return nullptr;
        // The full check of a texture already uploaded runs on a hasher thread, its result taken
        // at the next use (~2.7 ms a frame off the recorder in crowds). The fingerprint is still
        // taken now, so a texture rewritten as a whole (a video frame) is uploaded at once; a
        // partial rewrite shows one use later than before. LPN_SYNC_TEX_CHECKS=1: all in line.
        if (tex.img.image && asyncTexChecks_ && !asyncChanged && !idle && !fullTexChecks_) {
            tex.checkedFrame = frame_;
            if (sampleHash(t.base(), span) == tex.sample) {
                if (!tex.check) tex.check = std::make_shared<AsyncHash>();
                AsyncHash& c = *tex.check;
                const int st = c.state.load(std::memory_order_acquire);
                if (st == 2) {
                    c.state.store(0, std::memory_order_relaxed);
                    if (c.result == tex.hash) return settleTexture(tex, t.base(), span, tiled);
                    // changed (or unreadable): checked again in line below
                } else {
                    if (st == 0) queueTexHash(tex.check, t.base(), tiled);
                    return settleTexture(tex, t.base(), span, tiled);
                }
            }
        }
        uint64_t h64;
        {
            ProfScope ps(R().prof_, kProfTexRead);
            ++R().prof_.texChecks;
            h64 = hasher_.hash(t.base(), tiled);
            if (!h64) {
                warnOnce("texture at 0x" + hex(t.base()) + " is not readable");
                return nullptr;
            }
        }
        tex.checkedFrame = frame_;
        tex.sample = sampleHash(t.base(), span);
        if (!tex.img.image || tex.hash != h64) tex.changedFrame = frame_, tex.settled = false;
        if (tex.img.image && tex.hash == h64) return settleTexture(tex, t.base(), span, tiled);
        ProfScope pu(R().prof_, kProfTexUpload);
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
        // Debugging: LPN_DUMP_MIPS=<n> writes the first n mipmapped 2D textures up to 2048x2048 (LPN_DUMP_MIPS_MIN=<size>: at least that wide or tall) as
        // they sit in memory (twice level 0's size, enough for any layout of the other levels).
        if (static std::atomic<int> dumpMips = getenv("LPN_DUMP_MIPS") ? std::atoi(getenv("LPN_DUMP_MIPS")) : 0;
            levels > 1 && slices == 1 && t.type() == 9 && w <= 2048 && h <= 2048 &&
            std::max(w, h) >= (getenv("LPN_DUMP_MIPS_MIN") ? uint32_t(std::atoi(getenv("LPN_DUMP_MIPS_MIN"))) : 0u) &&
            dumpMips.load() > 0 &&
            dumpMips.fetch_sub(1) > 0) {
            std::vector<uint8_t> raw(mips[0].sliceBytes * 2);
            if (readGuest(t.base(), raw.data(), raw.size()))
                if (FILE* f = std::fopen(("mips-" + hex(t.base()) + "-" + std::to_string(w) + "x" + std::to_string(h) +
                                          "-df" + std::to_string(t.dataFormat()) + "-mode" + std::to_string(mode) + "-p" +
                                          std::to_string(t.pitch()) + "-last" + std::to_string(t.lastLevel()) + "-pow2" +
                                          std::to_string(t.pow2Pad()) + ".bin").c_str(), "wb")) {
                    std::fwrite(raw.data(), 1, raw.size(), f);
                    std::fclose(f);
                }
        }
        ++R().prof_.texUploads;
        R().prof_.texBytes += bytes;
        Prologue pg(*this); // on a recorder thread: ahead of the segments
        if (!ringFits(bytes)) {
            flush();
            begin();
        }
        const VkDeviceSize at = ringAlloc(bytes, 16);
        uint8_t* dst = static_cast<uint8_t*>(R().ringPtr_) + at;
        for (uint32_t m = 0; m < levels; ++m) {
            const MipLevel& l = mips[m];
            const size_t linSlice = size_t(l.width) * l.height * bpe;
            for (uint32_t s = 0; s < slices; ++s)
                detile(mode, scratch_.data() + l.offset + s * l.sliceBytes, dst + linearAt[m] + s * linSlice, l.width,
                       l.height, l.pitch, bpe);
        }
        if (!tex.img.image) {
            tex.phase = textureCount_++ % 8;
            const VkImageViewType vt = viewType(t.type());
            if (!createImage(tex.img, format, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false,
                             vt, slices, levels))
                return nullptr;
        }
        tex.hash = h64;
        fullBarrier();
        std::array<VkBufferImageCopy, kMaxMips> cs{};
        for (uint32_t m = 0; m < levels; ++m) {
            VkBufferImageCopy& c = cs[m];
            c.bufferOffset = at + linearAt[m];
            c.bufferRowLength = bc ? mips[m].width * 4 : mips[m].width;
            c.bufferImageHeight = bc ? mips[m].height * 4 : mips[m].height;
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, t.type() == 10 ? 1 : layers};
            c.imageExtent = {std::max(w >> m, 1u), std::max(h >> m, 1u), t.type() == 10 ? t.depth() : 1};
        }
        rec([cb = R().cmd_, buf = R().ring_, im = tex.img.image, cs, levels] {
            vkCmdCopyBufferToImage(cb, buf, im, VK_IMAGE_LAYOUT_GENERAL, levels, cs.data());
        });
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
    // LPN_FULL_TEX_CHECKS=1: every texture is fully hashed at its first use in each frame (slow;
    // rules stale textures in or out).
    const bool fullTexChecks_ = getenv("LPN_FULL_TEX_CHECKS") != nullptr;
    std::mutex texHashMutex_;
    std::condition_variable texHashCv_;
    std::deque<TexHashJob> texHashJobs_;

    struct ImageBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        float scale[2] = {1, 1};
    };

    void gpuWrote(Surface& s) {
        // The memory is sampled when the GPU renders into it, not when the target is next sampled
        // as a texture: by then the game may have loaded a texture there (a special move's banner,
        // in memory a render target had used), which became the baseline, so the old render
        // target was bound in its place. Once a frame per target.
        const bool again = s.writtenFrame == frame_;
        if (again && !s.cpuOwned && !(s.writtenBack && s.memSample)) return;
        if (sharedMiss()) return;
        if (!again || s.writtenBack) s.memSample = s.writtenBack || !s.memBytes ? 0 : sampleHash(s.addr, s.memBytes);
        s.writtenFrame = frame_;
        if (s.cpuOwned) ++imageGen_, ++R().prof_.imageGens; // sampled as a render target again, not from memory
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
        const uint64_t sample = s.writtenBack || !s.memBytes
                                    ? sampleHash(t.base(), size_t(t.pitch()) * t.height() * s.elementBytes)
                                    : sampleHash(s.addr, s.memBytes);
        if (!s.memSample) s.memSample = sample;
        else if (sample && sample != s.memSample) s.cpuOwned = true, ++imageGen_, ++R().prof_.imageGens;
        // A texture bound to a render target the GPU left alone for a while: what to look for
        // when a texture shows something else (once per target and write).
        if (!s.cpuOwned && frame_ > s.writtenFrame + 30 && s.loggedFrame != s.writtenFrame) {
            s.loggedFrame = s.writtenFrame;
            std::fprintf(stderr, "[gpu] texture 0x%" PRIx64 " %ux%u fmt %u/%u sampled from render target %ux%u drawn %" PRIu64
                                 " frames ago\n", t.base(), t.width(), t.height(), t.dataFormat(), t.numFormat(),
                         s.img.width, s.img.height, frame_ - s.writtenFrame);
        }
        if (s.cpuOwned)
            std::fprintf(stderr, "[gpu] render target 0x%" PRIx64 " rewritten in memory: sampled from memory\n", s.addr);
        return s.cpuOwned;
    }

    // Draws in a frame resolve the same few hundred T#/S# pairs over and over (a crowd repeats its
    // materials): the first resolution in a frame does the checks, later ones reuse its result
    // until an image is created or a render target goes back to the GPU (imageGen_).
    struct ImageMemo {
        uint32_t t[8], s[4];
        bool compare;
        bool persist = false; // a settled texture's: good in later frames too, while gen holds
        uint64_t frame = ~0ull, gen = 0;
        ImageBinding b;
        VkImage sampled; // the render target it samples, if any (for feedback checks)
    };
    // Two ways of 4,096 sets: a city view resolves thousands of pairs a frame, and with 1,024
    // single entries they evicted each other.
    static constexpr size_t kImageMemos = 8192;
    static size_t imageMemoSet(const gcn::ImageResource& t) {
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t w : t.t) h = (h ^ w) * 0x100000001b3ull;
        for (uint32_t w : t.s) h = (h ^ w) * 0x100000001b3ull;
        return ((h ^ (h >> 29) ^ t.compare) & (kImageMemos / 2 - 1)) * 2;
    }
    ImageMemo& imageMemo(const gcn::ImageResource& t) {
        const size_t set = imageMemoSet(t);
        ImageMemo& a = imageMemos_[set];
        ImageMemo& b = imageMemos_[set + 1];
        if (memoHit(b, t)) return b;
        if (memoHit(a, t)) return a;
        // A miss replaces a stale way first, else the older one (the way not hit last moves to b).
        const bool aLive = memoLive(a);
        if (aLive) std::swap(a, b);
        return a;
    }
    bool memoLive(const ImageMemo& m) const { return (m.frame == frame_ || m.persist) && m.gen == imageGen_; }
    bool memoHit(const ImageMemo& m, const gcn::ImageResource& t) const {
        return memoLive(m) && m.compare == t.compare && std::equal(t.t, t.t + 8, m.t) &&
               std::equal(t.s, t.s + 4, m.s);
    }
    // An image already resolved this frame that samples no render target: what fastDraw may bind
    // without recording anything (an upload) or affecting the pass.
    bool peekImage(const gcn::ImageResource& t, ImageBinding& b) {
        if (t.storage || trace_) return false;
        // Read only (recorder threads look up under a shared lock): no way swapped on a miss.
        const size_t set = imageMemoSet(t);
        const ImageMemo* m = &imageMemos_[set + 1];
        if (!memoHit(*m, t)) m = &imageMemos_[set];
        if (!memoHit(*m, t) || m->sampled) return false;
        b = m->b;
        return true;
    }
    ImageBinding resolveImage(const gcn::ImageResource& t) {
        if (R().lockShared_) { // read only: a hit, or the draw starts over
            if (t.storage || trace_) return sharedMiss(), ImageBinding{};
            const size_t set = imageMemoSet(t);
            const ImageMemo* m = &imageMemos_[set + 1];
            if (!memoHit(*m, t)) m = &imageMemos_[set];
            if (!memoHit(*m, t)) return sharedMiss(), ImageBinding{};
            if (m->sampled) R().sampledSurfaces_.push_back(m->sampled);
            return m->b;
        }
        if (t.storage || trace_) return resolveImageUncached(t); // storage: marks the target written
        ImageMemo& m = imageMemo(t);
        if (memoHit(m, t)) {
            if (m.sampled) R().sampledSurfaces_.push_back(m.sampled);
            return m.b;
        }
        const size_t before = R().sampledSurfaces_.size();
        ++R().prof_.imageMisses;
        R().lastTexture_ = nullptr;
        const ImageBinding b = resolveImageUncached(t);
        std::copy(t.t, t.t + 8, m.t);
        std::copy(t.s, t.s + 4, m.s);
        m.compare = t.compare;
        m.frame = frame_;
        m.gen = imageGen_;
        m.b = b;
        m.sampled = R().sampledSurfaces_.size() > before ? R().sampledSurfaces_.back() : VK_NULL_HANDLE;
        // A settled texture's binding holds until the sweep finds it changed (which moves imageGen_).
        m.persist = !m.sampled && R().lastTexture_ && R().lastTexture_->settled;
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
            R().sampledSurfaces_.push_back(img.image);
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
            R().sampledSurfaces_.push_back(s.img.image);
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
            // An all-zero T#: an image in code this launch skips (generic translation), bound to
            // the placeholder without a warning.
            if (t.dataFormat() || addr || (t.t[3] & 0x0FFFFFFF)) warnOnce("texture format " + std::to_string(t.dataFormat()) + "/" + std::to_string(nf) + " unsupported");
        } else if (GuestTexture* g = guestTexture(t, format)) {
            b.view = view(g->img, format, vt, swz, 0, t.baseLevel(), t.lastLevel() + 1 - std::min(t.baseLevel(), t.lastLevel()));
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

    // ---- DLSS (gpu/dlss.h) ---------------------------------------------------------------------
    // The game anti-aliases its scene with SMAA in three full-screen draws: edges, blend weights,
    // then a blend that writes the scene colour back. With DLSS on, the first two are skipped and
    // the blend becomes DLSS, given the scene colour, its depth and motion vectors this renderer
    // makes from the depth and the camera (so only the camera's motion: moving characters rely on
    // DLSS's own handling). Scene draws (with that depth buffer bound before SMAA, and 3D draws
    // tested against it after) are jittered by moving the viewport a sub-pixel amount each frame.

    // The SMAA pixel shaders, by a hash of their first 256 bytes of code.
    static uint8_t smaaRole(uint64_t codeHash) {
        switch (codeHash) {
        case 0x675ef5b209a40f60ull: return 1; // edges
        case 0x344b18cdf849ff76ull: return 2; // blend weights
        case 0xb267ed0abad15b02ull: return 3; // blend
        default: return 0;
        }
    }
    static uint64_t fnv(const uint8_t* p, size_t n) {
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
        return h;
    }

    // What the scene's vertex shaders read: projection rows (16 floats at 0x0 of the constant
    // buffer whose V# their user SGPRs s[12:13] point at), the view rotation rows (0x40, 0x50,
    // 0x60) and the camera position (0x220), positions being camera-relative; with the viewport.
    struct Camera {
        bool valid = false;
        double proj[16] = {}, rot[9] = {}, pos[3] = {};
        float xs = 0, xo = 0, ys = 0, yo = 0, zs = 1, zo = 0;
    };

    void captureCamera(const gcn::Program& vs, const DrawRegs& r) {
        if (sharedMiss()) return;
        if (++camTries_ > 8) return; // shaders of another layout: not this frame
        const uint64_t table = vs.launch.userData[12] | uint64_t(vs.launch.userData[13]) << 32;
        uint32_t sharp[4];
        float c[28], pos[3];
        if (!readGuest(table, sharp, sizeof sharp)) return;
        const uint64_t base = sharp[0] | uint64_t(sharp[1] & 0xFFF) << 32;
        if (!readGuest(base, c, sizeof c) || !readGuest(base + 0x220, pos, sizeof pos)) return;
        Camera k;
        for (int i = 0; i < 16; ++i) k.proj[i] = c[i];
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) k.rot[row * 3 + col] = c[16 + row * 4 + col];
        for (int i = 0; i < 3; ++i) k.pos[i] = pos[i];
        // A perspective projection (w = +-view z) and a rotation: otherwise not what we think.
        bool ok = k.proj[12] == 0 && k.proj[13] == 0 && std::fabs(std::fabs(k.proj[14]) - 1) < 1e-3 && k.proj[15] == 0;
        for (int row = 0; row < 3 && ok; ++row) {
            const double* v = &k.rot[row * 3];
            ok = std::fabs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1) < 0.01;
        }
        for (double x : pos) ok = ok && std::isfinite(x);
        if (!ok) {
            if (!camWarned_) std::fprintf(stderr, "[dlss] scene constants not as expected: camera motion off\n");
            camWarned_ = true;
            return;
        }
        const float* v = reinterpret_cast<const float*>(&r.ctx[reg::kVportXScale]);
        k.xs = v[0], k.xo = v[1], k.ys = v[2], k.yo = v[3], k.zs = v[4], k.zo = v[5];
        k.valid = k.xs != 0 && k.ys != 0 && k.zs != 0;
        cam_ = k;
    }

    // Row-major 4x4 inverse (Gauss-Jordan); false when singular.
    static bool invert(const double* m, double* out) {
        double a[4][8];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j) a[i][j] = j < 4 ? m[i * 4 + j] : double(j - 4 == i);
        for (int col = 0; col < 4; ++col) {
            int p = col;
            for (int i = col + 1; i < 4; ++i)
                if (std::fabs(a[i][col]) > std::fabs(a[p][col])) p = i;
            if (std::fabs(a[p][col]) < 1e-12) return false;
            std::swap(a[p], a[col]);
            const double d = a[col][col];
            for (double& x : a[col]) x /= d;
            for (int i = 0; i < 4; ++i)
                if (i != col) {
                    const double f = a[i][col];
                    for (int j = 0; j < 8; ++j) a[i][j] -= f * a[col][j];
                }
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) out[i * 4 + j] = a[i][4 + j];
        return true;
    }
    static void mul(const double* a, const double* b, double* out) {
        double t[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                t[i * 4 + j] = 0;
                for (int k = 0; k < 4; ++k) t[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        std::memcpy(out, t, sizeof t);
    }
    // Current frame NDC (x, y, depth, 1) to the previous frame's clip position.
    static bool reprojection(const Camera& cur, const Camera& prev, double* m) {
        auto view = [](const Camera& k, double* v) {
            for (int i = 0; i < 16; ++i) v[i] = i == 15;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) v[row * 4 + col] = k.rot[row * 3 + col];
        };
        double vc[16], vp[16], ivc[16], ipc[16], t[16];
        view(cur, vc);
        view(prev, vp);
        if (!invert(vc, ivc) || !invert(cur.proj, ipc)) return false;
        for (int i = 0; i < 16; ++i) t[i] = i % 5 == 0;
        for (int i = 0; i < 3; ++i) t[i * 4 + 3] = cur.pos[i] - prev.pos[i];
        mul(prev.proj, vp, m);
        mul(m, t, m);
        mul(m, ivc, m);
        mul(m, ipc, m);
        return true;
    }

    static constexpr const char* kMotionGlsl = R"(#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D depthTex;
layout(set = 0, binding = 1, rg16f) uniform writeonly image2D motion;
layout(push_constant) uniform P {
    mat4 reproject; // current NDC (x, y, depth, 1) to the previous clip position
    vec4 vp;        // viewport offset x, y and scale x, y: pixel = ndc * scale + offset
    vec4 z;         // depth offset, scale; this frame's viewport jitter x, y (pixels)
    ivec4 size;
} p;
void main() {
    ivec2 c = ivec2(gl_GlobalInvocationID.xy);
    if (c.x >= p.size.x || c.y >= p.size.y) return;
    float d = texelFetch(depthTex, c, 0).r;
    vec2 pix = vec2(c) + 0.5 - p.z.zw; // where the pixel's surface is without the jitter
    vec4 prev = p.reproject * vec4((pix - p.vp.xy) / p.vp.zw, (d - p.z.x) / p.z.y, 1.0);
    vec2 was = prev.xy / prev.w * p.vp.zw + p.vp.xy;
    imageStore(motion, c, vec4(prev.w > 0.0 ? was - pix : vec2(0.0), 0.0, 0.0));
}
)";
    struct MotionPush {
        float m[16];
        float vp[4], z[4];
        int32_t size[4];
    };

    bool motionPipeline() {
        if (mvPipe_) return true;
        const VkShaderModule sm = module(kMotionGlsl, gcn::GlslStage::Compute);
        if (!sm) return false;
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(vk_.device, &sci, nullptr, &mvSampler_));
        VkDescriptorSetLayoutBinding b[2]{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dl.bindingCount = 2;
        dl.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(vk_.device, &dl, nullptr, &mvSetLayout_));
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MotionPush)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &mvSetLayout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pr;
        VK_CHECK(vkCreatePipelineLayout(vk_.device, &pl, nullptr, &mvLayout_));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm;
        ci.stage.pName = "main";
        ci.layout = mvLayout_;
        return vkCreateComputePipelines(vk_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &mvPipe_) == VK_SUCCESS;
    }

    // An image of DLSS's own, (re)made at this size and format.
    bool dlssImage(Image& img, VkFormat format, uint32_t w, uint32_t h) {
        if (img.image && img.width == w && img.height == h && img.format == format) return true;
        destroyImage(img);
        return createImage(img, format, w, h,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           false);
    }

    // A draw of one of the SMAA shaders with DLSS on: true when it was handled (skipped, or the
    // blend replaced by DLSS).
    bool dlssSmaa(const DrawRegs& r, uint8_t role) {
        if (!dlss::usable()) return false;
        smaaSeen_ = true; // later draws are post-processing: not jittered
        if (role != 3) return true; // edges and weights: DLSS needs neither
        const uint32_t* cb = &r.ctx[reg::kCbColorBase];
        const uint32_t info = cb[::gpu::ctx::kCbColorInfo];
        const uint32_t fmt = (info >> 2) & 0x1F, num = (info >> 8) & 7;
        const VkFormat vf = colorFormat(fmt, num);
        const uint32_t zInfo = r.ctx[reg::kDbZInfo];
        if (vf == VK_FORMAT_UNDEFINED || !(zInfo & 3)) return false;
        const uint32_t pitch = ((cb[::gpu::ctx::kCbColorPitch] & 0x7FF) + 1) * 8;
        const uint32_t slice = ((cb[::gpu::ctx::kCbColorSlice] & 0x3FFFFF) + 1) * 64;
        Surface* out = colorSurface(uint64_t(cb[0]) << 8, pitch, slice / pitch, vf, bytesPerElement(fmt));
        const bool stencil = r.ctx[reg::kDbStencilInfo] & 1;
        const VkFormat df = stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : (zInfo & 3) == 1 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
        const uint32_t size = r.ctx[reg::kDbDepthSize];
        const uint64_t depthAddr = uint64_t(r.ctx[reg::kDbZReadBase]) << 8;
        Surface* depth = depthSurface(depthAddr, ((size & 0x7FF) + 1) * 8, (((size >> 11) & 0x7FF) + 1) * 8, df);
        if (!out || !depth) return false;
        sceneDepth_ = depthAddr;
        // The scene's size: the blend's full-screen viewport.
        const float* v = reinterpret_cast<const float*>(&r.ctx[reg::kVportXScale]);
        const uint32_t w = std::min(uint32_t(std::lround(std::fabs(v[0]) * 2)), std::min(out->img.width, depth->img.width));
        const uint32_t h = std::min(uint32_t(std::lround(std::fabs(v[2]) * 2)), std::min(out->img.height, depth->img.height));
        if (!w || !h) return false;
        if (!ringFits(1 << 16)) flush();
        begin();
        if (!dlssImage(dlssColor_, vf, out->img.width, out->img.height) ||
            !dlssImage(dlssMotion_, VK_FORMAT_R16G16_SFLOAT, w, h) || !motionPipeline())
            return false;
        fullBarrier();

        // The scene colour, before the blend: DLSS's input (it cannot write where it reads).
        const VkImageCopy copy{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
        rec([c = R().cmd_, src = out->img.image, dst = dlssColor_.image, copy] {
            vkCmdCopyImage(c, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        });

        // Motion vectors from the camera, or none without both frames' cameras or across a cut.
        MotionPush mp{};
        double m[16];
        const bool moved = cam_.valid && prevCam_.valid && dlssRanLast_ &&
                           std::hypot(cam_.pos[0] - prevCam_.pos[0], cam_.pos[1] - prevCam_.pos[1], cam_.pos[2] - prevCam_.pos[2]) < 10 &&
                           reprojection(cam_, prevCam_, m);
        const Camera& k = cam_.valid ? cam_ : Camera{};
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col) mp.m[col * 4 + row] = moved ? float(m[row * 4 + col]) : float(row == col);
        const float vp[4] = {k.valid ? k.xo : float(w) / 2, k.valid ? k.yo : float(h) / 2, k.valid ? k.xs : float(w) / 2,
                             k.valid ? k.ys : -float(h) / 2};
        std::memcpy(mp.vp, vp, sizeof vp);
        mp.z[0] = k.valid ? k.zo : 0, mp.z[1] = k.valid ? k.zs : 1;
        mp.z[2] = jitterOn_ ? jitterX_ : 0, mp.z[3] = jitterOn_ ? jitterY_ : 0;
        mp.size[0] = int32_t(w), mp.size[1] = int32_t(h);
        const VkImageView depthView = view(depth->img, df, VK_IMAGE_VIEW_TYPE_2D, kIdentity, VK_IMAGE_ASPECT_DEPTH_BIT);
        const VkImageView mvView = view(dlssMotion_, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D, kIdentity);
        rec([this, c = R().cmd_, mp, depthView, mvView] {
            const VkDescriptorImageInfo di[2] = {{mvSampler_, depthView, VK_IMAGE_LAYOUT_GENERAL},
                                                 {VK_NULL_HANDLE, mvView, VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet wr[2]{};
            for (uint32_t i = 0; i < 2; ++i) {
                wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstBinding = i;
                wr[i].descriptorCount = 1;
                wr[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wr[i].pImageInfo = &di[i];
            }
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, mvPipe_);
            vk_.cmdPushDescriptorSet(c, VK_PIPELINE_BIND_POINT_COMPUTE, mvLayout_, 0, 2, wr);
            vkCmdPushConstants(c, mvLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof mp, &mp);
            vkCmdDispatch(c, (uint32_t(mp.size[0]) + 7) / 8, (uint32_t(mp.size[1]) + 7) / 8, 1);
        });
        barrier();

        // DLSS: the jitter is how far the scene was moved on screen (as engines that jitter the
        // projection pass it; y down).
        dlss::Frame f;
        f.color = {dlssColor_.image, view(dlssColor_, vf, VK_IMAGE_VIEW_TYPE_2D, kIdentity), vf, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
        f.depth = {depth->img.image, depthView, df, VK_IMAGE_ASPECT_DEPTH_BIT, w, h};
        f.motion = {dlssMotion_.image, mvView, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
        f.output = {out->img.image, view(out->img, vf, VK_IMAGE_VIEW_TYPE_2D, kIdentity), vf, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
        f.renderW = f.outW = w;
        f.renderH = f.outH = h;
        f.jitterX = jitterOn_ ? jitterX_ * jitterSign_[0] : 0;
        f.jitterY = jitterOn_ ? jitterY_ * jitterSign_[1] : 0;
        f.reset = !moved;
        // Every 300 frames: how often the history was dropped, to tell a camera problem.
        ++dlssStats_[0], dlssStats_[1] += f.reset, dlssStats_[2] += cam_.valid, dlssStats_[3] += jitterOn_;
        if (dlssStats_[0] == 300) {
            std::fprintf(stderr, "[dlss] last 300 frames: history reset %u, camera read %u, jittered %u\n", dlssStats_[1],
                         dlssStats_[2], dlssStats_[3]);
            std::memset(dlssStats_, 0, sizeof dlssStats_);
        }
        f.mode = 1; // DLAA for now: the render size stays the game's
        rec([c = R().cmd_, dev = vk_.device, f] { dlss::evaluate(c, dev, f); });
        fullBarrier();
        // DLSS bound pipelines and descriptors of its own: everything is set again.
        R().bound_ = Bound{};
        R().fast_.valid = false;
        ++fastEpoch_;
        gpuWrote(*out);
        dlssRan_ = true;
        return true;
    }

    // Each frame: whether DLSS runs, the cameras and the next jitter.
    void dlssFrameEnd() {
        prevCam_ = cam_;
        cam_.valid = false;
        smaaSeen_ = false;
        camTries_ = 0;
        dlssRanLast_ = dlssRan_;
        dlssRan_ = false;
        dlssOn_ = displaySettings().dlss != DlssMode::Off && dlss::usable();
        // Jitter only while DLSS ran last frame: a frame without SMAA (menus) stops it.
        jitterOn_ = dlssOn_ && dlssRanLast_ && sceneDepth_ && !getenv("LPN_DLSS_NO_JITTER");
        if (jitterOn_) dlss::jitter(++jitterIndex_, 1, 1, &jitterX_, &jitterY_);
    }

    bool dlssOn_ = false, smaaSeen_ = false, dlssRan_ = false, dlssRanLast_ = false, jitterOn_ = false, camWarned_ = false;
    uint64_t sceneDepth_ = 0, jitterIndex_ = 0;
    uint32_t camTries_ = 0, dlssStats_[4] = {};
    float jitterX_ = 0, jitterY_ = 0;
    // LPN_DLSS_JITTER_SIGN=<x>,<y> (each 1 or -1): flips the jitter DLSS is told (debugging).
    float jitterSign_[2] = {1, 1};
    Camera cam_, prevCam_;
    Image dlssColor_, dlssMotion_;
    VkPipeline mvPipe_ = VK_NULL_HANDLE;
    VkPipelineLayout mvLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout mvSetLayout_ = VK_NULL_HANDLE;
    VkSampler mvSampler_ = VK_NULL_HANDLE;

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
        VkDeviceMemory mem = VK_NULL_HANDLE; // the arena block it is bound in
        VkDeviceSize memOffset = 0, memSize = 0;
        uint64_t sample = 0, hash = 0;
        uint64_t checkedFrame = ~0ull, changedFrame = 0, lastUse = 0, hostUntil = 0;
        uint32_t uses = 0, phase = 0, changes = 0;
        bool gpuWritten = false, dirty = false;
        std::shared_ptr<AsyncHash> check;
        // Unchanged for a while and checked once a frame by the sweep (see sweepVram) instead of
        // at its first use: analysis workers may then resolve it ahead (see resolveAhead).
        std::atomic<bool> settled{false};
        std::atomic<uint64_t> aheadUse{0}; // the frame a worker last resolved a draw to it in (for evictVram)
    };
    // A buffer binding resolved ahead on an analysis worker, valid while bufferEpoch_ is the
    // one it was resolved at. info.buffer null: resolved by the recorder as before.
    struct BufRes {
        VkDescriptorBufferInfo info{};
        uint32_t words[4] = {};
        VramBuffer* v = nullptr;
    };
    // Bumped (by the recorder, under cacheMutex_ when a lookup structure changes) whenever a
    // resolution done ahead may have gone stale: a VRAM copy made, freed, written or found changed.
    std::atomic<uint64_t> bufferEpoch_{1};
    std::atomic<uint64_t> frameNow_{0}; // frame_, for the workers
    std::shared_mutex cacheMutex_; // vram_, imports_ and VRAM copies' buffers, for the workers' reads
    const bool resolveAhead_ = getenv("LPN_NO_RESOLVE_AHEAD") == nullptr && getenv("LPN_RACE_CHECK") == nullptr;
    void unsettle(VramBuffer& v) {
        if (v.settled.exchange(false, std::memory_order_relaxed)) bufferEpoch_.fetch_add(1, std::memory_order_release);
    }
    struct PairHash {
        size_t operator()(const std::pair<uint64_t, uint64_t>& k) const { return std::hash<uint64_t>()(k.first * 0x9E3779B97F4A7C15ull ^ k.second); }
    };
    static constexpr uint64_t kVramMin = 64 << 10, kVramMax = 64ull << 20, kVramBudget = 4096ull << 20;
    // Copies share 128 MB blocks of device memory (VramArena), so their count is not held to the
    // driver's allocation limit. At 1,500 (one allocation each) the overlook's ~1,500 live copies
    // churned: 10-65 evicted, allocated and copied again a frame, the worst frames at 37-40 fps.
    static constexpr size_t kVramEntries = 6000;
    // Sub-allocates VRAM copies from large blocks: first fit over each block's free ranges,
    // merged again on free. Freed from the completion thread, hence the mutex.
    class VramArena {
    public:
        static constexpr VkDeviceSize kBlock = 128ull << 20;
        bool alloc(Vk& vk, const VkMemoryRequirements& req, VkDeviceMemory& mem, VkDeviceSize& off) {
            std::lock_guard lock(m_);
            for (int pass = 0; pass < 2; ++pass) {
                for (Block& b : blocks_) {
                    if (!((req.memoryTypeBits >> b.type) & 1)) continue;
                    for (auto it = b.free.begin(); it != b.free.end(); ++it) {
                        const VkDeviceSize start = (it->first + req.alignment - 1) / req.alignment * req.alignment;
                        const VkDeviceSize end = it->first + it->second;
                        if (start + req.size > end) continue;
                        const VkDeviceSize lo = it->first;
                        b.free.erase(it);
                        if (start > lo) b.free.emplace(lo, start - lo);
                        if (start + req.size < end) b.free.emplace(start + req.size, end - start - req.size);
                        mem = b.mem, off = start;
                        return true;
                    }
                }
                if (pass) break;
                VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                mai.allocationSize = std::max(kBlock, req.size);
                mai.memoryTypeIndex = vk.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                Block b;
                if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk.device, &mai, nullptr, &b.mem) != VK_SUCCESS)
                    return false;
                b.type = mai.memoryTypeIndex;
                b.free.emplace(0, mai.allocationSize);
                blocks_.push_back(std::move(b));
            }
            return false;
        }
        void free(VkDeviceMemory mem, VkDeviceSize off, VkDeviceSize size) {
            std::lock_guard lock(m_);
            for (Block& b : blocks_) {
                if (b.mem != mem) continue;
                auto [it, ok] = b.free.emplace(off, size);
                if (!ok) return;
                if (auto next = std::next(it); next != b.free.end() && it->first + it->second == next->first) {
                    it->second += next->second;
                    b.free.erase(next);
                }
                if (it != b.free.begin()) {
                    auto prev = std::prev(it);
                    if (prev->first + prev->second == it->first) {
                        prev->second += it->second;
                        b.free.erase(it);
                    }
                }
                return;
            }
        }

    private:
        struct Block {
            VkDeviceMemory mem = VK_NULL_HANDLE;
            uint32_t type = 0;
            std::map<VkDeviceSize, VkDeviceSize> free; // offset -> size
        };
        std::mutex m_;
        std::vector<Block> blocks_;
    };
    VramArena vramArena_;
    uint64_t evictNothingFrame_ = ~0ull; // evictVram found no copy to free in this frame
    const bool vramOn_ = getenv("LPN_NO_VRAM_BUFFERS") == nullptr;
    std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<VramBuffer>, PairHash> vram_;
    struct VramHit {
        uint64_t base = 0, size = 0;
        VramBuffer* v = nullptr;
    };
    static constexpr size_t kVramHits = 512;
    static size_t vramHitIndex(uint64_t base, uint64_t size) { return (base / 64 ^ size * 0x9E3779B97F4A7C15ull >> 40) % kVramHits; }
    std::multimap<uint64_t, VramBuffer*> vramByBase_; // for overlap queries
    std::map<uint64_t, uint64_t> gpuWrittenRanges_;  // [start, end), merged
    uint64_t vramBytes_ = 0, vramCount_ = 0;
    std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t, PairHash> vramCandidates_; // first seen in frame
    std::mutex candMutex_; // vramCandidates_ among recorder threads holding m_ shared

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
        if (sharedMiss()) return;
        uint64_t lo = addr, hi = end;
        if (it != gpuWrittenRanges_.begin() && std::prev(it)->second >= lo) --it;
        while (it != gpuWrittenRanges_.end() && it->first <= hi) {
            lo = std::min(lo, it->first);
            hi = std::max(hi, it->second);
            it = gpuWrittenRanges_.erase(it);
        }
        gpuWrittenRanges_[lo] = hi;
        forVramOverlapping(addr, end, [&](VramBuffer& v) {
            v.gpuWritten = true;
            unsettle(v);
        });
    }
    bool gpuWritten(uint64_t addr, uint64_t size) const {
        auto it = gpuWrittenRanges_.upper_bound(addr + size - 1);
        return it != gpuWrittenRanges_.begin() && std::prev(it)->second > addr;
    }
    void cpWrites(uint64_t addr, uint64_t size) {
        if (vramOn_)
            forVramOverlapping(addr, addr + size, [&](VramBuffer& v) {
                v.dirty = true;
                unsettle(v);
            });
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
    // inPass: only a copy already checked this frame, as nothing can be recorded (fastDraw).
    bool vramBuffer(uint64_t base, uint64_t size, VkBuffer& out, VkDeviceSize& off, bool inPass = false) {
        if (!vramOn_ || size < kVramMin || size > kVramMax || base % 4 || size % 4) return false;
        // The copy last found for this buffer, checked before the hash map (a miss per lookup).
        VramHit& hit = R().vramHits_[vramHitIndex(base, size)];
        VramBuffer* vp = nullptr;
        if (hit.v && hit.base == base && hit.size == size) {
            vp = hit.v;
        } else if (auto found = vram_.find({base, size}); found != vram_.end()) {
            vp = found->second.get();
            hit = {base, size, vp};
        }
        if (R().lockShared_ && !inPass) { // a full draw on a recorder thread: changes nothing shared
            if (!vp) {
                if (gpuWritten(base, size)) return false;
                std::lock_guard lock(candMutex_);
                auto [cand, fresh] = vramCandidates_.try_emplace({base, size}, frame_);
                if (fresh || cand->second == frame_) return false;
                return !sharedMiss(); // a copy to make
            }
            VramBuffer& v = *vp;
            if (!v.size) return !sharedMiss();
            if (v.gpuWritten || frame_ < v.hostUntil) return false;
            const bool settled = v.settled.load(std::memory_order_relaxed);
            const bool skip = frame_ - v.changedFrame >= 600 && (frame_ + v.phase) % 4 != 0; // as below
            if (!v.buf || v.dirty || (v.checkedFrame != frame_ && !settled && !skip)) return !sharedMiss();
            if (v.lastUse != frame_) std::atomic_ref(v.lastUse).store(frame_, std::memory_order_relaxed);
            out = v.buf;
            off = 0;
            return true;
        }
        if (inPass) {
            if (!vp) return false;
            const VramBuffer& v = *vp;
            if (!v.size || !v.buf || v.gpuWritten || frame_ < v.hostUntil || v.dirty ||
                (v.checkedFrame != frame_ && !v.settled.load(std::memory_order_relaxed)))
                return false;
            out = v.buf;
            off = 0;
            return true;
        }
        if (!vp) {
            // Only a buffer used again in a later frame gets a copy: per-frame data in rings sits at
            // a new address each frame, and copying it would cost more than it saves.
            if (gpuWritten(base, size)) return false;
            auto [cand, fresh] = vramCandidates_.try_emplace({base, size}, frame_);
            if (fresh || cand->second == frame_) return false;
            vramCandidates_.erase(cand);
            VkBuffer imported;
            VkDeviceSize at;
            if (!guestBuffer(base, size, imported, at)) return false; // mapped (importedSample reads it directly)
            std::unique_lock lock(cacheMutex_);
            if (vram_.size() >= kVramEntries) evictVram();
            vp = vram_.emplace(std::make_pair(base, size), std::make_unique<VramBuffer>()).first->second.get();
            hit = {base, size, vp};
        }
        VramBuffer& v = *vp;
        if (!v.size) {
            v.base = base, v.size = size, v.phase = uint32_t(vramCount_++ % 8);
            v.gpuWritten = gpuWritten(base, size);
            vramByBase_.emplace(base, vp);
        }
        if (v.gpuWritten || frame_ < v.hostUntil) return false;
        v.lastUse = frame_;
        // Settled: the sweep checks it (sweepVram). Otherwise, unchanged for 10 s: fingerprinted
        // every 4th frame only (as textures are on every other use). The fingerprints' reads of
        // host memory were ~10% of the recorder in the city.
        if (v.settled.load(std::memory_order_relaxed) && !v.dirty) v.checkedFrame = frame_;
        if (v.checkedFrame != frame_ && !v.dirty && v.buf && frame_ - v.changedFrame >= 600 && (frame_ + v.phase) % 4 != 0)
            v.checkedFrame = frame_;
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
                if (!v.buf) {
                    std::unique_lock lock(cacheMutex_);
                    if (!allocVram(v)) return false;
                }
                if (!changed) v.changedFrame = frame_; // the first copy
                v.hash = hasher_.hash(base, size); // what the copy holds (the game does not write memory the GPU is about to read)
                if (v.check) v.check->state.store(0, std::memory_order_relaxed); // a result in flight is about the old contents
                {
                    Prologue pg(*this); // on a recorder thread: ahead of the segments
                    begin();
                    fullBarrier();
                    const VkBufferCopy bc{soff, 0, size};
                    rec([c = R().cmd_, src, dst = v.buf, bc] { vkCmdCopyBuffer(c, src, dst, 1, &bc); });
                }
                ++R().prof_.vramCopies, R().prof_.vramCopyBytes += size;
            } else if (resolveAhead_ && frame_ - v.changedFrame >= kSettleFrames && !v.gpuWritten) {
                v.settled.store(true, std::memory_order_release); // checked and unchanged for a while
            }
        }
        out = v.buf;
        off = 0;
        return true;
    }
    static constexpr uint64_t kSettleFrames = 60;

    // Once a frame, the settled copies' and textures' fingerprints are taken on a thread of their
    // own, as their first use in a frame took them before; a copy that changed is copied again at
    // its next use, a texture uploaded again. Their full hashes run on the hasher threads every
    // 8th frame, as on every 8th use before.
    struct SweepItem {
        uint64_t base, size, sample;
        bool changed;
        GuestTexture* tex; // a texture's (sampleHash) rather than a VRAM copy's (importedSample)
    };
    std::mutex sweepMutex_;
    std::condition_variable sweepCv_;
    std::vector<SweepItem> sweep_;
    bool sweepQueued_ = false, sweepDone_ = true, sweepStarted_ = false;
    // A settled copy's or texture's full hash: every 8th frame, and every 32nd (LPN_HASH_FRAMES)
    // once unchanged for 10 s. Every 8th frame hashed all of them (GBs at the overlook), which kept
    // three hasher threads busy; the fingerprints taken every frame still catch a whole rewrite.
    bool hashDue(uint64_t changedFrame, uint32_t phase, uint64_t base) const {
        if (frame_ - changedFrame < 600) return (frame_ + phase) % 8 == 0;
        static const uint64_t every = std::max(8, getenv("LPN_HASH_FRAMES") ? std::atoi(getenv("LPN_HASH_FRAMES")) : 32);
        return (frame_ + (base >> 8) * 0x9E3779B1ull % every) % every == 0;
    }
    void sweepVram() {
        if (!resolveAhead_) return;
        {
            std::lock_guard lock(sweepMutex_);
            if (!sweepDone_) return; // the last one is still running: next frame
            for (const SweepItem& it : sweep_) {
                if (!it.changed) continue;
                if (it.tex) {
                    if (it.tex->settled && it.tex->sample == it.sample) unsettleTexture(*it.tex);
                    continue;
                }
                auto f = vram_.find({it.base, it.size});
                if (f == vram_.end() || f->second->sample != it.sample) continue; // gone, or checked since
                f->second->dirty = true;
                unsettle(*f->second);
            }
            sweep_.clear();
            for (auto& [k, up] : vram_) {
                VramBuffer& v = *up;
                if (!v.settled.load(std::memory_order_relaxed)) continue;
                if (!v.check) v.check = std::make_shared<AsyncHash>();
                const int st = v.check->state.load(std::memory_order_acquire);
                if (st == 2) {
                    v.check->state.store(0, std::memory_order_relaxed);
                    if (v.check->result != v.hash) {
                        v.dirty = true;
                        unsettle(v);
                        continue;
                    }
                } else if (st == 0 && hashDue(v.changedFrame, v.phase, v.base)) {
                    queueTexHash(v.check, v.base, v.size);
                }
                sweep_.push_back({v.base, v.size, v.sample, false, nullptr});
            }
            auto drop = [](GuestTexture* t) { return !(t->swept = false); }; // out of the list
            std::erase_if(sweptTextures_, [&](GuestTexture* t) {
                if (!t->settled) return drop(t);
                if (!t->check) t->check = std::make_shared<AsyncHash>();
                const int st = t->check->state.load(std::memory_order_acquire);
                if (st == 2) {
                    t->check->state.store(0, std::memory_order_relaxed);
                    if (t->check->result != t->hash) {
                        unsettleTexture(*t);
                        return drop(t);
                    }
                } else if (st == 0 && hashDue(t->changedFrame, t->phase, t->base)) {
                    queueTexHash(t->check, t->base, t->tiled);
                }
                sweep_.push_back({t->base, t->span, t->sample, false, t});
                return false;
            });
            if (sweep_.empty()) return;
            sweepDone_ = false;
            sweepQueued_ = true;
            if (!sweepStarted_) {
                sweepStarted_ = true;
                std::thread([this] { sweepThread(); }).detach();
            }
        }
        sweepCv_.notify_one();
    }
    void sweepThread() {
        profile::nameThread("lpn-sweep");
        std::vector<SweepItem> items;
        for (;;) {
            {
                std::unique_lock lock(sweepMutex_);
                sweepCv_.wait(lock, [&] { return sweepQueued_; });
                sweepQueued_ = false;
                items.swap(sweep_);
            }
            for (SweepItem& it : items)
                it.changed = (it.tex ? sampleHash(it.base, it.size) : importedSample(it.base, it.size)) != it.sample;
            {
                std::lock_guard lock(sweepMutex_);
                items.swap(sweep_);
                sweepDone_ = true;
            }
        }
    }
    bool allocVram(VramBuffer& v) { // cacheMutex_ held
        // A quarter of the largest device-local heap, at most kVramBudget.
        static const uint64_t budget = [&] {
            VkDeviceSize heap = 0;
            for (uint32_t i = 0; i < vk_.mem.memoryHeapCount; ++i)
                if (vk_.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) heap = std::max(heap, vk_.mem.memoryHeaps[i].size);
            return std::min<uint64_t>(kVramBudget, heap / 4);
        }();
        if (vramBytes_ + v.size > budget) evictVram();
        if (vramBytes_ + v.size > budget) return false;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = v.size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(vk_.device, &bci, nullptr, &v.buf) != VK_SUCCESS) return v.buf = VK_NULL_HANDLE, false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(vk_.device, v.buf, &req);
        if (!vramArena_.alloc(vk_, req, v.mem, v.memOffset)) {
            vkDestroyBuffer(vk_.device, v.buf, nullptr);
            v.buf = VK_NULL_HANDLE;
            return false;
        }
        v.memSize = req.size;
        VK_CHECK(vkBindBufferMemory(vk_.device, v.buf, v.mem, v.memOffset));
        vramBytes_ += v.size;
        ++R().prof_.vramAllocs;
        return true;
    }
    // Frees the copies used longest ago (a quarter of them), once the GPU is done with them.
    void evictVram() { // cacheMutex_ held
        // Only copies unused for 2 frames go: when none were, none will be until the next frame,
        // and scanning and sorting every copy again for each new one cost the recorder at the overlook.
        if (evictNothingFrame_ == frame_) return;
        ++R().prof_.vramEvicts;
        std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> byUse;
        for (auto& [k, v] : vram_)
            if (std::max(v->lastUse, v->aheadUse.load(std::memory_order_relaxed)) + 2 < frame_)
                byUse.push_back({std::max(v->lastUse, v->aheadUse.load(std::memory_order_relaxed)), k});
        if (byUse.empty()) {
            evictNothingFrame_ = frame_;
            ++R().prof_.vramEvictsEmpty;
            return;
        }
        bufferEpoch_.fetch_add(1, std::memory_order_release); // resolutions ahead may name a freed copy
        R().prof_.vramFreed += std::max<size_t>(byUse.size() / 4, std::min<size_t>(byUse.size(), 16));
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
                R().afterFlush_.push_back([this, buf = v->buf, mem = v->mem, off = v->memOffset, sz = v->memSize] {
                    vkDestroyBuffer(vk_.device, buf, nullptr);
                    vramArena_.free(mem, off, sz);
                });
            }
            vram_.erase(it);
        }
        rec0_.vramHits_.fill({}); // every thread's (m_ held exclusively: none is looking)
        pro_.vramHits_.fill({});
        for (auto& w : workerRecs_) w->vramHits_.fill({});
    }

    void bindResources(gcn::Program& p, uint32_t base, VkShaderStageFlags, std::vector<VkWriteDescriptorSet>& writes,
                       std::vector<VkDescriptorBufferInfo>& bufInfos, std::vector<VkDescriptorImageInfo>& imgInfos,
                       const std::vector<BufRes>* pre = nullptr) {
        ProfScope pb(R().prof_, kProfBind);
        uint32_t n = base + 1;
        for (size_t i = 0; i < p.buffers.size(); ++i) {
            const gcn::BufferResource& b = p.buffers[i];
            bufInfos.push_back(bindBuffer(p, b, false, pre ? &(*pre)[i] : nullptr));
            writes.push_back(kBufferWrite); // copied from a constant: built on the stack it stalled on store forwarding
            writes.back().dstBinding = n++;
        }
        for (auto& im : p.images) {
            // An image the analysis could not resolve (in code this launch skips) is bound to a
            // placeholder of the dimension the translation declared for it.
            gcn::ImageResource unresolved;
            const bool zero = !im.t[0] && !im.t[1] && !im.t[3];
            if (zero) unresolved = im, unresolved.t[3] = gcn::imageDimension(p, im) << 28;
            const ImageBinding ib = resolveImage(zero ? unresolved : im);
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
        const VkDeviceSize at = uploadData(p.data);
        bufInfos.push_back({R().ring_, at, bytes});
        writes.push_back(kBufferWrite);
        writes.back().dstBinding = base;
    }
    // Resolves buffer `b` of `p` and fills its words in the draw data. inPass: nothing may be recorded.
    uint64_t bindSize(const gcn::BufferResource& b) const {
        const uint64_t size = b.stride() ? uint64_t(b.stride()) * b.numRecords() : b.numRecords();
        return std::min<uint64_t>(size ? size : 16, std::min<uint64_t>(vk_.props.limits.maxStorageBufferRange, 64ull << 20));
    }
    // pre: its resolution on a worker (resolveAhead), if still good.
    VkDescriptorBufferInfo bindBuffer(gcn::Program& p, const gcn::BufferResource& b, bool inPass = false, const BufRes* pre = nullptr) {
        if (pre && pre->info.buffer) {
            std::memcpy(&p.data[b.data], pre->words, sizeof pre->words);
            ++R().prof_.resolvedAhead;
            return pre->info;
        }
        const uint64_t size = bindSize(b);
        BufferRange r;
        VkDeviceSize off = 0;
        if (b.store && b.known) gpuWrites(b.base(), size);
        if (b.known && ((!b.store && vramBuffer(b.base(), size, r.buffer, off, inPass)) || guestBuffer(b.base(), size, r.buffer, off))) {
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
        return {r.buffer, r.offset, r.range};
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
            if (gcn::genericTranslation()) { // the pixel shader's inputs, which it exports to (gcn::linkageData)
                std::memcpy(l.psInputCntl, &r.ctx[reg::kSpiPsInputCntl], sizeof(l.psInputCntl));
                l.numInterp = r.ctx[reg::kSpiPsInControl] & 0x3F;
            }
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
        // Worked out off the recorder, which compared these itself for every draw (fastDraw):
        uint64_t seq = 0;     // the draw's place in submission order
        uint64_t ctxHash = 0; // of its context registers (the analysis thread)
        // Its resources against the previous draw's (an analysis worker, see compareResources),
        // when that draw was in the same batch.
        struct Diff {
            int why = 0;
            uint32_t moved = 0, images = 0;
            int8_t data = -1; // 1: its draw data is the previous draw's but for resource words, 0: it is not
        } vsDiff, psDiff;
        bool diffPrev = false;
        // Its buffers' bindings, resolved on the worker (resolveAhead) at bufferEpoch_ resEpoch.
        std::vector<BufRes> vsRes, psRes;
        uint64_t resEpoch = 0;
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
    // On a worker: the bindings bindBuffer would make for the draw's buffers, where they depend
    // only on lookups that stay put while bufferEpoch_ does: a settled VRAM copy, or an import
    // for a buffer too small for one. The recorder takes them if the epoch has not moved since.
    // Those lookups (cache misses one after another) were the largest part of its time.
    // LPN_NO_RESOLVE_AHEAD=1: all on the recorder as before.
    void resolveAhead(DrawAnalysis& a) {
        std::shared_lock lock(cacheMutex_);
        a.resEpoch = bufferEpoch_.load(std::memory_order_acquire);
        resolveBuffers(a.vs, a.vsRes);
        if (a.psm) resolveBuffers(a.ps, a.psRes);
        else a.psRes.clear();
    }
    void resolveBuffers(const gcn::Program& p, std::vector<BufRes>& out) {
        out.resize(p.buffers.size());
        for (size_t i = 0; i < p.buffers.size(); ++i) {
            const gcn::BufferResource& b = p.buffers[i];
            BufRes& r = out[i];
            r = {};
            if (!b.known || b.store) continue;
            const uint64_t base = b.base(), size = bindSize(b);
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize off = 0;
            if (vramOn_ && size >= kVramMin && size <= kVramMax && base % 4 == 0 && size % 4 == 0) {
                auto it = vram_.find({base, size});
                if (it == vram_.end()) continue; // the recorder decides on a copy
                VramBuffer* v = it->second.get();
                if (!v->settled.load(std::memory_order_acquire) || !v->buf) continue;
                buf = v->buf;
                r.v = v;
                // Kept from eviction here: the recorder storing it per draw took a miss each.
                const uint64_t now = frameNow_.load(std::memory_order_relaxed);
                if (v->aheadUse.load(std::memory_order_relaxed) != now) v->aheadUse.store(now, std::memory_order_relaxed);
            } else if (!importFor(base, size, buf, off)) {
                continue;
            }
            const VkDeviceSize offset = off & ~(storageAlign_ - 1);
            const uint32_t byteOffset = uint32_t(off - offset);
            const VkDeviceSize range = std::min<VkDeviceSize>(size + byteOffset, vk_.props.limits.maxStorageBufferRange);
            r.info = {buf, offset, range};
            r.words[0] = byteOffset;
            r.words[1] = b.stride();
            r.words[2] = b.numRecords();
            r.words[3] = uint32_t(range);
        }
    }
    // guestBuffer's lookup of an existing import, without making one (cacheMutex_ held shared).
    bool importFor(uint64_t addr, uint64_t size, VkBuffer& buffer, VkDeviceSize& offset) const {
        if (!addr || addr >= (1ull << 40) || !size) return false;
        // The closest import below usually covers it; a larger one (the import thread's) may start
        // further down, behind small ones a draw made first.
        for (auto it = imports_.upper_bound(addr); it != imports_.begin();) {
            --it;
            if (it->second.end >= addr + size) {
                buffer = it->second.buffer;
                offset = addr - it->first;
                return true;
            }
            if (addr - it->first > kAheadChunk) break;
        }
        return false;
    }
    // A worker's resolution of `p`'s buffers, if still good (see resolveAhead).
    const std::vector<BufRes>* aheadFor(const DrawAnalysis& a, const gcn::Program& p) const {
        if (!a.resEpoch || a.resEpoch != bufferEpoch_.load(std::memory_order_relaxed)) return nullptr;
        const std::vector<BufRes>& r = &p == &a.vs ? a.vsRes : a.psRes;
        return r.size() == p.buffers.size() ? &r : nullptr;
    }

    void runBatch(AnalysisCtx& c, std::vector<DrawJob>& b) {
        for (DrawJob& j : b) runJob(c, j);
        for (size_t i = 1; i < b.size(); ++i) {
            DrawAnalysis &a = *b[i].out, &p = *b[i - 1].out;
            if (b[i].kind != DrawJob::kDraw || b[i - 1].kind != DrawJob::kDraw || !a.ok || !p.ok || a.vsm != p.vsm ||
                a.psm != p.psm)
                continue;
            a.vsDiff.why = compareResources(a.vs, p.vs, a.vsDiff.moved, &a.vsDiff.images);
            a.vsDiff.data = a.vsDiff.why < 0 ? int8_t(dataAlike(a.vs, p.vs)) : -1;
            if (a.psm) {
                a.psDiff.why = compareResources(a.ps, p.ps, a.psDiff.moved, &a.psDiff.images);
                a.psDiff.data = a.psDiff.why < 0 ? int8_t(dataAlike(a.ps, p.ps)) : -1;
            } else {
                a.psDiff = {-1, 0, 0, 1};
            }
            a.diffPrev = true;
        }
        if (resolveAhead_)
            for (DrawJob& j : b)
                if (j.kind == DrawJob::kDraw && j.out->ok) resolveAhead(*j.out);
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
        notePipeline(gs, a.vsm->name, a.psm ? a.psm->name : std::string(), c.prof.on, "ahead");
        const Layout& lay = layout(lb);
        PipelineJob job{gs, a.vsm->module, a.psm ? a.psm->module : VK_NULL_HANDLE, geometryFor(gs), lay.pipeline,
                        std::move(b), std::move(record), false};
        job.lay = &lay;
        job.vsName = a.vsm->name;
        job.psName = a.psm ? a.psm->name : std::string();
        queueJob(std::move(job), true);
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
        rec([c = R().cmd_, pipe] { vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); });
        pushDescriptors(VK_PIPELINE_BIND_POINT_COMPUTE, lay, writes, bufs, imgs);
        if (trace_) captureProgram(cs, m, "cs");
        if (trace_)
            for (const auto& b : cs.buffers) {
                uint32_t first = 0;
                readGuest(b.base(), &first, 4);
                std::fprintf(traceOut_, "[trace]   buffer 0x%" PRIx64 " stride %u records %u first 0x%x%s\n", b.base(),
                             b.stride(), b.numRecords(), first, b.store ? " (written)" : "");
            }
        rec([c = R().cmd_, x = d.groups[0], y = d.groups[1], z = d.groups[2]] { vkCmdDispatch(c, x, y, z); });
        ++R().stats_.dispatches;
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
        rec([c = R().cmd_, buffer, offset, bytes = records * 16, v = value[0]] { vkCmdFillBuffer(c, buffer, offset, bytes, v); });
        if (raceCheck_) noteWrite(b.base(), records * 16);
        gpuWrites(b.base(), records * 16);
        ++R().stats_.dispatches;
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
        if (R().lockShared_) {
            auto it = metadata_.find(meta);
            if (it != metadata_.end() && it->second.surface == surface && it->second.depth == depth &&
                it->second.clear[0] == c0 && it->second.clear[1] == c1)
                return;
            sharedMiss();
            return;
        }
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
            rec([c = R().cmd_, im = img.image, v, range] { vkCmdClearDepthStencilImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range); });
        } else {
            VkClearColorValue v;
            if (!clearColor(img.format, m.clear, v)) {
                warnOnce("fast clear of a render target in format " + std::to_string(img.format));
                return;
            }
            rec([c = R().cmd_, im = img.image, v, range] { vkCmdClearColorImage(c, im, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range); });
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
        s.writtenBack = true;
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
        rec([cb = R().cmd_, im = s.img.image, buffer, c] { vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &c); });
        if (trace_)
            std::fprintf(traceOut_, "[trace]   copied %ux%u back to 0x%" PRIx64 "\n", w, h, t.base());
    }

    void draw(const DrawCall& d, const DrawRegs& r, DrawAnalysis& a) {
        const uint32_t cbMode = (r.ctx[reg::kCbColorControl] >> 4) & 7;
        const uint32_t dbRender = r.ctx[reg::kDbRenderControl];
        if (trace_)
            std::fprintf(traceOut_, "[trace]   cb mode %u db render 0x%x depth control 0x%x z info 0x%x stencil info 0x%x control 0x%x"
                                 " ref/mask 0x%x z base 0x%" PRIx64 " htile 0x%" PRIx64 " clear %g prim %u count %u\n",
                         cbMode, dbRender, r.ctx[reg::kDbDepthControl], r.ctx[reg::kDbZInfo], r.ctx[reg::kDbStencilInfo],
                         r.ctx[reg::kDbStencilControl], r.ctx[reg::kDbStencilRefMask],
                         uint64_t(r.ctx[reg::kDbZReadBase]) << 8, uint64_t(r.ctx[reg::kDbHtileDataBase]) << 8,
                         std::bit_cast<float>(r.ctx[reg::kDbDepthClear]), r.prim, d.count);
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
        const uint32_t prim = r.prim;
        if (prim == 0 || prim == 8 || prim == 13) {
            warnOnce("primitive type " + std::to_string(prim) + " not rendered");
            return;
        }

        if (!a.ok) return;
        if (dlssOn_ && a.psm && a.psm->smaa && dlssSmaa(r, a.psm->smaa)) return;
        const bool wasMergeable = R().lastMergeable_;
        R().lastMergeable_ = false;
        if (fastDraw(d, r, a, wasMergeable) || switchDraw(d, r, a)) return;
        stashFast();
        // A recorder thread looks the caches up holding m_ shared first: almost every full draw
        // only finds what it needs. One that would change a cache (sharedMiss) starts over here
        // holding it exclusively. Everything up to wl.unlock() can run twice.
        bool sharedTry = R().worker_;
    retry:
        R().abort_ = false;
        WLock wl(*this, sharedTry);
        auto again = [&] { // true: start over exclusively
            if (!R().abort_) return false;
            wl.unlock();
            sharedTry = false;
            ++R().prof_.sharedRetries;
            return true;
        };
        gcn::Program& vs = a.vs;

        // Render targets. The pass and vectors below are reused from draw to draw (no allocations).
        Pass& pass = R().drawPass_;
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
        if (!colorCount && !hasDepth) {
            if (again()) goto retry;
            return;
        }
        // A scene draw (see dlssSmaa): every draw with the scene's depth buffer bound before SMAA,
        // full-screen passes too. Jittered alike, the lighting passes rebuild positions where the
        // depth was rendered: shadows flickered when only the geometry moved. After SMAA, 3D draws
        // that depth-test against it are jittered too: Ken's aura draws his body again, additively,
        // with an equal-or-nearer test, and unjittered it passed or failed by frame (it flickered).
        // 3D draws that write depth are where the camera is read.
        const bool sceneDepthBound = dlssOn_ && hasDepth && (uint64_t(r.ctx[reg::kDbZReadBase]) << 8) == sceneDepth_;
        const bool scene = sceneDepthBound && (!smaaSeen_ || (prim != 17 && ((depthControl >> 1) & 1)));
        if (scene && !smaaSeen_ && !cam_.valid && ((depthControl >> 2) & 1) && ((depthControl >> 1) & 1) && prim != 17 &&
            (r.ctx[reg::kPaClVteCntl] & 0xF))
            captureCamera(vs, r);
        const bool jitter = scene && jitterOn_;
        // A draw writing only depth into the open pass's depth buffer joins that pass, its color
        // targets attached but not written. Outside the city gate the game interleaves depth-only
        // draws with color ones on the same depth buffer: each switch was a new pass and barrier,
        // ~430 a frame (60 fps).
        if (!colorCount && hasDepth) ++R().prof_.depthOnly;
        if (!colorCount && hasDepth && R().pass_.open && R().pass_.colorCount && R().pass_.depth == depth.imageView &&
            R().pass_.stencil == gs.stencil && R().pass_.width <= width && R().pass_.height <= height && !noDepthJoin_) {
            // The pass's area is the smaller of its targets' (the depth buffer is often taller).
            ++R().prof_.depthJoins;
            colorCount = R().pass_.colorCount;
            width = R().pass_.width;
            height = R().pass_.height;
            for (uint32_t i = 0; i < colorCount; ++i) {
                colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                colors[i].imageView = R().pass_.views[i];
                colors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                colors[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                gs.colorFormats[i] = R().pass_.formats[i];
            }
        }

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

        std::vector<VkDescriptorSetLayoutBinding>& lb = R().drawBindings_;
        lb.clear();
        addBindings(lb, vs, 0, VK_SHADER_STAGE_VERTEX_BIT);
        if (psm) addBindings(lb, ps, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT);
        Layout* layp;
        VkPipeline pipe;
        {
            ProfScope pp(R().prof_, kProfPipeline);
            layp = &layout(lb);
            if (libraries_) libraryState(gs, dyn);
            pipe = graphicsPipeline(gs, vsm, psm, *layp, lb);
        }
        if (!pipe) {
            if (again()) goto retry;
            return;
        }
        Layout& lay = *layp;

        std::vector<VkWriteDescriptorSet>& writes = R().drawWrites_;
        std::vector<VkDescriptorBufferInfo>& bufs = R().drawBufs_;
        std::vector<VkDescriptorImageInfo>& imgs = R().drawImgs_;
        writes.clear();
        bufs.clear();
        imgs.clear();
        R().sampledSurfaces_.clear();
        if (trace_) {
            captureProgram(vs, vsm, "vs");
            if (psm) captureProgram(ps, psm, "ps");
            for (const auto* prog : {&vs, &ps}) {
                if (prog == &vs || psm) {
                    std::fprintf(traceOut_, "[trace]   %s user:", prog == &vs ? "vs" : "ps");
                    for (uint32_t i = 0; i < prog->launch.userSgprs() && i < 16; ++i) std::fprintf(traceOut_, " %x", prog->launch.userData[i]);
                    std::fprintf(traceOut_, "\n");
                }
                if (prog->data.size() <= gcn::data::smrd()) continue;
                std::fprintf(traceOut_, "[trace]   %s smrd:", prog == &vs ? "vs" : "ps");
                for (size_t i = gcn::data::smrd(); i < prog->data.size() && i < gcn::data::smrd() + 24; ++i) {
                    float f;
                    std::memcpy(&f, &prog->data[i], 4);
                    std::fprintf(traceOut_, " %g", f);
                }
                std::fprintf(traceOut_, "\n");
                for (const auto& w : prog->warnings) std::fprintf(traceOut_, "[trace]   warning: %s\n", w.c_str());
            }
        }
        { // set again on each try: hudTransform builds on it
            const bool pixels = !(r.ctx[reg::kPaClVteCntl] & 0xF);
            const float t[4] = {pixels ? 2.0f / float(width) : 1.0f, pixels ? 2.0f / float(height) : 1.0f,
                                pixels ? -1.0f : 0.0f, pixels ? -1.0f : 0.0f};
            std::memcpy(&vs.data[gcn::data::kPosTransform], t, sizeof(t));
        }
        // Before the draw data goes up with the resources (hudTransform changes it).
        const uint8_t hud = hudDraw(r, hasDepth, vs, psm ? &ps : nullptr, d, prim, width, height);
        if (hud) hudTransform(vs, r, width, height, hud);
        bindResources(vs, 0, VK_SHADER_STAGE_VERTEX_BIT, writes, bufs, imgs, aheadFor(a, vs));
        if (psm) bindResources(ps, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT, writes, bufs, imgs, aheadFor(a, ps));
        linkWrites(writes, bufs, imgs);

        // Index buffer.
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        if (d.kind == DrawCall::kIndexed) {
            const uint32_t isz = r.indexType ? 4 : 2;
            if (!vramBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset) &&
                !guestBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset)) {
                if (again()) goto retry;
                return;
            }
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
        if (again()) goto retry;
        if (wl.shared()) ++R().prof_.sharedDraws;
        // Everything below is this thread's own (its pass, bindings and command buffer): at the
        // overlook, holding m_ to the end of each full draw left the threads waiting for it half
        // their time.
        wl.unlock();

        // Consecutive draws into the same targets share a render pass. A new pass starts behind a
        // full barrier, as before every draw, so earlier passes, copies and dispatches are visible.
        // Inside a pass only its own attachments change, so a draw sampling one of them (or one
        // after a draw that wrote memory from a shader) needs a new pass.
        for (uint32_t i = 0; i < colorCount; ++i) pass.views[i] = colors[i].imageView, pass.formats[i] = gs.colorFormats[i];
        pass.depth = hasDepth ? depth.imageView : VK_NULL_HANDLE;
        pass.colorCount = colorCount;
        pass.stencil = hasDepth && gs.stencil;
        pass.width = width;
        pass.height = height;
        // Sampling an attachment needs a new pass (and its barrier) only once a draw in the open
        // pass has written it: decals and soft particles that read the depth buffer without
        // writing it then share one pass.
        bool feedback = false;
        for (VkImage im : R().sampledSurfaces_)
            if (std::find(R().drawWritten_.begin(), R().drawWritten_.end(), im) != R().drawWritten_.end()) {
                feedback = true;
                if (R().prof_.on && R().pass_.open && R().pass_.sameTargets(pass)) ++R().prof_.feedbackOn[im == R().pass_.depthImage ? 0 : 1];
            }
        if (!R().pass_.open || feedback || !R().pass_.sameTargets(pass)) {
            if (R().prof_.on) ++R().prof_.passWhy[!R().pass_.open ? R().passClosedBy_ : feedback ? kPassFeedback : kPassTargets];
            if (R().pass_.open) { // to other targets: no barrier unless something since the last one needs it
                rec([c = R().cmd_] { vkCmdEndRendering(c); });
                R().pass_.open = false;
            }
            // Draws since the last barrier wrote and sampled only images; another pass needs the
            // barrier when it samples what they wrote, or renders into what they wrote or sampled.
            if (R().needBarrier_ || feedback || overlaps(pass.images, R().drawWritten_) || overlaps(pass.images, R().drawSampled_))
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
            rec([c = R().cmd_, ri] { vkCmdBeginRendering(c, &ri); });
            std::swap(R().pass_, pass); // the old pass's storage is reused by the next draw
            R().pass_.open = true;
            ++R().passSerial_;
            ++R().prof_.passes;
        }
        if (R().bound_.pipeline != pipe) {
            rec([c = R().cmd_, pipe] { vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); });
            R().bound_.pipeline = pipe;
        }
        pushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, lay, writes, bufs, imgs);
        setViewportScissor(r, width, height, jitter, hud);
        if (!R().bound_.state || std::memcmp(R().bound_.blend, &r.ctx[reg::kCbBlendRed], 16) != 0) {
            std::memcpy(R().bound_.blend, &r.ctx[reg::kCbBlendRed], 16);
            rec([c = R().cmd_, b = std::bit_cast<std::array<float, 4>>(R().bound_.blend)] { vkCmdSetBlendConstants(c, b.data()); });
        }
        setDynamicState(dyn, gs.depthControl);
        R().bound_.state = true;
        if (clearDepth) {
            VkClearAttachment ca{};
            ca.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            float z;
            std::memcpy(&z, &r.ctx[reg::kDbDepthClear], 4);
            ca.clearValue.depthStencil = {z, r.ctx[reg::kDbStencilClear] & 0xFF};
            if (gs.stencil && (dbRender & 2)) ca.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            VkClearRect cr{{{0, 0}, {width, height}}, 0, 1};
            rec([c = R().cmd_, ca, cr] { vkCmdClearAttachments(c, 1, &ca, 1, &cr); });
        }
        if (d.kind == DrawCall::kIndexed) {
            const VkIndexType it = r.indexType ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (R().bound_.index != indexBuffer || R().bound_.indexOffset != indexOffset || R().bound_.indexType != it) {
                rec([c = R().cmd_, indexBuffer, indexOffset, it] { vkCmdBindIndexBuffer(c, indexBuffer, indexOffset, it); });
                R().bound_.index = indexBuffer, R().bound_.indexOffset = indexOffset, R().bound_.indexType = it;
            }
            rec([c = R().cmd_, n = d.count, inst = std::max(r.numInstances, 1u), first = d.indexOffset] {
                vkCmdDrawIndexed(c, n, inst, first, 0, 0);
            });
        } else {
            rec([c = R().cmd_, n = d.count, inst = std::max(r.numInstances, 1u)] { vkCmdDraw(c, n, inst, 0, 0); });
        }
        auto stores = [](const gcn::Program& p) {
            for (const auto& b : p.buffers)
                if (b.store) return true;
            for (const auto& i : p.images)
                if (i.storage) return true;
            return false;
        };
        for (VkImage im : R().pass_.images) // what this draw wrote
            if ((im != R().pass_.depthImage || clearDepth || (((depthControl >> 2) & 1) && ((depthControl >> 1) & 1)) ||
                 (gs.stencil && (depthControl & 1) && ((r.ctx[reg::kDbStencilRefMask] >> 16) & 0xFF) &&
                  (r.ctx[reg::kDbStencilControl] & 0xFFFFFF))) &&
                std::find(R().pass_.written.begin(), R().pass_.written.end(), im) == R().pass_.written.end()) {
                R().pass_.written.push_back(im);
                addOnce(R().drawWritten_, im);
            }
        for (VkImage im : R().sampledSurfaces_) addOnce(R().drawSampled_, im);
        if (stores(vs) || (psm && stores(ps))) {
            endPass();
            R().passClosedBy_ = kPassStore;
        }
        ++R().stats_.draws;
        if (trace_ && snap_ && colorCount && snapWanted()) snapshot(uint64_t(r.ctx[reg::kCbColorBase]) << 8);
        // What a repeat of this draw can take as is (see fastDraw).
        bool repeatable = R().pass_.open && !clearDepth && !trace_ && !raceCheck_ && !snap_;
        for (VkImage im : R().sampledSurfaces_) // sampling what it renders to: a repeat needs a new pass
            if (std::find(R().drawWritten_.begin(), R().drawWritten_.end(), im) != R().drawWritten_.end()) repeatable = false;
        if (repeatable && !noFastDraws_) {
            FastDraw& f = R().fast_;
            f.valid = true;
            f.rec = R().recCount_;
            f.frame = frame_;
            f.imageGen = imageGen_;
            f.kind = d.kind;
            f.indexType = r.indexType;
            f.numInstances = r.numInstances;
            f.prim = prim;
            f.ctxHash = a.ctxHash;
            if (R().prof_.on) f.ctx.assign(r.ctx, r.ctx + ::gpu::kContextRegs);
            f.layout = &lay;
            f.vsm = vsm;
            f.psm = psm;
            f.pipe = pipe;
            f.dyn = dyn;
            f.depthControl = gs.depthControl;
            f.jitter = jitter;
            f.hud = hud;
            f.width = width;
            f.height = height;
            f.pass = R().passSerial_;
            f.epoch = fastEpoch_;
            f.sampled = R().sampledSurfaces_;
            f.writes = writes;
            f.bufs = bufs;
            f.imgs = imgs;
            R().keepDraw_ = true;
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
        uint64_t ctxHash = 0;      // DrawAnalysis::ctxHash
        std::vector<uint32_t> ctx; // the registers themselves, with LPN_PROFILE (which ones differ)
        const Layout* layout = nullptr;
        ShaderModule *vsm = nullptr, *psm = nullptr;
        // What switchDraw sets again: the pipeline, dynamic state and every descriptor as the
        // full draw bound them, in pass `pass` with memory as of `epoch`.
        VkPipeline pipe = VK_NULL_HANDLE;
        DynamicState dyn{};
        uint32_t depthControl = 0, width = 0, height = 0;
        bool jitter = false;
        uint8_t hud = 0; // see hudDraw
        uint64_t pass = 0, epoch = 0;
        std::vector<VkImage> sampled;
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> bufs;
        std::vector<VkDescriptorImageInfo> imgs;
    };
    // Draws switching back and forth between a few shaders in one pass (~1,000 of the 6,350
    // looking up the steps after New Game): the fast states of the ones not bound now, with
    // their last draws.
    struct FastSlot {
        FastDraw f;
        std::unique_ptr<DrawAnalysis> prev = std::make_unique<DrawAnalysis>();
        uint64_t used = 0;
    };
    // 16: the city gate switches between more shaders in a pass than the steps (4 were enough there).
    // 64: at the rooftop ~840 draws a frame found no slot for their shaders with 16. Lookups go
    // through R().slotKeys_ (one cache line per 4 slots) instead of each slot's own state.
    static constexpr size_t kFastSlots = 64;
    // LPN_NO_FAST_TEXTURES=1: a draw sampling other textures than the last is recorded in full.
    const bool noFastTextures_ = getenv("LPN_NO_FAST_TEXTURES") != nullptr;
    struct SlotKey {
        const ShaderModule *vsm = nullptr, *psm = nullptr;
    };
    std::atomic<uint64_t> fastEpoch_ = 0; // read by recorder threads

    // The bound fast state, before a full draw replaces it: kept in a slot (its shaders' one, or
    // the least recently used) while its pass lasts.
    void stashFast() {
        if (R().fast_.valid && R().fast_.pass == R().passSerial_ && R().pass_.open && R().fast_.epoch == fastEpoch_ && R().fast_.frame == frame_) {
            size_t i = 0;
            for (size_t k = 0; k < kFastSlots; ++k) {
                if (R().slotKeys_[k].vsm == R().fast_.vsm && R().slotKeys_[k].psm == R().fast_.psm) {
                    i = k;
                    break;
                }
                if (R().slotUsed_[k] < R().slotUsed_[i]) i = k;
            }
            FastSlot* slot = &R().fastSlots_[i];
            R().slotKeys_[i] = {R().fast_.vsm, R().fast_.psm};
            R().slotUsed_[i] = R().fastUse_ + 1;
            std::swap(slot->f, R().fast_);
            std::swap(slot->prev, R().prevDraw_);
            slot->used = ++R().fastUse_;
        }
        R().fast_.valid = false;
    }

    // A draw repeating one in a slot (see FastSlot) but for its user data, constants and buffer
    // placement, with only draws recorded since in the same pass: its pipeline, dynamic state and
    // descriptors are set again from the slot, and its state becomes the bound one.
    bool switchDraw(const DrawCall& d, const DrawRegs& r, DrawAnalysis& a) {
        if (!R().recording_ || !R().pass_.open || !ringFits(1 << 16)) return false;
        auto miss = [&](int why) {
            if (R().prof_.on && (R().fast_.vsm != a.vsm || R().fast_.psm != a.psm)) ++R().prof_.switchMiss[why];
            return false;
        };
        FastSlot* slot = nullptr;
        size_t si = 0;
        for (size_t k = 0; k < kFastSlots; ++k)
            if (R().slotKeys_[k].vsm == a.vsm && R().slotKeys_[k].psm == a.psm && R().fastSlots_[k].f.valid) slot = &R().fastSlots_[si = k];
        if (!slot) return miss(0);
        FastDraw& f = slot->f;
        if (f.pass != R().passSerial_ || f.epoch != fastEpoch_ || f.frame != frame_ || f.imageGen != imageGen_) return miss(1);
        if (d.kind != f.kind || r.indexType != f.indexType || r.numInstances != f.numInstances ||
            r.prim != f.prim || a.ctxHash != f.ctxHash)
            return miss(2);
        for (VkImage im : f.sampled) // sampling what a draw since has rendered to needs a new pass
            if (std::find(R().drawWritten_.begin(), R().drawWritten_.end(), im) != R().drawWritten_.end()) return miss(3);
        if (!sameHudMode(f, d, a)) return miss(3);
        uint32_t moved, vsImages = 0, psImages = 0;
        ImageBinding vsImg[32], psImg[32];
        uint32_t psMoved = 0;
        if (sameButUserData(a.vs, slot->prev->vs, moved, noFastTextures_ ? nullptr : &vsImages) >= 0 ||
            (a.psm && sameButUserData(a.ps, slot->prev->ps, psMoved, noFastTextures_ ? nullptr : &psImages) >= 0))
            return miss(4);
        WLock wl(*this, true); // to the descriptors: the caches, shared (read only)
        if (!peekImages(a.vs, vsImages, vsImg) || !peekImages(a.ps, psImages, psImg)) return miss(4);
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        if (d.kind == DrawCall::kIndexed) {
            const uint32_t isz = r.indexType ? 4 : 2;
            if (!vramBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset, true) &&
                !guestBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset))
                return miss(5);
        }
        ProfScope pb(R().prof_, kProfBind);
        std::vector<VkWriteDescriptorSet>& writes = R().drawWrites_;
        std::vector<VkDescriptorImageInfo>& imgs = R().drawImgs_;
        // Other textures go into the slot's descriptors first (rebindImages also queues writes of its
        // own for fastDraw, dropped here: all of the slot's are pushed).
        writes.clear(), imgs.clear();
        rebindImages(a.vs, slot->prev->vs, vsImages, vsImg, 0, f);
        if (a.psm) rebindImages(a.ps, slot->prev->ps, psImages, psImg, kPsBindingBase, f);
        // The buffers that moved and the draw data, bound again in the slot's descriptors (bufs is in
        // write order); the rest stay as the slot has them. Rebinding every buffer (15-20 a draw
        // outside the city gate) was most of the recorder's bind time.
        // A scan of the slot's writes per buffer: a binding table filled for every draw cost more
        // than the few buffers that move.
        auto bufferAt = [&](uint32_t binding) -> VkDescriptorBufferInfo* {
            for (size_t k = 0, n = 0; k < f.writes.size(); ++k)
                if (f.writes[k].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    if (f.writes[k].dstBinding == binding) return &f.bufs[n];
                    ++n;
                }
            return nullptr;
        };
        auto bind = [&](gcn::Program& p, gcn::Program& bound, uint32_t moved, uint32_t base) {
            for (uint32_t i = 0; moved; ++i, moved >>= 1)
                if (moved & 1)
                    if (VkDescriptorBufferInfo* b = bufferAt(base + 1 + i)) {
                        const std::vector<BufRes>* pre = aheadFor(a, p);
                        *b = bindBuffer(p, p.buffers[i], true, pre ? &(*pre)[i] : nullptr);
                        std::memcpy(bound.buffers[i].sharp, p.buffers[i].sharp, sizeof p.buffers[i].sharp);
                    }
            const VkDeviceSize bytes = p.data.size() * 4;
            const VkDeviceSize at = uploadData(p.data);
            if (VkDescriptorBufferInfo* b = bufferAt(base)) *b = {R().ring_, at, bytes};
            std::swap(bound.data, p.data); // as fastDraw: the analysis is handed back after this draw
        };
        R().abort_ = false;
        bind(a.vs, slot->prev->vs, moved, 0);
        if (a.psm) bind(a.ps, slot->prev->ps, psMoved, kPsBindingBase);
        wl.unlock();
        if (R().abort_) { // a buffer needed a cache change: bound again holding m_
            R().abort_ = false;
            std::swap(slot->prev->vs.data, a.vs.data);
            if (a.psm) std::swap(slot->prev->ps.data, a.ps.data);
            WLock wx(*this);
            bind(a.vs, slot->prev->vs, moved, 0);
            if (a.psm) bind(a.ps, slot->prev->ps, psMoved, kPsBindingBase);
        }
        if (R().bound_.pipeline != f.pipe) {
            rec([c = R().cmd_, pipe = f.pipe] { vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); });
            R().bound_.pipeline = f.pipe;
        }
        pushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, *f.layout, f.writes, f.bufs, f.imgs);
        setViewportScissor(r, f.width, f.height, f.jitter, f.hud);
        if (!R().bound_.state || std::memcmp(R().bound_.blend, &r.ctx[reg::kCbBlendRed], 16) != 0) {
            std::memcpy(R().bound_.blend, &r.ctx[reg::kCbBlendRed], 16);
            rec([c = R().cmd_, b = std::bit_cast<std::array<float, 4>>(R().bound_.blend)] { vkCmdSetBlendConstants(c, b.data()); });
        }
        setDynamicState(f.dyn, f.depthControl);
        R().bound_.state = true;
        if (d.kind == DrawCall::kIndexed) {
            const VkIndexType it = r.indexType ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (R().bound_.index != indexBuffer || R().bound_.indexOffset != indexOffset || R().bound_.indexType != it) {
                rec([c = R().cmd_, indexBuffer, indexOffset, it] { vkCmdBindIndexBuffer(c, indexBuffer, indexOffset, it); });
                R().bound_.index = indexBuffer, R().bound_.indexOffset = indexOffset, R().bound_.indexType = it;
            }
            rec([c = R().cmd_, n = d.count, inst = std::max(r.numInstances, 1u), first = d.indexOffset] {
                vkCmdDrawIndexed(c, n, inst, first, 0, 0);
            });
        } else {
            rec([c = R().cmd_, n = d.count, inst = std::max(r.numInstances, 1u)] { vkCmdDraw(c, n, inst, 0, 0); });
        }
        // The slot's state is the bound one now; the one bound before takes its slot.
        const bool keepOld = R().fast_.valid && R().fast_.pass == R().passSerial_ && R().fast_.epoch == fastEpoch_ && R().fast_.frame == frame_;
        std::swap(slot->f, R().fast_);
        std::swap(slot->prev, R().prevDraw_);
        slot->f.valid = keepOld;
        slot->used = ++R().fastUse_;
        R().slotKeys_[si] = {slot->f.vsm, slot->f.psm};
        R().slotUsed_[si] = slot->used;
        R().fast_.rec = R().recCount_;
        R().recordedSeq_ = a.seq;
        ++R().stats_.draws;
        ++R().prof_.fastDraws;
        ++R().prof_.fastSwitches;
        return true;
    }
    // Aspect Ratio: whether a draw taking a kept draw's state (fastDraw, switchDraw) is mapped as that
    // one was (hudDraw's mode): its position transform comes along. The title menu's text, on the
    // backdrop's shaders, was drawn across the whole picture now and then.
    bool sameHudMode(const FastDraw& f, const DrawCall& d, const DrawAnalysis& a) {
        if (!hud().on) return true;
        if ((f.hud == 3) != (d.count == 3 && sceneCopy16x9(a.vs))) return false;
        if (f.hud == 1 || f.hud == 2) return (hudWide(a.vs, a.psm ? &a.ps : nullptr, d) ? 2 : 1) == f.hud;
        return true;
    }
    bool fastDraw(const DrawCall& d, const DrawRegs& r, DrawAnalysis& a, bool prevMergeable) {
        const FastDraw& f = R().fast_;
        auto miss = [&](FastMiss why) {
            if (R().prof_.on) ++R().prof_.fastMiss[why];
            return false;
        };
        if (!f.valid) return miss(kFastNotKept);
        if (f.rec != R().recCount_) return miss(kFastBetween);
        if (f.frame != frame_ || f.imageGen != imageGen_ || !R().recording_ || !R().pass_.open) return miss(kFastFrame);
        if (d.kind != f.kind || a.vsm != f.vsm || a.psm != f.psm || r.indexType != f.indexType ||
            r.numInstances != f.numInstances || r.prim != f.prim || !ringFits(1 << 16)) {
            if (R().prof_.on) ++R().prof_.fastKind[d.kind != f.kind ? 0 : a.vsm != f.vsm || a.psm != f.psm ? 1 : 2];
            return miss(kFastKind);
        }
        if (a.ctxHash != f.ctxHash) {
            if (R().prof_.on && f.ctx.size() == ::gpu::kContextRegs)
                for (uint32_t i = 0; i < ::gpu::kContextRegs; ++i)
                    if (r.ctx[i] != f.ctx[i]) ++R().prof_.fastCtx[i];
            return miss(kFastCtx);
        }
        uint32_t vsMoved = 0, psMoved = 0, vsImages = 0, psImages = 0;
        int why;
        int vsAlike = -1, psAlike = -1; // see DrawAnalysis::Diff::data
        if (a.diffPrev && R().prevSeq_ + 1 == a.seq) { // R().prevDraw_'s resources are the previous draw's
            ++R().prof_.diffsTaken;
            vsAlike = a.vsDiff.data, psAlike = a.psDiff.data;
            why = a.vsDiff.why;
            if (why < 0 && a.psm) why = a.psDiff.why;
            if (why < 0) {
                vsMoved = a.vsDiff.moved, vsImages = a.vsDiff.images;
                psMoved = a.psDiff.moved, psImages = a.psDiff.images;
                if (noFastTextures_ && (vsImages || psImages)) why = 2;
            }
            if (why < 0) {
                fillFromPrev(a.vs, R().prevDraw_->vs, vsMoved, vsImages);
                if (a.psm) fillFromPrev(a.ps, R().prevDraw_->ps, psMoved, psImages);
            }
        } else {
            uint32_t* const vi = noFastTextures_ ? nullptr : &vsImages;
            uint32_t* const pi = noFastTextures_ ? nullptr : &psImages;
            why = sameButUserData(a.vs, R().prevDraw_->vs, vsMoved, vi);
            if (why < 0 && a.psm) why = sameButUserData(a.ps, R().prevDraw_->ps, psMoved, pi);
        }
        // Other textures (another material on the same shaders): only ones already resolved this frame.
        ImageBinding vsImg[32], psImg[32];
        WLock wl(*this, true); // from here to the descriptors: the caches, shared (read only)
        if (why < 0 && !(peekImages(a.vs, vsImages, vsImg) && peekImages(a.ps, psImages, psImg))) why = 2;
        if (why >= 0) {
            if (R().prof_.on) ++R().prof_.fastRes[why];
            return miss(kFastResources);
        }
        if (!sameHudMode(f, d, a)) return miss(kFastResources);
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        if (d.kind == DrawCall::kIndexed) {
            const uint32_t isz = r.indexType ? 4 : 2;
            if (!vramBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset, true) &&
                !guestBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset))
                return miss(kFastIndex);
        }
        ProfScope pb(R().prof_, kProfBind);
        std::vector<VkWriteDescriptorSet>& writes = R().drawWrites_;
        std::vector<VkDescriptorBufferInfo>& bufs = R().drawBufs_;
        std::vector<VkDescriptorImageInfo>& imgs = R().drawImgs_;
        writes.clear();
        bufs.clear();
        imgs.clear();
        // Buffers that moved, then the draw data; R().prevDraw_ holds what is bound now.
        auto rebind = [&](gcn::Program& p, gcn::Program& bound, uint32_t moved, uint32_t base) {
            for (uint32_t i = 0; moved; ++i, moved >>= 1)
                if (moved & 1) {
                    const std::vector<BufRes>* pre = aheadFor(a, p);
                    bufs.push_back(bindBuffer(p, p.buffers[i], true, pre ? &(*pre)[i] : nullptr));
                    writes.push_back(kBufferWrite);
                    writes.back().dstBinding = base + 1 + i;
                    // Also in the fast state, which switchDraw binds again unchanged but for moved buffers.
                    for (size_t k = 0, n = 0; k < R().fast_.writes.size(); ++k)
                        if (R().fast_.writes[k].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                            if (R().fast_.writes[k].dstBinding == base + 1 + i) {
                                R().fast_.bufs[n] = bufs.back();
                                break;
                            }
                            ++n;
                        }
                    std::memcpy(bound.buffers[i].sharp, p.buffers[i].sharp, sizeof p.buffers[i].sharp);
                }
        };
        if (vsMoved || psMoved) ++R().prof_.fastMoved;
        if (vsImages || psImages) ++R().prof_.fastImages;
        auto rebindAll = [&] {
            rebind(a.vs, R().prevDraw_->vs, vsMoved, 0);
            if (a.psm) rebind(a.ps, R().prevDraw_->ps, psMoved, kPsBindingBase);
            rebindImages(a.vs, R().prevDraw_->vs, vsImages, vsImg, 0, R().fast_);
            if (a.psm) rebindImages(a.ps, R().prevDraw_->ps, psImages, psImg, kPsBindingBase, R().fast_);
        };
        const size_t bufsBefore = bufs.size(), writesBefore = writes.size(), imgsBefore = imgs.size();
        R().abort_ = false;
        rebindAll();
        wl.unlock();
        if (R().abort_) { // a buffer needed a cache change: bound again holding m_
            R().abort_ = false;
            bufs.resize(bufsBefore), writes.resize(writesBefore), imgs.resize(imgsBefore);
            WLock wx(*this);
            rebindAll();
        }
        auto data = [&](gcn::Program& p, gcn::Program& bound, uint32_t base, int alike, bool rebound) {
            const VkDeviceSize bytes = p.data.size() * 4;
            if (alike == 1 && !rebound) return; // the worker found it the same
            if (alike != 0 && !std::memcmp(p.data.data(), bound.data.data(), bytes)) return;
            const VkDeviceSize at = uploadData(p.data);
            bufs.push_back({R().ring_, at, bytes});
            writes.push_back(kBufferWrite);
            writes.back().dstBinding = base;
            // Now what is bound: swapped, not copied (the analysis is handed back after this draw).
            std::swap(bound.data, p.data);
        };
        data(a.vs, R().prevDraw_->vs, 0, vsAlike, vsMoved || vsImages);
        if (a.psm) data(a.ps, R().prevDraw_->ps, kPsBindingBase, psAlike, psMoved || psImages);
        if (!writes.empty()) pushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, *f.layout, writes, bufs, imgs);
        bool sameIndex = true;
        if (d.kind == DrawCall::kIndexed) {
            const VkIndexType it = r.indexType ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (R().bound_.index != indexBuffer || R().bound_.indexOffset != indexOffset || R().bound_.indexType != it) {
                sameIndex = false;
                rec([c = R().cmd_, indexBuffer, indexOffset, it] { vkCmdBindIndexBuffer(c, indexBuffer, indexOffset, it); });
                R().bound_.index = indexBuffer, R().bound_.indexOffset = indexOffset, R().bound_.indexType = it;
            }
            rec([c = R().cmd_, n = d.count, inst = std::max(r.numInstances, 1u), first = d.indexOffset] {
                vkCmdDrawIndexed(c, n, inst, first, 0, 0);
            });
        } else {
            rec([c = R().cmd_, n = d.count, inst = std::max(r.numInstances, 1u)] { vkCmdDraw(c, n, inst, 0, 0); });
        }
        R().fast_.rec = R().recCount_;
        R().recordedSeq_ = a.seq;
        ++R().stats_.draws;
        ++R().prof_.fastDraws;
        R().lastMergeable_ = sameIndex && !(vsMoved | psMoved | vsImages | psImages);
        if (R().lastMergeable_) {
            ++R().prof_.mergeable;
            if (!prevMergeable) ++R().prof_.mergeRuns;
        }
        return true;
    }
    bool peekImages(const gcn::Program& p, uint32_t mask, ImageBinding* out) {
        for (uint32_t i = 0; mask; ++i, mask >>= 1)
            if ((mask & 1) && !peekImage(p.images[i], out[i])) return false;
        return true;
    }
    // Binds the images in `mask` (resolved by peekImages) as fastDraw's next descriptor writes, and
    // keeps them in `f` and `bound` for a later switchDraw back to these shaders.
    void rebindImages(gcn::Program& p, gcn::Program& bound, uint32_t mask, const ImageBinding* ib, uint32_t base,
                      FastDraw& f) {
        for (uint32_t i = 0; mask; ++i, mask >>= 1) {
            if (!(mask & 1)) continue;
            gcn::ImageResource& im = p.images[i];
            std::memcpy(&p.data[im.data], ib[i].scale, 8);
            const VkDescriptorImageInfo info{ib[i].sampler, ib[i].view, VK_IMAGE_LAYOUT_GENERAL};
            const uint32_t binding = base + 1 + uint32_t(p.buffers.size()) + i;
            R().drawImgs_.push_back(info);
            R().drawWrites_.push_back(kSampledImageWrite);
            R().drawWrites_.back().dstBinding = binding;
            size_t n = 0;
            for (const auto& w : f.writes)
                if (w.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    if (w.dstBinding == binding) {
                        f.imgs[n] = info;
                        break;
                    }
                    ++n;
                }
            std::memcpy(bound.images[i].t, im.t, sizeof im.t);
            std::memcpy(bound.images[i].s, im.s, sizeof im.s);
        }
    }
    // Whether `cur` is `prev` (as bindResources and draw() left it) but for its user data words, the
    // scalar loads made from them (constants, uploaded with the draw data) and where read-only
    // buffers sit (per-draw vertex data): -1 if so, else why not (FastRes). Fills in the words the
    // resources fill from prev's (the same resources resolve alike); `moved` gets the buffers to
    // bind again. With `images`, sampled images may differ too, if the shader reads them alike
    // (same type, number kind, coordinates, compare): it gets those to bind again, their scale
    // words left for the caller.
    static int sameButUserData(gcn::Program& cur, const gcn::Program& prev, uint32_t& moved, uint32_t* images = nullptr) {
        const int why = compareResources(cur, prev, moved, images);
        if (why < 0) fillFromPrev(cur, prev, moved, images ? *images : 0);
        return why;
    }
    // sameButUserData's check, which reads only what analysis produced (an analysis worker can run it).
    static int compareResources(const gcn::Program& cur, const gcn::Program& prev, uint32_t& moved, uint32_t* images) {
        moved = 0;
        if (images) *images = 0;
        if (cur.data.size() != prev.data.size() || cur.buffers.size() != prev.buffers.size() ||
            cur.images.size() != prev.images.size() || cur.data.size() < gcn::data::smrd() || cur.buffers.size() > 32)
            return 0;
        for (size_t i = 0; i < cur.buffers.size(); ++i) {
            const gcn::BufferResource &x = cur.buffers[i], &y = prev.buffers[i];
            if (x.known != y.known || x.store != y.store || x.data != y.data || x.dfmt != y.dfmt || x.nfmt != y.nfmt) return 1;
            if (std::memcmp(x.sharp, y.sharp, sizeof x.sharp)) {
                if (x.store || !x.known || x.sharp[3] != y.sharp[3] || x.stride() != y.stride()) return 1;
                moved |= 1u << i;
            }
        }
        for (size_t i = 0; i < cur.images.size(); ++i) {
            const gcn::ImageResource &x = cur.images[i], &y = prev.images[i];
            if (x.storage != y.storage || x.compare != y.compare || x.data != y.data) return 2;
            if (std::memcmp(x.t, y.t, sizeof x.t) || std::memcmp(x.s, y.s, sizeof x.s)) {
                if (!images || x.storage || i >= 32 || x.type() != y.type() || x.numFormat() != y.numFormat() ||
                    ((x.s[0] ^ y.s[0]) >> 15 & 1))
                    return 2;
                *images |= 1u << i;
            }
        }
        return -1;
    }
    // Whether cur's draw data is prev's (both as analysed, same layout) but for the words resolution
    // fills in: then, filled from prev, it is what prev bound (fastDraw need not compare them). On a
    // worker: comparing on the recorder was ~6% of it, the data written on another core.
    static bool dataAlike(const gcn::Program& cur, const gcn::Program& prev) {
        const size_t n = cur.data.size();
        if (n != prev.data.size() || n < gcn::data::smrd()) return false;
        thread_local std::vector<uint8_t> skip;
        skip.assign(n, 0);
        // A pixel shader's words there are its colour swaps (generic translation), compared.
        if (cur.launch.stage != gcn::Stage::Pixel) std::fill_n(&skip[gcn::data::kPosTransform], 4, 1);
        for (const auto& b : cur.buffers)
            if (b.data + 4 <= n) std::fill_n(&skip[b.data], 4, 1);
        for (const auto& im : cur.images)
            if (im.data + 2 <= n) std::fill_n(&skip[im.data], 2, 1);
        const uint32_t *x = cur.data.data(), *y = prev.data.data();
        for (size_t i = 0; i < n; ++i)
            if (x[i] != y[i] && !skip[i]) return false;
        return true;
    }
    // sameButUserData's fill: the words resolution put in prev's data, for the resources that stay.
    static void fillFromPrev(gcn::Program& cur, const gcn::Program& prev, uint32_t moved, uint32_t images) {
        std::memcpy(&cur.data[gcn::data::kPosTransform], &prev.data[gcn::data::kPosTransform], 16);
        for (size_t i = 0; i < prev.buffers.size(); ++i)
            if (!(moved >> i & 1)) std::memcpy(&cur.data[prev.buffers[i].data], &prev.data[prev.buffers[i].data], 16);
        for (size_t i = 0; i < prev.images.size(); ++i)
            if (!(images >> i & 1)) std::memcpy(&cur.data[prev.images[i].data], &prev.data[prev.images[i].data], 8);
    }

    void endPass() {
        R().needBarrier_ = true; // something other than a draw comes next
        if (!R().pass_.open) return;
        rec([c = R().cmd_] { vkCmdEndRendering(c); });
        R().pass_.open = false;
        R().passClosedBy_ = kPassOther;
    }

    // ---- Aspect Ratio (display_settings.h hudLayout) ------------------------------------------
    // With a picture that is not 16:9 the game outputs at its size and draws its scene to all of
    // it, but lays its HUD out as 16:9 at the picture's width: past the edges of a wider picture. Draws into
    // the display buffers are mapped into the largest 16:9 rectangle in the middle, but for the
    // full-screen scene passes: draws that sample a render target of the picture's shape (the
    // scene) or have a depth buffer bound. Videos (1920x1080 textures) come out 16:9 that way too.
    struct Hud {
        bool on = false;
        uint32_t outW = 0, outH = 0;
        double aspect = 16.0 / 9;
        float x = 0, y = 0, sx = 1, sy = 1; // target pixel = (x, y) + HUD pixel * (sx, sy)
    };
    static const Hud& hud() {
        static const Hud h = [] {
            Hud k;
            if (!hudLayout(&k.outW, &k.outH) || !k.outH) return k;
            if (const char* e = getenv("LPN_HUD_STRETCH"); e && std::strcmp(e, "0") != 0) return k;
            k.on = true;
            k.aspect = double(k.outW) / k.outH;
            // The game lays the HUD out over the whole picture (1280x720 stretched to it), into its
            // HUD buffer, and copies that to the display as 16:9 at the picture's width, from the top
            // (sceneCopy16x9, fitted to the picture since). The layout is squeezed into the largest
            // 16:9 rectangle in the middle.
            const double w = std::min<double>(k.outW, k.outH * 16.0 / 9), h = std::min<double>(k.outH, k.outW * 9.0 / 16);
            k.sx = float(w / k.outW), k.sy = float(h / k.outH);
            k.x = float((k.outW - w) / 2), k.y = float((k.outH - h) / 2);
            std::fprintf(stderr, "[hud] HUD drawn %.0fx%.0f at %.0f,%.0f of the %ux%u picture\n", w, h, k.x, k.y, k.outW, k.outH);
            return k;
        }();
        return h;
    }
    // The display buffers' addresses, as flips show them (any thread).
    static constexpr int kMaxDisplays = 16;
    std::atomic<uint64_t> displayAddrs_[kMaxDisplays] = {};
    void noteDisplay(uint64_t addr) {
        for (auto& a : displayAddrs_) {
            uint64_t v = a.load(std::memory_order_relaxed);
            if (v == addr) return;
            if (!v && a.compare_exchange_strong(v, addr)) return;
            if (v == addr) return;
        }
    }
    bool isDisplay(uint64_t addr) const {
        for (const auto& a : displayAddrs_) {
            const uint64_t v = a.load(std::memory_order_relaxed);
            if (!v) return false;
            if (v == addr) return true;
        }
        return false;
    }
    // Whether a draw is HUD (above). Holding m_ (the surfaces are looked up).
    // The game draws its HUD over the finished scene in a picture-sized buffer of its own, which a
    // full-screen pass then copies into the display buffer. Such a buffer is told apart as one the
    // display buffers' draws sample that no draw with a depth buffer renders into (the scene's
    // targets have those). Draws into it, and into the display buffers, are HUD but for the ones
    // sampling a picture-shaped render target (the scene being copied in, post-processing).
    // 0: drawn as it is, 1: the HUD (into the 16:9 rectangle), 2: a HUD backdrop (hudWide:
    // stretched across the whole picture, only its height fitted), 3: sceneCopy16x9.
    uint8_t hudDraw(const DrawRegs& r, bool hasDepth, const gcn::Program& vs, const gcn::Program* ps, const DrawCall& d,
                    uint32_t prim, uint32_t width, uint32_t height) {
        const Hud& h = hud();
        if (!h.on) return 0;
        // A picture-sized target (a little padding allowed): the others are not the HUD's.
        if (width < h.outW || width > h.outW + 128 || height < h.outH || height > h.outH + 32) return 0;
        const uint64_t target = uint64_t(r.ctx[reg::kCbColorBase]) << 8;
        uint64_t picture[8];
        int pictures = 0;
        uint32_t sceneW = 0, sceneH = 0;
        for (const gcn::Program* p : {&vs, ps}) {
            if (!p) continue;
            for (const auto& t : p->images) {
                if (t.storage || !t.height() || t.width() < h.outW / 8) continue;
                if (std::fabs(double(t.width()) / t.height() - h.aspect) > 0.06 * h.aspect) continue;
                if (!colors_.count(t.base()) && !depths_.count(t.base())) continue;
                sceneW = t.width(), sceneH = t.height();
                if (pictures < 8) picture[pictures++] = t.base();
            }
        }
        const bool display = isDisplay(target);
        bool layer = false;
        {
            std::lock_guard lock(hudMutex_);
            if (hasDepth) {
                sceneTargets_.insert(target);
                hudLayers_.erase(target);
            } else {
                if (display)
                    for (int i = 0; i < pictures; ++i)
                        if (!sceneTargets_.count(picture[i]) && hudLayers_.insert(picture[i]).second)
                            std::fprintf(stderr, "[hud] 0x%" PRIx64 " taken for the HUD's buffer\n", picture[i]);
                layer = hudLayers_.count(target) != 0;
            }
        }
        // One untextured full-screen-style rectangle (RECTLIST, 3 vertices): a fill, tint or fade
        // over the whole picture. Squeezed, it left a seam at the 16:9 edges.
        bool textured = false;
        for (const gcn::Program* p : {&vs, ps})
            if (p)
                for (const auto& t : p->images) textured = textured || !t.storage;
        const bool fill = prim == 17 && d.count == 3 && !textured;
        const bool isHud = !hasDepth && (display || layer) && sceneW == 0 && !fill;
        uint8_t mode = !isHud ? 0 : hudWide(vs, ps, d) ? 2 : 1;
        if (!hasDepth && sceneW && prim == 17 && sceneCopy16x9(vs)) mode = 3;
        if (hudLog_) {
            std::lock_guard lock(hudLogMutex_);
            const uint64_t vsAddr = vs.launch.entry, psAddr = ps ? ps->launch.entry : 0;
            if (hudLogged_.insert({target ^ vsAddr, psAddr, (hasDepth ? 4 : 0) | mode}).second &&
                hudLogged_.size() <= 600) {
                std::string samples;
                for (int i = 0; i < pictures; ++i) samples += " 0x" + hex(picture[i]);
                std::fprintf(stderr, "[hud] frame %" PRIu64 " into 0x%" PRIx64 "%s%s%s: vs 0x%" PRIx64 " ps 0x%" PRIx64
                                     " prim %u count %u vte 0x%x: %s%s\n",
                             frame_, target, display ? " (display)" : "", layer ? " (HUD buffer)" : "",
                             hasDepth ? " with depth" : "", vsAddr, psAddr, prim, d.count, r.ctx[reg::kPaClVteCntl],
                             mode == 3   ? "scene copy laid out 16:9, height fitted"
                             : mode == 2 ? "HUD backdrop, full width"
                             : isHud     ? "HUD, 16:9 in the middle"
                                         : "full picture",
                             samples.empty() ? "" : ("; samples" + samples).c_str());
            }
        }
        return mode;
    }
    // A HUD draw covering the HUD's whole layout from side to side: the game's vignette, its top and
    // bottom shading, the pause screen's dimming, the title's backdrop. Squeezed into 16:9 they left
    // bands at the sides (21:9); they are stretched across the whole picture instead.
    // Textured ones (an image, not a gradient) only when they cover the layout's height too and are
    // few rectangles: the title menu's animated highlights, reaching across it now and then, jumped
    // between the two mappings.
    static bool hudWide(const gcn::Program& vs, const gcn::Program* ps, const DrawCall& d) {
        if (d.count > 256 || vs.buffers.empty()) return false;
        bool textured = false;
        for (const gcn::Program* p : {&vs, ps})
            if (p)
                for (const auto& t : p->images) textured = textured || !t.storage;
        if (textured && d.count > 32) return false;
        // The layout's width, from the draw data's (2/w, -2/h, -1, 1); else the game's 1280x720.
        auto at = [&](size_t i) {
            float v;
            std::memcpy(&v, &vs.data[i], 4);
            return v;
        };
        float layoutW = 0, layoutH = 0;
        for (size_t i = gcn::data::smrd(); i + 4 <= vs.data.size(); ++i)
            if (at(i) > 0 && at(i + 1) < 0 && at(i + 2) == -1.0f && at(i + 3) == 1.0f) {
                layoutW = 2 / at(i), layoutH = -2 / at(i + 1);
                break;
            }
        if (layoutW < 256) layoutW = 1280, layoutH = 720;
        const auto& b = vs.buffers[0]; // the positions (x, y first)
        const uint32_t n = b.numRecords(), stride = b.stride();
        if (!b.known || stride < 8 || !n || n > 256) return false;
        std::vector<uint8_t> bytes(size_t(stride) * n);
        if (!readGuest(b.base(), bytes.data(), bytes.size())) return false;
        float lo = INFINITY, hi = -INFINITY, top = INFINITY, bottom = -INFINITY;
        for (uint32_t i = 0; i < n; ++i) {
            float p[2];
            std::memcpy(p, &bytes[size_t(i) * stride], 8);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1])) return false;
            lo = std::min(lo, p[0]), hi = std::max(hi, p[0]), top = std::min(top, p[1]), bottom = std::max(bottom, p[1]);
        }
        if (textured && !(top < 2.5f && bottom > layoutH - 2.5f)) return false;
        return lo < 2.5f && hi > layoutW - 2.5f && hi - lo < 4 * layoutW;
    }
    // A copy of the scene laid out as 16:9 at the picture's width, from the top: one rectangle
    // (RECTLIST) from (0, 0) to (W, W * 9 / 16) in pixels. The pause screen's background is drawn
    // so: at 21:9 it ran past the bottom, the scene zoomed in and cut off at the top and bottom (#69).
    // Its height is fitted to the picture's (hudDraw's mode 3).
    static bool sceneCopy16x9(const gcn::Program& vs) {
        const Hud& h = hud();
        const double layoutH = h.outW * 9.0 / 16;
        if (std::fabs(layoutH - h.outH) < 1 || vs.buffers.empty()) return false;
        const auto& b = vs.buffers[0]; // the positions (x, y first)
        const uint32_t n = b.numRecords(), stride = b.stride();
        if (!b.known || stride < 8 || n != 3) return false;
        uint8_t bytes[3 * 64];
        if (stride > 64 || !readGuest(b.base(), bytes, size_t(stride) * n)) return false;
        float x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
        for (uint32_t i = 0; i < n; ++i) {
            float p[2];
            std::memcpy(p, &bytes[size_t(i) * stride], 8);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1])) return false;
            x0 = std::min(x0, p[0]), x1 = std::max(x1, p[0]), y0 = std::min(y0, p[1]), y1 = std::max(y1, p[1]);
        }
        return std::fabs(x0) < 0.5f && std::fabs(y0) < 0.5f && std::fabs(x1 - float(h.outW)) < 1.5f &&
               std::fabs(y1 - float(layoutH)) < 1.5f;
    }
    std::mutex hudMutex_;
    std::unordered_set<uint64_t> sceneTargets_, hudLayers_;
    const bool hudLog_ = getenv("LPN_HUD_LOG") != nullptr;
    std::mutex hudLogMutex_;
    std::set<std::tuple<uint64_t, uint64_t, int>> hudLogged_;

    // The game's viewport (x, y, width and height; width and height signed).
    static VkViewport gameViewport(const DrawRegs& r, uint32_t width, uint32_t height) {
        const float* v = reinterpret_cast<const float*>(&r.ctx[reg::kVportXScale]);
        float xs = v[0], xo = v[1], ys = v[2], yo = v[3];
        const uint32_t vte = r.ctx[reg::kPaClVteCntl];
        if (!(vte & 1)) xs = 1;
        if (!(vte & 2)) xo = 0;
        if (!(vte & 4)) ys = 1;
        if (!(vte & 8)) yo = 0;
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
        return vp;
    }
    // Aspect Ratio: a HUD draw moved into the 16:9 rectangle (hudDraw's mode 1; mode 2 only its
    // height) by its vertex shader's position transform, the viewport left as the game set it: a
    // smaller viewport cut off what reached past the rectangle (the minimap's ping).
    static void hudTransform(gcn::Program& vs, const DrawRegs& r, uint32_t width, uint32_t height, uint8_t mode) {
        const Hud& h = hud();
        const VkViewport vp = gameViewport(r, width, height);
        float t[4];
        std::memcpy(t, &vs.data[gcn::data::kPosTransform], sizeof t);
        // pixel' = offset + pixel * scale, as clip space for the same viewport: x * scale + o * w.
        auto fit = [](float& scale, float& offset, float s, float o, float vx, float vw) {
            if (vw == 0) return;
            const float add = s - 1 + 2 * (o + (s - 1) * vx) / vw;
            scale *= s, offset = offset * s + add;
        };
        if (mode == 3) { // sceneCopy16x9: from the top, the 16:9 layout's height to the picture's
            fit(t[1], t[3], float(h.outH / (h.outW * 9.0 / 16)), 0, vp.y, vp.height);
        } else {
            if (mode == 1) fit(t[0], t[2], h.sx, h.x, vp.x, vp.width);
            fit(t[1], t[3], h.sy, h.y, vp.y, vp.height);
        }
        std::memcpy(&vs.data[gcn::data::kPosTransform], t, sizeof t);
    }

    void setViewportScissor(const DrawRegs& r, uint32_t width, uint32_t height, bool jitter = false, uint8_t hudMode = 0) {
        const float* v = reinterpret_cast<const float*>(&r.ctx[reg::kVportXScale]);
        const uint32_t vte = r.ctx[reg::kPaClVteCntl];
        const float zs = (vte & 16) ? v[4] : 1, zo = (vte & 32) ? v[5] : 0;
        VkViewport vp = gameViewport(r, width, height);
        const Hud& h = hud();
        const bool dx = (r.ctx[reg::kPaClClipCntl] >> 19) & 1;
        vp.minDepth = dx ? zo : zo - zs;
        vp.maxDepth = zo + zs;
        vp.minDepth = std::clamp(vp.minDepth, 0.0f, 1.0f);
        vp.maxDepth = std::clamp(vp.maxDepth, 0.0f, 1.0f);
        if (vp.width < 0) {
            vp.x += vp.width;
            vp.width = -vp.width;
        }
        if (jitter) vp.x += jitterX_, vp.y += jitterY_;
        if (!R().bound_.state || std::memcmp(&R().bound_.viewport, &vp, sizeof vp) != 0) rec([c = R().cmd_, vp] { vkCmdSetViewport(c, 0, 1, &vp); }), R().bound_.viewport = vp;

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
        // The HUD's clipping (the minimap's, say) moves with it. Clipping to the whole picture keeps
        // its height and becomes the 16:9 rectangle with a margin, for glows reaching a little past
        // it (the minimap's ping) but not what the game slides in from off screen.
        const bool whole = x0 == 0 && y0 == 0 && x1 >= int32_t(h.outW) && y1 >= int32_t(h.outH);
        if (hudMode == 1 && whole) {
            const float w = h.outW * h.sx, margin = w * 0.06f;
            x0 = std::max<int32_t>(0, int32_t(std::floor(h.x - margin)));
            x1 = std::min<int32_t>(int32_t(width), int32_t(std::ceil(h.x + w + margin)));
        } else if ((hudMode == 1 || hudMode == 2) && !whole) {
            if (hudMode == 1) x0 = int32_t(std::floor(h.x + x0 * h.sx)), x1 = int32_t(std::ceil(h.x + x1 * h.sx));
            y0 = int32_t(std::floor(h.y + y0 * h.sy)), y1 = int32_t(std::ceil(h.y + y1 * h.sy));
            x1 = std::min<int32_t>(x1, int32_t(width)), y1 = std::min<int32_t>(y1, int32_t(height));
        }
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
        if (!R().bound_.state || std::memcmp(&R().bound_.scissor, &sc, sizeof sc) != 0) rec([c = R().cmd_, sc] { vkCmdSetScissor(c, 0, 1, &sc); }), R().bound_.scissor = sc;
    }

    Vk vk_;
    std::shared_mutex m_; // the renderer; recorder threads' fast draws share it for their lookups
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
    bool sync_ = false;
    // The command thread's queue (see CmdChunk): chunks handed over in order, and spare ones.
    bool syncRecord_ = getenv("LPN_SYNC_RECORD") != nullptr;
    bool deferUploads_ = getenv("LPN_NO_DEFER_UPLOADS") == nullptr; // see uploadData
    std::mutex chunkMutex_;
    std::condition_variable chunkCv_;
    std::deque<std::unique_ptr<CmdChunk>> chunks_, spareChunks_;
    VkBuffer dummyBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyMemory_ = VK_NULL_HANDLE;
    VkDeviceSize storageAlign_ = 16;
    std::vector<uint8_t> scratch_;
    GuestHasher hasher_;
    std::chrono::steady_clock::time_point lastFrameEnd_ = std::chrono::steady_clock::now();
    uint64_t windowFrames_ = 0, slowFrames_ = 0; // LPN_PROFILE: frames in the report window, and those over 17 ms
    double worstFrameMs_ = 0;
    uint64_t frameNs_[kProfCount] = {}, frameAnalyses_ = 0, framePipelines_ = 0;
    uint64_t windowSyncImports_ = 0, windowSyncImportNs_ = 0, windowAheadImports_ = 0, windowAheadImportNs_ = 0;

    std::map<uint64_t, Import> imports_; // never erased while rendering: importHits_ points into it
    static constexpr size_t kImportHits = 64;
    std::unordered_map<uint64_t, Surface> colors_, depths_;
    std::unordered_map<uint64_t, Metadata> metadata_; // by HTILE/CMASK address
    std::unordered_map<std::array<uint32_t, 6>, GuestTexture, WordsHash> textures_;
    static constexpr uint32_t kMaxMips = 15;
    const bool noMips_ = getenv("LPN_NO_MIPS") != nullptr;
    std::map<std::array<uint32_t, 6>, VkSampler> samplers_;
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
    Overlays overlays_;
    uint64_t overlayIds_ = 0;
    const bool noFastDraws_ = getenv("LPN_NO_FAST_DRAWS") != nullptr;
    const bool noDepthJoin_ = getenv("LPN_NO_DEPTH_JOIN") != nullptr; // depth-only draws get their own pass
    Profiler aprof_; // the analysis thread's slots and counts, merged into R().prof_'s reports
    std::mutex analysesMutex_;
    std::condition_variable analysesCv_; // a DrawAnalysis is ready
    std::deque<std::unique_ptr<DrawAnalysis>> analyses_; // one per draw or dispatch, for the recorder
    std::vector<std::unique_ptr<DrawAnalysis>> spareAnalyses_; // used ones, back from the recorder
    static constexpr size_t kTakeRun = 64;
    std::vector<std::unique_ptr<DrawAnalysis>> taken_; // the recorder's: ready ones taken
    size_t takenNext_ = 0;
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
        uint32_t formats[8] = {}; // VkFormat, as in GraphicsState
        uint32_t colorCount = 0, width = 0, height = 0;
        bool stencil = false;
        std::vector<VkImage> images; // attachments
        VkImage depthImage = VK_NULL_HANDLE;
        std::vector<VkImage> written; // attachments a draw in the pass has written
        bool sameTargets(const Pass& o) const {
            return colorCount == o.colorCount && depth == o.depth && stencil == o.stencil && width == o.width &&
                   height == o.height && std::equal(views, views + colorCount, o.views);
        }
    };
    std::array<ImageMemo, kImageMemos> imageMemos_{};
    uint64_t imageGen_ = 1;
    int anisoSeen_ = 0; // the Anisotropic Filtering setting the bindings were made with
    std::string spirvDir_; // "-": no SPIR-V cache
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::string pipelineCachePath_;
    std::atomic<uint32_t> newPipelines_{0};
    bool syncPipelines_ = false;
    bool libraries_ = false; // pipelines linked from libraries (see library())
    std::mutex setPoolMutex_; // setPool_, retiredPools_
    VkDescriptorPool setPool_ = VK_NULL_HANDLE;
    std::deque<std::pair<uint64_t, VkDescriptorPool>> retiredPools_; // reusable once completed_ reaches the first
    std::mutex buildMutex_, listMutex_; // built_ and jobs_; pipelines.list
    std::condition_variable buildCv_;
    std::unordered_map<std::string, std::shared_ptr<BuiltPipeline>> built_; // by pipelineRecord
    std::deque<PipelineJob> jobs_, backgroundJobs_;
    std::mutex layoutMutex_; // layouts_ and expanders_: the recorder, the analysis thread (prefetch) and startup builds
    std::string pipelineListPath_;
    FILE* pipelineList_ = nullptr;
    std::chrono::steady_clock::time_point lastPipelineSave_{};
    std::atomic<bool> pipelineSaving_{false};

    // What one thread recording draws keeps from draw to draw: its batch (command buffer, upload
    // ring, command chunk), the open pass, what is bound, the fast-draw states and its counters.
    // Each recorder thread has its own (see R()); the others use the main one.
    struct Rec {
        uint32_t cur_ = 0; // the batch
        VkCommandBuffer cmd_ = VK_NULL_HANDLE; // the current batch's
        VkBuffer ring_ = VK_NULL_HANDLE;
        void* ringPtr_ = nullptr;
        VkDeviceSize ringUsed_ = 0;
        bool recording_ = false;
        uint32_t drawsSinceFlush_ = 0, drawsSinceHandOff_ = 0;
        std::unique_ptr<CmdChunk> chunk_ = std::make_unique<CmdChunk>(); // being filled
        uint64_t recCount_ = 0; // calls rec()'d (see fastDraw)
        std::vector<std::function<void()>> pendingDone_; // due after the batch being recorded
        std::vector<std::function<void()>> afterFlush_;
        Pass pass_, drawPass_; // the open pass; the draw being recorded's
        PassWhy passClosedBy_ = kPassOther;
        uint64_t passSerial_ = 0;
        // Since the last barrier: images draws wrote and sampled, and whether anything else (a copy,
        // dispatch, upload, clear, shader store or new batch) needs one before the next pass.
        std::vector<VkImage> drawWritten_, drawSampled_;
        bool needBarrier_ = true;
        std::vector<VkImage> sampledSurfaces_;
        Bound bound_;
        std::vector<VkDescriptorSetLayoutBinding> drawBindings_;
        std::vector<VkWriteDescriptorSet> drawWrites_;
        std::vector<VkDescriptorBufferInfo> drawBufs_;
        std::vector<VkDescriptorImageInfo> drawImgs_;
        bool lastMergeable_ = false; // the last draw was a fast draw that only changed draw data
        FastDraw fast_;
        std::unique_ptr<DrawAnalysis> prevDraw_ = std::make_unique<DrawAnalysis>(); // the last draw recorded in full, if fast_.valid
        bool keepDraw_ = false; // onDraw keeps the draw as prevDraw_
        uint64_t prevSeq_ = 0, recordedSeq_ = 0; // DrawAnalysis::seq of prevDraw_'s draw / the last fast one
        std::array<FastSlot, kFastSlots> fastSlots_;
        std::array<SlotKey, kFastSlots> slotKeys_{}; // fastSlots_[i].f's shaders
        std::array<uint64_t, kFastSlots> slotUsed_{}; // fastSlots_[i].used, for the LRU scan
        uint64_t fastUse_ = 0;
        GuestTexture* lastTexture_ = nullptr; // the one resolveImageUncached bound, if any
        std::vector<std::unique_ptr<DrawAnalysis>> returned_; // used analyses to hand back
        struct {
            uint64_t draws = 0, dispatches = 0;
        } stats_;
        Profiler prof_;
        uint64_t countedPass_ = ~0ull, passDraws_ = 0; // LPN_PROFILE: the pass being counted, and its draws so far
        VkDeviceSize ringSize_ = kRingSize;
        // A recorder thread's (or the uploads ahead of them): Vulkan calls are made at once (not
        // queued for the command thread), into a secondary command buffer; it never flushes.
        bool immediate_ = false, worker_ = false;
        bool locked_ = false, lockShared_ = false; // holds m_, shared (see WLock)
        bool abort_ = false; // a full draw under the shared lock met a cache it would change (see sharedMiss)
        std::array<VramHit, kVramHits> vramHits_{}; // the copies last found (see vramBuffer); cleared when one is freed
        std::array<const Import*, kImportHits> importHits_{}; // the imports last found (see guestBuffer)
    };
    Rec rec0_;
    static inline thread_local Rec* tRec_ = nullptr;
    Rec& R() {
        Rec* r = tRec_;
        return r ? *r : rec0_;
    }
    const Rec& R() const {
        const Rec* r = tRec_;
        return r ? *r : rec0_;
    }

    // ---- recorder threads (LPN_RECORD_THREADS) -------------------------------------------------

    // At ~9,000 draws a frame (the city overlook) one recorder took ~15 ms a frame, most of it
    // waiting on memory, and capped the frame rate at ~55 while the GPU needed ~11 ms. With
    // LPN_RECORD_THREADS=N, runs of up to LPN_RECORD_SEGMENT draws (128) go to N threads, each
    // recording its run into a secondary command buffer of its own from a fresh state (its own
    // pass, bindings and fast-draw slots, starting behind a full barrier); the main recorder
    // queues an execution of each in order in its batch, which the command thread makes once the
    // run is recorded. Everything else (dispatches, DMA, memory writes, SMAA under DLSS, frame
    // ends) waits for the runs recorded so far and is recorded in order as before.
    // The caches are shared under m_: a thread holds it for a whole full draw and around a fast
    // draw's lookups. Uploads and copies a thread records go into a command buffer of their own
    // that runs before the next run is executed (none recorded earlier can miss them), and what
    // a thread frees waits for every run recorded by then.
    struct SegSlot {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkBuffer ring = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        void* ptr = nullptr;
        VkDeviceSize size = 0;
    };
    struct SegItem {
        DrawCall d;
        DrawRegs r;
        std::unique_ptr<DrawAnalysis> a;
    };
    struct Segment {
        SegSlot* slot = nullptr;
        std::vector<SegItem> items;
        std::vector<std::unique_ptr<uint32_t[]>> ctxs; // copies of the context registers the items read
        size_t ctxUsed = 0;
        bool done = false; // recorded (segMutex_)
    };
    static constexpr VkDeviceSize kSegRing = 4ull << 20, kProRing = 48ull << 20;

    std::unique_ptr<SegSlot> makeSlot(VkDeviceSize ringSize) {
        auto sl = std::make_unique<SegSlot>();
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = vk_.family;
        if (vkCreateCommandPool(vk_.device, &pci, nullptr, &sl->pool) != VK_SUCCESS) return nullptr;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = sl->pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vk_.device, &cai, &sl->cmd) != VK_SUCCESS) return nullptr;
        sl->ring = createHostBuffer(ringSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &sl->mem, &sl->ptr);
        if (!sl->ring) return nullptr;
        sl->size = ringSize;
        return sl;
    }
    void beginSecondary(SegSlot& sl) {
        VK_CHECK(vkResetCommandPool(vk_.device, sl.pool, 0));
        VkCommandBufferInheritanceInfo ii{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        bi.pInheritanceInfo = &ii;
        VK_CHECK(vkBeginCommandBuffer(sl.cmd, &bi));
    }
    // A Rec starting on a slot's command buffer: nothing bound, no pass, no fast states.
    static void startRec(Rec& rc, SegSlot& sl) {
        rc.cmd_ = sl.cmd;
        rc.ring_ = sl.ring;
        rc.ringPtr_ = sl.ptr;
        rc.ringUsed_ = 0;
        rc.ringSize_ = sl.size;
        rc.recording_ = true;
        rc.bound_ = Bound{};
        rc.needBarrier_ = true;
        rc.drawWritten_.clear();
        rc.drawSampled_.clear();
        rc.pass_.open = false;
        ++rc.passSerial_; // fast states from an earlier segment never match
        rc.fast_.valid = false;
        for (FastSlot& f : rc.fastSlots_) f.f.valid = false;
        rc.lastMergeable_ = false;
        rc.keepDraw_ = false;
        rc.prevSeq_ = rc.recordedSeq_ = 0;
    }

    void startRecorderThreads() {
        const char* n = getenv("LPN_RECORD_THREADS");
        // 6 on 16+ hardware threads (=0 turns them off): the overlook went from ~55 to ~76 fps on
        // an i7-12700K, where 4, 6 and 8 did alike before the lock was shared.
        const unsigned want = n ? unsigned(std::atoi(n)) : std::thread::hardware_concurrency() >= 16 ? 6 : 0;
        if (want < 1) return;
        const char* why = syncRecord_ ? "LPN_SYNC_RECORD" : sync_ ? "LPN_GPU_SYNC" : traceFrame_ != ~0ull ? "LPN_GPU_TRACE" : snap_ ? "LPN_GPU_SNAP"
                          : raceCheck_ ? "LPN_RACE_CHECK" : computeOnly_ ? "compute only" : !analysisWorkers_ ? "no analysis workers"
                                                                                                           : nullptr;
        if (why) {
            if (n) std::fprintf(stderr, "[gpu] LPN_RECORD_THREADS ignored (%s)\n", why);
            return;
        }
        if (const char* sz = getenv("LPN_RECORD_SEGMENT")) segDraws_ = std::max(8, std::atoi(sz));
        recordThreads_ = std::min(want, 16u);
        pro_.worker_ = pro_.immediate_ = true;
        pro_.prof_.on = R().prof_.on;
        for (unsigned i = 0; i < recordThreads_; ++i) {
            workerRecs_.push_back(std::make_unique<Rec>());
            Rec& rc = *workerRecs_.back();
            rc.worker_ = rc.immediate_ = true;
            rc.prof_.on = R().prof_.on;
            std::thread([this, &rc] { recorderThread(rc); }).detach();
        }
        std::fprintf(stderr, "[gpu] %u recorder threads, runs of up to %u draws\n", recordThreads_, segDraws_);
    }
    void recorderThread(Rec& rc) {
        profile::nameThread("lpn-record-w");
        tRec_ = &rc;
        for (;;) {
            Segment* g;
            {
                std::unique_lock lock(segMutex_);
                segCv_.wait(lock, [&] { return !segQueue_.empty(); });
                g = segQueue_.front();
                segQueue_.pop_front();
            }
            recordSegment(rc, *g);
            {
                std::lock_guard lock(segMutex_);
                g->done = true;
                --outstanding_;
            }
            segDoneCv_.notify_all();
        }
    }
    void recordSegment(Rec& rc, Segment& g) {
        SegSlot& sl = *g.slot;
        startRec(rc, sl);
        beginSecondary(sl);
        {
            ProfScope ps(rc.prof_, kProfRenderer);
            for (SegItem& it : g.items) {
                Recycle recycle{*this, it.a};
                recordDraw(it.d, it.r, it.a);
            }
            endPass();
        }
        VK_CHECK(vkEndCommandBuffer(sl.cmd));
        rc.recording_ = false;
        ++rc.prof_.segments;
        g.items.clear();
        g.ctxUsed = 0;
        if (!rc.returned_.empty()) spare(rc.returned_);
        if (!rc.afterFlush_.empty() || !rc.pendingDone_.empty()) { // freed: once every segment recorded by now is done
            std::lock_guard lock(lateMutex_);
            for (auto& f : rc.afterFlush_) lateFree_.push_back(std::move(f));
            for (auto& f : rc.pendingDone_) lateFree_.push_back(std::move(f));
            rc.afterFlush_.clear();
            rc.pendingDone_.clear();
        }
    }

    bool segmentable(const DrawCall& d, const DrawAnalysis& a) const {
        if (d.kind != DrawCall::kAuto && d.kind != DrawCall::kIndexed) return false;
        if (computeOnly_ || frame_ == traceFrame_ || frame_ == timingFrame_) return false;
        return !(dlssOn_ && a.psm && a.psm->smaa); // SMAA under DLSS changes what later draws do
    }
    void addToSegment(const DrawCall& d, const GpuRegs& r, std::unique_ptr<DrawAnalysis>& held) {
        ProfScope ps(R().prof_, kProfRenderer);
        if (!open_) {
            std::lock_guard lock(segMutex_);
            if (segFree_.empty()) {
                segAll_.push_back(std::make_unique<Segment>());
                segFree_.push_back(segAll_.back().get());
            }
            open_ = segFree_.back();
            segFree_.pop_back();
            openCtx_ = nullptr;
        }
        Segment& g = *open_;
        if (!openCtx_ || r.ctxSerial != openCtxSerial_) { // a copy of the context registers, while they hold
            if (g.ctxUsed == g.ctxs.size()) g.ctxs.push_back(std::make_unique<uint32_t[]>(::gpu::kContextRegs));
            uint32_t* c = g.ctxs[g.ctxUsed++].get();
            std::memcpy(c, r.ctx, sizeof r.ctx);
            openCtx_ = c;
            openCtxSerial_ = r.ctxSerial;
        }
        g.items.push_back({d, DrawRegs::of(r, openCtx_), std::move(held)});
        ++drawIndex_;
        if (g.items.size() >= segDraws_) dispatchSegment();
    }
    SegSlot* acquireSlot() {
        std::unique_lock lock(segMutex_);
        if (slotFree_.empty() && slotsAll_.size() < 256) {
            lock.unlock();
            auto sl = makeSlot(kSegRing);
            lock.lock();
            if (sl) {
                slotsAll_.push_back(std::move(sl));
                slotFree_.push_back(slotsAll_.back().get());
            }
        }
        if (slotFree_.empty()) {
            ProfScope pw(R().prof_, kProfGpuWait);
            segFreeCv_.wait(lock, [&] { return !slotFree_.empty(); });
        }
        SegSlot* sl = slotFree_.back();
        slotFree_.pop_back();
        return sl;
    }
    // The open segment to a recorder thread, its execution queued in the batch in order.
    void dispatchSegment() {
        Segment* g = open_;
        open_ = nullptr;
        openCtx_ = nullptr;
        if (!g) return;
        if (g->items.empty()) {
            std::lock_guard lock(segMutex_);
            segFree_.push_back(g);
            return;
        }
        g->slot = acquireSlot();
        const size_t n = g->items.size();
        {
            // At most a few batches ahead of the threads: the main recorder waits for a batch two
            // flushes back while holding m_ (begin), and that batch's segments must be recorded.
            const unsigned ahead = std::max(2u, flushDraws_ ? (kBatches - 2) * flushDraws_ / segDraws_ : 16u);
            std::unique_lock lock(segMutex_);
            if (outstanding_ >= ahead || paused_) {
                const auto t0 = std::chrono::steady_clock::now();
                segDoneCv_.wait(lock, [&] { return outstanding_ < ahead && !paused_; });
                R().prof_.aheadNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            }
            g->done = false;
            ++outstanding_;
            segQueue_.push_back(g);
        }
        segCv_.notify_one();
        std::lock_guard lock(m_);
        begin();
        endPass(); // secondaries run outside a render pass
        rec([this, c = R().cmd_, batch = R().cur_, g] { executeSegment(c, batch, *g); });
        R().afterFlush_.push_back([this, sl = g->slot] {
            {
                std::lock_guard lock(segMutex_);
                slotFree_.push_back(sl);
            }
            segFreeCv_.notify_one();
        });
        handOff();
        R().drawsSinceFlush_ += uint32_t(n);
        if (flushDraws_ && R().drawsSinceFlush_ >= flushDraws_) flush();
    }
    // On the command thread: once the segment is recorded, the uploads queued so far, then it.
    void executeSegment(VkCommandBuffer c, uint32_t batch, Segment& g) {
        {
            std::unique_lock lock(segMutex_);
            segDoneCv_.wait(lock, [&] { return g.done; });
        }
        executePrologues(c, batch);
        vkCmdExecuteCommands(c, 1, &g.slot->cmd);
        std::lock_guard lock(segMutex_);
        g.slot = nullptr;
        segFree_.push_back(&g);
    }
    // Waits until every segment so far is recorded: what comes next is recorded in order here.
    void drainSegments(int why) {
        dispatchSegment();
        std::unique_lock lock(segMutex_);
        if (!outstanding_) return;
        const auto t0 = std::chrono::steady_clock::now();
        segDoneCv_.wait(lock, [&] { return outstanding_ == 0; });
        R().prof_.drainNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        ++R().prof_.drains;
        ++R().prof_.drainWhy[why];
    }
    // Recorder threads' counters into the main ones (all threads idle: after drainSegments).
    void mergeRecorderProfiles() {
        auto take = [&](Rec& w) {
            Profiler& to = R().prof_;
            Profiler& p = w.prof_;
            to.workerNs += p.ns[kProfRenderer];
            p.ns[kProfRenderer] = 0;
            addProfile(to, p);
            R().stats_.draws += std::exchange(w.stats_.draws, 0);
            R().stats_.dispatches += std::exchange(w.stats_.dispatches, 0);
        };
        for (auto& w : workerRecs_) take(*w);
        std::lock_guard lock(proMutex_);
        take(pro_);
    }
    static void addProfile(Profiler& a, Profiler& b) {
        auto add = [](auto& x, auto& y) {
            for (size_t i = 0; i < std::size(x); ++i) x[i] += std::exchange(y[i], 0);
        };
        add(a.ns, b.ns), add(a.passWhy, b.passWhy), add(a.feedbackOn, b.feedbackOn), add(a.missWhy, b.missWhy);
        add(a.missUser, b.missUser), add(a.fastMiss, b.fastMiss), add(a.fastKind, b.fastKind), add(a.fastRes, b.fastRes);
        add(a.switchMiss, b.switchMiss), add(a.drainWhy, b.drainWhy), add(a.passBySize, b.passBySize), add(a.drawsBySize, b.drawsBySize);
        for (uint64_t Profiler::*f : {&Profiler::fastMoved, &Profiler::fastImages, &Profiler::diffsTaken, &Profiler::fastSwitches,
                                      &Profiler::mergeable, &Profiler::mergeRuns, &Profiler::depthJoins, &Profiler::depthOnly,
                                      &Profiler::barriers, &Profiler::calls, &Profiler::fastDraws, &Profiler::analysisHits,
                                      &Profiler::repeats, &Profiler::analyses, &Profiler::passes, &Profiler::texChecks,
                                      &Profiler::texUploads, &Profiler::texBytes, &Profiler::flushes, &Profiler::dmas,
                                      &Profiler::dmaBytes, &Profiler::pipelinesMade, &Profiler::pipelineWaits, &Profiler::vramCopies,
                                      &Profiler::vramCopyBytes, &Profiler::imageMisses, &Profiler::imageGens, &Profiler::resolvedAhead,
                                      &Profiler::vramAllocs, &Profiler::vramEvicts, &Profiler::vramFreed, &Profiler::vramEvictsEmpty,
                                      &Profiler::segments, &Profiler::drains, &Profiler::drainNs, &Profiler::workerNs, &Profiler::aheadNs, &Profiler::sharedDraws, &Profiler::sharedRetries, &Profiler::locks, &Profiler::lockWaits, &Profiler::lockNs})
            a.*f += std::exchange(b.*f, 0);
        a.largestPass = std::max(a.largestPass, std::exchange(b.largestPass, 0));
        for (auto& [k, v] : b.fastCtx) a.fastCtx[k] += v;
        b.fastCtx.clear();
    }

    // Uploads and copies a recorder thread makes (texture uploads, VRAM copies, new images' first
    // layout and clear) go into the next secondary of pro_, executed before the next segment.
    void openPrologue() { // proMutex_ held
        SegSlot* sl = nullptr;
        if (!proFree_.empty()) {
            sl = proFree_.back();
            proFree_.pop_back();
        } else {
            auto made = makeSlot(kProRing);
            if (!made) {
                warnOnce("cannot create an upload command buffer for the recorder threads");
                return;
            }
            sl = made.get();
            proAll_.push_back(std::move(made));
        }
        proSlot_ = sl;
        startRec(pro_, *sl);
        beginSecondary(*sl);
    }
    void closePrologue() { // proMutex_ held
        if (!pro_.recording_) return;
        VK_CHECK(vkEndCommandBuffer(proSlot_->cmd));
        pro_.recording_ = false;
        proPending_.push_back(proSlot_);
        proSlot_ = nullptr;
    }
    void executePrologues(VkCommandBuffer c, uint32_t batch) {
        std::vector<SegSlot*> run;
        {
            std::lock_guard lock(proMutex_);
            closePrologue();
            run.swap(proPending_);
        }
        if (run.empty()) return;
        for (SegSlot* sl : run) vkCmdExecuteCommands(c, 1, &sl->cmd);
        std::lock_guard lock(doneMutex_);
        batches_[batch].done.push_back([this, run] {
            std::lock_guard lock(proMutex_);
            for (SegSlot* sl : run) proFree_.push_back(sl);
        });
    }
    // Around code that records an upload or copy: on a recorder thread, into pro_.
    class Prologue {
    public:
        explicit Prologue(Renderer& r) : r_(r) {
            Rec* t = tRec_;
            if (!t || !t->worker_ || t == &r.pro_) return;
            lock_ = std::unique_lock(r.proMutex_);
            saved_ = t;
            tRec_ = &r.pro_;
            r.begin();
        }
        ~Prologue() {
            if (saved_) tRec_ = saved_;
        }

    private:
        Renderer& r_;
        std::unique_lock<std::mutex> lock_;
        Rec* saved_ = nullptr;
    };
    // On a recorder thread: holds m_ (the caches), unless already held.
    class WLock {
    public:
        explicit WLock(Renderer& r, bool shared = false) : shared_(shared) {
            Rec& x = r.R();
            if (!x.worker_ || x.locked_) return;
            m_ = &r.m_;
            // Held for microseconds at a time: spin first (a futex sleep and wake-up per handoff
            // cost more than the work it guards), counting the wait with LPN_PROFILE.
            ++x.prof_.locks;
            auto tryLock = [&] { return shared_ ? m_->try_lock_shared() : m_->try_lock(); };
            if (!tryLock()) {
                const auto t0 = x.prof_.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                int spins = 0;
                while (!tryLock())
                    if (++spins < 20000) __builtin_ia32_pause();
                    else {
                        shared_ ? m_->lock_shared() : m_->lock();
                        break;
                    }
                ++x.prof_.lockWaits;
                if (x.prof_.on)
                    x.prof_.lockNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            }
            x.locked_ = true;
            x.lockShared_ = shared_;
            rec_ = &x;
        }
        void unlock() {
            if (!rec_) return;
            rec_->locked_ = rec_->lockShared_ = false;
            shared_ ? m_->unlock_shared() : m_->unlock();
            rec_ = nullptr;
        }
        ~WLock() { unlock(); }
        bool shared() const { return rec_ && shared_; }

    private:
        bool shared_;
        std::shared_mutex* m_ = nullptr;
        Rec* rec_ = nullptr;
    };

    unsigned recordThreads_ = 0; // LPN_RECORD_THREADS
    uint32_t segDraws_ = 128;
    std::vector<std::unique_ptr<Rec>> workerRecs_;
    std::mutex segMutex_; // the queue, free lists, outstanding_ and Segment::done
    std::condition_variable segCv_, segDoneCv_, segFreeCv_;
    std::deque<Segment*> segQueue_;
    std::vector<std::unique_ptr<Segment>> segAll_;
    std::vector<Segment*> segFree_;
    std::vector<std::unique_ptr<SegSlot>> slotsAll_;
    std::vector<SegSlot*> slotFree_;
    unsigned outstanding_ = 0; // dispatched, not yet recorded
    unsigned paused_ = 0;      // threads waiting on the GPU with m_ held (see pauseSegments)
    Segment* open_ = nullptr;  // being filled (the main recorder's)
    const uint32_t* openCtx_ = nullptr;
    uint64_t openCtxSerial_ = 0;
    Rec pro_;             // uploads ahead of the segments
    std::mutex proMutex_; // pro_, proSlot_, proFree_, proPending_ (after m_)
    SegSlot* proSlot_ = nullptr;
    std::vector<SegSlot*> proFree_, proPending_;
    std::vector<std::unique_ptr<SegSlot>> proAll_;
    std::mutex lateMutex_;
    std::vector<std::function<void()>> lateFree_; // freed by recorder threads: after the next flush
};

Renderer* gRenderer = nullptr;

} // namespace

// LPN_DUMP_SHADER_TABLES=<flip> (stage 4 of docs/shader-precompile.md): once that many frames
// have been shown, writes the engine's shader tables to lpn-shader-tables.txt: per table its
// entry count and, per entry, the shader object's first 0x100 bytes and the code addresses found
// in it (dwords that, shifted left by 8, point at code followed by an OrbShdr block).
void dumpShaderTables(uint64_t flip) {
    while (flipCount() < flip) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    Module* eboot = process().eboot;
    if (!eboot) return;
    const uint64_t base = reinterpret_cast<uint64_t>(eboot->base());
    FILE* f = std::fopen("lpn-shader-tables.txt", "w");
    if (!f) return;
    struct Table {
        const char* name;
        uint64_t ptr, count;
    };
    const Table tables[] = {{"ps", 0x2698148, 0x269815c}, {"vs", 0x2698168, 0x269817c}, {"t3", 0x2698188, 0x269819c},
                            {"t4", 0x26981a8, 0x26981bc}, {"t5", 0x26981c8, 0x26981dc}, {"cs", 0x26981e8, 0x26981fc}};
    size_t withCode = 0, entries = 0;
    for (const Table& t : tables) {
        uint64_t arr = 0;
        uint32_t n = 0;
        uint8_t head[0x20] = {};
        readGuest(base + t.ptr, &arr, 8);
        readGuest(base + t.count, &n, 4);
        readGuest(base + t.ptr, head, sizeof head);
        std::fprintf(f, "table %s at +0x%" PRIx64 ": array 0x%" PRIx64 " count %u; header", t.name, t.ptr, arr, n);
        for (uint8_t b : head) std::fprintf(f, " %02x", b);
        std::fprintf(f, "\n");
        if (n > 100000) continue;
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t obj = 0;
            if (!readGuest(arr + i * 8ull, &obj, 8) || !obj) continue;
            ++entries;
            uint32_t words[64] = {};
            readGuest(obj, words, sizeof words);
            std::fprintf(f, "%s[%u] 0x%" PRIx64 ":", t.name, i, obj);
            for (int w = 0; w < 64; ++w) std::fprintf(f, "%s%08x", w % 8 ? " " : "\n  ", words[w]);
            bool found = false;
            // Code addresses: in the object itself, or one pointer away (a Gnmx shader it points at).
            auto scan = [&](const uint32_t* ws, int count, const char* where) {
                for (int w = 0; w < count; ++w) {
                    const uint64_t code = uint64_t(ws[w]) << 8;
                    if (code < 0x100000 || code >= (1ull << 40)) continue;
                    uint32_t first = 0;
                    if (!readGuest(code, &first, 4)) continue;
                    std::vector<uint8_t> body(0x10000);
                    size_t got = 0;
                    while (got < body.size() && readGuest(code + got, body.data() + got, 0x1000)) got += 0x1000;
                    for (size_t o = 0; o + 7 <= got; o += 4)
                        if (!std::memcmp(body.data() + o, "OrbShdr", 7)) {
                            std::fprintf(f, "\n  code %s+0x%x: 0x%" PRIx64 " (%zu bytes)", where, w * 4, code, o);
                            found = true;
                            break;
                        }
                }
            };
            scan(words, 64, "obj");
            for (int w = 0; w + 1 < 64 && !found; w += 2) {
                const uint64_t p = words[w] | (uint64_t(words[w + 1]) << 32);
                uint32_t inner[48] = {};
                if (p >= 0x100000 && p < (1ull << 40) && (p & 7) == 0 && readGuest(p, inner, sizeof inner)) {
                    char where[32];
                    std::snprintf(where, sizeof where, "*(obj+0x%x)", w * 4);
                    scan(inner, 48, where);
                }
            }
            withCode += found;
            std::fprintf(f, "\n");
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "[gpu] shader tables written to lpn-shader-tables.txt: %zu entries, %zu with code found\n", entries,
                 withCode);
}

bool startRenderer(bool computeOnly) {
    if (const char* e = getenv("LPN_DUMP_SHADER_TABLES"); e && *e)
        std::thread([flip = std::strtoull(e, nullptr, 10)] { dumpShaderTables(flip); }).detach();
    auto* r = new Renderer;
    r->computeOnly_ = computeOnly;
    if (!r->init()) {
        delete r;
        return false;
    }
    gRenderer = r;
    addDrawListener(r);
    r->waitForWarmup();
    return true;
}

bool warmupProgress(WarmupProgress& p) { return gRenderer && gRenderer->warmupProgress(p); }

bool bootCompileProgress(size_t& done, size_t& total) { return gRenderer && gRenderer->bootProgress(done, total); }

int maxAnisotropy() { return gRenderer ? gRenderer->maxAnisotropy() : 16; }

void skipWarmup() {
    if (gRenderer) gRenderer->skipWarmup();
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

void noteDirectMapping(uint64_t address, uint64_t size) {
    if (gRenderer) gRenderer->noteMapping(address, size);
}

void saveCaches() {
    if (gRenderer) gRenderer->savePipelineCacheNow();
}

} // namespace host::render
