// Vulkan renderer. Every draw and dispatch is recorded as it arrives, with full barriers
// between them, and a submission is flushed (submitted and waited for) when the game's own
// submission ends. Correctness first: batching and finer synchronisation come later.
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

#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
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
constexpr uint32_t kCbColorBase = 0x318, kCbColorStride = 0xF;
// sh window
constexpr uint32_t kPgmLoPs = 0x08, kRsrc1Ps = 0x0A, kRsrc2Ps = 0x0B, kUserDataPs = 0x0C;
constexpr uint32_t kPgmLoVs = 0x48, kRsrc1Vs = 0x4A, kRsrc2Vs = 0x4B, kUserDataVs = 0x4C;
constexpr uint32_t kNumThreadX = 0x207, kPgmLoCs = 0x20C, kRsrc1Cs = 0x212, kRsrc2Cs = 0x213, kUserDataCs = 0x240;
// uconfig window
constexpr uint32_t kPrimitiveType = 0x242;
} // namespace reg

constexpr uint32_t kPsBindingBase = 32;
constexpr VkDeviceSize kRingSize = 64ull << 20;
constexpr uint64_t kImportGranule = 2ull << 20;

bool readGuest(uint64_t addr, void* dst, size_t size) {
    if (!addr || addr >= (1ull << 40)) return false;
    iovec local{dst, size}, remote{reinterpret_cast<void*>(addr), size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t(size);
}

uint64_t shaderAddress(const GpuRegs& r, uint32_t lo) { return ((uint64_t(r.sh[lo + 1]) << 32) | r.sh[lo]) << 8; }

uint64_t hashBytes(const void* p, size_t n) {
    // FNV-1a over 8-byte words; good enough to notice a changed texture.
    const auto* w = static_cast<const uint64_t*>(p);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n / 8; ++i) h = (h ^ w[i]) * 0x100000001b3ull;
    return h;
}

std::string bytesKey(const void* p, size_t n) { return std::string(static_cast<const char*>(p), n); }

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
    std::map<std::string, VkImageView> views;
};

// A render target or depth buffer at a guest address.
struct Surface {
    Image img;
    uint32_t elementBytes = 4;
    bool depth = false;
};

struct GuestTexture {
    Image img;
    uint64_t hash = 0;
    uint64_t checkedFrame = ~0ull;
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
};

struct BufferRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, range = 0;
    uint32_t byteOffset = 0; // guest base relative to `offset`
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
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(vk_.device, &cai, &cmd_));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(vk_.device, &fci, nullptr, &fence_));

        ring_ = createHostBuffer(kRingSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 &ringMemory_, &ringPtr_);
        dummyBuffer_ = createHostBuffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dummyMemory_, nullptr);
        if (!ring_ || !dummyBuffer_) return false;
        storageAlign_ = std::max<VkDeviceSize>(vk_.props.limits.minStorageBufferOffsetAlignment, 16);
        if (const char* t = getenv("LPN_GPU_TRACE")) traceFrame_ = std::strtoull(t, nullptr, 0);
        snap_ = getenv("LPN_GPU_SNAP") != nullptr;
        return true;
    }

    void onDraw(const DrawCall& d, const GpuRegs& r) override {
        std::lock_guard lock(m_);
        ++drawIndex_;
        trace_ = frame_ == traceFrame_;
        if (trace_)
            std::fprintf(stderr, "[trace] #%" PRIu64 " %s vs 0x%" PRIx64 " ps 0x%" PRIx64 " cs 0x%" PRIx64 "\n", drawIndex_,
                         d.kind == DrawCall::kDispatch ? "dispatch" : "draw", shaderAddress(r, reg::kPgmLoVs),
                         shaderAddress(r, reg::kPgmLoPs), shaderAddress(r, reg::kPgmLoCs));
        if (d.kind == DrawCall::kDispatch) dispatch(d, r);
        else if (d.kind == DrawCall::kAuto || d.kind == DrawCall::kIndexed) draw(d, r);
        else warnOnce("indirect draws and dispatches are not rendered yet");
    }

    void onSubmitEnd() override {
        std::lock_guard lock(m_);
        flush();
    }

    void onFrameEnd() override {
        std::lock_guard lock(m_);
        flush();
        if (frame_ == traceFrame_) dumpSurfaces();
        ++frame_;
        drawIndex_ = 0;
    }

    bool readDisplay(uint64_t addr, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels) {
        std::lock_guard lock(m_);
        return readSurface(addr, width, height, pixels);
    }

private:
    // Debugging: a render target as it is after the current draw.
    void snapshot(uint64_t addr) {
        auto it = colors_.find(addr);
        if (it == colors_.end() || it->second.elementBytes != 4) return;
        std::vector<uint32_t> px;
        const uint32_t w = it->second.img.width, h = it->second.img.height;
        if (!readSurface(addr, w, h, px)) return;
        const std::string name = "gpu-draw-" + std::to_string(drawIndex_) + ".ppm";
        if (FILE* f = std::fopen(name.c_str(), "wb")) {
            std::fprintf(f, "P6\n%u %u\n255\n", w, h);
            for (uint32_t p : px) {
                const uint8_t rgb[3] = {uint8_t(p), uint8_t(p >> 8), uint8_t(p >> 16)};
                std::fwrite(rgb, 1, 3, f);
            }
            std::fclose(f);
        }
    }

    // Debugging: every 32-bit render target as gpu-rt-<address>.ppm (memory order BGRA assumed).
    void dumpSurfaces() {
        for (const auto& [addr, s] : colors_) {
            if (s.elementBytes != 4) continue;
            std::vector<uint32_t> px;
            if (!readSurface(addr, s.img.width, s.img.height, px)) continue;
            const std::string name = "gpu-rt-" + hex(addr) + ".ppm";
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
        c.bufferOffset = 0;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cmd_, s.img.image, VK_IMAGE_LAYOUT_GENERAL, ring_, 1, &c);
        ringUsed_ = bytes;
        flush();
        pixels.assign(size_t(width) * height, 0);
        for (uint32_t y = 0; y < h; ++y)
            std::memcpy(&pixels[size_t(y) * width], static_cast<const uint8_t*>(ringPtr_) + size_t(y) * w * 4, w * 4);
        return true;
    }

    // ---- recording -----------------------------------------------------------------------------

    void begin() {
        if (recording_) return;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
        recording_ = true;
    }

    void flush() {
        if (!recording_) return;
        VK_CHECK(vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VK_CHECK(vkQueueSubmit(vk_.queue, 1, &si, fence_));
        VK_CHECK(vkWaitForFences(vk_.device, 1, &fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vk_.device, 1, &fence_));
        VK_CHECK(vkResetCommandBuffer(cmd_, 0));
        recording_ = false;
        ringUsed_ = 0;
        for (auto& f : afterFlush_) f();
        afterFlush_.clear();
    }

    void fullBarrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
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
        if (warned_.insert(w).second) std::fprintf(stderr, "[gpu] %s\n", w.c_str());
    }

    // ---- memory --------------------------------------------------------------------------------

    VkBuffer createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory* mem, void** map) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        VkBuffer b;
        if (vkCreateBuffer(vk_.device, &bci, nullptr, &b) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(vk_.device, b, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex =
            vk_.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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
        for (const auto& [start, im] : imports_)
            if (start <= addr && im.end >= end) {
                buffer = im.buffer;
                offset = addr - start;
                return true;
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
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = img.image;
        ib.subresourceRange = {img.aspect, 0, 1, 0, ici.arrayLayers};
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &ib);
        // Clear so never-written surfaces read as black rather than garbage.
        const VkImageSubresourceRange range{img.aspect, 0, 1, 0, ici.arrayLayers};
        if (depth) {
            VkClearDepthStencilValue v{1.0f, 0};
            if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                vkCmdClearDepthStencilImage(cmd_, img.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range);
        } else if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
            VkClearColorValue v{};
            vkCmdClearColorImage(cmd_, img.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range);
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
        const uint32_t key[] = {uint32_t(format), uint32_t(type), uint32_t(swz.r), uint32_t(swz.g), uint32_t(swz.b),
                                uint32_t(swz.a), aspect};
        const std::string k = bytesKey(key, sizeof(key));
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
        m->module = module(m->translation.glsl, stage);
        m->id = uint32_t(shaders_.size() + 1);
        return (shaders_[key] = std::move(m)).get();
    }

    VkShaderModule module(const std::string& glsl, gcn::GlslStage stage) {
        std::string error;
        const auto spirv = gcn::compileGlsl(glsl, stage, &error);
        if (spirv.empty()) {
            std::fprintf(stderr, "[gpu] shader compile failed:\n%s\n--- source ---\n%s\n", error.c_str(), glsl.c_str());
            return VK_NULL_HANDLE;
        }
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spirv.size() * 4;
        ci.pCode = spirv.data();
        VkShaderModule m = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(vk_.device, &ci, nullptr, &m));
        return m;
    }

    static std::string hex(uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "%" PRIx64, v);
        return b;
    }

    // Geometry shader turning each primitive of a RECTLIST (3 corners) or QUADLIST into a strip.
    VkShaderModule expandShader(bool quads, uint32_t outputs) {
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
        std::string key;
        for (const auto& b : bindings) {
            const uint32_t k[] = {b.binding, uint32_t(b.descriptorType), b.stageFlags};
            key += bytesKey(k, sizeof(k));
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

    VkPipeline graphicsPipeline(const GraphicsState& s, ShaderModule* vs, ShaderModule* ps, VkPipelineLayout layout) {
        const std::string key = bytesKey(&s, sizeof(s));
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) return it->second;

        std::vector<VkPipelineShaderStageCreateInfo> stages;
        auto stage = [&](VkShaderStageFlagBits st, VkShaderModule m) {
            VkPipelineShaderStageCreateInfo i{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            i.stage = st;
            i.module = m;
            i.pName = "main";
            stages.push_back(i);
        };
        if (!vs->module || (ps && !ps->module)) return pipelines_[key] = VK_NULL_HANDLE;
        stage(VK_SHADER_STAGE_VERTEX_BIT, vs->module);
        VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        switch (s.topology) {
        case 1: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case 2: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case 5: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case 6: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 17: stage(VK_SHADER_STAGE_GEOMETRY_BIT, expandShader(false, s.vsOutputs)); break;
        case 19:
            topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
            stage(VK_SHADER_STAGE_GEOMETRY_BIT, expandShader(true, s.vsOutputs));
            break;
        default: break;
        }
        if (ps) stage(VK_SHADER_STAGE_FRAGMENT_BIT, ps->module);

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
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (s.depthFormat) {
            ds.depthTestEnable = (s.depthControl >> 1) & 1;
            ds.depthWriteEnable = ((s.depthControl >> 2) & 1) && !s.depthWriteOff;
            ds.depthCompareOp = VkCompareOp((s.depthControl >> 4) & 7);
            if (s.stencil) {
                ds.stencilTestEnable = s.depthControl & 1;
                auto face = [&](uint32_t func, uint32_t ops, uint32_t masks) {
                    VkStencilOpState f{};
                    f.compareOp = VkCompareOp(func);
                    f.failOp = stencilOp(ops & 0xF);
                    f.passOp = stencilOp((ops >> 4) & 0xF);
                    f.depthFailOp = stencilOp((ops >> 8) & 0xF);
                    f.compareMask = (masks >> 8) & 0xFF;
                    f.writeMask = (masks >> 16) & 0xFF;
                    f.reference = masks & 0xFF;
                    return f;
                };
                ds.front = face((s.depthControl >> 8) & 7, s.stencilControl, s.stencilMasks);
                ds.back = (s.depthControl >> 7) & 1 ? face((s.depthControl >> 20) & 7, s.stencilControl >> 12, s.stencilMasksBf)
                                                    : ds.front;
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
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 3;
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
        const VkResult r = vkCreateGraphicsPipelines(vk_.device, VK_NULL_HANDLE, 1, &gci, nullptr, &p);
        if (r != VK_SUCCESS) std::fprintf(stderr, "[gpu] graphics pipeline creation failed: %s\n", vkResultName(r));
        return pipelines_[key] = p;
    }

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
            VK_CHECK(vkCreateComputePipelines(vk_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
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
        case 8: return VK_IMAGE_VIEW_TYPE_1D;
        case 10: return VK_IMAGE_VIEW_TYPE_3D;
        case 11: return VK_IMAGE_VIEW_TYPE_CUBE;
        case 12: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        case 13: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
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
        const uint32_t key[] = {t.t[0], t.t[1], t.t[2], t.t[3] & ~0xFFFu, t.t[4], t.t[5]};
        const std::string k = bytesKey(key, sizeof(key));
        GuestTexture& tex = textures_[k];
        if (tex.img.image && tex.checkedFrame == frame_) return &tex;
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
        scratch_.resize(tiled);
        if (!readGuest(t.base(), scratch_.data(), tiled)) {
            warnOnce("texture at 0x" + hex(t.base()) + " is not readable");
            return nullptr;
        }
        const uint64_t h64 = hashBytes(scratch_.data(), tiled);
        tex.checkedFrame = frame_;
        if (tex.img.image && tex.hash == h64) return &tex;
        if (!ringFits(bytes)) {
            flush();
            begin();
        }
        const VkDeviceSize at = ringAlloc(bytes, 16);
        uint8_t* dst = static_cast<uint8_t*>(ringPtr_) + at;
        for (uint32_t s = 0; s < slices; ++s)
            detile(mode, scratch_.data() + s * slice, dst + size_t(s) * ew * eh * bpe, ew, eh, ep, bpe);
        if (!tex.img.image) {
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
        vkCmdCopyBufferToImage(cmd_, ring_, tex.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        return &tex;
    }

    struct ImageBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        float scale[2] = {1, 1};
    };

    ImageBinding resolveImage(const gcn::ImageResource& t) {
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
        if (auto it = colors_.find(addr); it != colors_.end()) {
            Surface& s = it->second;
            if (format == VK_FORMAT_UNDEFINED || bytesPerElement(t.dataFormat()) != s.elementBytes) {
                warnOnce("texture format " + std::to_string(t.dataFormat()) + " over a render target of different size");
                format = s.img.format;
            }
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
    void bindResources(gcn::Program& p, uint32_t base, VkShaderStageFlags, std::vector<VkWriteDescriptorSet>& writes,
                       std::vector<VkDescriptorBufferInfo>& bufInfos, std::vector<VkDescriptorImageInfo>& imgInfos) {
        uint32_t n = base + 1;
        for (auto& b : p.buffers) {
            uint64_t size = b.stride() ? uint64_t(b.stride()) * b.numRecords() : b.numRecords();
            size = std::min<uint64_t>(size ? size : 16, std::min<uint64_t>(vk_.props.limits.maxStorageBufferRange, 64ull << 20));
            BufferRange r;
            VkDeviceSize off = 0;
            if (b.known && guestBuffer(b.base(), size, r.buffer, off)) {
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
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstBinding = n++;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes.push_back(w);
        }
        for (auto& im : p.images) {
            const ImageBinding ib = resolveImage(im);
            if (trace_)
                std::fprintf(stderr, "[trace]   tex 0x%" PRIx64 " %ux%u fmt %u/%u type %u tile %u%s\n", im.base(), im.width(),
                             im.height(), im.dataFormat(), im.numFormat(), im.type(), im.tileMode(),
                             colors_.count(im.base()) ? " (render target)" : depths_.count(im.base()) ? " (depth)" : "");
            float scale[2] = {ib.scale[0], ib.scale[1]};
            std::memcpy(&p.data[im.data], scale, 8);
            imgInfos.push_back({ib.sampler, ib.view, VK_IMAGE_LAYOUT_GENERAL});
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstBinding = n++;
            w.descriptorCount = 1;
            w.descriptorType = im.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes.push_back(w);
        }
        // Draw data last: resource resolution above fills parts of it.
        const VkDeviceSize bytes = p.data.size() * 4;
        const VkDeviceSize at = ringAlloc(bytes, storageAlign_);
        std::memcpy(static_cast<uint8_t*>(ringPtr_) + at, p.data.data(), bytes);
        bufInfos.push_back({ring_, at, bytes});
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = base;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes.push_back(w);
    }

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

    void dispatch(const DrawCall& d, const GpuRegs& r) {
        gcn::Program cs = gcn::analyze(launch(r, gcn::Stage::Compute), readGuest);
        if (!cs.launch.entry) return;
        if (!ringFits(1 << 20)) flush();
        begin();
        ShaderModule* m = shader(cs, 0);
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
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vk_.cmdPushDescriptorSet(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, lay.pipeline, 0, uint32_t(writes.size()),
                                 writes.data());
        vkCmdDispatch(cmd_, d.groups[0], d.groups[1], d.groups[2]);
        ++stats_.dispatches;
    }

    void draw(const DrawCall& d, const GpuRegs& r) {
        const uint32_t cbMode = (r.ctx[reg::kCbColorControl] >> 4) & 7;
        const uint32_t dbRender = r.ctx[reg::kDbRenderControl];
        if (trace_)
            std::fprintf(stderr, "[trace]   cb mode %u db render 0x%x depth control 0x%x z info 0x%x prim %u count %u\n",
                         cbMode, dbRender, r.ctx[reg::kDbDepthControl], r.ctx[reg::kDbZInfo],
                         r.uconfig[reg::kPrimitiveType], d.count);
        if (cbMode > 1) return;           // fast-clear eliminate, resolve, decompress: nothing to do here
        if (dbRender & 0x1C) return;      // depth/stencil copy, resummarize
        const uint32_t prim = r.uconfig[reg::kPrimitiveType];
        if (prim == 0 || prim == 8 || prim == 13) {
            warnOnce("primitive type " + std::to_string(prim) + " not rendered");
            return;
        }

        gcn::Program vs = gcn::analyze(launch(r, gcn::Stage::Vertex), readGuest);
        if (!vs.launch.entry) return;
        const bool hasPs = shaderAddress(r, reg::kPgmLoPs) != 0;

        // Render targets.
        GraphicsState gs{};
        VkRenderingAttachmentInfo colors[8]{};
        uint32_t colorCount = 0;
        uint32_t width = 16384, height = 16384;
        gcn::LaunchState psLaunch = launch(r, gcn::Stage::Pixel);
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
            colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colors[i].imageView = view(s->img, vf, VK_IMAGE_VIEW_TYPE_2D, kIdentity);
            colors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            colors[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorCount = i + 1;
            if (trace_)
                std::fprintf(stderr, "[trace]   rt%u 0x%" PRIx64 " %ux%u fmt %u/%u swap %u mask 0x%x blend 0x%x\n", i,
                             uint64_t(cb[0]) << 8, s->img.width, s->img.height, fmt, num, (info >> 11) & 3,
                             (targetMask >> (i * 4)) & 0xF, r.ctx[reg::kCbBlendControl + i]);
            width = std::min(width, s->img.width);
            height = std::min(height, s->img.height);
            gs.colorFormats[i] = vf;
            gs.writeMasks |= ((targetMask >> (i * 4)) & 0xF) << (i * 4);
            gs.blend[i] = r.ctx[reg::kCbBlendControl + i];
            if (num == 4) psLaunch.colorUint |= 1u << i;
            if (num == 5) psLaunch.colorSint |= 1u << i;
            psLaunch.colorSwap[i] = (info >> 11) & 3;
            static const uint8_t kComps[32] = {0, 1, 1, 2, 1, 2, 3, 3, 4, 4, 4, 2, 4, 3, 4, 0,
                                               3, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            psLaunch.colorComps[i] = kComps[fmt];
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
                depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                gs.depthFormat = f;
                gs.stencil = stencil;
                gs.depthControl = depthControl;
                gs.stencilControl = r.ctx[reg::kDbStencilControl];
                gs.stencilMasks = r.ctx[reg::kDbStencilRefMask];
                gs.stencilMasksBf = r.ctx[reg::kDbStencilRefMaskBf];
                width = std::min(width, dw);
                height = std::min(height, dh);
                clearDepth = dbRender & 1;
                gs.depthWriteOff = clearDepth;
            }
        }
        if (!colorCount && !hasDepth) return;

        if (!ringFits(1 << 20)) flush();
        begin();
        ShaderModule* vsm = shader(vs, 0);
        gcn::Program ps;
        ShaderModule* psm = nullptr;
        if (hasPs) {
            ps = gcn::analyze(psLaunch, readGuest);
            psm = shader(ps, kPsBindingBase);
        }
        gs.vs = vsm->id;
        gs.ps = psm ? psm->id : 0;
        gs.topology = prim;
        gs.modeCntl = r.ctx[reg::kPaSuScModeCntl];
        gs.vsOutputs = vsm->translation.outputs;

        std::vector<VkDescriptorSetLayoutBinding> lb;
        addBindings(lb, vs, 0, VK_SHADER_STAGE_VERTEX_BIT);
        if (psm) addBindings(lb, ps, kPsBindingBase, VK_SHADER_STAGE_FRAGMENT_BIT);
        Layout& lay = layout(lb);
        VkPipeline pipe = graphicsPipeline(gs, vsm, psm, lay.pipeline);
        if (!pipe) return;

        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> bufs;
        std::vector<VkDescriptorImageInfo> imgs;
        bufs.reserve(64);
        imgs.reserve(64);
        if (trace_) {
            for (const auto* prog : {&vs, &ps}) {
                if (prog->data.size() <= gcn::data::kSmrd) continue;
                std::fprintf(stderr, "[trace]   %s smrd:", prog == &vs ? "vs" : "ps");
                for (size_t i = gcn::data::kSmrd; i < prog->data.size() && i < gcn::data::kSmrd + 24; ++i) {
                    float f;
                    std::memcpy(&f, &prog->data[i], 4);
                    std::fprintf(stderr, " %g", f);
                }
                std::fprintf(stderr, "\n");
                for (const auto& w : prog->warnings) std::fprintf(stderr, "[trace]   warning: %s\n", w.c_str());
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
            if (!guestBuffer(d.indexAddr, uint64_t(d.indexOffset + d.count) * isz, indexBuffer, indexOffset)) return;
        }

        fullBarrier();
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {width, height}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = colorCount;
        ri.pColorAttachments = colors;
        if (hasDepth) {
            ri.pDepthAttachment = &depth;
            if (gs.stencil) ri.pStencilAttachment = &depth;
        }
        vkCmdBeginRendering(cmd_, &ri);
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vk_.cmdPushDescriptorSet(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, lay.pipeline, 0, uint32_t(writes.size()),
                                 writes.data());
        setViewportScissor(r, width, height);
        vkCmdSetBlendConstants(cmd_, reinterpret_cast<const float*>(&r.ctx[reg::kCbBlendRed]));
        if (clearDepth) {
            VkClearAttachment ca{};
            ca.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            float z;
            std::memcpy(&z, &r.ctx[reg::kDbDepthClear], 4);
            ca.clearValue.depthStencil = {z, r.ctx[reg::kDbStencilClear] & 0xFF};
            if (gs.stencil && (dbRender & 2)) ca.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            VkClearRect cr{{{0, 0}, {width, height}}, 0, 1};
            vkCmdClearAttachments(cmd_, 1, &ca, 1, &cr);
        }
        if (d.kind == DrawCall::kIndexed) {
            vkCmdBindIndexBuffer(cmd_, indexBuffer, indexOffset, r.indexType ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
            vkCmdDrawIndexed(cmd_, d.count, std::max(r.numInstances, 1u), d.indexOffset, 0, 0);
        } else {
            vkCmdDraw(cmd_, d.count, std::max(r.numInstances, 1u), 0, 0);
        }
        vkCmdEndRendering(cmd_);
        ++stats_.draws;
        if (trace_ && snap_ && colorCount) snapshot(uint64_t(r.ctx[reg::kCbColorBase]) << 8);
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
        vkCmdSetViewport(cmd_, 0, 1, &vp);

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
            std::fprintf(stderr, "[trace]   scissors screen %x %x window %x %x generic %x %x vport %x %x mode0 %x\n",
                         r.ctx[reg::kScreenScissorTl], r.ctx[reg::kScreenScissorBr], r.ctx[reg::kWindowScissorTl],
                         r.ctx[reg::kWindowScissorBr], r.ctx[reg::kGenericScissorTl], r.ctx[reg::kGenericScissorBr],
                         r.ctx[reg::kVportScissorTl], r.ctx[reg::kVportScissorBr], r.ctx[0x292]);
        if (trace_)
            std::fprintf(stderr, "[trace]   viewport %.1f,%.1f %.1fx%.1f z %.2f..%.2f scissor %d,%d %ux%u vte 0x%x colfmt 0x%x\n",
                         vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth, sc.offset.x, sc.offset.y,
                         sc.extent.width, sc.extent.height, vte, r.ctx[reg::kSpiShaderColFormat]);
        vkCmdSetScissor(cmd_, 0, 1, &sc);
    }

    Vk vk_;
    std::mutex m_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool recording_ = false;
    VkBuffer ring_ = VK_NULL_HANDLE, dummyBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory ringMemory_ = VK_NULL_HANDLE, dummyMemory_ = VK_NULL_HANDLE;
    void* ringPtr_ = nullptr;
    VkDeviceSize ringUsed_ = 0, storageAlign_ = 16;
    std::vector<std::function<void()>> afterFlush_;
    std::vector<uint8_t> scratch_;

    std::map<uint64_t, Import> imports_;
    std::unordered_map<uint64_t, Surface> colors_, depths_;
    std::unordered_map<std::string, GuestTexture> textures_;
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
    bool trace_ = false, snap_ = false;
    struct {
        uint64_t draws = 0, dispatches = 0;
    } stats_;
};

Renderer* gRenderer = nullptr;

} // namespace

bool startRenderer() {
    auto* r = new Renderer;
    if (!r->init()) {
        delete r;
        return false;
    }
    gRenderer = r;
    addDrawListener(r);
    return true;
}

bool readDisplayBuffer(uint64_t address, uint32_t width, uint32_t height, std::vector<uint32_t>& pixels) {
    return gRenderer && gRenderer->readDisplay(address, width, height, pixels);
}

} // namespace host::render
