#include "host/gpu/fsr.h"

#include "host/gpu/vk.h"

#include <cfloat>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#ifdef LPN_HAVE_FSR
#include <ffx_vk_portable.h>
#endif

namespace host::render::fsr {

#ifdef LPN_HAVE_FSR

namespace {

bool gAvailable = false, gFailed = false;
FfxVkPortableDeviceInfo gDevice{};
FfxVkPortableUpscaleContext* gContext = nullptr;
// Contexts replaced by one of another size, with the dispatch count then: frames still on the GPU
// may use them, so they go a few frames later.
std::vector<std::pair<FfxVkPortableUpscaleContext*, uint64_t>> gRetired;
uint64_t gDispatches = 0;
uint32_t gKey[5] = {}; // render w, h, output w, h, flags: what the context was made for

FfxVkPortableImage image(const upscale::Image& i) {
    FfxVkPortableImage p{};
    p.structSize = sizeof p;
    p.image = i.image;
    p.format = i.format;
    p.extent = {i.width, i.height};
    p.mipCount = 1;
    p.arrayLayers = 1;
    p.usage = i.usage;
    p.aspect = i.aspect;
    p.state = FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS; // the GENERAL layout, kept so
    p.view = i.view;
    return p;
}

} // namespace

bool builtIn() { return true; }

void init(const Vk& vk) {
    FfxVkPortableDeviceCapabilities caps{};
    caps.structSize = sizeof caps;
    if (ffxVkPortableQueryDeviceCapabilities(vk.phys, &caps) != FFX_VK_PORTABLE_OK || !caps.fsr3ComputePrerequisites ||
        !vk.storageWriteWithoutFormat) {
        std::fprintf(stderr, "[fsr] unavailable: the GPU lacks a feature FSR 3.1 needs\n");
        return;
    }
    gDevice.structSize = sizeof gDevice;
    gDevice.instance = vk.instance;
    gDevice.physicalDevice = vk.phys;
    gDevice.device = vk.device;
    gDevice.getDeviceProcAddr = vkGetDeviceProcAddr;
    gDevice.queue = vk.queue;
    gDevice.queueFamilyIndex = vk.family;
    gDevice.shaderFloat16Enabled = vk.shaderFloat16;
    gDevice.subgroupSizeControlEnabled = vk.subgroupSizeControl;
    gDevice.computeFullSubgroupsEnabled = vk.computeFullSubgroups;
    gDevice.synchronization2Enabled = vk.synchronization2;
    gDevice.shaderStorageImageWriteWithoutFormatEnabled = VK_TRUE;
    gAvailable = true;
    std::fprintf(stderr, "[fsr] available (FSR 3.1%s)\n", vk.shaderFloat16 ? ", 16-bit" : "");
}

bool available() { return gAvailable; }
bool usable() { return gAvailable && !gFailed; }

void shutdown() {
    for (auto& [c, at] : gRetired) ffxVkPortableUpscaleContextDestroy(c);
    gRetired.clear();
    ffxVkPortableUpscaleContextDestroy(gContext);
    gContext = nullptr;
}

bool evaluate(VkCommandBuffer cmd, const upscale::Frame& f) {
    if (!usable()) return false;
    auto fail = [](const char* what, int r) {
        std::fprintf(stderr, "[fsr] %s failed (%d): FSR off for this run\n", what, r);
        gFailed = true;
        return false;
    };
    // The camera's depth range: near is the smaller view distance, whichever end of the depth
    // range it is at; past 100 km the projection is taken as infinite.
    const bool inverted = f.depth0 > f.depth1;
    const float nearZ = inverted ? f.depth1 : f.depth0, farZ = inverted ? f.depth0 : f.depth1;
    const bool infinite = farZ > 1e5f;
    uint32_t flags = 0;
    if (inverted) flags |= FFX_VK_PORTABLE_CONTEXT_DEPTH_INVERTED;
    if (infinite) flags |= FFX_VK_PORTABLE_CONTEXT_DEPTH_INFINITE;
    const uint32_t key[5] = {f.renderW, f.renderH, f.outW, f.outH, flags};
    if (!gContext || std::memcmp(key, gKey, sizeof key) != 0) {
        if (gContext) gRetired.push_back({gContext, gDispatches});
        gContext = nullptr;
        FfxVkPortableUpscaleCreateInfo c{};
        c.structSize = sizeof c;
        c.flags = flags; // tone-mapped (LDR) colour, motion vectors at the render size
        c.maxRenderSize = {f.renderW, f.renderH};
        c.maxOutputSize = {f.outW, f.outH};
        const FfxVkPortableResult r = ffxVkPortableUpscaleContextCreate(&gDevice, &c, &gContext);
        if (r != FFX_VK_PORTABLE_OK) return fail("creating the context", r);
        std::memcpy(gKey, key, sizeof key);
        std::fprintf(stderr, "[fsr] %s: %ux%u to %ux%u%s\n", f.quality == UpscaleQuality::Native ? "native AA" : "upscaling",
                     f.renderW, f.renderH, f.outW, f.outH, inverted ? ", depth inverted" : "");
    }
    FfxVkPortableUpscaleDispatchInfo d{};
    d.structSize = sizeof d;
    d.commandBuffer = cmd;
    d.color = image(f.color);
    d.depth = image(f.depth);
    d.motionVectors = image(f.motion);
    d.output = image(f.output);
    d.exposure.structSize = d.reactiveMask.structSize = d.transparencyAndCompositionMask.structSize = sizeof(FfxVkPortableImage);
    d.jitterOffset = {f.jitterX, f.jitterY};
    d.motionVectorScale = {1.0f, 1.0f}; // already in render pixels
    d.renderSize = {f.renderW, f.renderH};
    d.outputSize = {f.outW, f.outH};
    d.frameTimeMilliseconds = f.frameMs > 0 ? f.frameMs : 16.7f;
    d.preExposure = 1.0f;
    d.cameraNear = nearZ > 0 ? nearZ : 0.01f;
    d.cameraFar = infinite ? FLT_MAX : farZ;
    d.cameraVerticalFovRadians = f.fovY > 0 && f.fovY < 3.1f ? f.fovY : 1.0f;
    d.viewSpaceToMeters = 1.0f;
    // FSR's own sharpening (RCAS), lightly: the game's picture has none of its own after SMAA.
    d.enableSharpening = VK_TRUE;
    d.sharpness = 0.3f;
    d.reset = f.reset;
    d.frameId = ++gDispatches;
    for (size_t i = 0; i < gRetired.size();)
        if (gDispatches - gRetired[i].second > 16) {
            ffxVkPortableUpscaleContextDestroy(gRetired[i].first);
            gRetired.erase(gRetired.begin() + long(i));
        } else {
            ++i;
        }
    const FfxVkPortableResult r = ffxVkPortableUpscaleContextRecordDispatch(gContext, &d);
    if (r != FFX_VK_PORTABLE_OK) return fail("dispatching", r);
    return true;
}

#else

bool builtIn() { return false; }
void init(const Vk&) { std::fprintf(stderr, "[fsr] not built in (scripts/fetch-fsr.sh, then re-run cmake)\n"); }
bool available() { return false; }
bool usable() { return false; }
void shutdown() {}
bool evaluate(VkCommandBuffer, const upscale::Frame&) { return false; }

#endif

} // namespace host::render::fsr
