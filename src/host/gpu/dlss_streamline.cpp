// DLSS on Windows through NVIDIA Streamline (gpu/dlss.h's API). The DLSS SDK's NGX libraries are
// built for MSVC and do not link with MinGW; Streamline is a set of DLLs with a C entry point, so
// the build only needs its headers (scripts/fetch-streamline.sh, LPN_HAVE_STREAMLINE) and loads
// sl.interposer.dll from lpn-host's directory at run time. Streamline runs in manual hooking
// mode: the renderer makes its Vulkan instance and device itself (vk.cpp asks this file for the
// extensions, features and queues DLSS needs), then hands them over with slSetVulkanInfo. The
// renderer has no Vulkan swapchain, and DLSS super resolution does not need one.
#include "host/gpu/dlss.h"

#include "host/gpu/vk.h"

#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers_vk.h>

#include <windows.h>

#include <cstddef>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace host::render::dlss {

namespace {

constexpr const char* kProjectId = "6c1f5a8e-2b47-4d3a-9e61-0f8c3a5d7b24";

struct Api {
    PFun_slInit* init = nullptr;
    PFun_slShutdown* shutdown = nullptr;
    PFun_slIsFeatureSupported* isFeatureSupported = nullptr;
    PFun_slGetFeatureRequirements* getFeatureRequirements = nullptr;
    PFun_slGetFeatureFunction* getFeatureFunction = nullptr;
    PFun_slGetNewFrameToken* getNewFrameToken = nullptr;
    PFun_slSetConstants* setConstants = nullptr;
    PFun_slSetTagForFrame* setTagForFrame = nullptr;
    PFun_slEvaluateFeature* evaluateFeature = nullptr;
    PFun_slSetVulkanInfo* setVulkanInfo = nullptr;
    PFun_slDLSSSetOptions* setOptions = nullptr;
};

Api gSl;
bool gLoaded = false, gInit = false, gAvailable = false, gFailed = false, gDisabled = false;
sl::FeatureRequirements gReqs;
bool gHaveReqs = false;
uint32_t gExtraCompute = 0, gExtraGraphics = 0;
uint32_t gFrame = 0;
uint32_t gOptionsKey[3] = {}; // output w, h, quality the options were set for
bool gOptionsSet = false;

std::wstring exeDir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return L".";
    std::wstring s(buf, n);
    const size_t slash = s.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : s.substr(0, slash);
}

template <typename T> bool load(HMODULE m, const char* name, T** out) {
    *out = reinterpret_cast<T*>(reinterpret_cast<void*>(GetProcAddress(m, name)));
    return *out != nullptr;
}

const char* resultName(sl::Result r) {
    switch (r) {
    case sl::Result::eOk: return "ok";
    case sl::Result::eErrorIO: return "I/O error";
    case sl::Result::eErrorDriverOutOfDate: return "NVIDIA driver out of date";
    case sl::Result::eErrorOSOutOfDate: return "Windows out of date";
    case sl::Result::eErrorOSDisabledHWS: return "hardware GPU scheduling off";
    case sl::Result::eErrorDeviceNotCreated: return "device not created";
    case sl::Result::eErrorNoSupportedAdapterFound: return "no supported GPU";
    case sl::Result::eErrorAdapterNotSupported: return "GPU not supported";
    case sl::Result::eErrorNoPlugins: return "plugins not found";
    case sl::Result::eErrorVulkanAPI: return "Vulkan error";
    case sl::Result::eErrorFeatureMissing: return "feature missing";
    case sl::Result::eErrorFeatureNotSupported: return "feature not supported";
    case sl::Result::eErrorMissingInputParameter: return "missing input";
    case sl::Result::eErrorNotInitialized: return "not initialized";
    case sl::Result::eErrorNGXFailed: return "NGX failed";
    case sl::Result::eErrorInvalidIntegration: return "invalid integration";
    case sl::Result::eErrorMissingConstants: return "missing constants";
    case sl::Result::eErrorFeatureMissingHooks: return "feature needs hooks";
    case sl::Result::eErrorFeatureFailedToLoad: return "feature failed to load";
    default: return "error";
    }
}

void logMessage(sl::LogType type, const char* msg) {
    if (type == sl::LogType::eInfo && !getenv("LPN_DLSS_LOG")) return;
    std::fprintf(stderr, "[dlss] streamline: %s", msg);
}

// Loads Streamline and starts it, before the Vulkan instance exists (Streamline's rule).
void start() {
    if (gLoaded) return;
    gLoaded = true;
    if (getenv("LPN_NO_DLSS")) return;
    static const std::wstring dir = exeDir();
    const std::wstring path = dir + L"\\sl.interposer.dll";
    HMODULE m = LoadLibraryW(path.c_str());
    if (!m) {
        std::fprintf(stderr, "[dlss] unavailable: sl.interposer.dll not beside lpn-host.exe\n");
        return;
    }
    bool ok = load(m, "slInit", &gSl.init) && load(m, "slShutdown", &gSl.shutdown) &&
              load(m, "slIsFeatureSupported", &gSl.isFeatureSupported) &&
              load(m, "slGetFeatureRequirements", &gSl.getFeatureRequirements) &&
              load(m, "slGetFeatureFunction", &gSl.getFeatureFunction) &&
              load(m, "slGetNewFrameToken", &gSl.getNewFrameToken) && load(m, "slSetConstants", &gSl.setConstants) &&
              load(m, "slSetTagForFrame", &gSl.setTagForFrame) && load(m, "slEvaluateFeature", &gSl.evaluateFeature) &&
              load(m, "slSetVulkanInfo", &gSl.setVulkanInfo);
    if (!ok) {
        std::fprintf(stderr, "[dlss] unavailable: sl.interposer.dll is missing entry points\n");
        return;
    }
    static const wchar_t* paths[] = {dir.c_str()};
    static const sl::Feature features[] = {sl::kFeatureDLSS};
    sl::Preferences p{};
    p.showConsole = false;
    p.logLevel = getenv("LPN_DLSS_LOG") ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    p.pathsToPlugins = paths;
    p.numPathsToPlugins = 1;
    p.pathToLogsAndData = nullptr;
    p.logMessageCallback = logMessage;
    // No over-the-air updates: the DLLs shipped beside the exe are the ones that run.
    p.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eDisableCLStateTracking |
              sl::PreferenceFlags::eDisableDebugText | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    p.featuresToLoad = features;
    p.numFeaturesToLoad = 1;
    p.engine = sl::EngineType::eCustom;
    p.engineVersion = "1.0";
    p.projectId = kProjectId;
    p.renderAPI = sl::RenderAPI::eVulkan;
    const sl::Result r = gSl.init(p, sl::kSDKVersion);
    if (r != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] unavailable: Streamline init failed (%s)\n", resultName(r));
        return;
    }
    gInit = true;
    if (gSl.getFeatureRequirements(sl::kFeatureDLSS, gReqs) == sl::Result::eOk) gHaveReqs = true;
    if (gHaveReqs) {
        gExtraCompute = gReqs.vkNumComputeQueuesRequired;
        gExtraGraphics = gReqs.vkNumGraphicsQueuesRequired;
        // What Streamline adds to the Vulkan instance and device, for play.log.
        auto list = [](uint32_t n, const char** s) {
            std::string out;
            for (uint32_t i = 0; i < n; ++i)
                if (s && s[i]) out += (out.empty() ? "" : " ") + std::string(s[i]);
            return out.empty() ? std::string("none") : out;
        };
        std::fprintf(stderr, "[dlss] Streamline asks for instance extensions: %s\n",
                     list(gReqs.vkNumInstanceExtensions, gReqs.vkInstanceExtensions).c_str());
        std::fprintf(stderr, "[dlss] device extensions: %s\n",
                     list(gReqs.vkNumDeviceExtensions, gReqs.vkDeviceExtensions).c_str());
        std::fprintf(stderr, "[dlss] 1.2 features: %s; 1.3 features: %s; queues: %u compute, %u graphics\n",
                     list(gReqs.vkNumFeatures12, gReqs.vkFeatures12).c_str(),
                     list(gReqs.vkNumFeatures13, gReqs.vkFeatures13).c_str(), gExtraCompute, gExtraGraphics);
    } else {
        std::fprintf(stderr, "[dlss] Streamline gave no Vulkan requirements for DLSS\n");
    }
}

std::vector<std::string> names(uint32_t n, const char** s) {
    std::vector<std::string> out;
    for (uint32_t i = 0; i < n; ++i)
        if (s && s[i]) out.emplace_back(s[i]);
    return out;
}

// Sets in `want` each feature that is on in `need` and in `have`. The Vulkan feature structs are
// sType, pNext, then VkBool32s to the end.
template <typename T> void orFeatures(const T& need, const T& have, T* want) {
    constexpr size_t first = offsetof(T, pNext) + sizeof(void*);
    const size_t n = (sizeof(T) - first) / sizeof(VkBool32);
    const auto* a = reinterpret_cast<const VkBool32*>(reinterpret_cast<const char*>(&need) + first);
    const auto* b = reinterpret_cast<const VkBool32*>(reinterpret_cast<const char*>(&have) + first);
    auto* w = reinterpret_cast<VkBool32*>(reinterpret_cast<char*>(want) + first);
    for (size_t i = 0; i < n; ++i)
        if (a[i] && b[i]) w[i] = VK_TRUE;
}

sl::DLSSMode mode(UpscaleQuality q) {
    return q == UpscaleQuality::Native    ? sl::DLSSMode::eDLAA
           : q == UpscaleQuality::Quality ? sl::DLSSMode::eMaxQuality
           : q == UpscaleQuality::Balanced ? sl::DLSSMode::eBalanced
                                           : sl::DLSSMode::eMaxPerformance;
}

sl::Resource resource(const upscale::Image& i) {
    sl::Resource r(sl::ResourceType::eTex2d, reinterpret_cast<void*>(i.image), nullptr,
                   reinterpret_cast<void*>(i.view), uint32_t(VK_IMAGE_LAYOUT_GENERAL));
    r.width = i.width;
    r.height = i.height;
    r.nativeFormat = uint32_t(i.format);
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = i.usage;
    return r;
}

void identity(sl::float4x4& m) {
    for (uint32_t i = 0; i < 4; ++i)
        m.setRow(i, sl::float4(i == 0, i == 1, i == 2, i == 3));
}

} // namespace

std::vector<std::string> instanceExtensions() {
    start();
    if (!gHaveReqs) return {};
    return names(gReqs.vkNumInstanceExtensions, gReqs.vkInstanceExtensions);
}

std::vector<std::string> deviceExtensions(VkInstance, VkPhysicalDevice) {
    if (!gHaveReqs) return {};
    return names(gReqs.vkNumDeviceExtensions, gReqs.vkDeviceExtensions);
}

void deviceFeatures(const VkPhysicalDeviceVulkan12Features& have12, const VkPhysicalDeviceVulkan13Features& have13,
                    VkPhysicalDeviceVulkan12Features* want12, VkPhysicalDeviceVulkan13Features* want13) {
    if (!gHaveReqs) return;
    orFeatures(sl::getVkPhysicalDeviceVulkan12Features(gReqs.vkNumFeatures12, gReqs.vkFeatures12), have12, want12);
    orFeatures(sl::getVkPhysicalDeviceVulkan13Features(gReqs.vkNumFeatures13, gReqs.vkFeatures13), have13, want13);
}

uint32_t extraQueues() { return gInit ? gExtraCompute + gExtraGraphics : 0; }

void disable() {
    if (gInit) gSl.shutdown();
    gInit = gHaveReqs = false;
    gExtraCompute = gExtraGraphics = 0;
    gDisabled = true;
}

void init(const Vk& vk) {
    if (gDisabled) {
        std::fprintf(stderr, "[dlss] unavailable: off after the Vulkan device failed with Streamline's additions\n");
        return;
    }
    if (!gInit) return;
    sl::VulkanInfo info{};
    info.device = vk.device;
    info.instance = vk.instance;
    info.physicalDevice = vk.phys;
    // Streamline's queues come after the renderer's one, in the same family (vk.cpp).
    info.computeQueueFamily = vk.family;
    info.computeQueueIndex = 1;
    info.graphicsQueueFamily = vk.family;
    info.graphicsQueueIndex = 1 + gExtraCompute;
    sl::Result r = gSl.setVulkanInfo(info);
    if (r != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] unavailable: Streamline did not take the device (%s)\n", resultName(r));
        return;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = vk.phys;
    r = gSl.isFeatureSupported(sl::kFeatureDLSS, adapter);
    if (r != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] unavailable on this GPU (%s)\n", resultName(r));
        return;
    }
    void* fn = nullptr;
    if (gSl.getFeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", fn) != sl::Result::eOk || !fn) {
        std::fprintf(stderr, "[dlss] unavailable: no slDLSSSetOptions\n");
        return;
    }
    gSl.setOptions = reinterpret_cast<PFun_slDLSSSetOptions*>(fn);
    gAvailable = true;
    std::fprintf(stderr, "[dlss] available (Streamline)\n");
}

bool builtIn() { return true; }
bool available() { return gAvailable; }
bool usable() { return gAvailable && !gFailed; }

void shutdown() {
    if (gInit) gSl.shutdown();
    gInit = gAvailable = false;
}

namespace {
// Logs a Streamline call that took long on the CPU, to find stalls (at most 20 lines a run).
struct SlowCall {
    const char* what;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit SlowCall(const char* w) : what(w) {}
    ~SlowCall() {
        static int logged = 0;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms >= 20.0 && logged < 20) {
            ++logged;
            std::fprintf(stderr, "[dlss] %s took %.0f ms (frame %u)\n", what, ms, gFrame);
        }
    }
};
uint32_t gLastRender[2] = {0, 0};
} // namespace

bool evaluate(VkCommandBuffer cmd, VkDevice, const upscale::Frame& f) {
    if (!usable()) return false;
    if (f.renderW != gLastRender[0] || f.renderH != gLastRender[1]) {
        static int logged = 0;
        if (gLastRender[0] && logged++ < 10)
            std::fprintf(stderr, "[dlss] input size changed: %ux%u to %ux%u\n", gLastRender[0], gLastRender[1], f.renderW,
                         f.renderH);
        gLastRender[0] = f.renderW, gLastRender[1] = f.renderH;
    }
    auto fail = [](const char* what, sl::Result r) {
        std::fprintf(stderr, "[dlss] %s failed (%s): DLSS off for this run\n", what, resultName(r));
        gFailed = true;
        return false;
    };
    const sl::ViewportHandle viewport(0u);
    sl::FrameToken* token = nullptr;
    const uint32_t index = gFrame++;
    sl::Result r = gSl.getNewFrameToken(token, &index);
    if (r != sl::Result::eOk || !token) return fail("getting a frame token", r);

    const uint32_t key[3] = {f.outW, f.outH, uint32_t(f.quality)};
    if (!gOptionsSet || key[0] != gOptionsKey[0] || key[1] != gOptionsKey[1] || key[2] != gOptionsKey[2]) {
        sl::DLSSOptions o{};
        o.mode = mode(f.quality);
        o.outputWidth = f.outW;
        o.outputHeight = f.outH;
        o.colorBuffersHDR = sl::Boolean::eFalse; // the game's colour is already tone mapped (8-bit)
        {
            SlowCall t("slDLSSSetOptions");
            r = gSl.setOptions(viewport, o);
        }
        if (r != sl::Result::eOk) return fail("setting the options", r);
        gOptionsKey[0] = key[0], gOptionsKey[1] = key[1], gOptionsKey[2] = key[2];
        gOptionsSet = true;
        std::fprintf(stderr, "[dlss] %s: %ux%u to %ux%u\n", f.quality == UpscaleQuality::Native ? "DLAA" : "upscaling",
                     f.renderW, f.renderH, f.outW, f.outH);
    }

    sl::Constants c{};
    identity(c.cameraViewToClip);
    identity(c.clipToCameraView);
    identity(c.clipToLensClip);
    identity(c.clipToPrevClip);
    identity(c.prevClipToClip);
    c.jitterOffset = sl::float2(f.jitterX, f.jitterY);
    // Motion vectors are in render pixels; Streamline wants them scaled to [-1, 1].
    c.mvecScale = sl::float2(1.0f / float(f.renderW), 1.0f / float(f.renderH));
    c.cameraPinholeOffset = sl::float2(0, 0);
    c.cameraPos = sl::float3(0, 0, 0);
    c.cameraUp = sl::float3(0, 1, 0);
    c.cameraRight = sl::float3(1, 0, 0);
    c.cameraFwd = sl::float3(0, 0, 1);
    const bool inverted = f.depth0 > f.depth1;
    c.cameraNear = inverted ? f.depth1 : f.depth0;
    c.cameraFar = inverted ? f.depth0 : f.depth1;
    c.cameraFOV = f.fovY;
    c.cameraAspectRatio = float(f.renderW) / float(f.renderH);
    c.depthInverted = inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = f.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    {
        SlowCall t("slSetConstants");
        r = gSl.setConstants(c, *token, viewport);
    }
    if (r != sl::Result::eOk) return fail("setting the constants", r);

    sl::Resource color = resource(f.color), depth = resource(f.depth), motion = resource(f.motion),
                 output = resource(f.output);
    const sl::Extent inExtent{0, 0, f.renderW, f.renderH}, outExtent{0, 0, f.outW, f.outH};
    const sl::ResourceTag tags[] = {
        sl::ResourceTag(&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eOnlyValidNow, &inExtent),
        sl::ResourceTag(&output, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eOnlyValidNow, &outExtent),
        sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &inExtent),
        sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &inExtent),
    };
    auto* slCmd = reinterpret_cast<sl::CommandBuffer*>(cmd);
    {
        SlowCall t("slSetTagForFrame");
        r = gSl.setTagForFrame(*token, viewport, tags, 4, slCmd);
    }
    if (r != sl::Result::eOk) return fail("tagging the images", r);

    const sl::BaseStructure* inputs[] = {&viewport};
    {
        SlowCall t("slEvaluateFeature");
        r = gSl.evaluateFeature(sl::kFeatureDLSS, *token, inputs, 1, slCmd);
    }
    if (r != sl::Result::eOk) return fail("evaluating", r);
    return true;
}

} // namespace host::render::dlss
