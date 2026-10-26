#include "host/gpu/dlss.h"

#include "host/gpu/vk.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef LPN_HAVE_DLSS
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

#include <unistd.h>

#include <climits>
#endif

namespace host::render::dlss {

#ifdef LPN_HAVE_DLSS

namespace {

// NGX asks a custom engine for a GUID of its own.
constexpr const char* kProjectId = "6c1f5a8e-2b47-4d3a-9e61-0f8c3a5d7b24";

bool gInit = false, gAvailable = false, gFailed = false;
NVSDK_NGX_Parameter* gCaps = nullptr;
NVSDK_NGX_Parameter* gParams = nullptr; // creation and evaluation parameters
NVSDK_NGX_Handle* gFeature = nullptr;
uint32_t gFeatureKey[5] = {}; // render w, h, output w, h, mode the feature was made for

// lpn-host's directory: the build puts the DLSS runtime (libnvidia-ngx-dlss.so.*) beside it.
std::wstring exeDir() {
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return L".";
    std::string s(buf, size_t(n));
    s.erase(s.find_last_of('/'));
    return std::wstring(s.begin(), s.end());
}

const NVSDK_NGX_FeatureCommonInfo* featureInfo() {
    static const std::wstring dir = exeDir();
    static const wchar_t* paths[] = {dir.c_str()};
    static NVSDK_NGX_FeatureCommonInfo info = [] {
        NVSDK_NGX_FeatureCommonInfo i{};
        i.PathListInfo.Path = paths;
        i.PathListInfo.Length = 1;
        i.LoggingInfo.LoggingCallback = [](const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
            std::fprintf(stderr, "[dlss] ngx: %s", message);
        };
        i.LoggingInfo.MinimumLoggingLevel = getenv("LPN_DLSS_LOG") ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE
                                                                   : NVSDK_NGX_LOGGING_LEVEL_OFF;
        i.LoggingInfo.DisableOtherLoggingSinks = true;
        return i;
    }();
    return &info;
}

NVSDK_NGX_FeatureDiscoveryInfo discovery() {
    NVSDK_NGX_FeatureDiscoveryInfo d{};
    d.SDKVersion = NVSDK_NGX_Version_API;
    d.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    d.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    d.Identifier.v.ProjectDesc = {kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0"};
    d.ApplicationDataPath = L".";
    d.FeatureInfo = featureInfo();
    return d;
}

std::vector<std::string> names(uint32_t n, const VkExtensionProperties* props) {
    std::vector<std::string> out;
    for (uint32_t i = 0; i < n; ++i) out.emplace_back(props[i].extensionName);
    return out;
}

} // namespace

std::vector<std::string> instanceExtensions() {
    const NVSDK_NGX_FeatureDiscoveryInfo d = discovery();
    uint32_t n = 0;
    VkExtensionProperties* props = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&d, &n, &props))) return {};
    return names(n, props);
}

std::vector<std::string> deviceExtensions(VkInstance instance, VkPhysicalDevice phys) {
    const NVSDK_NGX_FeatureDiscoveryInfo d = discovery();
    uint32_t n = 0;
    VkExtensionProperties* props = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, phys, &d, &n, &props)))
        return {};
    return names(n, props);
}

void init(const Vk& vk) {
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L".", vk.instance, vk.phys, vk.device,
        vkGetInstanceProcAddr, vkGetDeviceProcAddr, featureInfo());
    if (NVSDK_NGX_FAILED(r)) {
        std::fprintf(stderr, "[dlss] unavailable: NGX init failed (0x%x; NVIDIA driver needed)\n", unsigned(r));
        return;
    }
    gInit = true;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&gCaps);
    if (NVSDK_NGX_FAILED(r)) {
        std::fprintf(stderr, "[dlss] unavailable: no capability parameters (0x%x)\n", unsigned(r));
        return;
    }
    int needsDriver = 0;
    unsigned major = 0, minor = 0;
    if (NVSDK_NGX_SUCCEED(gCaps->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver)) &&
        needsDriver) {
        gCaps->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &major);
        gCaps->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minor);
        std::fprintf(stderr, "[dlss] unavailable: needs NVIDIA driver %u.%u or newer\n", major, minor);
        return;
    }
    int ok = 0, initResult = 0;
    gCaps->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &ok);
    gCaps->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
    if (!ok) {
        std::fprintf(stderr, "[dlss] unavailable on this GPU (feature init 0x%x)\n", unsigned(initResult));
        return;
    }
    gAvailable = true;
    uint32_t w = 0, h = 0;
    renderSize(2, 2560, 1440, &w, &h);
    std::fprintf(stderr, "[dlss] available (Quality renders 2560x1440 at %ux%u)\n", w, h);
}

bool available() { return gAvailable; }
bool usable() { return gAvailable && !gFailed; }

void shutdown() {
    if (!gInit) return;
    if (gFeature) NVSDK_NGX_VULKAN_ReleaseFeature(gFeature);
    if (gParams) NVSDK_NGX_VULKAN_DestroyParameters(gParams);
    if (gCaps) NVSDK_NGX_VULKAN_DestroyParameters(gCaps);
    NVSDK_NGX_VULKAN_Shutdown1(nullptr);
    gFeature = nullptr;
    gParams = gCaps = nullptr;
    gInit = gAvailable = false;
}

bool renderSize(int mode, uint32_t outW, uint32_t outH, uint32_t* w, uint32_t* h) {
    if (!gAvailable || mode <= 0) return false;
    if (mode == 1) { // DLAA: anti-aliasing at the output size
        *w = outW, *h = outH;
        return true;
    }
    const NVSDK_NGX_PerfQuality_Value q = mode == 2   ? NVSDK_NGX_PerfQuality_Value_MaxQuality
                                          : mode == 3 ? NVSDK_NGX_PerfQuality_Value_Balanced
                                                      : NVSDK_NGX_PerfQuality_Value_MaxPerf;
    unsigned ow = 0, oh = 0, maxW, maxH, minW, minH;
    float sharpness;
    if (NVSDK_NGX_FAILED(NGX_DLSS_GET_OPTIMAL_SETTINGS(gCaps, outW, outH, q, &ow, &oh, &maxW, &maxH, &minW, &minH,
                                                       &sharpness)) ||
        !ow || !oh)
        return false;
    *w = ow, *h = oh;
    return true;
}

namespace {

NVSDK_NGX_PerfQuality_Value quality(int mode) {
    return mode == 1   ? NVSDK_NGX_PerfQuality_Value_DLAA
           : mode == 2 ? NVSDK_NGX_PerfQuality_Value_MaxQuality
           : mode == 3 ? NVSDK_NGX_PerfQuality_Value_Balanced
                       : NVSDK_NGX_PerfQuality_Value_MaxPerf;
}

NVSDK_NGX_Resource_VK resource(const Image& i, bool write) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(i.view, i.image, {i.aspect, 0, 1, 0, 1}, i.format, i.width, i.height,
                                                  write);
}

} // namespace

bool evaluate(VkCommandBuffer cmd, VkDevice device, const Frame& f) {
    if (!usable()) return false;
    auto fail = [](const char* what, NVSDK_NGX_Result r) {
        std::fprintf(stderr, "[dlss] %s failed (0x%x): DLSS off for this run\n", what, unsigned(r));
        gFailed = true;
        return false;
    };
    if (!gParams && NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&gParams)))
        return fail("allocating parameters", NVSDK_NGX_Result_Fail);
    const uint32_t key[5] = {f.renderW, f.renderH, f.outW, f.outH, uint32_t(f.mode)};
    if (!gFeature || std::memcmp(key, gFeatureKey, sizeof key) != 0) {
        if (gFeature) NVSDK_NGX_VULKAN_ReleaseFeature(gFeature), gFeature = nullptr;
        NVSDK_NGX_DLSS_Create_Params c{};
        c.Feature.InWidth = f.renderW;
        c.Feature.InHeight = f.renderH;
        c.Feature.InTargetWidth = f.outW;
        c.Feature.InTargetHeight = f.outH;
        c.Feature.InPerfQualityValue = quality(f.mode);
        // The game's colour is already tone mapped (8-bit); motion vectors are at render size.
        c.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(device, cmd, 1, 1, &gFeature, gParams, &c);
        if (NVSDK_NGX_FAILED(r)) return fail("creating the feature", r);
        std::memcpy(gFeatureKey, key, sizeof key);
        std::fprintf(stderr, "[dlss] %s: %ux%u to %ux%u\n", f.mode == 1 ? "DLAA" : "upscaling", f.renderW, f.renderH, f.outW,
                     f.outH);
    }
    NVSDK_NGX_Resource_VK color = resource(f.color, false), depth = resource(f.depth, false),
                          motion = resource(f.motion, false), output = resource(f.output, true);
    NVSDK_NGX_VK_DLSS_Eval_Params e{};
    e.Feature.pInColor = &color;
    e.Feature.pInOutput = &output;
    e.pInDepth = &depth;
    e.pInMotionVectors = &motion;
    NVSDK_NGX_Resource_VK bias = resource(f.bias, false);
    if (f.bias.image) e.pInBiasCurrentColorMask = &bias;
    e.InJitterOffsetX = f.jitterX;
    e.InJitterOffsetY = f.jitterY;
    e.InRenderSubrectDimensions = {f.renderW, f.renderH};
    e.InReset = f.reset;
    e.InMVScaleX = 1;
    e.InMVScaleY = 1;
    const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, gFeature, gParams, &e);
    if (NVSDK_NGX_FAILED(r)) return fail("evaluating", r);
    return true;
}

void jitter(uint64_t n, uint32_t renderW, uint32_t outW, float* x, float* y) {
    const float scale = renderW ? float(outW) / float(renderW) : 1.0f;
    const uint64_t phases = std::max<uint64_t>(8, uint64_t(8.0f * scale * scale + 0.5f));
    auto halton = [](uint64_t i, uint32_t base) {
        float f = 1, r = 0;
        for (; i; i /= base) f /= float(base), r += f * float(i % base);
        return r;
    };
    const uint64_t i = n % phases + 1; // index 0 of the sequence is 0, 0
    *x = halton(i, 2) - 0.5f;
    *y = halton(i, 3) - 0.5f;
}

#else

bool evaluate(VkCommandBuffer, VkDevice, const Frame&) { return false; }
void jitter(uint64_t, uint32_t, uint32_t, float* x, float* y) { *x = *y = 0; }

std::vector<std::string> instanceExtensions() { return {}; }
std::vector<std::string> deviceExtensions(VkInstance, VkPhysicalDevice) { return {}; }
void init(const Vk&) { std::fprintf(stderr, "[dlss] not built in (scripts/fetch-dlss.sh, then re-run cmake)\n"); }
bool available() { return false; }
bool usable() { return false; }
void shutdown() {}
bool renderSize(int, uint32_t, uint32_t, uint32_t*, uint32_t*) { return false; }

#endif

} // namespace host::render::dlss
