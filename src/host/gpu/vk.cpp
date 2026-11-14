#include "host/gpu/vk.h"

#include "host/gpu/dlss.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace host::render {

const char* vkResultName(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    default: return "VkResult(?)";
    }
}

uint32_t Vk::memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((bits >> i) & 1 && (mem.memoryTypes[i].propertyFlags & flags) == flags) return i;
    return UINT32_MAX;
}

bool Vk::init(std::string* error) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "lpn-host";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    // DLSS's instance extensions, those the loader has.
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> instanceExts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, instanceExts.data());
    const std::vector<std::string> dlssInstance = dlss::instanceExtensions();
    std::vector<const char*> enableInstance;
    for (const std::string& e : dlssInstance)
        for (const auto& have : instanceExts)
            if (e == have.extensionName) enableInstance.push_back(e.c_str());
    ici.enabledExtensionCount = uint32_t(enableInstance.size());
    ici.ppEnabledExtensionNames = enableInstance.data();
    VkResult r = vkCreateInstance(&ici, nullptr, &instance);
    if (r != VK_SUCCESS) {
        *error = std::string("vkCreateInstance: ") + vkResultName(r);
        return false;
    }
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(instance, &n, devices.data());
    if (devices.empty()) {
        *error = "no Vulkan devices";
        return false;
    }
    // Prefer a discrete GPU, then integrated, then anything (lavapipe).
    int best = -1, bestScore = -1;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                          : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                                                                   : 1;
        if (score > bestScore) best = int(i), bestScore = score;
    }
    if (best < 0) {
        *error = "no Vulkan 1.3 device";
        return false;
    }
    phys = devices[best];
    vkGetPhysicalDeviceProperties(phys, &props);
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);

    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
    auto has = [&](const char* name) {
        for (const auto& e : exts)
            if (!std::strcmp(e.extensionName, name)) return true;
        return false;
    };
    std::vector<const char*> enable;
    if (!has(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        *error = "VK_KHR_push_descriptor is required";
        return false;
    }
    enable.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    hostImport = has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (hostImport) enable.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    // Pipeline libraries (docs/shader-precompile.md stage 3): checked for the feature below.
    const bool gplExt = has(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) && has(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    // DLSS's device extensions, those the driver has (none without an NVIDIA driver).
    const std::vector<std::string> dlssDevice = dlss::deviceExtensions(instance, phys);
    for (const std::string& e : dlssDevice)
        if (has(e.c_str()) && std::find_if(enable.begin(), enable.end(), [&](const char* x) { return e == x; }) ==
                                  enable.end())
            enable.push_back(e.c_str());

    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, families.data());
    family = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            family = i;
            break;
        }
    if (family == UINT32_MAX) {
        *error = "no graphics+compute queue";
        return false;
    }

    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features have13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT haveGpl{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    have.pNext = &have12;
    have12.pNext = &have13;
    if (gplExt) have13.pNext = &haveGpl;
    vkGetPhysicalDeviceFeatures2(phys, &have);
    if (gplExt && haveGpl.graphicsPipelineLibrary) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gp{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &gp;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        pipelineLibraries = gp.graphicsPipelineLibraryFastLinking; // linking at draw time is only worth it when fast
    }
    if (pipelineLibraries) {
        enable.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
        enable.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
    }
    if (!have13.dynamicRendering || !have.features.geometryShader) {
        *error = "dynamic rendering and geometry shaders are required";
        return false;
    }

    VkPhysicalDeviceFeatures2 want{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features want12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features want13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT wantGpl{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    want.pNext = &want12;
    want12.pNext = &want13;
    if (pipelineLibraries) {
        wantGpl.graphicsPipelineLibrary = VK_TRUE;
        want13.pNext = &wantGpl;
    }
    auto& f = want.features;
    const auto& h = have.features;
    f.geometryShader = VK_TRUE;
    // Out-of-range buffer accesses read zero instead of faulting, as on the PS4.
    f.robustBufferAccess = h.robustBufferAccess;
    f.independentBlend = h.independentBlend;
    f.dualSrcBlend = h.dualSrcBlend;
    f.samplerAnisotropy = h.samplerAnisotropy;
    f.textureCompressionBC = h.textureCompressionBC;
    f.fragmentStoresAndAtomics = h.fragmentStoresAndAtomics;
    f.vertexPipelineStoresAndAtomics = h.vertexPipelineStoresAndAtomics;
    f.shaderStorageImageWriteWithoutFormat = h.shaderStorageImageWriteWithoutFormat;
    f.shaderStorageImageReadWithoutFormat = h.shaderStorageImageReadWithoutFormat;
    // rg16f storage images (the DLSS motion pass).
    f.shaderStorageImageExtendedFormats = h.shaderStorageImageExtendedFormats;
    f.shaderInt64 = h.shaderInt64;
    f.depthClamp = h.depthClamp;
    f.fillModeNonSolid = h.fillModeNonSolid;
    f.imageCubeArray = h.imageCubeArray;
    want12.scalarBlockLayout = have12.scalarBlockLayout;
    // DLSS reads its resources through buffer device addresses.
    if (!dlssDevice.empty()) want12.bufferDeviceAddress = have12.bufferDeviceAddress;
    want13.dynamicRendering = VK_TRUE;
    want13.synchronization2 = have13.synchronization2;
    want13.maintenance4 = have13.maintenance4;
    // FSR's 16-bit shader variants and its subgroup sizes (gpu/fsr.cpp), when the GPU has them.
    want12.shaderFloat16 = have12.shaderFloat16;
    want13.subgroupSizeControl = have13.subgroupSizeControl;
    want13.computeFullSubgroups = have13.computeFullSubgroups;
    dlss::deviceFeatures(have12, have13, &want12, &want13);
    storageWriteWithoutFormat = h.shaderStorageImageWriteWithoutFormat;
    shaderFloat16 = have12.shaderFloat16;
    subgroupSizeControl = have13.subgroupSizeControl;
    computeFullSubgroups = have13.computeFullSubgroups;
    synchronization2 = have13.synchronization2;

    // Queue 0 is the renderer's; DLSS (Streamline) may want more in the same family after it.
    const uint32_t queues = std::min(1 + dlss::extraQueues(), families[family].queueCount);
    const std::vector<float> priorities(queues, 1.0f);
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = queues;
    qci.pQueuePriorities = priorities.data();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &want;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable.size());
    dci.ppEnabledExtensionNames = enable.data();
    r = vkCreateDevice(phys, &dci, nullptr, &device);
    if (r != VK_SUCCESS) {
        *error = std::string("vkCreateDevice: ") + vkResultName(r);
        return false;
    }
    vkGetDeviceQueue(device, family, 0, &queue);
    cmdPushDescriptorSet =
        reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
    if (hostImport) {
        getMemoryHostPointerProperties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(device, "vkGetMemoryHostPointerPropertiesEXT"));
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &hp;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        hostPointerAlignment = hp.minImportedHostPointerAlignment;
    }
    std::fprintf(stderr, "[gpu] Vulkan device: %s (host memory import %s, fast-linked pipeline libraries %s)\n",
                 props.deviceName, hostImport ? "yes" : "no", pipelineLibraries ? "yes" : "no");
    return true;
}

} // namespace host::render
