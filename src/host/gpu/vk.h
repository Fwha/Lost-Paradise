// Vulkan device setup shared by the renderer.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace host::render {

struct Vk {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    VkDeviceSize hostPointerAlignment = 4096;
    bool hostImport = false; // VK_EXT_external_memory_host
    bool pipelineLibraries = false; // VK_EXT_graphics_pipeline_library with fast linking
    // Features enabled on the device that FSR (gpu/fsr.h) asks about.
    bool storageWriteWithoutFormat = false, shaderFloat16 = false, subgroupSizeControl = false,
         computeFullSubgroups = false, synchronization2 = false;
    PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet = nullptr;
    PFN_vkGetMemoryHostPointerPropertiesEXT getMemoryHostPointerProperties = nullptr;

    // Makes the instance and device. When that fails with DLSS's additions (gpu/dlss.h), tries
    // again without them and turns DLSS off.
    bool init(std::string* error);
    // Index of a memory type in `bits` with `flags`, or UINT32_MAX.
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;

private:
    // What of DLSS's Vulkan additions create() makes the instance and device with.
    struct DlssAdditions {
        bool on = true, features = true;
        std::string dropExt; // a device extension left out
    };
    bool create(std::string* error, const DlssAdditions& add);
    void destroy();
    std::vector<std::string> dlssDeviceExts_; // DLSS's device extensions the GPU has
};

const char* vkResultName(VkResult r);

} // namespace host::render
