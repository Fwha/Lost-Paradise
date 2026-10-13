// Vulkan device setup shared by the renderer.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

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
    PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet = nullptr;
    PFN_vkGetMemoryHostPointerPropertiesEXT getMemoryHostPointerProperties = nullptr;

    bool init(std::string* error);
    // Index of a memory type in `bits` with `flags`, or UINT32_MAX.
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;
};

const char* vkResultName(VkResult r);

} // namespace host::render
