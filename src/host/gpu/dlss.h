// NVIDIA DLSS on the renderer's Vulkan device. On Linux through the DLSS SDK's NGX library, built
// in when the SDK is in third_party/dlss (scripts/fetch-dlss.sh, LPN_HAVE_DLSS; gpu/dlss.cpp). On
// Windows through NVIDIA Streamline, built in when its SDK is in third_party/streamline
// (scripts/fetch-streamline.sh, LPN_HAVE_STREAMLINE; gpu/dlss_streamline.cpp). Used when the
// driver offers it (an RTX card with NVIDIA's driver). Otherwise every call here is a no-op.
#pragma once

#include "host/gpu/upscale.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace host::render {

struct Vk;

namespace dlss {

// The extensions NGX needs, to be enabled when the instance and the device are created (Vk::init
// enables those the driver has).
std::vector<std::string> instanceExtensions();
std::vector<std::string> deviceExtensions(VkInstance instance, VkPhysicalDevice phys);
// The Vulkan 1.2 and 1.3 features DLSS needs: set in want where the GPU has them.
void deviceFeatures(const VkPhysicalDeviceVulkan12Features& have12, const VkPhysicalDeviceVulkan13Features& have13,
                    VkPhysicalDeviceVulkan12Features* want12, VkPhysicalDeviceVulkan13Features* want13);
// Queues DLSS wants beside the renderer's, in the renderer's family (they follow its queue 0).
uint32_t extraQueues();
// Turns DLSS off for the run (vk.cpp, when the device could not be made with DLSS's additions).
void disable();

// After the device is made: starts NGX and asks the driver whether DLSS can run. Logs why not.
void init(const Vk& vk);
bool available();
// Available, and nothing failed since (evaluate turns it off for the run when it fails).
bool usable();
void shutdown();

// Whether the build has DLSS (LPN_HAVE_DLSS or LPN_HAVE_STREAMLINE).
bool builtIn();

// Records DLSS into cmd, which runs on the GPU queue the renderer uses; the images are in the
// GENERAL layout and the caller's barriers order everything around it. Makes the feature first
// (again when a size or the quality changes). False when DLSS cannot run (logged once).
bool evaluate(VkCommandBuffer cmd, VkDevice device, const upscale::Frame& f);

} // namespace dlss
} // namespace host::render
