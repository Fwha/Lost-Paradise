// NVIDIA DLSS through the DLSS SDK's NGX library, on the renderer's Vulkan device. Built in when
// the SDK is in third_party/dlss (scripts/fetch-dlss.sh, LPN_HAVE_DLSS); used when the driver
// offers it (an RTX card with NVIDIA's driver). Otherwise every call here is a no-op.
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

// After the device is made: starts NGX and asks the driver whether DLSS can run. Logs why not.
void init(const Vk& vk);
bool available();
// Available, and nothing failed since (evaluate turns it off for the run when it fails).
bool usable();
void shutdown();

// Whether the build has the DLSS SDK (LPN_HAVE_DLSS).
bool builtIn();

// Records DLSS into cmd, which runs on the GPU queue the renderer uses; the images are in the
// GENERAL layout and the caller's barriers order everything around it. Makes the feature first
// (again when a size or the quality changes). False when DLSS cannot run (logged once).
bool evaluate(VkCommandBuffer cmd, VkDevice device, const upscale::Frame& f);

} // namespace dlss
} // namespace host::render
