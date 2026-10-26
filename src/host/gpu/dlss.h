// NVIDIA DLSS through the DLSS SDK's NGX library, on the renderer's Vulkan device. Built in when
// the SDK is in third_party/dlss (scripts/fetch-dlss.sh, LPN_HAVE_DLSS); used when the driver
// offers it (an RTX card with NVIDIA's driver). Otherwise every call here is a no-op.
#pragma once

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
void shutdown();

// The size to render at for an output of outW x outH in a PC settings mode (display_settings.h's
// DlssMode); false when DLSS is not available.
bool renderSize(int mode, uint32_t outW, uint32_t outH, uint32_t* w, uint32_t* h);

} // namespace dlss
} // namespace host::render
