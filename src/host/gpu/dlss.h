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
// Available, and nothing failed since (evaluate turns it off for the run when it fails).
bool usable();
void shutdown();

// An image DLSS reads or writes: its view covers one 2D level and layer (depth: the depth aspect).
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t width = 0, height = 0;
};

struct Frame {
    Image color, depth, motion, output; // motion: RG16F, pixels to the previous frame (render size)
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    float jitterX = 0, jitterY = 0; // the sample's offset from the pixel centre, in render pixels
    bool reset = false;             // no usable history (first frame, camera cut)
    int mode = 1;                   // DlssMode
};

// Records DLSS into cmd, which runs on the GPU queue the renderer uses; the images are in the
// GENERAL layout and the caller's barriers order everything around it. Makes the feature first
// (again when a size or the mode changes). False when DLSS cannot run (logged once).
bool evaluate(VkCommandBuffer cmd, VkDevice device, const Frame& f);

// The sub-pixel offset to render frame `n` at: a Halton (2, 3) sequence, its length suited to the
// upscale, in render pixels within [-0.5, 0.5).
void jitter(uint64_t n, uint32_t renderW, uint32_t outW, float* x, float* y);

// The size to render at for an output of outW x outH in a PC settings mode (display_settings.h's
// DlssMode); false when DLSS is not available.
bool renderSize(int mode, uint32_t outW, uint32_t outH, uint32_t* w, uint32_t* h);

} // namespace dlss
} // namespace host::render
