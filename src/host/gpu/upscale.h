// The temporal upscalers that can replace the game's SMAA (common/settings_file.h's Upscaler):
// NVIDIA DLSS (gpu/dlss.h) and AMD FSR 3.1 (gpu/fsr.h). Both take the same frame: the scene's
// colour, depth and motion vectors at the render size, the sub-pixel jitter the scene was drawn
// with, and an output image at the render size (Native: anti-aliasing only) or larger.
// The renderer's side (finding the scene, the jitter, the motion vectors) is in renderer.cpp.
#pragma once

#include "common/settings_file.h"

#include <vulkan/vulkan.h>

#include <cstdint>

namespace host::render {

struct Vk;

namespace upscale {

// An image an upscaler reads or writes: its view covers one 2D level and layer (depth: the depth
// aspect). Every image is in the GENERAL layout.
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t width = 0, height = 0;
    VkImageUsageFlags usage = 0; // what the image was made with
};

struct Frame {
    Image color, depth, motion, output; // motion: RG16F, pixels to the previous frame (render size)
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    float jitterX = 0, jitterY = 0; // the sample's offset from the pixel centre, in render pixels (y down)
    bool reset = false;             // no usable history (first frame, camera cut)
    UpscaleQuality quality = UpscaleQuality::Native;
    // The camera, for FSR: view distance at depth 0 and at depth 1 (either may be huge), the
    // vertical field of view, and the time since the last frame.
    float depth0 = 0.1f, depth1 = 1000.0f, fovY = 1.0f, frameMs = 16.7f;
};

// Whether this build has the upscaler (its SDK was there at build time).
bool builtIn(Upscaler u);
// Whether it can run on this device (after init), and nothing failed since.
bool usable(Upscaler u);

// The sub-pixel offset to draw frame `n` at: a Halton (2, 3) sequence, its length suited to the
// upscale (8 phases at Native, 8 x ratio^2 above), in render pixels within [-0.5, 0.5).
void jitter(uint64_t n, uint32_t renderW, uint32_t outW, float* x, float* y);

} // namespace upscale
} // namespace host::render
