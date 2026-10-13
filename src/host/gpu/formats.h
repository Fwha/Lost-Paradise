// GCN surface and texture formats to Vulkan. Formats are in memory order: component 0 is the
// lowest bits in memory, whatever channel the hardware swizzles it to.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace host::render {

// Texture/buffer number formats (T# NUM_FORMAT).
enum NumFormat : uint32_t {
    kNumUnorm = 0, kNumSnorm = 1, kNumUscaled = 2, kNumSscaled = 3, kNumUint = 4, kNumSint = 5,
    kNumFloat = 7, kNumSrgb = 9,
};

// T# DATA_FORMAT and NUM_FORMAT. Returns VK_FORMAT_UNDEFINED when unsupported.
VkFormat textureFormat(uint32_t dataFormat, uint32_t numFormat);
// CB_COLOR_INFO FORMAT and NUMBER_TYPE.
VkFormat colorFormat(uint32_t format, uint32_t numberType);
// Number type of a color buffer as a texture number format.
uint32_t colorNumFormat(uint32_t numberType);

uint32_t bytesPerElement(uint32_t dataFormat); // block size for BC formats
bool isBlockCompressed(uint32_t dataFormat);
bool isDepthFormat(VkFormat f);

} // namespace host::render
