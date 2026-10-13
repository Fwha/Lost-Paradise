#include "host/gpu/formats.h"

namespace host::render {

namespace {

// One Vulkan format per number format: unorm, snorm, uscaled, sscaled, uint, sint, float, srgb.
struct Row {
    VkFormat unorm, snorm, uscaled, sscaled, uint, sint, sfloat, srgb;
};

VkFormat pick(const Row& r, uint32_t num) {
    switch (num) {
    case kNumUnorm: return r.unorm;
    case kNumSnorm: case 6: return r.snorm;
    case kNumUscaled: return r.uscaled;
    case kNumSscaled: return r.sscaled;
    case kNumUint: return r.uint;
    case kNumSint: return r.sint;
    case kNumFloat: return r.sfloat;
    case kNumSrgb: return r.srgb ? r.srgb : r.unorm;
    default: return VK_FORMAT_UNDEFINED;
    }
}

constexpr VkFormat U = VK_FORMAT_UNDEFINED;

} // namespace

VkFormat textureFormat(uint32_t df, uint32_t nf) {
    switch (df) {
    case 1: return pick({VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_USCALED, VK_FORMAT_R8_SSCALED,
                         VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, U, VK_FORMAT_R8_SRGB}, nf);
    case 2: return pick({VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_USCALED, VK_FORMAT_R16_SSCALED,
                         VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R16_SFLOAT, U}, nf);
    case 3: return pick({VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_SSCALED,
                         VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, U, VK_FORMAT_R8G8_SRGB}, nf);
    case 4: return pick({U, U, U, U, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT, U}, nf);
    case 5: return pick({VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_USCALED,
                         VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT,
                         VK_FORMAT_R16G16_SFLOAT, U}, nf);
    case 7: return nf == kNumFloat ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : U;
    case 8: case 9:
        return pick({VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_SNORM_PACK32,
                     VK_FORMAT_A2B10G10R10_USCALED_PACK32, VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
                     VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32, U, U}, nf);
    case 10: return pick({VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_USCALED,
                          VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, U,
                          VK_FORMAT_R8G8B8A8_SRGB}, nf);
    case 11: return pick({U, U, U, U, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SFLOAT, U}, nf);
    case 12: return pick({VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_USCALED,
                          VK_FORMAT_R16G16B16A16_SSCALED, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT,
                          VK_FORMAT_R16G16B16A16_SFLOAT, U}, nf);
    case 13: return pick({U, U, U, U, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
                          VK_FORMAT_R32G32B32_SFLOAT, U}, nf);
    case 14: return pick({U, U, U, U, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
                          VK_FORMAT_R32G32B32A32_SFLOAT, U}, nf);
    case 16: return nf == kNumUnorm ? VK_FORMAT_B5G6R5_UNORM_PACK16 : U;
    case 17: return nf == kNumUnorm ? VK_FORMAT_A1R5G5B5_UNORM_PACK16 : U;
    case 18: return nf == kNumUnorm ? VK_FORMAT_R5G5B5A1_UNORM_PACK16 : U;
    case 19: return nf == kNumUnorm ? VK_FORMAT_B4G4R4A4_UNORM_PACK16 : U;
    case 35: return nf == kNumSrgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case 36: return nf == kNumSrgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK;
    case 37: return nf == kNumSrgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
    case 38: return nf == kNumSnorm ? VK_FORMAT_BC4_SNORM_BLOCK : VK_FORMAT_BC4_UNORM_BLOCK;
    case 39: return nf == kNumSnorm ? VK_FORMAT_BC5_SNORM_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK;
    case 40: return nf == kNumSnorm ? VK_FORMAT_BC6H_SFLOAT_BLOCK : VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case 41: return nf == kNumSrgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    default: return U;
    }
}

uint32_t colorNumFormat(uint32_t numberType) { return numberType == 6 ? kNumSrgb : numberType; }

VkFormat colorFormat(uint32_t format, uint32_t numberType) { return textureFormat(format, colorNumFormat(numberType)); }

uint32_t bytesPerElement(uint32_t df) {
    switch (df) {
    case 1: return 1;
    case 2: case 3: case 16: case 17: case 18: case 19: return 2;
    case 4: case 5: case 6: case 7: case 8: case 9: case 10: return 4;
    case 11: case 12: return 8;
    case 13: return 12;
    case 14: return 16;
    case 35: case 38: return 8;
    case 36: case 37: case 39: case 40: case 41: return 16;
    default: return 4;
    }
}

bool isBlockCompressed(uint32_t df) { return df >= 35 && df <= 41; }

bool isDepthFormat(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           f == VK_FORMAT_D24_UNORM_S8_UINT;
}

} // namespace host::render
