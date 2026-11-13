// AMD FSR 3.1 temporal upscaling through FSR-Vulkan (github.com/FireBurn/FSR-Vulkan, MIT: AMD's
// FidelityFX SDK 1.1.4 upscaler and its Vulkan backend behind a C API), on the renderer's
// device. Built in when the source is in third_party/fsr-vulkan (scripts/fetch-fsr.sh,
// LPN_HAVE_FSR); plain compute shaders, so it runs on any GPU that has the few features it needs.
#pragma once

#include "host/gpu/upscale.h"

#include <vulkan/vulkan.h>

namespace host::render {

struct Vk;

namespace fsr {

bool builtIn();
// After the device is made: checks the device runs FSR. Logs why not.
void init(const Vk& vk);
bool available();
// Available, and nothing failed since (evaluate turns it off for the run when it fails).
bool usable();
// With the GPU idle.
void shutdown();

// Records FSR into cmd (the images in the GENERAL layout, the caller's barriers around it).
// Makes the context first (again when a size or the depth's direction changes). False when FSR
// cannot run (logged once).
bool evaluate(VkCommandBuffer cmd, const upscale::Frame& f);

} // namespace fsr
} // namespace host::render
