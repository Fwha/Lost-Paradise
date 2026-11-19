// The native renderer: passes of the game's frame drawn by our own Vulkan code and handwritten
// GLSL instead of the game's recompiled shaders (docs/native-renderer.md). For now a native pass
// takes the place of the game's draws of one pass, found by the names of their pixel shaders
// (shader_names.h), and draws into the same render targets, so the passes around it, still drawn
// by the translated renderer, see no difference. It knows nothing of PM4, GCN or the renderer's
// caches: the renderer hands it a Draw and a Gpu, and the same passes will later take the scene
// captured from the engine instead.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace host::native {

// A texture the game's draw samples, as the renderer bound it.
struct Texture {
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    // Multiplies normalized coordinates: the image can be larger than the texture the game
    // describes (a render target padded to 8 pixels, sampled as a texture of its picture's size).
    float scale[2] = {1, 1};
    uint64_t address = 0; // in guest memory
    uint32_t width = 0, height = 0; // as the game describes it
    uint32_t pc = 0; // the sample instruction it was found at, as an offset from the shader's start
};

// A render target the draw writes.
struct Target {
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0; // of the image
    uint64_t address = 0;
    // CB_COLOR_INFO.COMP_SWAP and the format's components: the shader's outputs go to memory in
    // the order swizzle() gives.
    uint8_t swap = 0, comps = 4;
};

// Where each of the target's channels comes from among the shader's outputs (x, y, z, w = 0..3),
// as the GCN colour export orders them.
void swizzle(const Target& t, int32_t out[4]);

// One of the game's draws that a native pass claimed.
struct Draw {
    const std::string* vsName = nullptr;
    const std::string* psName = nullptr;
    Target color[8];
    uint32_t colorCount = 0;     // targets 0..colorCount-1 (unwritten ones have no view)
    uint32_t writeMask = 0;      // 4 bits per target
    bool blend = false;          // any target blends
    bool depthTest = false, depthWrite = false, stencil = false;
    // The depth/stencil buffer bound (no view: none) and its state as the translated pipeline
    // would have it; the masks and reference per face (front, back) are dynamic.
    Target depth;
    bool hasStencil = false; // the depth format has a stencil aspect
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    uint32_t stencilRef[2] = {}, stencilCompare[2] = {}, stencilWrite[2] = {};
    float viewport[4] = {};      // x, y, width, height
    int32_t scissor[4] = {};     // x, y, width, height
    uint32_t primitive = 0;      // the game's primitive type (17: a rectangle list)
    uint32_t count = 0;          // vertices or indices
    // The pixel shader's sampled images, in the order the shader declares them.
    std::vector<Texture> textures;
    // `n` dwords of the scalar load at `pc` (an offset from the pixel shader's start), as the
    // shader read them for this draw. False when the shader has no such load.
    std::function<bool(uint32_t pc, uint32_t* out, uint32_t n)> load;
};

// What the renderer gives a native pass to draw with.
class Gpu {
public:
    virtual ~Gpu() = default;
    virtual VkDevice device() const = 0;
    virtual PFN_vkCmdPushDescriptorSetKHR pushDescriptorSet() const = 0;
    // A shader module from GLSL (compiled once, kept in the SPIR-V cache); null on an error.
    enum class Stage { Vertex, Fragment, Compute };
    virtual VkShaderModule shader(const std::string& glsl, Stage stage) = 0;
    // Queues Vulkan commands, in order with the translated draws around them, outside any render
    // pass and after a full barrier. `fn` runs later, on another thread: it must hold copies of
    // everything it needs.
    virtual void record(std::function<void(VkCommandBuffer)> fn) = 0;
};

class Pass {
public:
    virtual ~Pass() = default;
    virtual const char* name() const = 0;
    // The role this pass gives the game's pixel shader `psName` (an archive entry name, such as
    // "ps_aa_filter2.pso"), or 0 when the pass does not draw it. Asked once per shader.
    virtual int claims(const std::string& psName) const = 0;
    // Draws d in the game's place. False when this draw is not one the pass rebuilds (another
    // state, an input it does not know): nothing is recorded and the translated renderer draws it.
    virtual bool draw(int role, const Draw& d, Gpu& gpu) = 0;
    // Each frame's end (temporal state, statistics).
    virtual void frameEnd() {}
};

// Every native pass, in no particular order.
std::span<Pass* const> passes();

// The pass and role for a pixel shader (see Pass::claims): {nullptr, 0} when none claims it.
struct Claim {
    Pass* pass = nullptr;
    int role = 0;
};
Claim claim(const std::string& psName);

// A fullscreen-triangle vertex shader (no inputs) that native passes share.
extern const char* const kFullscreenVs;

} // namespace host::native
