// The engine's own description of its draws, read where it makes them: the step from drawing
// what the game's PS4 command buffers say to drawing what the engine holds (GR2port's capture).
//
// For now it covers the stage (static world) shadow casters: 0xa51100 walks the casters of
// stage+0x2268 and draws merged runs of their meshes through the engine's drawIndex (0xeac430),
// with the vertex buffer at state+0x288 and the index buffer at state+0x688 of the Gnm context's
// state block (docs/render-decomp.md). A hook on 0xeac430 keeps each such draw as the engine
// describes it, and the renderer checks it against the draw that reaches the command processor
// (LPN_ENGINE_CHECK=1): when they agree, the capture can take the game's place.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace host {
class Module;
}

namespace host::native {

// Before relocation.
bool installEngineCapture(Module& eboot);

// A draw as the engine made it.
struct EngineDraw {
    uint32_t site = 0;        // eboot offset of the return address of its drawIndex call
    uint64_t vertexBase = 0;  // the vertex buffer as bound, plus base vertex times stride
    uint32_t stride = 0;
    uint64_t indexAddr = 0;   // the first index drawn
    uint32_t count = 0;       // indices
    uint64_t constants[4] = {}; // the vertex stage's first constant buffer slots (state block)
    uint64_t view = 0;        // the view constants' V# (state block +0x20c8: base in the low 40 bits)
    uint32_t vertexFormat = 0; // state block +0x298 (stride in bits 4-19, low 4 bits from the buffer)
    // The second fetch slot (the alpha-tested casters' coordinates), from the fetch metadata at
    // state +0x698: its byte offset in the vertex, stream, and V# word 3 (formats, DST_SEL).
    uint32_t uvOffset = 0, uvStream = 0, uvWord3 = 0;
    bool uvKnown = false;
    uint32_t attributes[16][2] = {}; // the fetch metadata's table (for finding the coordinates' entry)
    uint32_t attributeCount = 0;
    // The pixel shader's texture read from user SGPR 0 (state +0x8e8 by the slot whose entry in
    // the shader's table at *(context+0xb8)+0x1c maps to it): its T#.
    uint32_t texture[8] = {};
    int32_t textureSlot = -1;
    // For finding the view constants: the first draws keep copies of the Gnm context (0x400
    // bytes) and its state block (0x4000) as they were at the call.
    std::shared_ptr<std::vector<uint8_t>> context, state, table; // table: *(context+0x68), 0x3000 bytes
};

// With Native Renderer on (LPN_ENGINE_SHADOWS=0 turns it off): the stage's shadow casters (both
// classes: the opaque sites 0xa5177a and 0xa518fe, the alpha-tested 0xa51b79 and 0xa51d37) are
// drawn from the engine's description alone. The game's
// state commit still runs, but in place of the draw packet the packet helper writes a NOP
// carrying the capture's id (kEngineTag), and the renderer draws the capture natively there.
constexpr uint32_t kEngineTag = 0x454E504C; // "LPNE"
// Set by the hook for the draw this thread writes next: its capture's id + 1.
inline thread_local uint32_t tEngineDrawId = 0;
struct EngineCaster {
    EngineDraw draw;
    uint32_t vertices = 0;     // the vertex buffer's capacity (state +0x294)
    bool index32 = false;
    bool alpha = false; // the alpha-tested class: draw.texture and draw.uv* are set
    float projection[16], rotation[12], camera[4]; // the view constants (proj rows, view rotation rows, camera)
};
// Render side: the capture behind a NOP's id, false when it is gone (overwritten).
bool engineCaster(uint32_t id, EngineCaster& out);
bool engineShadowsOn();

// Whether a draw's origin (DrawCall::origin[0]) is a site the capture covers.
bool engineSite(uint32_t origin);

// Command processor thread: the next captured draw, false when none is queued.
bool takeEngineDraw(EngineDraw& out);

bool engineCheckOn();

} // namespace host::native
