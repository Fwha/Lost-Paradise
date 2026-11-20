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
};

// Whether a draw's origin (DrawCall::origin[0]) is a site the capture covers.
bool engineSite(uint32_t origin);

// Command processor thread: the next captured draw, false when none is queued.
bool takeEngineDraw(EngineDraw& out);

bool engineCheckOn();

} // namespace host::native
