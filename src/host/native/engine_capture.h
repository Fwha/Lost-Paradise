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

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
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

// LPN_ENGINE_SCENE_CHECK: the scene's draw-command-list draws (back at 0x3700) described from the
// engine's state alone (docs/engine/user-data.md): each stage's code and the user data the engine
// is about to give it (descriptors in SGPRs, pointers into its descriptor table and that table),
// checked by the renderer against the draw's registers. A NOP {kEngineCheckTag, id} goes ahead of
// the draw (in place of its origin NOP); the draw itself still runs.
constexpr uint32_t kEngineCheckTag = 0x4B43454C; // "LECK"
inline thread_local uint32_t tEngineCheckId = 0;
struct EngineStage {
    uint64_t code = 0;
    uint32_t sgpr[16] = {};
    uint16_t sgprKnown = 0;          // bit per SGPR the engine sets with a descriptor or a value
    struct Pointer {
        uint8_t sgpr;                // the pair's first SGPR
        uint32_t dword;              // where it points in the table (dwords)
    } pointers[8] = {};
    uint32_t pointerCount = 0;
    uint8_t fetchSgpr = 0xFF;        // VS: the fetch shader's pair (M+4), set once drawIndex installs it
    std::vector<uint32_t> table;     // the descriptor table as the engine is about to copy it
    std::vector<uint8_t> tableKnown; // the dwords placed for this draw (the rest is stale)
    bool ok = false;
};
struct EngineSceneDraw {
    EngineStage vs, ps;
    bool attributes = false; // the vertex attributes generated for it are in vs's table
};
bool engineSceneCheckOn();
std::shared_ptr<const EngineSceneDraw> engineScene(uint32_t id);
// The placed table dwords as decodeStage had them before drawIndex: [0] the same after it, [1] not.
void engineTableAfter(uint64_t out[2]);

// LPN_ENGINE_FEED: the native passes' list draws (material, character) analysed from the engine's
// user data instead of the registers': each stage's SGPRs as the engine sets them, its pointers
// into the descriptor table moved to a copy of the engine's table at an address of its own (the
// feed range, served to the analysis by feedRead). The draws carry the capture's id as the scene
// check's do (DrawCall::engineCheck); with LPN_ENGINE_SCENE_CHECK too, each is also checked.
bool engineFeedOn();
uint64_t engineFeedCaptureNs(); // LPN_PROFILE: game threads' time capturing, in all
constexpr uint64_t kFeedBase = 0x7F0000000000ull, kFeedSlotBytes = 0x2000;
// Frames of draws: a slot is reused long after the analyses and the recorder are done with it
// (and never between two analyses of one worker, so the analysis' repeat shortcut holds).
constexpr uint32_t kFeedSlots = 1 << 16;
inline bool feedAddress(uint64_t a) { return a - kFeedBase < uint64_t(kFeedSlots) * kFeedSlotBytes; }
// The address a copy of these dwords is served at (the same for the same dwords while they stay
// in the ring), 0 when it doesn't fit.
uint64_t feedTable(const uint32_t* words, uint32_t n);
bool feedRead(uint64_t addr, void* dst, size_t size);

// LPN_ENGINE_PREPASS: the list prepass (vs_normal_rigid[_pt] with ps_normal[_pt]) drawn from the
// engine's description (docs/engine/, Codex's design): the engine still prepares the draw, but the
// packet helper writes a NOP {kEnginePrepassTag, id, count, index address} in place of its draw
// packet, and the renderer draws the capture natively there (or the draw as it was, from the NOP's
// arguments, when it cannot).
constexpr uint32_t kEnginePrepassTag = 0x5052504C; // "LPRP"
inline thread_local uint32_t tEnginePrepassId = 0;   // id + 1
struct EnginePrepass {
    std::string vsName, psName;
    EngineSceneDraw stages;
    struct Input {
        uint32_t sharp[4] = {};
        uint32_t slot = 0, components = 0, indexVgpr = 0, offset = 0;
    };
    struct Image {
        uint32_t t[8] = {}, s[4] = {};
        uint32_t pc = 0;
    };
    std::vector<Input> vertex;
    std::vector<Image> images;
    std::map<uint32_t, std::vector<uint32_t>> vsLoads, psLoads; // load pc: the values it reads
    uint32_t count = 0, first = 0, baseVertex = 0;
    uint32_t indexWord = 0, indexModifier = 0;
    uint64_t indexAddr = 0;
    bool alpha = false;
};
std::shared_ptr<const EnginePrepass> enginePrepass(uint32_t id);
// Whether the capture behind a NOP is there and is the draw the NOP stands for (the same answer on
// the command processor's analysis and drawing passes).
bool enginePrepassUsable(uint32_t id, uint32_t count, uint64_t indexAddr);
// Set by the renderer each frame: whether it draws these natively (Native Renderer on, no upscaler).
void setEnginePrepassAllowed(bool on);
bool enginePrepassOn();
// LPN_ENGINE_PREPASS_CHECK: the captures are made but the game's draws still run, each carrying its
// capture's id (DrawCall::engineCheck), and the renderer checks the capture against the draw's analysis.
bool enginePrepassCheckOn();

} // namespace host::native
