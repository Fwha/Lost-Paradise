// GCN shader programs: what a draw or dispatch runs, analysed against the state it was
// launched with, and translated to GLSL for Vulkan.
//
// The scalar half of a GCN program is uniform across the wave, so it is evaluated on the CPU
// at draw time: that resolves every resource descriptor (T#, S#, V#) and every scalar memory
// load to concrete values. The vector half becomes GLSL, one invocation per lane; per-lane
// masks (exec, vcc) are kept as that lane's own bit.
#pragma once

#include "gcn/decoder.h"

#include <cstdint>
#include <functional>
#include <atomic>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace gcn {

enum class Stage : uint8_t { Vertex, Pixel, Compute };

// Reads guest memory; returns false when the range is not readable.
using ReadMemory = std::function<bool(uint64_t addr, void* dst, size_t size)>;

// Hardware state a shader starts from (the SPI/COMPUTE registers of the draw).
struct LaunchState {
    Stage stage = Stage::Vertex;
    uint64_t entry = 0;
    uint32_t userData[16] = {};
    uint32_t rsrc1 = 0, rsrc2 = 0;
    // Vertex
    bool glClipZ = false; // clip space z in [-w, w] rather than [0, w]
    // Pixel
    uint32_t psInputEna = 0;
    uint32_t psInputCntl[32] = {};
    uint32_t numInterp = 0;
    uint32_t colorFormat = 0; // SPI_SHADER_COL_FORMAT, 4 bits per MRT
    uint32_t colorUint = 0;   // bit n: MRT n is an unsigned integer target
    uint32_t colorSint = 0;   // bit n: MRT n is a signed integer target
    uint8_t colorSwap[8] = {}; // CB_COLOR_INFO.COMP_SWAP per MRT
    uint8_t colorComps[8] = {}; // components of each MRT's format
    bool depthExport = false;
    // Compute
    uint32_t localSize[3] = {1, 1, 1};

    uint32_t userSgprs() const { return (rsrc2 >> 1) & 0x1F; }
};

// Layout of the per-draw data buffer every translated shader reads (binding 0 of its range).
namespace data {
constexpr uint32_t kUserData = 0;     // 16 dwords of user SGPRs
constexpr uint32_t kPosTransform = 16; // VS: position xy scale, then offset (times w); floats
constexpr uint32_t kColorSwizzle = 16; // PS, generic translation: per MRT a byte, components * 4 + COMP_SWAP (2 words)
constexpr uint32_t kSmrd = 20;        // scalar load results follow
} // namespace data

struct BufferResource {
    uint64_t pc = 0;
    uint32_t sharp[4] = {};
    uint32_t dfmt = 0, nfmt = 0; // element format (from the instruction for tbuffer, else the V#)
    bool store = false;
    bool known = true; // false when the V# could not be evaluated at draw time
    uint32_t data = 0; // dword offset in draw data: byte offset in binding, stride, num_records, size
    uint64_t base() const { return sharp[0] | (uint64_t(sharp[1] & 0xFFF) << 32); }
    uint32_t stride() const { return (sharp[1] >> 16) & 0x3FFF; }
    uint32_t numRecords() const { return sharp[2]; }
};

struct ImageResource {
    uint64_t pc = 0;
    uint32_t t[8] = {};
    uint32_t s[4] = {};
    bool storage = false; // image_store
    bool compare = false; // depth compare sampling
    uint32_t data = 0;    // dword offset in draw data: uv scale x, y (float)
    uint64_t base() const { return (t[0] | (uint64_t(t[1] & 0x3F) << 32)) << 8; }
    uint32_t dataFormat() const { return (t[1] >> 20) & 0x3F; }
    uint32_t numFormat() const { return (t[1] >> 26) & 0xF; }
    uint32_t width() const { return (t[2] & 0x3FFF) + 1; }
    uint32_t height() const { return ((t[2] >> 14) & 0x3FFF) + 1; }
    uint32_t dstSel() const { return t[3] & 0xFFF; }
    uint32_t type() const { return t[3] >> 28; }
    uint32_t tileMode() const { return (t[3] >> 20) & 0x1F; }
    uint32_t baseLevel() const { return (t[3] >> 12) & 0xF; }
    uint32_t lastLevel() const { return (t[3] >> 16) & 0xF; }
    bool pow2Pad() const { return (t[3] >> 25) & 1; }
    uint32_t depth() const { return (t[4] & 0x1FFF) + 1; }
    uint32_t pitch() const { return ((t[4] >> 13) & 0x3FFF) + 1; }
    uint32_t baseArray() const { return t[5] & 0x1FFF; }
    uint32_t lastArray() const { return (t[5] >> 13) & 0x1FFF; }
};

struct ReplayPlan; // a trace pre-decoded for fast replays (see replay())

// Holds a program's replay plan; copies share it. Built lazily by whichever thread replays first.
class PlanSlot {
public:
    PlanSlot() = default;
    PlanSlot(const PlanSlot& o) : p_(o.p_.load()) {}
    PlanSlot& operator=(const PlanSlot& o) {
        p_.store(o.p_.load());
        return *this;
    }
    std::shared_ptr<const ReplayPlan> get() const { return p_.load(std::memory_order_acquire); }
    void set(std::shared_ptr<const ReplayPlan> v) const { p_.store(std::move(v), std::memory_order_release); }

private:
    mutable std::atomic<std::shared_ptr<const ReplayPlan>> p_;
};

// A shader as launched by one draw: its reachable code, resolved resources and scalar loads.
struct Program {
    LaunchState launch;
    std::map<uint64_t, Instruction> code;
    std::set<uint64_t> labels; // addresses control can reach other than by falling through
    std::map<uint64_t, uint64_t> calls; // s_swappc_b64 site -> resolved callee
    bool controlFlow = false;
    bool writesExec = false;
    std::map<uint64_t, uint32_t> smrd; // pc -> dword offset of its result in `data`
    std::vector<uint32_t> data;        // per-draw data; resource fields are filled by the renderer
    std::vector<BufferResource> buffers;
    std::vector<ImageResource> images;
    std::vector<std::string> warnings;

    // The evaluation as it ran (work items started and instructions stepped, in order), so a
    // launch of the same code with other user data or memory can be re-evaluated without walking
    // and decoding the code again (see refresh()). Empty when it cannot be replayed.
    struct TraceStep {
        uint64_t pc = 0;
        Instruction in;
        bool start = false;  // a work item starts at `pc` (no instruction)
        bool result = false; // whether the path went on after the instruction
        uint32_t slot = ~0u;  // SMRD: its result's offset in `data`; MUBUF/MTBUF/MIMG: its resource's index
    };
    std::vector<TraceStep> trace;
    PlanSlot plan; // `trace` pre-decoded, once it has been replayed (LPN_FAST_REPLAY)

    // Identifies the translation: equal keys translate to the same GLSL.
    std::string key() const;
};

// A program as translate() needs it (no trace or warnings), for re-translating it in a later run
// without the game's memory: the startup pipeline builds keep these beside their SPIR-V.
std::string serialize(const Program& program);
// False when `bytes` is damaged or from a build whose structures differ.
bool deserialize(const std::string& bytes, Program& program);

// Generic translation (LPN_GENERIC_SHADERS=1; off until every stage has landed, as its cache starts
// empty): analyze() keeps the scalar branches a
// launch decided as run-time branches and walks the code they skip too, so launches of a shader
// that differ only in such flags translate alike (docs/shader-precompile.md, stage 1).
bool genericTranslation();

// The dimension the translation declares image `im` with (a T# type code): the T#'s when it was
// resolved, else worked out from the code before the instruction.
uint32_t imageDimension(const Program& program, const ImageResource& im);

// `readCode` fetches instructions, `readData` scalar loads.
Program analyze(const LaunchState& launch, const ReadMemory& readCode, const ReadMemory& readData);
inline Program analyze(const LaunchState& launch, const ReadMemory& read) { return analyze(launch, read, read); }

// Re-evaluates `program` (analysed from the same code and launch registers) for a launch with
// other user data or memory by replaying its trace. False when a branch would go another way;
// otherwise `out` holds the launch's data and resources (and no code).
bool replay(const Program& program, const LaunchState& launch, const ReadMemory& readData, Program& out);
// Fast replays (LPN_FAST_REPLAY=1, or =check to also run the full replay and compare): each
// trace is pre-decoded once into ops, and scalar loads, moves and SGPR kills run straight from
// them. Counts since the last call: replays run from a plan, their trace steps and how many ran
// directly, check-mode mismatches, and differences that two full replays showed as well (memory
// changing while it was read).
struct ReplayStats {
    uint64_t planned = 0, steps = 0, kept = 0, mismatches = 0, races = 0;
};
ReplayStats takeReplayStats();

// Moves a replay's launch, data and resources into `program` when they translate alike (the
// resources keep their formats, see Program::key). False leaves both unchanged.
bool adopt(Program& program, Program& replayed);
// replay() then adopt(): re-targets `program` to the launch, or returns false leaving it unchanged.
bool refresh(Program& program, const LaunchState& launch, const ReadMemory& readData);

struct Translation {
    std::string glsl;
    uint32_t outputs = 0; // VS: param exports written (bit = location)
    uint32_t inputs = 0;  // PS: input locations read
    uint32_t colorOutputs = 0; // PS: MRTs written
    std::vector<std::string> warnings;
};

// Bindings: draw data at `bindingBase`, then one per buffer, then one per image.
Translation translate(const Program& program, uint32_t bindingBase);

// GLSL to SPIR-V (glslang). Returns an empty vector and fills `error` on failure.
enum class GlslStage { Vertex, Geometry, Fragment, Compute };
std::vector<uint32_t> compileGlsl(const std::string& source, GlslStage stage, std::string* error);

} // namespace gcn
