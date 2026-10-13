#include "host/gpu/frame_dump.h"

#include "host/pm4.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>

namespace host {

namespace {

using namespace gpu;
namespace fs = std::filesystem;

uint64_t shaderAddress(const GpuRegs& r, uint32_t loReg) {
    return ((uint64_t(r.sh[loReg + 1]) << 32) | r.sh[loReg]) << 8;
}

// PS4 shader code is followed by a ShaderBinaryInfo block starting with "OrbShdr".
size_t shaderSize(const uint8_t* code) {
    for (size_t i = 0; i < 0x40000; i += 4)
        if (std::memcmp(code + i, "OrbShdr", 7) == 0) return i + 0x1C;
    return 0;
}

const char* kindName(DrawCall::Kind k) {
    switch (k) {
    case DrawCall::kIndexed: return "draw indexed";
    case DrawCall::kAuto: return "draw auto";
    case DrawCall::kIndexedIndirect: return "draw indexed indirect";
    case DrawCall::kIndirect: return "draw indirect";
    case DrawCall::kDispatch: return "dispatch";
    case DrawCall::kDispatchIndirect: return "dispatch indirect";
    }
    return "?";
}

class FrameDump : public DrawListener {
public:
    FrameDump(unsigned frame, std::string dir) : target_(frame), dir_(std::move(dir)) {}

    void onDraw(const DrawCall& d, const GpuRegs& r) override {
        if (frame_ != target_) return;
        if (!out_) {
            fs::create_directories(dir_ + "/shaders");
            out_ = std::fopen((dir_ + "/frame.txt").c_str(), "w");
            std::fprintf(stderr, "[gpu] dumping frame %u to %s\n", frame_, dir_.c_str());
        }
        ++index_;
        std::fprintf(out_, "#%u %s", index_, kindName(d.kind));
        if (d.kind == DrawCall::kDispatch)
            std::fprintf(out_, " %ux%ux%u groups", d.groups[0], d.groups[1], d.groups[2]);
        else if (d.kind < DrawCall::kIndexedIndirect)
            std::fprintf(out_, " count %u instances %u prim %u", d.count, r.numInstances,
                         r.uconfig[uconfig::kVgtPrimitiveType]);
        if (d.kind == DrawCall::kIndexed)
            std::fprintf(out_, " index 0x%" PRIx64 "+%u %s", d.indexAddr, d.indexOffset,
                         r.indexType ? "u32" : "u16");
        std::fprintf(out_, "\n");

        if (d.kind >= DrawCall::kDispatch) {
            shader("cs", shaderAddress(r, sh::kComputePgmLo), &r.sh[sh::kComputeUserData], 16);
            std::fprintf(out_, "  threads %ux%ux%u\n", r.sh[sh::kComputeNumThreadX], r.sh[sh::kComputeNumThreadX + 1],
                         r.sh[sh::kComputeNumThreadX + 2]);
            return;
        }
        shader("vs", shaderAddress(r, sh::kSpiShaderPgmLoVs), &r.sh[sh::kSpiShaderUserDataVs], 16);
        shader("ps", shaderAddress(r, sh::kSpiShaderPgmLoPs), &r.sh[sh::kSpiShaderUserDataPs], 16);

        const uint32_t targetMask = r.ctx[0x08E]; // CB_TARGET_MASK
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t* cb = &r.ctx[ctx::kCbColor0Base + i * ctx::kCbColorStride];
            const uint32_t info = cb[ctx::kCbColorInfo];
            const uint32_t format = (info >> 2) & 0x1F;
            if (!format || !((targetMask >> (i * 4)) & 0xF)) continue;
            const uint32_t pitch = ((cb[ctx::kCbColorPitch] & 0x7FF) + 1) * 8;
            const uint32_t slice = ((cb[ctx::kCbColorSlice] & 0x3FFFFF) + 1) * 64;
            std::fprintf(out_, "  rt%u 0x%" PRIx64 " %ux%u format %u numtype %u tile %u\n", i,
                         uint64_t(cb[0]) << 8, pitch, slice / pitch, format, (info >> 8) & 7,
                         cb[ctx::kCbColorAttrib] & 0x1F);
        }
        const uint32_t zInfo = r.ctx[ctx::kDbZInfo];
        if (zInfo & 3) {
            const uint32_t size = r.ctx[ctx::kDbDepthSize];
            std::fprintf(out_, "  depth 0x%" PRIx64 " %ux%u format %u control 0x%x\n",
                         uint64_t(r.ctx[ctx::kDbZReadBase]) << 8, ((size & 0x7FF) + 1) * 8,
                         (((size >> 11) & 0x7FF) + 1) * 8, zInfo & 3, r.ctx[ctx::kDbDepthControl]);
        }
        const float* vp = reinterpret_cast<const float*>(&r.ctx[ctx::kPaClVportXscale]);
        std::fprintf(out_, "  viewport scale %.1f,%.1f offset %.1f,%.1f  blend0 0x%x\n", vp[0], vp[2], vp[1], vp[3],
                     r.ctx[ctx::kCbBlend0Control]);
    }

    void onFrameEnd() override {
        if (frame_ == target_ && out_) {
            std::fclose(out_);
            out_ = nullptr;
            std::fprintf(stderr, "[gpu] frame %u: %u draws/dispatches, %zu shaders\n", frame_, index_, dumped_.size());
        }
        ++frame_;
    }

private:
    void shader(const char* stage, uint64_t addr, const uint32_t* userData, int n) {
        std::fprintf(out_, "  %s 0x%" PRIx64, stage, addr);
        if (!addr) {
            std::fprintf(out_, " (none)\n");
            return;
        }
        std::fprintf(out_, "  user");
        for (int i = 0; i < n; ++i) std::fprintf(out_, " %x", userData[i]);
        std::fprintf(out_, "\n");
        if (!dumped_.insert(addr).second) return;
        const auto* code = reinterpret_cast<const uint8_t*>(addr);
        const size_t size = shaderSize(code);
        if (!size) return;
        char name[96];
        std::snprintf(name, sizeof(name), "%s/shaders/%s_%" PRIx64 ".bin", dir_.c_str(), stage, addr);
        if (FILE* f = std::fopen(name, "wb")) {
            std::fwrite(code, 1, size, f);
            std::fclose(f);
        }
    }

    unsigned target_, frame_ = 0, index_ = 0;
    std::string dir_;
    FILE* out_ = nullptr;
    std::set<uint64_t> dumped_;
};

} // namespace

void enableFrameDump(unsigned frame, const std::string& dir) { addDrawListener(new FrameDump(frame, dir)); }

} // namespace host
