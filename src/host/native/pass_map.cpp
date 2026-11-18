#include "host/native/pass_map.h"

#include "host/gpu/registers.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace host::native {

namespace {

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

uint64_t pgm(const GpuRegs& r, uint32_t lo) { return ((uint64_t(r.sh[lo + 1]) << 32) | r.sh[lo]) << 8; }

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%" PRIx64, v);
    return b;
}

} // namespace

PassMap::PassMap() {
    if (const char* e = std::getenv("LPN_PASS_MAP")) {
        interval_ = std::atof(e);
        if (interval_ <= 1) interval_ = 10; // LPN_PASS_MAP=1: every 10 seconds
        last_ = now();
    }
}

void PassMap::closeRun(Pass& p) {
    if (p.lastCount) p.lines += "    " + std::to_string(p.lastCount) + "x " + p.lastPair + "\n";
    p.lastPair.clear();
    p.lastCount = 0;
}

void PassMap::onDraw(const DrawCall& d, const GpuRegs& r, ReadGuest read) {
    using namespace ::gpu;
    Pass p;
    const bool dispatch = d.kind == DrawCall::kDispatch || d.kind == DrawCall::kDispatchIndirect;
    if (!dispatch) {
        const uint32_t mask = r.ctx[ctx::kCbShaderMask] & r.ctx[0x08E];
        const bool colorOn = ((r.ctx[ctx::kCbColorControl] >> 4) & 7) == 1;
        for (uint32_t i = 0; i < 8 && colorOn; ++i) {
            const uint32_t* cb = &r.ctx[ctx::kCbColor0Base + i * ctx::kCbColorStride];
            const uint32_t info = cb[ctx::kCbColorInfo];
            if (!((info >> 2) & 0x1F) || !((mask >> (i * 4)) & 0xF)) continue;
            const uint32_t pitch = ((cb[ctx::kCbColorPitch] & 0x7FF) + 1) * 8;
            const uint32_t slice = ((cb[ctx::kCbColorSlice] & 0x3FFFFF) + 1) * 64;
            p.color[i] = {uint64_t(cb[0]) << 8, pitch, slice / pitch, ((info >> 2) & 0x1F) | ((info >> 8) & 7) << 8};
        }
        const uint32_t z = r.ctx[ctx::kDbZInfo];
        if (z & 3) {
            const uint32_t size = r.ctx[ctx::kDbDepthSize];
            p.depth = {uint64_t(r.ctx[ctx::kDbZReadBase]) << 8, ((size & 0x7FF) + 1) * 8, (((size >> 11) & 0x7FF) + 1) * 8, z & 3};
        }
    }
    bool same = open_;
    for (int i = 0; i < 8 && same; ++i) same = p.color[i] == cur_.color[i];
    same = same && p.depth == cur_.depth;
    if (!same && !dispatch) { // a dispatch stays in the pass it falls in
        if (open_) {
            closeRun(cur_);
            out_ += cur_.lines;
        }
        cur_ = p;
        open_ = true;
        ++passes_;
        out_ += "pass " + std::to_string(passes_) + ":";
        for (int i = 0; i < 8; ++i)
            if (p.color[i].addr)
                out_ += " rt" + std::to_string(i) + " " + hex(p.color[i].addr) + " " + std::to_string(p.color[i].w) + "x" +
                        std::to_string(p.color[i].h) + " fmt " + std::to_string(p.color[i].fmt & 0xFF) + "/" +
                        std::to_string(p.color[i].fmt >> 8);
        if (p.depth.addr)
            out_ += " depth " + hex(p.depth.addr) + " " + std::to_string(p.depth.w) + "x" + std::to_string(p.depth.h) +
                    (p.depth.fmt == 1 ? " d16" : " d32");
        if (!p.depth.addr && !p.color[0].addr) out_ += " (no targets)";
        out_ += "\n";
    } else if (!open_) {
        open_ = true;
        out_ += "pass 0: compute before any draw\n";
    }
    std::string pair;
    if (dispatch) pair = "cs " + shaderNameAt(pgm(r, sh::kComputePgmLo), read);
    else pair = shaderNameAt(pgm(r, sh::kSpiShaderPgmLoVs), read) + " | " + shaderNameAt(pgm(r, sh::kSpiShaderPgmLoPs), read);
    if (pair != cur_.lastPair) {
        closeRun(cur_);
        cur_.lastPair = pair;
    }
    ++cur_.lastCount;
    ++(dispatch ? cur_.dispatches : cur_.draws);
}

void PassMap::onFrameEnd(uint64_t frame, const std::string& captureDir) {
    if (active_) {
        if (open_) {
            closeRun(cur_);
            out_ += cur_.lines;
        }
        std::error_code ec;
        std::filesystem::create_directories(dir_, ec);
        const std::string path = dir_ + "/passmap-" + std::to_string(frame) + ".txt";
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            std::fprintf(f, "frame %" PRIu64 ": %u passes\n%s", frame, passes_, out_.c_str());
            std::fclose(f);
            std::fprintf(stderr, "[native] pass map: %s (%u passes)\n", path.c_str(), passes_);
        }
        active_ = false;
        out_.clear();
        cur_ = Pass{};
        open_ = false;
        passes_ = 0;
    }
    if (!captureDir.empty()) {
        dir_ = captureDir;
        active_ = true;
    } else if (interval_ > 0 && now() - last_ >= interval_) {
        last_ = now();
        dir_ = "lpn-captures";
        active_ = true;
    }
}

} // namespace host::native
