// GCN decoder tests against a real pixel shader's encoding (a textured copy: interpolate
// UVs, sample, pack to f16, export).

#include "gcn/decoder.h"
#include "gcn/shader.h"

#include <cstring>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(const std::string& got, const char* want) {
    if (got != want) {
        std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", want, got.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    const std::vector<uint32_t> code = {
        0xbeeb03ff, 0x0000000b, 0xbefc030c, 0xbeea047e, 0xbefe0a7e, 0xc8080000, 0xc80c0100, 0xc8090001,
        0xc80d0101, 0xf0800f00, 0x00400002, 0xbf8c0f70, 0x5e000300, 0x5e020702, 0xbefe046a, 0xf8001c0f,
        0x00000100, 0xbf810000,
    };
    const char* want[] = {
        "s_mov_b32 vcc_hi, 0xb",
        "s_mov_b32 m0, s12",
        "s_mov_b64 vcc_lo, exec_lo",
        "s_wqm_b64 exec_lo, exec_lo",
        "v_interp_p1_f32 v2, v0, attr0.x",
        "v_interp_p1_f32 v3, v0, attr0.y",
        "v_interp_p2_f32 v2, v1, attr0.x",
        "v_interp_p2_f32 v3, v1, attr0.y",
        "image_sample v0, v2, s[0:7], s[8:11] dmask:0xf",
        "s_waitcnt vmcnt(0) expcnt(7) lgkmcnt(15)",
        "v_cvt_pkrtz_f16_f32 v0, v0, v1",
        "v_cvt_pkrtz_f16_f32 v1, v2, v3",
        "s_mov_b64 exec_lo, vcc_lo",
        "exp mrt0 en:0xf v0 v1 v0 v0 compr done vm",
        "s_endpgm",
    };
    size_t i = 0, n = 0;
    for (; i < code.size() && n < std::size(want); ++n) {
        const gcn::Instruction in = gcn::decode(&code[i], code.size() - i);
        expect(gcn::disassemble(in, uint32_t(i * 4)), want[n]);
        i += in.size / 4;
    }
    if (i != code.size() || n != std::size(want)) {
        std::fprintf(stderr, "decoded %zu instructions over %zu dwords\n", n, i);
        ++failures;
    }

    // Translate the same shader to GLSL and compile it to SPIR-V. The T# and S# come from
    // user SGPRs s[0:11]; the analysis should find one sampled image.
    gcn::LaunchState l;
    l.stage = gcn::Stage::Pixel;
    l.entry = 0x1000;
    l.rsrc2 = 13 << 1;
    l.psInputEna = 0x2;  // perspective center i/j in v0, v1
    l.colorFormat = 0x4; // MRT0 exported as FP16
    l.colorComps[0] = 4;
    auto read = [&](uint64_t addr, void* dst, size_t size) {
        if (addr < l.entry || addr + size > l.entry + code.size() * 4) return false;
        std::memcpy(dst, reinterpret_cast<const char*>(code.data()) + (addr - l.entry), size);
        return true;
    };
    const gcn::Program p = gcn::analyze(l, read);
    if (p.code.size() != std::size(want) || p.images.size() != 1 || p.controlFlow) {
        std::fprintf(stderr, "analysis: %zu instructions, %zu images\n", p.code.size(), p.images.size());
        ++failures;
    }
    const gcn::Translation t = gcn::translate(p, 0);
    std::string error;
    if (gcn::compileGlsl(t.glsl, gcn::GlslStage::Fragment, &error).empty()) {
        std::fprintf(stderr, "GLSL did not compile:\n%s\n%s\n", error.c_str(), t.glsl.c_str());
        ++failures;
    }
    if (t.colorOutputs != 1 || t.inputs != 1 || !t.warnings.empty()) {
        std::fprintf(stderr, "translation: outputs 0x%x inputs 0x%x, %zu warnings\n", t.colorOutputs, t.inputs,
                     t.warnings.size());
        ++failures;
    }

    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
