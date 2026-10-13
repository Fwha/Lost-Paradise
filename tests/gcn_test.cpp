// GCN decoder tests against a real pixel shader's encoding (a textured copy: interpolate
// UVs, sample, pack to f16, export).

#include "gcn/decoder.h"

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
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
