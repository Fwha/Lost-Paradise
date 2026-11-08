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

    // A V# built from the program counter (a constant table inside the shader, as the game's AO
    // upsample does): s_getpc_b64, s_add_u32 and s_addc_u32 must leave the address known.
    const std::vector<uint32_t> pcRel = {
        0xbe9c1f00,             // s_getpc_b64 s[28:29]
        0x801c1cff, 0x00000328, // s_add_u32 s28, 0x328, s28
        0x821d1d80,             // s_addc_u32 s29, 0, s29
        0xbe9e03c0,             // s_mov_b32 s30, 64
        0xbe9f03ff, 0x2000c004, // s_mov_b32 s31, 0x2000c004
        0xebf31030, 0x8007000d, // tbuffer_load_format_xyzw v0, v13, s[28:31], 0 offset:48 offen
        0xbf810000,             // s_endpgm
    };
    gcn::LaunchState pl = l;
    pl.entry = 0x3'0000'1000ull; // above 4 GiB so the add's carry matters
    auto readPcRel = [&](uint64_t addr, void* dst, size_t size) {
        if (addr < pl.entry || addr + size > pl.entry + pcRel.size() * 4) return false;
        std::memcpy(dst, reinterpret_cast<const char*>(pcRel.data()) + (addr - pl.entry), size);
        return true;
    };
    const gcn::Program pp = gcn::analyze(pl, readPcRel);
    if (pp.buffers.size() != 1 || !pp.buffers[0].known || pp.buffers[0].base() != pl.entry + 4 + 0x328) {
        std::fprintf(stderr, "pc-relative buffer: %zu buffers, known %d, base 0x%llx\n", pp.buffers.size(),
                     pp.buffers.empty() ? 0 : int(pp.buffers[0].known),
                     pp.buffers.empty() ? 0ull : (unsigned long long)pp.buffers[0].base());
        ++failures;
    }

    // Reuse across launches: a V# loaded through a user pointer, then a branch on a loaded flag.
    const std::vector<uint32_t> loads = {
        0xc0820100,             // s_load_dwordx4 s[4:7], s[0:1], 0x0
        0xc0040104,             // s_load_dword s8, s[0:1], 0x4
        0xbf068008,             // s_cmp_eq_u32 s8, 0
        0xbf850002,             // s_cbranch_scc1 +2
        0xebf31030, 0x8001000d, // tbuffer_load_format_xyzw v0, v13, s[4:7], 0 offset:48 offen
        0xbf810000,             // s_endpgm
    };
    uint32_t table[2][8] = {{0x10000, 0, 64, 0x2000c004, 1}, {0x20000, 0, 64, 0x2000c004, 1}};
    gcn::LaunchState ll = l;
    ll.rsrc2 = 2 << 1;
    ll.userData[0] = 0x8000;
    auto readLoads = [&](uint64_t addr, void* dst, size_t size) {
        if (addr >= ll.entry && addr + size <= ll.entry + loads.size() * 4) {
            std::memcpy(dst, reinterpret_cast<const char*>(loads.data()) + (addr - ll.entry), size);
            return true;
        }
        if (addr >= 0x8000 && addr + size <= 0x8000 + sizeof(table)) {
            std::memcpy(dst, reinterpret_cast<const char*>(table) + (addr - 0x8000), size);
            return true;
        }
        return false;
    };
    gcn::Program lp = gcn::analyze(ll, readLoads);
    gcn::LaunchState other = ll;
    other.userData[0] = 0x8000 + sizeof(table[0]); // the next draw's table
    if (lp.buffers.size() != 1 || lp.buffers[0].base() != 0x10000 || !gcn::refresh(lp, other, readLoads) ||
        lp.buffers[0].base() != 0x20000 || lp.data[0] != other.userData[0]) {
        std::fprintf(stderr, "refresh did not re-target the program to another table\n");
        ++failures;
    }
    table[0][4] = 0; // the branch now skips the load
    if (gcn::refresh(lp, ll, readLoads)) {
        std::fprintf(stderr, "refresh accepted a different branch\n");
        ++failures;
    }
    table[0][4] = 1;
    table[0][3] ^= 0x7; // another dst_sel: another translation
    if (gcn::refresh(lp, ll, readLoads)) {
        std::fprintf(stderr, "refresh accepted a different buffer format\n");
        ++failures;
    }

    // Cross-lane and wide integer ops seen in gameplay shaders: a depth-slope pixel shader's quad
    // swizzles, an integer division's 64-bit multiply-add, and a 4-bit offset conversion.
    const std::vector<uint32_t> misc = {
        0xd8d48055, 0x03000000, // ds_swizzle_b32 v3, v0 offset0:85 offset1:128 (lane 1 of each quad)
        0xd2ec0a02, 0x02020300, // v_mad_u64_u32 v[2:3], s[10:11], v0, v1, 0
        0x7e081c84,             // v_cvt_off_f32_i4 v4, 4
        0xd2000006, 0x41aa0f07, // v_cndmask_b32 v6, v7, -v7, vcc (a gauge shader's atan2 sign)
        0xbf810000,             // s_endpgm
    };
    gcn::LaunchState ml = l;
    auto readMisc = [&](uint64_t addr, void* dst, size_t size) {
        if (addr < ml.entry || addr + size > ml.entry + misc.size() * 4) return false;
        std::memcpy(dst, reinterpret_cast<const char*>(misc.data()) + (addr - ml.entry), size);
        return true;
    };
    const gcn::Translation mt = gcn::translate(gcn::analyze(ml, readMisc), 0);
    if (!mt.warnings.empty() || mt.glsl.find("subgroupQuadBroadcast(v0, 1u)") == std::string::npos ||
        mt.glsl.find("umulExtended(v0, v1") == std::string::npos ||
        mt.glsl.find("(vcc_lo & 1u) != 0u ? (v7 ^ 0x80000000u) : v7") == std::string::npos ||
        gcn::compileGlsl(mt.glsl, gcn::GlslStage::Fragment, &error).empty()) {
        std::fprintf(stderr, "lane/wide ops: %zu warnings (%s)\n%s\n%s\n", mt.warnings.size(),
                     mt.warnings.empty() ? "" : mt.warnings[0].c_str(), error.c_str(), mt.glsl.c_str());
        ++failures;
    }

    // A copy shader storing into a BGRA render target through a T# whose DST_SEL swaps red and
    // blue: the store applies DST_SEL inversely, so memory stays BGRA.
    const std::vector<uint32_t> store = {
        0xf0200f00, 0x00020004, // image_store v0, v4, s[8:15], s[0:3] dmask:0xf
        0xbf810000,             // s_endpgm
    };
    gcn::LaunchState sl;
    sl.stage = gcn::Stage::Compute;
    sl.entry = 0x2000;
    sl.rsrc2 = 16 << 1;
    sl.userData[8 + 1] = 10u << 20;           // 8_8_8_8 unorm
    sl.userData[8 + 3] = 07456u | (9u << 28); // DST_SEL zyxw (B, G, R, A), 2D
    auto readStore = [&](uint64_t addr, void* dst, size_t size) {
        if (addr < sl.entry || addr + size > sl.entry + store.size() * 4) return false;
        std::memcpy(dst, reinterpret_cast<const char*>(store.data()) + (addr - sl.entry), size);
        return true;
    };
    const gcn::Translation st = gcn::translate(gcn::analyze(sl, readStore), 0);
    if (st.glsl.find("vec4(s_[2], s_[1], s_[0], s_[3])") == std::string::npos ||
        gcn::compileGlsl(st.glsl, gcn::GlslStage::Compute, &error).empty()) {
        std::fprintf(stderr, "swizzled image store:\n%s\n%s\n", error.c_str(), st.glsl.c_str());
        ++failures;
    }

    // A vertex shader calling a fetch shader (s_swappc_b64 to the address in s[2:3], which returns
    // with s_setpc_b64): moved elsewhere in memory, it must translate to the same GLSL.
    const std::vector<uint32_t> caller = {
        0xbe9e2102, // s_swappc_b64 s[30:31], s[2:3]
        0xbf810000, // s_endpgm
    };
    const std::vector<uint32_t> callee = {
        0xbe80201e, // s_setpc_b64 s[30:31]
    };
    auto translated = [&](uint64_t entry, uint64_t fetch) {
        gcn::LaunchState vl;
        vl.stage = gcn::Stage::Vertex;
        vl.entry = entry;
        vl.rsrc2 = 4 << 1;
        vl.userData[2] = uint32_t(fetch);
        vl.userData[3] = uint32_t(fetch >> 32);
        auto readCall = [&](uint64_t addr, void* dst, size_t size) {
            for (const auto& [at, words] : {std::pair{entry, &caller}, std::pair{fetch, &callee}})
                if (addr >= at && addr + size <= at + words->size() * 4) {
                    std::memcpy(dst, reinterpret_cast<const char*>(words->data()) + (addr - at), size);
                    return true;
                }
            return false;
        };
        const gcn::Program vp = gcn::analyze(vl, readCall);
        if (!vp.controlFlow || vp.calls.size() != 1) {
            std::fprintf(stderr, "call: control flow %d, %zu calls\n", int(vp.controlFlow), vp.calls.size());
            ++failures;
        }
        return gcn::translate(vp, 0).glsl;
    };
    const std::string at1 = translated(0x2'e822'6260ull, 0x3c3a'0000ull);
    const std::string at2 = translated(0x2'e822'6280ull, 0x3c3a'0040ull);
    if (at1 != at2) {
        std::fprintf(stderr, "a moved shader translated differently:\n%s\n---\n%s\n", at1.c_str(), at2.c_str());
        ++failures;
    }
    if (gcn::compileGlsl(at1, gcn::GlslStage::Vertex, &error).empty()) {
        std::fprintf(stderr, "GLSL with a call did not compile:\n%s\n%s\n", error.c_str(), at1.c_str());
        ++failures;
    }

    // A pixel shader with a loop (a backward branch on a vector compare) and an early exit out
    // of it: blocks in address order with a `for` around the loop, no loop over a switch.
    const std::vector<uint32_t> loop = {
        0x7e000280, // v_mov_b32 v0, 0
        0x060000f2, // v_add_f32 v0, 1.0, v0
        0x7c0800f6, // v_cmp_gt_f32 vcc, 4.0, v0
        0xbf860002, // s_cbranch_vccz 0x18 (out, past the loop)
        0xbf87fffc, // s_cbranch_vccnz 0x4 (back to the add)
        0xbf810000, // s_endpgm
        0x7e000280, // v_mov_b32 v0, 0
        0xbf810000, // s_endpgm
    };
    gcn::LaunchState loopL = l;
    auto readLoop = [&](uint64_t addr, void* dst, size_t size) {
        if (addr < loopL.entry || addr + size > loopL.entry + loop.size() * 4) return false;
        std::memcpy(dst, reinterpret_cast<const char*>(loop.data()) + (addr - loopL.entry), size);
        return true;
    };
    const gcn::Translation lt = gcn::translate(gcn::analyze(loopL, readLoop), 0);
    if (lt.glsl.find("switch (pc)") != std::string::npos || lt.glsl.find("continue;") == std::string::npos ||
        gcn::compileGlsl(lt.glsl, gcn::GlslStage::Fragment, &error).empty()) {
        std::fprintf(stderr, "structured loop:\n%s\n%s\n", error.c_str(), lt.glsl.c_str());
        ++failures;
    }

    // A program saved and loaded again (for the startup pipeline builds) translates the same.
    const gcn::Program saved = gcn::analyze(loopL, readLoop);
    gcn::Program loaded;
    const std::string bytes = gcn::serialize(saved);
    if (!gcn::deserialize(bytes, loaded) || gcn::translate(loaded, 3).glsl != gcn::translate(saved, 3).glsl ||
        gcn::deserialize(bytes.substr(0, bytes.size() - 1), loaded)) {
        std::fprintf(stderr, "a saved program loaded differently\n");
        ++failures;
    }

    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
