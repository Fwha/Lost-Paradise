#include "gcn/decoder.h"

#include <cstdio>
#include <cstring>

namespace gcn {

namespace {

// --- op name tables (Sea Islands ISA) ---------------------------------------------------

const char* const kSop2[] = {
    "s_add_u32", "s_sub_u32", "s_add_i32", "s_sub_i32", "s_addc_u32", "s_subb_u32", "s_min_i32", "s_min_u32",
    "s_max_i32", "s_max_u32", "s_cselect_b32", "s_cselect_b64", nullptr, nullptr, "s_and_b32", "s_and_b64",
    "s_or_b32", "s_or_b64", "s_xor_b32", "s_xor_b64", "s_andn2_b32", "s_andn2_b64", "s_orn2_b32", "s_orn2_b64",
    "s_nand_b32", "s_nand_b64", "s_nor_b32", "s_nor_b64", "s_xnor_b32", "s_xnor_b64", "s_lshl_b32", "s_lshl_b64",
    "s_lshr_b32", "s_lshr_b64", "s_ashr_i32", "s_ashr_i64", "s_bfm_b32", "s_bfm_b64", "s_mul_i32", "s_bfe_u32",
    "s_bfe_i32", "s_bfe_u64", "s_bfe_i64", "s_cbranch_g_fork", "s_absdiff_i32",
};
const char* const kSopk[] = {
    "s_movk_i32", nullptr, "s_cmovk_i32", "s_cmpk_eq_i32", "s_cmpk_lg_i32", "s_cmpk_gt_i32", "s_cmpk_ge_i32",
    "s_cmpk_lt_i32", "s_cmpk_le_i32", "s_cmpk_eq_u32", "s_cmpk_lg_u32", "s_cmpk_gt_u32", "s_cmpk_ge_u32",
    "s_cmpk_lt_u32", "s_cmpk_le_u32", "s_addk_i32", "s_mulk_i32", "s_cbranch_i_fork", "s_getreg_b32",
    "s_setreg_b32", nullptr, "s_setreg_imm32_b32",
};
const char* const kSop1[] = {
    nullptr, nullptr, nullptr, "s_mov_b32", "s_mov_b64", "s_cmov_b32", "s_cmov_b64", "s_not_b32", "s_not_b64",
    "s_wqm_b32", "s_wqm_b64", "s_brev_b32", "s_brev_b64", "s_bcnt0_i32_b32", "s_bcnt0_i32_b64",
    "s_bcnt1_i32_b32", "s_bcnt1_i32_b64", "s_ff0_i32_b32", "s_ff0_i32_b64", "s_ff1_i32_b32", "s_ff1_i32_b64",
    "s_flbit_i32_b32", "s_flbit_i32_b64", "s_flbit_i32", "s_flbit_i32_i64", "s_sext_i32_i8", "s_sext_i32_i16",
    "s_bitset0_b32", "s_bitset0_b64", "s_bitset1_b32", "s_bitset1_b64", "s_getpc_b64", "s_setpc_b64",
    "s_swappc_b64", "s_rfe_b64", nullptr, "s_and_saveexec_b64", "s_or_saveexec_b64", "s_xor_saveexec_b64",
    "s_andn2_saveexec_b64", "s_orn2_saveexec_b64", "s_nand_saveexec_b64", "s_nor_saveexec_b64",
    "s_xnor_saveexec_b64", "s_quadmask_b32", "s_quadmask_b64", "s_movrels_b32", "s_movrels_b64",
    "s_movreld_b32", "s_movreld_b64", "s_cbranch_join", nullptr, "s_abs_i32", "s_mov_fed_b32",
};
const char* const kSopc[] = {
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
    "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    "s_bitcmp0_b32", "s_bitcmp1_b32", "s_bitcmp0_b64", "s_bitcmp1_b64", "s_setvskip",
};
const char* const kSopp[] = {
    "s_nop", "s_endpgm", "s_branch", nullptr, "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz",
    "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz", "s_barrier", "s_setkill", "s_waitcnt",
    "s_sethalt", "s_sleep", "s_setprio", "s_sendmsg", "s_sendmsghalt", "s_trap", "s_icache_inv",
    "s_incperflevel", "s_decperflevel", "s_ttracedata", "s_cbranch_cdbgsys", "s_cbranch_cdbguser",
    "s_cbranch_cdbgsys_or_user", "s_cbranch_cdbgsys_and_user",
};
const char* const kSmrd[] = {
    "s_load_dword", "s_load_dwordx2", "s_load_dwordx4", "s_load_dwordx8", "s_load_dwordx16", nullptr, nullptr,
    nullptr, "s_buffer_load_dword", "s_buffer_load_dwordx2", "s_buffer_load_dwordx4", "s_buffer_load_dwordx8",
    "s_buffer_load_dwordx16", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "s_dcache_inv_vol", "s_memtime",
    "s_dcache_inv",
};
const char* const kVop2[] = {
    "v_cndmask_b32", "v_readlane_b32", "v_writelane_b32", "v_add_f32", "v_sub_f32", "v_subrev_f32",
    "v_mac_legacy_f32", "v_mul_legacy_f32", "v_mul_f32", "v_mul_i32_i24", "v_mul_hi_i32_i24", "v_mul_u32_u24",
    "v_mul_hi_u32_u24", "v_min_legacy_f32", "v_max_legacy_f32", "v_min_f32", "v_max_f32", "v_min_i32",
    "v_max_i32", "v_min_u32", "v_max_u32", "v_lshr_b32", "v_lshrrev_b32", "v_ashr_i32", "v_ashrrev_i32",
    "v_lshl_b32", "v_lshlrev_b32", "v_and_b32", "v_or_b32", "v_xor_b32", "v_bfm_b32", "v_mac_f32",
    "v_madmk_f32", "v_madak_f32", "v_bcnt_u32_b32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_add_i32",
    "v_sub_i32", "v_subrev_i32", "v_addc_u32", "v_subb_u32", "v_subbrev_u32", "v_ldexp_f32",
    "v_cvt_pkaccum_u8_f32", "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32", "v_cvt_pkrtz_f16_f32",
    "v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32",
};
const char* const kVop1[] = {
    "v_nop", "v_mov_b32", "v_readfirstlane_b32", "v_cvt_i32_f64", "v_cvt_f64_i32", "v_cvt_f32_i32",
    "v_cvt_f32_u32", "v_cvt_u32_f32", "v_cvt_i32_f32", "v_mov_fed_b32", "v_cvt_f16_f32", "v_cvt_f32_f16",
    "v_cvt_rpi_i32_f32", "v_cvt_flr_i32_f32", "v_cvt_off_f32_i4", "v_cvt_f32_f64", "v_cvt_f64_f32",
    "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3", "v_cvt_u32_f64",
    "v_cvt_f64_u32", "v_trunc_f64", "v_ceil_f64", "v_rndne_f64", "v_floor_f64", nullptr, nullptr, nullptr,
    nullptr, nullptr, "v_fract_f32", "v_trunc_f32", "v_ceil_f32", "v_rndne_f32", "v_floor_f32", "v_exp_f32",
    "v_log_clamp_f32", "v_log_f32", "v_rcp_clamp_f32", "v_rcp_legacy_f32", "v_rcp_f32", "v_rcp_iflag_f32",
    "v_rsq_clamp_f32", "v_rsq_legacy_f32", "v_rsq_f32", "v_rcp_f64", "v_rcp_clamp_f64", "v_rsq_f64",
    "v_rsq_clamp_f64", "v_sqrt_f32", "v_sqrt_f64", "v_sin_f32", "v_cos_f32", "v_not_b32", "v_bfrev_b32",
    "v_ffbh_u32", "v_ffbl_b32", "v_ffbh_i32", "v_frexp_exp_i32_f64", "v_frexp_mant_f64", "v_fract_f64",
    "v_frexp_exp_i32_f32", "v_frexp_mant_f32", "v_clrexcp", "v_movreld_b32", "v_movrels_b32",
    "v_movrelsd_b32", "v_log_legacy_f32", "v_exp_legacy_f32",
};
const char* const kVop3Only[] = { // ops 320..375
    "v_mad_legacy_f32", "v_mad_f32", "v_mad_i32_i24", "v_mad_u32_u24", "v_cubeid_f32", "v_cubesc_f32",
    "v_cubetc_f32", "v_cubema_f32", "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_fma_f32", "v_fma_f64",
    "v_lerp_u8", "v_alignbit_b32", "v_alignbyte_b32", "v_mullit_f32", "v_min3_f32", "v_min3_i32", "v_min3_u32",
    "v_max3_f32", "v_max3_i32", "v_max3_u32", "v_med3_f32", "v_med3_i32", "v_med3_u32", "v_sad_u8",
    "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_cvt_pk_u8_f32", "v_div_fixup_f32", "v_div_fixup_f64",
    "v_lshl_b64", "v_lshr_b64", "v_ashr_i64", "v_add_f64", "v_mul_f64", "v_min_f64", "v_max_f64",
    "v_ldexp_f64", "v_mul_lo_u32", "v_mul_hi_u32", "v_mul_lo_i32", "v_mul_hi_i32", "v_div_scale_f32",
    "v_div_scale_f64", "v_div_fmas_f32", "v_div_fmas_f64", "v_msad_u8", "v_qsad_pk_u16_u8",
    "v_mqsad_pk_u16_u8", "v_trig_preop_f64", "v_mqsad_u32_u8", "v_mad_u64_u32", "v_mad_i64_i32",
};
const char* const kVintrp[] = {"v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32"};
const char* const kMtbuf[] = {
    "tbuffer_load_format_x", "tbuffer_load_format_xy", "tbuffer_load_format_xyz", "tbuffer_load_format_xyzw",
    "tbuffer_store_format_x", "tbuffer_store_format_xy", "tbuffer_store_format_xyz", "tbuffer_store_format_xyzw",
};

std::string vopcName(uint32_t op) {
    static const char* const f[] = {"f", "lt", "eq", "le", "gt", "lg", "ge", "o",
                                    "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
    static const char* const i[] = {"f", "lt", "eq", "le", "gt", "ne", "ge", "t"};
    const uint32_t group = op >> 4, n = op & 15;
    const bool x = group & 1;
    switch (group) {
    case 0: case 1: return std::string(x ? "v_cmpx_" : "v_cmp_") + f[n] + "_f32";
    case 2: case 3: return std::string(x ? "v_cmpx_" : "v_cmp_") + f[n] + "_f64";
    case 4: case 5: return std::string(x ? "v_cmpsx_" : "v_cmps_") + f[n] + "_f32";
    case 6: case 7: return std::string(x ? "v_cmpsx_" : "v_cmps_") + f[n] + "_f64";
    default: break;
    }
    static const char* const types[] = {"i32", "i64", "u32", "u64"};
    const char* t = types[(op - 128) >> 5];
    const bool cmpx = (op >> 4) & 1;
    if (n < 8) return std::string(cmpx ? "v_cmpx_" : "v_cmp_") + i[n] + "_" + t;
    if (n == 8 && (op - 128) >> 5 <= 1) return std::string(cmpx ? "v_cmpx_class_" : "v_cmp_class_") + ((op - 128) >> 5 ? "f64" : "f32");
    return {};
}

std::string mimgName(uint32_t op) {
    static const char* const sample[] = {"", "_cl", "_d", "_d_cl", "_l", "_b", "_b_cl", "_lz",
                                         "_c", "_c_cl", "_c_d", "_c_d_cl", "_c_l", "_c_b", "_c_b_cl", "_c_lz"};
    static const char* const gather[] = {"", "_cl", "_l", "_b", "_b_cl", "_lz", "_c", "_c_cl", "_c_l", "_c_b",
                                         "_c_b_cl", "_c_lz"};
    static const char* const basic[] = {"image_load", "image_load_mip", "image_load_pck", "image_load_pck_sgn",
                                        "image_load_mip_pck", "image_load_mip_pck_sgn", nullptr, nullptr,
                                        "image_store", "image_store_mip", "image_store_pck",
                                        "image_store_mip_pck", nullptr, nullptr, "image_get_resinfo"};
    if (op < 15) return basic[op] ? basic[op] : "";
    if (op >= 32 && op < 64) return std::string("image_sample") + sample[op & 15] + (op >= 48 ? "_o" : "");
    if (op >= 64 && op < 96 && (op & 15) < 12)
        return std::string("image_gather4") + gather[op & 15] + (op >= 80 ? "_o" : "");
    if (op == 96) return "image_get_lod";
    if (op >= 104 && op < 112) return std::string("image_sample_cd") + sample[(op & 7) == 0 ? 0 : 0] + "";
    return {};
}

std::string mubufName(uint32_t op) {
    static const char* const t[] = {
        "buffer_load_format_x", "buffer_load_format_xy", "buffer_load_format_xyz", "buffer_load_format_xyzw",
        "buffer_store_format_x", "buffer_store_format_xy", "buffer_store_format_xyz", "buffer_store_format_xyzw",
        "buffer_load_ubyte", "buffer_load_sbyte", "buffer_load_ushort", "buffer_load_sshort", "buffer_load_dword",
        "buffer_load_dwordx2", "buffer_load_dwordx4", "buffer_load_dwordx3", nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, "buffer_store_byte", nullptr, "buffer_store_short", nullptr,
        "buffer_store_dword", "buffer_store_dwordx2", "buffer_store_dwordx4", "buffer_store_dwordx3"};
    if (op < 32) return t[op] ? t[op] : "";
    static const char* const atomics[] = {"swap", "cmpswap", "add", "sub", nullptr, "smin", "umin", "smax",
                                          "umax", "and", "or", "xor", "inc", "dec"};
    if (op >= 48 && op < 62 && atomics[op - 48]) return std::string("buffer_atomic_") + atomics[op - 48];
    if (op == 112) return "buffer_wbinvl1_vol";
    if (op == 113) return "buffer_wbinvl1";
    return {};
}

std::string dsName(uint32_t op) {
    switch (op) {
    case 0: return "ds_add_u32";
    case 1: return "ds_sub_u32";
    case 5: return "ds_min_i32";
    case 6: return "ds_max_i32";
    case 7: return "ds_min_u32";
    case 8: return "ds_max_u32";
    case 9: return "ds_and_b32";
    case 10: return "ds_or_b32";
    case 11: return "ds_xor_b32";
    case 13: return "ds_write_b32";
    case 14: return "ds_write2_b32";
    case 15: return "ds_write2st64_b32";
    case 30: return "ds_write_b8";
    case 31: return "ds_write_b16";
    case 32: return "ds_add_rtn_u32";
    case 45: return "ds_wrxchg_rtn_b32";
    case 53: return "ds_swizzle_b32";
    case 54: return "ds_read_b32";
    case 55: return "ds_read2_b32";
    case 56: return "ds_read2st64_b32";
    case 57: return "ds_read_i8";
    case 58: return "ds_read_u8";
    case 59: return "ds_read_i16";
    case 60: return "ds_read_u16";
    case 77: return "ds_write_b64";
    case 78: return "ds_write2_b64";
    case 118: return "ds_read_b64";
    case 119: return "ds_read2_b64";
    case 222: return "ds_write_b96";
    case 223: return "ds_write_b128";
    case 254: return "ds_read_b96";
    case 255: return "ds_read_b128";
    default: return {};
    }
}

template <size_t N> const char* lookup(const char* const (&t)[N], uint32_t op) { return op < N ? t[op] : nullptr; }

std::string scalar(uint32_t v) {
    char b[32];
    if (v <= 103) std::snprintf(b, sizeof b, "s%u", v);
    else if (v == 106) return "vcc_lo";
    else if (v == 107) return "vcc_hi";
    else if (v == 124) return "m0";
    else if (v == 126) return "exec_lo";
    else if (v == 127) return "exec_hi";
    else if (v >= 128 && v <= 192) std::snprintf(b, sizeof b, "%u", v - 128);
    else if (v >= 193 && v <= 208) std::snprintf(b, sizeof b, "-%u", v - 192);
    else if (v >= 240 && v <= 247) {
        static const char* const c[] = {"0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0"};
        return c[v - 240];
    } else if (v == 251) return "vccz";
    else if (v == 252) return "execz";
    else if (v == 253) return "scc";
    else if (v == 254) return "lds_direct";
    else if (v >= 256) std::snprintf(b, sizeof b, "v%u", v - 256);
    else std::snprintf(b, sizeof b, "src%u", v);
    return b;
}

std::string operand(uint32_t v, const Instruction& in) {
    if (v == 255) {
        char b[16];
        std::snprintf(b, sizeof b, "0x%x", in.literal);
        return b;
    }
    return scalar(v);
}

std::string vreg(uint32_t v) { return "v" + std::to_string(v); }
std::string sreg(uint32_t v) { return scalar(v); }

} // namespace

const char* formatName(Format f) {
    static const char* const n[] = {"sop2", "sopk", "sop1", "sopc", "sopp", "smrd", "vop2", "vop1", "vopc",
                                    "vop3", "vintrp", "ds", "mubuf", "mtbuf", "mimg", "exp", "flat", "unknown"};
    return n[int(f)];
}

Instruction decode(const uint32_t* code, size_t avail) {
    Instruction in;
    const uint32_t w = code[0];
    in.words[0] = w;
    auto two = [&] {
        in.size = 8;
        if (avail > 1) in.words[1] = code[1];
    };
    auto literal = [&] {
        in.hasLiteral = true;
        const size_t at = in.size / 4;
        if (avail > at) in.literal = in.words[at] = code[at];
        in.size += 4;
    };

    if ((w >> 31) == 0) {
        if ((w >> 25) == 0x3F) { // VOP1
            in.format = Format::VOP1;
            in.op = (w >> 9) & 0xFF;
        } else if ((w >> 25) == 0x3E) { // VOPC
            in.format = Format::VOPC;
            in.op = (w >> 17) & 0xFF;
        } else { // VOP2
            in.format = Format::VOP2;
            in.op = (w >> 25) & 0x3F;
            if (in.op == 32 || in.op == 33) { // v_madmk/madak carry a constant
                literal();
                return in;
            }
        }
        if ((w & 0x1FF) == 255) literal();
        return in;
    }
    if ((w >> 30) == 2) {
        const uint32_t top9 = w >> 23;
        if (top9 == 0x17D) {
            in.format = Format::SOP1;
            in.op = (w >> 8) & 0xFF;
            if ((w & 0xFF) == 255) literal();
        } else if (top9 == 0x17E) {
            in.format = Format::SOPC;
            in.op = (w >> 16) & 0x7F;
            if ((w & 0xFF) == 255 || ((w >> 8) & 0xFF) == 255) literal();
        } else if (top9 == 0x17F) {
            in.format = Format::SOPP;
            in.op = (w >> 16) & 0x7F;
        } else if ((w >> 28) == 0xB) {
            in.format = Format::SOPK;
            in.op = (w >> 23) & 0x1F;
            if (in.op == 21) literal();
        } else {
            in.format = Format::SOP2;
            in.op = (w >> 23) & 0x7F;
            if ((w & 0xFF) == 255 || ((w >> 8) & 0xFF) == 255) literal();
        }
        return in;
    }
    if ((w >> 27) == 0x18) {
        in.format = Format::SMRD;
        in.op = (w >> 22) & 0x1F;
        if (!((w >> 8) & 1) && (w & 0xFF) == 255) literal(); // 32-bit offset literal (CI)
        return in;
    }
    switch (w >> 26) {
    case 0x34: in.format = Format::VOP3; in.op = (w >> 17) & 0x1FF; two(); break;
    case 0x32: in.format = Format::VINTRP; in.op = (w >> 16) & 3; break;
    case 0x36: in.format = Format::DS; in.op = (w >> 18) & 0xFF; two(); break;
    case 0x37: in.format = Format::FLAT; in.op = (w >> 18) & 0x7F; two(); break;
    case 0x38: in.format = Format::MUBUF; in.op = (w >> 18) & 0x7F; two(); break;
    case 0x3A: in.format = Format::MTBUF; in.op = (w >> 16) & 7; two(); break;
    case 0x3C: in.format = Format::MIMG; in.op = (w >> 18) & 0x7F; two(); break;
    case 0x3E: in.format = Format::EXP; two(); break;
    default: in.format = Format::Unknown; break;
    }
    return in;
}

std::string mnemonic(const Instruction& in) {
    const char* n = nullptr;
    std::string s;
    switch (in.format) {
    case Format::SOP2: n = lookup(kSop2, in.op); break;
    case Format::SOPK: n = lookup(kSopk, in.op); break;
    case Format::SOP1: n = lookup(kSop1, in.op); break;
    case Format::SOPC: n = lookup(kSopc, in.op); break;
    case Format::SOPP: n = lookup(kSopp, in.op); break;
    case Format::SMRD: n = lookup(kSmrd, in.op); break;
    case Format::VOP2: n = lookup(kVop2, in.op); break;
    case Format::VOP1: n = lookup(kVop1, in.op); break;
    case Format::VOPC: s = vopcName(in.op); break;
    case Format::VOP3:
        if (in.op < 256) s = vopcName(in.op);
        else if (in.op < 320) n = lookup(kVop2, in.op - 256);
        else if (in.op < 384) n = lookup(kVop3Only, in.op - 320);
        else n = lookup(kVop1, in.op - 384);
        break;
    case Format::VINTRP: n = lookup(kVintrp, in.op); break;
    case Format::DS: s = dsName(in.op); break;
    case Format::MUBUF: s = mubufName(in.op); break;
    case Format::MTBUF: n = lookup(kMtbuf, in.op); break;
    case Format::MIMG: s = mimgName(in.op); break;
    case Format::EXP: return "exp";
    default: break;
    }
    if (n) return n;
    if (!s.empty()) return s;
    return std::string(formatName(in.format)) + "_op" + std::to_string(in.op);
}

std::string disassemble(const Instruction& in, uint32_t pc) {
    const uint32_t w = in.words[0], w1 = in.words[1];
    std::string m = mnemonic(in);
    char b[160];
    switch (in.format) {
    case Format::SOP2:
        std::snprintf(b, sizeof b, "%s %s, %s, %s", m.c_str(), sreg((w >> 16) & 0x7F).c_str(),
                      operand(w & 0xFF, in).c_str(), operand((w >> 8) & 0xFF, in).c_str());
        break;
    case Format::SOPK:
        std::snprintf(b, sizeof b, "%s %s, 0x%x", m.c_str(), sreg((w >> 16) & 0x7F).c_str(), w & 0xFFFF);
        break;
    case Format::SOP1:
        std::snprintf(b, sizeof b, "%s %s, %s", m.c_str(), sreg((w >> 16) & 0x7F).c_str(),
                      operand(w & 0xFF, in).c_str());
        break;
    case Format::SOPC:
        std::snprintf(b, sizeof b, "%s %s, %s", m.c_str(), operand(w & 0xFF, in).c_str(),
                      operand((w >> 8) & 0xFF, in).c_str());
        break;
    case Format::SOPP: {
        const int16_t imm = int16_t(w & 0xFFFF);
        if ((in.op >= 2 && in.op <= 9) || in.op >= 23) // branches: target = pc + 4 + imm*4
            std::snprintf(b, sizeof b, "%s 0x%x", m.c_str(), pc + 4 + int32_t(imm) * 4);
        else if (in.op == 12)
            std::snprintf(b, sizeof b, "%s vmcnt(%u) expcnt(%u) lgkmcnt(%u)", m.c_str(), imm & 0xF, (imm >> 4) & 7,
                          (imm >> 8) & 0x1F);
        else if (in.op == 1 || in.op == 10) std::snprintf(b, sizeof b, "%s", m.c_str());
        else std::snprintf(b, sizeof b, "%s 0x%x", m.c_str(), uint32_t(uint16_t(imm)));
        break;
    }
    case Format::SMRD: {
        const bool imm = (w >> 8) & 1;
        char off[24];
        if (imm) std::snprintf(off, sizeof off, "0x%x", (w & 0xFF) * 4);
        else if ((w & 0xFF) == 255) std::snprintf(off, sizeof off, "0x%x", in.literal);
        else std::snprintf(off, sizeof off, "%s", sreg(w & 0xFF).c_str());
        std::snprintf(b, sizeof b, "%s %s, s[%u:%u], %s", m.c_str(), sreg((w >> 15) & 0x7F).c_str(),
                      ((w >> 9) & 0x3F) * 2, ((w >> 9) & 0x3F) * 2 + (in.op >= 8 ? 3 : 1), off);
        break;
    }
    case Format::VOP2:
        if (in.op == 32 || in.op == 33)
            std::snprintf(b, sizeof b, "%s %s, %s, %s, 0x%x", m.c_str(), vreg((w >> 17) & 0xFF).c_str(),
                          scalar(w & 0x1FF).c_str(), vreg((w >> 9) & 0xFF).c_str(), in.literal);
        else
            std::snprintf(b, sizeof b, "%s %s, %s, %s", m.c_str(), vreg((w >> 17) & 0xFF).c_str(),
                          operand(w & 0x1FF, in).c_str(), vreg((w >> 9) & 0xFF).c_str());
        break;
    case Format::VOP1:
        std::snprintf(b, sizeof b, "%s %s, %s", m.c_str(), vreg((w >> 17) & 0xFF).c_str(),
                      operand(w & 0x1FF, in).c_str());
        break;
    case Format::VOPC:
        std::snprintf(b, sizeof b, "%s vcc, %s, %s", m.c_str(), operand(w & 0x1FF, in).c_str(),
                      vreg((w >> 9) & 0xFF).c_str());
        break;
    case Format::VOP3: {
        const uint32_t neg = w1 >> 29, abs = (w >> 8) & 7, omod = (w1 >> 27) & 3;
        const bool b3 = (in.op >= 256 + 37 && in.op <= 256 + 42) || in.op == 365 || in.op == 366 || in.op == 374 ||
                        in.op == 375;
        std::string dst = in.op < 256 ? sreg(w & 0xFF) : vreg(w & 0xFF);
        if (b3) dst += ", " + sreg((w >> 8) & 0x7F);
        auto src = [&](int i) {
            std::string s = scalar((w1 >> (i * 9)) & 0x1FF);
            if (!b3 && (abs >> i) & 1) s = "|" + s + "|";
            if ((neg >> i) & 1) s = "-" + s;
            return s;
        };
        const bool three = in.op >= 320 && in.op < 384;
        const bool one = in.op >= 384;
        std::string ops = dst + ", " + src(0);
        if (!one) ops += ", " + src(1);
        if (three || (in.op >= 256 + 40 && in.op <= 256 + 42) || in.op == 256) ops += ", " + src(2);
        if (!b3 && ((w >> 11) & 1)) ops += " clamp";
        if (omod) ops += omod == 1 ? " mul:2" : omod == 2 ? " mul:4" : " div:2";
        std::snprintf(b, sizeof b, "%s %s", m.c_str(), ops.c_str());
        break;
    }
    case Format::VINTRP:
        std::snprintf(b, sizeof b, "%s %s, %s, attr%u.%c", m.c_str(), vreg((w >> 18) & 0xFF).c_str(),
                      in.op == 2 ? (w & 3) == 0 ? "p10" : (w & 3) == 1 ? "p20" : "p0" : vreg(w & 0xFF).c_str(),
                      (w >> 10) & 0x3F, "xyzw"[(w >> 8) & 3]);
        break;
    case Format::DS:
        std::snprintf(b, sizeof b, "%s %s, %s, %s, %s offset0:%u offset1:%u%s", m.c_str(),
                      vreg(w1 >> 24).c_str(), vreg(w1 & 0xFF).c_str(), vreg((w1 >> 8) & 0xFF).c_str(),
                      vreg((w1 >> 16) & 0xFF).c_str(), w & 0xFF, (w >> 8) & 0xFF, (w >> 17) & 1 ? " gds" : "");
        break;
    case Format::MUBUF:
    case Format::MTBUF: {
        const bool mtbuf = in.format == Format::MTBUF;
        std::snprintf(b, sizeof b, "%s %s, %s, s[%u:%u], %s offset:%u%s%s%s%s", m.c_str(),
                      vreg((w1 >> 8) & 0xFF).c_str(), vreg(w1 & 0xFF).c_str(), ((w1 >> 16) & 0x1F) * 4,
                      ((w1 >> 16) & 0x1F) * 4 + 3, scalar(w1 >> 24).c_str(), w & 0xFFF, (w >> 12) & 1 ? " offen" : "",
                      (w >> 13) & 1 ? " idxen" : "", (w >> 14) & 1 ? " glc" : "", mtbuf ? "" : (w >> 16) & 1 ? " lds" : "");
        if (mtbuf) {
            const size_t len = std::strlen(b);
            std::snprintf(b + len, sizeof b - len, " dfmt:%u nfmt:%u", (w >> 19) & 0xF, (w >> 23) & 7);
        }
        break;
    }
    case Format::MIMG:
        std::snprintf(b, sizeof b, "%s %s, %s, s[%u:%u], s[%u:%u] dmask:0x%x%s%s%s", m.c_str(),
                      vreg((w1 >> 8) & 0xFF).c_str(), vreg(w1 & 0xFF).c_str(), ((w1 >> 16) & 0x1F) * 4,
                      ((w1 >> 16) & 0x1F) * 4 + 7, ((w1 >> 21) & 0x1F) * 4, ((w1 >> 21) & 0x1F) * 4 + 3,
                      (w >> 8) & 0xF, (w >> 12) & 1 ? " unorm" : "", (w >> 13) & 1 ? " glc" : "",
                      (w >> 25) & 1 ? " lwe" : "");
        break;
    case Format::EXP: {
        const uint32_t target = (w >> 4) & 0x3F;
        char t[16];
        if (target < 8) std::snprintf(t, sizeof t, "mrt%u", target);
        else if (target == 8) std::snprintf(t, sizeof t, "mrtz");
        else if (target == 9) std::snprintf(t, sizeof t, "null");
        else if (target >= 12 && target < 16) std::snprintf(t, sizeof t, "pos%u", target - 12);
        else if (target >= 32) std::snprintf(t, sizeof t, "param%u", target - 32);
        else std::snprintf(t, sizeof t, "target%u", target);
        std::snprintf(b, sizeof b, "exp %s en:0x%x %s %s %s %s%s%s%s", t, w & 0xF, vreg(w1 & 0xFF).c_str(),
                      vreg((w1 >> 8) & 0xFF).c_str(), vreg((w1 >> 16) & 0xFF).c_str(), vreg(w1 >> 24).c_str(),
                      (w >> 10) & 1 ? " compr" : "", (w >> 11) & 1 ? " done" : "", (w >> 12) & 1 ? " vm" : "");
        break;
    }
    default: std::snprintf(b, sizeof b, "%s 0x%08x", m.c_str(), w); break;
    }
    return b;
}

} // namespace gcn
