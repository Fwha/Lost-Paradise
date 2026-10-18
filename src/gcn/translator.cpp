// GCN to GLSL. Each GLSL invocation runs one lane: SGPRs and VGPRs become local uints, and
// 64-bit lane masks (exec, vcc, compare results) hold this lane's bit in bit 0. Programs with
// branches run as a loop over a switch on the program counter.
#include "gcn/shader.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace gcn {

namespace {

constexpr uint32_t kVcc = 106, kM0 = 124, kExec = 126;

std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

std::string hex(uint32_t v) { return fmt("0x%08xu", v); }

// Components of a buffer data format, or 0 if unsupported.
unsigned formatComponents(uint32_t dfmt) {
    switch (dfmt) {
    case 1: case 2: case 4: return 1;
    case 3: case 5: case 11: return 2;
    case 6: case 7: case 13: return 3;
    case 8: case 9: case 10: case 12: case 14: return 4;
    default: return 0;
    }
}

class Emitter {
public:
    Emitter(const Program& p, uint32_t base) : p_(p), base_(base) {}

    Translation run() {
        const Stage stage = p_.launch.stage;
        exec_ = p_.writesExec;
        prologue();
        if (p_.controlFlow) {
            out("for (uint guard = 0u; guard < 100000u; ++guard) {\nswitch (pc) {\n");
        }
        uint64_t expect = p_.launch.entry;
        bool first = true;
        for (const auto& [pc, in] : p_.code) {
            if (!first && pc != expect) out("return;\n");
            first = false;
            if (p_.controlFlow && p_.labels.count(pc)) out("case " + label(pc) + ":\n");
            pc_ = pc;
            next_ = pc + in.size;
            probe(pc);
            instruction(in);
            expect = next_;
        }
        out("return;\n");
        if (p_.controlFlow) out("default: return;\n}\n}\n");

        // Assemble the module.
        std::string g = "#version 450\n";
        if (need64_) g += "#extension GL_ARB_gpu_shader_int64 : require\n";
        if (needQuad_) g += "#extension GL_KHR_shader_subgroup_quad : require\n";
        if (stage == Stage::Compute)
            g += fmt("layout(local_size_x = %u, local_size_y = %u, local_size_z = %u) in;\n", p_.launch.localSize[0],
                     p_.launch.localSize[1], p_.launch.localSize[2]);
        g += fmt("layout(std430, set = 0, binding = %u) readonly buffer DrawData { uint D[]; };\n", base_);
        g += decls_;
        g += helpers();
        if (needLane_)
            g += stage == Stage::Compute ? "uint laneId() { return gl_LocalInvocationIndex & 63u; }\n"
                 : stage == Stage::Vertex ? "uint laneId() { return uint(gl_VertexIndex) & 63u; }\n"
                                          : "uint laneId() { return 0u; }\n";
        if (needQuad_)
            g += "uint quadSwizzle(uint v, uint sel) {\n"
                 "    uint q0 = subgroupQuadBroadcast(v, 0u), q1 = subgroupQuadBroadcast(v, 1u);\n"
                 "    uint q2 = subgroupQuadBroadcast(v, 2u), q3 = subgroupQuadBroadcast(v, 3u);\n"
                 "    uint s = (sel >> ((gl_SubgroupInvocationID & 3u) * 2u)) & 3u;\n"
                 "    return s == 0u ? q0 : s == 1u ? q1 : s == 2u ? q2 : q3;\n"
                 "}\n";
        g += "void main() {\n";
        for (const auto& r : regs_) g += "uint " + r + " = 0u;\n";
        g += "bool scc = false;\n";
        if (p_.controlFlow) g += "uint pc = " + label(p_.launch.entry) + ";\n";
        g += init_;
        g += body_;
        g += "}\n";
        t_.glsl = std::move(g);
        return std::move(t_);
    }

private:
    // ---- registers and operands -------------------------------------------------------------

    std::string sreg(uint32_t code) {
        std::string n;
        if (code < 104) n = fmt("s%u", code);
        else if (code == kVcc) n = "vcc_lo";
        else if (code == kVcc + 1) n = "vcc_hi";
        else if (code == kM0) n = "m0";
        else if (code == kExec) n = "exec_lo";
        else if (code == kExec + 1) n = "exec_hi";
        else if (code == 125) n = "s_null"; // writes to unknown destinations land here
        else n = fmt("s_x%u", code);
        regs_.insert(n);
        return n;
    }
    std::string vreg(uint32_t n) {
        std::string r = fmt("v%u", n);
        regs_.insert(r);
        return r;
    }

    // A 32-bit source operand as a uint expression.
    std::string src(uint32_t code) {
        if (code >= 256) return vreg(code - 256);
        if (code < 128) {
            if (code == kExec || code == kExec + 1 || code < 104 || code == kVcc || code == kVcc + 1 || code == kM0)
                return sreg(code);
            return "0u";
        }
        if (code <= 192) return fmt("%uu", code - 128);
        if (code <= 208) return hex(uint32_t(192 - int32_t(code)));
        if (code >= 240 && code <= 247) {
            static const uint32_t c[] = {0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                                         0x40000000, 0xc0000000, 0x40800000, 0xc0800000};
            return hex(c[code - 240]);
        }
        if (code == 251) return "uint((vcc_lo & 1u) == 0u)";
        if (code == 252) return "uint((exec_lo & 1u) == 0u)";
        if (code == 253) return "uint(scc)";
        if (code == 255) return hex(literal_);
        warn("source operand %u", code);
        return "0u";
    }
    // High half of a 64-bit source.
    std::string srcHi(uint32_t code) {
        if (code < 104 || code == kVcc || code == kExec) return sreg(code + 1);
        if (code >= 193 && code <= 208) return "0xffffffffu";
        return "0u";
    }
    std::string fsrc(uint32_t code, bool abs = false, bool neg = false) {
        std::string s;
        if (code >= 240 && code <= 247) {
            static const char* const c[] = {"0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0"};
            s = c[code - 240];
        } else {
            s = "F(" + src(code) + ")";
        }
        if (abs) s = "abs(" + s + ")";
        if (neg) s = "(-" + s + ")";
        return s;
    }

    // ---- output ----------------------------------------------------------------------------

    void out(const std::string& s) { body_ += s; }
    void warn(const char* f, ...) {
        char b[256];
        va_list ap;
        va_start(ap, f);
        std::vsnprintf(b, sizeof b, f, ap);
        va_end(ap);
        t_.warnings.push_back(fmt("0x%llx: %s", (unsigned long long)pc_, b));
        out(fmt("// unsupported: %s\n", b));
    }
    std::string guard() const { return exec_ ? "if ((exec_lo & 1u) != 0u) " : ""; }
    void writeV(uint32_t n, const std::string& e) { out(guard() + vreg(n) + " = " + e + ";\n"); }
    void writeS(uint32_t code, const std::string& e) {
        if (code == kExec || code == kExec + 1) exec_ = true;
        out(sreg(code) + " = " + e + ";\n");
    }
    void writeS64(uint32_t code, const std::string& lo, const std::string& hi) {
        out(fmt("{ uint lo_ = %s; uint hi_ = %s; ", lo.c_str(), hi.c_str()));
        out(sreg(code) + " = lo_; " + sreg(code + 1) + " = hi_; }\n");
    }
    // A lane-mask result (this lane's bit) into an SGPR pair.
    void writeMask(uint32_t code, const std::string& cond) {
        const std::string active = exec_ ? "(exec_lo & 1u) != 0u && " : "";
        out(fmt("%s = (%s(%s)) ? 1u : 0u; %s = 0u;\n", sreg(code).c_str(), active.c_str(), cond.c_str(),
                sreg(code + 1).c_str()));
    }

    // ---- prologue / epilogue -----------------------------------------------------------------

    void prologue() {
        const LaunchState& l = p_.launch;
        const uint32_t user = l.userSgprs();
        for (uint32_t i = 0; i < user && i < 16; ++i) init_ += fmt("%s = D[%u];\n", sreg(i).c_str(), i);
        init_ += sreg(kExec) + " = 1u;\n";
        switch (l.stage) {
        case Stage::Vertex: {
            init_ += vreg(0) + " = uint(gl_VertexIndex);\n";
            const uint32_t comps = (l.rsrc1 >> 24) & 3;
            if (comps >= 1) init_ += vreg(1) + " = uint(gl_InstanceIndex);\n";
            if (comps >= 2) init_ += vreg(2) + " = uint(gl_InstanceIndex);\n";
            if (comps >= 3) init_ += vreg(3) + " = uint(gl_InstanceIndex);\n";
            decls_ += "out gl_PerVertex { vec4 gl_Position; };\n";
            break;
        }
        case Stage::Pixel: {
            static const uint8_t kCount[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            uint32_t v = 0;
            for (uint32_t bit = 0; bit < 16; ++bit) {
                if (!((l.psInputEna >> bit) & 1)) continue;
                switch (bit) {
                case 8: init_ += vreg(v) + " = U(gl_FragCoord.x);\n"; break;
                case 9: init_ += vreg(v) + " = U(gl_FragCoord.y);\n"; break;
                case 10: init_ += vreg(v) + " = U(gl_FragCoord.z);\n"; break;
                case 11: init_ += vreg(v) + " = U(gl_FragCoord.w);\n"; break;
                case 12: init_ += vreg(v) + " = gl_FrontFacing ? 1u : 0u;\n"; break;
                default: break; // barycentrics: interpolation is done by the rasterizer
                }
                v += kCount[bit];
            }
            break;
        }
        case Stage::Compute: {
            uint32_t s = user;
            for (int i = 0; i < 3; ++i)
                if ((l.rsrc2 >> (7 + i)) & 1) init_ += fmt("%s = gl_WorkGroupID[%d];\n", sreg(s++).c_str(), i);
            if ((l.rsrc2 >> 10) & 1) init_ += fmt("%s = 0u;\n", sreg(s++).c_str());
            const uint32_t comps = (l.rsrc2 >> 11) & 3;
            for (uint32_t i = 0; i <= comps; ++i) init_ += fmt("%s = gl_LocalInvocationID[%u];\n", vreg(i).c_str(), i);
            break;
        }
        }
        // Resources.
        uint32_t binding = base_ + 1;
        for (size_t i = 0; i < p_.buffers.size(); ++i) {
            const auto& b = p_.buffers[i];
            decls_ += fmt("layout(std430, set = 0, binding = %u) %sbuffer Buf%zu { uint m[]; } buf%zu;\n", binding++,
                          b.store ? "" : "readonly ", i, i);
        }
        for (size_t i = 0; i < p_.images.size(); ++i) {
            const auto& im = p_.images[i];
            decls_ += fmt("layout(set = 0, binding = %u) uniform %s%s img%zu;\n", binding++,
                          im.storage ? "writeonly " : "", imageType(im).c_str(), i);
        }
    }

    // 1D images are declared 2D (one row): the host keeps every surface as a 2D image, and a
    // 1D T# often points at a render target (a 1x1 luminance result, for one).
    static const char* dimName(uint32_t type) {
        switch (type) {
        case 10: return "3D";
        case 11: return "Cube";
        case 12: case 13: return "2DArray";
        default: return "2D";
        }
    }
    static char typePrefix(const ImageResource& im) {
        const uint32_t nf = im.numFormat();
        return nf == 4 ? 'u' : nf == 5 ? 'i' : ' ';
    }
    std::string imageType(const ImageResource& im) {
        std::string t;
        const char pfx = typePrefix(im);
        if (pfx != ' ') t += pfx;
        t += im.storage ? "image" : "sampler";
        t += dimName(im.type());
        if (im.compare) t += "Shadow";
        return t;
    }

    std::string helpers() {
        return R"(float F(uint x) { return uintBitsToFloat(x); }
uint U(float x) { return floatBitsToUint(x); }
bool nanSeen_ = false;
uint sat(float x, bool c) { nanSeen_ = nanSeen_ || isnan(x); return U(c ? clamp(x, 0.0, 1.0) : x); }
vec4 nanDbg(vec4 v) { return any(isnan(v)) ? vec4(1.0, 0.0, 1.0, 1.0) : any(isinf(v)) ? vec4(0.0, 1.0, 0.0, 1.0) : v; }
vec4 nanSeenDbg(vec4 v) { return nanSeen_ ? vec4(1.0, 0.0, 1.0, 1.0) : v; }
uint med3u(uint a, uint b, uint c) { return max(min(a, b), min(max(a, b), c)); }
int med3i(int a, int b, int c) { return max(min(a, b), min(max(a, b), c)); }
float med3f(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }
uint bfeU(uint x, uint off, uint w) { off &= 31u; w &= 31u; return w == 0u ? 0u : bitfieldExtract(x, int(off), int(min(w, 32u - off))); }
uint bfeI(uint x, uint off, uint w) { off &= 31u; w &= 31u; return w == 0u ? 0u : uint(bitfieldExtract(int(x), int(off), int(min(w, 32u - off)))); }
uint i24(uint x) { return uint(bitfieldExtract(int(x), 0, 24)); }
bool cmpClass(float x, uint mask) {
    uint b = U(x);
    bool neg = (b >> 31) != 0u;
    uint e = (b >> 23) & 0xffu, m = b & 0x7fffffu;
    uint cls;
    if (e == 0xffu) cls = m != 0u ? ((m & 0x400000u) != 0u ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0u) cls = m == 0u ? (neg ? 5u : 6u) : (neg ? 4u : 7u);
    else cls = neg ? 3u : 8u;
    return ((mask >> cls) & 1u) != 0u;
}
// v_cube*: face id, face s/t and 2x the major axis of a direction (GCN face order +x -x +y -y +z -z).
float cubeId(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return d.z < 0.0 ? 5.0 : 4.0;
    if (a.y >= a.x) return d.y < 0.0 ? 3.0 : 2.0;
    return d.x < 0.0 ? 1.0 : 0.0;
}
float cubeSc(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return d.z < 0.0 ? -d.x : d.x;
    if (a.y >= a.x) return d.x;
    return d.x < 0.0 ? d.z : -d.z;
}
float cubeTc(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return -d.y;
    if (a.y >= a.x) return d.y < 0.0 ? -d.z : d.z;
    return -d.y;
}
float cubeMa(vec3 d) {
    vec3 a = abs(d);
    if (a.z >= a.x && a.z >= a.y) return 2.0 * d.z;
    if (a.y >= a.x) return 2.0 * d.y;
    return 2.0 * d.x;
}
// Cube image coordinates (s, t in [1, 2] as v_cube* produce them, face id) back to a direction.
vec3 cubeDir(vec3 c) {
    float u = 2.0 * c.x - 3.0, v = 2.0 * c.y - 3.0;
    switch (int(c.z + 0.5) & 7) {
    case 0: return vec3(1.0, -v, -u);
    case 1: return vec3(-1.0, -v, u);
    case 2: return vec3(u, 1.0, v);
    case 3: return vec3(u, -1.0, -v);
    case 4: return vec3(u, -v, 1.0);
    default: return vec3(-u, -v, -1.0);
    }
}
)";
    }

    // ---- instructions ------------------------------------------------------------------------

    // LPN_PROBE=<pixel shader entry>:<pc>:<vgpr>,<vgpr>,<vgpr> (hex, hex, decimal): that shader's
    // colour target 0 shows the three registers as they are just before the instruction at pc
    // (magenta where it is never reached), for working out where a shader goes wrong.
    struct Probe {
        uint64_t entry = 0, pc = 0;
        uint32_t regs[3] = {};
    };
    static const Probe* probeSetting() {
        static const Probe* p = [] () -> const Probe* {
            const char* e = std::getenv("LPN_PROBE");
            static Probe pr;
            if (!e || std::sscanf(e, "%" SCNx64 ":%" SCNx64 ":%u,%u,%u", &pr.entry, &pr.pc, &pr.regs[0], &pr.regs[1], &pr.regs[2]) != 5)
                return nullptr;
            return &pr;
        }();
        return p;
    }
    bool probing() const {
        const Probe* p = probeSetting();
        return p && p_.launch.stage == Stage::Pixel && p_.launch.entry == p->entry;
    }
    void probe(uint64_t pc) {
        if (!probing()) return;
        const Probe* p = probeSetting();
        if (pc - p_.launch.entry != p->pc) return;
        if (!probed_) init_ += "vec3 probe_ = vec3(1.0, 0.0, 1.0);\n";
        probed_ = true;
        out(fmt("probe_ = vec3(F(%s), F(%s), F(%s));\n", vreg(p->regs[0]).c_str(), vreg(p->regs[1]).c_str(), vreg(p->regs[2]).c_str()));
    }
    bool probed_ = false;

    void instruction(const Instruction& in) {
        literal_ = in.literal;
        switch (in.format) {
        case Format::SOPP: sopp(in); break;
        case Format::SOP1: sop1(in); break;
        case Format::SOP2: sop2(in); break;
        case Format::SOPC: sopc(in); break;
        case Format::SOPK: sopk(in); break;
        case Format::SMRD: smrd(in); break;
        case Format::VOP1:
        case Format::VOP2:
        case Format::VOPC:
        case Format::VOP3: vop(in); break;
        case Format::VINTRP: vintrp(in); break;
        case Format::EXP: exp(in); break;
        case Format::MUBUF:
        case Format::MTBUF: buffer(in); break;
        case Format::MIMG: image(in); break;
        case Format::DS: ds(in); break;
        default: warn("%s", mnemonic(in).c_str()); break;
        }
    }

    std::string label(uint64_t a) { return hex(uint32_t(a)).substr(0, 10) + "u"; }
    void jump(const std::string& cond, uint64_t target) {
        if (!p_.labels.count(target)) return; // the analysis proved this branch is never taken
        if (cond.empty()) out("pc = " + label(target) + "; break;\n");
        else out("if (" + cond + ") { pc = " + label(target) + "; break; }\n");
    }

    void sopp(const Instruction& in) {
        const uint64_t target = pc_ + 4 + int64_t(int16_t(in.words[0] & 0xFFFF)) * 4;
        switch (in.op) {
        case 0: case 12: case 13: case 14: case 15: case 16: case 17: case 19: break; // nop, waitcnt, prio, msgs
        case 1: out("return;\n"); break;
        case 2: jump("", target); break;
        case 4: jump("!scc", target); break;
        case 5: jump("scc", target); break;
        case 6: jump("(" + sreg(kVcc) + " & 1u) == 0u", target); break;
        case 7: jump("(" + sreg(kVcc) + " & 1u) != 0u", target); break;
        case 8: jump("(" + sreg(kExec) + " & 1u) == 0u", target); break;
        case 9: jump("(" + sreg(kExec) + " & 1u) != 0u", target); break;
        case 10:
            if (p_.launch.stage == Stage::Compute) out("barrier();\n");
            break;
        default: warn("%s", mnemonic(in).c_str()); break;
        }
    }

    void sop1(const Instruction& in) {
        const uint32_t w = in.words[0];
        const uint32_t d = (w >> 16) & 0x7F, s = w & 0xFF;
        const std::string a = src(s);
        switch (in.op) {
        case 3: writeS(d, a); break;
        case 4: writeS64(d, a, srcHi(s)); break;
        case 5: out("if (scc) " + sreg(d) + " = " + a + ";\n"); break;
        case 6: out("if (scc) { uint lo_ = " + a + "; uint hi_ = " + srcHi(s) + "; " + sreg(d) + " = lo_; " +
                    sreg(d + 1) + " = hi_; }\n"); break;
        case 7: writeS(d, "~" + a); out("scc = " + sreg(d) + " != 0u;\n"); break;
        case 8: writeS64(d, "~" + a, "~" + srcHi(s)); out("scc = (" + sreg(d) + " | " + sreg(d + 1) + ") != 0u;\n"); break;
        case 9: writeS(d, a); break;                   // s_wqm_b32: lanes are not grouped here
        case 10: writeS64(d, a, srcHi(s)); break;      // s_wqm_b64
        case 11: writeS(d, "bitfieldReverse(" + a + ")"); break;
        case 15: writeS(d, "uint(bitCount(" + a + "))"); out("scc = " + sreg(d) + " != 0u;\n"); break;
        case 16: writeS(d, "uint(bitCount(" + a + ") + bitCount(" + srcHi(s) + "))"); out("scc = " + sreg(d) + " != 0u;\n"); break;
        case 19: writeS(d, "uint(findLSB(" + a + "))"); break;
        case 21: writeS(d, "(" + a + ") == 0u ? 0xffffffffu : 31u - uint(findMSB(" + a + "))"); break;
        case 25: writeS(d, "uint(bitfieldExtract(int(" + a + "), 0, 8))"); break;
        case 26: writeS(d, "uint(bitfieldExtract(int(" + a + "), 0, 16))"); break;
        case 31: writeS64(d, label(next_), hex(uint32_t(next_ >> 32))); break; // s_getpc_b64
        case 32: out(p_.controlFlow ? "pc = " + a + "; break;\n" : "return;\n"); break; // s_setpc_b64
        case 33: { // s_swappc_b64
            writeS64(d, label(next_), hex(uint32_t(next_ >> 32)));
            auto call = p_.calls.find(pc_);
            if (call != p_.calls.end()) out("pc = " + label(call->second) + "; break;\n");
            break;
        }
        case 36: case 37: case 38: case 39: case 40: case 41: case 42: case 43: { // s_*_saveexec_b64
            static const char* const opn[] = {"%s & %s", "%s | %s", "%s ^ %s", "%s & ~%s",
                                              "%s | ~%s", "~(%s & %s)", "~(%s | %s)", "~(%s ^ %s)"};
            const char* o = opn[in.op - 36];
            exec_ = true;
            out("{ uint lo_ = " + a + "; uint hi_ = " + srcHi(s) + "; ");
            out(sreg(d) + " = exec_lo; " + sreg(d + 1) + " = exec_hi; ");
            out("exec_lo = " + fmt(o, "lo_", "exec_lo") + "; exec_hi = " + fmt(o, "hi_", "exec_hi") + "; ");
            out("scc = (exec_lo & 1u) != 0u; }\n");
            regs_.insert("exec_hi");
            break;
        }
        default: warn("%s", mnemonic(in).c_str()); break;
        }
    }

    void sop2(const Instruction& in) {
        const uint32_t w = in.words[0];
        const uint32_t d = (w >> 16) & 0x7F, s0 = w & 0xFF, s1 = (w >> 8) & 0xFF;
        const std::string a = src(s0), b = src(s1);
        const bool wide = in.op >= 15 && in.op <= 29 && (in.op & 1);
        if (wide) {
            static const char* const opn[] = {"%s & %s", "%s | %s", "%s ^ %s", "%s & ~%s",
                                              "%s | ~%s", "~(%s & %s)", "~(%s | %s)", "~(%s ^ %s)"};
            const char* o = opn[(in.op - 15) / 2];
            const std::string ah = srcHi(s0), bh = srcHi(s1);
            writeS64(d, fmt(o, a.c_str(), b.c_str()), fmt(o, ah.c_str(), bh.c_str()));
            out("scc = (" + sreg(d) + " | " + sreg(d + 1) + ") != 0u;\n");
            return;
        }
        auto set = [&](const std::string& e, bool scc = true) {
            writeS(d, e);
            if (scc) out("scc = " + sreg(d) + " != 0u;\n");
        };
        switch (in.op) {
        case 0: out(fmt("{ uint a_ = %s; uint r_ = a_ + %s; scc = r_ < a_; ", a.c_str(), b.c_str()) + sreg(d) + " = r_; }\n"); break;
        case 1: out(fmt("{ uint a_ = %s; uint b_ = %s; scc = b_ > a_; ", a.c_str(), b.c_str()) + sreg(d) + " = a_ - b_; }\n"); break;
        case 2: writeS(d, a + " + " + b); break;
        case 3: writeS(d, a + " - " + b); break;
        case 4: out(fmt("{ uint a_ = %s; uint r_ = a_ + %s + uint(scc); scc = r_ < a_ || (scc && r_ == a_); ", a.c_str(), b.c_str()) + sreg(d) + " = r_; }\n"); break;
        case 5: out(fmt("{ uint a_ = %s; uint b_ = %s; uint c_ = uint(scc); scc = b_ > a_ || (c_ != 0u && b_ == a_); ", a.c_str(), b.c_str()) + sreg(d) + " = a_ - b_ - c_; }\n"); break;
        case 6: writeS(d, "uint(min(int(" + a + "), int(" + b + ")))"); break;
        case 7: writeS(d, "min(" + a + ", " + b + ")"); break;
        case 8: writeS(d, "uint(max(int(" + a + "), int(" + b + ")))"); break;
        case 9: writeS(d, "max(" + a + ", " + b + ")"); break;
        case 10: writeS(d, "scc ? " + a + " : " + b); break;
        case 11: writeS64(d, "scc ? " + a + " : " + b, "scc ? " + srcHi(s0) + " : " + srcHi(s1)); break;
        case 14: set(a + " & " + b); break;
        case 16: set(a + " | " + b); break;
        case 18: set(a + " ^ " + b); break;
        case 20: set(a + " & ~" + b); break;
        case 22: set(a + " | ~" + b); break;
        case 24: set("~(" + a + " & " + b + ")"); break;
        case 26: set("~(" + a + " | " + b + ")"); break;
        case 28: set("~(" + a + " ^ " + b + ")"); break;
        case 30: set(a + " << (" + b + " & 31u)"); break;
        case 32: set(a + " >> (" + b + " & 31u)"); break;
        case 34: set("uint(int(" + a + ") >> (" + b + " & 31u))"); break;
        case 36: writeS(d, "bitfieldInsert(0u, 0xffffffffu, int(" + b + " & 31u), int(" + a + " & 31u))"); break;
        case 38: writeS(d, a + " * " + b); break;
        case 39: set("bfeU(" + a + ", " + b + ", " + b + " >> 16)"); break;
        case 40: set("bfeI(" + a + ", " + b + ", " + b + " >> 16)"); break;
        case 44: set("uint(abs(int(" + a + ") - int(" + b + ")))"); break;
        default: warn("%s", mnemonic(in).c_str()); break;
        }
    }

    static std::string cmpExpr(uint32_t op, const std::string& a, const std::string& b, bool isSigned) {
        const std::string x = isSigned ? "int(" + a + ")" : a, y = isSigned ? "int(" + b + ")" : b;
        static const char* const ops[] = {"==", "!=", ">", ">=", "<", "<="};
        return x + " " + ops[op] + " " + y;
    }

    void sopc(const Instruction& in) {
        const uint32_t w = in.words[0];
        const std::string a = src(w & 0xFF), b = src((w >> 8) & 0xFF);
        if (in.op < 12) out("scc = " + cmpExpr(in.op % 6, a, b, in.op < 6) + ";\n");
        else if (in.op == 12) out("scc = ((" + a + " >> (" + b + " & 31u)) & 1u) == 0u;\n");
        else if (in.op == 13) out("scc = ((" + a + " >> (" + b + " & 31u)) & 1u) != 0u;\n");
        else warn("%s", mnemonic(in).c_str());
    }

    void sopk(const Instruction& in) {
        const uint32_t w = in.words[0];
        const uint32_t d = (w >> 16) & 0x7F, imm = w & 0xFFFF;
        const std::string simm = hex(uint32_t(int32_t(int16_t(imm))));
        switch (in.op) {
        case 0: writeS(d, simm); break;
        case 2: out("if (scc) " + sreg(d) + " = " + simm + ";\n"); break;
        case 15: writeS(d, sreg(d) + " + " + simm); break;
        case 16: writeS(d, sreg(d) + " * " + simm); break;
        default:
            if (in.op >= 3 && in.op <= 14) {
                const uint32_t c = in.op - 3;
                out("scc = " + cmpExpr(c % 6, sreg(d), c < 6 ? simm : hex(imm), c < 6) + ";\n");
            } else {
                warn("%s", mnemonic(in).c_str());
            }
            break;
        }
    }

    void smrd(const Instruction& in) {
        if (in.op >= 29) return; // cache control, s_memtime
        auto it = p_.smrd.find(pc_);
        const uint32_t d = (in.words[0] >> 15) & 0x7F;
        static const unsigned kCount[] = {1, 2, 4, 8, 16};
        const unsigned n = kCount[in.op & 7];
        if (it == p_.smrd.end()) {
            warn("scalar load not analysed");
            return;
        }
        for (unsigned i = 0; i < n; ++i) out(sreg(d + i) + " = D[" + std::to_string(it->second + i) + "];\n");
    }

    // ---- vector ALU --------------------------------------------------------------------------

    void vop(const Instruction& in) {
        const uint32_t w = in.words[0], w1 = in.words[1];
        uint32_t op = in.op, s0 = 0, s1 = 0, s2 = 0, dst = 0, abs = 0, neg = 0, omod = 0, sdst = kVcc;
        bool clampBit = false, vop3 = false;
        switch (in.format) {
        case Format::VOPC: s0 = w & 0x1FF; s1 = 256 + ((w >> 9) & 0xFF); break;
        case Format::VOP2: op = 256 + in.op; s0 = w & 0x1FF; s1 = 256 + ((w >> 9) & 0xFF); dst = (w >> 17) & 0xFF; break;
        case Format::VOP1: op = 384 + in.op; s0 = w & 0x1FF; dst = (w >> 17) & 0xFF; break;
        default:
            vop3 = true;
            dst = w & 0xFF;
            s0 = w1 & 0x1FF;
            s1 = (w1 >> 9) & 0x1FF;
            s2 = (w1 >> 18) & 0x1FF;
            omod = (w1 >> 27) & 3;
            neg = w1 >> 29;
            if ((op >= 293 && op <= 298) || op == 365 || op == 366 || op == 374 || op == 375) {
                sdst = (w >> 8) & 0x7F;
            } else {
                abs = (w >> 8) & 7;
                clampBit = (w >> 11) & 1;
            }
            if (op < 256) sdst = dst;
            break;
        }
        const std::string a = src(s0), b = s1 ? src(s1) : "0u", c = vop3 ? src(s2) : "0u";
        const std::string fa = fsrc(s0, abs & 1, neg & 1), fb = fsrc(s1, abs & 2, neg & 2), fc = fsrc(s2, abs & 4, neg & 4);
        auto fres = [&](const std::string& e) {
            std::string r = e;
            if (omod == 1) r = "(" + r + ") * 2.0";
            else if (omod == 2) r = "(" + r + ") * 4.0";
            else if (omod == 3) r = "(" + r + ") * 0.5";
            writeV(dst, "sat(" + r + ", " + (clampBit ? "true" : "false") + ")");
        };
        auto ures = [&](const std::string& e) { writeV(dst, e); };
        const std::string vd = vreg(dst);

        if (op < 256) { // compares
            const uint32_t group = op >> 5, n = op & 15;
            const bool x = (op >> 4) & 1;
            std::string cond;
            if (group == 0 || group == 2) { // f32 (v_cmp, v_cmps)
                static const char* const f[] = {"false", "%s < %s", "%s == %s", "%s <= %s", "%s > %s",
                                                "%s != %s && !isnan(%s) && !isnan(%s)", "%s >= %s",
                                                "!isnan(%s) && !isnan(%s)", "isnan(%s) || isnan(%s)",
                                                "!(%s >= %s)", "!(%s != %s && !isnan(%s) && !isnan(%s))", "!(%s > %s)",
                                                "!(%s <= %s)", "!(%s == %s)", "!(%s < %s)", "true"};
                cond = fmt(f[n], fa.c_str(), fb.c_str(), fa.c_str(), fb.c_str());
            } else if (op >= 128 && (op < 160 || (op >= 192 && op < 224))) {
                const bool sgn = op < 160;
                if (n == 8 && sgn) {
                    cond = "cmpClass(" + fa + ", " + b + ")";
                } else if (n < 8) {
                    const std::string ia = sgn ? "int(" + a + ")" : a, ib = sgn ? "int(" + b + ")" : b;
                    static const char* const i[] = {"false", "<", "==", "<=", ">", "!=", ">=", "true"};
                    cond = (n == 0 || n == 7) ? i[n] : ia + " " + i[n] + " " + ib;
                } else {
                    warn("%s", mnemonic(in).c_str());
                    return;
                }
            } else {
                warn("%s", mnemonic(in).c_str());
                return;
            }
            if (x) {
                exec_ = true;
                out("{ bool c_ = (exec_lo & 1u) != 0u && (" + cond + "); exec_lo = c_ ? 1u : 0u; exec_hi = 0u; ");
                regs_.insert("exec_hi");
                if (sdst != kExec) out(sreg(sdst) + " = exec_lo; " + sreg(sdst + 1) + " = 0u; ");
                out("}\n");
            } else {
                writeMask(sdst, cond);
            }
            return;
        }

        switch (op) {
        // VOP2
        case 256: { // v_cndmask_b32
            const std::string mask = vop3 ? src(s2) : sreg(kVcc);
            ures("(" + mask + " & 1u) != 0u ? " + b + " : " + a);
            break;
        }
        case 257: writeS(dst, a); break; // v_readlane_b32 (lane-local)
        case 258: writeV(dst, a); break; // v_writelane_b32
        case 259: fres(fa + " + " + fb); break;
        case 260: fres(fa + " - " + fb); break;
        case 261: fres(fb + " - " + fa); break;
        case 262: fres("((" + fa + " == 0.0 || " + fb + " == 0.0) ? 0.0 : " + fa + " * " + fb + ") + F(" + vd + ")"); break;
        case 263: fres("(" + fa + " == 0.0 || " + fb + " == 0.0) ? 0.0 : " + fa + " * " + fb); break;
        case 264: fres(fa + " * " + fb); break;
        case 265: ures("uint(int(i24(" + a + ")) * int(i24(" + b + ")))"); break;
        case 267: ures("(" + a + " & 0xffffffu) * (" + b + " & 0xffffffu)"); break;
        case 269: case 271: fres("min(" + fa + ", " + fb + ")"); break;
        case 270: case 272: fres("max(" + fa + ", " + fb + ")"); break;
        case 273: ures("uint(min(int(" + a + "), int(" + b + ")))"); break;
        case 274: ures("uint(max(int(" + a + "), int(" + b + ")))"); break;
        case 275: ures("min(" + a + ", " + b + ")"); break;
        case 276: ures("max(" + a + ", " + b + ")"); break;
        case 277: ures(a + " >> (" + b + " & 31u)"); break;
        case 278: ures(b + " >> (" + a + " & 31u)"); break;
        case 279: ures("uint(int(" + a + ") >> (" + b + " & 31u))"); break;
        case 280: ures("uint(int(" + b + ") >> (" + a + " & 31u))"); break;
        case 281: ures(a + " << (" + b + " & 31u)"); break;
        case 282: ures(b + " << (" + a + " & 31u)"); break;
        case 283: ures(a + " & " + b); break;
        case 284: ures(a + " | " + b); break;
        case 285: ures(a + " ^ " + b); break;
        case 286: ures("bitfieldInsert(0u, 0xffffffffu, int(" + b + " & 31u), int(" + a + " & 31u))"); break;
        case 287: fres(fa + " * " + fb + " + F(" + vd + ")"); break;               // v_mac_f32
        case 288: fres(fa + " * F(" + hex(literal_) + ") + " + fb); break;         // v_madmk_f32
        case 289: fres(fa + " * " + fb + " + F(" + hex(literal_) + ")"); break;    // v_madak_f32
        case 290: ures("uint(bitCount(" + a + ")) + " + b); break;
        case 291: ures("uint(bitCount(" + a + " & ((1u << min(laneId(), 31u)) - 1u))) + " + b); needLane_ = true; break;
        case 292: ures("uint(bitCount(" + a + " & (laneId() > 32u ? ((1u << (laneId() - 32u)) - 1u) : 0u))) + " + b); needLane_ = true; break;
        case 293: // v_add_i32 (carry out)
            out(fmt("{ uint a_ = %s; uint r_ = a_ + %s; ", a.c_str(), b.c_str()));
            out(guard() + vd + " = r_; " + sreg(sdst) + " = uint(r_ < a_); " + sreg(sdst + 1) + " = 0u; }\n");
            break;
        case 294: // v_sub_i32
            out(fmt("{ uint a_ = %s; uint b_ = %s; ", a.c_str(), b.c_str()));
            out(guard() + vd + " = a_ - b_; " + sreg(sdst) + " = uint(b_ > a_); " + sreg(sdst + 1) + " = 0u; }\n");
            break;
        case 295: // v_subrev_i32
            out(fmt("{ uint a_ = %s; uint b_ = %s; ", a.c_str(), b.c_str()));
            out(guard() + vd + " = b_ - a_; " + sreg(sdst) + " = uint(a_ > b_); " + sreg(sdst + 1) + " = 0u; }\n");
            break;
        case 296: { // v_addc_u32
            const std::string cin = vop3 ? "(" + c + " & 1u)" : "(" + sreg(kVcc) + " & 1u)";
            out(fmt("{ uint a_ = %s; uint ci_ = %s; uint r_ = a_ + %s + ci_; ", a.c_str(), cin.c_str(), b.c_str()));
            out(guard() + vd + " = r_; " + sreg(sdst) + " = uint(r_ < a_ || (ci_ != 0u && r_ == a_)); " + sreg(sdst + 1) + " = 0u; }\n");
            break;
        }
        case 299: fres("ldexp(" + fa + ", int(" + b + "))"); break;
        case 303: ures("packHalf2x16(vec2(" + fa + ", " + fb + "))"); break; // v_cvt_pkrtz_f16_f32
        case 302: ures("packUnorm2x16(vec2(" + fa + ", " + fb + "))"); break;
        case 301: ures("packSnorm2x16(vec2(" + fa + ", " + fb + "))"); break;
        case 304: ures("(" + a + " & 0xffffu) | (" + b + " << 16)"); break;
        case 305: ures("(" + a + " & 0xffffu) | (" + b + " << 16)"); break;
        // VOP3 only
        case 320: case 321: fres(fa + " * " + fb + " + " + fc); break;
        case 322: ures("uint(int(i24(" + a + ")) * int(i24(" + b + "))) + " + c); break;
        case 323: ures("(" + a + " & 0xffffffu) * (" + b + " & 0xffffffu) + " + c); break;
        case 324: fres("cubeId(vec3(" + fa + ", " + fb + ", " + fc + "))"); break;
        case 325: fres("cubeSc(vec3(" + fa + ", " + fb + ", " + fc + "))"); break;
        case 326: fres("cubeTc(vec3(" + fa + ", " + fb + ", " + fc + "))"); break;
        case 327: fres("cubeMa(vec3(" + fa + ", " + fb + ", " + fc + "))"); break;
        case 328: ures("bfeU(" + a + ", " + b + ", " + c + ")"); break;
        case 329: ures("bfeI(" + a + ", " + b + ", " + c + ")"); break;
        case 330: ures("(" + a + " & " + b + ") | (~" + a + " & " + c + ")"); break;
        case 331: fres("fma(" + fa + ", " + fb + ", " + fc + ")"); break;
        case 334: ures("uint((uint64_t(" + a + ") << 32 | uint64_t(" + b + ")) >> (" + c + " & 31u))"); need64_ = true; break;
        case 335: ures("uint((uint64_t(" + a + ") << 32 | uint64_t(" + b + ")) >> ((" + c + " & 3u) * 8u))"); need64_ = true; break;
        case 337: fres("min(min(" + fa + ", " + fb + "), " + fc + ")"); break;
        case 338: ures("uint(min(min(int(" + a + "), int(" + b + ")), int(" + c + ")))"); break;
        case 339: ures("min(min(" + a + ", " + b + "), " + c + ")"); break;
        case 340: fres("max(max(" + fa + ", " + fb + "), " + fc + ")"); break;
        case 341: ures("uint(max(max(int(" + a + "), int(" + b + ")), int(" + c + ")))"); break;
        case 342: ures("max(max(" + a + ", " + b + "), " + c + ")"); break;
        case 343: fres("med3f(" + fa + ", " + fb + ", " + fc + ")"); break;
        case 344: ures("uint(med3i(int(" + a + "), int(" + b + "), int(" + c + ")))"); break;
        case 345: ures("med3u(" + a + ", " + b + ", " + c + ")"); break;
        case 349: ures("(max(" + a + ", " + b + ") - min(" + a + ", " + b + ")) + " + c); break; // v_sad_u32
        case 350: ures("(" + c + " & ~(0xffu << ((" + b + " & 3u) * 8u))) | ((uint(clamp(round(" + fa +
                       "), 0.0, 255.0)) & 0xffu) << ((" + b + " & 3u) * 8u))"); break;
        case 351: fres(fc + " / " + fb); break; // v_div_fixup_f32: result of the division sequence
        case 361: case 363: ures(a + " * " + b); break;
        case 362: out(fmt("{ uint h_, l_; umulExtended(%s, %s, h_, l_); ", a.c_str(), b.c_str()) + guard() + vd + " = h_; }\n"); break;
        case 364: out(fmt("{ int h_, l_; imulExtended(int(%s), int(%s), h_, l_); ", a.c_str(), b.c_str()) + guard() + vd + " = uint(h_); }\n"); break;
        case 365: // v_div_scale_f32: no scaling needed at GLSL precision
            out(guard() + vd + " = " + a + "; " + sreg(sdst) + " = 0u; " + sreg(sdst + 1) + " = 0u;\n");
            break;
        case 367: fres(fa + " * " + fb + " + " + fc); break; // v_div_fmas_f32
        case 374: { // v_mad_u64_u32: v[dst:dst+1] = a * b + c (64-bit c), carry out to sdst
            const std::string ch = s2 >= 256 ? vreg(s2 - 255) : srcHi(s2);
            out(fmt("{ uint h_, l_, c0_, c1_, c2_; umulExtended(%s, %s, h_, l_); l_ = uaddCarry(l_, %s, c0_); ",
                    a.c_str(), b.c_str(), c.c_str()));
            out(fmt("h_ = uaddCarry(uaddCarry(h_, %s, c1_), c0_, c2_); ", ch.c_str()));
            out(guard() + "{ " + vd + " = l_; " + vreg(dst + 1) + " = h_; } " + sreg(sdst) + " = c1_ | c2_; " +
                sreg(sdst + 1) + " = 0u; }\n");
            break;
        }
        // VOP1
        case 384: break;
        case 385: // v_mov_b32; in VOP3 form its abs/neg/omod/clamp apply as for a float move (games negate with it)
            if (clampBit || omod) fres(fa);
            else ures("(" + a + (abs & 1 ? " & 0x7fffffffu" : "") + ")" + (neg & 1 ? " ^ 0x80000000u" : ""));
            break;
        case 386: writeS(dst, a); break; // v_readfirstlane_b32 (lane-local)
        case 389: fres("float(int(" + a + "))"); break;
        case 398: fres("float(bitfieldExtract(int(" + a + "), 0, 4)) * 0.0625"); break; // v_cvt_off_f32_i4
        case 390: fres("float(" + a + ")"); break;
        case 391: ures("uint(clamp(" + fa + ", 0.0, 4294967040.0))"); break;
        case 392: ures("uint(int(clamp(" + fa + ", -2147483648.0, 2147483520.0)))"); break;
        case 394: ures("packHalf2x16(vec2(" + fa + ", 0.0))"); break;
        case 395: fres("unpackHalf2x16(" + a + ").x"); break;
        case 396: ures("uint(int(floor(" + fa + " + 0.5)))"); break;
        case 397: ures("uint(int(floor(" + fa + ")))"); break;
        case 401: case 402: case 403: case 404:
            fres(fmt("float((%s >> %du) & 0xffu)", a.c_str(), (op - 401) * 8));
            break;
        case 416: fres("fract(" + fa + ")"); break;
        case 417: fres("trunc(" + fa + ")"); break;
        case 418: fres("ceil(" + fa + ")"); break;
        case 419: fres("roundEven(" + fa + ")"); break;
        case 420: fres("floor(" + fa + ")"); break;
        case 421: fres("exp2(" + fa + ")"); break;
        case 422: case 423: fres("log2(" + fa + ")"); break;
        case 424: case 425: case 426: case 427: fres("1.0 / " + fa); break;
        case 428: case 429: case 430: fres("inversesqrt(" + fa + ")"); break;
        case 435: fres("sqrt(" + fa + ")"); break;
        case 437: fres("sin(" + fa + " * 6.283185307179586)"); break;
        case 438: fres("cos(" + fa + " * 6.283185307179586)"); break;
        case 439: ures("~" + a); break;
        case 440: ures("bitfieldReverse(" + a + ")"); break;
        case 441: ures("(" + a + ") == 0u ? 0xffffffffu : 31u - uint(findMSB(" + a + "))"); break;
        case 442: ures("uint(findLSB(" + a + "))"); break;
        case 443: ures("uint(findMSB(int(" + a + "))) == 0xffffffffu ? 0xffffffffu : 31u - uint(findMSB(int(" + a + ")))"); break;
        case 447: ures("uint(" + fa + " == 0.0 || isinf(" + fa + ") || isnan(" + fa + ") ? 0 : int(floor(log2(abs(" + fa + ")))) + 1)"); break;
        case 448: fres(fa + " == 0.0 || isinf(" + fa + ") || isnan(" + fa + ") ? " + fa + " : " + fa + " / exp2(floor(log2(abs(" + fa + "))) + 1.0)"); break;
        default: warn("%s", mnemonic(in).c_str()); break;
        }
    }

    // ---- interpolation and exports -------------------------------------------------------------

    // LDS and GDS are not emulated; ds_swizzle_b32 only moves data between lanes.
    void ds(const Instruction& in) {
        const uint32_t w = in.words[0], w1 = in.words[1];
        const uint32_t offset = w & 0xFFFF; // offset1:offset0
        if (in.op != 53 || !(offset & 0x8000)) { // the bit-mask mode would need GCN's lane layout
            warn("%s", mnemonic(in).c_str());
            return;
        }
        // Quad mode: lane i of each quad reads lane (offset >> 2i) & 3 of its quad. GCN and Vulkan
        // number a fragment quad's pixels alike (top-left, top-right, bottom-left, bottom-right).
        // The broadcast runs outside the exec guard so the whole quad takes part.
        const std::string v = vreg(w1 & 0xFF);
        const uint32_t sel = offset & 0xFF;
        needQuad_ = true;
        const std::string e = sel == (sel & 3) * 0x55 ? fmt("subgroupQuadBroadcast(%s, %uu)", v.c_str(), sel & 3)
                                                      : fmt("quadSwizzle(%s, %uu)", v.c_str(), sel);
        out("{ uint q_ = " + e + "; " + guard() + vreg(w1 >> 24) + " = q_; }\n");
    }

    void vintrp(const Instruction& in) {
        const uint32_t w = in.words[0];
        const uint32_t vdst = (w >> 18) & 0xFF, attr = (w >> 10) & 0x3F, chan = (w >> 8) & 3;
        if (in.op == 0) return; // v_interp_p1_f32: p2 produces the whole interpolated value
        const uint32_t cntl = p_.launch.psInputCntl[attr];
        const uint32_t offset = cntl & 0x3F;
        if (offset & 0x20) { // default value
            static const float defaults[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}};
            writeV(vdst, fmt("U(%.1f)", defaults[(cntl >> 8) & 3][chan]));
            return;
        }
        const bool flat = in.op == 2 || ((cntl >> 10) & 1);
        if (!(t_.inputs & (1u << offset))) {
            t_.inputs |= 1u << offset;
            decls_ += fmt("layout(location = %u) %sin vec4 iA%u;\n", offset, flat ? "flat " : "", offset);
        }
        writeV(vdst, fmt("U(iA%u[%u])", offset, chan));
    }

    void exp(const Instruction& in) {
        const uint32_t w = in.words[0], w1 = in.words[1];
        const uint32_t en = w & 0xF, target = (w >> 4) & 0x3F;
        const bool compr = (w >> 10) & 1, vm = (w >> 12) & 1;
        const uint32_t vs[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, w1 >> 24};
        auto comp = [&](int i) { return (en >> i) & 1 ? "F(" + vreg(vs[i]) + ")" : std::string("0.0"); };
        auto vec = [&] { return "vec4(" + comp(0) + ", " + comp(1) + ", " + comp(2) + ", " + comp(3) + ")"; };
        if (p_.launch.stage == Stage::Pixel && vm && exec_) out("if ((exec_lo & 1u) == 0u) discard;\n");

        if (target == 12) { // pos0
            out("gl_Position = " + vec() + ";\n");
            out(fmt("gl_Position.xy = gl_Position.xy * vec2(F(D[%u]), F(D[%u])) + vec2(F(D[%u]), F(D[%u])) * gl_Position.w;\n",
                    data::kPosTransform, data::kPosTransform + 1, data::kPosTransform + 2, data::kPosTransform + 3));
            if (p_.launch.glClipZ) out("gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;\n");
            return;
        }
        if (target >= 13 && target <= 15) return; // misc vectors, clip distances
        if (target >= 32) {
            const uint32_t n = target - 32;
            if (!(t_.outputs & (1u << n))) {
                t_.outputs |= 1u << n;
                decls_ += fmt("layout(location = %u) out vec4 oP%u;\n", n, n);
            }
            if (compr) out(fmt("oP%u = vec4(unpackHalf2x16(%s), unpackHalf2x16(%s));\n", n, vreg(vs[0]).c_str(), vreg(vs[1]).c_str()));
            else out(fmt("oP%u = %s;\n", n, vec().c_str()));
            return;
        }
        if (target == 8) { // mrtz
            if (p_.launch.depthExport && (en & 1)) {
                if (!depthDeclared_) decls_ += "layout(depth_any) out float gl_FragDepth;\n";
                depthDeclared_ = true;
                out("gl_FragDepth = F(" + vreg(vs[0]) + ");\n");
            }
            return;
        }
        if (target == 9) return; // null
        if (target < 8) {
            // Export slots skip MRTs whose export format is ZERO.
            uint32_t mrt = 0, slot = 0;
            for (mrt = 0; mrt < 8; ++mrt) {
                if (!((p_.launch.colorFormat >> (mrt * 4)) & 0xF)) continue;
                if (slot++ == target) break;
            }
            if (mrt == 8) return;
            const uint32_t format = (p_.launch.colorFormat >> (mrt * 4)) & 0xF;
            const bool isU = (p_.launch.colorUint >> mrt) & 1, isS = (p_.launch.colorSint >> mrt) & 1;
            const char* type = isU ? "uvec4" : isS ? "ivec4" : "vec4";
            if (!(t_.colorOutputs & (1u << mrt))) {
                t_.colorOutputs |= 1u << mrt;
                decls_ += fmt("layout(location = %u) out %s oC%u;\n", mrt, type, mrt);
            }
            std::string v;
            if (compr) {
                const std::string a = vreg(vs[0]), b = vreg(vs[1]);
                switch (format) {
                case 5: v = fmt("vec4(unpackUnorm2x16(%s), unpackUnorm2x16(%s))", a.c_str(), b.c_str()); break;
                case 6: v = fmt("vec4(unpackSnorm2x16(%s), unpackSnorm2x16(%s))", a.c_str(), b.c_str()); break;
                case 7: v = fmt("uvec4(%s & 0xffffu, %s >> 16, %s & 0xffffu, %s >> 16)", a.c_str(), a.c_str(), b.c_str(), b.c_str()); break;
                case 8: v = fmt("ivec4(bitfieldExtract(int(%s), 0, 16), int(%s) >> 16, bitfieldExtract(int(%s), 0, 16), int(%s) >> 16)", a.c_str(), a.c_str(), b.c_str(), b.c_str()); break;
                default: v = fmt("vec4(unpackHalf2x16(%s), unpackHalf2x16(%s))", a.c_str(), b.c_str()); break;
                }
            } else if (isU || isS) {
                auto r = [&](int i) { return (en >> i) & 1 ? vreg(vs[i]) : std::string("0u"); };
                v = fmt("uvec4(%s, %s, %s, %s)", r(0).c_str(), r(1).c_str(), r(2).c_str(), r(3).c_str());
            } else {
                v = vec();
            }
            // Shader components to memory order (CB_COLOR_INFO.COMP_SWAP).
            static const char* const kSwizzle[5][4] = {{"xyzw", "xyzw", "xyzw", "xyzw"},
                                                       {"xxxx", "yyyy", "zzzz", "wwww"},
                                                       {"xyzw", "xwzw", "yxzw", "wxzw"},
                                                       {"xyzw", "xywz", "zyxw", "wyxz"},
                                                       {"xyzw", "zyxw", "wzyx", "wxyz"}};
            const uint32_t comps = std::min<uint32_t>(p_.launch.colorComps[mrt], 4);
            const char* swz = kSwizzle[comps][p_.launch.colorSwap[mrt] & 3];
            // LPN_DEBUG_NAN=1: colour outputs holding a NaN show magenta, an infinity green.
            // LPN_DEBUG_NAN=2: magenta wherever any float result in the pixel shader was a NaN.
            static const char* debugNan = std::getenv("LPN_DEBUG_NAN");
            if (debugNan && !isU && !isS)
                v = std::string(std::strcmp(debugNan, "2") ? "nanDbg(" : "nanSeenDbg(") + v + ")";
            if (probing() && mrt == 0) {
                if (!probed_) init_ += "vec3 probe_ = vec3(1.0, 0.0, 1.0);\n";
                probed_ = true;
                v = "vec4(probe_, 1.0)";
            }
            if (std::strcmp(swz, "xyzw")) out(fmt("oC%u = %s(%s).%s;\n", mrt, type, v.c_str(), swz));
            else out(fmt("oC%u = %s(%s);\n", mrt, type, v.c_str()));
            return;
        }
        warn("export target %u", target);
    }

    // ---- memory ------------------------------------------------------------------------------

    void buffer(const Instruction& in) {
        const uint32_t w = in.words[0], w1 = in.words[1];
        size_t index = 0;
        while (index < p_.buffers.size() && p_.buffers[index].pc != pc_) ++index;
        if (index == p_.buffers.size()) {
            warn("buffer access not analysed");
            return;
        }
        const BufferResource& r = p_.buffers[index];
        const bool mtbuf = in.format == Format::MTBUF;
        const uint32_t instOffset = w & 0xFFF;
        const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
        const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF, soff = w1 >> 24;
        const std::string bn = fmt("buf%zu", index);
        const uint32_t o = r.data;

        out("{\n");
        out(fmt("uint base_ = D[%u], stride_ = D[%u], nrec_ = D[%u], size_ = D[%u];\n", o, o + 1, o + 2, o + 3));
        out("uint idx_ = " + (idxen ? vreg(vaddr) : std::string("0u")) + ";\n");
        out("uint off_ = " + (offen ? vreg(vaddr + (idxen ? 1 : 0)) : std::string("0u")) + " + " + src(soff) +
            fmt(" + %uu;\n", instOffset));
        out("uint a_ = base_ + idx_ * stride_ + off_;\n");
        out("bool in_ = (stride_ != 0u ? idx_ < nrec_ : off_ < nrec_) && a_ < size_;\n");

        enum Kind { kFormat, kDwords, kU8, kS8, kU16, kS16 } kind = kFormat;
        unsigned count = 0;
        bool store = false;
        if (mtbuf) {
            count = (in.op & 3) + 1;
            store = in.op >= 4;
        } else if (in.op <= 7) {
            count = (in.op & 3) + 1;
            store = in.op >= 4;
        } else if (in.op >= 8 && in.op <= 11) {
            kind = Kind(kU8 + (in.op - 8));
            count = 1;
        } else if (in.op >= 12 && in.op <= 15) {
            kind = kDwords;
            static const unsigned n[] = {1, 2, 4, 3};
            count = n[in.op - 12];
        } else if (in.op == 24 || in.op == 26) {
            kind = in.op == 24 ? kU8 : kU16;
            count = 1;
            store = true;
        } else if (in.op >= 28 && in.op <= 31) {
            kind = kDwords;
            static const unsigned n[] = {1, 2, 4, 3};
            count = n[in.op - 28];
            store = true;
        } else {
            warn("%s", mnemonic(in).c_str());
            out("}\n");
            return;
        }

        if (store) {
            out("if (in_" + std::string(exec_ ? " && (exec_lo & 1u) != 0u" : "") + ") {\n");
            if (kind == kDwords) {
                for (unsigned i = 0; i < count; ++i) out(fmt("%s.m[(a_ >> 2) + %uu] = %s;\n", bn.c_str(), i, vreg(vdata + i).c_str()));
            } else if (kind == kU8 || kind == kU16) {
                const char* mask = kind == kU8 ? "0xffu" : "0xffffu";
                const char* shift = kind == kU8 ? "(a_ & 3u) * 8u" : "(a_ & 2u) * 8u";
                out(fmt("atomicAnd(%s.m[a_ >> 2], ~(%s << (%s)));\n", bn.c_str(), mask, shift));
                out(fmt("atomicOr(%s.m[a_ >> 2], (%s & %s) << (%s));\n", bn.c_str(), vreg(vdata).c_str(), mask, shift));
            } else {
                storeFormat(r, bn, vdata, count);
            }
            out("}\n}\n");
            return;
        }
        // Loads.
        out("uint r_[4] = uint[4](0u, 0u, 0u, 0u);\n");
        out("if (in_) {\n");
        switch (kind) {
        case kDwords:
            for (unsigned i = 0; i < count; ++i) out(fmt("r_[%u] = %s.m[(a_ >> 2) + %uu];\n", i, bn.c_str(), i));
            break;
        case kU8: out(fmt("r_[0] = (%s.m[a_ >> 2] >> ((a_ & 3u) * 8u)) & 0xffu;\n", bn.c_str())); break;
        case kS8: out(fmt("r_[0] = uint(bitfieldExtract(int(%s.m[a_ >> 2]), int((a_ & 3u) * 8u), 8));\n", bn.c_str())); break;
        case kU16: out(fmt("r_[0] = (%s.m[a_ >> 2] >> ((a_ & 2u) * 8u)) & 0xffffu;\n", bn.c_str())); break;
        case kS16: out(fmt("r_[0] = uint(bitfieldExtract(int(%s.m[a_ >> 2]), int((a_ & 2u) * 8u), 16));\n", bn.c_str())); break;
        case kFormat: loadFormat(r, bn); break;
        }
        out("}\n");
        for (unsigned i = 0; i < count; ++i) writeV(vdata + i, fmt("r_[%u]", i));
        out("}\n");
    }

    // Formatted element load into r_[0..3], with the V# swizzle applied.
    void loadFormat(const BufferResource& r, const std::string& bn) {
        const uint32_t dfmt = r.dfmt, nfmt = r.nfmt;
        const unsigned comps = formatComponents(dfmt);
        const bool isInt = nfmt == 4 || nfmt == 5;
        out("uint e_[4] = uint[4](0u, 0u, 0u, " + std::string(isInt ? "1u" : "0x3f800000u") + ");\n");
        auto conv = [&](const std::string& raw, unsigned bits) -> std::string {
            const std::string maxv = bits == 8 ? "255.0" : bits == 16 ? "65535.0" : "1023.0";
            const std::string half = bits == 8 ? "127.0" : bits == 16 ? "32767.0" : "511.0";
            switch (nfmt) {
            case 0: return "U(float(" + raw + ") / " + maxv + ")";
            case 1: return fmt("U(max(float(bitfieldExtract(int(%s), 0, %u)) / %s, -1.0))", raw.c_str(), bits, half.c_str());
            case 2: return "U(float(" + raw + "))";
            case 3: return fmt("U(float(bitfieldExtract(int(%s), 0, %u)))", raw.c_str(), bits);
            case 5: return fmt("uint(bitfieldExtract(int(%s), 0, %u))", raw.c_str(), bits);
            case 7: return bits == 16 ? "U(unpackHalf2x16(" + raw + ").x)" : raw;
            default: return raw;
            }
        };
        switch (dfmt) {
        case 4: case 11: case 13: case 14:
            for (unsigned i = 0; i < comps; ++i) out(fmt("e_[%u] = %s.m[(a_ >> 2) + %uu];\n", i, bn.c_str(), i));
            break;
        case 2: case 5: case 12:
            for (unsigned i = 0; i < comps; ++i)
                out(fmt("e_[%u] = %s;\n", i, conv(fmt("((%s.m[(a_ >> 2) + %uu] >> %uu) & 0xffffu)", bn.c_str(), i / 2, (i & 1) * 16), 16).c_str()));
            break;
        case 1: case 3: case 10:
            for (unsigned i = 0; i < comps; ++i)
                out(fmt("e_[%u] = %s;\n", i, conv(fmt("((%s.m[a_ >> 2] >> ((a_ & 3u) * 8u + %uu)) & 0xffu)", bn.c_str(), i * 8), 8).c_str()));
            break;
        case 8: case 9: { // 10_10_10_2 / 2_10_10_10: x in the low bits
            out(fmt("uint p_ = %s.m[a_ >> 2];\n", bn.c_str()));
            for (unsigned i = 0; i < 3; ++i) out(fmt("e_[%u] = %s;\n", i, conv(fmt("((p_ >> %uu) & 0x3ffu)", i * 10), 10).c_str()));
            out(nfmt == 0 ? "e_[3] = U(float(p_ >> 30) / 3.0);\n" : "e_[3] = p_ >> 30;\n");
            break;
        }
        default: warn("buffer data format %u", dfmt); break;
        }
        const uint32_t sel = r.sharp[3] & 0xFFF;
        for (int i = 0; i < 4; ++i) {
            const uint32_t s = (sel >> (i * 3)) & 7;
            if (s >= 4) out(fmt("r_[%d] = e_[%u];\n", i, s - 4));
            else if (s == 1) out(fmt("r_[%d] = %s;\n", i, isInt ? "1u" : "0x3f800000u"));
            else if (s == 0 && sel != 0) out(fmt("r_[%d] = 0u;\n", i));
            else if (sel == 0) out(fmt("r_[%d] = e_[%d];\n", i, i));
        }
    }

    void storeFormat(const BufferResource& r, const std::string& bn, uint32_t vdata, unsigned count) {
        const unsigned comps = formatComponents(r.dfmt);
        switch (r.dfmt) {
        case 4: case 11: case 13: case 14:
            for (unsigned i = 0; i < comps && i < count; ++i)
                out(fmt("%s.m[(a_ >> 2) + %uu] = %s;\n", bn.c_str(), i, vreg(vdata + i).c_str()));
            break;
        case 10:
            if (r.nfmt == 0) {
                std::string v = "vec4(";
                for (unsigned i = 0; i < 4; ++i) v += (i ? ", " : "") + (i < count ? "F(" + vreg(vdata + i) + ")" : std::string("0.0"));
                out(fmt("%s.m[a_ >> 2] = packUnorm4x8(%s));\n", bn.c_str(), v.c_str()));
                break;
            }
            [[fallthrough]];
        default: warn("buffer store format %u/%u", r.dfmt, r.nfmt); break;
        }
    }

    void image(const Instruction& in) {
        const uint32_t w = in.words[0], w1 = in.words[1];
        size_t index = 0;
        while (index < p_.images.size() && p_.images[index].pc != pc_) ++index;
        if (index == p_.images.size()) {
            warn("image access not analysed");
            return;
        }
        const ImageResource& im = p_.images[index];
        const uint32_t dmask = (w >> 8) & 0xF;
        const bool unorm = ((w >> 12) & 1) || ((im.s[0] >> 15) & 1);
        const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
        const std::string tn = fmt("img%zu", index);
        const uint32_t type = im.type();
        unsigned dims = 2;
        bool array = false;
        switch (type) {
        case 8: dims = 1; break;
        case 10: case 11: dims = 3; break;
        case 12: dims = 1; array = true; break;
        case 13: dims = 2; array = true; break;
        default: break;
        }
        const unsigned coords = dims + (array ? 1 : 0); // address registers
        const bool oneD = dims == 1;                     // declared 2D: y is row 0
        const char pfx = typePrefix(im);
        const bool isInt = pfx != ' ';
        auto fv = [&](uint32_t r) { return "F(" + vreg(r) + ")"; };
        auto iv = [&](uint32_t r) { return "int(" + vreg(r) + ")"; };
        auto vecOf = [&](uint32_t first, unsigned n, bool asInt) {
            if (n == 1) return asInt ? iv(first) : fv(first);
            std::string s = std::string(asInt ? "ivec" : "vec") + char('0' + n) + "(";
            for (unsigned i = 0; i < n; ++i) s += (i ? ", " : "") + (asInt ? iv(first + i) : fv(first + i));
            return s + ")";
        };
        auto writeResult = [&](const std::string& vec4expr) {
            out("{ " + std::string(isInt ? "uvec4" : "vec4") + " t_ = " + vec4expr + ";\n");
            uint32_t r = vdata;
            for (int c = 0; c < 4; ++c)
                if ((dmask >> c) & 1) writeV(r++, isInt ? fmt("t_[%d]", c) : fmt("U(t_[%d])", c));
            out("}\n");
        };
        // Integer texel coordinates in the GLSL image's dimensions.
        auto texel = [&](uint32_t first) {
            if (!oneD) return vecOf(first, coords, true);
            return array ? "ivec3(" + iv(first) + ", 0, " + iv(first + 1) + ")" : "ivec2(" + iv(first) + ", 0)";
        };
        const std::string cast = isInt ? "uvec4" : "vec4";
        // The T#'s DST_SEL swizzles stores (inversely) as well as loads. Storage image views cannot
        // swizzle, so it is applied here; a store with BGRA DST_SEL into a BGRA-swapped render
        // target keeps memory in BGRA order, which a copy shader relies on.
        const uint32_t sel = im.dstSel();
        uint32_t used = 0; // memory components DST_SEL names; only a full permutation is applied
        for (int c = 0; c < 4; ++c)
            if (((sel >> (3 * c)) & 7) >= 4) used |= 1u << (((sel >> (3 * c)) & 7) - 4);
        const bool swizzled = im.storage && sel != 07654 && used == 0xF;

        if (in.op <= 1) { // image_load, image_load_mip
            const std::string lod = in.op == 1 ? iv(vaddr + coords) : "0";
            if (swizzled) { // shader component c reads memory component DST_SEL[c]
                out("{ " + cast + " m_ = " + cast + "(imageLoad(" + tn + ", " + texel(vaddr) + "));\n");
                std::string r = cast + "(";
                for (int c = 0; c < 4; ++c) {
                    const uint32_t s = (sel >> (3 * c)) & 7;
                    r += c ? ", " : "";
                    r += s >= 4 ? fmt("m_[%u]", s - 4) : s == 1 ? (isInt ? "1u" : "1.0") : (isInt ? "0u" : "0.0");
                }
                writeResult(r + ")");
                out("}\n");
            } else if (im.storage) writeResult(cast + "(imageLoad(" + tn + ", " + texel(vaddr) + "))");
            else writeResult(cast + "(texelFetch(" + tn + ", " + texel(vaddr) + ", " + lod + "))");
            return;
        }
        if (in.op == 8 || in.op == 9) { // image_store, image_store_mip
            std::string v = isInt ? "uvec4(" : "vec4(";
            uint32_t r = vdata;
            for (int c = 0; c < 4; ++c) {
                v += c ? ", " : "";
                if ((dmask >> c) & 1) v += isInt ? vreg(r++) : fv(r++);
                else v += isInt ? "0u" : "0.0";
            }
            v += ")";
            if (swizzled) { // memory component j takes the shader component whose DST_SEL is j
                out("{ " + cast + " s_ = " + v + ";\n");
                v = cast + "(";
                for (uint32_t j = 0; j < 4; ++j) {
                    int from = -1;
                    for (int c = 0; c < 4; ++c)
                        if (((sel >> (3 * c)) & 7) == 4 + j) from = c;
                    v += j ? ", " : "";
                    v += from >= 0 ? fmt("s_[%d]", from) : (isInt ? "0u" : "0.0");
                }
                v += ")";
            }
            if (pfx == 'i') v = "ivec4(" + v + ")";
            out(guard() + "imageStore(" + tn + ", " + texel(vaddr) + ", " + v + ");\n");
            if (swizzled) out("}\n");
            return;
        }
        if (in.op == 14) { // image_get_resinfo
            std::string size = "textureSize(" + tn + ", " + iv(vaddr) + ")";
            std::string v;
            if (oneD && array) v = "uvec4(uint(" + size + ".x), uint(" + size + ".z), 1u, uint(textureQueryLevels(" + tn + ")))";
            else if (oneD) v = "uvec4(uint(" + size + ".x), 1u, 1u, uint(textureQueryLevels(" + tn + ")))";
            else if (coords == 2) v = "uvec4(uvec2(" + size + "), 1u, uint(textureQueryLevels(" + tn + ")))";
            else v = "uvec4(uvec3(" + size + "), uint(textureQueryLevels(" + tn + ")))";
            out("{ uvec4 t_ = " + v + ";\n");
            uint32_t r = vdata;
            for (int c = 0; c < 4; ++c)
                if ((dmask >> c) & 1) writeV(r++, fmt("t_[%d]", c));
            out("}\n");
            return;
        }
        const bool sample = in.op >= 32 && in.op < 64, gather = in.op >= 64 && in.op < 96;
        if (!sample && !gather) {
            warn("%s", mnemonic(in).c_str());
            return;
        }
        // Address layout: [offset] [bias] [z-compare] [derivatives] coords [lod/clamp].
        const uint32_t variant = in.op & 15;
        bool hasOffset, bias, compare, deriv = false, lod = false, lz = false, clampLod = false;
        if (sample) {
            hasOffset = in.op >= 48;
            static const uint8_t flags[16] = {0, 8, 2, 10, 4, 1, 9, 16, 32, 40, 34, 42, 36, 33, 41, 48};
            const uint8_t f = flags[variant];
            bias = f & 1;
            deriv = f & 2;
            lod = f & 4;
            clampLod = f & 8;
            lz = f & 16;
            compare = f & 32;
        } else {
            hasOffset = in.op >= 80;
            static const uint8_t flags[12] = {0, 8, 4, 1, 9, 16, 32, 40, 36, 33, 41, 48};
            const uint8_t f = variant < 12 ? flags[variant] : 0;
            bias = f & 1;
            lod = f & 4;
            clampLod = f & 8;
            lz = f & 16;
            compare = f & 32;
        }
        (void)clampLod;
        uint32_t r = vaddr;
        const uint32_t offReg = hasOffset ? r++ : 0;
        const uint32_t biasReg = bias ? r++ : 0;
        const uint32_t zReg = compare ? r++ : 0;
        const uint32_t dReg = r;
        if (deriv) r += dims * 2;
        const uint32_t cReg = r;
        r += coords;
        const uint32_t lodReg = r;

        out("{\n");
        const unsigned glslCoords = oneD ? coords + 1 : coords;
        out("vec" + std::to_string(glslCoords) + " uv_ = " +
            (!oneD ? vecOf(cReg, coords, false)
                   : array ? "vec3(" + fv(cReg) + ", 0.0, " + fv(cReg + 1) + ")" : "vec2(" + fv(cReg) + ", 0.0)") +
            ";\n");
        const std::string sizeExpr = dims == 3 ? "vec3(textureSize(" + tn + ", 0))" : "vec2(textureSize(" + tn + ", 0).xy)";
        if (unorm) out("uv_.x" + std::string(oneD ? "" : "y") + " /= " + sizeExpr + ".x" + (oneD ? "" : "y") + ";\n");
        else if (oneD) out(fmt("uv_.x *= F(D[%u]);\n", im.data));
        else if (dims == 2) out(fmt("uv_.xy *= vec2(F(D[%u]), F(D[%u]));\n", im.data, im.data + 1));
        if (hasOffset) {
            out(fmt("{ uint o_ = %s; vec2 oo_ = vec2(float(bitfieldExtract(int(o_), 0, 6)), float(bitfieldExtract(int(o_), 8, 6)));\n",
                    vreg(offReg).c_str()));
            out("uv_.xy += oo_ / " + sizeExpr + ".xy; }\n");
        }
        if (oneD) out("uv_.y = 0.5 / " + sizeExpr + ".y;\n"); // the middle of row 0
        if (type == 11) out("uv_ = cubeDir(uv_);\n"); // samplerCube wants a direction
        const std::string uv = "uv_";
        auto grad = [&](uint32_t first) { return oneD ? "vec2(" + fv(first) + ", 0.0)" : vecOf(first, dims, false); };
        std::string call;
        const bool fragment = p_.launch.stage == Stage::Pixel;
        if (compare) {
            std::string uvz = glslCoords == 2 ? "vec3(uv_, " + fv(zReg) + ")" : "vec4(uv_, " + fv(zReg) + ")";
            if (gather) call = "textureGather(" + tn + ", " + uv + ", " + fv(zReg) + ")";
            else if (lod || lz || !fragment) call = "vec4(textureLod(" + tn + ", " + uvz + ", " + (lod ? fv(lodReg) : "0.0") + "), 0.0, 0.0, 0.0)";
            else if (deriv) call = "vec4(textureGrad(" + tn + ", " + uvz + ", " + grad(dReg) + ", " + grad(dReg + dims) + "), 0.0, 0.0, 0.0)";
            else call = "vec4(texture(" + tn + ", " + uvz + (bias ? ", " + fv(biasReg) : "") + "), 0.0, 0.0, 0.0)";
        } else if (gather) {
            int compIdx = 0;
            while (compIdx < 3 && !((dmask >> compIdx) & 1)) ++compIdx;
            call = cast + "(textureGather(" + tn + ", " + uv + ", " + std::to_string(compIdx) + "))";
        } else if (lod || lz || !fragment) {
            call = cast + "(textureLod(" + tn + ", " + uv + ", " + (lod ? fv(lodReg) : "0.0") + "))";
        } else if (deriv) {
            call = cast + "(textureGrad(" + tn + ", " + uv + ", " + grad(dReg) + ", " + grad(dReg + dims) + "))";
        } else {
            call = cast + "(texture(" + tn + ", " + uv + (bias ? ", " + fv(biasReg) : "") + "))";
        }
        if (gather) {
            // Gather returns four texels of one component, in vdata order.
            out("{ " + std::string(isInt ? "uvec4" : "vec4") + " t_ = " + call + ";\n");
            for (int c = 0; c < 4; ++c) writeV(vdata + c, isInt ? fmt("t_[%d]", c) : fmt("U(t_[%d])", c));
            out("}\n}\n");
            return;
        }
        writeResult(call);
        out("}\n");
    }

    const Program& p_;
    uint32_t base_;
    Translation t_;
    std::string decls_, init_, body_;
    std::set<std::string> regs_;
    uint64_t pc_ = 0, next_ = 0;
    uint32_t literal_ = 0;
    bool exec_ = false;
    bool needLane_ = false, need64_ = false, needQuad_ = false, depthDeclared_ = false;
};

} // namespace

Translation translate(const Program& program, uint32_t bindingBase) {
    Translation t = Emitter(program, bindingBase).run();
    return t;
}

} // namespace gcn
