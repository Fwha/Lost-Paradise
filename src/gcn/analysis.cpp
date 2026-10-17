// Scalar evaluation of a GCN program at launch: discovers the reachable code (following
// branches, fetch-shader calls and returns) and resolves scalar loads and descriptors.
#include "gcn/shader.h"

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <deque>

namespace gcn {

namespace {

constexpr uint32_t kVcc = 106, kM0 = 124, kExec = 126;

struct State {
    uint32_t s[128] = {};
    bool known[128] = {};
    bool scc = false, sccKnown = false;
};

struct Work {
    uint64_t pc;
    State st;
};

class Analyzer {
public:
    Analyzer(const LaunchState& l, const ReadMemory& code, const ReadMemory& data, const Program* prior = nullptr)
        : readCode_(code), readData_(data), prior_(prior) {
        p_.launch = l;
    }

    Program run() {
        p_.data.assign(data::kSmrd, 0);
        std::deque<Work> work{{p_.launch.entry, start()}};
        p_.labels.insert(p_.launch.entry);
        unsigned steps = 0;
        while (!work.empty() && steps < 100000) {
            Work w = std::move(work.front());
            work.pop_front();
            p_.trace.push_back({w.pc, {}, true, false});
            while (++steps < 100000) {
                if (p_.code.count(w.pc)) break; // already walked from here
                // Read only the instruction's own bytes: the page after a shader's last
                // instruction may hold data that changes, and the analysis depends on what it reads.
                uint32_t words[3] = {};
                if (!readCode_(w.pc, words, 4)) {
                    warn("unreadable code at 0x%llx", (unsigned long long)w.pc);
                    break;
                }
                Instruction in = decode(words, 1);
                if (in.size > 4 && !readCode_(w.pc, words, in.size)) {
                    warn("unreadable code at 0x%llx", (unsigned long long)w.pc);
                    break;
                }
                in = decode(words, in.size / 4);
                p_.code[w.pc] = in;
                const uint64_t next = w.pc + in.size;
                const bool more = step(in, w.pc, w.st, next, work);
                if (!more || !inert(in)) p_.trace.push_back({w.pc, in, false, more});
                if (!more) break;
                w.pc = next;
            }
        }
        if (steps >= 100000) {
            warn("analysis did not terminate");
            p_.trace.clear();
        }
        return std::move(p_);
    }

    // Re-evaluates prior_'s trace for this launch; false when the walk would differ.
    bool replay() {
        const Program& prior = *prior_;
        p_.data.assign(prior.data.size(), 0);
        p_.buffers = prior.buffers;
        p_.images = prior.images;
        seenBuffers_.assign(p_.buffers.size(), false);
        seenImages_.assign(p_.images.size(), false);
        std::deque<Work> work{{p_.launch.entry, start()}};
        State st;
        for (const Program::TraceStep& t : prior.trace) {
            if (t.start) {
                if (work.empty() || work.front().pc != t.pc) return false;
                st = std::move(work.front().st);
                work.pop_front();
            } else if (step(t.in, t.pc, st, t.pc + t.in.size, work) != t.result || diverged_) {
                return false;
            }
        }
        return work.empty();
    }

    Program& program() { return p_; }

private:
    // The state at the entry point; also fills in the launch's part of the draw data.
    State start() {
        std::memcpy(p_.data.data(), p_.launch.userData, sizeof(p_.launch.userData));
        const float one = 1.0f;
        std::memcpy(&p_.data[data::kPosTransform], &one, 4);
        std::memcpy(&p_.data[data::kPosTransform + 1], &one, 4);
        State st;
        const uint32_t user = p_.launch.userSgprs();
        for (uint32_t i = 0; i < user && i < 16; ++i) {
            st.s[i] = p_.launch.userData[i];
            st.known[i] = true;
        }
        return st;
    }

    template <typename... A> void warn(const char* fmt, A... a) {
        if (prior_) return; // a replay reports nothing new
        char b[256];
        std::snprintf(b, sizeof b, fmt, a...);
        p_.warnings.emplace_back(b);
    }

    // Value of a scalar source operand, if known.
    bool src(const State& st, uint32_t code, const Instruction& in, uint32_t& v) {
        if (code < 104 || code == kVcc || code == kVcc + 1 || code == kM0) {
            v = st.s[code];
            return st.known[code];
        }
        if (code >= 128 && code <= 192) return v = code - 128, true;
        if (code >= 193 && code <= 208) return v = uint32_t(192 - int32_t(code)), true;
        if (code >= 240 && code <= 247) {
            static const float c[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f};
            std::memcpy(&v, &c[code - 240], 4);
            return true;
        }
        if (code == 255) return v = in.literal, true;
        if (code == 253) return v = st.scc, st.sccKnown;
        return false; // exec, vcc-derived, vector
    }
    bool src64(const State& st, uint32_t code, const Instruction& in, uint64_t& v) {
        uint32_t lo, hi = 0;
        if (code < 104 || code == kVcc) {
            if (!src(st, code, in, lo) || !src(st, code + 1, in, hi)) return false;
        } else {
            if (!src(st, code, in, lo)) return false;
            if (code >= 193 && code <= 208) hi = 0xFFFFFFFF;
        }
        v = lo | (uint64_t(hi) << 32);
        return true;
    }
    void set(State& st, uint32_t d, uint32_t v, bool known = true) {
        if (d < 128) {
            st.s[d] = v;
            st.known[d] = known;
        }
        if (d == kExec || d == kExec + 1) p_.writesExec = true;
    }
    void set64(State& st, uint32_t d, uint64_t v, bool known = true) {
        set(st, d, uint32_t(v), known);
        set(st, d + 1, uint32_t(v >> 32), known);
    }
    void unknown(State& st, uint32_t d, unsigned n = 1) {
        for (unsigned i = 0; i < n; ++i) set(st, d + i, 0, false);
    }

    void branchTo(uint64_t target, const State& st, std::deque<Work>& work) {
        label(target);
        p_.controlFlow = true;
        work.push_back({target, st});
    }
    void label(uint64_t pc) {
        if (!prior_) p_.labels.insert(pc); // a replay keeps the prior labels
    }

    // Instructions step() does nothing for: a replay can skip them.
    static bool inert(const Instruction& in) {
        switch (in.format) {
        case Format::VOP1: return in.op != 2;
        case Format::VOP2: return in.op != 1 && !(in.op >= 37 && in.op <= 42);
        case Format::EXP:
        case Format::VINTRP:
        case Format::DS: return true;
        default: return false;
        }
    }

    // Returns false when this path ends.
    bool step(const Instruction& in, uint64_t pc, State& st, uint64_t next, std::deque<Work>& work) {
        const uint32_t w = in.words[0];
        switch (in.format) {
        case Format::SOPP: {
            const uint64_t target = pc + 4 + int64_t(int16_t(w & 0xFFFF)) * 4;
            switch (in.op) {
            case 1: return false; // s_endpgm
            case 2: branchTo(target, st, work); return false;
            case 4: case 5: // s_cbranch_scc0/1
                if (st.sccKnown) {
                    if (st.scc == (in.op == 5)) {
                        branchTo(target, st, work);
                        return false;
                    }
                    return true;
                }
                [[fallthrough]];
            case 6: case 7: case 8: case 9: // vccz/vccnz/execz/execnz: lane-dependent
                branchTo(target, st, work);
                label(next);
                return true;
            default: return true;
            }
        }
        case Format::SOP1: return sop1(in, pc, st, next, work);
        case Format::SOP2: sop2(in, st); return true;
        case Format::SOPC: {
            uint32_t a = 0, b = 0;
            const bool k = src(st, w & 0xFF, in, a) & src(st, (w >> 8) & 0xFF, in, b);
            st.sccKnown = k;
            if (k) st.scc = sopc(in.op, a, b);
            return true;
        }
        case Format::SOPK: {
            const uint32_t d = (w >> 16) & 0x7F;
            const uint32_t imm = w & 0xFFFF;
            const uint32_t simm = uint32_t(int32_t(int16_t(imm)));
            uint32_t cur;
            const bool k = src(st, d, in, cur);
            switch (in.op) {
            case 0: set(st, d, simm); break;
            case 15: if (k) set(st, d, cur + simm); else unknown(st, d); st.sccKnown = false; break;
            case 16: if (k) set(st, d, cur * simm); else unknown(st, d); break;
            default:
                if (in.op >= 3 && in.op <= 14) {
                    st.sccKnown = k;
                    if (k) st.scc = sopc(in.op - 3, cur, in.op <= 8 ? simm : imm);
                } else if (in.op != 18) {
                    unknown(st, d);
                }
                if (in.op == 18) unknown(st, d);
                break;
            }
            return true;
        }
        case Format::SMRD: smrd(in, pc, st); return true;
        case Format::VOP3:
            if (in.op < 256) unknown(st, w & 0xFF, 2); // compare into an SGPR pair
            else if ((in.op >= 293 && in.op <= 298) || in.op == 365 || in.op == 366 || in.op == 374 || in.op == 375)
                unknown(st, (w >> 8) & 0x7F, 2); // VOP3b: carry or flags into an SGPR pair
            return true;
        case Format::VOPC:
            if (in.op & 16) p_.writesExec = true; // v_cmpx
            unknown(st, kVcc, 2);
            return true;
        case Format::VOP1:
            if (in.op == 2) unknown(st, (w >> 17) & 0xFF); // v_readfirstlane_b32
            return true;
        case Format::VOP2:
            if (in.op == 1) unknown(st, (w >> 17) & 0xFF); // v_readlane_b32
            if (in.op >= 37 && in.op <= 42) unknown(st, kVcc, 2);
            return true;
        case Format::MUBUF:
        case Format::MTBUF: buffer(in, pc, st); return true;
        case Format::MIMG: image(in, pc, st); return true;
        case Format::EXP:
        case Format::VINTRP:
        case Format::DS: return true;
        default:
            warn("unhandled %s at 0x%llx", mnemonic(in).c_str(), (unsigned long long)pc);
            return true;
        }
    }

    static bool sopc(uint32_t op, uint32_t a, uint32_t b) {
        const int32_t x = int32_t(a), y = int32_t(b);
        switch (op) {
        case 0: return x == y;
        case 1: return x != y;
        case 2: return x > y;
        case 3: return x >= y;
        case 4: return x < y;
        case 5: return x <= y;
        case 6: return a == b;
        case 7: return a != b;
        case 8: return a > b;
        case 9: return a >= b;
        case 10: return a < b;
        case 11: return a <= b;
        case 12: return !((a >> (b & 31)) & 1);
        case 13: return (a >> (b & 31)) & 1;
        default: return false;
        }
    }

    bool sop1(const Instruction& in, uint64_t pc, State& st, uint64_t next, std::deque<Work>& work) {
        const uint32_t w = in.words[0];
        const uint32_t d = (w >> 16) & 0x7F, s0 = w & 0xFF;
        uint32_t a;
        uint64_t a64;
        switch (in.op) {
        case 3: if (src(st, s0, in, a)) set(st, d, a); else unknown(st, d); break;
        case 4: if (src64(st, s0, in, a64)) set64(st, d, a64); else unknown(st, d, 2); break;
        case 7: if (src(st, s0, in, a)) { set(st, d, ~a); st.scc = ~a != 0; st.sccKnown = true; } else { unknown(st, d); st.sccKnown = false; } break;
        case 31: set64(st, d, next); break; // s_getpc_b64
        case 32: // s_setpc_b64
            if (src64(st, s0, in, a64)) branchTo(a64 & 0xFFFFFFFFFFFFull, st, work);
            else warn("s_setpc to unknown address at 0x%llx", (unsigned long long)pc);
            return false;
        case 33: { // s_swappc_b64
            const bool k = src64(st, s0, in, a64) && a64 != 0;
            set64(st, d, next);
            label(next);
            if (k) {
                if (!prior_) p_.calls[pc] = a64 & 0xFFFFFFFFFFFFull;
                branchTo(a64 & 0xFFFFFFFFFFFFull, st, work);
            } else {
                warn("call to unknown address at 0x%llx (fetch shader skipped)", (unsigned long long)pc);
                return true;
            }
            return false;
        }
        default:
            if (in.op >= 36 && in.op <= 43) { // s_*_saveexec_b64
                p_.writesExec = true;
                unknown(st, d, 2);
                st.sccKnown = false;
            } else {
                const bool wide = in.op == 4 || in.op == 6 || in.op == 8 || in.op == 10 || in.op == 12 ||
                                  in.op == 14 || in.op == 16 || in.op == 28 || in.op == 30 || in.op == 45;
                unknown(st, d, wide ? 2 : 1);
                st.sccKnown = false;
            }
            break;
        }
        return true;
    }

    void sop2(const Instruction& in, State& st) {
        const uint32_t w = in.words[0];
        const uint32_t d = (w >> 16) & 0x7F, s0 = w & 0xFF, s1 = (w >> 8) & 0xFF;
        const bool wide = in.op >= 15 && in.op <= 37 && (in.op & 1);
        if (wide || in.op == 11 || in.op == 41 || in.op == 42) {
            uint64_t a = 0, b = 0;
            const bool k = src64(st, s0, in, a) & src64(st, s1, in, b);
            uint64_t r = 0;
            bool ok = k;
            switch (in.op) {
            case 15: r = a & b; break;
            case 17: r = a | b; break;
            case 19: r = a ^ b; break;
            case 21: r = a & ~b; break;
            case 23: r = a | ~b; break;
            case 25: r = ~(a & b); break;
            case 27: r = ~(a | b); break;
            case 29: r = ~(a ^ b); break;
            case 31: r = a << (b & 63); break;
            case 33: r = a >> (b & 63); break;
            case 11: ok = k && st.sccKnown; r = st.scc ? a : b; break;
            default: ok = false; break;
            }
            if (ok) set64(st, d, r);
            else unknown(st, d, 2);
            st.sccKnown = ok && in.op != 11;
            st.scc = r != 0;
            return;
        }
        uint32_t a = 0, b = 0;
        const bool k = src(st, s0, in, a) & src(st, s1, in, b);
        uint32_t r = 0;
        bool ok = k, setsScc = true;
        switch (in.op) {
        case 0: r = a + b; st.scc = r < a; break;
        case 1: r = a - b; st.scc = b > a; break;
        case 2: r = a + b; st.scc = ((~(a ^ b) & (a ^ r)) >> 31) != 0; break; // signed overflow
        case 3: r = a - b; st.scc = (((a ^ b) & (a ^ r)) >> 31) != 0; break;
        case 4: { // s_addc_u32
            ok = k && st.sccKnown;
            const uint64_t r64 = uint64_t(a) + b + st.scc;
            r = uint32_t(r64);
            st.scc = r64 >> 32;
            break;
        }
        case 5: { // s_subb_u32
            ok = k && st.sccKnown;
            const uint64_t sub = uint64_t(b) + st.scc;
            r = uint32_t(a - sub);
            st.scc = sub > a;
            break;
        }
        case 6: r = int32_t(a) < int32_t(b) ? a : b; break;
        case 7: r = a < b ? a : b; break;
        case 8: r = int32_t(a) > int32_t(b) ? a : b; break;
        case 9: r = a > b ? a : b; break;
        case 10: ok = k && st.sccKnown; r = st.scc ? a : b; setsScc = false; break;
        case 14: r = a & b; break;
        case 16: r = a | b; break;
        case 18: r = a ^ b; break;
        case 20: r = a & ~b; break;
        case 22: r = a | ~b; break;
        case 26: r = ~(a | b); break;
        case 30: r = a << (b & 31); break;
        case 32: r = a >> (b & 31); break;
        case 34: r = uint32_t(int32_t(a) >> (b & 31)); break;
        case 36: r = ((1u << (a & 31)) - 1) << (b & 31); setsScc = false; break;
        case 38: r = a * b; setsScc = false; break;
        case 39: { const uint32_t off = b & 31, width = (b >> 16) & 0x7F;
                   r = width ? (a >> off) & (width >= 32 ? ~0u : (1u << width) - 1) : 0; break; }
        default: ok = false; break;
        }
        if (ok) set(st, d, r);
        else unknown(st, d);
        if (setsScc) {
            if (in.op > 5) st.scc = r != 0;
            st.sccKnown = ok;
        }
    }

    void smrd(const Instruction& in, uint64_t pc, State& st) {
        const uint32_t w = in.words[0];
        const uint32_t d = (w >> 15) & 0x7F, base = ((w >> 9) & 0x3F) * 2;
        static const unsigned kCount[] = {1, 2, 4, 8, 16};
        const unsigned op = in.op & 7;
        if (in.op >= 29 || op > 4) return; // s_memtime, cache control
        const unsigned n = kCount[op];
        const bool buf = in.op >= 8;
        uint32_t off = 0;
        bool known = true;
        if ((w >> 8) & 1) off = (w & 0xFF) * 4;
        else if ((w & 0xFF) == 255) off = in.literal;
        else known = src(st, w & 0xFF, in, off);

        uint32_t at;
        if (prior_) {
            auto pit = prior_->smrd.find(pc);
            if (pit == prior_->smrd.end()) {
                diverged_ = true;
                return;
            }
            at = pit->second;
        } else if (auto it = p_.smrd.find(pc); it == p_.smrd.end()) {
            at = uint32_t(p_.data.size());
            p_.smrd[pc] = at;
            p_.data.resize(at + n, 0);
        } else {
            at = it->second;
        }
        if (at + n > p_.data.size()) {
            diverged_ = true;
            return;
        }

        uint32_t lo, hi;
        known &= src(st, base, in, lo) & src(st, base + 1, in, hi);
        if (!known) {
            warn("scalar load from unknown address at 0x%llx", (unsigned long long)pc);
            unknown(st, d, n);
            return;
        }
        uint64_t addr;
        uint32_t limit = ~0u;
        if (buf) {
            uint32_t sharp2 = 0;
            src(st, base + 2, in, sharp2);
            addr = lo | (uint64_t(hi & 0xFFF) << 32);
            const uint32_t stride = (hi >> 16) & 0x3FFF;
            limit = stride ? uint32_t(std::min<uint64_t>(uint64_t(stride) * sharp2, ~0u)) : sharp2;
        } else {
            addr = (lo | (uint64_t(hi & 0xFFFF) << 32)) & ~3ull;
        }
        uint32_t vals[16] = {};
        unsigned inside = n; // dwords within the buffer
        if (buf) inside = off >= limit ? 0 : std::min<unsigned>(n, (limit - off) / 4);
        if (inside && !readData_(addr + off, vals, inside * 4)) { // else find the unreadable dword
            std::memset(vals, 0, sizeof vals);
            for (unsigned i = 0; i < inside; ++i)
                if (!readData_(addr + off + i * 4, &vals[i], 4)) {
                    warn("scalar load from unreadable 0x%llx", (unsigned long long)(addr + off + i * 4));
                    break;
                }
        }
        for (unsigned i = 0; i < n; ++i) {
            p_.data[at + i] = vals[i];
            set(st, d + i, vals[i]);
        }
    }

    bool sharp(const State& st, uint32_t first, unsigned n, uint32_t* out, uint64_t pc, const char* what) {
        bool ok = true;
        for (unsigned i = 0; i < n; ++i) {
            out[i] = st.s[first + i];
            ok &= st.known[first + i];
        }
        if (!ok) warn("%s descriptor at 0x%llx is not uniform-known", what, (unsigned long long)pc);
        return ok;
    }

    void buffer(const Instruction& in, uint64_t pc, State& st) {
        const uint32_t w = in.words[0], w1 = in.words[1];
        int found = -1;
        for (size_t i = 0; i < p_.buffers.size(); ++i)
            if (p_.buffers[i].pc == pc) found = int(i);
        if (prior_) { // the first visit in a replay re-resolves the prior resource
            if (found < 0) diverged_ = true;
            if (found < 0 || seenBuffers_[found]) return;
            seenBuffers_[found] = true;
        } else if (found >= 0) {
            return;
        }
        BufferResource r;
        r.pc = pc;
        r.known = sharp(st, ((w1 >> 16) & 0x1F) * 4, 4, r.sharp, pc, "buffer");
        if (in.format == Format::MTBUF) {
            r.dfmt = (w >> 19) & 0xF;
            r.nfmt = (w >> 23) & 7;
            r.store = in.op >= 4;
        } else {
            r.dfmt = (r.sharp[3] >> 15) & 0xF;
            r.nfmt = (r.sharp[3] >> 12) & 7;
            r.store = (in.op >= 4 && in.op <= 7) || (in.op >= 24 && in.op <= 31) || (in.op >= 48 && in.op < 112);
        }
        if (prior_) {
            r.data = p_.buffers[found].data;
            p_.buffers[found] = r;
            return;
        }
        r.data = uint32_t(p_.data.size());
        p_.data.resize(r.data + 4, 0);
        p_.buffers.push_back(r);
    }

    void image(const Instruction& in, uint64_t pc, State& st) {
        const uint32_t w1 = in.words[1];
        int found = -1;
        for (size_t i = 0; i < p_.images.size(); ++i)
            if (p_.images[i].pc == pc) found = int(i);
        if (prior_) {
            if (found < 0) diverged_ = true;
            if (found < 0 || seenImages_[found]) return;
            seenImages_[found] = true;
        } else if (found >= 0) {
            return;
        }
        ImageResource r;
        r.pc = pc;
        sharp(st, ((w1 >> 16) & 0x1F) * 4, 8, r.t, pc, "image");
        const bool sampled = in.op >= 32;
        if (sampled) sharp(st, ((w1 >> 21) & 0x1F) * 4, 4, r.s, pc, "sampler");
        r.storage = in.op >= 8 && in.op <= 11;
        if (in.op >= 32 && in.op < 64) r.compare = (in.op & 15) >= 8;
        if (in.op >= 64 && in.op < 96) r.compare = (in.op & 15) >= 6;
        if (prior_) {
            r.data = p_.images[found].data;
            p_.images[found] = r;
            return;
        }
        r.data = uint32_t(p_.data.size());
        p_.data.resize(r.data + 4, 0);
        p_.images.push_back(r);
    }

    const ReadMemory& readCode_;
    const ReadMemory& readData_;
    const Program* prior_; // replaying this program's trace
    bool diverged_ = false;
    std::vector<bool> seenBuffers_, seenImages_;
    Program p_;
};

} // namespace

Program analyze(const LaunchState& launch, const ReadMemory& readCode, const ReadMemory& readData) {
    return Analyzer(launch, readCode, readData).run();
}

bool replay(const Program& p, const LaunchState& launch, const ReadMemory& readData, Program& out) {
    if (p.trace.empty()) return false;
    Analyzer a(launch, readData, readData, &p);
    if (!a.replay()) return false;
    out = std::move(a.program());
    return true;
}

bool adopt(Program& p, Program& r) {
    if (p.buffers.size() != r.buffers.size() || p.images.size() != r.images.size() || p.data.size() != r.data.size())
        return false;
    // The translation depends on these (see Program::key).
    for (size_t i = 0; i < p.buffers.size(); ++i) {
        const BufferResource &x = p.buffers[i], &y = r.buffers[i];
        if (x.dfmt != y.dfmt || x.nfmt != y.nfmt || (x.sharp[3] & 0xFFF) != (y.sharp[3] & 0xFFF) || x.known != y.known)
            return false;
    }
    for (size_t i = 0; i < p.images.size(); ++i) {
        const ImageResource &x = p.images[i], &y = r.images[i];
        if (x.type() != y.type() || x.dataFormat() != y.dataFormat() || x.numFormat() != y.numFormat() ||
            x.dstSel() != y.dstSel())
            return false;
    }
    p.launch = r.launch;
    p.data = std::move(r.data);
    p.buffers = std::move(r.buffers);
    p.images = std::move(r.images);
    return true;
}

bool refresh(Program& p, const LaunchState& launch, const ReadMemory& readData) {
    Program r;
    return replay(p, launch, readData, r) && adopt(p, r);
}

std::string Program::key() const {
    std::string k;
    auto put = [&](const void* p, size_t n) { k.append(static_cast<const char*>(p), n); };
    const LaunchState& l = launch;
    put(&l.stage, 1);
    put(&l.entry, 8);
    put(&l.rsrc1, 4);
    put(&l.rsrc2, 4);
    put(&l.glClipZ, 1);
    if (l.stage == Stage::Pixel) {
        put(&l.psInputEna, 4);
        put(l.psInputCntl, sizeof(l.psInputCntl));
        put(&l.colorFormat, 4);
        put(&l.colorUint, 4);
        put(&l.colorSint, 4);
        put(l.colorSwap, sizeof(l.colorSwap));
        put(l.colorComps, sizeof(l.colorComps));
        put(&l.depthExport, 1);
    }
    if (l.stage == Stage::Compute) put(l.localSize, sizeof(l.localSize));
    for (const auto& [pc, in] : code) {
        put(&pc, 8);
        put(in.words, in.size);
    }
    for (const auto& b : buffers) {
        put(&b.dfmt, 4);
        put(&b.nfmt, 4);
        const uint32_t sel = b.sharp[3] & 0xFFF;
        put(&sel, 4);
    }
    for (const auto& i : images) {
        const uint32_t shape[] = {i.type(), i.dataFormat(), i.numFormat(), i.dstSel(), i.compare, i.storage};
        put(shape, sizeof(shape));
    }
    return k;
}

} // namespace gcn
