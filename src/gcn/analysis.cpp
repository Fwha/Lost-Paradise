// Scalar evaluation of a GCN program at launch: discovers the reachable code (following
// branches, fetch-shader calls and returns) and resolves scalar loads and descriptors.
#include "gcn/shader.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <type_traits>

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

// Work items in order. A deque allocates per item here (they are large), and a replay runs for
// every draw: a vector with a read position, kept per thread (one queue runs at a time), does not
// allocate at all once it has grown.
class WorkQueue {
public:
    explicit WorkQueue(Work first) : items_(storage()) {
        items_.clear();
        items_.push_back(std::move(first));
    }
    bool empty() const { return head_ == items_.size(); }
    Work& front() { return items_[head_]; }
    void pop_front() { ++head_; }
    void push_back(Work w) { items_.push_back(std::move(w)); }

private:
    static std::vector<Work>& storage() {
        thread_local std::vector<Work> v;
        return v;
    }
    std::vector<Work>& items_;
    size_t head_ = 0;
};

class Analyzer {
public:
    Analyzer(const LaunchState& l, const ReadMemory& code, const ReadMemory& data, const Program* prior = nullptr)
        : readCode_(code), readData_(data), prior_(prior) {
        p_.launch = l;
    }

    Program run() {
        p_.data.assign(data::kSmrd, 0);
        p_.labels.insert(p_.launch.entry);
        unsigned steps = 0;
        {
            WorkQueue work{Work{p_.launch.entry, start()}};
            walk(work, steps);
        }
        // Generic: the scalar branches this launch decided are kept as branches, and the code
        // they skip is walked too (with nothing known, so its resources and loads are left
        // unresolved; this launch never runs them). Every launch then finds the same code, and
        // the slots are numbered by address, so launches that differ only in such flags (a
        // vertex shader turning blocks on with bits of a constant) share one translation.
        if (generic() && steps < 100000) {
            speculative_ = true;
            if (!untaken_.empty()) {
                WorkQueue work{Work{untaken_[0], State{}}};
                for (size_t i = 1; i < untaken_.size(); ++i) work.push_back({untaken_[i], State{}});
                walk(work, steps);
            }
            speculative_ = false;
        }
        if (steps >= 100000) {
            warn("analysis did not terminate");
            p_.trace.clear();
        }
        if (generic()) canonicalize();
        return std::move(p_);
    }

    // LPN_GENERIC_SHADERS=0: translations specialised to the scalar branches each launch takes.
    static bool generic() { return genericTranslation(); }

    void walk(WorkQueue& work, unsigned& steps) {
        while (!work.empty() && steps < 100000) {
            Work w = std::move(work.front());
            work.pop_front();
            if (!speculative_) p_.trace.push_back({w.pc, {}, true, false});
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
                slot_ = ~0u;
                const bool more = step(in, w.pc, w.st, next, work);
                if (!speculative_ && (!more || !inert(in))) p_.trace.push_back({w.pc, in, false, more, slot_});
                if (!more) break;
                w.pc = next;
            }
        }
    }

    // Numbers the scalar-load slots and resources by address rather than by the order this
    // launch's walk met them, so the translation does not depend on which way it went.
    void canonicalize() {
        std::map<uint32_t, uint32_t> slotMap; // old data offset -> new
        std::vector<uint32_t> data(p_.data.begin(), p_.data.begin() + std::min<size_t>(p_.data.size(), data::kSmrd));
        data.resize(data::kSmrd, 0);
        auto move = [&](uint32_t from, unsigned n) {
            const uint32_t to = uint32_t(data.size());
            for (unsigned i = 0; i < n; ++i) data.push_back(from + i < p_.data.size() ? p_.data[from + i] : 0);
            slotMap[from] = to;
            return to;
        };
        static const unsigned kCount[] = {1, 2, 4, 8, 16};
        for (auto& [pc, at] : p_.smrd) { // by address
            auto it = p_.code.find(pc);
            const unsigned n = it != p_.code.end() && (it->second.op & 7) <= 4 ? kCount[it->second.op & 7] : 1;
            at = move(at, n);
        }
        std::vector<uint32_t> bufOrder(p_.buffers.size()), imgOrder(p_.images.size());
        for (uint32_t i = 0; i < bufOrder.size(); ++i) bufOrder[i] = i;
        for (uint32_t i = 0; i < imgOrder.size(); ++i) imgOrder[i] = i;
        std::stable_sort(bufOrder.begin(), bufOrder.end(), [&](uint32_t a, uint32_t b) { return p_.buffers[a].pc < p_.buffers[b].pc; });
        std::stable_sort(imgOrder.begin(), imgOrder.end(), [&](uint32_t a, uint32_t b) { return p_.images[a].pc < p_.images[b].pc; });
        std::vector<uint32_t> bufIndex(bufOrder.size()), imgIndex(imgOrder.size()); // old -> new
        std::vector<BufferResource> buffers;
        std::vector<ImageResource> images;
        for (uint32_t i = 0; i < bufOrder.size(); ++i) {
            bufIndex[bufOrder[i]] = i;
            buffers.push_back(p_.buffers[bufOrder[i]]);
            buffers.back().data = move(buffers.back().data, 4);
        }
        for (uint32_t i = 0; i < imgOrder.size(); ++i) {
            imgIndex[imgOrder[i]] = i;
            images.push_back(p_.images[imgOrder[i]]);
            images.back().data = move(images.back().data, 4);
        }
        for (auto& t : p_.trace) {
            if (t.start || t.slot == ~0u) continue;
            switch (t.in.format) {
            case Format::SMRD:
                if (auto it = slotMap.find(t.slot); it != slotMap.end()) t.slot = it->second;
                break;
            case Format::MUBUF: case Format::MTBUF:
                if (t.slot < bufIndex.size()) t.slot = bufIndex[t.slot];
                break;
            case Format::MIMG:
                if (t.slot < imgIndex.size()) t.slot = imgIndex[t.slot];
                break;
            default: break;
            }
        }
        p_.data = std::move(data);
        p_.buffers = std::move(buffers);
        p_.images = std::move(images);
    }

    // Re-evaluates prior_'s trace for this launch; false when the walk would differ. With a plan,
    // only the steps it keeps (see ReplayPlan).
    bool replay(const ReplayPlan* plan = nullptr);
    bool replayAll() {
        const Program& prior = *prior_;
        p_.data.assign(prior.data.size(), 0);
        p_.buffers = prior.buffers;
        p_.images = prior.images;
        seenBuffers_.assign(p_.buffers.size(), false);
        seenImages_.assign(p_.images.size(), false);
        WorkQueue work{Work{p_.launch.entry, start()}};
        State st;
        for (const Program::TraceStep& t : prior.trace) {
            if (t.start) {
                if (work.empty() || work.front().pc != t.pc) return false;
                st = std::move(work.front().st);
                work.pop_front();
            } else if ((slot_ = t.slot, step(t.in, t.pc, st, t.pc + t.in.size, work) != t.result) || diverged_) {
                return false;
            }
        }
        return work.empty();
    }
    bool replayOps(const ReplayPlan& plan);

    Program& program() { return p_; }

    // Where a scalar operand comes from, worked out once (see src()): an SGPR-like register,
    // a value fixed by the instruction, or neither (unknown to a replay, or not handled here).
    struct Operand {
        enum Kind : uint8_t { kReg, kValue, kOther } kind = kOther;
        uint32_t code = 0, value = 0, hi = 0; // hi: the upper word src64 gives a fixed value
    };
    static Operand operand(uint32_t code, const Instruction& in) {
        Operand o;
        o.code = code;
        if (code < 104 || code == kVcc || code == kVcc + 1 || code == kM0) o.kind = Operand::kReg;
        else if (code >= 128 && code <= 192) o.kind = Operand::kValue, o.value = code - 128;
        else if (code >= 193 && code <= 208) o.kind = Operand::kValue, o.value = uint32_t(192 - int32_t(code)), o.hi = ~0u;
        else if (code >= 240 && code <= 247) {
            static const float c[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f};
            o.kind = Operand::kValue;
            std::memcpy(&o.value, &c[code - 240], 4);
        } else if (code == 255) o.kind = Operand::kValue, o.value = in.literal;
        return o;
    }

    // The SGPRs (0-127 and vcc) a step only marks unknown, when it does nothing else a replay keeps.
    static bool onlyUnknowns(const Instruction& in, uint32_t& d, unsigned& n) {
        const uint32_t w = in.words[0];
        switch (in.format) {
        case Format::VOP3:
            if (in.op < 256) return d = w & 0xFF, n = 2, true;
            if (writesSgprPair(in)) return d = (w >> 8) & 0x7F, n = 2, true;
            return false;
        case Format::VOPC: return d = kVcc, n = 2, (in.op & 16) == 0; // v_cmpx also writes exec
        case Format::VOP1: return in.op == 2 && (d = (w >> 17) & 0xFF, n = 1, true);
        case Format::VOP2:
            if (in.op == 1) return d = (w >> 17) & 0xFF, n = 1, true;
            if (in.op >= 37 && in.op <= 42) return d = kVcc, n = 2, true;
            return false;
        default: return false;
        }
    }

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
        if (prior_ || speculative_) return; // a replay reports nothing new; skipped code is unknown by design
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

    void branchTo(uint64_t target, const State& st, WorkQueue& work) {
        label(target);
        p_.controlFlow = true;
        work.push_back({target, st});
    }
    void label(uint64_t pc) {
        if (!prior_) p_.labels.insert(pc); // a replay keeps the prior labels
    }

    // VOP3 instructions that write an SGPR pair: compares, and VOP3b's carry or flags.
    static bool writesSgprPair(const Instruction& in) {
        return in.op < 256 || (in.op >= 293 && in.op <= 298) || in.op == 365 || in.op == 366 || in.op == 374 ||
               in.op == 375;
    }

    // Instructions step() does nothing for: a replay can skip them. Most of a shader is vector
    // ALU work, so this keeps traces short.
    static bool inert(const Instruction& in) {
        switch (in.format) {
        case Format::SOPP: return in.op != 1 && in.op != 2 && !(in.op >= 4 && in.op <= 9);
        case Format::SMRD: return in.op >= 29 || (in.op & 7) > 4;
        case Format::VOP3: return !writesSgprPair(in);
        case Format::VOP1: return in.op != 2;
        case Format::VOP2: return in.op != 1 && !(in.op >= 37 && in.op <= 42);
        case Format::EXP:
        case Format::VINTRP:
        case Format::DS: return true;
        default: return false;
        }
    }

    // Returns false when this path ends.
    bool step(const Instruction& in, uint64_t pc, State& st, uint64_t next, WorkQueue& work) {
        const uint32_t w = in.words[0];
        switch (in.format) {
        case Format::SOPP: {
            const uint64_t target = pc + 4 + int64_t(int16_t(w & 0xFFFF)) * 4;
            switch (in.op) {
            case 1: return false; // s_endpgm
            case 2: branchTo(target, st, work); return false;
            case 4: case 5: // s_cbranch_scc0/1
                if (st.sccKnown) {
                    const bool taken = st.scc == (in.op == 5);
                    if (!prior_ && generic()) { // the other way is kept and walked afterwards
                        const uint64_t other = taken ? next : target;
                        label(target);
                        label(next);
                        p_.controlFlow = true;
                        if (speculative_) work.push_back({other, State{}});
                        else untaken_.push_back(other);
                    }
                    if (taken) {
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
            else if (writesSgprPair(in)) unknown(st, (w >> 8) & 0x7F, 2); // VOP3b: carry or flags into an SGPR pair
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

    bool sop1(const Instruction& in, uint64_t pc, State& st, uint64_t next, WorkQueue& work) {
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
        else if ((w & 0xFF) == 255) off = in.literal * 4; // CI's 32-bit literal offset counts dwords, like the 8-bit one
        else known = src(st, w & 0xFF, in, off);

        uint32_t at;
        if (prior_) { // the trace step holds where its result goes
            if (slot_ == ~0u) {
                diverged_ = true;
                return;
            }
            at = slot_;
        } else if (auto it = p_.smrd.find(pc); it == p_.smrd.end()) {
            at = uint32_t(p_.data.size());
            p_.smrd[pc] = at;
            p_.data.resize(at + n, 0);
        } else {
            at = it->second;
        }
        slot_ = at;
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
        if (prior_) { // the trace step holds the resource's index
            if (slot_ < p_.buffers.size() && p_.buffers[slot_].pc == pc) found = int(slot_);
        } else {
            for (size_t i = 0; i < p_.buffers.size(); ++i)
                if (p_.buffers[i].pc == pc) found = int(i);
            slot_ = found >= 0 ? uint32_t(found) : uint32_t(p_.buffers.size());
        }
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
        if (prior_) {
            if (slot_ < p_.images.size() && p_.images[slot_].pc == pc) found = int(slot_);
        } else {
            for (size_t i = 0; i < p_.images.size(); ++i)
                if (p_.images[i].pc == pc) found = int(i);
            slot_ = found >= 0 ? uint32_t(found) : uint32_t(p_.images.size());
        }
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
    bool speculative_ = false;          // walking code this launch skips (generic())
    std::vector<uint64_t> untaken_;     // where the scalar branches this launch decided did not go
    uint32_t slot_ = ~0u; // the stepped instruction's TraceStep::slot
    // Kept per thread: a replay runs for every launch, and these would allocate each time.
    static std::vector<bool>& seenStorage(int i) {
        thread_local std::vector<bool> v[2];
        return v[i];
    }
    std::vector<bool>&seenBuffers_ = seenStorage(0), &seenImages_ = seenStorage(1);
    Program p_;
};

} // namespace

// A trace pre-decoded: one op per step, with the fields a replay of the step needs worked out.
// Scalar loads, moves and the vector steps that only mark SGPRs unknown run straight from their
// op; anything else steps its instruction as the full replay does.
struct ReplayOp {
    enum Kind : uint8_t { kStart, kStep, kLoad, kMov, kMov64, kUnknown } kind = kStep;
    uint8_t n = 0;     // kLoad: dwords; kUnknown: SGPRs
    bool buf = false;  // kLoad: s_buffer_load (its address is a V#)
    uint8_t d = 0;     // destination SGPR
    uint8_t base = 0;  // kLoad: first SGPR of the address
    uint32_t slot = 0; // kLoad: offset of its result in the draw data
    uint32_t trace = 0; // index into the trace
    Analyzer::Operand src; // kLoad: the offset; kMov/kMov64: the source
};

struct ReplayPlan {
    std::vector<ReplayOp> ops;
    uint32_t direct = 0; // ops run without stepping
};

namespace {

std::shared_ptr<const ReplayPlan> makePlan(const std::vector<Program::TraceStep>& trace) {
    auto plan = std::make_shared<ReplayPlan>();
    plan->ops.reserve(trace.size());
    static const unsigned kCount[] = {1, 2, 4, 8, 16};
    for (size_t i = 0; i < trace.size(); ++i) {
        const Program::TraceStep& t = trace[i];
        ReplayOp o;
        o.trace = uint32_t(i);
        const uint32_t w = t.in.words[0];
        uint32_t d = 0;
        unsigned n = 0;
        if (t.start) {
            o.kind = ReplayOp::kStart;
        } else if (t.in.format == Format::SMRD && t.in.op < 29 && (t.in.op & 7) <= 4 && t.slot != ~0u && t.result) {
            o.kind = ReplayOp::kLoad;
            o.n = uint8_t(kCount[t.in.op & 7]);
            o.buf = t.in.op >= 8;
            o.d = uint8_t((w >> 15) & 0x7F);
            o.base = uint8_t(((w >> 9) & 0x3F) * 2);
            o.slot = t.slot;
            if ((w >> 8) & 1) o.src.kind = Analyzer::Operand::kValue, o.src.value = (w & 0xFF) * 4;
            else if ((w & 0xFF) == 255) o.src.kind = Analyzer::Operand::kValue, o.src.value = t.in.literal * 4;
            else o.src = Analyzer::operand(w & 0xFF, t.in);
            // Address registers src() reads as plain SGPRs (the buffer form also reads base + 2).
            if (o.src.kind == Analyzer::Operand::kOther || o.base + 3 >= 104) o.kind = ReplayOp::kStep;
        } else if (t.in.format == Format::SOP1 && (t.in.op == 3 || t.in.op == 4) && t.result) {
            o.kind = t.in.op == 3 ? ReplayOp::kMov : ReplayOp::kMov64;
            o.d = uint8_t((w >> 16) & 0x7F);
            o.src = Analyzer::operand(w & 0xFF, t.in);
            if (o.src.kind == Analyzer::Operand::kOther) o.kind = ReplayOp::kStep;
            // src64 reads these as one register (high word 0): left to the full step.
            if (o.kind == ReplayOp::kMov64 && o.src.kind == Analyzer::Operand::kReg &&
                (o.src.code == kVcc + 1 || o.src.code == kM0))
                o.kind = ReplayOp::kStep;
        } else if (t.result && Analyzer::onlyUnknowns(t.in, d, n) && d + n <= 128) {
            o.kind = ReplayOp::kUnknown;
            o.d = uint8_t(d);
            o.n = uint8_t(n);
        }
        if (o.kind != ReplayOp::kStep && o.kind != ReplayOp::kStart) ++plan->direct;
        plan->ops.push_back(o);
    }
    return plan;
}

int fastReplayMode() { // 0 off, 1 on, 2 check
    static const int mode = [] {
        const char* e = std::getenv("LPN_FAST_REPLAY");
        if (!e || !*e || *e == '0') return 0;
        return std::strcmp(e, "check") == 0 ? 2 : 1;
    }();
    return mode;
}

std::atomic<uint64_t> gPlanned{0}, gSteps{0}, gKept{0}, gMismatches{0}, gRaces{0};

bool sameReplay(const Program& a, const Program& b) {
    if (a.data != b.data || a.buffers.size() != b.buffers.size() || a.images.size() != b.images.size()) return false;
    for (size_t i = 0; i < a.buffers.size(); ++i) {
        const BufferResource &x = a.buffers[i], &y = b.buffers[i];
        if (x.pc != y.pc || std::memcmp(x.sharp, y.sharp, sizeof x.sharp) || x.dfmt != y.dfmt || x.nfmt != y.nfmt ||
            x.store != y.store || x.known != y.known || x.data != y.data)
            return false;
    }
    for (size_t i = 0; i < a.images.size(); ++i) {
        const ImageResource &x = a.images[i], &y = b.images[i];
        if (x.pc != y.pc || std::memcmp(x.t, y.t, sizeof x.t) || std::memcmp(x.s, y.s, sizeof x.s) ||
            x.storage != y.storage || x.compare != y.compare || x.data != y.data)
            return false;
    }
    return true;
}

} // namespace

bool Analyzer::replay(const ReplayPlan* plan) { return plan ? replayOps(*plan) : replayAll(); }

// replayAll() over the plan's ops: the same state changes, without decoding or dispatching on
// the instruction for the ops that run directly.
bool Analyzer::replayOps(const ReplayPlan& plan) {
    const Program& prior = *prior_;
    p_.data.assign(prior.data.size(), 0);
    p_.buffers = prior.buffers;
    p_.images = prior.images;
    seenBuffers_.assign(p_.buffers.size(), false);
    seenImages_.assign(p_.images.size(), false);
    WorkQueue work{Work{p_.launch.entry, start()}};
    State st;
    auto value = [&](const Operand& o, uint32_t& v) {
        if (o.kind == Operand::kValue) return v = o.value, true;
        v = st.s[o.code];
        return st.known[o.code];
    };
    for (const ReplayOp& o : plan.ops) {
        const Program::TraceStep& t = prior.trace[o.trace];
        switch (o.kind) {
        case ReplayOp::kStart:
            if (work.empty() || work.front().pc != t.pc) return false;
            st = std::move(work.front().st);
            work.pop_front();
            break;
        case ReplayOp::kUnknown: unknown(st, o.d, o.n); break;
        case ReplayOp::kMov: {
            uint32_t v;
            if (value(o.src, v)) set(st, o.d, v);
            else unknown(st, o.d);
            break;
        }
        case ReplayOp::kMov64: {
            uint32_t lo, hi = o.src.hi;
            bool k = value(o.src, lo);
            if (o.src.kind == Operand::kReg) {
                hi = st.s[o.src.code + 1];
                k &= st.known[o.src.code + 1];
            }
            if (k) set64(st, o.d, lo | (uint64_t(hi) << 32));
            else unknown(st, o.d, 2);
            break;
        }
        case ReplayOp::kLoad: {
            const unsigned n = o.n;
            const uint32_t at = o.slot;
            if (at + n > p_.data.size()) return false;
            uint32_t off = 0;
            bool known = value(o.src, off);
            const uint32_t lo = st.s[o.base], hi = st.s[o.base + 1];
            known &= st.known[o.base] & st.known[o.base + 1];
            if (!known) {
                unknown(st, o.d, n);
                break;
            }
            uint64_t addr;
            uint32_t limit = ~0u;
            if (o.buf) {
                const uint32_t sharp2 = st.s[o.base + 2];
                addr = lo | (uint64_t(hi & 0xFFF) << 32);
                const uint32_t stride = (hi >> 16) & 0x3FFF;
                limit = stride ? uint32_t(std::min<uint64_t>(uint64_t(stride) * sharp2, ~0u)) : sharp2;
            } else {
                addr = (lo | (uint64_t(hi & 0xFFFF) << 32)) & ~3ull;
            }
            uint32_t vals[16] = {};
            unsigned inside = n;
            if (o.buf) inside = off >= limit ? 0 : std::min<unsigned>(n, (limit - off) / 4);
            if (inside && !readData_(addr + off, vals, inside * 4)) {
                std::memset(vals, 0, sizeof vals);
                for (unsigned i = 0; i < inside; ++i)
                    if (!readData_(addr + off + i * 4, &vals[i], 4)) break;
            }
            uint32_t* out = &p_.data[at];
            for (unsigned i = 0; i < n; ++i) {
                out[i] = vals[i];
                set(st, o.d + i, vals[i]);
            }
            break;
        }
        case ReplayOp::kStep:
            slot_ = t.slot;
            if (step(t.in, t.pc, st, t.pc + t.in.size, work) != t.result || diverged_) return false;
            break;
        }
    }
    return work.empty();
}

ReplayStats takeReplayStats() {
    ReplayStats s;
    s.planned = gPlanned.exchange(0, std::memory_order_relaxed);
    s.steps = gSteps.exchange(0, std::memory_order_relaxed);
    s.kept = gKept.exchange(0, std::memory_order_relaxed);
    s.mismatches = gMismatches.exchange(0, std::memory_order_relaxed);
    s.races = gRaces.exchange(0, std::memory_order_relaxed);
    return s;
}

bool genericTranslation() {
    static const bool on = [] {
        const char* e = std::getenv("LPN_GENERIC_SHADERS");
        return !(e && !std::strcmp(e, "0"));
    }();
    return on;
}

Program analyze(const LaunchState& launch, const ReadMemory& readCode, const ReadMemory& readData) {
    return Analyzer(launch, readCode, readData).run();
}

bool replay(const Program& p, const LaunchState& launch, const ReadMemory& readData, Program& out) {
    if (p.trace.empty()) return false;
    const int mode = fastReplayMode();
    std::shared_ptr<const ReplayPlan> plan;
    if (mode) {
        plan = p.plan.get();
        if (!plan) {
            plan = makePlan(p.trace);
            p.plan.set(plan);
        }
        gPlanned.fetch_add(1, std::memory_order_relaxed);
        gSteps.fetch_add(p.trace.size(), std::memory_order_relaxed);
        gKept.fetch_add(plan->direct, std::memory_order_relaxed);
    }
    Analyzer a(launch, readData, readData, &p);
    // A replay runs for every launch: it fills out's vectors, keeping their storage.
    Program& q = a.program();
    q.data.swap(out.data);
    q.buffers.swap(out.buffers);
    q.images.swap(out.images);
    bool ok = a.replay(plan.get());
    if (mode == 2) {
        Analyzer full(launch, readData, readData, &p);
        const bool fullOk = full.replay();
        if (fullOk != ok || (ok && !sameReplay(q, full.program()))) {
            // Memory the game rewrites while we read it makes two full replays differ too:
            // that is not the plan's doing.
            Analyzer again(launch, readData, readData, &p);
            const bool againOk = again.replay();
            const bool race = againOk != fullOk || (fullOk && !sameReplay(full.program(), again.program()));
            static std::atomic<int> logged{0};
            if (!race && logged.fetch_add(1) < 20)
                std::fprintf(stderr, "[replay] fast replay differs from full replay for shader 0x%llx (%s): %u of %zu steps direct\n",
                             (unsigned long long)launch.entry, ok != fullOk ? "result" : "output", plan->direct,
                             p.trace.size());
            (race ? gRaces : gMismatches).fetch_add(1, std::memory_order_relaxed);
            q = std::move(full.program()); // the full replay's answer stands
            ok = fullOk;
        }
    }
    out.launch = q.launch;
    out.data.swap(q.data);
    out.buffers.swap(q.buffers);
    out.images.swap(q.images);
    return ok;
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
    // Swapped rather than moved: r takes p's old storage, for the caller's next replay.
    p.launch = r.launch;
    p.data.swap(r.data);
    p.buffers.swap(r.buffers);
    p.images.swap(r.images);
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
        // What the translation reads of an image: the data format and a sampled image's swizzle
        // only matter to the view, so images that differ in them share a translation.
        const uint32_t nf = i.numFormat();
        const uint32_t t = i.type();
        const uint32_t dim = t == 8 || t == 10 || t == 11 || t == 12 || t == 13 ? t : 9u; // the declared dimension
        const uint32_t shape[] = {dim, nf == 4 || nf == 5 ? nf : 0u, i.storage ? i.dstSel() : 0u, i.compare,
                                  i.storage, (i.s[0] >> 15) & 1};
        put(shape, sizeof(shape));
    }
    return k;
}


namespace {
constexpr uint32_t kSerialVersion = 1;
template <class T> void putPod(std::string& s, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    s.append(reinterpret_cast<const char*>(&v), sizeof v);
}
struct Reader {
    const std::string& s;
    size_t at = 0;
    template <class T> bool get(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (s.size() - at < sizeof v) return false;
        std::memcpy(&v, s.data() + at, sizeof v);
        at += sizeof v;
        return true;
    }
    bool count(uint32_t& n) { return get(n) && n <= (s.size() - at); } // each element is at least a byte
};
} // namespace

std::string serialize(const Program& p) {
    std::string s;
    const uint32_t sizes[] = {kSerialVersion, uint32_t(sizeof(LaunchState)), uint32_t(sizeof(Instruction)),
                              uint32_t(sizeof(BufferResource)), uint32_t(sizeof(ImageResource))};
    putPod(s, sizes);
    putPod(s, p.launch);
    putPod(s, uint8_t(p.controlFlow | p.writesExec << 1));
    putPod(s, uint32_t(p.code.size()));
    for (const auto& [pc, in] : p.code) putPod(s, pc), putPod(s, in);
    putPod(s, uint32_t(p.labels.size()));
    for (uint64_t l : p.labels) putPod(s, l);
    putPod(s, uint32_t(p.calls.size()));
    for (const auto& [site, callee] : p.calls) putPod(s, site), putPod(s, callee);
    putPod(s, uint32_t(p.smrd.size()));
    for (const auto& [pc, slot] : p.smrd) putPod(s, pc), putPod(s, slot);
    putPod(s, uint32_t(p.data.size()));
    for (uint32_t d : p.data) putPod(s, d);
    putPod(s, uint32_t(p.buffers.size()));
    for (const auto& b : p.buffers) putPod(s, b);
    putPod(s, uint32_t(p.images.size()));
    for (const auto& i : p.images) putPod(s, i);
    return s;
}

bool deserialize(const std::string& bytes, Program& p) {
    p = Program{};
    Reader r{bytes};
    uint32_t sizes[5];
    if (!r.get(sizes) || sizes[0] != kSerialVersion || sizes[1] != sizeof(LaunchState) || sizes[2] != sizeof(Instruction) ||
        sizes[3] != sizeof(BufferResource) || sizes[4] != sizeof(ImageResource))
        return false;
    uint8_t flags;
    uint32_t n;
    if (!r.get(p.launch) || !r.get(flags)) return false;
    p.controlFlow = flags & 1, p.writesExec = flags & 2;
    if (!r.count(n)) return false;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t pc;
        Instruction in;
        if (!r.get(pc) || !r.get(in)) return false;
        p.code.emplace(pc, in);
    }
    if (!r.count(n)) return false;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t l;
        if (!r.get(l)) return false;
        p.labels.insert(l);
    }
    if (!r.count(n)) return false;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t site, callee;
        if (!r.get(site) || !r.get(callee)) return false;
        p.calls.emplace(site, callee);
    }
    if (!r.count(n)) return false;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t pc;
        uint32_t slot;
        if (!r.get(pc) || !r.get(slot)) return false;
        p.smrd.emplace(pc, slot);
    }
    if (!r.count(n)) return false;
    p.data.resize(n);
    for (auto& d : p.data)
        if (!r.get(d)) return false;
    if (!r.count(n)) return false;
    p.buffers.resize(n);
    for (auto& b : p.buffers)
        if (!r.get(b)) return false;
    if (!r.count(n)) return false;
    p.images.resize(n);
    for (auto& i : p.images)
        if (!r.get(i)) return false;
    return r.at == bytes.size();
}

} // namespace gcn
