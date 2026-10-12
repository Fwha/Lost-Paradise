#include "ps4elf/ehframe.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>

namespace ps4elf {

namespace {

enum : uint8_t {
    DW_EH_PE_absptr = 0x00,
    DW_EH_PE_uleb128 = 0x01,
    DW_EH_PE_udata2 = 0x02,
    DW_EH_PE_udata4 = 0x03,
    DW_EH_PE_udata8 = 0x04,
    DW_EH_PE_sleb128 = 0x09,
    DW_EH_PE_sdata2 = 0x0A,
    DW_EH_PE_sdata4 = 0x0B,
    DW_EH_PE_sdata8 = 0x0C,
    DW_EH_PE_pcrel = 0x10,
    DW_EH_PE_datarel = 0x30,
    DW_EH_PE_omit = 0xFF,
};

// Sequential reader over the image, addressed by virtual address.
class Cursor {
public:
    Cursor(const Elf& elf, uint64_t vaddr) : elf_(elf), pos_(vaddr) {}

    uint64_t pos() const { return pos_; }
    void seek(uint64_t vaddr) { pos_ = vaddr; }
    bool ok() const { return ok_; }

    template <typename T>
    T get() {
        const uint8_t* p = elf_.at(pos_, sizeof(T));
        if (!p) {
            ok_ = false;
            return T{};
        }
        T v;
        std::memcpy(&v, p, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    uint64_t uleb() {
        uint64_t v = 0;
        for (int shift = 0; ok_ && shift < 64; shift += 7) {
            const uint8_t b = get<uint8_t>();
            v |= uint64_t(b & 0x7F) << shift;
            if (!(b & 0x80)) break;
        }
        return v;
    }

    int64_t sleb() {
        int64_t v = 0;
        int shift = 0;
        uint8_t b = 0;
        do {
            b = get<uint8_t>();
            v |= int64_t(b & 0x7F) << shift;
            shift += 7;
        } while (ok_ && (b & 0x80) && shift < 64);
        if (shift < 64 && (b & 0x40)) v |= -(int64_t(1) << shift);
        return v;
    }

    std::string cstr() {
        std::string s;
        for (char c; ok_ && (c = char(get<uint8_t>()));) s += c;
        return s;
    }

    // Read an encoded pointer. dataBase is the eh_frame_hdr address for datarel.
    uint64_t pointer(uint8_t enc, uint64_t dataBase = 0) {
        if (enc == DW_EH_PE_omit) return 0;
        const uint64_t fieldAddr = pos_;
        uint64_t v = 0;
        switch (enc & 0x0F) {
        case DW_EH_PE_absptr: v = get<uint64_t>(); break;
        case DW_EH_PE_uleb128: v = uleb(); break;
        case DW_EH_PE_udata2: v = get<uint16_t>(); break;
        case DW_EH_PE_udata4: v = get<uint32_t>(); break;
        case DW_EH_PE_udata8: v = get<uint64_t>(); break;
        case DW_EH_PE_sleb128: v = uint64_t(sleb()); break;
        case DW_EH_PE_sdata2: v = uint64_t(int64_t(get<int16_t>())); break;
        case DW_EH_PE_sdata4: v = uint64_t(int64_t(get<int32_t>())); break;
        case DW_EH_PE_sdata8: v = get<uint64_t>(); break;
        default: ok_ = false; return 0;
        }
        switch (enc & 0x70) {
        case 0: break;
        case DW_EH_PE_pcrel: v += fieldAddr; break;
        case DW_EH_PE_datarel: v += dataBase; break;
        default: ok_ = false; break;
        }
        return v;
    }

private:
    const Elf& elf_;
    uint64_t pos_;
    bool ok_ = true;
};

// The FDE pointer encoding is declared in the CIE's 'R' augmentation.
std::optional<uint8_t> cieFdeEncoding(const Elf& elf, uint64_t cieAddr) {
    Cursor c(elf, cieAddr);
    uint64_t length = c.get<uint32_t>();
    if (length == 0xFFFFFFFF) length = c.get<uint64_t>();
    if (c.get<uint32_t>() != 0) return std::nullopt; // not a CIE
    const uint8_t version = c.get<uint8_t>();
    const std::string aug = c.cstr();
    if (aug.find("eh") != std::string::npos) c.get<uint64_t>();
    c.uleb(); // code alignment
    c.sleb(); // data alignment
    if (version == 1) c.get<uint8_t>(); else c.uleb(); // return address register

    uint8_t fdeEnc = DW_EH_PE_absptr;
    if (!aug.empty() && aug[0] == 'z') {
        c.uleb(); // augmentation data length
        for (size_t i = 1; i < aug.size(); ++i) {
            switch (aug[i]) {
            case 'R': fdeEnc = c.get<uint8_t>(); break;
            case 'L': c.get<uint8_t>(); break;
            case 'P': c.pointer(c.get<uint8_t>()); break;
            case 'S': break;
            default: return c.ok() ? std::optional<uint8_t>(fdeEnc) : std::nullopt;
            }
        }
    }
    if (!c.ok()) return std::nullopt;
    return fdeEnc;
}

} // namespace

std::vector<FunctionRange> functionsFromEhFrame(const Elf& elf, std::string* error) {
    const Segment* hdrSeg = elf.findSegment(PT_GNU_EH_FRAME);
    if (!hdrSeg) {
        if (error) *error = "no PT_GNU_EH_FRAME segment";
        return {};
    }
    const uint64_t hdr = hdrSeg->vaddr;
    Cursor c(elf, hdr);
    const uint8_t version = c.get<uint8_t>();
    const uint8_t ehFramePtrEnc = c.get<uint8_t>();
    const uint8_t countEnc = c.get<uint8_t>();
    const uint8_t tableEnc = c.get<uint8_t>();
    c.pointer(ehFramePtrEnc, hdr);
    const uint64_t count = c.pointer(countEnc, hdr);
    if (!c.ok() || version != 1 || countEnc == DW_EH_PE_omit || tableEnc == DW_EH_PE_omit) {
        if (error) *error = "unsupported .eh_frame_hdr";
        return {};
    }

    std::map<uint64_t, std::optional<uint8_t>> cieCache;
    std::vector<FunctionRange> out;
    out.reserve(count);
    for (uint64_t i = 0; i < count && c.ok(); ++i) {
        c.pointer(tableEnc, hdr); // initial location (also in the FDE itself)
        const uint64_t fde = c.pointer(tableEnc, hdr);

        Cursor f(elf, fde);
        uint64_t length = f.get<uint32_t>();
        if (length == 0xFFFFFFFF) length = f.get<uint64_t>();
        const uint64_t cieFieldAddr = f.pos();
        const uint32_t ciePtr = f.get<uint32_t>();
        if (!f.ok() || ciePtr == 0) continue;
        const uint64_t cie = cieFieldAddr - ciePtr;

        auto it = cieCache.find(cie);
        if (it == cieCache.end()) it = cieCache.emplace(cie, cieFdeEncoding(elf, cie)).first;
        if (!it->second) continue;

        const uint64_t start = f.pointer(*it->second, hdr);
        const uint64_t size = f.pointer(*it->second & 0x0F, hdr);
        if (f.ok() && size) out.push_back({start, size});
    }
    if (!c.ok()) {
        if (error) *error = ".eh_frame_hdr table runs past the image";
        return {};
    }

    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.start < b.start; });
    out.erase(std::unique(out.begin(), out.end(),
                          [](auto& a, auto& b) { return a.start == b.start; }),
              out.end());
    return out;
}

} // namespace ps4elf
