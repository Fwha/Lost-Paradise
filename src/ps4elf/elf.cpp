#include "ps4elf/elf.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace ps4elf {

namespace {

template <typename T>
T read(std::span<const uint8_t> d, uint64_t off) {
    T v{};
    std::memcpy(&v, d.data() + off, sizeof(T));
    return v;
}

// A SELF (signed ELF) holds the ELF header and program headers after its own header and entry
// table, and each segment's bytes in an entry of their own. Fake-signed dumps (fSELF, as some
// releases' eboot.bin and modules come) keep those bytes in the clear: they are put back at each
// segment's file offset, giving the ELF the console's loader would have seen. Encrypted or
// compressed entries are refused.
constexpr uint8_t kSelfMagic[4] = {0x4f, 0x15, 0x3d, 0x1d};
constexpr uint64_t kSelfEncrypted = 1u << 1, kSelfCompressed = 1u << 3, kSelfBlocks = 1u << 11;

bool unwrapSelf(std::vector<uint8_t>& data, std::string* error) {
    auto fail = [&](const char* why) {
        if (error) *error = why;
        return false;
    };
    std::span<const uint8_t> d = data;
    if (d.size() < 0x20) return fail("truncated SELF header");
    const uint16_t entries = read<uint16_t>(d, 0x18);
    const uint64_t elfAt = 0x20 + uint64_t(entries) * 0x20;
    if (elfAt + 64 > d.size() || std::memcmp(d.data() + elfAt, "\x7f" "ELF", 4) != 0)
        return fail("SELF without an ELF header");
    const uint64_t phoff = read<uint64_t>(d, elfAt + 32);
    const uint16_t phentsize = read<uint16_t>(d, elfAt + 54), phnum = read<uint16_t>(d, elfAt + 56);
    if (phentsize != 56 || elfAt + phoff + uint64_t(phnum) * 56 > d.size()) return fail("bad SELF program header table");
    struct Entry {
        uint64_t props, offset, size;
    };
    std::vector<Entry> list;
    for (uint16_t i = 0; i < entries; ++i) {
        const uint64_t e = 0x20 + uint64_t(i) * 0x20;
        list.push_back({read<uint64_t>(d, e), read<uint64_t>(d, e + 8), read<uint64_t>(d, e + 16)});
    }
    uint64_t size = phoff + uint64_t(phnum) * 56;
    for (uint16_t i = 0; i < phnum; ++i) {
        const uint64_t p = elfAt + phoff + uint64_t(i) * 56;
        size = std::max(size, read<uint64_t>(d, p + 8) + read<uint64_t>(d, p + 32));
    }
    if (size > (uint64_t(1) << 32)) return fail("bad SELF program header table");
    std::vector<uint8_t> out(size);
    std::memcpy(out.data(), d.data() + elfAt, phoff + uint64_t(phnum) * 56);
    for (const Entry& e : list) {
        if (!(e.props & kSelfBlocks)) continue; // signatures and digests
        if (e.props & kSelfEncrypted) return fail("encrypted SELF: decrypt it first (only fake-signed SELFs load)");
        if (e.props & kSelfCompressed) return fail("compressed SELF segments are not supported");
        const uint64_t seg = (e.props >> 20) & 0xfff;
        if (seg >= phnum || e.offset + e.size > d.size()) return fail("bad SELF segment entry");
        const uint64_t p = elfAt + phoff + seg * 56;
        const uint64_t at = read<uint64_t>(d, p + 8), filesz = read<uint64_t>(d, p + 32);
        if (e.size > filesz) return fail("bad SELF segment entry");
        std::memcpy(out.data() + at, d.data() + e.offset, e.size);
    }
    data = std::move(out);
    return true;
}

} // namespace

std::optional<Elf> Elf::load(const std::string& path, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = "cannot open " + path;
        return std::nullopt;
    }
    Elf elf;
    elf.data_.assign(std::istreambuf_iterator<char>(in), {});
    if (elf.data_.size() >= 4 && std::memcmp(elf.data_.data(), kSelfMagic, 4) == 0 && !unwrapSelf(elf.data_, error))
        return std::nullopt;
    std::span<const uint8_t> d = elf.data_;

    if (d.size() < 64 || std::memcmp(d.data(), "\x7f" "ELF", 4) != 0) {
        if (error) *error = "not an ELF or SELF file";
        return std::nullopt;
    }
    if (d[4] != 2 || d[5] != 1) {
        if (error) *error = "expected 64-bit little-endian ELF";
        return std::nullopt;
    }

    elf.type_ = read<uint16_t>(d, 16);
    elf.entry_ = read<uint64_t>(d, 24);
    const uint64_t phoff = read<uint64_t>(d, 32);
    const uint16_t phentsize = read<uint16_t>(d, 54);
    const uint16_t phnum = read<uint16_t>(d, 56);
    if (phentsize != 56 || phoff + uint64_t(phnum) * phentsize > d.size()) {
        if (error) *error = "bad program header table";
        return std::nullopt;
    }

    for (uint16_t i = 0; i < phnum; ++i) {
        const uint64_t p = phoff + uint64_t(i) * phentsize;
        Segment s;
        s.type = read<uint32_t>(d, p + 0);
        s.flags = read<uint32_t>(d, p + 4);
        s.offset = read<uint64_t>(d, p + 8);
        s.vaddr = read<uint64_t>(d, p + 16);
        s.filesz = read<uint64_t>(d, p + 32);
        s.memsz = read<uint64_t>(d, p + 40);
        s.align = read<uint64_t>(d, p + 48);
        elf.segments_.push_back(s);
    }
    return elf;
}

const Segment* Elf::findSegment(uint32_t type) const {
    for (const auto& s : segments_)
        if (s.type == type) return &s;
    return nullptr;
}

std::optional<uint64_t> Elf::vaddrToOffset(uint64_t vaddr) const {
    // PT_LOAD and PT_SCE_RELRO are the segments that actually get mapped.
    for (const auto& s : segments_) {
        if (s.type != PT_LOAD && s.type != PT_SCE_RELRO) continue;
        if (vaddr >= s.vaddr && vaddr < s.vaddr + s.filesz) return s.offset + (vaddr - s.vaddr);
    }
    return std::nullopt;
}

const uint8_t* Elf::at(uint64_t vaddr, uint64_t size) const {
    auto off = vaddrToOffset(vaddr);
    if (!off || *off + size > data_.size()) return nullptr;
    return data_.data() + *off;
}

uint64_t Elf::imageSize() const {
    uint64_t end = 0;
    for (const auto& s : segments_)
        if (s.type == PT_LOAD || s.type == PT_SCE_RELRO) end = std::max(end, s.vaddr + s.memsz);
    return end;
}

} // namespace ps4elf
