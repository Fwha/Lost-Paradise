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

} // namespace

std::optional<Elf> Elf::load(const std::string& path, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = "cannot open " + path;
        return std::nullopt;
    }
    Elf elf;
    elf.data_.assign(std::istreambuf_iterator<char>(in), {});
    std::span<const uint8_t> d = elf.data_;

    if (d.size() < 64 || std::memcmp(d.data(), "\x7f" "ELF", 4) != 0) {
        if (error) *error = "not an ELF file (still encrypted SELF?)";
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
