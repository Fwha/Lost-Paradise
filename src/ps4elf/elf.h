// Reader for decrypted PS4 ELF executables (eboot.bin, .prx/.sprx after decryption).
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ps4elf {

// PS4-specific values that standard ELF tools don't know about.
constexpr uint16_t ET_SCE_EXEC = 0xFE00;
constexpr uint16_t ET_SCE_DYNEXEC = 0xFE10;
constexpr uint16_t ET_SCE_DYNAMIC = 0xFE18;

constexpr uint32_t PT_LOAD = 1;
constexpr uint32_t PT_DYNAMIC = 2;
constexpr uint32_t PT_TLS = 7;
constexpr uint32_t PT_GNU_EH_FRAME = 0x6474E550;
constexpr uint32_t PT_SCE_DYNLIBDATA = 0x61000000;
constexpr uint32_t PT_SCE_PROCPARAM = 0x61000001;
constexpr uint32_t PT_SCE_RELRO = 0x61000010;

struct Segment {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t vaddr;
    uint64_t filesz;
    uint64_t memsz;
    uint64_t align;
};

class Elf {
public:
    static std::optional<Elf> load(const std::string& path, std::string* error);

    uint16_t type() const { return type_; }
    uint64_t entry() const { return entry_; }
    const std::vector<Segment>& segments() const { return segments_; }
    const Segment* findSegment(uint32_t type) const;

    // Bytes of the whole file.
    std::span<const uint8_t> file() const { return data_; }

    // Translate a virtual address in a loaded segment to a file offset.
    std::optional<uint64_t> vaddrToOffset(uint64_t vaddr) const;
    // Pointer into the file for a virtual address, or nullptr if it isn't file-backed.
    const uint8_t* at(uint64_t vaddr, uint64_t size = 1) const;

    // Highest end address of any PT_LOAD segment.
    uint64_t imageSize() const;

private:
    std::vector<uint8_t> data_;
    std::vector<Segment> segments_;
    uint16_t type_ = 0;
    uint64_t entry_ = 0;
};

} // namespace ps4elf
