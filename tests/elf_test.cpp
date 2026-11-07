// Tests for loading fake-signed SELF files (synthetic: one PT_LOAD segment), as some releases'
// eboot.bin and modules come.

#include "ps4elf/elf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

template <class T> void put(std::vector<uint8_t>& v, size_t at, T x) {
    if (v.size() < at + sizeof x) v.resize(at + sizeof x);
    std::memcpy(v.data() + at, &x, sizeof x);
}

constexpr uint64_t kSegOffset = 0x4000, kVaddr = 0, kSegSize = 0x20;

// An ELF header and one program header (PT_LOAD at file offset 0x4000).
std::vector<uint8_t> elfHeaders() {
    std::vector<uint8_t> h(64 + 56);
    std::memcpy(h.data(), "\x7f" "ELF", 4);
    h[4] = 2, h[5] = 1;
    put<uint16_t>(h, 16, 0xfe10);
    put<uint64_t>(h, 24, 0x10);  // entry
    put<uint64_t>(h, 32, 64);    // phoff
    put<uint16_t>(h, 54, 56);
    put<uint16_t>(h, 56, 1);
    put<uint32_t>(h, 64, 1);     // PT_LOAD
    put<uint32_t>(h, 68, 5);
    put<uint64_t>(h, 72, kSegOffset);
    put<uint64_t>(h, 80, kVaddr);
    put<uint64_t>(h, 96, kSegSize);
    put<uint64_t>(h, 104, kSegSize);
    put<uint64_t>(h, 112, 0x4000);
    return h;
}

std::vector<uint8_t> segment() {
    std::vector<uint8_t> s(kSegSize);
    for (size_t i = 0; i < s.size(); ++i) s[i] = uint8_t(0xa0 + i);
    return s;
}

// A SELF: header, a digest entry and the segment's entry, the ELF headers, then the bytes.
std::vector<uint8_t> self(uint64_t extraProps) {
    std::vector<uint8_t> f(0x20);
    put<uint32_t>(f, 0, 0x1d3d154f);
    put<uint16_t>(f, 0x18, 2);
    const std::vector<uint8_t> h = elfHeaders(), s = segment();
    const uint64_t dataAt = 0x20 + 2 * 0x20 + h.size();
    put<uint64_t>(f, 0x20, 0x110004), put<uint64_t>(f, 0x28, dataAt), put<uint64_t>(f, 0x30, 8);
    put<uint64_t>(f, 0x40, 0x2804 | extraProps), put<uint64_t>(f, 0x48, dataAt + 8), put<uint64_t>(f, 0x50, s.size());
    f.resize(0x20 + 2 * 0x20);
    f.insert(f.end(), h.begin(), h.end());
    f.insert(f.end(), 8, 0xee); // the digest
    f.insert(f.end(), s.begin(), s.end());
    return f;
}

std::string write(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    return p.string();
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "lpn_elf_test";
    fs::create_directories(dir);

    std::string error;
    auto elf = ps4elf::Elf::load(write(dir / "fself.bin", self(0)), &error);
    CHECK(elf.has_value());
    if (!elf) std::fprintf(stderr, "  %s\n", error.c_str());
    if (elf) {
        CHECK(elf->type() == 0xfe10);
        CHECK(elf->entry() == 0x10);
        CHECK(elf->segments().size() == 1);
        const uint8_t* p = elf->at(kVaddr, kSegSize);
        const std::vector<uint8_t> s = segment();
        CHECK(p && std::memcmp(p, s.data(), s.size()) == 0);
    }

    // The same as a plain ELF loads the same.
    std::vector<uint8_t> plain = elfHeaders();
    plain.resize(kSegOffset);
    const std::vector<uint8_t> s = segment();
    plain.insert(plain.end(), s.begin(), s.end());
    auto plainElf = ps4elf::Elf::load(write(dir / "plain.bin", plain), &error);
    CHECK(plainElf.has_value());
    if (elf && plainElf) CHECK(elf->file().size() == plainElf->file().size());

    error.clear();
    CHECK(!ps4elf::Elf::load(write(dir / "encrypted.bin", self(1u << 1)), &error));
    CHECK(error.find("encrypted") != std::string::npos);
    CHECK(!ps4elf::Elf::load(write(dir / "compressed.bin", self(1u << 3)), &error));

    fs::remove_all(dir);
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    else std::printf("elf_test: all passed\n");
    return failures ? 1 : 0;
}
