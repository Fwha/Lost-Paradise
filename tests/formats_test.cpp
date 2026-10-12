// Tests for the .par and SLLZ readers, using synthetic data (no game files needed).

#include "formats/par.h"
#include "formats/sllz.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

void putBe32(std::vector<uint8_t>& v, size_t at, uint32_t x) {
    for (int i = 0; i < 4; ++i) v[at + i] = uint8_t(x >> (24 - 8 * i));
}

std::vector<uint8_t> bytesOf(const std::string& s) { return {s.begin(), s.end()}; }

void testSllzRoundTrip() {
    std::string text;
    for (int i = 0; i < 200; ++i) text += "kenshiro hokuto shinken " + std::to_string(i % 7);
    const auto input = bytesOf(text);
    const auto packed = formats::sllzCompress(input);
    CHECK(formats::isSllz(packed));
    CHECK(packed.size() < input.size());

    std::vector<uint8_t> out;
    std::string err;
    CHECK(formats::sllzDecompress(packed, out, &err));
    CHECK(out == input);

    const std::vector<uint8_t> empty;
    CHECK(formats::sllzDecompress(formats::sllzCompress(empty), out, &err));
    CHECK(out.empty());
}

void testSllzKnownStream() {
    // Hand-assembled: literals "abc", then a back-reference (distance 3, length 6) -> "abcabcabc".
    std::vector<uint8_t> blob = {'S', 'L', 'L', 'Z', 0, 1, 16, 0, 9, 0, 0, 0, 0, 0, 0, 0,
                                 0x10, 'a', 'b', 'c', 0x23, 0x00};
    blob[12] = uint8_t(blob.size());
    std::vector<uint8_t> out;
    std::string err;
    CHECK(formats::sllzDecompress(blob, out, &err));
    CHECK(std::string(out.begin(), out.end()) == "abcabcabc");

    blob[20] = 0xF3; // distance far beyond the output so far
    CHECK(!formats::sllzDecompress(blob, out, &err));
}

// Build an archive: root "." with file "a.txt" (raw) and folder "sub" holding "b.bin" (SLLZ).
std::vector<uint8_t> buildPar() {
    const auto a = bytesOf("hello");
    const auto bPlain = bytesOf(std::string(100, 'x'));
    const auto b = formats::sllzCompress(bPlain);

    const uint32_t folders = 2, files = 2;
    const uint32_t folderTable = 0x20 + (folders + files) * 64;
    const uint32_t fileTable = folderTable + folders * 32;
    const uint32_t aOff = fileTable + files * 32;
    const uint32_t bOff = aOff + uint32_t(a.size());

    std::vector<uint8_t> v(bOff + b.size(), 0);
    std::memcpy(v.data(), "PARC", 4);
    v[4] = 2;
    v[5] = 1;
    putBe32(v, 0x08, 0x00020001);
    putBe32(v, 0x10, folders);
    putBe32(v, 0x14, folderTable);
    putBe32(v, 0x18, files);
    putBe32(v, 0x1C, fileTable);
    const char* names[] = {".", "sub", "a.txt", "b.bin"};
    for (int i = 0; i < 4; ++i) std::memcpy(&v[0x20 + i * 64], names[i], std::strlen(names[i]));

    // root: 1 child folder at index 1, 1 file at index 0
    putBe32(v, folderTable + 0, 1);
    putBe32(v, folderTable + 4, 1);
    putBe32(v, folderTable + 8, 1);
    putBe32(v, folderTable + 12, 0);
    // sub: no folders, 1 file at index 1
    putBe32(v, folderTable + 32 + 4, 2);
    putBe32(v, folderTable + 32 + 8, 1);
    putBe32(v, folderTable + 32 + 12, 1);

    putBe32(v, fileTable + 4, uint32_t(a.size()));
    putBe32(v, fileTable + 8, uint32_t(a.size()));
    putBe32(v, fileTable + 12, aOff);
    putBe32(v, fileTable + 32 + 0, 0x80000000);
    putBe32(v, fileTable + 32 + 4, uint32_t(bPlain.size()));
    putBe32(v, fileTable + 32 + 8, uint32_t(b.size()));
    putBe32(v, fileTable + 32 + 12, bOff);

    std::memcpy(&v[aOff], a.data(), a.size());
    std::memcpy(&v[bOff], b.data(), b.size());
    return v;
}

void testPar() {
    const auto data = buildPar();
    std::string err;
    auto ar = formats::ParArchive::parse(data, &err);
    CHECK(ar.has_value());
    if (!ar) return;
    CHECK(ar->files().size() == 2);
    CHECK(ar->files()[0].path == "a.txt");
    CHECK(ar->files()[1].path == "sub/b.bin");
    CHECK(ar->files()[1].compressed);

    std::vector<uint8_t> out;
    CHECK(ar->read(ar->files()[0], out, &err));
    CHECK(std::string(out.begin(), out.end()) == "hello");
    CHECK(ar->read(ar->files()[1], out, &err));
    CHECK(out == bytesOf(std::string(100, 'x')));

    auto truncated = data;
    truncated.resize(truncated.size() - 4);
    CHECK(!formats::ParArchive::parse(truncated, &err));
}

} // namespace

int main() {
    testSllzRoundTrip();
    testSllzKnownStream();
    testPar();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all formats tests passed\n");
    return 0;
}
