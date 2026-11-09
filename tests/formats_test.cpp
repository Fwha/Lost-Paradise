// Tests for the .par and SLLZ readers, using synthetic data (no game files needed).

#include "formats/gmd.h"
#include "formats/gmt.h"
#include "formats/skeleton.h"

#include <cmath>
#include "formats/par.h"
#include "formats/sllz.h"

#include <cstdio>
#include <cstring>
#include <map>
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
std::vector<uint8_t> buildPar(const std::vector<uint8_t>& a = bytesOf("hello")) {
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

std::vector<uint8_t> fileOf(const formats::ParArchive& ar, const std::string& path) {
    std::vector<uint8_t> out;
    std::string err;
    for (const auto& f : ar.files())
        if (f.path == path) ar.read(f, out, &err);
    return out;
}

// Mods' changes inside an archive, and inside the archive in it (a.txt holds one here).
void testPatchPar() {
    const auto outer = buildPar(buildPar());
    const std::map<std::string, std::vector<uint8_t>> changes = {
        {"sub/b.bin", bytesOf("new b")},
        {"a.txt/a.txt", bytesOf("HI")},
        {"a.txt/nope", bytesOf("y")},
        {"missing.dds", bytesOf("x")},
    };
    std::vector<std::string> unmatched;
    std::string err;
    const auto out = formats::patchPar(outer, changes, &unmatched, &err);
    CHECK(!out.empty());
    CHECK(out.size() % 0x800 == 0);
    CHECK((unmatched == std::vector<std::string>{"a.txt/nope", "missing.dds"}));
    auto ar = formats::ParArchive::parse(out, &err);
    CHECK(ar.has_value());
    if (!ar) return;
    CHECK(fileOf(*ar, "sub/b.bin") == bytesOf("new b"));
    CHECK(!ar->files()[1].compressed);
    const auto innerBytes = fileOf(*ar, "a.txt");
    auto inner = formats::ParArchive::parse(innerBytes, &err);
    CHECK(inner.has_value());
    if (!inner) return;
    CHECK(fileOf(*inner, "a.txt") == bytesOf("HI"));
    CHECK(fileOf(*inner, "sub/b.bin") == bytesOf(std::string(100, 'x')));

    unmatched.clear();
    CHECK(formats::patchPar(outer, {{"other", bytesOf("z")}}, &unmatched, &err).empty());
    CHECK(unmatched.size() == 1);
}

void testGmdVertexLayout() {
    // The three layouts seen in Kenshiro's model, with their declared strides.
    auto a = formats::gmdVertexLayout(0x41fe0ffc3ull, nullptr);
    CHECK(a && a->stride == 40 && a->normal == 20 && a->tangent == 24 && a->uv == 36);
    auto b = formats::gmdVertexLayout(0x41800ffc3ull, nullptr);
    CHECK(b && b->stride == 32 && b->uv == 28);
    auto c = formats::gmdVertexLayout(0x418001fc3ull, nullptr);
    CHECK(c && c->stride == 28 && c->tangent == -1 && c->uv == 24);

    std::string err;
    CHECK(!formats::gmdVertexLayout(0x3 | 0x100000, &err)); // unknown bit
    CHECK(!formats::gmdVertexLayout(0x2, &err));            // not float3 positions
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

void testGmtSampling() {
    formats::GmtTrack<formats::Vec3> loc;
    loc.frames = {0, 10};
    loc.values = {{0, 0, 0}, {10, 20, 30}};
    const auto mid = formats::sampleTrack(loc, 5);
    CHECK(near(mid[0], 5) && near(mid[1], 10) && near(mid[2], 15));
    CHECK(near(formats::sampleTrack(loc, -3)[0], 0));  // clamps before the first key
    CHECK(near(formats::sampleTrack(loc, 99)[2], 30)); // and after the last

    // Opposite-sign quaternions are the same rotation; interpolation must take the short way.
    formats::GmtTrack<formats::Quat> rot;
    rot.frames = {0, 2};
    rot.values = {{0, 0, 0, 1}, {0, 0, 0, -1}};
    const auto q = formats::sampleTrack(rot, 1);
    CHECK(near(std::fabs(q[3]), 1));
}

void testSkeletonMath() {
    // 90 degrees about Y maps +X to -Z, then translate.
    const float s = std::sqrt(0.5f);
    const auto m = formats::matFromTrs({1, 2, 3}, {0, s, 0, s}, {1, 1, 1});
    const auto p = formats::matTransformPoint(m, {1, 0, 0});
    CHECK(near(p[0], 1) && near(p[1], 2) && near(p[2], 2));
    const auto id = formats::matMultiply(m, formats::matIdentity());
    CHECK(id == m);
}

} // namespace

int main() {
    testSllzRoundTrip();
    testSllzKnownStream();
    testPar();
    testPatchPar();
    testGmdVertexLayout();
    testGmtSampling();
    testSkeletonMath();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all formats tests passed\n");
    return 0;
}
