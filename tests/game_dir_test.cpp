// Tests for the title ID lookup that picks each version's save folder (synthetic param.sfo files).

#include "common/game_dir.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        if ((a) != (b)) {                                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            ++failures;                                                                             \
        }                                                                                           \
    } while (0)

void put32(std::vector<uint8_t>& v, size_t at, uint32_t x) {
    for (int i = 0; i < 4; ++i) v[at + i] = uint8_t(x >> (8 * i));
}

// A param.sfo with CATEGORY then TITLE_ID, both utf-8 strings.
void writeSfo(const fs::path& dir, const std::string& titleId) {
    const std::vector<std::pair<std::string, std::string>> entries = {{"CATEGORY", "gd"}, {"TITLE_ID", titleId}};
    std::string keys, data;
    std::vector<uint8_t> v(20 + 16 * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        const size_t e = 20 + 16 * i;
        v[e] = uint8_t(keys.size()), v[e + 1] = uint8_t(keys.size() >> 8);
        v[e + 2] = 0x04, v[e + 3] = 0x02;
        put32(v, e + 4, uint32_t(entries[i].second.size() + 1));
        put32(v, e + 8, 16);
        put32(v, e + 12, uint32_t(data.size()));
        keys += entries[i].first + '\0';
        data += entries[i].second;
        data.resize(data.size() + 16 - entries[i].second.size(), '\0');
    }
    while (keys.size() % 4) keys += '\0';
    std::memcpy(v.data(), "\0PSF", 4);
    put32(v, 4, 0x101);
    put32(v, 8, uint32_t(v.size()));
    put32(v, 12, uint32_t(v.size() + keys.size()));
    put32(v, 16, uint32_t(entries.size()));
    fs::create_directories(dir / "sce_sys");
    std::ofstream f(dir / "sce_sys" / "param.sfo", std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size()));
    f << keys << data;
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "lpn-game-dir-test";
    fs::remove_all(root);

    // param.sfo wins over the folder's name.
    writeSfo(root / "renamed", "CUSA12781");
    CHECK_EQ(host::gameTitleId((root / "renamed").string()), std::string("CUSA12781"));

    // Without one, a folder named like a title ID gives it, with or without a trailing slash.
    fs::create_directories(root / "CUSA12781");
    CHECK_EQ(host::gameTitleId((root / "CUSA12781").string()), std::string("CUSA12781"));
    CHECK_EQ(host::gameTitleId((root / "CUSA12781").string() + "/"), std::string("CUSA12781"));

    // Otherwise the version the patches are written for.
    fs::create_directories(root / "my dump");
    CHECK_EQ(host::gameTitleId((root / "my dump").string()), std::string(host::kGameId));

    fs::remove_all(root);
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    else std::printf("game_dir_test: all passed\n");
    return failures ? 1 : 0;
}
