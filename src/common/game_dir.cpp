#include "common/game_dir.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace host {

namespace {

bool hasEboot(const fs::path& dir) {
    std::error_code ec;
    return fs::is_regular_file(dir / "eboot.bin", ec);
}

// A title ID: four capital letters and five digits.
bool isTitleId(const std::string& s) {
    if (s.size() != 9) return false;
    for (size_t i = 0; i < 9; ++i)
        if (i < 4 ? !(s[i] >= 'A' && s[i] <= 'Z') : !(s[i] >= '0' && s[i] <= '9')) return false;
    return true;
}

uint32_t le32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }

// TITLE_ID from a param.sfo: "\0PSF", key table and data table offsets, entry count, then
// 16-byte entries (key offset u16, format u16, length, max length, data offset).
std::string sfoTitleId(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    const std::string d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto* b = reinterpret_cast<const uint8_t*>(d.data());
    if (d.size() < 20 || std::memcmp(b, "\0PSF", 4) != 0) return {};
    const uint32_t keys = le32(b + 8), data = le32(b + 12), n = le32(b + 16);
    for (uint32_t i = 0; i < n && 20 + 16 * (i + 1) <= d.size(); ++i) {
        const uint8_t* e = b + 20 + 16 * i;
        const size_t key = keys + (e[0] | e[1] << 8), len = le32(e + 4), at = data + le32(e + 12);
        if (key >= d.size() || at + len > d.size()) continue;
        if (std::strcmp(d.c_str() + key, "TITLE_ID") != 0) continue;
        std::string id(d.data() + at, len);
        id.resize(std::strlen(id.c_str()));
        return isTitleId(id) ? id : std::string();
    }
    return {};
}

} // namespace

std::string gameTitleId(const std::string& gameDir) {
    const fs::path dir(gameDir);
    if (auto id = sfoTitleId(dir / "sce_sys" / "param.sfo"); !id.empty()) return id;
    fs::path abs = fs::absolute(dir).lexically_normal();
    if (abs.filename().empty()) abs = abs.parent_path(); // a trailing slash
    if (const std::string name = abs.filename().string(); isTitleId(name)) return name;
    return kGameId;
}

std::string findGameDir(const std::string& installDir) {
    const fs::path base = installDir.empty() ? fs::path(".") : fs::path(installDir);
    const fs::path games = base / kGameFolder;
    if (hasEboot(games / kGameId)) return (games / kGameId).string();
    std::error_code ec;
    std::vector<fs::path> others;
    for (const auto& e : fs::directory_iterator(games, ec)) {
        // A .part folder is a .pkg the launcher is still unpacking (pkg/pkg.h).
        if (e.path().extension() == ".part") continue;
        if (e.is_directory(ec) && hasEboot(e.path())) others.push_back(e.path());
    }
    if (!others.empty()) return std::min_element(others.begin(), others.end())->string();
    return {};
}

} // namespace host
