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

} // namespace

// "\0PSF", key table and data table offsets, entry count, then 16-byte entries (key offset u16,
// format u16, length, max length, data offset).
std::string sfoString(const std::string& path, const char* want) {
    std::ifstream f(path, std::ios::binary);
    const std::string d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto* b = reinterpret_cast<const uint8_t*>(d.data());
    if (d.size() < 20 || std::memcmp(b, "\0PSF", 4) != 0) return {};
    const uint32_t keys = le32(b + 8), data = le32(b + 12), n = le32(b + 16);
    for (uint32_t i = 0; i < n && 20 + 16 * size_t(i + 1) <= d.size(); ++i) {
        const uint8_t* e = b + 20 + 16 * i;
        const size_t key = size_t(keys) + (e[0] | e[1] << 8), len = le32(e + 4), at = size_t(data) + le32(e + 12);
        if (key >= d.size() || at + len > d.size()) continue;
        if (std::strcmp(d.c_str() + key, want) != 0) continue;
        std::string s(d.data() + at, len);
        s.resize(std::strlen(s.c_str()));
        return s;
    }
    return {};
}

namespace {

std::string sfoTitleId(const fs::path& path) {
    const std::string id = sfoString(path.string(), "TITLE_ID");
    return isTitleId(id) ? id : std::string();
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

std::vector<GameVersion> findGames(const std::string& installDir) {
    const fs::path base = installDir.empty() ? fs::path(".") : fs::path(installDir);
    const fs::path games = base / kGameFolder;
    std::error_code ec;
    std::vector<GameVersion> out;
    auto endsWith = [](const std::string& s, const char* tail) {
        const size_t n = std::strlen(tail);
        return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
    };
    for (const auto& e : fs::directory_iterator(games, ec)) {
        const std::string name = e.path().filename().string();
        // A .part folder is a .pkg the launcher is still unpacking (pkg/pkg.h).
        if (name.empty() || name[0] == '.' || endsWith(name, ".part") || endsWith(name, "-UPDATE") ||
            endsWith(name, "-patch"))
            continue;
        if (!e.is_directory(ec) || !hasEboot(e.path())) continue;
        const std::string sfo = (e.path() / "sce_sys" / "param.sfo").string();
        GameVersion g{e.path().string(), name, gameTitleId(e.path().string()), sfoString(sfo, "TITLE"),
                      sfoString(sfo, "APP_VER"), sfoString(sfo, "CATEGORY")};
        if (g.version.empty()) g.version = sfoString(sfo, "VERSION");
        out.push_back(g);
    }
    std::sort(out.begin(), out.end(), [](const GameVersion& a, const GameVersion& b) {
        const bool ka = a.name == kGameId, kb = b.name == kGameId;
        return ka != kb ? ka : a.name < b.name;
    });
    return out;
}

std::string findGameDir(const std::string& installDir) {
    const std::vector<GameVersion> g = findGames(installDir);
    return g.empty() ? std::string() : g.front().dir;
}

} // namespace host
