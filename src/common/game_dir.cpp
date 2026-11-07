#include "common/game_dir.h"

#include <algorithm>
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

} // namespace

std::string findGameDir(const std::string& installDir) {
    const fs::path base = installDir.empty() ? fs::path(".") : fs::path(installDir);
    const fs::path games = base / kGameFolder;
    if (hasEboot(games / kGameId)) return (games / kGameId).string();
    std::error_code ec;
    std::vector<fs::path> others;
    for (const auto& e : fs::directory_iterator(games, ec))
        if (e.is_directory(ec) && hasEboot(e.path())) others.push_back(e.path());
    if (!others.empty()) return std::min_element(others.begin(), others.end())->string();
    return {};
}

} // namespace host
