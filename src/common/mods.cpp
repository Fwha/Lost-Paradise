#include "common/mods.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace host {

namespace fs = std::filesystem;

std::string modKey(const std::string& rel) {
    std::string s = fs::path(rel).lexically_normal().generic_string();
    while (!s.empty() && s.front() == '/') s.erase(0, 1);
    while (!s.empty() && s.back() == '/') s.pop_back();
    if (s == "." || s.rfind("../", 0) == 0 || s == "..") return {};
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

void ModIndex::scan(const std::string& modsDir) {
    mods_.clear();
    files_.clear();
    std::error_code ec;
    if (!fs::is_directory(modsDir, ec)) return;

    // mods/data and mods/sce_sys are the loose files' folders, not mods.
    auto looseFolder = [](const std::string& n) { return modKey(n) == "data" || modKey(n) == "sce_sys"; };
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(modsDir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.empty() || n[0] == '.' || n[0] == '_' || looseFolder(n) || !e.is_directory(ec)) continue;
        names.push_back(n);
    }
    std::sort(names.begin(), names.end());

    // Lowest priority first: the loose files in mods/data and mods/sce_sys, then the mods by
    // name, each one replacing what came before.
    mods_.push_back({"", fs::path(modsDir).string()});
    for (const auto& n : names) mods_.push_back({n, (fs::path(modsDir) / n).string()});

    for (size_t i = 0; i < mods_.size(); ++i) {
        Mod& m = mods_[i];
        const fs::path root = m.dir;
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            const fs::path p = it->path();
            if (it.depth() == 0) {
                const std::string n = p.filename().string();
                const bool skip = i == 0 ? !looseFolder(n) : (n.empty() || n[0] == '.' || n[0] == '_');
                if (skip) {
                    if (it->is_directory(ec)) it.disable_recursion_pending();
                    continue;
                }
            }
            if (!it->is_regular_file(ec)) continue;
            const std::string key = modKey(p.lexically_relative(root).generic_string());
            if (key.empty()) continue;
            files_[key] = p.string();
            ++m.files;
        }
        ec.clear();
    }
    if (mods_.front().files == 0) mods_.erase(mods_.begin());
}

std::string ModIndex::find(const std::string& rel) const {
    if (files_.empty()) return {};
    auto it = files_.find(modKey(rel));
    return it == files_.end() ? std::string() : it->second;
}

const Mod* ModIndex::modOf(const std::string& hostPath) const {
    const Mod* best = nullptr;
    for (const Mod& m : mods_) {
        const std::string d = (fs::path(m.dir) / "").string();
        if (hostPath.compare(0, d.size(), d) == 0 && (!best || m.dir.size() > best->dir.size())) best = &m;
    }
    return best;
}

} // namespace host
