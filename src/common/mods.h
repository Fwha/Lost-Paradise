// Mods: replacement game files in a mods/ folder in the install folder (beside game/).
// Each folder in mods/ is one mod, laid out like the game folder, e.g.
//   mods/Red Kenshiro/data/2dpar/sprite_e.par
// replaces game/CUSA12776/data/2dpar/sprite_e.par. Files laid out right in mods/
// (mods/data/..., mods/sce_sys/...) count as one more mod, with the lowest priority. When two
// mods hold the same file, the one whose folder name sorts last wins, so "20-foo" beats
// "10-bar". A folder whose name starts with "." or "_" is off. Paths match without regard to
// case. Mods replace whole files: one .dds inside a .par means a whole new .par, and a file
// a mod adds is opened by name but does not show when the game lists a folder.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace host {

constexpr const char* kModsFolder = "mods";

struct Mod {
    std::string name; // the folder's name ("" for loose files right in mods/)
    std::string dir;
    size_t files = 0;
};

class ModIndex {
public:
    // Reads modsDir (missing: no mods). Mods in priority order, lowest first.
    void scan(const std::string& modsDir);
    const std::vector<Mod>& mods() const { return mods_; }
    size_t size() const { return files_.size(); }
    // The mod's file for rel, a path in the game folder ("data/2dpar/sprite_e.par", a leading
    // "/" and "." or ".." parts allowed), or empty when no mod replaces it.
    std::string find(const std::string& rel) const;
    // The mod a host path from find() came from, for logs.
    const Mod* modOf(const std::string& hostPath) const;

private:
    std::vector<Mod> mods_;
    std::unordered_map<std::string, std::string> files_; // key(rel) -> host path
};

// The key a game-relative path is matched by: normalised, '/'-separated, lower case.
std::string modKey(const std::string& rel);

} // namespace host
