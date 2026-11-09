// Mods: replacement game files in a mods/ folder in the install folder (beside game/).
// Each folder in mods/ is one mod, laid out like the game folder, e.g.
//   mods/Red Kenshiro/data/2dpar/sprite_e.par     replaces the game's data/2dpar/sprite_e.par
//   mods/Red Kenshiro/data/2dpar/sprite_e.par/2d_mr_icon_button_ps4.dds
//                                                 replaces only that file inside sprite_e.par
// A folder named like an archive holds files to change inside it, by their paths in it; an
// archive in an archive works the same way (ui_e.par/inner.par/x.dds). Files laid out right in
// mods/ (mods/data/..., mods/sce_sys/...) count as one more mod, with the lowest priority. When
// two mods hold the same file, the one whose folder name sorts last wins, so "20-foo" beats
// "10-bar"; mods that change different files in one archive both apply. A folder whose name
// starts with "." or "_" is off. Paths match without regard to case. A file a mod adds is
// opened by name but does not show when the game lists a folder, and cannot be added to an
// archive (only files already in it are changed).
#pragma once

#include <string>
#include <map>
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
    size_t size() const { return files_.size(); }                   // whole files replaced
    size_t archives() const { return patches_.size(); }             // archives changed inside
    // The mod's file for rel, a path in the game folder ("data/2dpar/sprite_e.par", a leading
    // "/" and "." or ".." parts allowed), or empty when no mod replaces it.
    std::string find(const std::string& rel) const;
    // Files to change inside an archive: path in the archive (lower case) -> the mod's file.
    using Patches = std::map<std::string, std::string>;
    // The changes a mod makes inside rel (an archive in the game folder), or null.
    const Patches* patches(const std::string& rel) const;
    // The mod a host path from find() or patches() came from, for logs.
    const Mod* modOf(const std::string& hostPath) const;

private:
    std::vector<Mod> mods_;
    std::unordered_map<std::string, std::string> files_; // key(rel) -> host path
    std::unordered_map<std::string, Patches> patches_;   // key(archive) -> its changes
};

// The key a game-relative path is matched by: normalised, '/'-separated, lower case.
std::string modKey(const std::string& rel);

} // namespace host
