// Putting what a .pkg or an unpacked folder holds into the game folder (the launcher's Tools):
//   a game     -> game/<title ID>
//   an update  -> game/<title ID>-v<APP_VER>: a copy of the game it is for (hard links, so it
//                 takes no room for the files the update leaves alone) with the update's files
//                 over it. The game as it was stays, and Play lists both versions.
//   DLC        -> game/<title ID>-DLC/<content label>, beside the game it is for.
// The game an update or DLC is for is the one in game/ with the same title ID.
#pragma once

#include "pkg/pkg.h"

#include <string>

namespace pkg {

// An unpacked update or DLC folder (sce_sys/param.sfo CATEGORY gp or ac) applied to its game.
// move: take the folder's files rather than copy them (a folder made from a .pkg). message says
// what was done; false with err saying why not.
bool applyFolder(const std::string& folder, const std::string& gamesDir, bool move, Progress* progress,
                 std::string& message, std::string& err);

// A .pkg unpacked: a game into game/<title ID>, an update or DLC applied as applyFolder does.
bool installPkg(const std::string& pkgPath, const std::string& gamesDir, Progress* progress, std::string& message,
                std::string& err);

} // namespace pkg
