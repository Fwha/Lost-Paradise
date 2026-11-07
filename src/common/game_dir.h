// Where the game goes: a game/ folder in the install folder (beside launch.sh or the launcher
// AppImage), holding the dump's folder, e.g. game/CUSA12776 with eboot.bin and sce_module/ in it.
#pragma once

#include <string>

namespace host {

constexpr const char* kGameFolder = "game";
constexpr const char* kGameId = "CUSA12776"; // the version the host's patches are written for

// The dump under <installDir>/game: game/CUSA12776 when it holds an eboot.bin, else the first
// other folder there that does. Empty when there is none.
std::string findGameDir(const std::string& installDir);

// The dump's title ID (e.g. CUSA12776): TITLE_ID from sce_sys/param.sfo, else the folder's name
// when it looks like one, else kGameId. Each version keeps its saves under its own title ID.
std::string gameTitleId(const std::string& gameDir);

} // namespace host
