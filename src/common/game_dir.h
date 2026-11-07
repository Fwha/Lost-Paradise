// Where the game goes: a game/ folder in the install folder (beside launch.sh or the launcher
// AppImage), holding the dump's folder, e.g. game/CUSA12776 with eboot.bin and sce_module/ in it.
#pragma once

#include <string>
#include <vector>

namespace host {

constexpr const char* kGameFolder = "game";
constexpr const char* kGameId = "CUSA12776"; // the version the host's patches are written for

// A game folder under game/, one per version: a dump, or one an update was applied to
// (pkg/install.h makes those as <title ID>-v<APP_VER>).
struct GameVersion {
    std::string dir;      // its path
    std::string name;     // the folder's name
    std::string titleId;  // gameTitleId
    std::string title;    // param.sfo TITLE
    std::string version;  // param.sfo APP_VER (e.g. 01.05), else VERSION
    std::string category; // param.sfo CATEGORY: gd a game, gp one with an update applied
};

// The folders under <installDir>/game that hold an eboot.bin: game/CUSA12776 first, then by name.
// Left out: .part folders (a .pkg being unpacked), unpacked updates on their own (-UPDATE, -patch:
// only the files the update changes) and hidden folders.
std::vector<GameVersion> findGames(const std::string& installDir);

// The first of findGames, or empty when there is none.
std::string findGameDir(const std::string& installDir);

// A string from a param.sfo file (empty when it or the key is missing).
std::string sfoString(const std::string& sfoPath, const char* key);

// The dump's title ID (e.g. CUSA12776): TITLE_ID from sce_sys/param.sfo, else the folder's name
// when it looks like one, else kGameId. Each version keeps its saves under its own title ID.
std::string gameTitleId(const std::string& gameDir);

} // namespace host
