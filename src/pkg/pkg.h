// PS4 .pkg extraction, for the launcher's Tools panel and lpn-pkg.
//
// A PKG holds a list of entries (param.sfo, icon0.png and the other sce_sys files, plus keys) and
// a PFS image with the game's files. In a retail PKG the PFS image's key is encrypted with keys
// only a PS4 has, so only fake PKGs (fpkg, made with the homebrew tools' published keys) and
// PKGs with an unencrypted image can be opened here. The image is usually a second, compressed
// (PFSC) PFS image inside the first: pfs_image.dat, whose uroot folder is the game folder.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace pkg {

struct Info {
    std::string contentId; // e.g. UP0700-CUSA12776_00-HOKUTOGAAPP00000
    std::string titleId;   // CUSA12776
    std::string category;  // param.sfo CATEGORY: gd a game, gp a patch, ac add-on content
    std::string title;     // param.sfo TITLE
    bool fake = false;     // its keys open with the fake keys (else a retail PKG, which cannot be opened)
    uint64_t imageSize = 0;
};

// Shared with the thread running extract.
struct Progress {
    std::atomic<uint64_t> done{0}, total{0}; // bytes of game files
    std::atomic<bool> cancel{false};
    std::mutex mutex;
    std::string file; // the file being written, relative to the output folder
};

// Reads the header and entries; false with err saying why (not a PKG, unreadable).
bool readInfo(const std::string& path, Info& info, std::string& err);

// The folder name the game goes into under game/: the title ID, with -UPDATE for a patch and
// -DLC for add-on content.
std::string folderName(const Info& info);

// Extracts the PKG's files into outDir (which must not exist yet): the PFS image's files and
// the entries' sce_sys files. Writes into outDir.part and renames it at the end, so a cancelled or
// failed extraction leaves no half folder behind. False with err saying why.
bool extract(const std::string& path, const std::string& outDir, Progress* progress, std::string& err);

} // namespace pkg
