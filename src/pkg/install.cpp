#include "pkg/install.h"

#include "common/game_dir.h"

#include <cerrno>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace pkg {

namespace {

struct Failed {
    std::string why;
};
struct Cancelled {};

void setStep(Progress* pr, const std::string& step) {
    if (!pr) return;
    std::lock_guard lock(pr->mutex);
    pr->step = step;
    pr->file.clear();
}

void setFile(Progress* pr, const std::string& file) {
    if (!pr) return;
    if (pr->cancel) throw Cancelled{};
    std::lock_guard lock(pr->mutex);
    pr->file = file;
}

// Copies a file in pieces, so a big one shows progress and can be cancelled.
void copyFile(const fs::path& from, const fs::path& to, Progress* pr) {
    std::FILE* in = std::fopen(from.string().c_str(), "rb");
    if (!in) throw Failed{"cannot read " + from.string() + ": " + std::strerror(errno)};
    std::FILE* out = std::fopen(to.string().c_str(), "wb");
    if (!out) {
        std::fclose(in);
        throw Failed{"cannot write " + to.string() + ": " + std::strerror(errno)};
    }
    std::vector<char> buf(1 << 20);
    bool ok = true;
    for (size_t n; (n = std::fread(buf.data(), 1, buf.size(), in)) > 0;) {
        if (std::fwrite(buf.data(), 1, n, out) != n) {
            ok = false;
            break;
        }
        if (pr) {
            pr->done += n;
            if (pr->cancel) break;
        }
    }
    ok = !std::ferror(in) && ok;
    std::fclose(in);
    ok = std::fclose(out) == 0 && ok;
    if (pr && pr->cancel) throw Cancelled{};
    if (!ok) throw Failed{"cannot copy " + from.filename().string() + ": " + std::strerror(errno)};
}

uint64_t treeSize(const fs::path& dir) {
    uint64_t n = 0;
    for (const auto& e : fs::recursive_directory_iterator(dir))
        if (e.is_regular_file()) n += e.file_size();
    return n;
}

// from's files put into to (made as needed), over what is there: moved or copied.
void overlay(const fs::path& from, const fs::path& to, bool move, Progress* pr) {
    for (const auto& e : fs::recursive_directory_iterator(from)) {
        const fs::path rel = fs::relative(e.path(), from), dst = to / rel;
        if (e.is_directory()) {
            fs::create_directories(dst);
            continue;
        }
        if (!e.is_regular_file()) continue;
        setFile(pr, rel.string());
        fs::create_directories(dst.parent_path());
        // Removed first: it may be a hard link to the game's own copy, which must not change.
        std::error_code ec;
        fs::remove(dst, ec);
        if (move) {
            const uint64_t size = e.file_size();
            fs::rename(e.path(), dst, ec);
            if (!ec) {
                if (pr) pr->done += size;
                continue;
            }
        }
        copyFile(e.path(), dst, pr);
    }
}

// A copy of a game folder made of hard links (a plain copy where links cannot be made).
void linkTree(const fs::path& from, const fs::path& to, Progress* pr) {
    fs::create_directories(to);
    for (const auto& e : fs::recursive_directory_iterator(from)) {
        const fs::path rel = fs::relative(e.path(), from), dst = to / rel;
        if (e.is_directory()) {
            fs::create_directories(dst);
        } else if (e.is_regular_file()) {
            std::error_code ec;
            fs::create_hard_link(e.path(), dst, ec);
            if (ec) {
                setFile(pr, rel.string());
                copyFile(e.path(), dst, nullptr);
            }
            if (pr && pr->cancel) throw Cancelled{};
        }
    }
}

std::string safe(std::string s) {
    for (char& c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_')) c = '_';
    return s;
}

const host::GameVersion* gameFor(const std::vector<host::GameVersion>& games, const std::string& titleId,
                                 bool baseOnly) {
    const host::GameVersion* pick = nullptr;
    for (const host::GameVersion& g : games) {
        if (g.titleId != titleId || (baseOnly && g.category.rfind("gp", 0) == 0)) continue;
        if (!pick || g.name == titleId) pick = &g;
    }
    return pick;
}

// Builds into <target>.part, renamed to target at the end; the .part is removed if it stops.
template <class F> void build(const fs::path& target, F&& fill) {
    const fs::path part = target.string() + ".part";
    std::error_code ec;
    fs::remove_all(part, ec);
    try {
        fill(part);
        fs::rename(part, target);
    } catch (...) {
        fs::remove_all(part, ec);
        throw;
    }
}

} // namespace

bool applyFolder(const std::string& folder, const std::string& gamesDir, bool move, Progress* pr,
                 std::string& message, std::string& err) {
    try {
        const fs::path src(folder), games(gamesDir);
        const std::string sfo = (src / "sce_sys" / "param.sfo").string();
        const std::string category = host::sfoString(sfo, "CATEGORY"), titleId = host::sfoString(sfo, "TITLE_ID");
        const std::string title = host::sfoString(sfo, "TITLE");
        if (titleId.empty()) throw Failed{"no sce_sys/param.sfo with a title ID in " + src.filename().string()};
        const bool update = category.rfind("gp", 0) == 0, dlc = category == "ac";
        if (!update && !dlc)
            throw Failed{src.filename().string() + " is a full game, not an update or DLC: unpack or copy it into the game folder"};
        const std::vector<host::GameVersion> list = host::findGames(games.parent_path().string());
        const host::GameVersion* game = gameFor(list, titleId, update);
        if (!game) throw Failed{"there is no " + titleId + " game in the game folder to apply it to"};
        if (pr) pr->total = treeSize(src), pr->done = 0;

        if (update) {
            std::string version = host::sfoString(sfo, "APP_VER");
            if (version.empty()) version = host::sfoString(sfo, "VERSION");
            if (version.empty()) version = "update";
            const fs::path target = games / (titleId + "-v" + safe(version));
            if (fs::exists(target)) throw Failed{target.filename().string() + " is already in the game folder: this update was applied before"};
            build(target, [&](const fs::path& part) {
                setStep(pr, "Copying " + game->name);
                linkTree(game->dir, part, pr);
                setStep(pr, "Applying update " + version);
                overlay(src, part, move, pr);
            });
            message = "Applied update " + version + " to " + game->name + ": Play now offers game/" +
                      target.filename().string();
        } else {
            std::string contentId = host::sfoString(sfo, "CONTENT_ID");
            const std::string label = contentId.size() >= 36 ? contentId.substr(20, 16) : src.filename().string();
            const fs::path target = games / (titleId + "-DLC") / safe(label);
            if (fs::exists(target)) throw Failed{"this DLC is already in game/" + titleId + "-DLC"};
            fs::create_directories(target.parent_path());
            build(target, [&](const fs::path& part) {
                setStep(pr, "Adding DLC " + label);
                overlay(src, part, move, pr);
            });
            message = "Added DLC " + (title.empty() ? label : title) + " for " + titleId + " (game/" + titleId +
                      "-DLC). The game does not load DLC yet";
        }
        return true;
    } catch (const Failed& f) {
        err = f.why;
    } catch (const Cancelled&) {
        err = "cancelled";
    } catch (const fs::filesystem_error& e) {
        err = e.what();
    }
    return false;
}

bool installPkg(const std::string& pkgPath, const std::string& gamesDir, Progress* pr, std::string& message,
                std::string& err) {
    Info info;
    if (!readInfo(pkgPath, info, err)) return false;
    const fs::path games(gamesDir);
    const bool update = info.category.rfind("gp", 0) == 0, dlc = info.category == "ac";
    setStep(pr, "Unpacking");
    if (!update && !dlc) {
        const std::string folder = folderName(info);
        if (!extract(pkgPath, (games / folder).string(), pr, err)) return false;
        message = "Unpacked " + (info.title.empty() ? info.titleId : info.title) + " into game/" + folder;
        return true;
    }
    // An update or DLC: unpacked into a hidden folder, then applied from it.
    const fs::path tmp = games / ("." + safe(info.titleId) + "-pkg-unpacking");
    std::error_code ec;
    fs::remove_all(tmp, ec);
    const bool ok = extract(pkgPath, tmp.string(), pr, err) && applyFolder(tmp.string(), gamesDir, true, pr, message, err);
    fs::remove_all(tmp, ec);
    return ok;
}

} // namespace pkg
