// Mods (common/mods.h): which host file an /app0 path opens. An archive the mods change files
// in is opened as a copy with those files in it, made at the first open and kept in
// <data>/mod-archives/<key>/<its path>, the key a hash of every file it is made from, so the
// copy is made again when one of them changes.
#include "host/kernel.h"

#include "common/mods.h"
#include "formats/par.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

namespace host {

namespace {

namespace fs = std::filesystem;

ModIndex gMods; // filled by loadMods before the game starts, read only after

const char* modName(const std::string& hostPath) {
    const Mod* m = gMods.modOf(hostPath);
    return m && !m->name.empty() ? m->name.c_str() : "mods/";
}

bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), {});
    return bool(in) || in.eof();
}

// FNV-1a over where each source is, its size and its time.
struct Hash {
    uint64_t h = 0xcbf29ce484222325ull;
    void add(const void* p, size_t n) {
        for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 0x100000001b3ull;
    }
    void addFile(const std::string& path) {
        add(path.data(), path.size() + 1);
        std::error_code ec;
        const uint64_t size = fs::file_size(path, ec);
        const auto t = fs::last_write_time(path, ec).time_since_epoch().count();
        add(&size, sizeof size), add(&t, sizeof t);
    }
};

// The copy of base (the game's archive, or a mod's replacement of it) with the changes in it.
std::string patchedArchive(const std::string& rel, const std::string& base, const ModIndex::Patches& patches) {
    Hash h;
    const int version = 1; // bump when the copies are made differently
    h.add(&version, sizeof version);
    h.addFile(base);
    for (const auto& [inner, file] : patches) h.add(inner.data(), inner.size() + 1), h.addFile(file);
    char key[17];
    std::snprintf(key, sizeof key, "%016llx", (unsigned long long)h.h);
    const fs::path copy = fs::path(process().dataDir) / "mod-archives" / key / fs::path(rel).relative_path();
    std::error_code ec;
    if (fs::is_regular_file(copy, ec)) return copy.string();

    std::vector<uint8_t> par;
    if (!readFile(base, par)) {
        std::fprintf(stderr, "[mods] cannot read %s\n", base.c_str());
        return {};
    }
    std::map<std::string, std::vector<uint8_t>> changes;
    for (const auto& [inner, file] : patches) {
        std::vector<uint8_t> data;
        if (!readFile(file, data)) {
            std::fprintf(stderr, "[mods] cannot read %s\n", file.c_str());
            continue;
        }
        changes.emplace(inner, std::move(data));
    }
    std::vector<std::string> unmatched;
    std::string err;
    const std::vector<uint8_t> out = formats::patchPar(par, changes, &unmatched, &err);
    for (const auto& u : unmatched)
        std::fprintf(stderr, "[mods] %s: no %s in it to replace (from %s); a mod can only change files already there\n",
                     rel.c_str(), u.c_str(), modName(patches.at(u)));
    if (out.empty()) {
        std::fprintf(stderr, "[mods] %s: nothing changed%s%s\n", rel.c_str(), err.empty() ? "" : ": ", err.c_str());
        return {};
    }
    fs::create_directories(copy.parent_path(), ec);
    const fs::path tmp = copy.string() + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        o.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
        if (!o) {
            std::fprintf(stderr, "[mods] cannot write %s\n", tmp.string().c_str());
            return {};
        }
    }
    fs::rename(tmp, copy, ec);
    if (ec) {
        std::fprintf(stderr, "[mods] cannot write %s: %s\n", copy.string().c_str(), ec.message().c_str());
        return {};
    }
    std::fprintf(stderr, "[mods] %s: %zu files changed inside it\n", rel.c_str(), changes.size() - unmatched.size());
    return copy.string();
}

} // namespace

std::string modFile(const std::string& rel) {
    const ModIndex::Patches* patches = gMods.patches(rel);
    std::string whole = gMods.find(rel);
    if (patches) {
        static std::mutex m;
        static std::map<std::string, std::string> done; // rel -> copy ("" when it failed)
        std::lock_guard lock(m);
        auto it = done.find(rel);
        if (it == done.end())
            it = done.emplace(rel, patchedArchive(rel, whole.empty() ? process().gameDir + rel : whole, *patches)).first;
        if (!it->second.empty()) {
            if (process().traceFiles) std::fprintf(stderr, "[mods] /app0%s with changes from mods\n", rel.c_str());
            return it->second;
        }
    }
    if (!whole.empty() && process().traceFiles) std::fprintf(stderr, "[mods] /app0%s from %s\n", rel.c_str(), modName(whole));
    return whole;
}

std::string gameFile(const std::string& rel) {
    std::string m = modFile(rel);
    return m.empty() ? process().gameDir + rel : m;
}

void loadMods(const std::string& modsDir) {
    gMods.scan(modsDir);
    if (gMods.mods().empty()) return;
    std::fprintf(stderr, "[mods] %zu mods in %s: %zu files replaced, %zu archives changed inside\n", gMods.mods().size(),
                 modsDir.c_str(), gMods.size(), gMods.archives());
    for (const Mod& m : gMods.mods())
        std::fprintf(stderr, "[mods]   %s (%zu files)\n", m.name.empty() ? "loose files in mods/" : m.name.c_str(), m.files);
}

} // namespace host
