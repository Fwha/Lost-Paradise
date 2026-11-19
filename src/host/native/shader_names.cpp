#include "host/native/shader_names.h"

#include "formats/par.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace host::native {

namespace {

constexpr uint8_t kStart[4] = {0xff, 0x03, 0xeb, 0xbe}; // s_mov_b32 vcc_hi, <literal>
constexpr char kFooter[] = "OrbShdr";
constexpr size_t kFooterLen = sizeof kFooter - 1;
constexpr const char* kArchives[] = {"sh_devil_ps4.par", "mars_shader_ps4.par", "sh_ptc_ps4.par", "sh_ptc_all_ps4.par"};

std::shared_mutex gMutex;
std::unordered_map<uint64_t, std::string> gNames; // filled once, then read only
std::atomic<bool> gReady{false};

struct Seen {
    std::array<uint8_t, 64> head{};
    std::string name;
};
std::mutex gSeenMutex;
std::unordered_map<uint64_t, Seen> gSeen;
const std::string kUnknown = "?";

uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

bool readFile(const std::filesystem::path& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

// Every program in one archive entry, under its name.
void indexEntry(const std::vector<uint8_t>& d, const std::string& name, std::unordered_map<uint64_t, std::vector<std::string>>& out) {
    for (size_t o = 0; o + 4 <= d.size(); o += 4) {
        if (std::memcmp(&d[o], kStart, 4) != 0) continue;
        const size_t len = keyLength(&d[o], std::min(d.size() - o, kMaxShaderBytes));
        if (!len) continue;
        auto& names = out[fnv(&d[o], len)];
        if (names.empty() || names.back() != name) names.push_back(name);
        o += len - len % 4 - 4;
    }
}

void build(std::string dir) {
    std::unordered_map<uint64_t, std::vector<std::string>> found;
    size_t entries = 0;
    for (const char* a : kArchives) {
        std::vector<uint8_t> bytes;
        if (!readFile(std::filesystem::path(dir) / a, bytes)) continue;
        std::string error;
        auto par = formats::ParArchive::parse(bytes, &error);
        if (!par) {
            std::fprintf(stderr, "[native] %s: %s\n", a, error.c_str());
            continue;
        }
        std::vector<uint8_t> data;
        for (const auto& f : par->files()) {
            if (!par->read(f, data, &error)) continue;
            indexEntry(data, f.name, found);
            ++entries;
        }
    }
    std::unordered_map<uint64_t, std::string> names;
    names.reserve(found.size());
    for (auto& [k, v] : found) names[k] = v.size() > 1 ? v[0] + " +" + std::to_string(v.size() - 1) : v[0];
    {
        std::unique_lock lock(gMutex);
        gNames = std::move(names);
    }
    gReady = true;
    std::fprintf(stderr, "[native] %zu shader programs named from %zu archive entries\n", gNames.size(), entries);
}

} // namespace

size_t keyLength(const uint8_t* code, size_t n) {
    if (n < kFooterLen) return 0;
    const uint8_t* f = std::search(code, code + n, kFooter, kFooter + kFooterLen);
    return f != code + n ? size_t(f - code) + kFooterLen : 0;
}

uint64_t shaderKey(const uint8_t* code, size_t n) { return fnv(code, n); }

void indexShaderNames(const std::string& gameDir) {
    std::thread(build, (std::filesystem::path(gameDir) / "data" / "shader").string()).detach();
}

bool shaderNamesReady() { return gReady; }

const std::string* shaderName(uint64_t key) {
    if (!gReady) return nullptr;
    std::shared_lock lock(gMutex);
    auto it = gNames.find(key);
    return it == gNames.end() ? nullptr : &it->second;
}

const std::string& shaderNameAt(uint64_t addr, ReadGuest read) {
    if (!addr || !gReady) return kUnknown;
    std::array<uint8_t, 64> head{};
    if (!read(addr, head.data(), head.size())) return kUnknown;
    std::lock_guard lock(gSeenMutex);
    auto it = gSeen.find(addr);
    if (it != gSeen.end() && it->second.head == head) return it->second.name;
    // Read in steps until the footer turns up.
    std::vector<uint8_t> code;
    size_t len = 0;
    for (size_t n = 1024; !len && n <= kMaxShaderBytes; n *= 4) {
        code.resize(n);
        if (!read(addr, code.data(), n)) break;
        len = keyLength(code.data(), n);
    }
    const std::string* name = len ? shaderName(shaderKey(code.data(), len)) : nullptr;
    Seen& s = gSeen[addr];
    s.head = head;
    s.name = name ? *name : kUnknown;
    return s.name;
}

} // namespace host::native
