#include "host/button_icon_files.h"

#include "host/display_settings.h"
#include "host/kernel.h"

#ifdef LPN_HAVE_SDL2
#include "formats/par.h"
#include "host/pad_glyphs.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>
#endif

namespace host {

#ifndef LPN_HAVE_SDL2
std::string buttonIconFile(const std::string& path) { return path; }
#else

namespace {

namespace fs = std::filesystem;

// Bump when the redrawing changes, so old copies are made again.
constexpr int kVersion = 1;

// The four face buttons, as SDL names them: cross is A, circle B, square X, triangle Y.
constexpr SDL_GameControllerButton kCircle = SDL_CONTROLLER_BUTTON_B, kCross = SDL_CONTROLLER_BUTTON_A,
                                   kSquare = SDL_CONTROLLER_BUTTON_X, kTriangle = SDL_CONTROLLER_BUTTON_Y;
// The sheets' top row, left to right.
constexpr SDL_GameControllerButton kSheetRow[4] = {kCircle, kCross, kSquare, kTriangle};

uint32_t be32(const std::vector<uint8_t>& d, size_t o) {
    return uint32_t(d[o]) << 24 | uint32_t(d[o + 1]) << 16 | uint32_t(d[o + 2]) << 8 | d[o + 3];
}
void putBe32(std::vector<uint8_t>& d, size_t o, uint32_t v) {
    d[o] = uint8_t(v >> 24), d[o + 1] = uint8_t(v >> 16), d[o + 2] = uint8_t(v >> 8), d[o + 3] = uint8_t(v);
}
uint32_t le32(const std::vector<uint8_t>& d, size_t o) {
    uint32_t v;
    std::memcpy(&v, &d[o], 4);
    return v;
}

struct Dds {
    uint32_t w = 0, h = 0;
    bool bgra = false, dxt5 = false;
};
Dds ddsInfo(const std::vector<uint8_t>& d) {
    Dds i;
    if (d.size() < 128 || std::memcmp(d.data(), "DDS ", 4) != 0) return i;
    i.h = le32(d, 12), i.w = le32(d, 16);
    const uint32_t flags = le32(d, 80), bits = le32(d, 88);
    i.bgra = (flags & 0x40) && bits == 32 && le32(d, 92) == 0x00FF0000u && le32(d, 104) == 0xFF000000u &&
             d.size() >= 128 + size_t(i.w) * i.h * 4;
    i.dxt5 = (flags & 0x4) && !std::memcmp(&d[84], "DXT5", 4) && d.size() >= 128 + size_t(i.w) * i.h;
    return i;
}

// RGBA icon into a BGRA image at (x, y), replacing what was there.
void putBgra(std::vector<uint8_t>& d, const Dds& i, int x, int y, int size, const std::vector<uint8_t>& px) {
    for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c) {
            const uint8_t* s = &px[(size_t(r) * size + c) * 4];
            uint8_t* o = &d[128 + ((size_t(y) + r) * i.w + x + c) * 4];
            o[0] = s[2], o[1] = s[1], o[2] = s[0], o[3] = s[3];
        }
}

uint16_t rgb565(const uint8_t* p) { return uint16_t((p[0] >> 3) << 11 | (p[1] >> 2) << 5 | p[2] >> 3); }

// One 4x4 DXT5 block from 16 RGBA pixels: the colours' and alphas' ranges as the end points.
void dxt5Block(const uint8_t px[16][4], uint8_t out[16]) {
    uint8_t amin = 255, amax = 0, lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0};
    for (int i = 0; i < 16; ++i) {
        amin = std::min(amin, px[i][3]), amax = std::max(amax, px[i][3]);
        if (px[i][3] < 8) continue; // a clear pixel's colour does not show
        for (int c = 0; c < 3; ++c) lo[c] = std::min(lo[c], px[i][c]), hi[c] = std::max(hi[c], px[i][c]);
    }
    if (lo[0] > hi[0]) std::memset(lo, 0, 3), std::memset(hi, 0, 3);
    // Alpha: 8 steps from a0 (max) to a1 (min).
    out[0] = amax, out[1] = amin;
    uint64_t abits = 0;
    for (int i = 0; i < 16; ++i) {
        int idx = 0;
        if (amax > amin) {
            const int t = ((amax - px[i][3]) * 7 + (amax - amin) / 2) / (amax - amin); // 0 = a0 .. 7 = a1
            idx = t == 0 ? 0 : t == 7 ? 1 : t + 1;
        }
        abits |= uint64_t(idx) << (3 * i);
    }
    for (int b = 0; b < 6; ++b) out[2 + b] = uint8_t(abits >> (8 * b));
    // Colour: four steps between c0 (bright end) and c1; c0 > c1 keeps the four-colour mode.
    uint16_t c0 = rgb565(hi), c1 = rgb565(lo);
    if (c0 < c1) std::swap(c0, c1), std::swap(lo, hi);
    if (c0 == c1) c1 = c0 > 0 ? uint16_t(c0 - 1) : 0;
    out[8] = uint8_t(c0), out[9] = uint8_t(c0 >> 8), out[10] = uint8_t(c1), out[11] = uint8_t(c1 >> 8);
    uint32_t cbits = 0;
    for (int i = 0; i < 16; ++i) {
        // Where the pixel sits between lo and hi, by its brightest-spread channel sum.
        int num = 0, den = 0;
        for (int c = 0; c < 3; ++c) num += px[i][c] - lo[c], den += hi[c] - lo[c];
        const int t = den > 0 ? std::clamp((num * 3 + den / 2) / den, 0, 3) : 0; // 0 = lo .. 3 = hi
        static const int kIdx[4] = {1, 3, 2, 0}; // lo is c1 (1), hi is c0 (0)
        cbits |= uint32_t(kIdx[t]) << (2 * i);
    }
    for (int b = 0; b < 4; ++b) out[12 + b] = uint8_t(cbits >> (8 * b));
}

// RGBA icon into a DXT5 image at (x, y), both on 4-pixel boundaries.
void putDxt5(std::vector<uint8_t>& d, const Dds& i, int x, int y, int size, const std::vector<uint8_t>& px) {
    for (int by = 0; by < size / 4; ++by)
        for (int bx = 0; bx < size / 4; ++bx) {
            uint8_t block[16][4];
            for (int p = 0; p < 16; ++p)
                std::memcpy(block[p], &px[((size_t(by) * 4 + p / 4) * size + bx * 4 + p % 4) * 4], 4);
            const size_t o = 128 + ((size_t(y / 4 + by) * (i.w / 4)) + x / 4 + bx) * 16;
            dxt5Block(block, &d[o]);
        }
}

// Redraws a DDS's face buttons; false when it is not one of ours or not the expected format.
bool redraw(const std::string& name, std::vector<uint8_t>& dds, PadStyle style) {
    const Dds i = ddsInfo(dds);
    // The font's symbols: 32x32 each.
    static const std::map<std::string, SDL_GameControllerButton> kGlyphs = {
        {"kigou00.dds", kCircle}, {"kigou01.dds", kTriangle}, {"kigou02.dds", kSquare}, {"kigou03.dds", kCross}};
    if (auto g = kGlyphs.find(name); g != kGlyphs.end()) {
        if (!i.bgra || i.w != 32 || i.h != 32) return false;
        const std::vector<uint8_t> px = padGlyphPixels(style, g->second, 32, 1.12f);
        if (px.empty()) return false;
        putBgra(dds, i, 0, 0, 32, px);
        return true;
    }
    // The button sheets: 256x256, 64x64 icons, face buttons along the top.
    if (name == "2d_mr_icon_button_ps4.dds" || name == "2d_cf_icon_button_ps4.dds" || name == "2d_cf_icon_button.dds") {
        if (i.w != 256 || i.h != 256 || !(i.bgra || i.dxt5)) return false;
        for (int c = 0; c < 4; ++c) {
            const std::vector<uint8_t> px = padGlyphPixels(style, kSheetRow[c], 64, 1.12f);
            if (px.empty()) return false;
            if (i.bgra) putBgra(dds, i, c * 64, 0, 64, px);
            else putDxt5(dds, i, c * 64, 0, 64, px);
        }
        return true;
    }
    return false;
}

// The archive with its button icons redrawn: each changed file is stored again, uncompressed, at
// the end, and its entry pointed there. Empty on failure.
std::vector<uint8_t> patchArchive(const std::string& path, PadStyle style) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> par(std::istreambuf_iterator<char>(in), {});
    std::string err;
    auto ar = formats::ParArchive::parse(par, &err);
    if (!ar) {
        std::fprintf(stderr, "[button icons] cannot read %s: %s\n", path.c_str(), err.c_str());
        return {};
    }
    std::vector<std::pair<size_t, std::vector<uint8_t>>> changed;
    for (size_t k = 0; k < ar->files().size(); ++k) {
        const formats::ParFile& f = ar->files()[k];
        std::vector<uint8_t> dds;
        if (f.name.size() < 5 || !(f.name.rfind("kigou0", 0) == 0 || f.name.find("icon_button") != std::string::npos))
            continue;
        if (!ar->read(f, dds, &err)) continue;
        if (redraw(f.name, dds, style)) changed.emplace_back(k, std::move(dds));
    }
    if (changed.empty()) {
        std::fprintf(stderr, "[button icons] no button icons found in %s\n", path.c_str());
        return {};
    }
    std::vector<uint8_t> out = par; // ar views par, which stays as it is
    const uint32_t fileTable = be32(par, 0x1C);
    for (auto& [k, dds] : changed) {
        out.resize((out.size() + 0x7FF) & ~size_t(0x7FF), 0); // the archive keeps files 2 KiB aligned
        const uint64_t off = out.size();
        out.insert(out.end(), dds.begin(), dds.end());
        const size_t e = fileTable + 32 * k;
        putBe32(out, e, be32(out, e) & ~0x80000000u); // not compressed
        putBe32(out, e + 4, uint32_t(dds.size()));
        putBe32(out, e + 8, uint32_t(dds.size()));
        putBe32(out, e + 12, uint32_t(off));
        putBe32(out, e + 20, uint32_t(off >> 32));
    }
    out.resize((out.size() + 0x7FF) & ~size_t(0x7FF), 0);
    std::fprintf(stderr, "[button icons] redrew %zu icon files of %s\n", changed.size(), path.c_str());
    return out;
}

const char* styleKey(PadStyle s) { return s == PadStyle::Xbox ? "xbox" : s == PadStyle::Switch ? "switch" : "playstation"; }

bool endsWith(const std::string& s, const char* tail) {
    const size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

} // namespace

std::string buttonIconFile(const std::string& path) {
    if (!endsWith(path, "fontpar/font.par") && !endsWith(path, "2dpar/sprite_e.par")) return path;
    static std::mutex m;
    static std::map<std::string, std::string> done;
    // Decided once, at the first of these files the game opens, so both archives agree.
    static const PadStyle style = padStyle(displaySettings().buttonIcons);
    if (style == PadStyle::PlayStation) return path;
    std::lock_guard lock(m);
    if (auto it = done.find(path); it != done.end()) return it->second;
    std::string& result = done[path];
    result = path;

    std::error_code ec;
    const fs::path dir = fs::path(process().dataDir) / "button-icons" / (std::string(styleKey(style)) + "-v" + std::to_string(kVersion));
    const fs::path copy = dir / fs::path(path).filename();
    const auto srcTime = fs::last_write_time(path, ec);
    if (!ec && fs::exists(copy, ec) && fs::last_write_time(copy, ec) >= srcTime && !ec) return result = copy.string();

    const std::vector<uint8_t> out = patchArchive(path, style);
    if (out.empty()) return result;
    fs::create_directories(dir, ec);
    const fs::path tmp = copy.string() + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        o.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
        if (!o) {
            std::fprintf(stderr, "[button icons] cannot write %s\n", tmp.c_str());
            return result;
        }
    }
    fs::rename(tmp, copy, ec);
    if (ec) {
        std::fprintf(stderr, "[button icons] cannot write %s: %s\n", copy.c_str(), ec.message().c_str());
        return result;
    }
    return result = copy.string();
}
#endif

} // namespace host
