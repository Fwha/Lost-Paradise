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
constexpr int kVersion = 4;

// The four face buttons, as SDL names them: cross is A, circle B, square X, triangle Y.
constexpr SDL_GameControllerButton kCircle = SDL_CONTROLLER_BUTTON_B, kCross = SDL_CONTROLLER_BUTTON_A,
                                   kSquare = SDL_CONTROLLER_BUTTON_X, kTriangle = SDL_CONTROLLER_BUTTON_Y;

// Where a texture holds a face button: the rectangle is cleared and the icon drawn to fill it.
struct Spot {
    int x, y, w, h;
    SDL_GameControllerButton b;
    bool white = false; // only the letter or symbol, in white (the game's flash over a prompt)
};
struct Sheet {
    const char* name;
    uint32_t w, h;
    float zoom; // the art leaves a margin round its button; this much larger fills the spot
    std::vector<Spot> spots;
};
// The 64x64 face buttons along the top of a sheet, left to right.
std::vector<Spot> topRow(std::initializer_list<SDL_GameControllerButton> bs) {
    std::vector<Spot> out;
    int x = 0;
    for (SDL_GameControllerButton b : bs) out.push_back({x, 0, 64, 64, b}), x += 64;
    return out;
}
const std::vector<Sheet>& sheets() {
    static const std::vector<Sheet> s = {
        // fontpar/font.par: the font's symbols, which the menus' Confirm/Back bar and text use.
        {"kigou00.dds", 32, 32, 1.12f, {{0, 0, 32, 32, kCircle}}},
        {"kigou01.dds", 32, 32, 1.12f, {{0, 0, 32, 32, kTriangle}}},
        {"kigou02.dds", 32, 32, 1.12f, {{0, 0, 32, 32, kSquare}}},
        {"kigou03.dds", 32, 32, 1.12f, {{0, 0, 32, 32, kCross}}},
        // 2dpar/sprite_e.par: the HUD's button sheets.
        {"2d_mr_icon_button_ps4.dds", 256, 256, 1.12f, topRow({kCircle, kCross, kSquare, kTriangle})},
        {"2d_cf_icon_button_ps4.dds", 256, 256, 1.12f, topRow({kCircle, kCross, kSquare, kTriangle})},
        {"2d_cf_icon_button.dds", 256, 256, 1.12f, topRow({kCircle, kCross, kSquare, kTriangle})},
        // 2dpar/ui_e.par, surfboard/pjcm_action_button.par: the fights' prompts (finishers, Heat
        // actions). jm_btn and jm_qte3 are whole buttons; jm_qte and jm_qte2 hold the symbols
        // the game lays over its own black button, jm_qte2's drawn squashed for the tilted ones.
        {"2d_jm_btn.dds", 64, 64, 1.2f, {{0, 0, 64, 64, kCircle}}},
        {"2d_jm_qte3.dds", 256, 64, 1.2f, topRow({kCross, kTriangle, kSquare})},
        {"2d_jm_qte.dds", 256, 256, 1.12f, [] {
             std::vector<Spot> v = topRow({kCircle, kCross, kSquare, kTriangle});
             for (Spot w : {Spot{150, 178, 44, 36, kCircle}, Spot{206, 178, 44, 36, kCross},
                            Spot{150, 217, 44, 36, kTriangle}, Spot{206, 219, 44, 36, kSquare}})
                 w.white = true, v.push_back(w);
             return v;
         }()},
        {"2d_jm_qte2.dds", 256, 128, 1.12f,
         {{150, 2, 44, 36, kCircle}, {206, 2, 44, 36, kCross}, {150, 40, 44, 36, kTriangle}, {206, 42, 44, 36, kSquare}}},
        // surfboard/pjcm_btl_ougibutton_add.par: the finisher and Heat buttons over an enemy and
        // in the button-press scenes (BC7; a 128x128 cell each, the button 84 across).
        {"2d_mr_ougiadd_01.dds", 512, 256, 1.1f,
         {{16, 16, 96, 96, kCircle}, {144, 16, 96, 96, kCross}, {272, 16, 96, 96, kTriangle}, {400, 16, 96, 96, kSquare}}},
        // pjcm_btl_ovhd_gauge.par and pjcm_btl_hikou.par: the circle beside an enemy's health bar
        // when a finisher is ready.
        {"2d_mr_ovhd_gauge02.dds", 128, 128, 1.2f, {{80, 0, 47, 47, kCircle}}},
        {"2d_mr_btl_hikou01.dds", 128, 128, 1.2f, {{0, 0, 46, 46, kCircle}}},
        // pjcm_btl_life_gauge.par: the R2 that starts rage mode, beside the player's health bar.
        {"2d_btl_life_gauge01.dds", 512, 128, 1.0f, {{398, 56, 32, 32, kRightTrigger}}},
    };
    return s;
}
// Archives inside the archives that hold some of the above.
bool isContainer(const std::string& name) {
    return name == "pjcm_action_button.par" || name == "pjcm_kyaba.par" || name == "pjcm_btl_ougibutton_add.par" ||
           name == "pjcm_btl_ovhd_gauge.par" || name == "pjcm_btl_hikou.par" || name == "pjcm_btl_life_gauge.par";
}

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
    bool bgra = false, dxt5 = false, bc7 = false;
    size_t data = 128; // where the pixels start: after the DX10 header extension for BC7
};
Dds ddsInfo(const std::vector<uint8_t>& d) {
    Dds i;
    if (d.size() < 128 || std::memcmp(d.data(), "DDS ", 4) != 0) return i;
    i.h = le32(d, 12), i.w = le32(d, 16);
    const uint32_t flags = le32(d, 80), bits = le32(d, 88);
    i.bgra = (flags & 0x40) && bits == 32 && le32(d, 92) == 0x00FF0000u && le32(d, 104) == 0xFF000000u &&
             d.size() >= 128 + size_t(i.w) * i.h * 4;
    const bool blocks = (flags & 0x4) && i.w % 4 == 0 && i.h % 4 == 0;
    i.dxt5 = blocks && !std::memcmp(&d[84], "DXT5", 4) && d.size() >= 128 + size_t(i.w) * i.h;
    if (blocks && !std::memcmp(&d[84], "DX10", 4) && d.size() >= 148) {
        const uint32_t dxgi = le32(d, 128); // 98 BC7_UNORM, 99 BC7_UNORM_SRGB
        i.bc7 = (dxgi == 98 || dxgi == 99) && d.size() >= 148 + size_t(i.w) * i.h;
        i.data = 148;
    }
    return i;
}

void rgb565(uint16_t c, uint8_t* o) {
    const int r = c >> 11, g = (c >> 5) & 63, b = c & 31;
    o[0] = uint8_t(r << 3 | r >> 2), o[1] = uint8_t(g << 2 | g >> 4), o[2] = uint8_t(b << 3 | b >> 2);
}
uint16_t to565(const uint8_t* p) { return uint16_t((p[0] >> 3) << 11 | (p[1] >> 2) << 5 | p[2] >> 3); }

// One 4x4 DXT5 block to 16 RGBA pixels.
void dxt5Decode(const uint8_t* in, uint8_t px[16][4]) {
    uint8_t a[8] = {in[0], in[1]};
    if (a[0] > a[1])
        for (int k = 1; k < 7; ++k) a[k + 1] = uint8_t(((7 - k) * a[0] + k * a[1]) / 7);
    else {
        for (int k = 1; k < 5; ++k) a[k + 1] = uint8_t(((5 - k) * a[0] + k * a[1]) / 5);
        a[6] = 0, a[7] = 255;
    }
    uint64_t abits = 0;
    for (int b = 0; b < 6; ++b) abits |= uint64_t(in[2 + b]) << (8 * b);
    uint8_t c[4][3];
    const uint16_t c0 = uint16_t(in[8] | in[9] << 8), c1 = uint16_t(in[10] | in[11] << 8);
    rgb565(c0, c[0]), rgb565(c1, c[1]);
    for (int k = 0; k < 3; ++k) {
        c[2][k] = uint8_t((2 * c[0][k] + c[1][k]) / 3); // DXT5 colours are always the four-colour kind
        c[3][k] = uint8_t((c[0][k] + 2 * c[1][k]) / 3);
    }
    const uint32_t cbits = uint32_t(in[12] | in[13] << 8 | in[14] << 16 | uint32_t(in[15]) << 24);
    for (int i = 0; i < 16; ++i) {
        std::memcpy(px[i], c[(cbits >> (2 * i)) & 3], 3);
        px[i][3] = a[(abits >> (3 * i)) & 7];
    }
}

// 16 RGBA pixels to one DXT5 block: the colours' and alphas' ranges as the end points.
void dxt5Encode(const uint8_t px[16][4], uint8_t out[16]) {
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
    // Colour: four steps between c0 (the bright end) and c1.
    uint16_t c0 = to565(hi), c1 = to565(lo);
    if (c0 < c1) std::swap(c0, c1), std::swap(lo, hi);
    if (c0 == c1) c1 = c0 > 0 ? uint16_t(c0 - 1) : 0;
    out[8] = uint8_t(c0), out[9] = uint8_t(c0 >> 8), out[10] = uint8_t(c1), out[11] = uint8_t(c1 >> 8);
    uint32_t cbits = 0;
    for (int i = 0; i < 16; ++i) {
        int num = 0, den = 0;
        for (int c = 0; c < 3; ++c) num += px[i][c] - lo[c], den += hi[c] - lo[c];
        const int t = den > 0 ? std::clamp((num * 3 + den / 2) / den, 0, 3) : 0; // 0 = lo .. 3 = hi
        static const int kIdx[4] = {1, 3, 2, 0};                               // lo is c1, hi is c0
        cbits |= uint32_t(kIdx[t]) << (2 * i);
    }
    for (int b = 0; b < 4; ++b) out[12 + b] = uint8_t(cbits >> (8 * b));
}

// 16 RGBA pixels to one BC7 block in mode 6: one pair of RGBA end points (the block's range),
// sixteen steps between them.
void bc7Encode(const uint8_t px[16][4], uint8_t out[16]) {
    int lo[4] = {255, 255, 255, 255}, hi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 16; ++i)
        for (int c = 0; c < 4; ++c) lo[c] = std::min(lo[c], int(px[i][c])), hi[c] = std::max(hi[c], int(px[i][c]));
    // 7 bits a channel and a shared low bit (0 for lo, 1 for hi, which keeps both ends reachable).
    int e[2][4];
    for (int c = 0; c < 4; ++c) e[0][c] = lo[c] >> 1, e[1][c] = hi[c] >> 1;
    int p[2] = {0, 1};
    auto value = [&](int k, int c) { return e[k][c] << 1 | p[k]; };
    static const int kWeight[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
    int idx[16];
    for (int i = 0; i < 16; ++i) {
        int best = 0, bestErr = 1 << 30;
        for (int k = 0; k < 16; ++k) {
            int err = 0;
            for (int c = 0; c < 4; ++c) {
                const int v = (value(0, c) * (64 - kWeight[k]) + value(1, c) * kWeight[k] + 32) >> 6;
                err += (v - px[i][c]) * (v - px[i][c]);
            }
            if (err < bestErr) best = k, bestErr = err;
        }
        idx[i] = best;
    }
    // The first pixel's index is stored in 3 bits, so it must be under 8: swap the ends if not.
    if (idx[0] >= 8) {
        std::swap(e[0], e[1]), std::swap(p[0], p[1]);
        for (int& k : idx) k = 15 - k;
    }
    std::memset(out, 0, 16);
    int bit = 0;
    auto put = [&](uint32_t v, int n) {
        for (int k = 0; k < n; ++k, ++bit)
            if (v >> k & 1) out[bit / 8] |= uint8_t(1 << (bit % 8));
    };
    put(1u << 6, 7); // mode 6
    for (int c = 0; c < 4; ++c) put(uint32_t(e[0][c]), 7), put(uint32_t(e[1][c]), 7);
    put(uint32_t(p[0]), 1), put(uint32_t(p[1]), 1);
    for (int i = 0; i < 16; ++i) put(uint32_t(idx[i]), i == 0 ? 3 : 4);
}

// The icon at w x h: drawn square at the larger side, then stretched.
std::vector<uint8_t> icon(PadStyle style, SDL_GameControllerButton b, int w, int h, float zoom) {
    const int n = std::max(w, h);
    const std::vector<uint8_t> sq = padGlyphPixels(style, b, n, zoom);
    if (sq.empty() || (w == n && h == n)) return sq;
    std::vector<uint8_t> out(size_t(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float sx = (x + 0.5f) * n / w - 0.5f, sy = (y + 0.5f) * n / h - 0.5f;
            const int x0 = std::clamp(int(sx), 0, n - 1), y0 = std::clamp(int(sy), 0, n - 1);
            const int x1 = std::min(x0 + 1, n - 1), y1 = std::min(y0 + 1, n - 1);
            const float fx = std::clamp(sx - x0, 0.0f, 1.0f), fy = std::clamp(sy - y0, 0.0f, 1.0f);
            for (int c = 0; c < 4; ++c) {
                auto at = [&](int xx, int yy) { return float(sq[(size_t(yy) * n + xx) * 4 + c]); };
                const float v = (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
                out[(size_t(y) * w + x) * 4 + c] = uint8_t(std::lround(v));
            }
        }
    return out;
}

// Redraws a DDS's face buttons; false when it is not one of ours or not the expected format.
bool redraw(const std::string& name, std::vector<uint8_t>& dds, PadStyle style) {
    const Sheet* sheet = nullptr;
    for (const Sheet& s : sheets())
        if (name == s.name) sheet = &s;
    if (!sheet) return false;
    const Dds i = ddsInfo(dds);
    if (i.w != sheet->w || i.h != sheet->h || !(i.bgra || i.dxt5 || i.bc7)) {
        std::fprintf(stderr, "[button icons] %s is not the expected %ux%u texture\n", name.c_str(), sheet->w, sheet->h);
        return false;
    }
    // The image as RGBA.
    std::vector<uint8_t> img(size_t(i.w) * i.h * 4);
    const uint32_t bw = i.w / 4;
    if (i.bgra) {
        for (size_t p = 0; p < size_t(i.w) * i.h; ++p) {
            const uint8_t* s = &dds[128 + p * 4];
            img[p * 4] = s[2], img[p * 4 + 1] = s[1], img[p * 4 + 2] = s[0], img[p * 4 + 3] = s[3];
        }
    } else if (i.bc7) {
        // Not decoded: only whole blocks inside the spots are written.
        for (const Spot& s : sheet->spots)
            if (s.x % 4 || s.y % 4 || s.w % 4 || s.h % 4) return false;
    } else {
        for (uint32_t by = 0; by < i.h / 4; ++by)
            for (uint32_t bx = 0; bx < bw; ++bx) {
                uint8_t px[16][4];
                dxt5Decode(&dds[128 + (size_t(by) * bw + bx) * 16], px);
                for (int k = 0; k < 16; ++k) std::memcpy(&img[((size_t(by) * 4 + k / 4) * i.w + bx * 4 + k % 4) * 4], px[k], 4);
            }
    }
    std::vector<bool> dirty(size_t(bw) * (i.h / 4));
    for (const Spot& s : sheet->spots) {
        std::vector<uint8_t> px = icon(style, s.b, s.w, s.h, sheet->zoom);
        if (px.empty()) return false;
        if (s.white) {
            // The letter is the brightest part of the art, the button round it darker.
            int top = 1;
            for (size_t p = 0; p < px.size(); p += 4) top = std::max({top, int(px[p]), int(px[p + 1]), int(px[p + 2])});
            for (size_t p = 0; p < px.size(); p += 4) {
                const int v = std::max({px[p], px[p + 1], px[p + 2]});
                px[p + 3] = uint8_t(px[p + 3] * std::clamp((v * 100 / top - 60) * 255 / 25, 0, 255) / 255);
                px[p] = px[p + 1] = px[p + 2] = 255;
            }
        }
        for (int y = 0; y < s.h; ++y)
            std::memcpy(&img[((size_t(s.y) + y) * i.w + s.x) * 4], &px[size_t(y) * s.w * 4], size_t(s.w) * 4);
        for (int by = s.y / 4; by <= (s.y + s.h - 1) / 4; ++by)
            for (int bx = s.x / 4; bx <= (s.x + s.w - 1) / 4; ++bx) dirty[size_t(by) * bw + bx] = true;
    }
    if (i.bgra) {
        for (size_t p = 0; p < size_t(i.w) * i.h; ++p) {
            uint8_t* o = &dds[128 + p * 4];
            o[0] = img[p * 4 + 2], o[1] = img[p * 4 + 1], o[2] = img[p * 4], o[3] = img[p * 4 + 3];
        }
    } else {
        // Only the blocks the icons touch are encoded again, so the rest keeps its quality.
        for (uint32_t by = 0; by < i.h / 4; ++by)
            for (uint32_t bx = 0; bx < bw; ++bx) {
                if (!dirty[size_t(by) * bw + bx]) continue;
                uint8_t px[16][4];
                for (int k = 0; k < 16; ++k) std::memcpy(px[k], &img[((size_t(by) * 4 + k / 4) * i.w + bx * 4 + k % 4) * 4], 4);
                uint8_t* block = &dds[i.data + (size_t(by) * bw + bx) * 16];
                if (i.bc7) bc7Encode(px, block);
                else dxt5Encode(px, block);
            }
    }
    return true;
}

// The archive with its button icons redrawn (and those of the archives in it that hold some):
// each changed file is stored again, uncompressed, at the end, and its entry pointed there.
// Empty when nothing in it changed.
std::vector<uint8_t> patchArchive(const std::vector<uint8_t>& par, PadStyle style, const std::string& what, int* count) {
    std::string err;
    auto ar = formats::ParArchive::parse(par, &err);
    if (!ar) {
        std::fprintf(stderr, "[button icons] cannot read %s: %s\n", what.c_str(), err.c_str());
        return {};
    }
    std::vector<std::pair<size_t, std::vector<uint8_t>>> changed;
    for (size_t k = 0; k < ar->files().size(); ++k) {
        const formats::ParFile& f = ar->files()[k];
        const bool icons = std::any_of(sheets().begin(), sheets().end(), [&](const Sheet& s) { return f.name == s.name; });
        if (!icons && !isContainer(f.name)) continue;
        std::vector<uint8_t> data;
        if (!ar->read(f, data, &err)) continue;
        if (icons) {
            if (redraw(f.name, data, style)) changed.emplace_back(k, std::move(data)), ++*count;
        } else if (std::vector<uint8_t> inner = patchArchive(data, style, what + "/" + f.name, count); !inner.empty()) {
            changed.emplace_back(k, std::move(inner));
        }
    }
    if (changed.empty()) return {};
    std::vector<uint8_t> out = par; // ar views par, which stays as it is
    const uint32_t fileTable = be32(par, 0x1C);
    for (auto& [k, data] : changed) {
        out.resize((out.size() + 0x7FF) & ~size_t(0x7FF), 0); // the archives keep files 2 KiB aligned
        const uint64_t off = out.size();
        out.insert(out.end(), data.begin(), data.end());
        const size_t e = fileTable + 32 * k;
        putBe32(out, e, be32(out, e) & ~0x80000000u); // not compressed
        putBe32(out, e + 4, uint32_t(data.size()));
        putBe32(out, e + 8, uint32_t(data.size()));
        putBe32(out, e + 12, uint32_t(off));
        putBe32(out, e + 20, uint32_t(off >> 32));
    }
    out.resize((out.size() + 0x7FF) & ~size_t(0x7FF), 0);
    return out;
}

const char* styleKey(PadStyle s) { return s == PadStyle::Xbox ? "xbox" : s == PadStyle::Switch ? "switch" : "playstation"; }

bool endsWith(const std::string& s, const char* tail) {
    const size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

} // namespace

std::string buttonIconFile(const std::string& path) {
    if (!endsWith(path, "fontpar/font.par") && !endsWith(path, "2dpar/sprite_e.par") && !endsWith(path, "2dpar/ui_e.par"))
        return path;
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

    std::ifstream in(path, std::ios::binary);
    const std::vector<uint8_t> par(std::istreambuf_iterator<char>(in), {});
    int count = 0;
    const std::vector<uint8_t> out = patchArchive(par, style, path, &count);
    if (out.empty()) {
        std::fprintf(stderr, "[button icons] no button icons found in %s\n", path.c_str());
        return result;
    }
    std::fprintf(stderr, "[button icons] redrew %d icon textures of %s\n", count, path.c_str());
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
