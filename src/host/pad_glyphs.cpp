// Built without SDL2, lpn-host has no window and no menus to draw these in.
#ifdef LPN_HAVE_SDL2
#include "host/pad_glyphs.h"

#include <cstdio>
#include <string>
#include <vector>

#define NANOSVG_IMPLEMENTATION
#define NANOSVGRAST_IMPLEMENTATION
#include "nanosvg.h"
#include "nanosvgrast.h"

#include "host/pad_glyphs_data.h"

namespace host {

namespace {

// pad_glyphs_data.h's order: south, east, west, north.
enum Pos { South, East, West, North };

int position(PadStyle style, SDL_GameControllerButton b) {
    bool labels = style == PadStyle::Switch;
#ifdef SDL_HINT_GAMECONTROLLER_USE_BUTTON_LABELS
    labels = labels && SDL_GetHintBoolean(SDL_HINT_GAMECONTROLLER_USE_BUTTON_LABELS, SDL_TRUE);
#endif
    switch (b) {
    case SDL_CONTROLLER_BUTTON_A: return labels ? East : South;
    case SDL_CONTROLLER_BUTTON_B: return labels ? South : East;
    case SDL_CONTROLLER_BUTTON_X: return labels ? North : West;
    case SDL_CONTROLLER_BUTTON_Y: return labels ? West : North;
    default: return -1;
    }
}

struct Cached {
    int style, pos, size;
    SDL_Texture* tex; // null: it failed, do not try again
};
SDL_Renderer* gRen = nullptr;
std::vector<Cached> gCache;

std::vector<uint8_t> pixels(int style, int pos, int size, float zoom) {
    std::string svg = glyph_data::kSvgs[style][pos]; // nsvgParse writes into the text
    NSVGimage* img = nsvgParse(svg.data(), "px", 96.0f);
    if (!img || img->width <= 0) {
        if (img) nsvgDelete(img);
        return {};
    }
    NSVGrasterizer* rast = nsvgCreateRasterizer();
    std::vector<uint8_t> px(size_t(size) * size * 4);
    const float scale = size * zoom / img->width, off = -size * (zoom - 1) / 2;
    nsvgRasterize(rast, img, off, off, scale, px.data(), size, size, size * 4);
    nsvgDeleteRasterizer(rast);
    nsvgDelete(img);
    return px;
}

SDL_Texture* rasterise(SDL_Renderer* ren, int style, int pos, int size) {
    const std::vector<uint8_t> px = pixels(style, pos, size, 1.0f);
    if (px.empty()) return nullptr;
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, size, size);
    if (!tex) return nullptr;
    SDL_UpdateTexture(tex, nullptr, px.data(), size * 4);
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
    return tex;
}

} // namespace

PadStyle padStyle(ButtonIcons setting) {
    switch (setting) {
    case ButtonIcons::PlayStation: return PadStyle::PlayStation;
    case ButtonIcons::Xbox: return PadStyle::Xbox;
    case ButtonIcons::Switch: return PadStyle::Switch;
    case ButtonIcons::Auto: break;
    }
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        if (!SDL_IsGameController(i)) continue;
        switch (SDL_GameControllerTypeForIndex(i)) {
        case SDL_CONTROLLER_TYPE_PS3:
        case SDL_CONTROLLER_TYPE_PS4:
        case SDL_CONTROLLER_TYPE_PS5: return PadStyle::PlayStation;
        case SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_PRO:
#if SDL_VERSION_ATLEAST(2, 24, 0)
        case SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
        case SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
        case SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
#endif
            return PadStyle::Switch;
        default: return PadStyle::Xbox;
        }
    }
    return PadStyle::PlayStation;
}

std::vector<uint8_t> padGlyphPixels(PadStyle style, SDL_GameControllerButton b, int size, float zoom) {
    const int pos = position(style, b);
    if (pos < 0 || size <= 0 || size > 1024) return {};
    return pixels(int(style), pos, size, zoom);
}

bool drawPadGlyph(SDL_Renderer* ren, int x, int y, int size, PadStyle style, SDL_GameControllerButton b) {
    const int pos = position(style, b);
    if (pos < 0 || size <= 0 || size > 512) return false;
    if (ren != gRen) {
        // A new renderer (the window was made again): the old one's textures went with it.
        gCache.clear();
        gRen = ren;
    }
    SDL_Texture* tex = nullptr;
    bool found = false;
    for (const Cached& c : gCache)
        if (c.style == int(style) && c.pos == pos && c.size == size) tex = c.tex, found = true;
    if (!found) {
        tex = rasterise(ren, int(style), pos, size);
        if (!tex) std::fprintf(stderr, "[pad glyphs] cannot draw button %d of style %d\n", pos, int(style));
        gCache.push_back({int(style), pos, size, tex});
    }
    if (!tex) return false;
    SDL_Rect dst{x, y, size, size};
    SDL_RenderCopy(ren, tex, nullptr, &dst);
    return true;
}

} // namespace host
#endif
