#include "host/menu_font.h"

#ifdef LPN_HAVE_SDL2
#include <stb_truetype.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>

namespace host::menufont {

namespace {

struct Font {
    std::vector<unsigned char> data;
    stbtt_fontinfo info{};
    bool ok = false;
};
Font gRegular, gBold;
bool gLoaded = false;

std::string fcMatch(const char* pattern) {
#ifdef _WIN32
    (void)pattern; // no fontconfig; running it would flash a console window per call
    return {};
#endif
    std::string cmd = std::string("fc-match -f '%{file}' '") + pattern + "' 2>/dev/null";
    std::string out;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[512];
        while (size_t n = std::fread(buf, 1, sizeof buf, p)) out.append(buf, n);
        pclose(p);
    }
    return out;
}

bool load(Font& f, const std::string& path) {
    if (path.empty()) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    f.data.assign(std::istreambuf_iterator<char>(in), {});
    const int off = stbtt_GetFontOffsetForIndex(f.data.data(), 0);
    f.ok = off >= 0 && stbtt_InitFont(&f.info, f.data.data(), off);
    if (f.ok) std::fprintf(stderr, "[menu font] %s\n", path.c_str());
    return f.ok;
}

// Arial first, then fonts with Arial's metrics, then fontconfig's own pick for "Arial".
bool loadFirst(Font& f, bool bold, const char* env) {
    if (env && *env) return load(f, env);
#ifdef _WIN32
    // Windows' own Arial, in %WINDIR%\Fonts.
    const char* win = std::getenv("WINDIR");
    const std::string fonts = std::string(win && *win ? win : "C:\\Windows") + "\\Fonts\\";
    if (load(f, fonts + (bold ? "arialbd.ttf" : "arial.ttf"))) return true;
    return load(f, fonts + (bold ? "segoeuib.ttf" : "segoeui.ttf"));
#endif
    static const char* const kDirs[] = {"/usr/share/fonts/truetype/msttcorefonts/", "/usr/share/fonts/TTF/",
                                        "/usr/share/fonts/truetype/", "/usr/share/fonts/corefonts/",
                                        "/usr/share/fonts/microsoft/", "/usr/local/share/fonts/"};
    static const char* const kRegular[] = {"Arial.ttf", "arial.ttf"};
    static const char* const kBold[] = {"Arial_Bold.ttf", "arialbd.ttf", "Arial-Bold.ttf"};
    for (const char* d : kDirs)
        for (const char* n : bold ? std::vector<const char*>(std::begin(kBold), std::end(kBold))
                                  : std::vector<const char*>(std::begin(kRegular), std::end(kRegular)))
            if (std::filesystem::exists(std::string(d) + n) && load(f, std::string(d) + n)) return true;
    for (const char* family : {"Arial", "Liberation Sans", "Arimo"}) {
        const std::string path = fcMatch((std::string(family) + (bold ? ":bold" : "")).c_str());
        // fontconfig always answers with something; only take a sans that stands in for Arial.
        const std::string name = std::filesystem::path(path).filename().string();
        bool good = false;
        for (const char* k : {"rial", "Liberation", "Arimo"})
            if (name.find(k) != std::string::npos) good = true;
        if (good && load(f, path)) return true;
    }
    static const char* const kFallback[][2] = {
        {"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"},
        {"/usr/share/fonts/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"},
        {"/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"},
        {"/usr/share/fonts/TTF/LiberationSans-Regular.ttf", "/usr/share/fonts/TTF/LiberationSans-Bold.ttf"},
    };
    for (const auto& p : kFallback)
        if (load(f, p[bold ? 1 : 0])) return true;
    return false;
}

void ensureLoaded() {
    if (gLoaded) return;
    gLoaded = true;
    if (!loadFirst(gRegular, false, std::getenv("LPN_MENU_FONT")))
        std::fprintf(stderr, "[menu font] no Arial or Liberation Sans found: menus use the pixel font\n");
    if (!loadFirst(gBold, true, std::getenv("LPN_MENU_FONT_BOLD"))) gBold.ok = false;
}

const Font& pick(bool bold) { return bold && gBold.ok ? gBold : gRegular; }

std::vector<int> codepoints(const std::string& s) {
    std::vector<int> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int cp = c, n = 1;
        if (c >= 0xF0) cp = c & 0x07, n = 4;
        else if (c >= 0xE0) cp = c & 0x0F, n = 3;
        else if (c >= 0xC0) cp = c & 0x1F, n = 2;
        if (i + n > s.size()) break;
        for (int k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

float measure(const Font& f, const std::vector<int>& cps, float scale) {
    float x = 0;
    for (size_t i = 0; i < cps.size(); ++i) {
        int adv, lsb;
        stbtt_GetCodepointHMetrics(&f.info, cps[i], &adv, &lsb);
        x += adv * scale;
        if (i + 1 < cps.size()) x += stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]) * scale;
    }
    return x;
}

struct Tex {
    SDL_Texture* tex = nullptr;
    int w = 0, h = 0;
};
std::map<std::tuple<bool, int, std::string>, Tex> gCache;
SDL_Renderer* gCacheRen = nullptr;

void clearCache() {
    for (auto& [k, t] : gCache) SDL_DestroyTexture(t.tex);
    gCache.clear();
}

Tex render(SDL_Renderer* ren, const Font& f, int size, const std::string& s) {
    const auto cps = codepoints(s);
    const float scale = stbtt_ScaleForPixelHeight(&f.info, float(size));
    int ascent, descent, gap;
    stbtt_GetFontVMetrics(&f.info, &ascent, &descent, &gap);
    const int baseline = int(std::lround(ascent * scale));
    const int w = int(std::ceil(measure(f, cps, scale))) + 2, h = size + 2;
    std::vector<unsigned char> alpha(size_t(w) * size_t(h), 0);
    float x = 1;
    for (size_t i = 0; i < cps.size(); ++i) {
        int adv, lsb, x0, y0, x1, y1;
        stbtt_GetCodepointHMetrics(&f.info, cps[i], &adv, &lsb);
        const float shift = x - std::floor(x);
        stbtt_GetCodepointBitmapBoxSubpixel(&f.info, cps[i], scale, scale, shift, 0, &x0, &y0, &x1, &y1);
        const int gw = x1 - x0, gh = y1 - y0;
        if (gw > 0 && gh > 0) {
            std::vector<unsigned char> g(size_t(gw) * size_t(gh));
            stbtt_MakeCodepointBitmapSubpixel(&f.info, g.data(), gw, gh, gw, scale, scale, shift, 0, cps[i]);
            const int ox = int(std::floor(x)) + x0, oy = baseline + y0 + 1;
            for (int yy = 0; yy < gh; ++yy)
                for (int xx = 0; xx < gw; ++xx) {
                    const int px = ox + xx, py = oy + yy;
                    if (px < 0 || py < 0 || px >= w || py >= h) continue;
                    unsigned char& d = alpha[size_t(py) * size_t(w) + size_t(px)];
                    d = std::max(d, g[size_t(yy) * size_t(gw) + size_t(xx)]);
                }
        }
        x += adv * scale;
        if (i + 1 < cps.size()) x += stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]) * scale;
    }
    std::vector<uint32_t> argb(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) argb[i] = (uint32_t(alpha[i]) << 24) | 0xFFFFFFu;
    Tex t;
    t.w = w, t.h = h;
    t.tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, w, h);
    if (t.tex) {
        SDL_UpdateTexture(t.tex, nullptr, argb.data(), w * 4);
        SDL_SetTextureBlendMode(t.tex, SDL_BLENDMODE_BLEND);
    }
    return t;
}

} // namespace

bool available() {
    ensureLoaded();
    return gRegular.ok;
}

int width(const std::string& s, int size, bool bold) {
    if (!available()) return 0;
    const Font& f = pick(bold);
    return int(std::ceil(measure(f, codepoints(s), stbtt_ScaleForPixelHeight(&f.info, float(size)))));
}

std::string fit(const std::string& s, int size, int maxW, bool bold) {
    if (width(s, size, bold) <= maxW) return s;
    std::string cut = s;
    while (!cut.empty()) {
        // Drop one whole UTF-8 character: its continuation bytes, then the byte that starts it.
        while (!cut.empty()) {
            const unsigned char c = static_cast<unsigned char>(cut.back());
            cut.pop_back();
            if ((c & 0xC0) != 0x80) break;
        }
        while (!cut.empty() && cut.back() == ' ') cut.pop_back();
        if (width(cut + "...", size, bold) <= maxW) return cut + "...";
    }
    return "";
}

void draw(SDL_Renderer* ren, int x, int y, int size, const std::string& s, bool bold) {
    if (s.empty() || size <= 0 || !available()) return;
    if (ren != gCacheRen) {
        clearCache();
        gCacheRen = ren;
    }
    if (gCache.size() > 400) clearCache(); // menus show few distinct strings; start over now and then
    bold = bold && gBold.ok;
    auto key = std::make_tuple(bold, size, s);
    auto it = gCache.find(key);
    if (it == gCache.end()) it = gCache.emplace(key, render(ren, pick(bold), size, s)).first;
    const Tex& t = it->second;
    if (!t.tex) return;
    Uint8 r, g, b, a;
    SDL_GetRenderDrawColor(ren, &r, &g, &b, &a);
    SDL_SetTextureColorMod(t.tex, r, g, b);
    SDL_SetTextureAlphaMod(t.tex, a);
    SDL_Rect dst{x - 1, y - 1, t.w, t.h};
    SDL_RenderCopy(ren, t.tex, nullptr, &dst);
}

} // namespace host::menufont
#endif
