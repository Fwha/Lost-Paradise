// The Settings window's options and tabs (settings_dialog.h). The controls are in
// settings_dialog_win32.cpp and settings_dialog_gtk.cpp.
#include "launcher/settings_dialog.h"

#include <algorithm>
#include <cmath>

namespace launcher::settings {

using host::DisplaySettings;

namespace {

std::vector<std::string> names(const char* const* n, int count) { return {n, n + count}; }

// A resolution from the ini that the list does not hold is shown after the list's own.
int resIndex(const DisplaySettings& s) {
    for (int i = 0; i < host::kResCount; ++i)
        if (host::kResolutions[i].w == s.width && host::kResolutions[i].h == s.height) return i;
    return host::kResCount;
}
std::vector<std::string> resChoices(const DisplaySettings& s) {
    std::vector<std::string> out;
    for (const host::Resolution& r : host::kResolutions) out.push_back(r.label);
    if (resIndex(s) == host::kResCount) out.push_back(std::to_string(s.width) + "x" + std::to_string(s.height));
    return out;
}
void resSet(DisplaySettings& s, int i) {
    if (i >= 0 && i < host::kResCount) s.width = host::kResolutions[i].w, s.height = host::kResolutions[i].h;
}

int fpsIndex(const DisplaySettings& s) {
    for (int i = 0; i < host::kFpsCount; ++i)
        if (host::kFpsChoices[i] == s.fps) return i;
    return host::kFpsCount;
}
std::vector<std::string> fpsChoices(const DisplaySettings& s) {
    std::vector<std::string> out;
    for (int f : host::kFpsChoices) out.push_back(f ? std::to_string(f) + " FPS" : "Uncapped");
    if (fpsIndex(s) == host::kFpsCount) out.push_back(std::to_string(s.fps) + " FPS");
    return out;
}

int anisoIndex(const DisplaySettings& s) {
    for (int i = 0; i < host::kAnisoCount; ++i)
        if (host::kAnisoChoices[i] == s.anisotropy) return i;
    return 0;
}

const std::vector<Tab> kTabs = {
    {"Display",
     {
         {"Window Mode", "Play in a window, a borderless window or fullscreen.", false,
          [](const DisplaySettings&) { return names(host::kWindowModeNames, 3); },
          [](const DisplaySettings& s) { return int(s.window); },
          [](DisplaySettings& s, int i) { s.window = host::WindowMode(i); }},
         {"Resolution", "The 3D rendering resolution.", true, resChoices, resIndex, resSet},
         {"Aspect Ratio", "Match your monitor's shape: the 3D view widens to fit, the HUD stays 16:9 in the middle.", true,
          [](const DisplaySettings&) { return names(host::kAspectNames, host::kAspectChoices); },
          [](const DisplaySettings& s) { return int(s.aspect); },
          [](DisplaySettings& s, int i) { s.aspect = host::AspectRatio(i); }},
         {"Frame Rate Cap", "The most frames shown per second. Uncapped follows the monitor's refresh rate.", false,
          fpsChoices, fpsIndex,
          [](DisplaySettings& s, int i) {
              if (i >= 0 && i < host::kFpsCount) s.fps = host::kFpsChoices[i];
          }},
         {"VSync", "Show each frame at the monitor's refresh, without tearing. Off can lower input lag.", false, nullptr,
          [](const DisplaySettings& s) { return int(s.vsync); }, [](DisplaySettings& s, int v) { s.vsync = v; }},
         {"FPS Counter", "Show the frame rate in the top right corner of the screen.", false, nullptr,
          [](const DisplaySettings& s) { return int(s.fpsCounter); },
          [](DisplaySettings& s, int v) { s.fpsCounter = v; }},
     }},
    {"Graphics",
     {
         {"DLSS", "NVIDIA DLAA: DLSS anti-aliasing in place of the game's. Needs an RTX card; Off on other GPUs.", true,
          [](const DisplaySettings&) { return std::vector<std::string>{"Off", "DLAA"}; },
          [](const DisplaySettings& s) { return int(s.dlss != host::DlssMode::Off); },
          [](DisplaySettings& s, int i) { s.dlss = i ? host::DlssMode::Dlaa : host::DlssMode::Off; }},
         {"Anisotropic Filtering", "Keep textures sharp at a glancing angle, such as the ground ahead. Up to what your GPU supports.",
          false, [](const DisplaySettings&) { return names(host::kAnisoNames, host::kAnisoCount); }, anisoIndex,
          [](DisplaySettings& s, int i) { s.anisotropy = host::kAnisoChoices[std::clamp(i, 0, host::kAnisoCount - 1)]; }},
         {"Compile Shaders",
          "Build every shader seen in earlier runs before the game starts, so they don't stutter in play. In Background: they build while you play.",
          false, [](const DisplaySettings&) { return std::vector<std::string>{"At Startup", "In Background"}; },
          [](const DisplaySettings& s) { return int(!s.precompileShaders); },
          [](DisplaySettings& s, int i) { s.precompileShaders = i == 0; }},
     }},
    {"Controller",
     {
         {"Button Icons", "Which controller's buttons to show. Auto matches your controller.", true,
          [](const DisplaySettings&) { return names(host::kButtonIconNames, host::kButtonIconChoices); },
          [](const DisplaySettings& s) { return int(s.buttonIcons); },
          [](DisplaySettings& s, int i) { s.buttonIcons = host::ButtonIcons(i); }},
     }},
    {"Extras",
     {
         {"Skip Intro Logos", "Go straight to the title screen when the game starts.", false, nullptr,
          [](const DisplaySettings& s) { return int(s.skipIntro); }, [](DisplaySettings& s, int v) { s.skipIntro = v; }},
     }},
};

} // namespace

const std::vector<Tab>& tabs() { return kTabs; }

std::string tabNote(const Tab& t) {
    std::vector<const char*> later;
    for (const Option& o : t.options)
        if (o.nextStart) later.push_back(o.label);
    if (later.empty()) return {};
    std::string s;
    for (size_t i = 0; i < later.size(); ++i)
        s += (i == 0 ? "" : i + 1 == later.size() ? " and " : ", ") + std::string(later[i]);
    return s + (later.size() == 1 ? " takes" : " take") + " effect the next time the game starts.";
}

Image coverCrop(const Image& art, int w, int h) {
    Image out;
    if (art.w <= 0 || art.h <= 0 || w <= 0 || h <= 0) return out;
    out.w = w, out.h = h;
    out.rgba.resize(size_t(w) * h * 4);
    const double scale = std::max(double(w) / art.w, double(h) / art.h);
    const double x0 = (art.w * scale - w) / 2, y0 = (art.h * scale - h) * 0.4;
    // Each output pixel averages the source pixels under it (the art is larger than the banner).
    const double step = 1 / scale;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double sx = (x + x0) * step, sy = (y + y0) * step;
            const int ax = std::clamp(int(sx), 0, art.w - 1), ay = std::clamp(int(sy), 0, art.h - 1);
            const int bx = std::clamp(int(std::ceil(sx + step)), ax + 1, art.w), by = std::clamp(int(std::ceil(sy + step)), ay + 1, art.h);
            unsigned sum[4] = {};
            for (int j = ay; j < by; ++j)
                for (int i = ax; i < bx; ++i)
                    for (int c = 0; c < 4; ++c) sum[c] += art.rgba[(size_t(j) * art.w + i) * 4 + c];
            const unsigned n = unsigned((bx - ax) * (by - ay));
            for (int c = 0; c < 4; ++c) out.rgba[(size_t(y) * w + x) * 4 + c] = uint8_t(sum[c] / n);
        }
    return out;
}

} // namespace launcher::settings
