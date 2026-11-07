#include "common/settings_file.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace host {

namespace {

const char* modeKey(WindowMode m) {
    return m == WindowMode::Borderless ? "borderless" : m == WindowMode::Fullscreen ? "fullscreen" : "windowed";
}

const char* const kDlssKeys[kDlssModes] = {"off", "dlaa", "quality", "balanced", "performance"};
const char* const kButtonIconKeys[kButtonIconChoices] = {"playstation", "xbox", "switch", "auto"};
const char* const kAspectKeys[kAspectChoices] = {"16:9", "16:10", "21:9"};

} // namespace

double aspectValue(AspectRatio a) {
    return a == AspectRatio::Wide16x10 ? 16.0 / 10 : a == AspectRatio::Ultrawide21x9 ? 64.0 / 27 : 16.0 / 9;
}

void renderSize(const DisplaySettings& s, uint32_t* width, uint32_t* height) {
    *width = s.width, *height = s.height;
    if (s.aspect == AspectRatio::Wide16x9) return;
    const uint32_t h = s.height ? s.height : 896;
    uint32_t w = uint32_t(h * aspectValue(s.aspect) / 32 + 0.5) * 32;
    // The game packs its display size into 12 bits a side (display_settings.cpp).
    w = std::min<uint32_t>(w, 4064);
    *width = w, *height = h;
}

const char* dlssModeName(DlssMode m) {
    static const char* const names[kDlssModes] = {"Off", "DLAA", "Quality", "Balanced", "Performance"};
    return names[int(m)];
}

void readSettings(const std::string& path, DisplaySettings& s) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        char key[64], value[64];
        if (std::sscanf(line, " %63[^= ] = %63s", key, value) != 2) continue;
        if (!std::strcmp(key, "window")) {
            for (WindowMode m : {WindowMode::Windowed, WindowMode::Borderless, WindowMode::Fullscreen})
                if (!std::strcmp(value, modeKey(m))) s.window = m;
        } else if (!std::strcmp(key, "resolution")) {
            unsigned w = 0, h = 0;
            if (std::sscanf(value, "%ux%u", &w, &h) == 2 && w >= 320 && h >= 180 && w <= 7680 && h <= 4320)
                s.width = w, s.height = h;
            else s.width = s.height = 0;
        } else if (!std::strcmp(key, "fps")) {
            const int fps = std::atoi(value);
            if (!std::strcmp(value, "uncapped")) s.fps = 0;
            else if (fps >= 10 && fps <= 240) s.fps = fps;
        } else if (!std::strcmp(key, "fps_counter")) {
            s.fpsCounter = !std::strcmp(value, "on");
        } else if (!std::strcmp(key, "dlss")) {
            for (int i = 0; i < kDlssModes; ++i)
                if (!std::strcmp(value, kDlssKeys[i])) s.dlss = DlssMode(i);
        } else if (!std::strcmp(key, "skip_intro")) {
            s.skipIntro = !std::strcmp(value, "on");
        } else if (!std::strcmp(key, "button_icons")) {
            for (int i = 0; i < kButtonIconChoices; ++i)
                if (!std::strcmp(value, kButtonIconKeys[i])) s.buttonIcons = ButtonIcons(i);
        } else if (!std::strcmp(key, "aspect")) {
            for (int i = 0; i < kAspectChoices; ++i)
                if (!std::strcmp(value, kAspectKeys[i])) s.aspect = AspectRatio(i);
        }
    }
    std::fclose(f);
}

bool writeSettings(const std::string& path, const DisplaySettings& s) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fprintf(f, "window = %s\n", modeKey(s.window));
    if (s.width) std::fprintf(f, "resolution = %ux%u\n", s.width, s.height);
    else std::fprintf(f, "resolution = default\n");
    if (s.fps) std::fprintf(f, "fps = %d\n", s.fps);
    else std::fprintf(f, "fps = uncapped\n");
    std::fprintf(f, "fps_counter = %s\n", s.fpsCounter ? "on" : "off");
    std::fprintf(f, "dlss = %s\n", kDlssKeys[int(s.dlss)]);
    std::fprintf(f, "skip_intro = %s\n", s.skipIntro ? "on" : "off");
    std::fprintf(f, "button_icons = %s\n", kButtonIconKeys[int(s.buttonIcons)]);
    std::fprintf(f, "aspect = %s\n", kAspectKeys[int(s.aspect)]);
    return std::fclose(f) == 0;
}

} // namespace host
