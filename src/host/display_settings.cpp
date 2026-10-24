#include "host/display_settings.h"

#include "host/module.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace host {

namespace {

constexpr const char* kFile = "lpn-display.ini";
DisplaySettings gSettings;
uint32_t gActiveW = 0, gActiveH = 0;

const char* modeName(WindowMode m) {
    return m == WindowMode::Borderless ? "borderless" : m == WindowMode::Fullscreen ? "fullscreen" : "windowed";
}

} // namespace

DisplaySettings& displaySettings() { return gSettings; }

void loadDisplaySettings() {
    FILE* f = std::fopen(kFile, "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        char key[64], value[64];
        if (std::sscanf(line, " %63[^= ] = %63s", key, value) != 2) continue;
        if (!std::strcmp(key, "window")) {
            for (WindowMode m : {WindowMode::Windowed, WindowMode::Borderless, WindowMode::Fullscreen})
                if (!std::strcmp(value, modeName(m))) gSettings.window = m;
        } else if (!std::strcmp(key, "resolution")) {
            unsigned w = 0, h = 0;
            if (std::sscanf(value, "%ux%u", &w, &h) == 2 && w >= 320 && h >= 180 && w <= 7680 && h <= 4320)
                gSettings.width = w, gSettings.height = h;
            else gSettings.width = gSettings.height = 0;
        } else if (!std::strcmp(key, "fps")) {
            const int fps = std::atoi(value);
            if (!std::strcmp(value, "uncapped")) gSettings.fps = 0;
            else if (fps >= 10 && fps <= 240) gSettings.fps = fps;
        }
    }
    std::fclose(f);
}

void saveDisplaySettings() {
    FILE* f = std::fopen(kFile, "w");
    if (!f) {
        std::fprintf(stderr, "[display] cannot write %s\n", kFile);
        return;
    }
    std::fprintf(f, "window = %s\n", modeName(gSettings.window));
    if (gSettings.width) std::fprintf(f, "resolution = %ux%u\n", gSettings.width, gSettings.height);
    else std::fprintf(f, "resolution = default\n");
    if (gSettings.fps) std::fprintf(f, "fps = %d\n", gSettings.fps);
    else std::fprintf(f, "fps = uncapped\n");
    std::fclose(f);
}

bool patchRenderResolution(Module& eboot, uint32_t width, uint32_t height) {
    if (!width || !height) return true;
    // The eboot's display setup (0x55d5e0, given argc/argv) stores the render width and height
    // (0x239d5ec/ee) and the output size (0x239d5f0/f2) as 16-bit values: 1600x896 on a base
    // PS4, 1920x1080 on a Pro, 1920x1080 output for both. Each is a `movw $imm16, (reg)`.
    struct Store {
        uint32_t at;
        uint8_t reg;     // ModRM byte
        uint16_t was;
        bool isWidth;
        bool output;
    };
    static const Store kStores[] = {
        {0x55d861, 0x00, 0x780, true, false},  {0x55d866, 0x01, 0x438, false, false}, // PS4 Pro
        {0x55d9f1, 0x00, 0x640, true, false},  {0x55d9f6, 0x01, 0x380, false, false}, // base PS4
        {0x55da09, 0x00, 0x780, true, true},   {0x55da0e, 0x01, 0x438, false, true},  // output
    };
    uint8_t* text = eboot.base();
    for (const Store& s : kStores) {
        const uint8_t* p = text + s.at;
        if (p[0] != 0x66 || p[1] != 0xC7 || p[2] != s.reg || uint16_t(p[3] | p[4] << 8) != s.was) {
            std::fprintf(stderr, "[display] unexpected eboot code at 0x%x: resolution left as the game's\n", s.at);
            return false;
        }
    }
    // Above 1080p the game also outputs at that size (its 4K modes do the same), so the window
    // gets every rendered pixel; at or below it keeps the 1920x1080 output and scales up.
    const bool bigger = width > 1920 || height > 1080;
    for (const Store& s : kStores) {
        if (s.output && !bigger) continue;
        const uint16_t v = uint16_t(s.isWidth ? width : height);
        text[s.at + 3] = uint8_t(v);
        text[s.at + 4] = uint8_t(v >> 8);
    }
    gActiveW = width, gActiveH = height;
    std::fprintf(stderr, "[display] rendering at %ux%u (output %ux%u)\n", width, height, bigger ? width : 1920,
                 bigger ? height : 1080);
    return true;
}

void activeRenderResolution(uint32_t* width, uint32_t* height) {
    *width = gActiveW;
    *height = gActiveH;
}

} // namespace host
