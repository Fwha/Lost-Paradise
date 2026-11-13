// Tests for lpn-display.ini's upscaler settings (common/settings_file.h): the older dlss key, a
// round trip, and the scene size an upscale ratio gives.

#include "common/settings_file.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        if ((a) != (b)) {                                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            ++failures;                                                                             \
        }                                                                                           \
    } while (0)

host::DisplaySettings readText(const fs::path& p, const std::string& text) {
    std::ofstream(p) << text;
    host::DisplaySettings s;
    host::readSettings(p.string(), s);
    return s;
}

} // namespace

int main() {
    const fs::path ini = fs::temp_directory_path() / "lpn-settings-test.ini";

    // Older files: dlss = dlaa / quality / ... becomes the DLSS upscaler at that quality.
    host::DisplaySettings s = readText(ini, "dlss = dlaa\n");
    CHECK_EQ(s.upscaler, host::Upscaler::Dlss);
    CHECK_EQ(s.upscaleQuality, host::UpscaleQuality::Native);
    s = readText(ini, "dlss = performance\n");
    CHECK_EQ(s.upscaler, host::Upscaler::Dlss);
    CHECK_EQ(s.upscaleQuality, host::UpscaleQuality::Performance);
    s = readText(ini, "dlss = off\n");
    CHECK_EQ(s.upscaler, host::Upscaler::Off);

    // A round trip of the new keys.
    s = host::DisplaySettings{};
    s.upscaler = host::Upscaler::Fsr;
    s.upscaleQuality = host::UpscaleQuality::Balanced;
    CHECK_EQ(host::writeSettings(ini.string(), s), true);
    host::DisplaySettings t;
    host::readSettings(ini.string(), t);
    CHECK_EQ(t.upscaler, host::Upscaler::Fsr);
    CHECK_EQ(t.upscaleQuality, host::UpscaleQuality::Balanced);
    CHECK_EQ(host::upscaleLabel(t.upscaler, t.upscaleQuality), std::string("FSR Balanced"));
    CHECK_EQ(host::upscaleLabel(host::Upscaler::Dlss, host::UpscaleQuality::Native), std::string("DLAA"));

    // Ratios: Native and Off draw at the resolution; the others smaller, to 8 pixels.
    CHECK_EQ(host::upscaleRatio(t), 1.7);
    t.upscaler = host::Upscaler::Off;
    CHECK_EQ(host::upscaleRatio(t), 1.0);
    uint32_t w = 0, h = 0;
    host::upscaledSceneSize(1920, 1080, 1.5, &w, &h);
    CHECK_EQ(w, 1280u);
    CHECK_EQ(h, 720u);
    host::upscaledSceneSize(2560, 1440, 2.0, &w, &h);
    CHECK_EQ(w, 1280u);
    CHECK_EQ(h, 720u);
    host::upscaledSceneSize(1920, 1080, 1.0, &w, &h);
    CHECK_EQ(w, 1920u);
    CHECK_EQ(h, 1080u);

    fs::remove(ini);
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
