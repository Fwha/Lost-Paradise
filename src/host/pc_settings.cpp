#include "host/pc_settings.h"

#include "host/display_menu.h"
#include "host/display_settings.h"
#include "host/gpu/dlss.h"
#include "host/gpu/fsr.h"
#include "host/gpu/renderer.h"

#include <algorithm>

namespace host {

namespace {

// The F11 menu's settings: row r of display_menu.h's choices.
template <int R> std::vector<std::string> displayChoices() {
    std::vector<std::string> out;
    for (int i = 0, n = displayOptionCount(R); i < n; ++i) out.push_back(displayOptionName(R, i));
    return out;
}
template <int R> int displayValue() { return displayOptionValue(R); }
template <int R> void displayChoose(int i) { setDisplayOption(R, i); }
template <int R> constexpr PcSetting display(const char* label, const char* description) {
    return {label, description, &displayChoices<R>, &displayValue<R>, &displayChoose<R>};
}

// Off and the upscalers this GPU runs: DLSS needs an RTX card and NVIDIA's driver (and a build
// with the DLSS SDK), FSR runs on any GPU (a build with FSR-Vulkan).
std::vector<Upscaler> upscalers() {
    std::vector<Upscaler> out{Upscaler::Off};
    if (render::dlss::available()) out.push_back(Upscaler::Dlss);
    if (render::fsr::available()) out.push_back(Upscaler::Fsr);
    return out;
}
std::vector<std::string> upscalerChoices() {
    std::vector<std::string> out;
    for (Upscaler u : upscalers()) out.push_back(kUpscalerNames[int(u)]);
    return out;
}
int upscalerValue() {
    const std::vector<Upscaler> u = upscalers();
    const auto it = std::find(u.begin(), u.end(), displaySettings().upscaler);
    return it == u.end() ? 0 : int(it - u.begin());
}
void upscalerChoose(int i) {
    const std::vector<Upscaler> u = upscalers();
    if (i < 0 || i >= int(u.size())) return;
    displaySettings().upscaler = u[i];
    saveDisplaySettings();
}
std::vector<std::string> qualityChoices() { return {std::begin(kUpscaleQualityNames), std::end(kUpscaleQualityNames)}; }
int qualityValue() { return int(displaySettings().upscaleQuality); }
void qualityChoose(int i) {
    if (i < 0 || i >= kUpscaleQualities) return;
    displaySettings().upscaleQuality = UpscaleQuality(i);
    saveDisplaySettings();
}

std::vector<std::string> aspectChoices() { return {std::begin(kAspectNames), std::end(kAspectNames)}; }
int aspectSetting() { return int(displaySettings().aspect); }
void aspectChoose(int i) {
    if (i < 0 || i >= kAspectChoices) return;
    displaySettings().aspect = AspectRatio(i);
    saveDisplaySettings();
}

// The levels the GPU does (render::maxAnisotropy): Off and up to its most.
std::vector<std::string> anisoChoices() {
    std::vector<std::string> out;
    for (int i = 0; i < kAnisoCount; ++i)
        if (i == 0 || kAnisoChoices[i] <= render::maxAnisotropy()) out.push_back(kAnisoNames[i]);
    return out;
}
int anisoValue() {
    int v = 0;
    for (int i = 0; i < kAnisoCount; ++i)
        if (kAnisoChoices[i] == displaySettings().anisotropy) v = i;
    return std::min(v, int(anisoChoices().size()) - 1);
}
void anisoChoose(int i) {
    if (i < 0 || i >= int(anisoChoices().size())) return;
    displaySettings().anisotropy = kAnisoChoices[i];
    saveDisplaySettings();
}

std::vector<std::string> onOff() { return {"Off", "On"}; }
int vsyncValue() { return displaySettings().vsync ? 1 : 0; }
void vsyncChoose(int i) {
    displaySettings().vsync = i != 0; // the window thread applies it (window.cpp)
    saveDisplaySettings();
}
int skipIntroValue() { return displaySettings().skipIntro ? 1 : 0; }
void skipIntroChoose(int i) {
    displaySettings().skipIntro = i != 0;
    saveDisplaySettings();
}

std::vector<std::string> buttonIconChoices() { return {std::begin(kButtonIconNames), std::end(kButtonIconNames)}; }
int buttonIconsValue() { return int(displaySettings().buttonIcons); }
void buttonIconsChoose(int i) {
    if (i < 0 || i >= kButtonIconChoices) return;
    displaySettings().buttonIcons = ButtonIcons(i);
    saveDisplaySettings();
}

} // namespace

// Descriptions stay on one line under the menu (about 90 characters): the game does not wrap them.
const std::vector<PcSetting>& pcSettings() {
    static const std::vector<PcSetting> settings = {
        display<0>("Window Mode", "Play in a window, a borderless window or fullscreen."),
        display<1>("Resolution", "Set the 3D rendering resolution. Takes effect after restarting the game."),
        {"Aspect Ratio",
         "Match your monitor's shape; the HUD stays 16:9. Takes effect after restarting the game.",
         &aspectChoices, &aspectSetting, &aspectChoose},
        display<2>("Frame Rate Cap",
                   "Set the most frames shown per second. Uncapped follows the monitor's refresh rate."),
        {"VSync", "Show each frame at the monitor's refresh, without tearing. Off can lower input lag.", &onOff,
         &vsyncValue, &vsyncChoose},
        display<3>("FPS Counter", "Show the frame rate in the top right corner of the screen."),
        {"Upscaler",
         "Swap the game's anti-aliasing for NVIDIA DLSS (RTX cards) or AMD FSR 3.1 (any GPU).",
         &upscalerChoices, &upscalerValue, &upscalerChoose},
        {"Upscale Quality",
         "Native: anti-aliasing only. Others render smaller, faster. Takes effect after a restart.",
         &qualityChoices, &qualityValue, &qualityChoose},
        {"Anisotropic Filtering", "Keep textures sharp at glancing angles, like the ground ahead. Off: as the game sets them.",
         &anisoChoices, &anisoValue, &anisoChoose},
        {"Skip Intro Logos", "Go straight to the title screen when the game starts. Takes effect at the next start.",
         &onOff, &skipIntroValue, &skipIntroChoose},
        {"Button Icons",
         "Controller buttons to show. Auto matches your controller. Takes effect at the next start.",
         &buttonIconChoices, &buttonIconsValue, &buttonIconsChoose},
        display<4>("Exit Game", "Close the game. Progress since your last save is lost."),
    };
    return settings;
}

} // namespace host
