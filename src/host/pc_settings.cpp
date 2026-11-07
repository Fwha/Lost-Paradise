#include "host/pc_settings.h"

#include "host/display_menu.h"
#include "host/display_settings.h"
#include "host/gpu/dlss.h"

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

// Off and DLAA (upscaling is not done yet); only Off without an RTX card and NVIDIA's driver
// (or a build without the DLSS SDK).
std::vector<std::string> dlssChoices() {
    std::vector<std::string> out{dlssModeName(DlssMode::Off)};
    if (render::dlss::available()) out.push_back(dlssModeName(DlssMode::Dlaa));
    return out;
}
int dlssValue() { return render::dlss::available() && displaySettings().dlss != DlssMode::Off ? 1 : 0; }
void dlssChoose(int i) {
    if (!render::dlss::available() || i < 0 || i > 1) return;
    displaySettings().dlss = i ? DlssMode::Dlaa : DlssMode::Off;
    saveDisplaySettings();
}

std::vector<std::string> onOff() { return {"Off", "On"}; }
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

const std::vector<PcSetting>& pcSettings() {
    static const std::vector<PcSetting> settings = {
        display<0>("Window Mode", "Play in a window, a borderless window or fullscreen."),
        display<1>("Resolution", "Set the 3D rendering resolution. Takes effect after restarting the game."),
        display<2>("Frame Rate Cap",
                   "Set the most frames shown per second. Uncapped follows the monitor's refresh rate."),
        display<3>("FPS Counter", "Show the frame rate in the top right corner of the screen."),
        {"DLSS",
         "NVIDIA DLAA: DLSS anti-aliasing in place of the game's. Needs an RTX card; Off on other GPUs.",
         &dlssChoices, &dlssValue, &dlssChoose},
        {"Skip Intro Logos", "Go straight to the title screen when the game starts. Takes effect at the next start.",
         &onOff, &skipIntroValue, &skipIntroChoose},
        {"Button Icons",
         "Which controller's buttons to show. Auto matches your controller. Takes effect at the next start.",
         &buttonIconChoices, &buttonIconsValue, &buttonIconsChoose},
        display<4>("Exit Game", "Close the game. Progress since your last save is lost."),
    };
    return settings;
}

} // namespace host
