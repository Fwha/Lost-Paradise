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

// Only Off without an RTX card and NVIDIA's driver (or a build without the DLSS SDK).
std::vector<std::string> dlssChoices() {
    std::vector<std::string> out;
    for (int i = 0; i < (render::dlss::available() ? kDlssModes : 1); ++i) out.push_back(dlssModeName(DlssMode(i)));
    return out;
}
int dlssValue() { return render::dlss::available() ? int(displaySettings().dlss) : 0; }
void dlssChoose(int i) {
    if (!render::dlss::available() || i < 0 || i >= kDlssModes) return;
    displaySettings().dlss = DlssMode(i);
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
         "NVIDIA DLSS anti-aliasing (DLAA) or upscaling. Needs an RTX card; Off on other GPUs.",
         &dlssChoices, &dlssValue, &dlssChoose},
        display<4>("Exit Game", "Close the game. Progress since your last save is lost."),
    };
    return settings;
}

} // namespace host
