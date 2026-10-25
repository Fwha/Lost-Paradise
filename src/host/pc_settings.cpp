#include "host/pc_settings.h"

#include "host/display_menu.h"

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

} // namespace

const std::vector<PcSetting>& pcSettings() {
    static const std::vector<PcSetting> settings = {
        display<0>("Window Mode", "Play in a window, a borderless window or fullscreen."),
        display<1>("Resolution", "Set the 3D rendering resolution. Takes effect after restarting the game."),
        display<2>("Frame Rate Cap",
                   "Set the most frames shown per second. Uncapped follows the monitor's refresh rate."),
        display<3>("FPS Counter", "Show the frame rate in the top right corner of the screen."),
        display<4>("Exit Game", "Close the game. Progress since your last save is lost."),
    };
    return settings;
}

} // namespace host
