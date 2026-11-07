// The PC settings file, lpn-display.ini in the install folder: what the F11 menu, the game's PC
// SETTINGS screen and the launcher (src/launcher) all read and change. Kept free of the host so
// the launcher can share it.
#pragma once

#include <cstdint>
#include <string>

namespace host {

enum class WindowMode { Windowed, Borderless, Fullscreen };
// NVIDIA DLSS (gpu/dlss.h): anti-aliasing at the render size (DLAA) or upscaling to it.
enum class DlssMode { Off, Dlaa, Quality, Balanced, Performance };
constexpr int kDlssModes = 5;
const char* dlssModeName(DlssMode m); // "Off", "DLAA", "Quality", ...
// Which controller's buttons the menus' control hints show (host/pad_glyphs.h). Auto follows the
// controller plugged in.
enum class ButtonIcons { PlayStation, Xbox, Switch, Auto };
constexpr int kButtonIconChoices = 4;
constexpr const char* kButtonIconNames[kButtonIconChoices] = {"PlayStation", "Xbox", "Switch", "Auto"};

struct DisplaySettings {
    WindowMode window = WindowMode::Windowed;
    // The game's 3D resolution; 0 keeps the game's own (1600x896, as on a base PS4).
    uint32_t width = 0, height = 0;
    // Frames a second shown at most (see setFpsCap); 0 leaves the rate to the game.
    int fps = 30;
    bool fpsCounter = false; // frames shown a second, in the top right corner
    DlssMode dlss = DlssMode::Off;
    bool skipIntro = false; // fast-forward the boot logos to the title screen
    ButtonIcons buttonIcons = ButtonIcons::PlayStation; // as the game draws its own
};

constexpr const char* kSettingsFile = "lpn-display.ini";

// Missing or unreadable: s is left as it is.
void readSettings(const std::string& path, DisplaySettings& s);
bool writeSettings(const std::string& path, const DisplaySettings& s);

// The choices the menus offer.
constexpr const char* kWindowModeNames[] = {"Windowed", "Borderless Fullscreen", "Fullscreen"};
struct Resolution {
    uint32_t w, h; // 0x0: the game's own
    const char* label;
};
constexpr Resolution kResolutions[] = {
    {1280, 720, "1280x720"},
    {0, 0, "1600x896 (PS4)"},
    {1920, 1080, "1920x1080 (PS4 Pro)"},
    {2560, 1440, "2560x1440"},
};
constexpr int kResCount = int(sizeof kResolutions / sizeof kResolutions[0]);
// The game runs at 60 and drops itself to 30 in cutscenes and 20 on loading screens. Above 60
// (90, or uncapped at the monitor's refresh rate) the frame clock is patched to keep its speed,
// and the pad history keeps one entry per 1/60 s so mashing moves still count (input_history.h).
// Uncapped, the title screen runs at the monitor's rate too; cutscenes stay at 30.
constexpr int kFpsChoices[] = {30, 60, 90, 0};
constexpr int kFpsCount = int(sizeof kFpsChoices / sizeof kFpsChoices[0]);

} // namespace host
