// The rows of the title menu's PC SETTINGS screen (title_menu.h), in order. To add a setting, add
// an entry to the list in pc_settings.cpp; the screen takes up to kMaxPcSettings.
#pragma once

#include <string>
#include <vector>

namespace host {

struct PcSetting {
    const char* label;       // as the game's settings name theirs ("Frame Rate Cap")
    const char* description; // the line under the list while the row is highlighted
    // The names of its choices (at most 15), as the screen opens. None leaves the game's own
    // settings on the screen.
    std::vector<std::string> (*choices)();
    int (*value)();          // the index of the choice it has
    void (*choose)(int i);   // the player confirmed choice i (on the game's thread)
};

constexpr int kMaxPcSettings = 13;
const std::vector<PcSetting>& pcSettings();

} // namespace host
