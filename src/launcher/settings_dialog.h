// The launcher's Settings window: a separate window of the desktop's own controls (tabs,
// drop-downs, checkboxes and Set To Default, Ok and Cancel buttons) over the same lpn-display.ini
// settings as the in-game PC Settings (common/settings_file.h).
//
// Windows: Win32 common controls (settings_dialog_win32.cpp). Linux: GTK 3, loaded when the window
// opens (settings_dialog_gtk.cpp), so the launcher still starts where it is missing and falls back
// to its own Settings screen. The options and their tabs are listed once, here, for both.
#pragma once

#include "common/settings_file.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace launcher::settings {

// One row: a drop-down of choices, or a checkbox (choices empty, the value 0 or 1).
struct Option {
    const char* label;
    const char* tip; // the row's tooltip
    bool nextStart;  // takes effect when the game is started again
    std::vector<std::string> (*choices)(const host::DisplaySettings&);
    int (*get)(const host::DisplaySettings&);
    void (*set)(host::DisplaySettings&, int);
    bool checkbox() const { return !choices; }
};

struct Tab {
    const char* name;
    std::vector<Option> options;
};

const std::vector<Tab>& tabs();
// The line under a tab's rows: which of them take effect at the next start ("" for none).
std::string tabNote(const Tab& t);

// An RGBA picture (the game's key art) for the banner at the top of the window.
struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;
};
// art scaled to cover w x h, cropped to it (a little above the middle, where the faces are).
Image coverCrop(const Image& art, int w, int h);

enum class Result { Ok, Cancel, Unavailable };

// Shows the window over parent (Windows: its HWND; Linux: unused) until Ok, Cancel or its close
// button. Ok leaves the choices in s, Cancel leaves s as it was. idle is called while it is open,
// for the launcher to keep its own window drawn. Unavailable: no window could be shown.
Result show(host::DisplaySettings& s, void* parent, const Image& art, const std::function<void()>& idle);

} // namespace launcher::settings
