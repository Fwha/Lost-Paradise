// The accent colour of our own screens: the save/load menu, the F11 display menu, the FPS
// counter and the launcher. Highlights, outlines, focus bars and selected buttons all use it, so
// changing the accent here changes it everywhere.
//
// LPN_UI_ACCENT picks one of the looks below at build time (cmake -DLPN_UI_ACCENT=1 ...);
// 2, steel blue, is the default.
#pragma once

#include <cstdint>

#ifndef LPN_UI_ACCENT
#define LPN_UI_ACCENT 2
#endif

namespace ui {

struct Rgb {
    uint8_t r, g, b;
};

struct Accent {
    Rgb line;   // outlines, arrows, labels and other accent text
    Rgb fill;   // the selected item's background, switches, tags, the scroll thumb
    Rgb bright; // focus that needs to stand out more than line (the focused launcher button)
    Rgb on;     // text drawn on fill
    Rgb tint;   // the highlighted row's background, under its line outline
};

#if LPN_UI_ACCENT == 1 // pure white
inline constexpr Accent kAccent{{236, 236, 238}, {222, 222, 226}, {255, 255, 255}, {14, 14, 18}, {48, 48, 56}};
#elif LPN_UI_ACCENT == 2 // steel blue
inline constexpr Accent kAccent{{140, 176, 214}, {120, 158, 200}, {170, 200, 232}, {10, 16, 26}, {28, 40, 58}};
#elif LPN_UI_ACCENT == 3 // crimson
inline constexpr Accent kAccent{{206, 78, 78}, {184, 58, 58}, {228, 104, 104}, {250, 244, 240}, {58, 24, 26}};
#else // soft silver
inline constexpr Accent kAccent{{204, 208, 216}, {186, 191, 200}, {228, 231, 236}, {16, 17, 22}, {46, 49, 58}};
#endif

} // namespace ui
