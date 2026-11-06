// Smooth menu text: Arial when it is installed, else a metric-compatible sans (Liberation Sans,
// Arimo) or whatever fontconfig picks for "Arial", rasterised with stb_truetype. LPN_MENU_FONT
// names a .ttf to use instead. When no font file is found, available() is false and the menus
// draw with their blocky pixel font.
#pragma once

#ifdef LPN_HAVE_SDL2
#include <SDL.h>

#include <string>

namespace host::menufont {

bool available();
// Draws UTF-8 text with its top-left corner at (x, y) in the renderer's current draw colour.
// size is the line height in pixels.
void draw(SDL_Renderer* ren, int x, int y, int size, const std::string& s, bool bold = false);
int width(const std::string& s, int size, bool bold = false);
// s, cut short with "..." so it is at most maxW pixels wide.
std::string fit(const std::string& s, int size, int maxW, bool bold = false);

} // namespace host::menufont
#endif
