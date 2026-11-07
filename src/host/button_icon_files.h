// The game's own button prompts in the Button Icons setting's style (pad_glyphs.h). Its icons
// are textures in two archives: the font's symbol glyphs (fontpar/font.par, kigou00-03: the
// circle, triangle, square and cross the menus' "Confirm"/"Back" bar and the text use) and the
// button sheets in 2dpar/sprite_e.par (the HUD's prompts). For Xbox or Switch the game reads a
// copy of each archive with those four face buttons redrawn, made once in lpn-data/button-icons;
// the other icons (L1, R2, OPTIONS, ...) stay as the game draws them. The game loads these at
// boot, so a change of the setting shows in its own screens from the next start.
#pragma once

#include <string>

namespace host {

// The host file to open for `path` (a game file): the redrawn copy when it is one of the two
// archives and the setting is not PlayStation, else path itself.
std::string buttonIconFile(const std::string& path);

} // namespace host
