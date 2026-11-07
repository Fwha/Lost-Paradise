// Controller button icons for the control hints in our menus and the launcher: PlayStation,
// Xbox and Nintendo Switch face buttons, from ControllerImage's public-domain art
// (external/controllerimage), rasterised with nanosvg at the size asked for.
#pragma once

#include <SDL.h>

#include "common/settings_file.h"

namespace host {

enum class PadStyle { PlayStation, Xbox, Switch };

// The style the setting asks for. Auto: the first controller plugged in (PlayStation and Switch
// pads as themselves, every other pad as Xbox), PlayStation with none.
PadStyle padStyle(ButtonIcons setting);

// Draws SDL's face button b (A, B, X or Y) as style labels it, in a size x size box at (x, y).
// Xbox and PlayStation by position (A is the bottom button, cross on a DualSense); Switch by
// label, as SDL2 maps a Nintendo pad (A is its right button), unless
// SDL_HINT_GAMECONTROLLER_USE_BUTTON_LABELS is off. False when the icon could not be drawn.
bool drawPadGlyph(SDL_Renderer* ren, int x, int y, int size, PadStyle style, SDL_GameControllerButton b);

} // namespace host
