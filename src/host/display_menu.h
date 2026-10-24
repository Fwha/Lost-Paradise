// The F11 display menu: window mode, the game's render resolution and the fps cap, drawn over
// the game.
// The game keeps running while it is open but gets no input.
#pragma once

#ifdef LPN_HAVE_SDL2
#include <SDL.h>
#endif

namespace host {

bool displayMenuOpen();

#ifdef LPN_HAVE_SDL2
// Shows the window as the settings say (at startup and after a change).
void applyWindowMode(SDL_Window* win);
// Runs the vblank clock at the monitor's refresh rate while uncapped, else at 59.94 Hz (at
// startup and after a change).
void applyFrameRate(SDL_Window* win);
// Called by the window thread: input (true when the menu used the event; F11 opens and closes
// it) and drawing over the frame.
bool displayMenuEvent(const SDL_Event& e, SDL_Window* win);
void drawDisplayMenu(SDL_Renderer* ren);
#endif

} // namespace host
