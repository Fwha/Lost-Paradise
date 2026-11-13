// The F11 display menu: window mode, the game's render resolution and the fps cap, drawn over
// the game.
// The game keeps running while it is open but gets no input.
#pragma once

#include <string>

#ifdef LPN_HAVE_SDL2
#include <SDL.h>
#endif

namespace host {

bool displayMenuOpen();

// The menu's settings as choices, for the game's own PC SETTINGS screen (title_menu.h): row 0
// window mode, 1 resolution, 2 fps cap, 3 fps counter, 4 exit game (choice 1 quits). A value is a
// choice's index; a setting none of them names (a size or cap from the ini or the command line)
// is a last choice of its own.
constexpr int kDisplayOptions = 5;
int displayOptionCount(int row);
std::string displayOptionName(int row, int i);
int displayOptionValue(int row); // a choice set but not applied yet counts
// From any thread: applied by the window thread, as if picked in the menu.
void setDisplayOption(int row, int i);

#ifdef LPN_HAVE_SDL2
// Shows the window as the settings say (at startup and after a change).
void applyWindowMode(SDL_Window* win);
// Runs the vblank clock at the monitor's refresh rate while uncapped, at the cap above 60, else
// at 59.94 Hz (at startup and after a change).
void applyFrameRate(SDL_Window* win);
// Called by the window thread: input (true when the menu used the event; F11 opens and closes
// it) and drawing over the frame.
bool displayMenuEvent(const SDL_Event& e, SDL_Window* win);
void drawDisplayMenu(SDL_Renderer* ren);
// Applies the choices setDisplayOption queued (each frame).
void applyDisplayOptions(SDL_Window* win);
// Compile Shaders at Startup: the progress screen while the game waits for the startup compile
// (gpu/renderer.h warmupProgress). False when there is none.
bool drawShaderWarmup(SDL_Renderer* ren);
// The boot compile's progress in the bottom left corner while it runs.
void drawBootCompile(SDL_Renderer* ren);
// The frames shown a second, in the top right corner, when the menu's FPS counter is on.
void drawFpsCounter(SDL_Renderer* ren);
#endif

} // namespace host
