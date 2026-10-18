// Our own save-data list dialog, drawn over the game window: pick a slot to save to or load
// from, overwrite or delete one. The game waits while it is open, as it does on a PS4.
#pragma once

#include <string>
#include <vector>

#ifdef LPN_HAVE_SDL2
#include <SDL.h>
#endif

namespace host {

struct SaveSlot {
    std::string dir;  // directory name the game listed, e.g. "02"
    std::string path; // its folder on disk
};

enum class SaveMenuKind { Save, Load, Delete };

// Open the menu over the game window. Returns false when there is no window to show it in
// (headless, built without SDL, scripted presses, LPN_SAVE_MENU=0); the caller then picks itself.
// newItem adds an "EMPTY" row under the saves when saving: picking it makes a new save.
bool openSaveMenu(SaveMenuKind kind, std::vector<SaveSlot> slots, int focus, bool newItem);
bool saveMenuOpen();
// After the menu closes: the chosen slot's directory, or "" when cancelled.
std::string saveMenuChoice();
// After the menu closes: true when the "EMPTY" new-save row was picked (the choice is then "").
bool saveMenuChoseNew();
void closeSaveMenu();

#ifdef LPN_HAVE_SDL2
// Called by the window thread: input while open (returns true when the menu used the event),
// and drawing over the frame.
bool saveMenuEvent(const SDL_Event& e);
void drawSaveMenu(SDL_Renderer* ren);
#endif

} // namespace host
