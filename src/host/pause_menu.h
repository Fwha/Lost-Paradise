// "PC Settings" in the pause menu's Load/Settings list, after Settings: it opens the game's
// settings screen with the PC settings, as the title menu's PC SETTINGS does (title_menu.h), so
// both show and change the same settings (pc_settings.h).
// "Exit Game" after Return to Title: the game's own Yes/No box, as Return to Title opens, asks
// "Exit the game?"; Yes exits as the F11 menu's Exit Game does, No goes back to the list.
// LPN_PC_SETTINGS=0 leaves the pause menu as the game's.
#pragma once

namespace host {

class Module;

// Before relocation, after patchTitleMenu (whose settings screen hooks it uses). Returns false
// when the eboot does not hold the expected code (nothing is changed then).
bool patchPauseMenu(Module& eboot);

} // namespace host
