// "PC SETTINGS" on the title menu: a row after SETTINGS that opens the F11 display menu.
//
// The game builds its title rows from a list of kinds (0xb429b0: NEW GAME 0 ... SETTINGS 6);
// kind 7 (an "attention" screen) is never added on this release but still has a label frame,
// a description and a confirm state. This adds it last and repurposes those:
//  - the layout (pjcm_titlemenu.sbb) gets an 8th frame per label slot; kind 7's frame points
//    at a new label in the atlas (2d_mr_titlemenu_parts04.dds), built from the letters of
//    REMINISCE, CONTINUE and SETTINGS in an empty band of it. Both are patched as they come
//    out of the game's SLLZ decompressor (0xe2ebc0).
//  - its description (text table 7, row 8) is swapped for one of ours.
//  - its confirm state (0xb4f090) opens the F11 menu and returns to the title menu when that
//    closes.
// LPN_PC_SETTINGS=0 leaves the title menu as the game's.
#pragma once

#include <cstddef>
#include <cstdint>

namespace host {

class Module;

// Before relocation. Returns false when the eboot does not hold the expected code (nothing is
// changed then).
bool patchTitleMenu(Module& eboot);

// The data patches, for a decompressed file; each returns false (and leaves it) unless the
// buffer is exactly the game's file.
bool patchTitleMenuLayout(uint8_t* sbb, size_t size);
bool patchTitleMenuAtlas(uint8_t* dds, size_t size);

} // namespace host
