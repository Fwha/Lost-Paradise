// Each frame the game appends the player's pad state to a 60-entry history (0x3f01f0, ring at
// actor+0x1478), meant as one second at 60 fps. Moves that need mashing count presses in it,
// e.g. Hokuto Hyakuretsu after the kick: square pressed 5 times within the last second
// (condition 0x1b, 0x3f0360). Above 60 fps the 60 entries cover less than a second (0.42 s at
// 144), so the same mashing no longer counts 5 presses. This replaces the function so that,
// with the frame clock uncapped, an entry is added at most once per 1/60 s of game time; the
// press bits of the frames in between are merged into the next entry.
#pragma once

namespace host {

class Module;

// Returns false when the eboot does not hold the expected code (the game's own is kept).
bool patchInputHistory(Module& eboot);

} // namespace host
