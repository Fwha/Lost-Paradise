// Display settings the F11 menu changes: how the window is shown, the resolution the game
// renders at and the frame rate cap. They are kept in lpn-display.ini in the working directory
// (common/settings_file.h), which the launcher changes too.
#pragma once

#include "common/settings_file.h"

#include <cstdint>

namespace host {

class Module;

DisplaySettings& displaySettings();
void loadDisplaySettings();
void saveDisplaySettings();

// The game picks its render size in one place at boot: rewrite the sizes it stores there.
// Changing the resolution therefore takes a restart. Returns false when the eboot does not
// hold the expected instructions (another game version).
bool patchRenderResolution(Module& eboot, uint32_t width, uint32_t height);
// With a picture that is not 16:9 (renderSize with an Aspect Ratio), it also outputs at that size
// and builds its 3D projections for that shape instead of 16:9.
// The resolution the running game was started with (0x0: the game's own).
void activeRenderResolution(uint32_t* width, uint32_t* height);
// When the running game's picture is not 16:9: its output size (true). The renderer then draws
// the HUD (what goes into the display buffers but the full-screen scene) in the largest 16:9
// rectangle in the middle of it. LPN_HUD_STRETCH=1 leaves it stretched to the whole picture.
bool hudLayout(uint32_t* outWidth, uint32_t* outHeight);

// The game's frame clock (0xd9ee40) steps the world by a fixed 1/60 s a frame in some scenes
// (1/30 in cutscenes, 1/20 while loading, 1/60 through flag bit 2 in the title's menus) and by
// the measured frame time, but never less than 1/60 s, in others. Held at 30 fps it would run
// at half speed, shown above 60 it would run fast. With a 30 fps cap or none, the host picks the
// step (stepNs) and takes the measured time wherever the game would step 1/60. Cutscenes and
// loading keep their fixed 1/30 and 1/20. Returns false when the eboot does not hold the expected instructions; the clock is then
// left as the game's.
bool prepareFrameClock(Module& eboot);
// The frame rate cap the clock should suit (0: uncapped; above 60 is treated as uncapped). Takes effect at the next flip.
void setFrameClock(int fps);
// Called on the game's thread as it flips, between two runs of the clock: applies a change.
void syncFrameClock();
// Whether the clock now steps by the measured time with the low floor (above 60 fps).
bool frameClockUncapped();
// Whether the game's 30 fps flip rate is shown at the monitor's rate: uncapped, on the title
// (its 1/30 step is then turned off), unless LPN_KEEP_30FPS_SCENES=1. Elsewhere it is kept: the
// cutscenes ask for 30 fps with a fixed 1/30 step or, like the one before the Shin fight, with
// measured steps, and flash when shown faster.
bool uncap30FpsScenes();

// With Skip intro logos on: from boot until the title sequence starts, the game is run as fast as
// it draws (the vblank clock as on loading screens, the frame clock's longest step), so the
// logos and the photosensitivity notice go by in a moment. Off again at the title, or after 60 s.
bool skippingIntro();

} // namespace host
