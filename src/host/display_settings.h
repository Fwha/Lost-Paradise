// Display settings the F11 menu changes: how the window is shown, the resolution the game
// renders at and the frame rate cap. They are kept in lpn-display.ini in the working directory.
#pragma once

#include <cstdint>

namespace host {

class Module;

enum class WindowMode { Windowed, Borderless, Fullscreen };

struct DisplaySettings {
    WindowMode window = WindowMode::Windowed;
    // The game's 3D resolution; 0 keeps the game's own (1600x896, as on a base PS4).
    uint32_t width = 0, height = 0;
    // Frames a second shown at most (see setFpsCap); 0 leaves the rate to the game.
    int fps = 30;
    bool fpsCounter = false; // frames shown a second, in the top right corner
};

DisplaySettings& displaySettings();
void loadDisplaySettings();
void saveDisplaySettings();

// The game picks its render size in one place at boot: rewrite the sizes it stores there.
// Changing the resolution therefore takes a restart. Returns false when the eboot does not
// hold the expected instructions (another game version).
bool patchRenderResolution(Module& eboot, uint32_t width, uint32_t height);
// The resolution the running game was started with (0x0: the game's own).
void activeRenderResolution(uint32_t* width, uint32_t* height);

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
// Whether the game's 30 fps flip rate is shown at the monitor's rate: uncapped, outside the fixed
// 1/30 step of cutscenes (the title screen asks for 30 with a measured step), unless
// LPN_KEEP_30FPS_SCENES=1.
bool uncap30FpsScenes();

} // namespace host
