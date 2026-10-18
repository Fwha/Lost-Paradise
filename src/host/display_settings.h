// Display settings the F11 menu changes: how the window is shown and the resolution the game
// renders at. They are kept in lpn-display.ini in the working directory.
#pragma once

#include <cstdint>

namespace host {

class Module;

enum class WindowMode { Windowed, Borderless, Fullscreen };

struct DisplaySettings {
    WindowMode window = WindowMode::Windowed;
    // The game's 3D resolution; 0 keeps the game's own (1600x896, as on a base PS4).
    uint32_t width = 0, height = 0;
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

} // namespace host
