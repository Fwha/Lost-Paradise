// The game window. One thread owns SDL and shows whatever VideoOut flips to.
#pragma once

#include <cstdint>

namespace host {

struct Frame {
    const void* pixels = nullptr;
    uint32_t width = 0, height = 0, pitchPixels = 0;
    uint32_t pixelFormat = 0; // SCE_VIDEO_OUT_PIXEL_FORMAT_*
    uint32_t tilingMode = 0;  // 0 tiled, 1 linear
};

// Open the window (or do nothing when headless or built without SDL).
void startPresenter(bool headless);
// Sound through SDL (on by default); off, audio ports are only paced by the clock.
void setAudioEnabled(bool on);
// Copy a flipped buffer for display. Safe to call from any thread.
void presentFrame(const Frame& f);

// Controller state from the keyboard or a connected game controller, in PS4 terms.
struct PadState {
    uint32_t buttons = 0; // SCE_PAD_BUTTON_* bits
    uint8_t lx = 0x80, ly = 0x80, rx = 0x80, ry = 0x80, l2 = 0, r2 = 0;
    // Motion, as a DualShock 4 reports it: x right, y up, z toward the player. Acceleration in g
    // (1 up when lying flat), angular velocity in rad/s, orientation a quaternion (x, y, z, w).
    // From the controller's sensors where it has them (DualShock 4, DualSense, Switch Pro);
    // otherwise the right stick tilts and shakes a virtual pad.
    float acceleration[3] = {0, 1, 0}, angularVelocity[3] = {0, 0, 0}, orientation[4] = {0, 0, 0, 1};
};
PadState padState();

// Scripted presses for headless runs: "cross@1800,options@2400" holds each button for 10
// displayed frames starting at that frame; lsleft/lsright/lsup/lsdown and rsleft/... push a
// stick fully instead. Returns false on a bad spec.
bool addScriptedPresses(const char* spec);

} // namespace host
