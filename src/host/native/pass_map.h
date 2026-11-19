// The pass map: one frame's passes in the order the game draws them, each with its render
// targets and the names of the shaders of its draws (see shader_names.h). It is the map the
// native renderer is rebuilt from, a pass at a time.
//
// LPN_PASS_MAP=1 writes one every 10 seconds (LPN_PASS_MAP=<seconds> sets the interval) to
// lpn-captures/passmap-<frame>.txt; F12's frame capture writes one beside its trace.
#pragma once

#include "host/pm4.h"
#include "host/native/shader_names.h"

#include <cstdint>
#include <string>

namespace host::native {

class PassMap {
public:
    PassMap();
    // Whether this frame is being mapped: onDraw is only called then.
    bool active() const { return active_; }
    void onDraw(const DrawCall& d, const GpuRegs& r, ReadGuest read);
    // Ends the frame: writes the map if one was taken. `captureDir` (may be empty): also map the
    // next frame into this folder.
    void onFrameEnd(uint64_t frame, const std::string& captureDir);

private:
    struct Target {
        uint64_t addr = 0;
        uint32_t w = 0, h = 0, fmt = 0;
        bool operator==(const Target&) const = default;
    };
    struct Pass {
        Target color[8];
        Target depth;
        uint32_t draws = 0, dispatches = 0;
        std::string lines; // one per distinct shader pair, with its draw count
        std::string lastPair;
        uint32_t lastCount = 0;
    };
    void closeRun(Pass& p);

    bool active_ = false;
    double interval_ = 0; // seconds; 0: off
    double last_ = 0;
    std::string dir_ = "lpn-captures";
    std::string out_; // the map so far
    Pass cur_;
    bool open_ = false;
    uint32_t passes_ = 0;
};

} // namespace host::native
