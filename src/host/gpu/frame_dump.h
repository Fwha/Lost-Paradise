// Writes a readable report of one frame's draws and dispatches, plus every shader they use,
// so the renderer can be built against real data. Enabled with --dump-frame N.
#pragma once

#include <string>

namespace host {

void enableFrameDump(unsigned frame, const std::string& dir);

} // namespace host
