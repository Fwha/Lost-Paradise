// LPN_DRAW_SITES=1: counts the game's draws and dispatches by where they come from (the
// return addresses up the %rbp chain from the eboot's Gnm draw wrappers) and logs the busiest
// call chains once a second, to pick which engine draw paths to decompile and replace first.
#pragma once

namespace host {

class Module;

bool patchDrawSites(Module& eboot);

} // namespace host
