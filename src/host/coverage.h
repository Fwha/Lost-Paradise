// LPN_COVERAGE=1: which game functions run for the first time after a key press, to find the
// code behind one action. F9 arms a one-shot breakpoint on every function entry (from
// analysis/functions.csv, or the file LPN_COVERAGE names); each hit is restored at once.
// A second F9 starts phase 2: from then on every first hit is logged as a [cover] line with
// its caller, so doing everything else in phase 1 and only the action in phase 2 leaves a short
// list of the functions behind that action.
#pragma once

namespace host::coverage {

// F9 from the window thread: arm (first press), then each press starts the next phase.
void onKey();

} // namespace host::coverage
