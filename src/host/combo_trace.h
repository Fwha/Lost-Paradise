// LPN_COMBO_TRACE=1: F8 starts and stops a trace of how the player's attack picks its next move.
// Each frame of an attack the game resolves the moves that may follow (0x3f77b0), tests each
// candidate's conditions (0x3eb7d0 -> 0x3eafa0) and runs the one that passes (0x3f7a20). Hooks
// on those log [combo] lines: the current move, each candidate with its result, each condition
// with its type, parameter and result, and the move that runs.
#pragma once

namespace host::combo_trace {

void onKey(); // F8, from the window thread

} // namespace host::combo_trace
