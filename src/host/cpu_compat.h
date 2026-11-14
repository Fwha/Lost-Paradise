// Running on CPUs older than the PS4's: the game's code was built for AMD Jaguar, which has AVX,
// BMI1, LZCNT, MOVBE and F16C. CPUs with AVX but without the rest (Intel Sandy Bridge and Ivy
// Bridge, AMD Bulldozer and Piledriver) get those instructions in software (common/x86_emu.h).
#pragma once

#include <cstddef>

namespace host {

class Module;

// Reads the CPU's features and logs the ones the game uses that it lacks. Returns false (with
// a message) when the CPU or OS has no AVX, which cannot be made up for.
// LPN_CPU_EMULATE=1 emulates every instruction above even when the CPU has it, to try the
// fallbacks on a new CPU; LPN_CPU_EMULATE=lzcnt,movbe (names as in the log) only those.
bool checkCpu();

// Finds the instructions patchMissingInstructions will change, on the code as loaded: call
// before any other code patch, so their jumps do not throw the decoding off.
void scanMissingInstructions(Module& m);

// Where the CPU would run lzcnt as bsr and tzcnt as bsf (silently wrong results), makes those
// instructions trap so they can be emulated; with LPN_CPU_EMULATE=1, all the instructions above.
// Call after linking and before Module::protect(). Returns the number of instructions changed.
size_t patchMissingInstructions(Module& m);

// From the illegal-instruction handler: runs the instruction at the fault, if it is one of the
// above, and returns true with the context moved past it. ctx is a ucontext_t* on Linux and a
// CONTEXT* on Windows.
bool emulateIllegalInstruction(void* ctx);

// How many times each instruction was emulated (at exit).
void printCpuCompatSummary();

} // namespace host
