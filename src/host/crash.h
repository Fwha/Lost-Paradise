// Fatal signal reporting: prints the faulting guest address as module+offset for Ghidra.
#pragma once

namespace host {

void installCrashHandler();

} // namespace host
