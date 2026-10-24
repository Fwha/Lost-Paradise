// Fatal signal reporting: prints the faulting guest address as module+offset for Ghidra.
#pragma once

namespace host {

void installCrashHandler();

// Runs fn(ctx) and returns true, or returns false if it touched memory that is not mapped
// readable (SIGSEGV/SIGBUS): a read of guest memory without the cost of a syscall. fn must
// hold no locks and own nothing that needs cleanup, as a fault leaves it midway.
bool guarded(void (*fn)(void*), void* ctx);

} // namespace host
