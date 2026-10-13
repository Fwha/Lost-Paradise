// High-level emulation: our C++ implementations of PS4 system library functions.
//
// The PS4 uses the same System V x86-64 calling convention as Linux, so an HLE function is
// an ordinary C++ function with the guest's signature, bound directly into the import slot.
#pragma once

#include <cstdint>
#include <string>

namespace host {

// Everything the trampoline for an unimplemented import needs to report it.
struct ImportInfo {
    std::string library;
    std::string nid;
    std::string name; // empty when the NID isn't in the database
    uint64_t calls = 0;
    void* target = nullptr;      // traced imports: the implementation to run
    uint64_t lastTracedFlip = 0; // traced imports: flip of the last logged call
};

// Register an implementation by library and symbol name. Call before linking.
void registerHle(const std::string& library, const std::string& name, void* fn);

// The registered implementation, or nullptr.
void* findHle(const std::string& library, const std::string& name);

// Install the built-in implementations (see hle_*.cpp).
void registerBuiltinHle();

// Logging switches, set from the command line.
struct HleOptions {
    bool abortOnUnimplemented = false;
    uint64_t logLimitPerImport = 3; // further calls to the same import are counted, not logged
    bool traceHle = false;          // route implemented imports through a logging trampoline
    uint64_t traceFrom = 0;         // with traceHle: also log each import's first call after this flip
};
HleOptions& hleOptions();

// Summary of unimplemented imports hit so far, most-called first.
void printUnimplementedSummary();

} // namespace host
