// Resolves imports between loaded modules and our HLE code, and applies relocations.
#pragma once

#include "ps4elf/nid.h"

#include <cstdint>
#include <string>
#include <vector>

namespace host {

class Module;

struct LinkStats {
    size_t relocations = 0;
    size_t toModules = 0;      // imports bound to another module's export (LLE)
    size_t toHle = 0;          // imports bound to our C++ implementations
    size_t unimplemented = 0;  // imports bound to logging trampolines
    std::vector<std::string> unimplementedNames; // "library::name", one per distinct import
};

// Link every module against the others' exports, then HLE, then trampolines.
// `nids` names imports so HLE lookup and logs can use symbol names.
bool linkModules(const std::vector<Module*>& modules, const ps4elf::NidDb& nids, LinkStats* stats,
                 std::string* error);

// Module containing addr, or nullptr; used by crash reports.
Module* findModule(uint64_t addr);

} // namespace host
