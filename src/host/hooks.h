// Jumps from the eboot's functions to ours, for the menu patches.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace host {

struct Hook {
    uint32_t at;
    std::vector<uint8_t> was; // the function's first instructions, position independent
    const void* fn;
    void** original; // set to a copy of `was` followed by a jump back (null: replaced outright)
};

// Before relocation. Checks every hook's bytes first and changes nothing unless all match;
// `tag` names the patch in the log.
bool installHooks(uint8_t* base, const Hook* hooks, size_t n, const char* tag);

// Writes data the game has mapped read only.
void writeData(uint8_t* p, const void* v, size_t n);

} // namespace host
