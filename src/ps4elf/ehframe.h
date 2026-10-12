// Function boundary recovery from .eh_frame_hdr / .eh_frame.
//
// The eboot is stripped, but every non-leaf C++ function the compiler emitted has an FDE,
// which gives us its exact start address and length.
#pragma once

#include "ps4elf/elf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ps4elf {

struct FunctionRange {
    uint64_t start = 0;
    uint64_t size = 0;
};

// Returns functions sorted by start address. Empty on failure, with *error set.
std::vector<FunctionRange> functionsFromEhFrame(const Elf& elf, std::string* error);

} // namespace ps4elf
