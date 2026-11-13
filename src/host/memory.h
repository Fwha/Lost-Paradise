// Guest address space.
//
// PS4 GPU packets carry 40-bit addresses, so everything the guest may hand the GPU
// (modules, direct/flexible/pool memory, mmap) is placed below 1 TiB.
#pragma once

#include <cstddef>
#include <cstdint>

namespace host {

constexpr uint64_t kGuestPage = 0x4000;

// mmap that keeps non-fixed mappings in the guest range, aligned to `align`.
// Returns MAP_FAILED with errno set on failure.
void* guestMap(void* hint, uint64_t len, uint64_t align, int prot, int flags, int fd, uint64_t off);

// Backs direct memory with anonymous private mappings instead of one shared memfd, for GPU
// drivers that cannot import file-backed pages (amdgpu). Mapping a physical range twice then no
// longer aliases it. Only before the game maps any direct memory; false when too late.
bool useAnonymousDirectMemory();

} // namespace host
