// Memory mapping. On Linux this is <sys/mman.h>. On Windows (mman_win.cpp) the same calls are
// built on placeholders (VirtualAlloc2 / MapViewOfFile3), covering what the host uses: anonymous
// and shared-memory mappings anywhere in the guest range, fixed or not, partial munmap and
// mprotect across mappings. Views there sit on 64 KiB boundaries (kMapGranule).
#pragma once

#include <cstddef>
#include <cstdint>

#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace host {

#ifdef _WIN32
constexpr uint64_t kMapGranule = 0x10000;
#else
constexpr uint64_t kMapGranule = 0x1000;
#endif

// A shared memory object of `size` bytes (memfd on Linux, a pagefile-backed section on
// Windows), mappable with MAP_SHARED. Returns a descriptor for mmap, or -1.
int createSharedMemory(const char* name, uint64_t size);

// Gives back the memory under [off, off + len) of a shared memory object (no-op on Windows).
void releaseSharedMemory(int fd, uint64_t off, uint64_t len);

} // namespace host

#ifdef _WIN32
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4

#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_NORESERVE 0x4000
#define MAP_FIXED_NOREPLACE 0x100000
#define MAP_FAILED (reinterpret_cast<void*>(-1))

#define MADV_DONTNEED 4
#define MS_ASYNC 1
#define MS_SYNC 4

void* mmap(void* addr, size_t len, int prot, int flags, int fd, int64_t off);
int munmap(void* addr, size_t len);
int mprotect(void* addr, size_t len, int prot);
int madvise(void* addr, size_t len, int advice);
int msync(void* addr, size_t len, int flags);
int mincore(void* addr, size_t len, unsigned char* vec);
#endif
