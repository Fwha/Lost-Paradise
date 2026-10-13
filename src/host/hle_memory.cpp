// libkernel memory: direct (physical) memory, flexible memory and mmap.
//
// Direct memory is backed by one sparse memfd the size of the PS4's, so mapping the same
// physical range twice aliases it exactly like on the console.
#include "host/hle.h"
#include "host/kernel.h"

#include <fcntl.h>
#include <linux/falloc.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <map>
#include <mutex>

namespace host {

namespace {

constexpr uint64_t kDirectMemorySize = 0x13C000000ull; // 5056 MiB, the base PS4's game budget
constexpr uint64_t kPage = 0x4000;

// PS4 protection bits: CPU read/write/exec, GPU read/write. The GPU is emulated on the CPU,
// so GPU access needs CPU access here.
enum : int32_t { kCpuRead = 1, kCpuWrite = 2, kCpuExec = 4, kGpuRead = 0x10, kGpuWrite = 0x20 };
enum : int32_t { kMapFixed = 0x10, kMapNoOverwrite = 0x80 };

std::mutex gMutex;
int gDirectFd = -1;
std::map<uint64_t, uint64_t> gAllocated; // start -> length, physical offsets

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

int hostProt(int32_t prot) {
    int p = 0;
    if (prot & (kCpuRead | kGpuRead)) p |= PROT_READ;
    if (prot & (kCpuWrite | kGpuWrite)) p |= PROT_READ | PROT_WRITE;
    if (prot & kCpuExec) p |= PROT_READ | PROT_EXEC;
    return p;
}

int directFd() {
    if (gDirectFd < 0) {
        gDirectFd = memfd_create("ps4-direct-memory", MFD_CLOEXEC);
        if (gDirectFd < 0 || ftruncate(gDirectFd, kDirectMemorySize) != 0) {
            std::perror("[memory] cannot create direct memory backing");
            std::abort();
        }
    }
    return gDirectFd;
}

// mmap at an alignment above the host page size by over-reserving and trimming.
void* mapAligned(void* hint, uint64_t len, uint64_t align, int prot, int flags, int fd, uint64_t off) {
    if (flags & MAP_FIXED) return mmap(hint, len, prot, flags, fd, off);
    if (align < kPage) align = kPage;
    void* r = mmap(hint, len + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) return r;
    const uint64_t start = alignUp(uint64_t(r), align);
    if (start > uint64_t(r)) munmap(r, start - uint64_t(r));
    const uint64_t end = uint64_t(r) + len + align;
    if (end > start + len) munmap(reinterpret_cast<void*>(start + len), end - start - len);
    return mmap(reinterpret_cast<void*>(start), len, prot, flags | MAP_FIXED, fd, off);
}

uint64_t sceKernelGetDirectMemorySize() { return kDirectMemorySize; }

int32_t sceKernelAllocateDirectMemory(int64_t searchStart, int64_t searchEnd, uint64_t len, uint64_t align,
                                      int32_t /*memoryType*/, int64_t* physOut) {
    if (!physOut || len == 0 || (len & (kPage - 1))) return sceError(ps4errno::kEINVAL);
    if (align < kPage) align = kPage;
    std::lock_guard lock(gMutex);
    uint64_t candidate = alignUp(uint64_t(searchStart), align);
    for (const auto& [start, length] : gAllocated) {
        if (candidate + len <= start) break;
        if (start + length > candidate) candidate = alignUp(start + length, align);
    }
    if (candidate + len > uint64_t(searchEnd) || candidate + len > kDirectMemorySize) {
        std::fprintf(stderr, "[memory] direct memory exhausted (0x%lx bytes requested)\n", len);
        return sceError(ps4errno::kEAGAIN);
    }
    gAllocated[candidate] = len;
    *physOut = int64_t(candidate);
    return 0;
}

int32_t sceKernelReleaseDirectMemory(int64_t start, uint64_t len) {
    std::lock_guard lock(gMutex);
    auto it = gAllocated.find(uint64_t(start));
    if (it == gAllocated.end()) return sceError(ps4errno::kENOENT);
    if (it->second != len) std::fprintf(stderr, "[memory] partial direct memory release not supported\n");
    gAllocated.erase(it);
    fallocate(directFd(), FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, start, len);
    return 0;
}

int32_t sceKernelMapDirectMemory(void** addr, uint64_t len, int32_t prot, int32_t flags, int64_t directStart,
                                 uint64_t align) {
    if (!addr || len == 0) return sceError(ps4errno::kEINVAL);
    int hflags = MAP_SHARED;
    if (flags & kMapFixed) hflags |= (flags & kMapNoOverwrite) ? MAP_FIXED_NOREPLACE : MAP_FIXED;
    void* r = mapAligned(*addr, len, align, hostProt(prot), hflags, directFd(), uint64_t(directStart));
    if (r == MAP_FAILED) return sceErrorFromHost(errno);
    *addr = r;
    return 0;
}

int32_t sceKernelMapNamedFlexibleMemory(void** addr, uint64_t len, int32_t prot, int32_t flags, const char*) {
    if (!addr || len == 0) return sceError(ps4errno::kEINVAL);
    int hflags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (flags & kMapFixed) hflags |= MAP_FIXED;
    void* r = mapAligned(*addr, len, kPage, hostProt(prot), hflags, -1, 0);
    if (r == MAP_FAILED) return sceErrorFromHost(errno);
    *addr = r;
    return 0;
}
int32_t sceKernelMapFlexibleMemory(void** addr, uint64_t len, int32_t prot, int32_t flags) {
    return sceKernelMapNamedFlexibleMemory(addr, len, prot, flags, nullptr);
}

int32_t sceKernelMunmap(void* addr, uint64_t len) {
    return munmap(addr, len) == 0 ? 0 : sceErrorFromHost(errno);
}

int32_t sceKernelMprotect(void* addr, uint64_t len, int32_t prot) {
    return mprotect(addr, len, hostProt(prot)) == 0 ? 0 : sceErrorFromHost(errno);
}

int32_t sceKernelMlock(const void*, uint64_t) { return 0; }

// POSIX mmap with FreeBSD flag values.
void* mmap_(void* addr, uint64_t len, int32_t prot, int32_t flags, int32_t fd, int64_t off) {
    int hflags = (flags & 1) ? MAP_SHARED : MAP_PRIVATE;
    if (flags & 0x10) hflags |= MAP_FIXED;
    if (flags & 0x1000) {
        hflags |= MAP_ANONYMOUS;
        fd = -1;
    }
    void* r = mmap(addr, len, prot & 7, hflags, fd, off);
    if (r == MAP_FAILED) *guestErrno() = toPs4Errno(errno);
    return r;
}
int32_t munmap_(void* addr, uint64_t len) {
    const int r = munmap(addr, len);
    if (r) *guestErrno() = toPs4Errno(errno);
    return r;
}
int32_t msync_(void* addr, uint64_t len, int32_t flags) { return msync(addr, len, flags ? MS_SYNC : MS_ASYNC); }

} // namespace

void registerMemoryHle() {
    const char* k = "libkernel";
    HLE(k, sceKernelGetDirectMemorySize);
    HLE(k, sceKernelAllocateDirectMemory);
    HLE(k, sceKernelReleaseDirectMemory);
    HLE(k, sceKernelMapDirectMemory);
    HLE(k, sceKernelMapNamedFlexibleMemory);
    HLE(k, sceKernelMapFlexibleMemory);
    HLE(k, sceKernelMunmap);
    HLE(k, sceKernelMprotect);
    HLE(k, sceKernelMlock);
    HLE_NAMED(k, "mmap", mmap_);
    HLE_NAMED(k, "munmap", munmap_);
    HLE_NAMED(k, "msync", msync_);
}

} // namespace host
