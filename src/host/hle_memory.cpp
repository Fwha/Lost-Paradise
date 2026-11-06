// libkernel memory: direct (physical) memory, flexible memory and mmap.
//
// Direct memory is backed by one sparse memfd the size of the PS4's, so mapping the same
// physical range twice aliases it exactly like on the console.
#include "host/hle.h"
#include "host/kernel.h"
#include "host/memory.h"

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
// Past the budget the game is told about: taken only when an allocation no longer fits in it.
// The game fills the whole budget, and above 1080p its display pool grows by 100 MB (see
// patchRenderResolution), which the PS4 Pro branch it copies would find in the Pro's extra memory.
constexpr uint64_t kOverflowSize = 0x10000000ull;
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
        if (gDirectFd < 0 || ftruncate(gDirectFd, kDirectMemorySize + kOverflowSize) != 0) {
            std::perror("[memory] cannot create direct memory backing");
            std::abort();
        }
    }
    return gDirectFd;
}

void* mapAligned(void* hint, uint64_t len, uint64_t align, int prot, int flags, int fd, uint64_t off) {
    return guestMap(hint, len, align, prot, flags, fd, off);
}

uint64_t sceKernelGetDirectMemorySize() { return kDirectMemorySize; }

int32_t sceKernelAllocateDirectMemory(int64_t searchStart, int64_t searchEnd, uint64_t len, uint64_t align,
                                      int32_t /*memoryType*/, int64_t* physOut) {
    if (!physOut || len == 0 || (len & (kPage - 1))) return sceError(ps4errno::kEINVAL);
    if (align < kPage) align = kPage;
    std::lock_guard lock(gMutex);
    auto find = [&](uint64_t from) {
        uint64_t candidate = alignUp(from, align);
        for (const auto& [start, length] : gAllocated) {
            if (candidate + len <= start) break;
            if (start + length > candidate) candidate = alignUp(start + length, align);
        }
        return candidate;
    };
    uint64_t candidate = find(uint64_t(searchStart));
    if (candidate + len > uint64_t(searchEnd) || candidate + len > kDirectMemorySize) {
        candidate = find(kDirectMemorySize);
        if (candidate + len <= kDirectMemorySize + kOverflowSize) {
            std::fprintf(stderr, "[memory] 0x%lx bytes of direct memory placed past the budget\n", len);
            gAllocated[candidate] = len;
            *physOut = int64_t(candidate);
            return 0;
        }
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
    void* r = guestMap(addr, len, kPage, prot & 7, hflags, fd, off);
    if (r == MAP_FAILED) *guestErrno() = toPs4Errno(errno);
    return r;
}
int32_t munmap_(void* addr, uint64_t len) {
    const int r = munmap(addr, len);
    if (r) *guestErrno() = toPs4Errno(errno);
    return r;
}
int32_t msync_(void* addr, uint64_t len, int32_t flags) { return msync(addr, len, flags ? MS_SYNC : MS_ASYNC); }

// Memory pools: a reserved virtual range whose pages are committed and decommitted at will.
// Pool physical memory is only bookkeeping on the PS4 side, so commits are plain mprotect.
int32_t sceKernelMemoryPoolExpand(int64_t searchStart, int64_t searchEnd, uint64_t len, uint64_t align,
                                  int64_t* physOut) {
    return sceKernelAllocateDirectMemory(searchStart, searchEnd, len, align, 0, physOut);
}

int32_t sceKernelMemoryPoolReserve(void* addrIn, uint64_t len, uint64_t align, int32_t flags, void** addrOut) {
    if (!addrOut || len == 0) return sceError(ps4errno::kEINVAL);
    int hflags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (flags & kMapFixed) hflags |= MAP_FIXED;
    void* r = mapAligned(addrIn, len, align ? align : 0x200000, PROT_NONE, hflags, -1, 0);
    if (r == MAP_FAILED) return sceErrorFromHost(errno);
    *addrOut = r;
    return 0;
}

int32_t sceKernelMemoryPoolCommit(void* addr, uint64_t len, int32_t /*type*/, int32_t prot, int32_t) {
    return mprotect(addr, len, hostProt(prot)) == 0 ? 0 : sceErrorFromHost(errno);
}

int32_t sceKernelMemoryPoolDecommit(void* addr, uint64_t len, int32_t) {
    madvise(addr, len, MADV_DONTNEED);
    return mprotect(addr, len, PROT_NONE) == 0 ? 0 : sceErrorFromHost(errno);
}

struct PoolBatchEntry {
    uint32_t op; // 1 commit, 2 decommit, 3 protect, 4 type+protect, 5 move
    uint32_t flags;
    uint64_t a, b, c; // op-specific: {addr, len, type|prot<<32}, {addr, len, prot}, {dst, src, len}
};
static_assert(sizeof(PoolBatchEntry) == 32);

int32_t sceKernelMemoryPoolBatch(const PoolBatchEntry* entries, int32_t n, int32_t* processed, int32_t) {
    int32_t done = 0, r = 0;
    for (; done < n && r == 0; ++done) {
        const auto& e = entries[done];
        void* addr = reinterpret_cast<void*>(e.a);
        switch (e.op) {
        case 1: r = sceKernelMemoryPoolCommit(addr, e.b, int32_t(e.c), int32_t(e.c >> 32), 0); break;
        case 2: r = sceKernelMemoryPoolDecommit(addr, e.b, 0); break;
        case 3: r = sceKernelMprotect(addr, e.b, int32_t(e.c)); break;
        case 4: r = sceKernelMprotect(addr, e.b, int32_t(e.c >> 32)); break;
        default:
            std::fprintf(stderr, "[memory] unsupported pool batch op %u\n", e.op);
            r = sceError(ps4errno::kEINVAL);
        }
    }
    if (processed) *processed = r ? done - 1 : done;
    return r;
}

} // namespace

void registerMemoryHle() {
    HLE("libkernel", sceKernelMemoryPoolExpand);
    HLE("libkernel", sceKernelMemoryPoolReserve);
    HLE("libkernel", sceKernelMemoryPoolCommit);
    HLE("libkernel", sceKernelMemoryPoolDecommit);
    HLE("libkernel", sceKernelMemoryPoolBatch);
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
