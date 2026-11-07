// mmap and friends for Windows (see mman.h).
//
// The guest range [8 GiB, 1 TiB) is reserved up front as placeholders. Every mapping in it is a
// view of a section: shared memory (createSharedMemory) or, for anonymous memory, a section of
// its own. Sections are SEC_RESERVE, so pages are committed only once they are made accessible.
// Because every mapping is a view, a partial munmap can unmap the view and map the parts that
// stay back at the same section offsets, keeping their contents.
//
// Mappings outside the guest range (the host's own executable pages) are plain VirtualAlloc.
#ifdef _WIN32

#include "host/mman.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>

#include "common/compat.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif

namespace {

constexpr uint64_t kWindowStart = 0x200000000ull;  // memory.cpp's guest range
constexpr uint64_t kWindowEnd = 0x10000000000ull;
constexpr uint64_t kGranule = host::kMapGranule;
constexpr uint64_t kPage = 0x1000;

// Windows 10 1803+ (kernelbase), looked up at run time so the import table stays plain.
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);
VirtualAlloc2Fn pVirtualAlloc2;
MapViewOfFile3Fn pMapViewOfFile3;
UnmapViewOfFile2Fn pUnmapViewOfFile2;

struct Section {
    HANDLE h;
    uint64_t size;
    explicit Section(HANDLE handle, uint64_t s) : h(handle), size(s) {}
    ~Section() { CloseHandle(h); }
};
using SectionRef = std::shared_ptr<Section>;

struct View {
    uint64_t len;
    SectionRef section;
    uint64_t off;
};

std::recursive_mutex gMutex;
bool gReady = false;
std::map<uint64_t, uint64_t> gHoles; // placeholders: start -> length
std::map<uint64_t, View> gViews;     // start -> view
std::map<int, SectionRef> gShared;   // createSharedMemory descriptors
int gNextFd = 0x40000000;

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
uint64_t alignDown(uint64_t v, uint64_t a) { return v & ~(a - 1); }

DWORD winProt(int prot) {
    const bool r = prot & PROT_READ, w = prot & PROT_WRITE, x = prot & PROT_EXEC;
    if (x) return w ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    if (w) return PAGE_READWRITE;
    if (r) return PAGE_READONLY;
    return PAGE_NOACCESS;
}

bool init() {
    if (gReady) return true;
    HMODULE k = GetModuleHandleA("kernelbase.dll");
    if (k) {
        pVirtualAlloc2 = reinterpret_cast<VirtualAlloc2Fn>(reinterpret_cast<void*>(GetProcAddress(k, "VirtualAlloc2")));
        pMapViewOfFile3 = reinterpret_cast<MapViewOfFile3Fn>(reinterpret_cast<void*>(GetProcAddress(k, "MapViewOfFile3")));
        pUnmapViewOfFile2 = reinterpret_cast<UnmapViewOfFile2Fn>(reinterpret_cast<void*>(GetProcAddress(k, "UnmapViewOfFile2")));
    }
    if (!pVirtualAlloc2 || !pMapViewOfFile3 || !pUnmapViewOfFile2) {
        std::fprintf(stderr, "[mman] Windows 10 version 1803 or later is needed\n");
        std::abort();
    }
    // Every free stretch of the range becomes a placeholder. Whatever Windows already put there
    // (with address space layout randomisation, a heap can land in it) is left out, and mmap
    // steps past it as it would past a host mapping on Linux.
    uint64_t reserved = 0;
    for (uint64_t a = kWindowStart; a < kWindowEnd;) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof mbi)) break;
        const uint64_t end = std::min(kWindowEnd, uint64_t(mbi.BaseAddress) + mbi.RegionSize);
        if (mbi.State == MEM_FREE) {
            const uint64_t s = alignUp(a, kGranule), e = alignDown(end, kGranule);
            if (s < e && pVirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(s), e - s,
                                        MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0)) {
                gHoles[s] = e - s;
                reserved += e - s;
            }
        }
        a = end;
    }
    if (reserved < (kWindowEnd - kWindowStart) / 2) {
        std::fprintf(stderr, "[mman] cannot reserve the guest address range (error %lu)\n", GetLastError());
        std::abort();
    }
    gReady = true;
    return true;
}

bool inWindow(uint64_t a, uint64_t len) { return a >= kWindowStart && a + len <= kWindowEnd && a + len >= a; }

bool overlapsViews(uint64_t a, uint64_t b) {
    auto it = gViews.lower_bound(a);
    if (it != gViews.begin()) {
        auto p = std::prev(it);
        if (p->first + p->second.len > a) return true;
    }
    return it != gViews.end() && it->first < b;
}

// Makes [a, b) exactly one placeholder. It must lie in holes only.
bool carvePlaceholder(uint64_t a, uint64_t b) {
    auto it = gHoles.upper_bound(a);
    if (it == gHoles.begin()) return false;
    --it;
    if (it->first + it->second <= a) return false;
    const uint64_t s = it->first;
    uint64_t e = it->first + it->second;
    auto last = it;
    while (e < b) {
        auto n = std::next(last);
        if (n == gHoles.end() || n->first != e) return false;
        last = n;
        e = n->first + n->second;
    }
    if (last != it) {
        if (!VirtualFree(reinterpret_cast<void*>(s), e - s, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) return false;
        gHoles.erase(std::next(it), std::next(last));
        it->second = e - s;
    }
    if (s != a || e != b) {
        if (!VirtualFree(reinterpret_cast<void*>(a), b - a, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return false;
        gHoles.erase(s);
        if (s < a) gHoles[s] = a - s;
        if (b < e) gHoles[b] = e - b;
        gHoles[a] = b - a;
    }
    return true;
}

// Commits (when needed) and protects [a, a + len) of mapped memory.
bool protectRange(uint64_t a, uint64_t len, int prot) {
    const uint64_t end = a + len;
    while (a < end) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof mbi)) return false;
        const uint64_t rEnd = std::min(end, uint64_t(mbi.BaseAddress) + mbi.RegionSize);
        if (mbi.State == MEM_FREE) return false;
        void* p = reinterpret_cast<void*>(a);
        if (mbi.State == MEM_RESERVE) {
            if (prot != PROT_NONE && !VirtualAlloc(p, rEnd - a, MEM_COMMIT, winProt(prot))) return false;
        } else {
            DWORD old;
            if (!VirtualProtect(p, rEnd - a, winProt(prot), &old)) return false;
        }
        a = rEnd;
    }
    return true;
}

bool mapView(uint64_t a, uint64_t len, const SectionRef& s, uint64_t off) {
    void* p = pMapViewOfFile3(s->h, GetCurrentProcess(), reinterpret_cast<void*>(a), off, len, MEM_REPLACE_PLACEHOLDER,
                              PAGE_EXECUTE_READWRITE, nullptr, 0);
    if (p != reinterpret_cast<void*>(a)) {
        std::fprintf(stderr, "[mman] MapViewOfFile3 at 0x%llx (0x%llx bytes, offset 0x%llx) failed: %lu\n",
                     (unsigned long long)a, (unsigned long long)len, (unsigned long long)off, GetLastError());
        return false;
    }
    gHoles.erase(a);
    gViews[a] = View{len, s, off};
    return true;
}

SectionRef newSection(uint64_t size) {
    HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE,
                                  DWORD(size >> 32), DWORD(size), nullptr);
    if (!h) return nullptr;
    return std::make_shared<Section>(h, size);
}

// The protection runs of [a, b), so a remapped view can get them back.
struct Run {
    uint64_t a, b;
    DWORD prot; // 0: not committed
};
std::vector<Run> protections(uint64_t a, uint64_t b) {
    std::vector<Run> runs;
    while (a < b) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof mbi)) break;
        const uint64_t e = std::min(b, uint64_t(mbi.BaseAddress) + mbi.RegionSize);
        runs.push_back({a, e, mbi.State == MEM_COMMIT ? mbi.Protect : 0});
        a = e;
    }
    return runs;
}

void restore(const std::vector<Run>& runs, uint64_t a, uint64_t b) {
    for (const Run& r : runs) {
        const uint64_t s = std::max(r.a, a), e = std::min(r.b, b);
        if (s >= e || !r.prot) continue;
        DWORD old;
        if (!VirtualProtect(reinterpret_cast<void*>(s), e - s, r.prot, &old))
            VirtualAlloc(reinterpret_cast<void*>(s), e - s, MEM_COMMIT, r.prot);
    }
}

// Unmaps [a, b) of the guest range. Views cut at a 64 KiB boundary are split, keeping the
// parts outside; where a cut is not on one, that part stays mapped but inaccessible.
bool unmapWindow(uint64_t a, uint64_t b) {
    auto it = gViews.lower_bound(a);
    if (it != gViews.begin() && std::prev(it)->first + std::prev(it)->second.len > a) --it;
    std::vector<uint64_t> starts;
    for (; it != gViews.end() && it->first < b; ++it) starts.push_back(it->first);
    for (uint64_t vs : starts) {
        const View v = gViews[vs];
        const uint64_t ve = vs + v.len;
        uint64_t cutA = std::max(a, vs), cutB = std::min(b, ve);
        if ((cutA != vs && cutA % kGranule) || (cutB != ve && cutB % kGranule)) {
            // Can't split here: make the unaligned part inaccessible instead.
            const uint64_t inA = cutA == vs ? vs : alignUp(cutA, kGranule);
            const uint64_t inB = cutB == ve ? ve : alignDown(cutB, kGranule);
            DWORD old;
            for (const Run& r : protections(cutA, cutB))
                if (r.prot) VirtualProtect(reinterpret_cast<void*>(r.a), r.b - r.a, PAGE_NOACCESS, &old);
            if (inA >= inB) continue;
            cutA = inA, cutB = inB;
        }
        const auto runs = protections(vs, ve);
        if (!pUnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(vs), MEM_PRESERVE_PLACEHOLDER)) return false;
        gViews.erase(vs);
        gHoles[vs] = v.len;
        if (cutA > vs) {
            if (!carvePlaceholder(vs, cutA) || !mapView(vs, cutA - vs, v.section, v.off)) return false;
            restore(runs, vs, cutA);
        }
        if (cutB < ve) {
            if (!carvePlaceholder(cutB, ve) || !mapView(cutB, ve - cutB, v.section, v.off + (cutB - vs))) return false;
            restore(runs, cutB, ve);
        }
        if (cutA > vs || cutB < ve) carvePlaceholder(cutA, cutB);
    }
    return true;
}

void* fail(int e) {
    errno = e;
    return MAP_FAILED;
}

} // namespace

namespace host {

int createSharedMemory(const char*, uint64_t size) {
    std::lock_guard lock(gMutex);
    SectionRef s = newSection(size);
    if (!s) return -1;
    const int fd = gNextFd++;
    gShared[fd] = s;
    return fd;
}

void releaseSharedMemory(int, uint64_t, uint64_t) {}

} // namespace host

void* mmap(void* addr, size_t len, int prot, int flags, int fd, int64_t off) {
    std::lock_guard lock(gMutex);
    init();
    if (len == 0) return fail(EINVAL);
    const uint64_t a = uint64_t(addr);
    const bool fixed = flags & (MAP_FIXED | MAP_FIXED_NOREPLACE);
    SectionRef shared;
    if (!(flags & MAP_ANONYMOUS)) {
        auto it = gShared.find(fd);
        if (it != gShared.end()) shared = it->second;
    }

    if (!fixed || !inWindow(a, len)) {
        if (fixed) return fail(EINVAL);
        // The host's own memory, anywhere.
        void* p = VirtualAlloc(nullptr, len, prot == PROT_NONE ? MEM_RESERVE : MEM_RESERVE | MEM_COMMIT,
                               prot == PROT_NONE ? PAGE_NOACCESS : winProt(prot));
        if (!p) return fail(ENOMEM);
        if (!(flags & MAP_ANONYMOUS) && !shared && fd >= 0) pread(fd, p, len, off);
        return p;
    }

    if (a % kGranule || (shared && uint64_t(off) % kGranule)) {
        // Anonymous memory over an existing view of its own (a reservation being filled in):
        // fresh zeroed pages in place.
        if (!shared && (flags & MAP_FIXED) && !(flags & MAP_FIXED_NOREPLACE)) {
            auto it = gViews.upper_bound(a);
            if (it != gViews.begin() && (--it, it->first + it->second.len >= a + len)) {
                if (!protectRange(a, len, PROT_READ | PROT_WRITE)) return fail(ENOMEM);
                std::memset(addr, 0, len);
                if (!protectRange(a, len, prot)) return fail(ENOMEM);
                return addr;
            }
        }
        std::fprintf(stderr, "[mman] mapping at 0x%llx (offset 0x%llx) is not on a 64 KiB boundary\n",
                     (unsigned long long)a, (unsigned long long)off);
        return fail(EINVAL);
    }

    const uint64_t b = a + alignUp(len, kPage);
    if (overlapsViews(a, b)) {
        if (flags & MAP_FIXED_NOREPLACE) return fail(EEXIST);
        if (!unmapWindow(a, b)) return fail(ENOMEM);
    }
    if (!carvePlaceholder(a, b)) return fail(EEXIST); // something of Windows' is in the way
    SectionRef s = shared ? shared : newSection(b - a);
    if (!s) return fail(ENOMEM);
    if (!mapView(a, b - a, s, shared ? uint64_t(off) : 0)) return fail(ENOMEM);
    if (prot != PROT_NONE && !protectRange(a, b - a, prot)) {
        unmapWindow(a, b);
        return fail(ENOMEM);
    }
    if (!(flags & MAP_ANONYMOUS) && !shared && fd >= 0) {
        // A file: its contents, copied (MAP_PRIVATE semantics; writes don't reach the file).
        if (flags & MAP_SHARED) std::fprintf(stderr, "[mman] MAP_SHARED file mapping copied, not shared\n");
        DWORD old;
        VirtualProtect(addr, len, PAGE_READWRITE, &old);
        pread(fd, addr, len, off);
        protectRange(a, b - a, prot);
    }
    return addr;
}

int munmap(void* addr, size_t len) {
    std::lock_guard lock(gMutex);
    const uint64_t a = uint64_t(addr);
    if (gReady && inWindow(a, len)) {
        if (!unmapWindow(alignDown(a, kPage), alignUp(a + len, kPage))) {
            errno = EINVAL;
            return -1;
        }
        return 0;
    }
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(addr, &mbi, sizeof mbi) && mbi.AllocationBase == addr && mbi.RegionSize >= len)
        return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
    return VirtualFree(addr, len, MEM_DECOMMIT) ? 0 : -1;
}

int mprotect(void* addr, size_t len, int prot) {
    std::lock_guard lock(gMutex);
    const uint64_t a = alignDown(uint64_t(addr), kPage);
    if (!protectRange(a, alignUp(uint64_t(addr) + len, kPage) - a, prot)) {
        errno = ENOMEM;
        return -1;
    }
    return 0;
}

int madvise(void*, size_t, int) { return 0; }

int msync(void* addr, size_t len, int) { return FlushViewOfFile(addr, len) ? 0 : 0; }

int mincore(void* addr, size_t len, unsigned char* vec) {
    uint64_t a = alignDown(uint64_t(addr), kPage);
    const uint64_t end = uint64_t(addr) + len;
    size_t i = 0;
    while (a < end) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof mbi) || mbi.State == MEM_FREE) {
            errno = ENOMEM;
            return -1;
        }
        const uint64_t e = std::min(end, uint64_t(mbi.BaseAddress) + mbi.RegionSize);
        for (; a < e; a += kPage) vec[i++] = mbi.State == MEM_COMMIT ? 1 : 0;
    }
    return 0;
}

#endif // _WIN32
