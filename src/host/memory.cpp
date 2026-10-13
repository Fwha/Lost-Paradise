#include "host/memory.h"

#include <sys/mman.h>

#include <atomic>
#include <cerrno>

namespace host {

namespace {

constexpr uint64_t kGuestStart = 0x200000000ull;  // 8 GiB: clear of the host executable and heap
constexpr uint64_t kGuestEnd = 0x10000000000ull;  // 1 TiB: the GPU's 40-bit limit
std::atomic<uint64_t> gNext{kGuestStart};

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

} // namespace

void* guestMap(void* hint, uint64_t len, uint64_t align, int prot, int flags, int fd, uint64_t off) {
    if (flags & (MAP_FIXED | MAP_FIXED_NOREPLACE)) return mmap(hint, len, prot, flags, fd, off);
    if (align < kGuestPage) align = kGuestPage;
    len = alignUp(len, kGuestPage);

    if (hint && uint64_t(hint) + len <= kGuestEnd) {
        void* r = mmap(reinterpret_cast<void*>(alignUp(uint64_t(hint), align)), len, prot,
                       flags | MAP_FIXED_NOREPLACE, fd, off);
        if (r != MAP_FAILED) return r;
    }
    // Bump allocation; ranges are not reused, which 1 TiB of address space affords for now.
    for (;;) {
        uint64_t cur = gNext.load();
        const uint64_t start = alignUp(cur, align);
        if (start + len > kGuestEnd) {
            errno = ENOMEM;
            return MAP_FAILED;
        }
        if (!gNext.compare_exchange_weak(cur, start + len)) continue;
        void* r = mmap(reinterpret_cast<void*>(start), len, prot, flags | MAP_FIXED_NOREPLACE, fd, off);
        if (r != MAP_FAILED || errno != EEXIST) return r;
        // Something of the host's sits there; move past it.
    }
}

} // namespace host
