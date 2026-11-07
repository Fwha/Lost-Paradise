#include "host/hooks.h"

#include <cstdio>
#include <cstring>

#include "host/mman.h"
#include "common/compat.h"
#include <unistd.h>

namespace host {

namespace {

void writeJump(uint8_t* p, const void* fn) {
    // movabs $fn, %rax; jmp *%rax
    p[0] = 0x48, p[1] = 0xb8;
    const uint64_t target = reinterpret_cast<uint64_t>(fn);
    std::memcpy(p + 2, &target, 8);
    p[10] = 0xff, p[11] = 0xe0;
}

} // namespace

bool installHooks(uint8_t* base, const Hook* hooks, size_t n, const char* tag) {
    constexpr size_t kSlot = 32;
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(base + hooks[i].at, hooks[i].was.data(), hooks[i].was.size()) != 0) {
            std::fprintf(stderr, "[%s] unexpected eboot code at 0x%x: not patched\n", tag, hooks[i].at);
            return false;
        }
    void* mem = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED || n * kSlot > 4096) return false;
    uint8_t* tramp = static_cast<uint8_t*>(mem);
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    for (size_t i = 0; i < n; ++i) {
        const Hook& h = hooks[i];
        if (h.original) {
            std::memcpy(tramp, h.was.data(), h.was.size());
            writeJump(tramp + h.was.size(), base + h.at + h.was.size());
            *h.original = tramp;
            tramp += kSlot;
        }
        uint8_t* p = base + h.at;
        // Left writable: the loader still rewrites thread pointer reads on these pages, and
        // Module::protect sets the final protection afterwards.
        mprotect(reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1)), page * 2, PROT_READ | PROT_WRITE | PROT_EXEC);
        writeJump(p, h.fn);
    }
    return true;
}

void writeData(uint8_t* p, const void* v, size_t n) {
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    void* start = reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1));
    mprotect(start, page * 2, PROT_READ | PROT_WRITE);
    std::memcpy(p, v, n);
}

} // namespace host
