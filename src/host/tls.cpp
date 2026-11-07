#include "host/tls.h"

#include "host/module.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace host {

namespace {

// Read through patched guest instructions as `mov reg, fs:[offset]`, so it must use the
// initial-exec TLS model (see CMakeLists.txt) to sit at a fixed offset from %fs. On Windows the
// TEB slot (gs:[0x1480 + 8 * index]) is what the game reads, and this is a copy for our use.
thread_local void* tGuestThreadPointer = nullptr;
#ifdef _WIN32
DWORD gTlsSlot = TLS_OUT_OF_INDEXES;
#endif

struct TlsImage {
    const uint8_t* init;
    uint64_t initSize;
    uint64_t offset; // block = thread pointer - offset
};
std::vector<TlsImage> gImages; // indexed by tlsIndex - 1
uint64_t gStaticSize = 0;
uint64_t gStaticAlign = 16;

constexpr uint64_t kTcbSize = 0x100; // self pointer, dtv, thread; generous for unknown fields

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

} // namespace

ThreadPointerSlot guestThreadPointerSlot() {
#ifdef _WIN32
    if (gTlsSlot == TLS_OUT_OF_INDEXES) gTlsSlot = TlsAlloc();
    if (gTlsSlot >= 64) { // the TEB holds the first 64 slots inline
        std::fprintf(stderr, "[tls] no inline TLS slot free (got %lu)\n", gTlsSlot);
        std::abort();
    }
    return {0x65, int32_t(0x1480 + 8 * gTlsSlot)};
#else
    char* fsBase;
    asm("mov %%fs:0, %0" : "=r"(fsBase)); // glibc keeps its own TCB self-pointer at fs:0
    const intptr_t off = reinterpret_cast<char*>(&tGuestThreadPointer) - fsBase;
    if (off < INT32_MIN || off > INT32_MAX) {
        std::fprintf(stderr, "[tls] guest thread pointer slot is out of disp32 range\n");
        std::abort();
    }
    return {0x64, int32_t(off)};
#endif
}

void layoutStaticTls(const std::vector<Module*>& modules) {
    // Variant II: the first module's block sits directly below the TCB, later ones below it.
    uint64_t offset = 0;
    for (Module* m : modules) {
        const auto* seg = m->tlsSegment();
        if (!seg || seg->memsz == 0) continue;
        const uint64_t align = seg->align ? seg->align : 1;
        offset = alignUp(offset + seg->memsz, align);
        m->tlsOffset = offset;
        m->tlsIndex = uint32_t(gImages.size() + 1);
        gImages.push_back({m->base() + seg->vaddr, seg->filesz, offset});
        if (align > gStaticAlign) gStaticAlign = align;
    }
    gStaticSize = alignUp(offset, gStaticAlign);
}

void* createGuestTls() {
#ifdef _WIN32
    uint8_t* mem = static_cast<uint8_t*>(_aligned_malloc(gStaticSize + kTcbSize, gStaticAlign));
#else
    uint8_t* mem = static_cast<uint8_t*>(std::aligned_alloc(gStaticAlign, gStaticSize + kTcbSize));
#endif
    std::memset(mem, 0, gStaticSize + kTcbSize);
    uint8_t* tp = mem + gStaticSize;
    for (const auto& img : gImages) std::memcpy(tp - img.offset, img.init, img.initSize);
    *reinterpret_cast<void**>(tp) = tp; // tcb_self
    tGuestThreadPointer = tp;
#ifdef _WIN32
    if (gTlsSlot == TLS_OUT_OF_INDEXES) guestThreadPointerSlot();
    TlsSetValue(gTlsSlot, tp);
#endif
    return tp;
}

void* currentGuestTls() { return tGuestThreadPointer; }

void* guestTlsGetAddr(const TlsIndex* ti) {
    if (ti->module == 0 || ti->module > gImages.size()) {
        std::fprintf(stderr, "[tls] __tls_get_addr for unknown module %llu\n",
                     static_cast<unsigned long long>(ti->module));
        std::abort();
    }
    auto* tp = static_cast<uint8_t*>(tGuestThreadPointer);
    return tp - gImages[ti->module - 1].offset + ti->offset;
}

} // namespace host
