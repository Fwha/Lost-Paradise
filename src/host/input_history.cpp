#include "host/guest_abi.h"
#include "host/input_history.h"

#include "host/display_settings.h"
#include "host/module.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "host/mman.h"
#include "common/compat.h"
#include <unistd.h>

namespace host {

namespace {

constexpr uint32_t kFn = 0x3f01f0;         // appends one entry (the game's thread, per actor update)
constexpr uint32_t kGameTime = 0x247f150;  // double, the time stamped on each entry
constexpr uint32_t kFrameDt = 0x24860e0;   // float, the frame clock's step (0x2486080 + 0x60)
constexpr int kEntries = 60;
constexpr float kStep = 1.0f / 60.0f;

// Offsets in the actor. The live state (0x20 bytes at 0x1458) is filled by the input source;
// entries are a copy of it plus the time, 0x30 bytes apart.
constexpr size_t kSource = 0x2108, kState = 0x1458, kPressed = 0x1460, kHeld = 0x1464, kReleased = 0x1468,
                 kRing = 0x1478, kEntrySize = 0x30, kEntryTime = 0x20, kNext = 0x1fb9, kOldest = 0x1fba;

struct Pending {
    float time = 0;        // game time since the last entry
    uint32_t pressed = 0;  // bits seen in frames not given an entry yet
    uint32_t released = 0;
    uint32_t held = 0;     // the previous frame's held buttons
};
std::mutex gM;
std::unordered_map<uint8_t*, Pending> gPending;
uint8_t* gEboot = nullptr;

template <class T> T& at(uint8_t* p, size_t off) { return *reinterpret_cast<T*>(p + off); }

void appendEntry(uint8_t* actor) {
    // The input source fills the live state: vtable +0x68 (source, &state).
    void* source = at<void*>(actor, kSource);
    auto read = reinterpret_cast<void (LPN_GUEST*)(void*, void*)>((*reinterpret_cast<void***>(source))[0x68 / 8]);
    read(source, actor + kState);

    int next = int8_t(at<uint8_t>(actor, kNext));
    const int last = next > 0 ? next - 1 : kEntries - 1;
    const uint32_t held = at<uint32_t>(actor, kHeld);
    uint8_t entry[0x20];
    if (frameClockUncapped()) {
        std::lock_guard lock(gM);
        Pending& p = gPending[actor];
        at<uint32_t>(actor, kReleased) = ~held & p.held; // against the previous frame, as at 60
        p.held = held;
        p.pressed |= at<uint32_t>(actor, kPressed);
        p.released |= at<uint32_t>(actor, kReleased);
        p.time += *reinterpret_cast<const float*>(gEboot + kFrameDt);
        if (p.time < kStep * 0.999f) return;
        p.time = std::min(p.time - kStep, kStep);
        // The entry gets every press and release since the last one; the live state keeps
        // this frame's own for code that reads it directly.
        std::memcpy(entry, actor + kState, sizeof entry);
        std::memcpy(entry + (kPressed - kState), &p.pressed, 4);
        std::memcpy(entry + (kReleased - kState), &p.released, 4);
        p.pressed = p.released = 0;
    } else {
        // The game's own: released = held in the last entry (the previous frame) but not now.
        at<uint32_t>(actor, kReleased) = ~held & at<uint32_t>(actor, kRing + last * kEntrySize + (kHeld - kState));
        std::memcpy(entry, actor + kState, sizeof entry);
    }
    uint8_t* e = actor + kRing + next * kEntrySize;
    std::memcpy(e, entry, sizeof entry);
    std::memcpy(e + kEntryTime, gEboot + kGameTime, 8);
    if (++next == kEntries) next = 0;
    at<uint8_t>(actor, kNext) = uint8_t(next);
    if (at<uint8_t>(actor, kOldest) == uint8_t(next)) {
        int oldest = next + 1;
        if (oldest == kEntries) oldest = 0;
        at<uint8_t>(actor, kOldest) = uint8_t(oldest);
    }
}

} // namespace

bool patchInputHistory(Module& eboot) {
    static const uint8_t kWas[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0xbb, 0x08, 0x21};
    uint8_t* fn = eboot.base() + kFn;
    if (std::memcmp(fn, kWas, sizeof kWas) != 0) {
        std::fprintf(stderr, "[input] unexpected eboot code at 0x%x: pad history left as the game's\n", kFn);
        return false;
    }
    gEboot = eboot.base();
    // movabs $appendEntry, %rax; jmp *%rax
    uint8_t jump[12] = {0x48, 0xb8};
    const uint64_t target = reinterpret_cast<uint64_t>(guestEntry<&appendEntry>());
    std::memcpy(jump + 2, &target, 8);
    jump[10] = 0xff, jump[11] = 0xe0;
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    void* start = reinterpret_cast<void*>(uintptr_t(fn) & ~(page - 1));
    mprotect(start, page * 2, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::memcpy(fn, jump, sizeof jump);
    mprotect(start, page * 2, PROT_READ | PROT_EXEC);
    return true;
}

} // namespace host
