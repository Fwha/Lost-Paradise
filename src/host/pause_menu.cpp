#include "host/guest_abi.h"
#include "host/pause_menu.h"

#include "host/hooks.h"
#include "host/module.h"
#include "host/title_menu.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace host {

namespace {

// The Load/Settings list (CPause..., vtable 0x232caf0) is a generic list menu: its build
// (0x893440) adds each row with addRow, a row's data pointing at its action, the sequence its
// confirm state (0x893ab0) opens: Load 0x500000001, Settings 0x12 (the settings screen), Return
// to Title 0xc0000001b. PC Settings is a row after Settings with an action of its own, also 0x12;
// confirming it sets the next settings screen to be the PC one.
constexpr uint32_t kAddRow = 0x580a90;          // (list, text, description, enabled, data)
constexpr uint32_t kAfterSettings = 0x8935a5;   // the build's return from adding Settings
constexpr uint32_t kConfirm = 0x893ab0;         // the list's confirm state
constexpr uint32_t kListOf = 0x57daa0;          // (menu, 0) -> its cursor list
// Rows: 0x20 bytes each from +0x98, count +0xa4; a row's data at +0x18.
constexpr size_t kRows = 0x98, kRowCount = 0xa4, kRowSize = 0x20, kRowData = 0x18;

constexpr char kLabel[] = "PC Settings";
constexpr char kNoDesc[] = "";
const uint64_t kPcAction = 0x12; // read as the settings screen's sequence; its address marks our row

uint8_t* gEboot = nullptr;
void (LPN_GUEST* gAddRow)(uint8_t*, const char*, const char*, uint32_t, const void*) = nullptr;
void (LPN_GUEST* gConfirm)(uint8_t*) = nullptr;

template <class T> T& at(uint8_t* p, size_t off) { return *reinterpret_cast<T*>(p + off); }

void addRow(uint8_t* self, const char* text, const char* desc, uint32_t enabled, const void* data) {
    gAddRow(self, text, desc, enabled, data);
    if (LPN_GUEST_RETURN_ADDRESS() == gEboot + kAfterSettings) gAddRow(self, kLabel, kNoDesc, 1, &kPcAction);
}

void confirm(uint8_t* self) {
    auto listOf = reinterpret_cast<uint8_t* (LPN_GUEST*)(uint8_t*, uint32_t)>(gEboot + kListOf);
    uint8_t* list = listOf(self, 0);
    using Cursor = uint32_t (LPN_GUEST*)(uint8_t*);
    const uint32_t i = (*reinterpret_cast<Cursor**>(list))[2](list);
    if (i < at<uint32_t>(self, kRowCount) && at<const void*>(at<uint8_t*>(self, kRows) + kRowSize * i, kRowData) == &kPcAction) {
        std::fprintf(stderr, "[pause] PC Settings\n");
        openPcSettingsNext();
    }
    gConfirm(self);
}

} // namespace

bool patchPauseMenu(Module& eboot) {
    if (const char* e = std::getenv("LPN_PC_SETTINGS"); e && std::strcmp(e, "0") == 0) return false;
    const Hook hooks[] = {
        {kAddRow, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54},
         guestHook<&addRow>(), reinterpret_cast<void**>(&gAddRow)},
        {kConfirm, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x31, 0xf6, 0x48, 0x89, 0xfb},
         guestEntry<&confirm>(), reinterpret_cast<void**>(&gConfirm)},
    };
    uint8_t* base = eboot.base();
    gEboot = base;
    if (!installHooks(base, hooks, std::size(hooks), "pause")) return false;
    std::fprintf(stderr, "[pause] PC Settings row in the Load/Settings menu\n");
    return true;
}

} // namespace host
