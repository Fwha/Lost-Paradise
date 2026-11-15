#include "host/guest_abi.h"
#include "host/pause_menu.h"

#include "host/display_menu.h"
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
constexpr uint32_t kAfterTitle = 0x893652;      // and from adding Return to Title
constexpr uint32_t kConfirm = 0x893ab0;         // the list's confirm state
constexpr uint32_t kListOf = 0x57daa0;          // (menu, 0) -> its cursor list
// Rows: 0x20 bytes each from +0x98, count +0xa4; a row's data at +0x18.
constexpr size_t kRows = 0x98, kRowCount = 0xa4, kRowSize = 0x20, kRowData = 0x18;

// Exit Game: Return to Title's action is sequence 0x1b, the game's Yes/No box (CMessageBox, made
// by the menu factory 0x58f690, vtable 0x230e1f0), with message 0xc of the box's table (Yes, No,
// "Return to title screen?"). The confirm state opens it as the list's child (0x57cc30 keeps it
// at +0x60) and makes it a Yes/No one (+0x160 = 1, No first). It shows its message from +0x100 each frame
// (0x58acc0), so ours is put there once it is open. Its decide (vtable +0xd0, 0x58a570) stores
// the answer at +0xa0, 1 for Yes; the list's child-closed handler (0x8937f0) then fades to the
// title for a row whose action is 0x1b and a Yes. For our row a Yes exits instead, as the F11
// menu's Exit Game does, and the answer is cleared so the list does not leave for the title.
// Only our box's decide is ours: it gets a copy of the vtable with that entry replaced (a hook
// at 0x58a570 would be every box's, and its first instructions load rax, which the hooks'
// jump back uses).
constexpr uint32_t kBoxDecide = 0x58a570;
constexpr size_t kNewChild = 0x60, kBoxText = 0x100, kBoxAnswer = 0xa0; // the child 0x57cc30 opened
constexpr size_t kVtDecide = 0xd0;
constexpr size_t kVtHead = 0x10, kVtSize = 0x400; // offset to top and type info; the entries copied

constexpr char kLabel[] = "PC Settings";
constexpr char kExitLabel[] = "Exit Game";
constexpr char kExitText[] = "Exit the game?";
constexpr char kNoDesc[] = "";
const uint64_t kPcAction = 0x12; // read as the settings screen's sequence; its address marks our row
const uint64_t kExitAction = 0xc0000001b; // Return to Title's box; its address marks our row

uint8_t* gEboot = nullptr;
void (LPN_GUEST* gAddRow)(uint8_t*, const char*, const char*, uint32_t, const void*) = nullptr;
void (LPN_GUEST* gConfirm)(uint8_t*) = nullptr;
alignas(16) void* gBoxVtable[(kVtHead + kVtSize) / sizeof(void*)]; // the game thread's

template <class T> T& at(uint8_t* p, size_t off) { return *reinterpret_cast<T*>(p + off); }

void addRow(uint8_t* self, const char* text, const char* desc, uint32_t enabled, const void* data) {
    const void* from = LPN_GUEST_RETURN_ADDRESS();
    gAddRow(self, text, desc, enabled, data);
    if (from == gEboot + kAfterSettings) gAddRow(self, kLabel, kNoDesc, 1, &kPcAction);
    else if (from == gEboot + kAfterTitle) gAddRow(self, kExitLabel, kNoDesc, 1, &kExitAction);
}

uint32_t boxDecide(uint8_t* self, void* e);

void confirm(uint8_t* self) {
    auto listOf = reinterpret_cast<uint8_t* (LPN_GUEST*)(uint8_t*, uint32_t)>(gEboot + kListOf);
    uint8_t* list = listOf(self, 0);
    using Cursor = uint32_t (LPN_GUEST*)(uint8_t*);
    const uint32_t i = (*reinterpret_cast<Cursor**>(list))[2](list);
    const void* data =
        i < at<uint32_t>(self, kRowCount) ? at<const void*>(at<uint8_t*>(self, kRows) + kRowSize * i, kRowData) : nullptr;
    if (data == &kPcAction) {
        std::fprintf(stderr, "[pause] PC Settings\n");
        openPcSettingsNext();
    }
    gConfirm(self);
    if (data != &kExitAction) return;
    uint8_t* box = at<uint8_t*>(self, kNewChild);
    if (!box) return;
    void** vt = at<void**>(box, 0);
    if (vt[kVtDecide / sizeof(void*)] != gEboot + kBoxDecide) {
        std::fprintf(stderr, "[pause] Exit Game: not the expected box\n");
        return; // left as Return to Title's
    }
    std::memcpy(gBoxVtable, reinterpret_cast<uint8_t*>(vt) - kVtHead, sizeof(gBoxVtable));
    void** ours = gBoxVtable + kVtHead / sizeof(void*);
    ours[kVtDecide / sizeof(void*)] = guestEntry<&boxDecide>();
    at<void**>(box, 0) = ours;
    at<const char*>(box, kBoxText) = kExitText;
    std::fprintf(stderr, "[pause] Exit Game?\n");
}

uint32_t boxDecide(uint8_t* self, void* e) {
    const uint32_t r = reinterpret_cast<uint32_t (LPN_GUEST*)(uint8_t*, void*)>(gEboot + kBoxDecide)(self, e);
    if (at<uint32_t>(self, kBoxAnswer) != 1) return r; // No: back to the list
    at<uint32_t>(self, kBoxAnswer) = 0;
    std::fprintf(stderr, "[pause] Exit Game\n");
    setDisplayOption(kDisplayExit, 1); // the window thread exits as from the F11 menu
    return r;
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
    // The box's decide, called as it is from our copy of its vtable.
    static const uint8_t kDecide[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x43, 0x70};
    if (std::memcmp(base + kBoxDecide, kDecide, sizeof(kDecide)) != 0) {
        std::fprintf(stderr, "[pause] unexpected eboot code at 0x%x: not patched\n", kBoxDecide);
        return false;
    }
    gEboot = base;
    if (!installHooks(base, hooks, std::size(hooks), "pause")) return false;
    std::fprintf(stderr, "[pause] PC Settings and Exit Game rows in the Load/Settings menu\n");
    return true;
}

} // namespace host
