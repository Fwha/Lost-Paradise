#include "host/guest_abi.h"
#include "host/title_menu.h"

#include "host/display_menu.h"
#include "host/hooks.h"
#include "host/module.h"
#include "host/pc_settings.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

namespace host {

namespace {

constexpr uint32_t kDecompress = 0xe2ebc0; // SLLZ (dst, capacity, src): every compressed file
constexpr uint32_t kBuildItems = 0xb429b0; // fills the title's row kinds (self+0x338, count +0x340)
constexpr uint32_t kText = 0x84a320;       // text lookup (out, table, row) -> out, *out = the string
constexpr uint32_t kChosen = 0xb4f090;     // kind 7's confirm state (the unused attention screen)

constexpr uint64_t kKind = 7;
constexpr int kMaxItems = 8; // the rows' inline storage
// The rows the title shows: the layout places them at six anchors (its parts at 0xb0c0, 0x144
// apart: x 110, y 325 + 44 per row), and its script makes six label slots for them (group 1:
// one and 5 clones). The game adds 2, 5 or 6 rows (0xb429b0), so ours go in while there is room:
// both after 2, PC SETTINGS alone after 5, neither after 6 (the New Game+ menu).
constexpr int kMaxRows = 6;
constexpr uint32_t kDescTable = 7, kDescRow = kKind + 1;
constexpr char kGameDesc[] = "Read before starting the game.";
constexpr char kDesc[] = "Change PC display and frame rate settings.";
constexpr char kExitDesc[] = "Close the game.";

// EXIT GAME is kind 7 too, as the row tables the game indexes by kind (confirm states, label
// frames, descriptions) have no room for an 8th: an item is the kind and, above it, a word the
// game only tests for 0 (a disabled row). Ours is 2, which marks the row as EXIT GAME.
constexpr uint64_t kPcItem = kKind | 1ull << 32, kExitItem = kKind | 2ull << 32;

// DOWNLOAD INFO (kind 4) leaves the title for PC SETTINGS and EXIT GAME; the PC SETTINGS screen
// opened from the title gets a Download Info row instead, which closes that screen as Back does.
// The title's state for a closed settings screen (0xb4e170: it frees the screen, +0x2b0, and
// picks the next state) is then followed by DOWNLOAD INFO's (0xb4df10, kind 4's in the table at
// 0x246bc00), set as confirming its row did: the state at +0x218 (+0x228 set), the kind at
// +0x4850 and +0x438 set. Anything else clears the request (a new screen, the title rebuilt).
constexpr uint32_t kDownloadKind = 4;
constexpr uint32_t kDownloadState = 0xb4df10;
constexpr uint32_t kScreenClosed = 0xb4e170;
constexpr size_t kState = 0x218, kStateSet = 0x228, kBusy = 0x438, kChosenKind = 0x4850;
constexpr char kDownloadLabel[] = "Download Info";
constexpr char kDownloadDesc[] = "View Online Services and the download history.";
constexpr char kDownloadOpen[] = "Open";

// The title sequence (rdi of the states above).
constexpr size_t kItems = 0x338, kItemCount = 0x340;
constexpr size_t kMenu = 0x168; // its list menu: (kMenu, 1) -> the rows' list, the cursor at +0x20
constexpr uint32_t kListOf = 0x57daa0; // (menu, n) -> its nth list: the title menu's rows (1), a screen's (0)
constexpr size_t kCursor = 0x20;

// A row's label: the rows (+0x2a8) add each item with 0xb4bbb0 (rows, kind, enabled, flag),
// which gives their ui (+0x48) to 0xc281a0 (ui, kind, flag, alpha). That shows the kind's frame
// (the table at 0x1ff42d0) in the row's slot and keeps it in the ui's entries (+0x1a10, 16 bytes
// each, +0 the frame, count +0x1a18), which the cursor's highlight finds rows by (0xc283a0).
// 0xb4bbb0 is replaced: EXIT GAME's row then gets frame 8 instead of kind 7's 6.
constexpr uint32_t kAddRow = 0xb4bbb0;
constexpr uint32_t kShowLabel = 0xc281a0;
constexpr uint32_t kDimAlpha = 0x1fef994;  // float: a disabled row's label alpha
constexpr uint32_t kSlotPart = 0xb5aa60;   // (slot, id) -> its part with that id
constexpr uint32_t kSetFrame = 0xb588b0;   // (part, float frame)
constexpr uint32_t kSetShown = 0xb58aa0;   // (part, shown)
constexpr size_t kUi = 0x48, kEntries = 0x1a10, kEntryCount = 0x1a18, kEntrySize = 0x10;
constexpr size_t kSlotGroups = 0x1870, kSlotGroup = 0x430, kSlotList = 0x220, kSlotCount = 0x228,
                 kParts = 0x1888;
constexpr uint32_t kExitFrame = 8;
// Each row's NEW badge: group 8's slot, its part with the ui's part id 5. 0xc282e0 shows or hides
// it for a kind, finding the row by the kind's frame; EXIT GAME's is not kind 7's, so it is
// hidden here (left alone, it shows).
constexpr int kLabelGroup = 1, kBadgeGroup = 8;
constexpr uint32_t kLabelId = 0, kBadgeId = 5;

uint8_t* gEboot = nullptr;
uint8_t* gLayout = nullptr; // the title layout, as decompressed (then loaded in place)
uint8_t* gTitle = nullptr;  // the title sequence (the game thread's)
std::atomic<bool> gDownloadPending{false}; // Download Info chosen: the title opens DOWNLOAD INFO
int32_t (LPN_GUEST* gDecompress)(uint8_t*, uint32_t, const uint8_t*) = nullptr;
void (LPN_GUEST* gBuildItems)(uint8_t*) = nullptr;
void** (LPN_GUEST* gText)(void**, uint32_t, uint32_t) = nullptr;

template <class T> T& at(uint8_t* p, size_t off) { return *reinterpret_cast<T*>(p + off); }

// The item under the title menu's cursor (0 when there is none).
uint64_t cursorItem(uint8_t* title) {
    if (!title) return 0;
    auto list = reinterpret_cast<uint8_t* (LPN_GUEST*)(uint8_t*, uint32_t)>(gEboot + kListOf);
    const int32_t i = at<int32_t>(list(title + kMenu, 1), kCursor);
    if (i < 0 || i >= at<int64_t>(title, kItemCount)) return 0;
    return at<uint64_t*>(title, kItems)[i];
}

// LPN_TITLE_TRACE=1 (debugging): each hooked function's first call.
void traceFirst(const char* name, std::atomic<bool>& seen) {
    static const bool on = std::getenv("LPN_TITLE_TRACE") != nullptr;
    if (on && !seen.exchange(true)) std::fprintf(stderr, "[title] first call: %s\n", name);
}

uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
void putBe16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v >> 8), p[1] = uint8_t(v); }
void putBe32(uint8_t* p, uint32_t v) { putBe16(p, uint16_t(v >> 16)), putBe16(p + 2, uint16_t(v)); }
void putBeFloat(uint8_t* p, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    putBe32(p, v);
}

int32_t decompress(uint8_t* dst, uint32_t cap, const uint8_t* src) {
    const int32_t r = gDecompress(dst, cap, src);
    if (r < 0) return r;
    uint32_t size;
    std::memcpy(&size, src + 8, 4);
    if (src[4]) size = __builtin_bswap32(size);
    if (size > cap) return r;
    if (patchTitleMenuLayout(dst, size)) {
        gLayout = dst;
        std::fprintf(stderr, "[title] layout: PC SETTINGS and EXIT GAME rows added\n");
    }
    else if (patchTitleMenuAtlas(dst, size))
        std::fprintf(stderr, "[title] atlas: PC SETTINGS and EXIT GAME labels drawn\n");
    return r;
}

void buildItems(uint8_t* self) {
    static std::atomic<bool> seen{false};
    traceFirst("build items", seen);
    gBuildItems(self);
    gTitle = self;
    gDownloadPending = false;
    int64_t& n = at<int64_t>(self, kItemCount);
    static_assert(kMaxRows <= kMaxItems);
    if (n < 0 || n > kMaxItems) return;
    uint64_t* items = at<uint64_t*>(self, kItems);
    int64_t kept = 0;
    for (int64_t i = 0; i < n; ++i)
        if (uint32_t(items[i]) != kDownloadKind) items[kept++] = items[i];
    n = kept;
    if (n >= kMaxRows) return;
    at<uint64_t*>(self, kItems)[n++] = kPcItem;
    if (n < kMaxRows) at<uint64_t*>(self, kItems)[n++] = kExitItem;
}

// 0xb4bbb0, replaced: as the game's, then EXIT GAME's label frame.
void addRow(uint8_t* rows, uint32_t kind, uint32_t enabled, uint32_t flag) {
    uint8_t* ui = at<uint8_t*>(rows, kUi);
    if (!ui) return;
    const float alpha = enabled ? 0.0f : *reinterpret_cast<const float*>(gEboot + kDimAlpha);
    const uint64_t i = at<uint64_t>(ui, kEntryCount);
    // Never past the label slots (buildItems keeps to them): 0xc281a0 does not check.
    if (i >= at<uint64_t>(at<uint8_t*>(ui, kSlotGroups) + kSlotGroup, kSlotCount)) {
        std::fprintf(stderr, "[title] row %llu has no label slot\n", static_cast<unsigned long long>(i));
        return;
    }
    using ShowLabel = void (LPN_GUEST*)(uint8_t*, uint32_t, uint32_t, float);
    reinterpret_cast<ShowLabel>(gEboot + kShowLabel)(ui, kind, flag, alpha);
    if (kind != kKind || enabled != uint32_t(kExitItem >> 32) || at<uint64_t>(ui, kEntryCount) != i + 1) return;
    // As 0xc281a0 sets kind 7's frame, with ours: the slot's label part, its id from the ui's parts.
    at<uint32_t>(at<uint8_t*>(ui, kEntries) + kEntrySize * i, 0) = kExitFrame;
    uint8_t* parts = at<uint8_t*>(ui, kParts);
    auto part = [&](int group, uint32_t which) -> uint8_t* {
        uint8_t* g = at<uint8_t*>(ui, kSlotGroups) + kSlotGroup * group;
        if (i >= at<uint64_t>(g, kSlotCount)) return nullptr;
        using Id = uint32_t (LPN_GUEST*)(uint8_t*, uint32_t);
        Id* vt = *reinterpret_cast<Id**>(parts);
        const uint32_t id = vt[0x58 / 8](parts, vt[0x50 / 8](parts, which));
        using SlotPart = uint8_t* (LPN_GUEST*)(uint8_t*, uint32_t);
        return reinterpret_cast<SlotPart>(gEboot + kSlotPart)(at<uint8_t**>(g, kSlotList)[i], id);
    };
    if (uint8_t* label = part(kLabelGroup, kLabelId))
        reinterpret_cast<void (LPN_GUEST*)(uint8_t*, float)>(gEboot + kSetFrame)(label, float(kExitFrame));
    if (uint8_t* badge = part(kBadgeGroup, kBadgeId))
        reinterpret_cast<void (LPN_GUEST*)(uint8_t*, uint32_t)>(gEboot + kSetShown)(badge, 0);
}

void** text(void** out, uint32_t table, uint32_t row) {
    gText(out, table, row);
    if (table == kDescTable && row == kDescRow && *out && std::strcmp(static_cast<const char*>(*out), kGameDesc) == 0)
        *out = const_cast<char*>(cursorItem(gTitle) == kExitItem ? kExitDesc : kDesc);
    return out;
}

// PC SETTINGS opens the game's own settings screen ("option", 0x8b6170), as SETTINGS does, with
// the F11 menu's settings as its rows. The screen is data driven: a list of 0xa0-byte rows, each
// a label, a description, up to 15 value names and the value shown; the game's code draws and
// edits them by row. Its writes to the game's options for a row are undone here and the value
// goes to the display settings instead.
//   +0 u8 kind (0: a choice of names), +1 u8 field (in the options, at +8), +2 u8 id,
//   +8 label, +0x10 u32 value, +0x18 description, +0x20 names[15]
constexpr size_t kRowSize = 0xa0, kRowField = 1, kRowId = 2, kRowLabel = 8, kRowValue = 0x10, kRowDesc = 0x18,
                 kRowNames = 0x20, kMaxNames = 15;
// The screen: rows (+0xe0, capacity +0xe8, count +0xec), the values set so far (+0x140, by field;
// cancelling an edit puts the row's back), its state (+0xb0, as the title's).
constexpr size_t kRows = 0xe0, kRowCap = 0xe8, kRowCount = 0xec, kCommitted = 0x140, kScreenState = 0xb0;
constexpr uint32_t kSettingsState = 0xb4e0c0; // the title's SETTINGS confirm state: opens the screen
constexpr uint32_t kScreenInit = 0x8b63d0;    // builds the rows and the list
constexpr uint32_t kScreenDraw = 0x8b9490;    // each frame: the shown rows' text
constexpr uint32_t kValueChange = 0x8ba160;   // editing: the value moved (writes the option)
constexpr uint32_t kCommit = 0x8b9bc0;        // editing: confirmed
constexpr uint32_t kCancelEdit = 0x8b9d70;    // editing: cancelled (writes the option back)
constexpr uint32_t kEditState = 0x8ba330;     // the screen's state while a row is edited
constexpr uint32_t kListSize = 0x5828f0;      // (list, rows shown, rows)
constexpr uint32_t kOptions = 0x2638df8;      // -> the game's options (relocated)
constexpr size_t kOptionsSize = 0x58;         // what the screen edits, from +0
// Each id's value range, 20 bytes apart: +4 the last value. Editing shows that many + 1.
constexpr uint32_t kRanges = 0x23a2700;
constexpr size_t kRangeSize = 20, kRangeLast = 4;
// The ids and fields our rows take: ids of plain choices the screen has no special case for
// (not 0, 0xa, 0xb, 0xe..0x10; 8 is a plain one but a special field), fields within what the screen
// keeps (its +0x140 copy) and not 8.
constexpr uint8_t kPcIds[kMaxPcSettings] = {1, 2, 3, 4, 5, 6, 7, 9, 0xc, 0xd, 0x11, 0x12, 0x13};
constexpr uint8_t kPcFields[kMaxPcSettings] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13};

// The header: the title layout shows frame 5 of three lists for SETTINGS (rect 39); PC SETTINGS
// shows the menu label (rect 37) there instead, while its screen is up. The loaded layout is the
// decompressed buffer, its fields byte-swapped in place.
constexpr size_t kHeaderLists[] = {0x18bb0, 0x18bfc, 0x18c48};
constexpr size_t kHeaderRecord = 4 * 17; // frame 5's
constexpr uint16_t kRectHeader = 39, kRectLabel = 37;


void (LPN_GUEST* gSettingsState)(uint8_t*) = nullptr;
void (LPN_GUEST* gScreenInit)(uint8_t*) = nullptr;
void (LPN_GUEST* gScreenDraw)(uint8_t*) = nullptr;
void (LPN_GUEST* gValueChange)(uint8_t*, uint32_t, uint32_t) = nullptr;
uint32_t (LPN_GUEST* gCommit)(uint8_t*, void*) = nullptr;
uint32_t (LPN_GUEST* gCancelEdit)(uint8_t*, void*) = nullptr;

std::atomic<bool> gPcPending{false}; // PC SETTINGS chosen: the next screen built is ours
std::atomic<bool> gPcTitle{false};   // and chosen on the title (it gets the Download Info row)
int gDownloadRow = -1;               // the PC screen's Download Info row
void (LPN_GUEST* gScreenClosed)(uint8_t*) = nullptr;
// The screen's decide (vtable +0xd0) and Back (+0xd8); our screen's vtable is a copy with decide
// replaced (screenDecide), as the pause menu does for its box.
constexpr uint32_t kScreenDecide = 0x8b9810, kScreenBack = 0x8b9cb0;
constexpr size_t kVtDecide = 0xd0, kVtHead = 0x10, kVtSize = 0x400, kEventKind = 0x1c;
alignas(16) void* gScreenVtable[(kVtHead + kVtSize) / sizeof(void*)];
uint8_t* gPcScreen = nullptr;        // the game thread's, as all below
int gPcRows = 0;
int32_t gOldLast[kMaxPcSettings];
std::vector<std::string> gNames[kMaxPcSettings];

uint8_t* row(uint8_t* screen, int i) { return at<uint8_t*>(screen, kRows) + kRowSize * i; }
uint8_t* options() { return at<uint8_t*>(gEboot, kOptions) + 8; }

// The value ranges of the ids our rows take: ours while our screen is up, else the game's.
void setRanges(bool pc) {
    static bool saved = false;
    uint8_t* r = gEboot + kRanges + kRangeLast;
    if (!saved) {
        for (int i = 0; i < kMaxPcSettings; ++i) std::memcpy(&gOldLast[i], r + kRangeSize * kPcIds[i], 4);
        saved = true;
    }
    for (int i = 0; i < kMaxPcSettings; ++i) {
        const int32_t last = pc && i < gPcRows ? int32_t(gNames[i].size()) - 1 : pc && i == gDownloadRow ? 0 : gOldLast[i];
        writeData(r + kRangeSize * kPcIds[i], &last, 4);
    }
}

void setHeader(uint16_t rect) {
    uint8_t* s = gLayout;
    if (!s) return;
    // Still the loaded layout: our label's rect, now in native floats, and frame 5's records.
    const float want[4] = {1 / 1024.0f, 400 / 512.0f, 545 / 1024.0f, 446 / 512.0f};
    if (std::memcmp(s + 0x130 + kRectLabel * 16, want, 16) != 0) {
        std::fprintf(stderr, "[title] header: the layout is not where it was decompressed\n");
        return;
    }
    const uint16_t other = rect == kRectLabel ? kRectHeader : kRectLabel;
    for (size_t list : kHeaderLists) {
        const uint8_t* rec = s + list + kHeaderRecord;
        uint16_t frame, tex, r;
        // tex << 16 | rect, swapped as one u32
        std::memcpy(&frame, rec, 2), std::memcpy(&r, rec + 4, 2), std::memcpy(&tex, rec + 6, 2);
        if (frame != 5 || tex != 3 || (r != rect && r != other)) {
            std::fprintf(stderr, "[title] header: frame 5 is %u %u %u at +0x%zx\n", frame, tex, r, list);
            return;
        }
    }
    for (size_t list : kHeaderLists) std::memcpy(s + list + kHeaderRecord + 4, &rect, 2);
}

void settingsState(uint8_t* self) {
    static std::atomic<bool> seen{false};
    traceFirst("settings state", seen);
    const bool pc = gPcPending;
    setHeader(pc ? kRectLabel : kRectHeader);
    gSettingsState(self);
}

void chosen(uint8_t* self) {
    static std::atomic<bool> seen{false};
    traceFirst("chosen", seen);
    // EXIT GAME exits as the F11 menu's Exit Game does. The title has no Yes/No box of its own:
    // the game's (sequence 0x1b, which the pause menu opens) would need the title's states to
    // wait on it, as they do on their other children. The state stays this one until the window
    // thread exits, so it is asked once.
    if (cursorItem(self) == kExitItem) {
        static std::atomic<bool> asked{false};
        if (!asked.exchange(true)) {
            std::fprintf(stderr, "[title] EXIT GAME\n");
            setDisplayOption(kDisplayExit, 1);
        }
        return;
    }
    std::fprintf(stderr, "[title] PC SETTINGS\n");
    gPcPending = true;
    gPcTitle = true;
    settingsState(self);
}

// The screen's index under its cursor.
uint32_t screenCursor(uint8_t* screen) {
    uint8_t* list = reinterpret_cast<uint8_t* (LPN_GUEST*)(uint8_t*, uint32_t)>(gEboot + kListOf)(screen, 0);
    using Cursor = uint32_t (LPN_GUEST*)(uint8_t*);
    return (*reinterpret_cast<Cursor**>(list))[2](list);
}

uint32_t screenDecide(uint8_t* self, uint8_t* e) {
    using Handler = uint32_t (LPN_GUEST*)(uint8_t*, uint8_t*);
    if (self == gPcScreen && gDownloadRow >= 0 && at<uint32_t>(e, kEventKind) == 0 &&
        screenCursor(self) == uint32_t(gDownloadRow)) {
        std::fprintf(stderr, "[title] Download Info\n");
        gDownloadPending = true;
        return reinterpret_cast<Handler>(gEboot + kScreenBack)(self, e);
    }
    return reinterpret_cast<Handler>(gEboot + kScreenDecide)(self, e);
}

void screenClosed(uint8_t* self) {
    gScreenClosed(self);
    if (!gDownloadPending.exchange(false)) return;
    std::fprintf(stderr, "[title] DOWNLOAD INFO\n");
    at<uint32_t>(self, kChosenKind) = kDownloadKind;
    at<const void*>(self, kState) = gEboot + kDownloadState;
    at<uint64_t>(self, kState + 8) = 0;
    at<uint64_t>(self, kStateSet) = 1;
    at<uint32_t>(self, kBusy) = 1;
}

void screenInit(uint8_t* self) {
    static std::atomic<bool> seen{false};
    traceFirst("screen init", seen);
    gScreenInit(self);
    const bool pc = gPcPending.exchange(false);
    const bool fromTitle = gPcTitle.exchange(false);
    gDownloadPending = false;
    gPcScreen = pc ? self : nullptr;
    gDownloadRow = -1;
    if (!pc) return setRanges(false);
    const std::vector<PcSetting>& settings = pcSettings();
    const int n = std::min(int(settings.size()), kMaxPcSettings);
    bool ok = n > 0 && at<uint32_t>(self, kRowCap) >= uint32_t(n); // always: the game's own are more
    for (int i = 0; i < n && ok; ++i) {
        gNames[i] = settings[i].choices();
        if (gNames[i].size() > kMaxNames) gNames[i].resize(kMaxNames);
        ok = !gNames[i].empty(); // none without a window
    }
    if (!ok) {
        gPcScreen = nullptr;
        gPcRows = 0;
        return setRanges(false);
    }
    gPcRows = n;
    for (int i = 0; i < n; ++i) {
        uint8_t* r = row(self, i);
        std::memset(r, 0, kRowSize);
        r[kRowField] = kPcFields[i];
        r[kRowId] = kPcIds[i];
        at<const char*>(r, kRowLabel) = settings[i].label;
        at<const char*>(r, kRowDesc) = settings[i].description;
        for (size_t k = 0; k < gNames[i].size(); ++k) at<const char*>(r, kRowNames + 8 * k) = gNames[i][k].c_str();
        const uint32_t v = uint32_t(std::clamp(settings[i].value(), 0, int(gNames[i].size()) - 1));
        at<uint32_t>(r, kRowValue) = v;
        self[kCommitted + kPcFields[i]] = uint8_t(v);
    }
    int rows = n;
    void** vt = at<void**>(self, 0);
    if (fromTitle && n < kMaxPcSettings && at<uint32_t>(self, kRowCap) > uint32_t(n) &&
        vt[kVtDecide / sizeof(void*)] == gEboot + kScreenDecide) {
        // Download Info: one name, never edited (screenDecide takes it).
        uint8_t* r = row(self, n);
        std::memset(r, 0, kRowSize);
        r[kRowField] = kPcFields[n];
        r[kRowId] = kPcIds[n];
        at<const char*>(r, kRowLabel) = kDownloadLabel;
        at<const char*>(r, kRowDesc) = kDownloadDesc;
        at<const char*>(r, kRowNames) = kDownloadOpen;
        self[kCommitted + kPcFields[n]] = 0;
        gDownloadRow = n;
        rows = n + 1;
        std::memcpy(gScreenVtable, reinterpret_cast<uint8_t*>(vt) - kVtHead, sizeof(gScreenVtable));
        void** ours = gScreenVtable + kVtHead / sizeof(void*);
        ours[kVtDecide / sizeof(void*)] = guestEntry<&screenDecide>();
        at<void**>(self, 0) = ours;
    }
    at<uint32_t>(self, kRowCount) = uint32_t(rows);
    setRanges(true);
    auto list = reinterpret_cast<uint8_t* (LPN_GUEST*)(uint8_t*, uint32_t)>(gEboot + kListOf);
    auto listSize = reinterpret_cast<void (LPN_GUEST*)(uint8_t*, uint32_t, uint32_t)>(gEboot + kListSize);
    listSize(list(self, 0), 0xb, uint32_t(rows));
}

// Keeps the rows as the settings are, which the F11 menu can change too (not while one is edited).
void screenDraw(uint8_t* self) {
    static std::atomic<bool> seen{false};
    traceFirst("screen draw", seen);
    if (self == gPcScreen && at<uint64_t>(self, kScreenState) != uint64_t(gEboot + kEditState))
        for (int i = 0; i < gPcRows; ++i) {
            const int v = pcSettings()[i].value();
            if (v < 0 || v >= int(gNames[i].size())) continue;
            at<uint32_t>(row(self, i), kRowValue) = uint32_t(v);
            self[kCommitted + kPcFields[i]] = uint8_t(v);
        }
    gScreenDraw(self);
}

// The game writes the option of the row's field; for our rows that is put back.
struct KeepOptions {
    explicit KeepOptions(uint8_t* screen) : on(screen == gPcScreen) {
        if (on) std::memcpy(saved, options(), kOptionsSize);
    }
    ~KeepOptions() {
        if (on) std::memcpy(options(), saved, kOptionsSize);
    }
    bool on;
    uint8_t saved[kOptionsSize];
};

void valueChange(uint8_t* self, uint32_t a, uint32_t b) {
    static std::atomic<bool> seen{false};
    traceFirst("value change", seen);
    KeepOptions keep(self);
    gValueChange(self, a, b);
}

uint32_t cancelEdit(uint8_t* self, void* e) {
    static std::atomic<bool> seen{false};
    traceFirst("cancel edit", seen);
    KeepOptions keep(self);
    return gCancelEdit(self, e);
}

uint32_t commit(uint8_t* self, void* e) {
    static std::atomic<bool> seen{false};
    traceFirst("commit", seen);
    const uint32_t r = gCommit(self, e);
    if (self == gPcScreen)
        for (int i = 0; i < gPcRows; ++i) {
            const int v = int(at<uint32_t>(row(self, i), kRowValue));
            if (v != pcSettings()[i].value()) pcSettings()[i].choose(v);
        }
    return r;
}

} // namespace

void openPcSettingsNext() {
    if (gEboot) gPcPending = true;
}

bool patchTitleMenuLayout(uint8_t* s, size_t size) {
    constexpr size_t kSize = 0x19800, kRects = 0x130, kSlots = 0x7be0, kSlotStride = 0x100, kLists = 0x18568,
                     kOldList = 88, kNewList = 112, kSpare = 0x19700;
    constexpr int kRectLabel = 37, kRectExit = 13, kRectSettings = 28, kRectNewGame = 23, kOldFrames = 7;
    static const uint8_t kSlot[] = {0x00, 0x00, 0x00, 0x06, 0x00, 0x07, 0x00, 0x00};
    static const uint8_t kSlotValue[] = {0x3e, 0x2a, 0xaa, 0xab};
    static const uint8_t kRect37[] = {0x3f, 0x7a, 0x80, 0x00, 0, 0, 0, 0, 0x3f, 0x80, 0, 0, 0x3d, 0x30, 0, 0};
    // Rect 13 (920,328 - 936,352): as 37, no sprite or frame list of the layout uses it.
    static const uint8_t kRect13[] = {0x3f, 0x66, 0, 0, 0x3f, 0x24, 0, 0, 0x3f, 0x6a, 0, 0, 0x3f, 0x30, 0, 0};
    static const uint8_t kRect28[] = {0x3a, 0x80, 0, 0, 0x3f, 0x0d, 0x80, 0, 0x3f, 0x08, 0x40, 0, 0x3f, 0x24, 0x80, 0};
    // Frame k -> record k: NG/CO, CONTINUE, ENDLESS EDEN, DOWNLOAD INFO, REMINISCE, SETTINGS,
    // NEW GAME (a kind's frame, from the table at 0x1ff42d0: 0 -> 7 (the last), 1..6 -> 0..5, 7 -> 6).
    static const uint8_t kList[kOldList] = {
        0x00, 0x00, 0x00, 0x08, 0x00, 0x01, 0x00, 0x0a, 0x00, 0x02, 0x00, 0x0c, 0x00, 0x03, 0x00, 0x0e,
        0x00, 0x04, 0x00, 0x10, 0x00, 0x05, 0x00, 0x12, 0x00, 0x06, 0x00, 0x14, 0xff, 0xff, 0x00, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x00, 0x03, 0x00, 0x16, 0x00, 0x01, 0x08, 0x08, 0x00, 0x03, 0x00, 0x18,
        0x00, 0x02, 0x08, 0x08, 0x00, 0x03, 0x00, 0x19, 0x00, 0x03, 0x08, 0x08, 0x00, 0x03, 0x00, 0x1a,
        0x00, 0x04, 0x08, 0x08, 0x00, 0x03, 0x00, 0x1b, 0x00, 0x05, 0x08, 0x08, 0x00, 0x03, 0x00, 0x1c,
        0x00, 0x06, 0x08, 0x08, 0x00, 0x03, 0x00, 0x17};

    if (size != kSize || std::memcmp(s, "SURB", 4) != 0 || s[5] != 1 || be32(s + 0x18) != kRects ||
        be32(s + 0x24) != 43 || std::memcmp(s + kRects + kRectLabel * 16, kRect37, 16) != 0 ||
        std::memcmp(s + kRects + kRectExit * 16, kRect13, 16) != 0 ||
        std::memcmp(s + kRects + kRectSettings * 16, kRect28, 16) != 0)
        return false;
    for (int i = 0; i < kMaxItems; ++i) {
        const uint8_t* slot = s + kSlots + kSlotStride * i;
        if (std::memcmp(slot, kSlot, 8) != 0 || be32(slot + 8) != kLists + kOldList * i ||
            std::memcmp(slot + 12, kSlotValue, 4) != 0 || std::memcmp(s + kLists + kOldList * i, kList, kOldList) != 0)
            return false;
    }
    for (size_t i = kSpare; i < kSize; ++i)
        if (s[i]) return false;

    // The labels' rects, in the atlas bands patchTitleMenuAtlas draws in.
    uint8_t* r = s + kRects + kRectLabel * 16;
    putBeFloat(r, 1 / 1024.0f), putBeFloat(r + 4, 400 / 512.0f), putBeFloat(r + 8, 545 / 1024.0f),
        putBeFloat(r + 12, 446 / 512.0f);
    r = s + kRects + kRectExit * 16;
    putBeFloat(r, 479 / 1024.0f), putBeFloat(r + 4, 330 / 512.0f), putBeFloat(r + 8, 1023 / 1024.0f),
        putBeFloat(r + 12, 376 / 512.0f);

    // Each slot gets its own list (the loader byte-swaps each in place, so none may be shared):
    // 9 keys, then the old frames 0..5, frame 6 the PC SETTINGS label, frame 7 NEW GAME, frame 8
    // the EXIT GAME label (addRow gives it to its row). Six take the old lists' place, the last
    // two the zeros after the data.
    uint8_t list[kNewList] = {};
    constexpr int kFrames = kOldFrames + 2;
    constexpr int kInPlace = kOldList * kMaxItems / kNewList;
    static_assert(kNewList * (kMaxItems - kInPlace) <= kSize - kSpare);
    constexpr int kRecords = kFrames + 1; // in 4-byte units: the keys and their terminator
    for (int k = 0; k < kFrames; ++k) putBe16(list + 4 * k, uint16_t(k)), putBe16(list + 4 * k + 2, uint16_t(kRecords + 2 * k));
    putBe16(list + 4 * kFrames, 0xffff);
    std::memcpy(list + 4 * kRecords, kList + 4 * kOldFrames + 4, 8 * 6);
    uint8_t* rec = list + 4 * kRecords + 8 * 6;
    putBe16(rec, 6), putBe16(rec + 2, 0x0808), putBe16(rec + 4, 3), putBe16(rec + 6, kRectLabel);
    putBe16(rec + 8, 7), putBe16(rec + 10, 0x0808), putBe16(rec + 12, 3), putBe16(rec + 14, kRectNewGame);
    putBe16(rec + 16, kExitFrame), putBe16(rec + 18, 0x0808), putBe16(rec + 20, 3), putBe16(rec + 22, kRectExit);
    static_assert(4 * kRecords + 8 * kFrames == kNewList);
    for (int i = 0; i < kMaxItems; ++i) {
        const uint32_t off = uint32_t(i < kInPlace ? kLists + kNewList * i : kSpare + kNewList * (i - kInPlace));
        std::memcpy(s + off, list, kNewList);
        uint8_t* slot = s + kSlots + kSlotStride * i;
        // The row's frame comes from the game's table at 0x1ff42d0 (kind 0, NEW GAME: 7; kind k:
        // k - 1) and is clamped to the slot's last frame (+2) and its reciprocal (+12). Left at 6
        // and 1/6, NEW GAME showed frame 6, the PC SETTINGS label.
        putBe16(slot + 2, kFrames - 1);
        putBe16(slot + 4, kFrames);
        putBe32(slot + 8, off);
        putBeFloat(slot + 12, 1.0f / (kFrames - 1));
    }
    return true;
}

bool patchTitleMenuAtlas(uint8_t* d, size_t size) {
    constexpr int kW = 1024, kH = 512, kHeader = 128, kRowH = 46, kX0 = 1, kBand = 400, kBandW = 544;
    constexpr int kCarryover = 1, kNewGame = 48, kContinue = 95, kReminisce = 236, kSettings = 283, kSettingsW = 320;
    constexpr uint64_t kSource = 0xc5af36fab8684185; // FNV-1a of rows 95..329, x 1..544: the letters used
    constexpr uint64_t kSource2 = 0x347c94ecaf18d5c8; // and of rows 1..94 (NEW GAME/DATA CARRYOVER, NEW GAME)
    // EXIT GAME's band: right of the NEW badge, above the glow at the bottom.
    constexpr int kExitX = 479, kExitY = 330;
    // The letters lean right: a glyph's columns shift by this much per row.
    static const int8_t kSlant[kRowH] = {11, 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4,
                                         4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2, -3};

    if (size != size_t(kHeader) + kW * kH * 4 || std::memcmp(d, "DDS ", 4) != 0) return false;
    uint32_t h, w, bits;
    std::memcpy(&h, d + 12, 4), std::memcpy(&w, d + 16, 4), std::memcpy(&bits, d + 88, 4);
    if (h != uint32_t(kH) || w != uint32_t(kW) || bits != 32) return false;
    auto px = [&](int x, int y) { return d + kHeader + (size_t(y) * kW + x) * 4; };
    auto fnv = [&](int y0, int y1) {
        uint64_t hash = 0xcbf29ce484222325;
        for (int y = y0; y <= y1; ++y)
            for (const uint8_t *p = px(kX0, y), *e = px(kX0 + kBandW, y); p < e; ++p) hash = (hash ^ *p) * 0x100000001b3;
        return hash;
    };
    if (fnv(kContinue, kSettings + kRowH) != kSource || fnv(kCarryover, kContinue - 1) != kSource2) return false;
    for (int y = 0; y < kRowH; ++y)
        for (int x = 0; x < kBandW; ++x)
            if (px(kX0 + x, kBand + y)[3] || px(kExitX + x, kExitY + y)[3]) return false;

    // x is from the label's left edge (atlas x 1); a pixel is kept where it is the more opaque.
    auto paste = [&](int toX, int y, const uint8_t* from) {
        uint8_t* to = px(kX0 + toX, kBand + y);
        if (from[3] > to[3]) std::memcpy(to, from, 4);
    };
    for (int y = 0; y < kRowH; ++y) {
        const int sh = kSlant[y];
        // P: REMINISCE's R without its leg; C: CONTINUE's.
        for (int x = 3 + sh; x < 28 + sh; ++x)
            if (y < 25 || x < 14 + sh) paste(2 + x - 3, y, px(kX0 + x, kReminisce + y));
        for (int x = 3 + sh; x < 27 + sh; ++x) paste(28 + x - 3, y, px(kX0 + x, kContinue + y));
        for (int x = 0; x < kSettingsW; ++x) paste(62 + x, y, px(kX0 + x, kSettings + y));
    }

    // EXIT GAME, by columns with the slant taken out (a glyph's column c is at x c + kSlant[y]
    // of its row). E, I, T: SETTINGS'; X: CARRYOVER's V, each stroke's rows shifted to an X's
    // slope; GAME: NEW GAME's.
    auto put = [&](int c, int y, const uint8_t* from) {
        uint8_t* to = px(kExitX + c + kSlant[y], kExitY + y);
        if (from[3] > to[3]) std::memcpy(to, from, 4);
    };
    auto src = [&](int row, int c, int y) { return px(kX0 + c + kSlant[y], row + y); };
    auto glyph = [&](int row, int c0, int c1, int to) {
        for (int y = 0; y < kRowH; ++y)
            for (int c = c0; c < c1; ++c) put(to + c - c0, y, src(row, c, y));
    };
    int col = 3;
    glyph(kSettings, 26, 46, col), col += 21; // E
    for (int y = 0; y < kRowH; ++y) {
        // The V's strokes are 8 wide; the cap height is rows 8..35.
        const int shift = int(std::lround((y - 8) * 16 / 27.0));
        const int left = 463 + int(std::lround((y - 10) * 7 / 24.0)), right = 486 - int(std::lround((y - 10) * 8 / 24.0));
        for (int k = 0; k < 8; ++k) {
            put(col + 2 + shift + k, y, src(kCarryover, left - 1 + k, y));
            put(col + 16 - shift + k, y, src(kCarryover, right - 8 + k, y));
        }
    }
    col += 27;
    glyph(kSettings, 93, 103, col), col += 11; // I
    glyph(kSettings, 69, 93, col), col += 24;  // T
    glyph(kNewGame, 94, 202, col + 8);         // GAME, after a space
    return true;
}

bool patchTitleMenu(Module& eboot) {
    if (const char* e = std::getenv("LPN_PC_SETTINGS"); e && std::strcmp(e, "0") == 0) return false;
    Hook hooks[] = {
        {kDecompress, {0x41, 0x57, 0x41, 0x56, 0x53, 0x81, 0x3a, 0x53, 0x4c, 0x4c, 0x5a, 0xb8, 0x05, 0x40, 0x00, 0x80},
         guestEntry<&decompress>(), reinterpret_cast<void**>(&gDecompress)},
        {kBuildItems, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xfb},
         guestEntry<&buildItems>(), reinterpret_cast<void**>(&gBuildItems)},
        {kText, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xff},
         guestEntry<&text>(), reinterpret_cast<void**>(&gText)},
        {kChosen, {0xc5, 0xfa, 0x7e, 0x05, 0x40, 0x00, 0x84, 0x01, 0xc5, 0xfa, 0x7e, 0x0d},
         guestEntry<&chosen>(), nullptr},
        {kAddRow, {0x48, 0x8b, 0x7f, 0x48, 0x48, 0x85, 0xff, 0x74, 0x0f, 0x85, 0xd2, 0x74, 0x0c},
         guestEntry<&addRow>(), nullptr},
        {kScreenClosed, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0xbb, 0xa8, 0x02, 0x00, 0x00},
         guestEntry<&screenClosed>(), reinterpret_cast<void**>(&gScreenClosed)},
        {kSettingsState, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0xbb, 0xa8, 0x02, 0x00, 0x00},
         guestEntry<&settingsState>(), reinterpret_cast<void**>(&gSettingsState)},
        {kScreenInit, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53},
         guestEntry<&screenInit>(), reinterpret_cast<void**>(&gScreenInit)},
        {kScreenDraw, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53},
         guestEntry<&screenDraw>(), reinterpret_cast<void**>(&gScreenDraw)},
        {kValueChange, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50},
         guestEntry<&valueChange>(), reinterpret_cast<void**>(&gValueChange)},
        {kCommit, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x31, 0xf6},
         guestEntry<&commit>(), reinterpret_cast<void**>(&gCommit)},
        {kCancelEdit, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x31, 0xf6},
         guestEntry<&cancelEdit>(), reinterpret_cast<void**>(&gCancelEdit)},
    };
    uint8_t* base = eboot.base();
    // The screen's decide and Back, called as they are from our copy of its vtable.
    static const uint8_t kDecide[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x8b, 0x4e, 0x1c, 0x48, 0x89, 0xfb};
    static const uint8_t kBack[] = {0x8b, 0x4e, 0x1c, 0x83, 0xf9, 0x01, 0x74, 0x27, 0x31, 0xc0, 0x85, 0xc9};
    if (std::memcmp(base + kScreenDecide, kDecide, sizeof(kDecide)) != 0 ||
        std::memcmp(base + kScreenBack, kBack, sizeof(kBack)) != 0) {
        std::fprintf(stderr, "[title] unexpected eboot code at the settings screen: not patched\n");
        return false;
    }
    if (!installHooks(base, hooks, std::size(hooks), "title")) return false;
    gEboot = base;
    std::fprintf(stderr, "[title] PC SETTINGS and EXIT GAME rows (DOWNLOAD INFO in PC SETTINGS)\n");
    return true;
}

} // namespace host
