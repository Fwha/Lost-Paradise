#include "host/title_menu.h"

#include "host/display_menu.h"
#include "host/module.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace host {

namespace {

constexpr uint32_t kDecompress = 0xe2ebc0; // SLLZ (dst, capacity, src): every compressed file
constexpr uint32_t kBuildItems = 0xb429b0; // fills the title's row kinds (self+0x338, count +0x340)
constexpr uint32_t kText = 0x84a320;       // text lookup (out, table, row) -> out, *out = the string
constexpr uint32_t kChosen = 0xb4f090;     // kind 7's confirm state (the unused attention screen)

constexpr uint64_t kKind = 7;
constexpr int kMaxItems = 8; // the rows' inline storage
constexpr uint32_t kDescTable = 7, kDescRow = kKind + 1;
constexpr char kGameDesc[] = "Read before starting the game.";
constexpr char kDesc[] = "Change PC display and frame rate settings.";

// The title sequence (rdi of the states above).
constexpr size_t kItems = 0x338, kItemCount = 0x340;

uint8_t* gEboot = nullptr;
uint8_t* gLayout = nullptr; // the title layout, as decompressed (then loaded in place)
int32_t (*gDecompress)(uint8_t*, uint32_t, const uint8_t*) = nullptr;
void (*gBuildItems)(uint8_t*) = nullptr;
void** (*gText)(void**, uint32_t, uint32_t) = nullptr;

template <class T> T& at(uint8_t* p, size_t off) { return *reinterpret_cast<T*>(p + off); }

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
        std::fprintf(stderr, "[title] layout: PC SETTINGS row added\n");
    }
    else if (patchTitleMenuAtlas(dst, size)) std::fprintf(stderr, "[title] atlas: PC SETTINGS label drawn\n");
    return r;
}

void buildItems(uint8_t* self) {
    gBuildItems(self);
    int64_t& n = at<int64_t>(self, kItemCount);
    if (n < 0 || n >= kMaxItems) return;
    at<uint64_t*>(self, kItems)[n++] = kKind | 1ull << 32; // enabled
}

void** text(void** out, uint32_t table, uint32_t row) {
    gText(out, table, row);
    if (table == kDescTable && row == kDescRow && *out && std::strcmp(static_cast<const char*>(*out), kGameDesc) == 0)
        *out = const_cast<char*>(kDesc);
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
constexpr uint32_t kListOf = 0x57daa0;        // (screen, 0) -> its row list
constexpr uint32_t kListSize = 0x5828f0;      // (list, rows shown, rows)
constexpr uint32_t kOptions = 0x2638df8;      // -> the game's options (relocated)
constexpr size_t kOptionsSize = 0x58;         // what the screen edits, from +0
// Each id's value range, 20 bytes apart: +4 the last value. Editing shows that many + 1.
constexpr uint32_t kRanges = 0x23a2700;
constexpr size_t kRangeSize = 20, kRangeLast = 4;
constexpr uint8_t kFirstId = 1; // ids 1..4 (tips, cameras): plain choices the screen has no special case for

// The header: the title layout shows frame 5 of three lists for SETTINGS (rect 39); PC SETTINGS
// shows the menu label (rect 37) there instead, while its screen is up. The loaded layout is the
// decompressed buffer, its fields byte-swapped in place.
constexpr size_t kHeaderLists[] = {0x18bb0, 0x18bfc, 0x18c48};
constexpr size_t kHeaderRecord = 4 * 17; // frame 5's
constexpr uint16_t kRectHeader = 39, kRectLabel = 37;

constexpr int kPcRows = 4;
const char* const kPcLabels[kPcRows] = {"Window Mode", "Resolution", "Frame Rate Cap", "FPS Counter"};
const char* const kPcDescs[kPcRows] = {
    "Play in a window, a borderless window or fullscreen.",
    "Set the 3D rendering resolution. Takes effect after restarting the game.",
    "Set the most frames shown per second. Uncapped follows the monitor's refresh rate.",
    "Show the frame rate in the top right corner of the screen.",
};

void (*gSettingsState)(uint8_t*) = nullptr;
void (*gScreenInit)(uint8_t*) = nullptr;
void (*gScreenDraw)(uint8_t*) = nullptr;
void (*gValueChange)(uint8_t*, uint32_t, uint32_t) = nullptr;
uint32_t (*gCommit)(uint8_t*, void*) = nullptr;
uint32_t (*gCancelEdit)(uint8_t*, void*) = nullptr;

std::atomic<bool> gPcPending{false}; // PC SETTINGS chosen: the next screen built is ours
uint8_t* gPcScreen = nullptr;        // the game thread's, as all below
int32_t gOldLast[kPcRows];
std::vector<std::string> gNames[kPcRows];

uint8_t* row(uint8_t* screen, int i) { return at<uint8_t*>(screen, kRows) + kRowSize * i; }
uint8_t* options() { return at<uint8_t*>(gEboot, kOptions) + 8; }

void writeData(uint8_t* p, const void* v, size_t n) {
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    void* start = reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1));
    mprotect(start, page * 2, PROT_READ | PROT_WRITE);
    std::memcpy(p, v, n);
}

// The value ranges of the ids our rows take: ours while our screen is up, else the game's.
void setRanges(bool pc) {
    static bool saved = false;
    uint8_t* r = gEboot + kRanges + kRangeLast;
    if (!saved) {
        for (int i = 0; i < kPcRows; ++i) std::memcpy(&gOldLast[i], r + kRangeSize * (kFirstId + i), 4);
        saved = true;
    }
    for (int i = 0; i < kPcRows; ++i) {
        const int32_t last = pc ? int32_t(gNames[i].size()) - 1 : gOldLast[i];
        writeData(r + kRangeSize * (kFirstId + i), &last, 4);
    }
}

void setHeader(uint16_t rect) {
    uint8_t* s = gLayout;
    if (!s) return;
    // Still the loaded layout: our label's rect, now in native floats, and frame 5's records.
    const float want[4] = {1 / 1024.0f, 400 / 512.0f, 545 / 1024.0f, 446 / 512.0f};
    if (std::memcmp(s + 0x130 + kRectLabel * 16, want, 16) != 0) return;
    const uint16_t other = rect == kRectLabel ? kRectHeader : kRectLabel;
    for (size_t list : kHeaderLists) {
        const uint8_t* rec = s + list + kHeaderRecord;
        uint16_t frame, tex, r;
        std::memcpy(&frame, rec, 2), std::memcpy(&tex, rec + 4, 2), std::memcpy(&r, rec + 6, 2);
        if (frame != 5 || tex != 3 || (r != rect && r != other)) return;
    }
    for (size_t list : kHeaderLists) std::memcpy(s + list + kHeaderRecord + 6, &rect, 2);
}

void settingsState(uint8_t* self) {
    const bool pc = gPcPending;
    setHeader(pc ? kRectLabel : kRectHeader);
    gSettingsState(self);
}

void chosen(uint8_t* self) {
    std::fprintf(stderr, "[title] PC SETTINGS\n");
    gPcPending = true;
    settingsState(self);
}

void screenInit(uint8_t* self) {
    gScreenInit(self);
    const bool pc = gPcPending.exchange(false);
    gPcScreen = pc ? self : nullptr;
    if (!pc) return setRanges(false);
    bool ok = at<uint32_t>(self, kRowCap) >= uint32_t(kPcRows); // always: the game's own are more
    for (int i = 0; i < kPcRows; ++i) ok = ok && displayOptionCount(i) > 0; // none without a window
    if (!ok) {
        gPcScreen = nullptr;
        return setRanges(false);
    }
    for (int i = 0; i < kPcRows; ++i) {
        gNames[i].clear();
        for (int k = 0, n = std::min(displayOptionCount(i), int(kMaxNames)); k < n; ++k)
            gNames[i].push_back(displayOptionName(i, k));
        uint8_t* r = row(self, i);
        std::memset(r, 0, kRowSize);
        r[kRowField] = uint8_t(i);
        r[kRowId] = uint8_t(kFirstId + i);
        at<const char*>(r, kRowLabel) = kPcLabels[i];
        at<const char*>(r, kRowDesc) = kPcDescs[i];
        for (size_t k = 0; k < gNames[i].size(); ++k) at<const char*>(r, kRowNames + 8 * k) = gNames[i][k].c_str();
        const uint32_t v = uint32_t(std::clamp(displayOptionValue(i), 0, int(gNames[i].size()) - 1));
        at<uint32_t>(r, kRowValue) = v;
        self[kCommitted + i] = uint8_t(v);
    }
    at<uint32_t>(self, kRowCount) = kPcRows;
    setRanges(true);
    auto list = reinterpret_cast<uint8_t* (*)(uint8_t*, uint32_t)>(gEboot + kListOf);
    auto listSize = reinterpret_cast<void (*)(uint8_t*, uint32_t, uint32_t)>(gEboot + kListSize);
    listSize(list(self, 0), 0xb, kPcRows);
}

// Keeps the rows as the settings are, which the F11 menu can change too (not while one is edited).
void screenDraw(uint8_t* self) {
    if (self == gPcScreen && at<uint64_t>(self, kScreenState) != uint64_t(gEboot + kEditState))
        for (int i = 0; i < kPcRows; ++i) {
            const int v = displayOptionValue(i);
            if (v < 0 || v >= int(gNames[i].size())) continue;
            at<uint32_t>(row(self, i), kRowValue) = uint32_t(v);
            self[kCommitted + i] = uint8_t(v);
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
    KeepOptions keep(self);
    gValueChange(self, a, b);
}

uint32_t cancelEdit(uint8_t* self, void* e) {
    KeepOptions keep(self);
    return gCancelEdit(self, e);
}

uint32_t commit(uint8_t* self, void* e) {
    const uint32_t r = gCommit(self, e);
    if (self == gPcScreen)
        for (int i = 0; i < kPcRows; ++i) {
            const int v = int(at<uint32_t>(row(self, i), kRowValue));
            if (v != displayOptionValue(i)) setDisplayOption(i, v);
        }
    return r;
}

void writeJump(uint8_t* p, const void* fn) {
    // movabs $fn, %rax; jmp *%rax
    p[0] = 0x48, p[1] = 0xb8;
    const uint64_t target = reinterpret_cast<uint64_t>(fn);
    std::memcpy(p + 2, &target, 8);
    p[10] = 0xff, p[11] = 0xe0;
}

struct Hook {
    uint32_t at;
    std::vector<uint8_t> was; // the function's first instructions, position independent
    const void* fn;
    void** original; // set to a copy of `was` followed by a jump back (null: replaced outright)
};

} // namespace

bool patchTitleMenuLayout(uint8_t* s, size_t size) {
    constexpr size_t kSize = 0x19800, kRects = 0x130, kSlots = 0x7be0, kSlotStride = 0x100, kLists = 0x18568,
                     kOldList = 88, kNewList = 100, kSpare = 0x19700;
    constexpr int kRectLabel = 37, kRectSettings = 28, kRectNewGame = 23, kOldFrames = 7;
    static const uint8_t kSlot[] = {0x00, 0x00, 0x00, 0x06, 0x00, 0x07, 0x00, 0x00};
    static const uint8_t kSlotValue[] = {0x3e, 0x2a, 0xaa, 0xab};
    static const uint8_t kRect37[] = {0x3f, 0x7a, 0x80, 0x00, 0, 0, 0, 0, 0x3f, 0x80, 0, 0, 0x3d, 0x30, 0, 0};
    static const uint8_t kRect28[] = {0x3a, 0x80, 0, 0, 0x3f, 0x0d, 0x80, 0, 0x3f, 0x08, 0x40, 0, 0x3f, 0x24, 0x80, 0};
    // Frame k -> record k: NG/CO, CONTINUE, ENDLESS EDEN, DOWNLOAD INFO, REMINISCE, SETTINGS,
    // NEW GAME (a kind's frame: 0 -> 7, clamped to the last; 1..6 -> 0..5; 7 -> 6).
    static const uint8_t kList[kOldList] = {
        0x00, 0x00, 0x00, 0x08, 0x00, 0x01, 0x00, 0x0a, 0x00, 0x02, 0x00, 0x0c, 0x00, 0x03, 0x00, 0x0e,
        0x00, 0x04, 0x00, 0x10, 0x00, 0x05, 0x00, 0x12, 0x00, 0x06, 0x00, 0x14, 0xff, 0xff, 0x00, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x00, 0x03, 0x00, 0x16, 0x00, 0x01, 0x08, 0x08, 0x00, 0x03, 0x00, 0x18,
        0x00, 0x02, 0x08, 0x08, 0x00, 0x03, 0x00, 0x19, 0x00, 0x03, 0x08, 0x08, 0x00, 0x03, 0x00, 0x1a,
        0x00, 0x04, 0x08, 0x08, 0x00, 0x03, 0x00, 0x1b, 0x00, 0x05, 0x08, 0x08, 0x00, 0x03, 0x00, 0x1c,
        0x00, 0x06, 0x08, 0x08, 0x00, 0x03, 0x00, 0x17};

    if (size != kSize || std::memcmp(s, "SURB", 4) != 0 || s[5] != 1 || be32(s + 0x18) != kRects ||
        be32(s + 0x24) != 43 || std::memcmp(s + kRects + kRectLabel * 16, kRect37, 16) != 0 ||
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

    // The label's rect, in the atlas band patchTitleMenuAtlas draws in.
    uint8_t* r = s + kRects + kRectLabel * 16;
    putBeFloat(r, 1 / 1024.0f), putBeFloat(r + 4, 400 / 512.0f), putBeFloat(r + 8, 545 / 1024.0f),
        putBeFloat(r + 12, 446 / 512.0f);

    // Each slot gets its own list (the loader byte-swaps each in place, so none may be shared):
    // 8 keys, then the old frames 0..5, frame 6 the new label, frame 7 NEW GAME. Seven take the
    // old lists' place, the last the zeros after the data.
    uint8_t list[kNewList] = {};
    constexpr int kFrames = kOldFrames + 1;
    constexpr int kRecords = kFrames + 1; // in 4-byte units: the keys and their terminator
    for (int k = 0; k < kFrames; ++k) putBe16(list + 4 * k, uint16_t(k)), putBe16(list + 4 * k + 2, uint16_t(kRecords + 2 * k));
    putBe16(list + 4 * kFrames, 0xffff);
    std::memcpy(list + 4 * kRecords, kList + 4 * kOldFrames + 4, 8 * 6);
    uint8_t* rec = list + 4 * kRecords + 8 * 6;
    putBe16(rec, 6), putBe16(rec + 2, 0x0808), putBe16(rec + 4, 3), putBe16(rec + 6, kRectLabel);
    putBe16(rec + 8, 7), putBe16(rec + 10, 0x0808), putBe16(rec + 12, 3), putBe16(rec + 14, kRectNewGame);
    for (int i = 0; i < kMaxItems; ++i) {
        const uint32_t off = uint32_t(i < kMaxItems - 1 ? kLists + kNewList * i : kSpare);
        std::memcpy(s + off, list, kNewList);
        uint8_t* slot = s + kSlots + kSlotStride * i;
        putBe16(slot + 4, kFrames);
        putBe32(slot + 8, off);
    }
    return true;
}

bool patchTitleMenuAtlas(uint8_t* d, size_t size) {
    constexpr int kW = 1024, kH = 512, kHeader = 128, kRowH = 46, kX0 = 1, kBand = 400, kBandW = 544;
    constexpr int kContinue = 95, kReminisce = 236, kSettings = 283, kSettingsW = 320;
    constexpr uint64_t kSource = 0xc5af36fab8684185; // FNV-1a of rows 95..329, x 1..544: the letters used
    // The letters lean right: a glyph's columns shift by this much per row.
    static const int8_t kSlant[kRowH] = {11, 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4,
                                         4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2, -3};

    if (size != size_t(kHeader) + kW * kH * 4 || std::memcmp(d, "DDS ", 4) != 0) return false;
    uint32_t h, w, bits;
    std::memcpy(&h, d + 12, 4), std::memcpy(&w, d + 16, 4), std::memcpy(&bits, d + 88, 4);
    if (h != uint32_t(kH) || w != uint32_t(kW) || bits != 32) return false;
    auto px = [&](int x, int y) { return d + kHeader + (size_t(y) * kW + x) * 4; };
    uint64_t hash = 0xcbf29ce484222325;
    for (int y = kContinue; y <= kSettings + kRowH; ++y)
        for (const uint8_t *p = px(kX0, y), *e = px(kX0 + kBandW, y); p < e; ++p) hash = (hash ^ *p) * 0x100000001b3;
    if (hash != kSource) return false;
    for (int y = kBand; y < kBand + kRowH; ++y)
        for (int x = kX0; x < kX0 + kBandW; ++x)
            if (px(x, y)[3]) return false;

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
    return true;
}

bool patchTitleMenu(Module& eboot) {
    if (const char* e = std::getenv("LPN_PC_SETTINGS"); e && std::strcmp(e, "0") == 0) return false;
    Hook hooks[] = {
        {kDecompress, {0x41, 0x57, 0x41, 0x56, 0x53, 0x81, 0x3a, 0x53, 0x4c, 0x4c, 0x5a, 0xb8, 0x05, 0x40, 0x00, 0x80},
         reinterpret_cast<const void*>(&decompress), reinterpret_cast<void**>(&gDecompress)},
        {kBuildItems, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xfb},
         reinterpret_cast<const void*>(&buildItems), reinterpret_cast<void**>(&gBuildItems)},
        {kText, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xff},
         reinterpret_cast<const void*>(&text), reinterpret_cast<void**>(&gText)},
        {kChosen, {0xc5, 0xfa, 0x7e, 0x05, 0x40, 0x00, 0x84, 0x01, 0xc5, 0xfa, 0x7e, 0x0d},
         reinterpret_cast<const void*>(&chosen), nullptr},
        {kSettingsState, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0xbb, 0xa8, 0x02, 0x00, 0x00},
         reinterpret_cast<const void*>(&settingsState), reinterpret_cast<void**>(&gSettingsState)},
        {kScreenInit, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53},
         reinterpret_cast<const void*>(&screenInit), reinterpret_cast<void**>(&gScreenInit)},
        {kScreenDraw, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53},
         reinterpret_cast<const void*>(&screenDraw), reinterpret_cast<void**>(&gScreenDraw)},
        {kValueChange, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50},
         reinterpret_cast<const void*>(&valueChange), reinterpret_cast<void**>(&gValueChange)},
        {kCommit, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x31, 0xf6},
         reinterpret_cast<const void*>(&commit), reinterpret_cast<void**>(&gCommit)},
        {kCancelEdit, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x31, 0xf6},
         reinterpret_cast<const void*>(&cancelEdit), reinterpret_cast<void**>(&gCancelEdit)},
    };
    uint8_t* base = eboot.base();
    for (const Hook& h : hooks)
        if (std::memcmp(base + h.at, h.was.data(), h.was.size()) != 0) {
            std::fprintf(stderr, "[title] unexpected eboot code at 0x%x: no PC SETTINGS row\n", h.at);
            return false;
        }
    gEboot = base;
    void* mem = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return false;
    uint8_t* tramp = static_cast<uint8_t*>(mem);
    const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
    for (const Hook& h : hooks) {
        if (h.original) {
            std::memcpy(tramp, h.was.data(), h.was.size());
            writeJump(tramp + h.was.size(), base + h.at + h.was.size());
            *h.original = tramp;
            tramp += 32;
        }
        uint8_t* p = base + h.at;
        // Left writable: the loader still rewrites thread pointer reads on these pages, and
        // Module::protect sets the final protection afterwards.
        mprotect(reinterpret_cast<void*>(uintptr_t(p) & ~(page - 1)), page * 2, PROT_READ | PROT_WRITE | PROT_EXEC);
        writeJump(p, h.fn);
    }
    std::fprintf(stderr, "[title] PC SETTINGS row on the title menu\n");
    return true;
}

} // namespace host
