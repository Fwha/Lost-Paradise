#include "host/title_menu.h"

#include "host/display_menu.h"
#include "host/module.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace host {

namespace {

constexpr uint32_t kDecompress = 0xe2ebc0; // SLLZ (dst, capacity, src): every compressed file
constexpr uint32_t kBuildItems = 0xb429b0; // fills the title's row kinds (self+0x338, count +0x340)
constexpr uint32_t kText = 0x84a320;       // text lookup (out, table, row) -> out, *out = the string
constexpr uint32_t kChosen = 0xb4f090;     // kind 7's confirm state (the unused attention screen)
constexpr uint32_t kMenuState = 0x2393e28; // the title menu's setup state (b43f90), relocated

constexpr uint64_t kKind = 7;
constexpr int kMaxItems = 8; // the rows' inline storage
constexpr uint32_t kDescTable = 7, kDescRow = kKind + 1;
constexpr char kGameDesc[] = "Read before starting the game.";
constexpr char kDesc[] = "Change PC display and frame rate settings.";

// The title sequence (rdi of the states above).
constexpr size_t kItems = 0x338, kItemCount = 0x340, kNextState = 0x218, kNextAdj = 0x220, kPending = 0x228,
                 kReturning = 0x388;

uint8_t* gEboot = nullptr;
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
    if (patchTitleMenuLayout(dst, size)) std::fprintf(stderr, "[title] layout: PC SETTINGS row added\n");
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

void setState(uint8_t* self, uint64_t fn) {
    at<uint64_t>(self, kNextState) = fn;
    at<uint64_t>(self, kNextAdj) = 0;
    at<uint64_t>(self, kPending) = 1; // and no transition time
}

// A state: the title menu waits (getting no input) while the F11 menu is open, then sets itself up
// again as when coming back from SETTINGS. Even, as a member function pointer must be.
__attribute__((aligned(16))) void waitForMenu(uint8_t* self) {
    if (displayMenuOpen()) return;
    at<int32_t>(self, kReturning) = 1;
    setState(self, at<uint64_t>(gEboot, kMenuState));
}

void chosen(uint8_t* self) {
    std::fprintf(stderr, "[title] PC SETTINGS: display menu\n");
    openDisplayMenu();
    setState(self, reinterpret_cast<uint64_t>(&waitForMenu));
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
