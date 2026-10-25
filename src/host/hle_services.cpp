// System services the game expects but that have no PC meaning, or only a small one:
// user accounts, controller, save data, system dialogs, trophies, PlayGo, networking,
// clocks, audio pacing and zlib. Each answers the way an offline console would.
#include "host/hle.h"
#include "host/kernel.h"
#include "host/profile.h"
#include "host/save_menu.h"
#include "host/videoout.h"
#include "host/window.h"

#include <zlib.h>
#ifdef LPN_HAVE_SDL2
#include <SDL.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace host {

namespace {

namespace fs = std::filesystem;

constexpr int32_t kUserId = 1;

int32_t ok() { return 0; }
int32_t one() { return 1; }

// --- UserService ----------------------------------------------------------------------

int32_t sceUserServiceGetInitialUser(int32_t* id) { return *id = kUserId, 0; }
int32_t sceUserServiceGetLoginUserIdList(int32_t* ids) {
    ids[0] = kUserId;
    ids[1] = ids[2] = ids[3] = -1;
    return 0;
}
int32_t sceUserServiceGetUserName(int32_t, char* name, uint64_t size) {
    std::snprintf(name, size, "%s", "Player");
    return 0;
}
int32_t sceUserServiceGetUserColor(int32_t, int32_t* color) { return *color = 0, 0; }
// The initial user logs in once at startup, as on a console; games may wait for it before
// assigning a controller to the player.
int32_t sceUserServiceGetEvent(int32_t* event) {
    static std::atomic<bool> loggedIn{false};
    if (loggedIn.exchange(true)) return int32_t(0x80960007); // NO_EVENT
    event[0] = 0;                                            // SCE_USER_SERVICE_EVENT_TYPE_LOGIN
    event[1] = kUserId;
    return 0;
}

// --- SystemService --------------------------------------------------------------------

int32_t sceSystemServiceGetStatus(uint8_t* status) {
    std::memset(status, 0, 132); // eventNum 0, not overlaid, not in background
    return 0;
}
int32_t sceSystemServiceReceiveEvent(void*) { return int32_t(0x80A10003); } // NO_EVENT
int32_t sceSystemServiceGetDisplaySafeAreaInfo(float* ratio) { return *ratio = 1.0f, 0; }
int32_t sceSystemServiceLoadExec(const char* path, char* const*) {
    std::fprintf(stderr, "[system] game asked to restart into %s; exiting\n", path ? path : "(null)");
    printUnimplementedSummary();
    std::_Exit(0);
}

// --- AppContent -----------------------------------------------------------------------

// Integer from sce_sys/param.sfo (the game's USER_DEFINED_PARAM_n pick its region/language).
int32_t sfoInt(const std::string& key) {
    static std::map<std::string, int32_t> values = [] {
        std::map<std::string, int32_t> v;
        FILE* f = std::fopen((process().gameDir + "/sce_sys/param.sfo").c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[appcontent] sce_sys/param.sfo not found; app params read as 0\n");
            return v;
        }
        std::string d;
        char buf[4096];
        for (size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) d.append(buf, n);
        std::fclose(f);
        auto u32 = [&](size_t o) { uint32_t x = 0; if (o + 4 <= d.size()) std::memcpy(&x, &d[o], 4); return x; };
        auto u16 = [&](size_t o) { uint16_t x = 0; if (o + 2 <= d.size()) std::memcpy(&x, &d[o], 2); return x; };
        if (u32(0) != 0x46535000) return v; // "\0PSF"
        const uint32_t keys = u32(8), data = u32(12), count = u32(16);
        for (uint32_t i = 0; i < count; ++i) {
            const size_t e = 20 + size_t(i) * 16;
            if (u16(e + 2) != 0x0404) continue; // integer entries only
            v[std::string(d.c_str() + keys + u16(e))] = int32_t(u32(data + u32(e + 12)));
        }
        return v;
    }();
    auto it = values.find(key);
    return it != values.end() ? it->second : 0;
}

int32_t sceAppContentAppParamGetInt(int32_t id, int32_t* value) {
    if (id == 0) *value = 3; // SKU flag: full game
    else if (id >= 1 && id <= 4) *value = sfoInt("USER_DEFINED_PARAM_" + std::to_string(id));
    else return int32_t(0x80D90002); // PARAMETER
    return 0;
}
int32_t sceAppContentGetAddcontInfo(void*, void*, void*) { return int32_t(0x80D90005); } // no DLC
int32_t sceAppContentTemporaryDataMount2(int32_t, char* mountPoint) {
    std::snprintf(mountPoint, 16, "/temp0");
    return 0;
}

// --- Pad ------------------------------------------------------------------------------

struct PadData {
    uint32_t buttons;
    uint8_t lx, ly, rx, ry, l2, r2, pad0[2];
    float orientation[4], acceleration[3], angularVelocity[3];
    uint8_t touchNum, touchReserved[3];
    uint32_t touchReserved1;
    uint8_t touches[16];
    bool connected;
    uint8_t pad1[3];
    uint64_t timestamp;
    uint8_t extension[16];
    uint8_t connectedCount, reserved[2], deviceUniqueDataLen, deviceUniqueData[12];
};
static_assert(sizeof(PadData) == 120);

int32_t scePadOpen(int32_t, int32_t, int32_t, const void*) { return 1; }
int32_t scePadReadState(int32_t, PadData* d) {
    std::memset(d, 0, sizeof(*d));
    const PadState s = padState();
    static uint32_t lastButtons = ~0u;
    static uint64_t reads = 0;
    ++reads;
    if (s.buttons != lastButtons) {
        std::fprintf(stderr, "[pad] buttons 0x%x at flip %llu (read %llu)\n", s.buttons,
                     (unsigned long long)flipCount(), (unsigned long long)reads);
        lastButtons = s.buttons;
    }
    d->buttons = s.buttons;
    d->lx = s.lx, d->ly = s.ly, d->rx = s.rx, d->ry = s.ry, d->l2 = s.l2, d->r2 = s.r2;
    d->orientation[3] = 1.0f;
    d->acceleration[1] = 1.0f; // gravity, pad lying flat
    d->connected = true;
    d->connectedCount = 1;
    d->timestamp = gameTimeScale() != 1.0 ? gameClockNs() / 1000
                                          : uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                                         std::chrono::steady_clock::now().time_since_epoch())
                                                         .count());
    return 0;
}
struct PadControllerInformation {
    float touchPadPixelDensity;
    uint16_t touchPadWidth, touchPadHeight;
    uint8_t deadZoneLeft, deadZoneRight, connectionType, connectedCount;
    bool connected;
    uint8_t pad[3];
    int32_t deviceClass;
    uint8_t reserved[8];
};
int32_t scePadGetControllerInformation(int32_t, PadControllerInformation* i) {
    *i = {};
    i->touchPadPixelDensity = 44.86f;
    i->touchPadWidth = 1920, i->touchPadHeight = 943;
    i->deadZoneLeft = i->deadZoneRight = 13;
    i->connectedCount = 1;
    i->connected = true;
    return 0;
}

// --- SaveData -------------------------------------------------------------------------

std::string saveRoot() { return process().dataDir + "/savedata"; }

struct SaveDataMount2 {
    int32_t userId, pad;
    const char* dirName; // SceSaveDataDirName, char[32]
    uint64_t blocks;
    uint32_t mountMode;
};
struct SaveDataMountResult {
    char mountPoint[16];
    uint64_t requiredBlocks;
    uint32_t unused, mountStatus;
};

int32_t sceSaveDataMount2(const SaveDataMount2* m, SaveDataMountResult* r) {
    const std::string dir = saveRoot() + "/" + m->dirName;
    const bool exists = fs::exists(dir);
    constexpr uint32_t kCreate = 4, kCreate2 = 32;
    if (!exists && !(m->mountMode & (kCreate | kCreate2))) return int32_t(0x809F0008); // NOT_FOUND
    fs::create_directories(dir);
    process().saveDir = dir;
    std::memset(r, 0, sizeof(*r));
    std::snprintf(r->mountPoint, sizeof(r->mountPoint), "/savedata0");
    r->mountStatus = exists ? 0 : 1; // CREATED
    std::fprintf(stderr, "[savedata] mounted %s%s\n", m->dirName, exists ? "" : " (new)");
    return 0;
}
int32_t sceSaveDataUmount(const void*) {
    process().saveDir.clear();
    return 0;
}

struct DirNameSearchResult {
    uint32_t hitNum, pad;
    char (*dirNames)[32];
    uint32_t dirNamesNum, setNum;
    void* params;
    void* infos;
};
int32_t sceSaveDataDirNameSearch(const void*, DirNameSearchResult* r) {
    uint32_t hits = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(saveRoot(), ec)) {
        if (!e.is_directory()) continue;
        if (r->dirNames && hits < r->dirNamesNum)
            std::snprintf(r->dirNames[hits], 32, "%s", e.path().filename().c_str());
        ++hits;
    }
    r->hitNum = hits;
    r->setNum = r->dirNames ? std::min(hits, r->dirNamesNum) : 0;
    return 0;
}

// --- Common dialogs: no UI, each finishes immediately with "OK" -------------------------

// There is no dialog UI: what the game would show goes to the terminal instead, so a message like
// "no saved data" is not silently swallowed.
const char* saveDataSystemMessage(int32_t type) {
    switch (type) {
    case 1: return "There is no saved data.";
    case 2: return "Confirm the operation.";
    case 3: return "Overwrite the saved data?";
    case 4: return "Not enough free space.";
    case 5: return "Processing...";
    case 6: return "The saved data is corrupted.";
    case 7: return "Done.";
    default: return nullptr;
    }
}

// SceSaveDataDialogParam after the 48-byte common base: size, mode, dispType, then pointers
// (animation, items, user message, system message, ...).
// A list dialog shows our save menu over the game window; with no window it "chooses" itself:
// the first slot offered when saving, the newest listed save otherwise (see GetResult).
std::string gDialogChoice;
bool gDialogMenu = false; // the current list dialog is our menu

int32_t sceSaveDataDialogOpen(const uint8_t* p) {
    int32_t mode, disp;
    std::memcpy(&mode, p + 52, 4);
    std::memcpy(&disp, p + 56, 4);
    const uint8_t* ptrs[6];
    std::memcpy(ptrs, p + 64, sizeof ptrs);
    const char* what = disp == 1 ? "save" : disp == 2 ? "load" : disp == 3 ? "delete" : "save data";
    gDialogMenu = false;
    if (mode == 1 && ptrs[1]) { // list: items are userId, pad, titleId*, dirName*, dirNameNum, pad, newItem*, focusPos
        const char* dirs = nullptr;
        const void* newItem = nullptr;
        uint32_t count = 0;
        int32_t focusPos = 0;
        std::memcpy(&dirs, ptrs[1] + 16, 8);
        std::memcpy(&count, ptrs[1] + 24, 4);
        std::memcpy(&newItem, ptrs[1] + 32, 8);
        std::memcpy(&focusPos, ptrs[1] + 40, 4);
        gDialogChoice.clear();
        std::vector<SaveSlot> slots;
        int newestIdx = -1, firstEmpty = -1;
        fs::file_time_type newest{};
        for (uint32_t i = 0; dirs && i < count; ++i) {
            const std::string name(dirs + i * 32, strnlen(dirs + i * 32, 32));
            slots.push_back({name, saveRoot() + "/" + name});
            std::error_code ec;
            const auto t = fs::last_write_time(saveRoot() + "/" + name + "/data.saveps4", ec);
            if (ec && firstEmpty < 0) firstEmpty = int(i);
            if (!ec && (newestIdx < 0 || t > newest)) newestIdx = int(i), newest = t;
        }
        std::fprintf(stderr, "[dialog] %s: %u saves listed (new item %s, focus %d)\n", what, count,
                     newItem ? "offered" : "none", focusPos);
        // The game's own new save picks the lowest free number from 01 to 30 (eboot 0x9a5d60), so
        // there is no new-save row once 30 saves exist.
        constexpr uint32_t kMaxSaves = 30;
        if (count >= kMaxSaves) newItem = nullptr;
        // Saving starts on an empty slot: a listed one, else the new-save row under the saves.
        if (disp == 1 && firstEmpty < 0 && newItem) firstEmpty = int(slots.size());
        const int focus = disp == 1 && firstEmpty >= 0 ? firstEmpty : std::max(newestIdx, 0);
        const SaveMenuKind kind = disp == 1 ? SaveMenuKind::Save : disp == 3 ? SaveMenuKind::Delete : SaveMenuKind::Load;
        gDialogMenu = openSaveMenu(kind, slots, focus, newItem != nullptr);
        if (!gDialogMenu) {
            if (disp == 1 && !slots.empty()) gDialogChoice = slots[0].dir; // saving: the first slot offered
            else if (newestIdx >= 0) gDialogChoice = slots[size_t(newestIdx)].dir;
            std::fprintf(stderr, "[dialog] %s: choosing %s\n", what, gDialogChoice.empty() ? "none" : gDialogChoice.c_str());
        }
    } else if (mode == 2 && ptrs[2]) { // user message: buttonType, msgType, msg
        const char* msg = nullptr;
        std::memcpy(&msg, ptrs[2] + 8, 8);
        if (msg) std::fprintf(stderr, "[dialog] %s: %s\n", what, msg);
    } else if (mode == 3 && ptrs[3]) { // system message: sysMsgType
        int32_t type;
        std::memcpy(&type, ptrs[3], 4);
        const char* msg = saveDataSystemMessage(type);
        if (msg) std::fprintf(stderr, "[dialog] %s: %s\n", what, msg);
        else std::fprintf(stderr, "[dialog] %s: system message %d\n", what, type);
    }
    return 0;
}

// SceMsgDialogParam: base (48 bytes), size, mode, then user message, progress bar and system
// message pointers. A user message holds buttonType and the text.
int32_t sceMsgDialogOpen(const uint8_t* p) {
    int32_t mode;
    std::memcpy(&mode, p + 56, 4);
    const uint8_t* user = nullptr;
    std::memcpy(&user, p + 64, 8);
    const char* msg = nullptr;
    if (mode == 1 && user) std::memcpy(&msg, user + 8, 8);
    if (msg) std::fprintf(stderr, "[dialog] %s\n", msg);
    return 0;
}

constexpr int32_t kDialogRunning = 2, kDialogFinished = 3;
int32_t dialogFinished() { return kDialogFinished; }
int32_t saveDataDialogStatus() { return saveMenuOpen() ? kDialogRunning : kDialogFinished; }
int32_t saveDataDialogClose(const void*) {
    closeSaveMenu();
    return 0;
}
// SceSaveDataDialogResult: mode, result, buttonId, pad, dirName*, param*, userData.
int32_t saveDataDialogResult(int32_t* r) {
    // An OK result with an empty dirName is the new-save item: the game then saves to the lowest
    // free slot number itself.
    const bool choseNew = gDialogMenu && saveMenuChoseNew();
    if (gDialogMenu) gDialogChoice = saveMenuChoice();
    std::memset(r, 0, 12);
    r[2] = 1; // YES / OK
    if (gDialogMenu && gDialogChoice.empty() && !choseNew) r[1] = 1; // USER_CANCELED
    char* dir = nullptr;
    std::memcpy(&dir, reinterpret_cast<uint8_t*>(r) + 16, 8);
    if (dir && (choseNew || !gDialogChoice.empty())) std::snprintf(dir, 32, "%s", gDialogChoice.c_str());
    return 0;
}

int32_t dialogResultOk(int32_t* r) {
    std::memset(r, 0, 12); // mode, result OK, buttonId set below
    r[2] = 1;              // YES / OK
    return 0;
}

// --- NpTrophy -------------------------------------------------------------------------

int32_t sceNpTrophyCreateContext(int32_t* ctx, int32_t, uint32_t, uint64_t) { return *ctx = 1, 0; }
int32_t sceNpTrophyCreateHandle(int32_t* h) { return *h = 1, 0; }
int32_t sceNpTrophyUnlockTrophy(int32_t, int32_t, int32_t id, int32_t* platinum) {
    std::fprintf(stderr, "[trophy] unlocked trophy %d\n", id);
    if (platinum) *platinum = -1;
    return 0;
}

// --- PlayGo: the whole game is installed ------------------------------------------------

int32_t scePlayGoOpen(int32_t* handle, const void*) { return *handle = 1, 0; }
int32_t scePlayGoGetToDoList(int32_t, void*, uint32_t, uint32_t* n) { return *n = 0, 0; }
int32_t scePlayGoGetEta(int32_t, const void*, uint32_t, int64_t* eta) { return *eta = 0, 0; }
int32_t scePlayGoGetInstallSpeed(int32_t, int32_t* speed) { return *speed = 2, 0; }

// --- Live streaming -------------------------------------------------------------------

int32_t sceGameLiveStreamingGetCurrentStatus(uint8_t* status) {
    std::memset(status, 0, 16);
    return 0;
}

// --- Rtc: ticks are microseconds since 0001-01-01 --------------------------------------

constexpr uint64_t kUnixEpochTicks = 62135596800ull * 1000000ull;

struct RtcDateTime {
    uint16_t year, month, day, hour, minute, second;
    uint32_t microsecond;
};

uint64_t unixMicros() {
    using namespace std::chrono;
    return uint64_t(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
}
int32_t sceRtcGetCurrentNetworkTick(uint64_t* tick) { return *tick = kUnixEpochTicks + unixMicros(), 0; }
int32_t sceRtcGetTick(const RtcDateTime* d, uint64_t* tick) {
    std::tm t{};
    t.tm_year = d->year - 1900, t.tm_mon = d->month - 1, t.tm_mday = d->day;
    t.tm_hour = d->hour, t.tm_min = d->minute, t.tm_sec = d->second;
    *tick = kUnixEpochTicks + uint64_t(int64_t(timegm(&t))) * 1000000ull + d->microsecond;
    return 0;
}
int32_t sceRtcSetTick(RtcDateTime* d, const uint64_t* tick) {
    const int64_t us = int64_t(*tick - kUnixEpochTicks);
    const time_t secs = time_t(us / 1000000);
    std::tm t{};
    gmtime_r(&secs, &t);
    *d = {uint16_t(t.tm_year + 1900), uint16_t(t.tm_mon + 1), uint16_t(t.tm_mday), uint16_t(t.tm_hour),
          uint16_t(t.tm_min), uint16_t(t.tm_sec), uint32_t(us % 1000000)};
    return 0;
}
int64_t utcOffsetMicros() {
    const time_t now = time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    return int64_t(local.tm_gmtoff) * 1000000;
}
int32_t sceRtcConvertUtcToLocalTime(const uint64_t* utc, uint64_t* local) {
    return *local = *utc + utcOffsetMicros(), 0;
}
int32_t sceRtcConvertLocalTimeToUtc(const uint64_t* local, uint64_t* utc) {
    return *utc = *local - utcOffsetMicros(), 0;
}

// --- AudioOut: each port plays through its own SDL audio device ------------------------
// Output blocks like the hardware, one grain at a time on a steady clock, keeping about 40 ms
// queued on the device. Without a device (headless, --no-audio, no SDL) the samples are dropped.

bool gAudioEnabled = true;

struct AudioPort {
    uint32_t grain = 256, freq = 48000, channels = 2, sampleBytes = 2;
    std::chrono::steady_clock::time_point next{};
    uint64_t outputs = 0, underruns = 0;
    bool primed = false;
#ifdef LPN_HAVE_SDL2
    SDL_AudioDeviceID dev = 0;
#endif
};
std::mutex gAudioMutex;
std::map<int32_t, AudioPort> gPorts;
int32_t gNextPort = 1;

#ifdef LPN_HAVE_SDL2
// The port's own format; SDL converts it (8 channels down to stereo too) for the device.
SDL_AudioDeviceID openAudioDevice(const AudioPort& p) {
    static const bool ok = [] {
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) == 0) return true;
        std::fprintf(stderr, "[audio] SDL audio init failed: %s\n", SDL_GetError());
        return false;
    }();
    if (!ok) return 0;
    SDL_AudioSpec want{}, have{};
    want.freq = int(p.freq);
    want.format = p.sampleBytes == 4 ? AUDIO_F32SYS : AUDIO_S16SYS;
    want.channels = uint8_t(p.channels);
    want.samples = 512;
    const SDL_AudioDeviceID dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (!dev) {
        std::fprintf(stderr, "[audio] cannot open device: %s\n", SDL_GetError());
        return 0;
    }
    std::fprintf(stderr, "[audio] port %u ch %s %u Hz on %s\n", p.channels, p.sampleBytes == 4 ? "float" : "s16",
                 p.freq, SDL_GetCurrentAudioDriver());
    SDL_PauseAudioDevice(dev, 0);
    return dev;
}
#endif

int32_t sceAudioOutOpen(int32_t, int32_t, int32_t, uint32_t grain, uint32_t freq, uint32_t param) {
    // param bits 0-7: S16 mono/stereo/8ch, float mono/stereo/8ch, S16 8ch std, float 8ch std.
    static const uint8_t kChannels[8] = {1, 2, 8, 1, 2, 8, 8, 8};
    const uint32_t format = param & 0xff;
    if (format > 7) return int32_t(0x80260005); // INVALID_FORMAT
    AudioPort p;
    p.grain = grain;
    p.freq = freq ? freq : 48000;
    p.channels = kChannels[format];
    p.sampleBytes = (format >= 3 && format <= 5) || format == 7 ? 4 : 2;
    p.next = std::chrono::steady_clock::now();
#ifdef LPN_HAVE_SDL2
    if (gAudioEnabled) p.dev = openAudioDevice(p);
#endif
    std::lock_guard lock(gAudioMutex);
    const int32_t h = gNextPort++;
    gPorts[h] = p;
    return h;
}
int32_t sceAudioOutClose(int32_t h) {
    std::lock_guard lock(gAudioMutex);
    auto it = gPorts.find(h);
    if (it == gPorts.end()) return int32_t(0x80260004); // INVALID_PORT
#ifdef LPN_HAVE_SDL2
    if (it->second.dev) SDL_CloseAudioDevice(it->second.dev);
#endif
    gPorts.erase(it);
    return 0;
}
int32_t sceAudioOutOutput(int32_t h, const void* samples) {
    AudioPort p;
    {
        std::lock_guard lock(gAudioMutex);
        auto it = gPorts.find(h);
        if (it == gPorts.end()) return int32_t(0x80260004); // INVALID_PORT
        p = it->second;
        it->second.outputs++;
#ifdef LPN_HAVE_SDL2
        if (!p.dev) {
#endif
            const auto now = std::chrono::steady_clock::now();
            if (it->second.next < now - std::chrono::milliseconds(100)) it->second.next = now; // resync after a stall
            it->second.next += std::chrono::microseconds(uint64_t(p.grain) * 1000000 / p.freq);
            p.next = it->second.next;
#ifdef LPN_HAVE_SDL2
        }
#endif
    }
#ifdef LPN_HAVE_SDL2
    if (p.dev) {
        const uint32_t frameBytes = p.channels * p.sampleBytes, bytesPerMs = p.freq * frameBytes / 1000;
        // LPN_AUDIO_MS: how much sound to keep queued (default 40 ms); more hides game hitches.
        static const uint32_t latencyMs = [] {
            const char* e = std::getenv("LPN_AUDIO_MS");
            return e ? uint32_t(std::atoi(e)) : 40u;
        }();
        const uint32_t target = std::max(4 * p.grain * frameBytes, latencyMs * bytesPerMs);
        if (p.outputs > 0 && SDL_GetQueuedAudioSize(p.dev) == 0) { // ran dry: an audible gap
            uint64_t n;
            {
                std::lock_guard lock(gAudioMutex);
                n = ++gPorts[h].underruns;
            }
            if ((n & (n - 1)) == 0 || n % 100 == 0)
                std::fprintf(stderr, "[audio] port %d ran dry (%llu times, output %llu)\n", h, (unsigned long long)n,
                             (unsigned long long)p.outputs);
        }
        // LPN_AUDIO_DUMP=dir: also write each port's raw samples to dir/port-N.raw.
        static const char* dumpDir = std::getenv("LPN_AUDIO_DUMP");
        if (dumpDir && samples) {
            static std::map<int32_t, FILE*> files;
            std::lock_guard lock(gAudioMutex);
            FILE*& f = files[h];
            if (!f) f = std::fopen((std::string(dumpDir) + "/port-" + std::to_string(h) + ".raw").c_str(), "wb");
            if (f) std::fwrite(samples, frameBytes, p.grain, f);
        }
        if (samples) SDL_QueueAudio(p.dev, samples, p.grain * frameBytes);
        // Block one grain at a time on a steady clock, like the hardware. Waiting for the queue
        // to drain instead releases the game in bursts as the device takes 1024-frame chunks,
        // and the game then reads its mix before its mixer thread has finished it (a 47 Hz
        // flutter). The queue level only fills the queue up front and trims clock drift.
        const auto grainTime = std::chrono::microseconds(uint64_t(p.grain) * 1000000 / p.freq);
        const uint32_t queued = SDL_GetQueuedAudioSize(p.dev);
        std::chrono::steady_clock::time_point wake;
        {
            std::lock_guard lock(gAudioMutex);
            auto& next = gPorts[h].next;
            const auto now = std::chrono::steady_clock::now();
            auto& primed = gPorts[h].primed;
            if (queued == 0 || next < now - std::chrono::milliseconds(100)) primed = false; // ran dry or stalled
            if (!primed) {
                next = now; // fill up without waiting
                primed = queued >= target;
            } else if (queued > target * 3 / 2) {
                next += grainTime + grainTime / 16;
            } else if (queued < target / 2) {
                next += grainTime - grainTime / 16;
            } else {
                next += grainTime;
            }
            wake = next;
        }
        std::this_thread::sleep_until(wake);
        return 0;
    }
#else
    (void)samples;
#endif
    std::this_thread::sleep_until(p.next);
    return 0;
}
struct AudioOutOutputParam {
    int32_t handle, pad;
    const void* ptr;
};
int32_t sceAudioOutOutputs(const AudioOutOutputParam* params, uint32_t num) {
    for (uint32_t i = 0; i < num; ++i) sceAudioOutOutput(params[i].handle, params[i].ptr); // all share the clock
    return 0;
}

// --- Zlib: decompression requests, completed synchronously -----------------------------

struct ZlibRequest {
    uint32_t outLen;
    int32_t status;
};
std::mutex gZlibMutex;
std::map<uint64_t, ZlibRequest> gZlib;
uint64_t gNextZlib = 1;

int32_t sceZlibInflate(const void* src, uint32_t srcLen, void* dst, uint32_t dstLen, uint64_t* id) {
    const auto t0 = std::chrono::steady_clock::now();
    z_stream z{};
    // Raw deflate unless the data carries a zlib header.
    const auto* s = static_cast<const uint8_t*>(src);
    const bool wrapped = srcLen >= 2 && (s[0] & 0x0F) == 8 && ((s[0] << 8) | s[1]) % 31 == 0;
    inflateInit2(&z, wrapped ? 15 : -15);
    z.next_in = const_cast<Bytef*>(s);
    z.avail_in = srcLen;
    z.next_out = static_cast<Bytef*>(dst);
    z.avail_out = dstLen;
    const int r = inflate(&z, Z_FINISH);
    inflateEnd(&z);
    auto& io = profile::io();
    io.inflates.fetch_add(1, std::memory_order_relaxed);
    io.inflateNs.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                           std::memory_order_relaxed);
    std::lock_guard lock(gZlibMutex);
    *id = gNextZlib++;
    gZlib[*id] = {uint32_t(z.total_out), r == Z_STREAM_END ? 0 : int32_t(0x81120002)};
    return 0;
}
int32_t sceZlibWaitForDone(uint64_t* id, const uint32_t*) {
    std::lock_guard lock(gZlibMutex);
    if (gZlib.empty()) return int32_t(0x81120027); // NOT_FOUND
    *id = gZlib.begin()->first;
    return 0;
}
int32_t sceZlibGetResult(uint64_t id, uint32_t* outLen, int32_t* status) {
    std::lock_guard lock(gZlibMutex);
    auto it = gZlib.find(id);
    if (it == gZlib.end()) return int32_t(0x81120027);
    *outLen = it->second.outLen;
    *status = it->second.status;
    gZlib.erase(it);
    return 0;
}

// --- DiscMap: report no bitmap so Fios reads normally ---------------------------------

int32_t discMapNoBitmap() { return int32_t(0x81100004); }

// --- small libkernel leftovers ----------------------------------------------------------

int32_t sceKernelConvertUtcToLocaltime(int64_t utc, int64_t* local, void* tz, int32_t* dst) {
    std::tm t{};
    const time_t secs = time_t(utc);
    localtime_r(&secs, &t);
    if (local) *local = utc + t.tm_gmtoff;
    if (tz) std::memset(tz, 0, 8);
    if (dst) *dst = t.tm_isdst > 0 ? 3600 : 0;
    return 0;
}
int32_t getrusage_(int32_t, void* usage) {
    std::memset(usage, 0, 144);
    return 0;
}

void regs(const char* lib, std::initializer_list<const char*> names, void* fn) {
    for (const char* n : names) registerHle(lib, n, fn);
}

} // namespace

void setAudioEnabled(bool on) { gAudioEnabled = on; }

void registerServicesHle() {
    void* zero = reinterpret_cast<void*>(&ok);
    void* positive = reinterpret_cast<void*>(&one);

    const char* us = "libSceUserService";
    HLE(us, sceUserServiceGetInitialUser);
    HLE(us, sceUserServiceGetLoginUserIdList);
    HLE(us, sceUserServiceGetUserName);
    HLE(us, sceUserServiceGetUserColor);
    HLE(us, sceUserServiceGetEvent);
    regs(us, {"sceUserServiceInitialize", "sceUserServiceTerminate"}, zero);

    const char* ss = "libSceSystemService";
    HLE(ss, sceSystemServiceGetStatus);
    HLE(ss, sceSystemServiceReceiveEvent);
    HLE(ss, sceSystemServiceGetDisplaySafeAreaInfo);
    HLE(ss, sceSystemServiceLoadExec);
    regs(ss, {"sceSystemServiceHideSplashScreen"}, zero);

    const char* ac = "libSceAppContent";
    HLE(ac, sceAppContentAppParamGetInt);
    HLE(ac, sceAppContentGetAddcontInfo);
    HLE(ac, sceAppContentTemporaryDataMount2);
    regs(ac, {"sceAppContentInitialize"}, zero);

    regs("libSceSysmodule",
         {"sceSysmoduleLoadModule", "sceSysmoduleIsLoaded", "sceSysmoduleUnloadModule",
          "sceSysmoduleLoadModuleInternalWithArg"},
         zero);

    const char* pad = "libScePad";
    HLE(pad, scePadOpen);
    HLE(pad, scePadReadState);
    HLE(pad, scePadGetControllerInformation);
    regs(pad,
         {"scePadInit", "scePadClose", "scePadResetLightBar", "scePadSetLightBar", "scePadSetVibration",
          "scePadSetMotionSensorState", "scePadSetTiltCorrectionState", "scePadSetAngularVelocityDeadbandState"},
         zero);

    const char* sd = "libSceSaveData";
    HLE(sd, sceSaveDataMount2);
    HLE(sd, sceSaveDataUmount);
    HLE(sd, sceSaveDataDirNameSearch);
    regs(sd, {"sceSaveDataInitialize3", "sceSaveDataTerminate", "sceSaveDataSetParam", "sceSaveDataSaveIcon"}, zero);

    void* finished = reinterpret_cast<void*>(&dialogFinished);
    void* resultOk = reinterpret_cast<void*>(&dialogResultOk);
    regs("libSceCommonDialog", {"sceCommonDialogInitialize"}, zero);
    const char* md = "libSceMsgDialog";
    regs(md, {"sceMsgDialogInitialize", "sceMsgDialogTerminate"}, zero);
    HLE(md, sceMsgDialogOpen);
    regs(md, {"sceMsgDialogUpdateStatus"}, finished);
    regs(md, {"sceMsgDialogGetResult"}, resultOk);
    const char* ed = "libSceErrorDialog";
    regs(ed, {"sceErrorDialogInitialize", "sceErrorDialogOpen", "sceErrorDialogTerminate"}, zero);
    regs(ed, {"sceErrorDialogUpdateStatus"}, finished);
    const char* sdd = "libSceSaveDataDialog";
    regs(sdd,
         {"sceSaveDataDialogInitialize", "sceSaveDataDialogTerminate", "sceSaveDataDialogProgressBarSetValue"},
         zero);
    HLE(sdd, sceSaveDataDialogOpen);
    regs(sdd, {"sceSaveDataDialogUpdateStatus"}, reinterpret_cast<void*>(&saveDataDialogStatus));
    regs(sdd, {"sceSaveDataDialogClose"}, reinterpret_cast<void*>(&saveDataDialogClose));
    regs(sdd, {"sceSaveDataDialogIsReadyToDisplay"}, positive);
    regs(sdd, {"sceSaveDataDialogGetResult"}, reinterpret_cast<void*>(&saveDataDialogResult));
    const char* ime = "libSceImeDialog";
    regs(ime, {"sceImeDialogInit", "sceImeDialogTerm", "sceImeDialogGetPanelSize"}, zero);
    regs(ime, {"sceImeDialogGetStatus"}, finished);
    regs(ime, {"sceImeDialogGetResult"}, resultOk);

    const char* tr = "libSceNpTrophy";
    HLE(tr, sceNpTrophyCreateContext);
    HLE(tr, sceNpTrophyCreateHandle);
    HLE(tr, sceNpTrophyUnlockTrophy);
    regs(tr, {"sceNpTrophyRegisterContext", "sceNpTrophyDestroyContext", "sceNpTrophyDestroyHandle"}, zero);

    const char* pg = "libScePlayGo";
    HLE(pg, scePlayGoOpen);
    HLE(pg, scePlayGoGetToDoList);
    HLE(pg, scePlayGoGetEta);
    HLE(pg, scePlayGoGetInstallSpeed);
    regs(pg, {"scePlayGoInitialize", "scePlayGoTerminate", "scePlayGoClose", "scePlayGoSetInstallSpeed"}, zero);

    // Online features: initialise fine, never connect.
    regs("libSceNet", {"sceNetPoolCreate"}, positive);
    regs("libSceNet", {"sceNetPoolDestroy"}, zero);
    regs("libSceSsl", {"sceSslInit"}, positive);
    regs("libSceSsl", {"sceSslTerm"}, zero);
    regs("libSceHttp", {"sceHttpInit"}, positive);
    regs("libSceHttp", {"sceHttpTerm"}, zero);
    regs("libSceNpWebApi", {"sceNpWebApiInitialize"}, positive);
    regs("libSceNpWebApi", {"sceNpWebApiTerminate"}, zero);
    regs("libSceRudp", {"sceRudpInit", "sceRudpEnd", "sceRudpEnableInternalIOThread"}, zero);
    regs("libSceNpManager", {"sceNpSetNpTitleId"}, zero);
    regs("libSceRemoteplay", {"sceRemoteplayInitialize", "sceRemoteplayTerminate"}, zero);
    regs("libSceSharePlay", {"sceSharePlayInitialize", "sceSharePlayTerminate", "sceSharePlaySetProhibition"}, zero);
    regs("libSceScreenShot", {"sceScreenShotEnable", "sceScreenShotDisable", "sceScreenShotSetOverlayImage"}, zero);
    regs("libSceVideoRecording", {"sceVideoRecordingSetInfo"}, zero);
    const char* gls = "libSceGameLiveStreaming";
    HLE(gls, sceGameLiveStreamingGetCurrentStatus);
    regs(gls,
         {"sceGameLiveStreamingInitialize", "sceGameLiveStreamingTerminate", "sceGameLiveStreamingEnableLiveStreaming",
          "sceGameLiveStreamingPermitLiveStreaming", "sceGameLiveStreamingPermitServerSideRecording",
          "sceGameLiveStreamingSetCameraFrameSetting", "sceGameLiveStreamingSetInvitationSessionId",
          "sceGameLiveStreamingSetMaxBitrate", "sceGameLiveStreamingSetMetadata",
          "sceGameLiveStreamingGetProgramInfo"},
         zero);

    const char* rtc = "libSceRtc";
    HLE(rtc, sceRtcGetCurrentNetworkTick);
    HLE(rtc, sceRtcGetTick);
    HLE(rtc, sceRtcSetTick);
    HLE(rtc, sceRtcConvertUtcToLocalTime);
    HLE(rtc, sceRtcConvertLocalTimeToUtc);

    const char* ao = "libSceAudioOut";
    HLE(ao, sceAudioOutOpen);
    HLE(ao, sceAudioOutClose);
    HLE(ao, sceAudioOutOutput);
    HLE(ao, sceAudioOutOutputs);
    regs(ao, {"sceAudioOutInit"}, zero);

    const char* z = "libSceZlib";
    HLE(z, sceZlibInflate);
    HLE(z, sceZlibWaitForDone);
    HLE(z, sceZlibGetResult);
    regs(z, {"sceZlibInitialize", "sceZlibFinalize"}, zero);

    regs("libSceDiscMap", {"sceDiscMapIsRequestOnHDD", "fJgP+wqifno"}, reinterpret_cast<void*>(&discMapNoBitmap));

    HLE("libkernel", sceKernelConvertUtcToLocaltime);
    HLE_NAMED("libkernel", "getrusage", getrusage_);
}

} // namespace host
