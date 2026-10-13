// libkernel files: PS4 paths (/app0, /savedata0, ...) mapped onto host folders.
#include "host/hle.h"
#include "host/kernel.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace host {

namespace {

namespace fs = std::filesystem;

std::string hostPath(const char* guest) {
    const std::string p = guest ? guest : "";
    auto mount = [&](const char* prefix, const std::string& root) -> std::string {
        const size_t n = std::strlen(prefix);
        if (p.compare(0, n, prefix) != 0 || (p.size() > n && p[n] != '/')) return {};
        return root + p.substr(n);
    };
    const Process& pr = process();
    if (auto h = mount("/app0", pr.gameDir); !h.empty()) return h;
    if (!pr.saveDir.empty())
        if (auto h = mount("/savedata0", pr.saveDir); !h.empty()) return h;
    for (const char* m : {"/download0", "/savedata0", "/temp0", "/data"})
        if (auto h = mount(m, pr.dataDir + m); !h.empty()) {
            fs::create_directories(fs::path(pr.dataDir + m));
            return h;
        }
    std::fprintf(stderr, "[file] unmapped guest path %s\n", p.c_str());
    return pr.dataDir + "/unmapped" + p;
}

int hostOpenFlags(int32_t f) {
    int h = (f & 3) == 1 ? O_WRONLY : (f & 3) == 2 ? O_RDWR : O_RDONLY;
    if (f & 0x4) h |= O_NONBLOCK;
    if (f & 0x8) h |= O_APPEND;
    if (f & 0x200) h |= O_CREAT;
    if (f & 0x400) h |= O_TRUNC;
    if (f & 0x800) h |= O_EXCL;
    if (f & 0x20000) h |= O_DIRECTORY;
    return h | O_CLOEXEC;
}

// FreeBSD struct stat as the PS4 sees it.
struct Ps4Stat {
    uint32_t dev, ino;
    uint16_t mode, nlink;
    uint32_t uid, gid, rdev;
    int64_t atimeSec, atimeNsec, mtimeSec, mtimeNsec, ctimeSec, ctimeNsec;
    int64_t size, blocks;
    uint32_t blksize, flags, gen;
    int32_t lspare;
    int64_t birthSec, birthNsec;
};
static_assert(sizeof(Ps4Stat) == 120);

void convertStat(const struct stat& s, Ps4Stat* o) {
    std::memset(o, 0, sizeof(*o));
    o->dev = uint32_t(s.st_dev);
    o->ino = uint32_t(s.st_ino);
    o->mode = uint16_t(s.st_mode);
    o->nlink = uint16_t(s.st_nlink);
    o->atimeSec = s.st_atim.tv_sec, o->atimeNsec = s.st_atim.tv_nsec;
    o->mtimeSec = s.st_mtim.tv_sec, o->mtimeNsec = s.st_mtim.tv_nsec;
    o->ctimeSec = s.st_ctim.tv_sec, o->ctimeNsec = s.st_ctim.tv_nsec;
    o->birthSec = o->ctimeSec, o->birthNsec = o->ctimeNsec;
    o->size = s.st_size;
    o->blocks = s.st_blocks;
    o->blksize = 0x10000;
}

int32_t check(int64_t r) { return r < 0 ? sceErrorFromHost(errno) : int32_t(r); }
int64_t check64(int64_t r) { return r < 0 ? sceErrorFromHost(errno) : r; }

int32_t sceKernelOpen(const char* path, int32_t flags, uint16_t mode) {
    const std::string h = hostPath(path);
    const int fd = ::open(h.c_str(), hostOpenFlags(flags), mode ? mode : 0644);
    if (fd < 0) std::fprintf(stderr, "[file] open %s failed: %s\n", path, std::strerror(errno));
    else if (process().traceFiles) std::fprintf(stderr, "[file] open %s\n", path);
    return check(fd);
}
int32_t sceKernelClose(int32_t fd) { return check(::close(fd)); }
int64_t sceKernelRead(int32_t fd, void* buf, uint64_t n) { return check64(::read(fd, buf, n)); }
int64_t sceKernelWrite(int32_t fd, const void* buf, uint64_t n) {
    if (fd == 1 || fd == 2) return ::write(fd, buf, n);
    return check64(::write(fd, buf, n));
}
int64_t sceKernelPread(int32_t fd, void* buf, uint64_t n, int64_t off) { return check64(::pread(fd, buf, n, off)); }
int64_t sceKernelPwrite(int32_t fd, const void* buf, uint64_t n, int64_t off) {
    return check64(::pwrite(fd, buf, n, off));
}
int64_t sceKernelLseek(int32_t fd, int64_t off, int32_t whence) { return check64(::lseek(fd, off, whence)); }
int32_t sceKernelFsync(int32_t fd) { return check(::fsync(fd)); }
int32_t sceKernelFtruncate(int32_t fd, int64_t len) { return check(::ftruncate(fd, len)); }
int32_t sceKernelTruncate(const char* path, int64_t len) { return check(::truncate(hostPath(path).c_str(), len)); }
int32_t sceKernelStat(const char* path, Ps4Stat* out) {
    struct stat s;
    if (::stat(hostPath(path).c_str(), &s) != 0) return sceErrorFromHost(errno);
    convertStat(s, out);
    return 0;
}
int32_t sceKernelFstat(int32_t fd, Ps4Stat* out) {
    struct stat s;
    if (::fstat(fd, &s) != 0) return sceErrorFromHost(errno);
    convertStat(s, out);
    return 0;
}
int32_t sceKernelMkdir(const char* path, uint16_t mode) { return check(::mkdir(hostPath(path).c_str(), mode)); }
int32_t sceKernelRmdir(const char* path) { return check(::rmdir(hostPath(path).c_str())); }
int32_t sceKernelUnlink(const char* path) { return check(::unlink(hostPath(path).c_str())); }
int32_t sceKernelRename(const char* from, const char* to) {
    return check(::rename(hostPath(from).c_str(), hostPath(to).c_str()));
}

// FreeBSD dirent records: { u32 fileno; u16 reclen; u8 type; u8 namlen; char name[]; }, 4-aligned.
std::mutex gDirMutex;
std::map<int32_t, size_t> gDirPositions; // fd -> entries already returned

int32_t sceKernelGetdirentries(int32_t fd, char* buf, int32_t nbytes, int64_t* basep) {
    std::vector<std::pair<std::string, uint8_t>> entries;
    {
        char link[64];
        std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
        std::error_code ec;
        const fs::path dir = fs::read_symlink(link, ec);
        if (ec) return sceError(ps4errno::kEBADF);
        entries.push_back({".", 4});
        entries.push_back({"..", 4});
        for (const auto& e : fs::directory_iterator(dir, ec))
            entries.push_back({e.path().filename().string(), uint8_t(e.is_directory() ? 4 : 8)});
    }
    std::lock_guard lock(gDirMutex);
    size_t& pos = gDirPositions[fd];
    if (basep) *basep = int64_t(pos);
    int32_t used = 0;
    for (; pos < entries.size(); ++pos) {
        const auto& [name, type] = entries[pos];
        const uint16_t reclen = uint16_t((8 + name.size() + 1 + 3) & ~3u);
        if (used + reclen > nbytes) break;
        char* r = buf + used;
        std::memset(r, 0, reclen);
        const uint32_t ino = uint32_t(pos + 1);
        std::memcpy(r, &ino, 4);
        std::memcpy(r + 4, &reclen, 2);
        r[6] = char(type);
        r[7] = char(name.size());
        std::memcpy(r + 8, name.c_str(), name.size());
        used += reclen;
    }
    return used;
}

int32_t sceKernelSetFsstParam(int32_t, uint64_t) { return 0; }
int32_t sceKernelCheckReachability(const char* path) {
    return ::access(hostPath(path).c_str(), F_OK) == 0 ? 0 : sceErrorFromHost(errno);
}

} // namespace

void registerFileHle() {
    const char* k = "libkernel";
    HLE(k, sceKernelOpen);
    HLE(k, sceKernelClose);
    HLE(k, sceKernelRead);
    HLE(k, sceKernelWrite);
    HLE(k, sceKernelPread);
    HLE(k, sceKernelPwrite);
    HLE(k, sceKernelLseek);
    HLE(k, sceKernelFsync);
    HLE(k, sceKernelFtruncate);
    HLE(k, sceKernelTruncate);
    HLE(k, sceKernelStat);
    HLE(k, sceKernelFstat);
    HLE(k, sceKernelMkdir);
    HLE(k, sceKernelRmdir);
    HLE(k, sceKernelUnlink);
    HLE(k, sceKernelRename);
    HLE(k, sceKernelGetdirentries);
    HLE(k, sceKernelSetFsstParam);
    HLE(k, sceKernelCheckReachability);
}

} // namespace host
