// The few POSIX calls the host, launcher and pkg tools use that Windows (MinGW-w64) lacks or
// spells differently. On Linux this only adds LPN_O_BINARY (0) and lpnMkdir.
#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <ctime>
#include <string>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <unistd.h>

// Files are opened in binary mode: text mode would turn \r\n into \n in game data.
#define LPN_O_BINARY O_BINARY
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif

int64_t pread(int fd, void* buf, size_t n, int64_t off);
int64_t pwrite(int fd, const void* buf, size_t n, int64_t off);
inline int fsync(int fd) { return _commit(fd); }
inline std::tm* localtime_r(const std::time_t* t, std::tm* out) { return localtime_s(out, t) == 0 ? out : nullptr; }
inline std::tm* gmtime_r(const std::time_t* t, std::tm* out) { return gmtime_s(out, t) == 0 ? out : nullptr; }
inline std::time_t timegm(std::tm* t) { return _mkgmtime(t); }
inline int lpnMkdir(const char* path, int) { return ::mkdir(path); }
#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 30
#endif
inline long sysconf(int) { return 4096; } // only asked for the page size
// Seconds east of UTC of local time at `t` (struct tm has no tm_gmtoff here).
long lpnUtcOffset(std::time_t t);
#else
#include <unistd.h>
#define LPN_O_BINARY 0
inline int lpnMkdir(const char* path, int mode) { return ::mkdir(path, mode); }
inline long lpnUtcOffset(std::time_t t) {
    std::tm lt{};
    localtime_r(&t, &lt);
    return lt.tm_gmtoff;
}
#endif

namespace host {

// The per-user folder for caches (shader cache): $XDG_CACHE_HOME or ~/.cache on Linux,
// %LOCALAPPDATA% on Windows. Empty when none is known.
std::string userCacheDir();

} // namespace host
