#include "common/compat.h"

#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cerrno>

namespace {

int64_t positioned(int fd, void* buf, size_t n, int64_t off, bool write) {
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    OVERLAPPED ov{};
    ov.Offset = DWORD(uint64_t(off));
    ov.OffsetHigh = DWORD(uint64_t(off) >> 32);
    DWORD done = 0;
    const DWORD len = n > 0x7fffffff ? 0x7fffffff : DWORD(n);
    // An offset in OVERLAPPED moves the file position on a synchronous handle; POSIX keeps it.
    const int64_t pos = _lseeki64(fd, 0, SEEK_CUR);
    const BOOL ok = write ? WriteFile(h, buf, len, &done, &ov) : ReadFile(h, buf, len, &done, &ov);
    if (pos >= 0) _lseeki64(fd, pos, SEEK_SET);
    if (!ok) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) return 0;
        errno = e == ERROR_ACCESS_DENIED ? EACCES : EIO;
        return -1;
    }
    return int64_t(done);
}

} // namespace

int64_t pread(int fd, void* buf, size_t n, int64_t off) { return positioned(fd, buf, n, off, false); }
int64_t pwrite(int fd, const void* buf, size_t n, int64_t off) {
    return positioned(fd, const_cast<void*>(buf), n, off, true);
}

long lpnUtcOffset(std::time_t t) {
    std::tm lt{}, gt{};
    localtime_s(&lt, &t);
    gmtime_s(&gt, &t);
    gt.tm_isdst = lt.tm_isdst;
    return long(_mkgmtime(&lt) - _mkgmtime(&gt));
}
#endif

namespace host {

std::string userCacheDir() {
#ifdef _WIN32
    if (const char* a = std::getenv("LOCALAPPDATA"); a && *a) return a;
    return {};
#else
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return x;
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache";
    return {};
#endif
}

} // namespace host
