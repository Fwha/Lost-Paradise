// Shared state and conventions for the libkernel HLE (hle_kernel/memory/pthread/file.cpp).
#pragma once

#include <cstdint>
#include <string>

namespace host {

class Module;

struct Process {
    Module* eboot = nullptr;
    std::string gameDir; // /app0
    std::string dataDir; // host folder for /download0, /savedata0, /temp0
};
Process& process();

// PS4 errno values follow FreeBSD, which differ from Linux for several codes.
int toPs4Errno(int linuxErrno);

// libkernel's sce* functions return 0x80020000 | errno on failure.
constexpr int32_t kSceKernelError = int32_t(0x80020000);
inline int32_t sceError(int ps4Errno) { return kSceKernelError | ps4Errno; }
inline int32_t sceErrorFromHost(int linuxErrno) { return sceError(toPs4Errno(linuxErrno)); }

// The guest's errno, as returned by libkernel's __error().
int* guestErrno();

namespace ps4errno {
constexpr int kEPERM = 1, kENOENT = 2, kESRCH = 3, kEINTR = 4, kEIO = 5, kEBADF = 9, kEDEADLK = 11, kENOMEM = 12,
              kEACCES = 13, kEFAULT = 14, kEBUSY = 16, kEEXIST = 17, kEINVAL = 22, kEAGAIN = 35, kETIMEDOUT = 60,
              kENAMETOOLONG = 63, kENOTEMPTY = 66, kENOSYS = 78;
}

// Register the HLE functions of each libkernel area.
void registerKernelHle();
void registerMemoryHle();
void registerPthreadHle();
void registerFileHle();

} // namespace host

// Registration helper: HLE("libkernel", sceKernelFoo) binds the symbol of that name.
#define HLE(lib, fn) ::host::registerHle(lib, #fn, reinterpret_cast<void*>(&fn))
// For PS4 names that clash with the host's libc (mmap, _write, ...).
#define HLE_NAMED(lib, name, fn) ::host::registerHle(lib, name, reinterpret_cast<void*>(&fn))
