// Guest thread-local storage.
//
// PS4 code finds its TLS through the thread pointer: `mov reg, fs:[0]` yields a TCB whose
// first word points at itself, with each module's static TLS block below it (variant II).
// On Linux %fs belongs to glibc, so Module::patchThreadPointerReads rewrites those reads to
// load a host thread_local instead, and each guest thread gets its own TCB here.
#pragma once

#include <cstdint>
#include <vector>

namespace host {

class Module;

// %fs-relative offset of the host slot holding the guest thread pointer.
int32_t guestThreadPointerFsOffset();

// Lay out static TLS for the loaded modules (sets Module::tlsOffset / tlsIndex).
void layoutStaticTls(const std::vector<Module*>& modules);

// Allocate and initialise a TCB + static TLS for the calling thread and make it current.
void* createGuestTls();
void* currentGuestTls();

// tls_index as passed to __tls_get_addr by general-dynamic TLS code.
struct TlsIndex {
    uint64_t module;
    uint64_t offset;
};
void* guestTlsGetAddr(const TlsIndex* ti);

} // namespace host
