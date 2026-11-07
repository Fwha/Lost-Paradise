// A PS4 executable or library mapped into this process.
#pragma once

#include "ps4elf/dynlib.h"
#include "ps4elf/elf.h"

#include <cstdint>
#include <memory>
#include <string>

namespace host {

class Module {
public:
    // Reserve memory and copy the module's segments in. Nothing is executable yet:
    // relocations are applied by the linker, then protect() finalises permissions.
    static std::unique_ptr<Module> load(const std::string& path, std::string* error);
    ~Module();

    const std::string& name() const { return name_; }
    const ps4elf::Elf& elf() const { return elf_; }
    const ps4elf::Dynlib& dynlib() const { return dyn_; }

    uint8_t* base() const { return base_; }
    size_t size() const { return size_; }
    bool contains(uint64_t addr) const {
        return addr >= uint64_t(base_) && addr < uint64_t(base_) + size_;
    }
    uint64_t entry() const { return uint64_t(base_) + elf_.entry(); }

    // Static TLS: this module's block lives at (thread pointer - tlsOffset).
    const ps4elf::Segment* tlsSegment() const { return elf_.findSegment(ps4elf::PT_TLS); }
    uint64_t tlsOffset = 0;
    uint32_t tlsIndex = 0; // 1-based module id for __tls_get_addr; 0 when the module has no TLS

    // Rewrite every `mov reg, fs:[0]` so the guest reads its own thread pointer instead of
    // glibc's. Returns the number of instructions patched.
    size_t patchThreadPointerReads(uint8_t segment, int32_t disp);

    // Apply final page permissions (code RX, RELRO R, data RW).
    bool protect(std::string* error);

private:
    Module() = default;

    std::string name_;
    ps4elf::Elf elf_;
    ps4elf::Dynlib dyn_;
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
};

} // namespace host
