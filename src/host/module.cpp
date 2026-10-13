#include "host/module.h"

#include "host/memory.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <filesystem>

namespace host {

namespace {

constexpr uint64_t kPageSize = 0x4000; // PS4 segment alignment; a multiple of the host page

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
uint64_t alignDown(uint64_t v, uint64_t a) { return v & ~(a - 1); }

bool isMapped(uint32_t type) { return type == ps4elf::PT_LOAD || type == ps4elf::PT_SCE_RELRO; }

} // namespace

std::unique_ptr<Module> Module::load(const std::string& path, std::string* error) {
    auto elf = ps4elf::Elf::load(path, error);
    if (!elf) return nullptr;
    auto dyn = ps4elf::Dynlib::parse(*elf, error);
    if (!dyn) return nullptr;

    std::unique_ptr<Module> m(new Module);
    m->name_ = std::filesystem::path(path).filename().string();
    m->size_ = alignUp(elf->imageSize(), kPageSize);
    void* p = guestMap(nullptr, m->size_, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        if (error) *error = "cannot reserve " + std::to_string(m->size_) + " bytes for " + m->name_;
        return nullptr;
    }
    m->base_ = static_cast<uint8_t*>(p);

    for (const auto& s : elf->segments()) {
        if (!isMapped(s.type)) continue;
        if (s.vaddr + s.memsz > m->size_ || s.offset + s.filesz > elf->file().size()) {
            if (error) *error = m->name_ + ": segment out of range";
            return nullptr;
        }
        std::memcpy(m->base_ + s.vaddr, elf->file().data() + s.offset, s.filesz);
        // Anonymous memory is already zero, which covers .bss (memsz > filesz).
    }
    m->elf_ = std::move(*elf);
    m->dyn_ = std::move(*dyn);
    return m;
}

Module::~Module() {
    if (base_) munmap(base_, size_);
}

size_t Module::patchThreadPointerReads(int32_t fsOffset) {
    size_t patched = 0;
    for (const auto& s : elf_.segments()) {
        if (s.type != ps4elf::PT_LOAD || !(s.flags & 1)) continue; // executable segments only
        uint8_t* p = base_ + s.vaddr;
        const uint8_t* end = p + s.filesz;
        // 64 REX.W(+R) 8B /r with mod=00 rm=100, SIB 0x25 (no base, no index), disp32 = 0:
        //   mov r64, qword ptr fs:[0]
        for (; p + 9 <= end; ++p) {
            if (p[0] != 0x64 || (p[1] != 0x48 && p[1] != 0x4C) || p[2] != 0x8B) continue;
            if ((p[3] & 0xC7) != 0x04 || p[4] != 0x25) continue;
            if (p[5] | p[6] | p[7] | p[8]) continue;
            std::memcpy(p + 5, &fsOffset, 4);
            ++patched;
            p += 8;
        }
    }
    return patched;
}

bool Module::protect(std::string* error) {
    for (const auto& s : elf_.segments()) {
        if (!isMapped(s.type)) continue;
        int prot = 0;
        if (s.type == ps4elf::PT_SCE_RELRO) prot = PROT_READ;
        else {
            if (s.flags & 4) prot |= PROT_READ;
            if (s.flags & 2) prot |= PROT_WRITE;
            if (s.flags & 1) prot |= PROT_EXEC;
        }
        const uint64_t start = alignDown(s.vaddr, kPageSize);
        const uint64_t end = alignUp(s.vaddr + s.memsz, kPageSize);
        if (mprotect(base_ + start, end - start, prot) != 0) {
            if (error) *error = name_ + ": mprotect failed";
            return false;
        }
    }
    return true;
}

} // namespace host
