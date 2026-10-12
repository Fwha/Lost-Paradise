// PS4 dynamic linking info: needed modules, import libraries, symbols and relocations.
#pragma once

#include "ps4elf/elf.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ps4elf {

// Tags in PT_DYNAMIC that point into PT_SCE_DYNLIBDATA.
enum : int64_t {
    DT_SCE_NEEDED_MODULE = 0x6100000F,
    DT_SCE_EXPORT_LIB = 0x61000013,
    DT_SCE_IMPORT_LIB = 0x61000015,
    DT_SCE_PLTGOT = 0x61000027,
    DT_SCE_JMPREL = 0x61000029,
    DT_SCE_PLTRELSZ = 0x6100002D,
    DT_SCE_RELA = 0x6100002F,
    DT_SCE_RELASZ = 0x61000031,
    DT_SCE_STRTAB = 0x61000035,
    DT_SCE_STRSZ = 0x61000037,
    DT_SCE_SYMTAB = 0x61000039,
    DT_SCE_SYMTABSZ = 0x6100003F,
};

constexpr uint32_t R_X86_64_64 = 1;
constexpr uint32_t R_X86_64_GLOB_DAT = 6;
constexpr uint32_t R_X86_64_JUMP_SLOT = 7;
constexpr uint32_t R_X86_64_RELATIVE = 8;

struct Symbol {
    std::string name;   // raw, e.g. "Ou3iL1abvng#C#A" for an import
    uint8_t info = 0;
    uint16_t shndx = 0;
    uint64_t value = 0;
    uint64_t size = 0;

    bool isImport() const { return shndx == 0 && !name.empty(); }
    uint8_t bind() const { return info >> 4; }
    uint8_t type() const { return info & 0xF; }
};

struct Relocation {
    uint64_t offset = 0;   // vaddr patched
    uint32_t type = 0;
    uint32_t symbol = 0;   // index into Dynlib::symbols
    int64_t addend = 0;
};

// One imported function or object, resolved to the library and module it comes from.
struct Import {
    std::string nid;       // 11-char base64 NID
    std::string library;   // e.g. "libSceGnmDriver"
    std::string module;    // e.g. "libSceGnmDriver"
    uint64_t slot = 0;     // GOT slot vaddr the loader fills in (0 if none)
    uint32_t relocType = 0;
};

class Dynlib {
public:
    static std::optional<Dynlib> parse(const Elf& elf, std::string* error);

    const std::vector<std::string>& needed() const { return needed_; }
    const std::map<uint16_t, std::string>& modules() const { return modules_; }
    const std::map<uint16_t, std::string>& libraries() const { return libraries_; }
    const std::vector<Symbol>& symbols() const { return symbols_; }
    const std::vector<Relocation>& relocations() const { return relocs_; }   // RELA + JMPREL
    std::vector<Import> imports() const;

private:
    std::vector<std::string> needed_;
    std::map<uint16_t, std::string> modules_;
    std::map<uint16_t, std::string> libraries_;
    std::vector<Symbol> symbols_;
    std::vector<Relocation> relocs_;
};

// Decode the PS4 base64 variant used for library/module ids in symbol names.
uint64_t decodeId(const std::string& s);

} // namespace ps4elf
