#include "ps4elf/dynlib.h"

#include <cstring>
#include <set>

namespace ps4elf {

namespace {

template <typename T>
T read(const uint8_t* p) {
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return v;
}

} // namespace

uint64_t decodeId(const std::string& s) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    uint64_t v = 0;
    for (char c : s) {
        const char* p = std::strchr(kAlphabet, c);
        if (!p || !c) return ~0ull;
        v = v * 64 + uint64_t(p - kAlphabet);
    }
    return v;
}

std::optional<Dynlib> Dynlib::parse(const Elf& elf, std::string* error) {
    const Segment* dyn = elf.findSegment(PT_DYNAMIC);
    const Segment* data = elf.findSegment(PT_SCE_DYNLIBDATA);
    if (!dyn || !data) {
        if (error) *error = "missing PT_DYNAMIC or PT_SCE_DYNLIBDATA";
        return std::nullopt;
    }
    auto file = elf.file();
    if (dyn->offset + dyn->filesz > file.size() || data->offset + data->filesz > file.size()) {
        if (error) *error = "dynamic segments out of range";
        return std::nullopt;
    }
    const uint8_t* base = file.data() + data->offset;
    const uint64_t baseSize = data->filesz;

    std::multimap<int64_t, uint64_t> tags;
    for (uint64_t off = 0; off + 16 <= dyn->filesz; off += 16) {
        const uint8_t* p = file.data() + dyn->offset + off;
        const int64_t tag = read<int64_t>(p);
        if (tag == 0) break;
        tags.emplace(tag, read<uint64_t>(p + 8));
    }
    auto one = [&](int64_t tag) -> std::optional<uint64_t> {
        auto it = tags.find(tag);
        if (it == tags.end()) return std::nullopt;
        return it->second;
    };

    auto strtab = one(DT_SCE_STRTAB), strsz = one(DT_SCE_STRSZ);
    auto symtab = one(DT_SCE_SYMTAB), symsz = one(DT_SCE_SYMTABSZ);
    if (!strtab || !strsz || !symtab || !symsz || *strtab + *strsz > baseSize ||
        *symtab + *symsz > baseSize) {
        if (error) *error = "missing or bad string/symbol table";
        return std::nullopt;
    }
    auto str = [&](uint64_t off) -> std::string {
        if (off >= *strsz) return {};
        const char* s = reinterpret_cast<const char*>(base + *strtab + off);
        return std::string(s, strnlen(s, *strsz - off));
    };

    Dynlib out;
    auto [nb, ne] = tags.equal_range(1); // DT_NEEDED
    for (auto it = nb; it != ne; ++it) out.needed_.push_back(str(it->second));
    auto [mb, me] = tags.equal_range(DT_SCE_NEEDED_MODULE);
    for (auto it = mb; it != me; ++it)
        out.modules_[uint16_t(it->second >> 48)] = str(it->second & 0xFFFFFFFF);
    auto [lb, le] = tags.equal_range(DT_SCE_IMPORT_LIB);
    for (auto it = lb; it != le; ++it)
        out.libraries_[uint16_t(it->second >> 48)] = str(it->second & 0xFFFFFFFF);

    for (uint64_t off = 0; off + 24 <= *symsz; off += 24) {
        const uint8_t* p = base + *symtab + off;
        Symbol s;
        s.name = str(read<uint32_t>(p));
        s.info = p[4];
        s.shndx = read<uint16_t>(p + 6);
        s.value = read<uint64_t>(p + 8);
        s.size = read<uint64_t>(p + 16);
        out.symbols_.push_back(std::move(s));
    }

    auto addRelocs = [&](std::optional<uint64_t> off, std::optional<uint64_t> size) {
        if (!off || !size || *off + *size > baseSize) return;
        for (uint64_t r = 0; r + 24 <= *size; r += 24) {
            const uint8_t* p = base + *off + r;
            const uint64_t info = read<uint64_t>(p + 8);
            out.relocs_.push_back({read<uint64_t>(p), uint32_t(info), uint32_t(info >> 32),
                                   read<int64_t>(p + 16)});
        }
    };
    addRelocs(one(DT_SCE_RELA), one(DT_SCE_RELASZ));
    addRelocs(one(DT_SCE_JMPREL), one(DT_SCE_PLTRELSZ));
    return out;
}

std::vector<Import> Dynlib::imports() const {
    auto resolve = [&](const Symbol& s) {
        Import imp;
        const auto a = s.name.find('#');
        const auto b = s.name.find('#', a + 1);
        imp.nid = s.name.substr(0, a);
        if (a != std::string::npos && b != std::string::npos) {
            const auto lib = libraries_.find(uint16_t(decodeId(s.name.substr(a + 1, b - a - 1))));
            const auto mod = modules_.find(uint16_t(decodeId(s.name.substr(b + 1))));
            imp.library = lib != libraries_.end() ? lib->second : s.name.substr(a + 1, b - a - 1);
            imp.module = mod != modules_.end() ? mod->second : s.name.substr(b + 1);
        }
        return imp;
    };

    std::vector<Import> out;
    std::set<uint32_t> seen;
    for (const auto& r : relocs_) {
        if (r.symbol == 0 || r.symbol >= symbols_.size()) continue;
        const Symbol& s = symbols_[r.symbol];
        if (!s.isImport()) continue;
        Import imp = resolve(s);
        imp.slot = r.offset;
        imp.relocType = r.type;
        out.push_back(std::move(imp));
        seen.insert(r.symbol);
    }
    // Imports with no relocation (unusual) are still worth listing.
    for (uint32_t i = 1; i < symbols_.size(); ++i)
        if (symbols_[i].isImport() && !seen.count(i)) out.push_back(resolve(symbols_[i]));
    return out;
}

} // namespace ps4elf
