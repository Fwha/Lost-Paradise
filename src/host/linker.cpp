#include "host/linker.h"

#include "host/hle.h"
#include "host/module.h"

#include <sys/mman.h>

#include <cstring>
#include <map>
#include <memory>

extern "C" void lpn_unimplemented_stub();
extern "C" void lpn_trace_stub();

namespace host {

namespace {

constexpr uint32_t R_X86_64_DTPMOD64 = 16;
constexpr uint32_t R_X86_64_DTPOFF64 = 17;
constexpr uint32_t R_X86_64_TPOFF64 = 18;
constexpr uint8_t STT_OBJECT = 1;
constexpr uint8_t STT_TLS = 6;

std::vector<Module*> gModules;

// Executable memory for per-import trampolines:
//   movabs rax, ImportInfo* ; movabs r11, lpn_unimplemented_stub ; jmp r11
// Traced imports use r10 instead of rax, which carries the vector count of varargs calls.
class TrampolinePool {
public:
    void* make(ImportInfo* info, bool traced = false) {
        constexpr size_t kSize = 32;
        if (!page_ || used_ + kSize > kPage) {
            void* p = mmap(nullptr, kPage, PROT_READ | PROT_WRITE | PROT_EXEC,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) return nullptr;
            page_ = static_cast<uint8_t*>(p);
            used_ = 0;
        }
        uint8_t* t = page_ + used_;
        used_ += kSize;
        const uint64_t infoAddr = uint64_t(info);
        const uint64_t stubAddr = uint64_t(traced ? &lpn_trace_stub : &lpn_unimplemented_stub);
        t[0] = traced ? 0x49 : 0x48; t[1] = traced ? 0xBA : 0xB8; std::memcpy(t + 2, &infoAddr, 8);
        t[10] = 0x49; t[11] = 0xBB; std::memcpy(t + 12, &stubAddr, 8);
        t[20] = 0x41; t[21] = 0xFF; t[22] = 0xE3;
        return t;
    }

private:
    static constexpr size_t kPage = 0x10000;
    uint8_t* page_ = nullptr;
    size_t used_ = 0;
};

struct Target {
    uint64_t addr = 0;
    Module* module = nullptr; // defining module (for TLS relocations)
};

std::string key(const std::string& library, const std::string& nid) { return library + "#" + nid; }

} // namespace

Module* findModule(uint64_t addr) {
    for (Module* m : gModules)
        if (m->contains(addr)) return m;
    return nullptr;
}

bool linkModules(const std::vector<Module*>& modules, const ps4elf::NidDb& nids, LinkStats* stats,
                 std::string* error) {
    gModules = modules;
    static TrampolinePool pool;
    static std::map<std::string, std::unique_ptr<ImportInfo>> infos;    // one per library#nid
    static std::map<std::string, void*> unimplementedTargets;
    static std::map<std::string, void*> tracedTargets;
    LinkStats st;

    std::map<std::string, Target> exports;
    for (Module* m : modules)
        for (const auto& e : m->dynlib().exports())
            exports.try_emplace(key(e.library, e.nid), Target{uint64_t(m->base()) + e.value, m});

    for (Module* m : modules) {
        const auto& dyn = m->dynlib();
        const auto& syms = dyn.symbols();
        std::vector<Target> resolved(syms.size());
        std::vector<bool> done(syms.size());

        auto resolve = [&](uint32_t index) -> Target {
            if (done[index]) return resolved[index];
            done[index] = true;
            const ps4elf::Symbol& s = syms[index];
            if (!s.isImport()) return resolved[index] = {uint64_t(m->base()) + s.value, m};

            const ps4elf::Import imp = dyn.importOf(s);
            const std::string k = key(imp.library, imp.nid);
            auto it = exports.find(k);
            // The system libc (libSceLibcInternal) shares NIDs with the game's bundled libc.prx.
            if (it == exports.end() && imp.library == "libSceLibcInternal") it = exports.find(key("libc", imp.nid));
            if (it != exports.end()) {
                ++st.toModules;
                return resolved[index] = it->second;
            }
            const std::string& name = nids.lookup(imp.nid);
            // Imports missing from the NID database are registered under their NID.
            if (void* fn = findHle(imp.library, name.empty() ? imp.nid : name)) {
                ++st.toHle;
                if (hleOptions().traceHle && s.type() != STT_OBJECT) {
                    void*& traced = tracedTargets[k];
                    if (!traced) {
                        auto& info = infos[k];
                        info = std::make_unique<ImportInfo>(ImportInfo{imp.library, imp.nid, name, 0, fn});
                        traced = pool.make(info.get(), true);
                    }
                    return resolved[index] = {uint64_t(traced), nullptr};
                }
                return resolved[index] = {uint64_t(fn), nullptr};
            }
            ++st.unimplemented;
            void*& target = unimplementedTargets[k];
            if (!target) {
                st.unimplementedNames.push_back(imp.library + "::" + (name.empty() ? imp.nid : name));
                auto& info = infos[k];
                info = std::make_unique<ImportInfo>(ImportInfo{imp.library, imp.nid, name, 0});
                if (s.type() == STT_OBJECT) {
                    // Unknown data: hand out zeroed memory rather than code.
                    target = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                                  -1, 0);
                    std::fprintf(stderr, "[link] unimplemented data import %s::%s\n", imp.library.c_str(),
                                 name.empty() ? imp.nid.c_str() : name.c_str());
                } else {
                    target = pool.make(info.get());
                }
            }
            return resolved[index] = {uint64_t(target), nullptr};
        };

        for (const auto& r : dyn.relocations()) {
            if (r.offset + 8 > m->size()) {
                if (error) *error = m->name() + ": relocation outside image";
                return false;
            }
            auto* where = reinterpret_cast<uint64_t*>(m->base() + r.offset);
            if (r.symbol >= syms.size()) {
                if (error) *error = m->name() + ": relocation symbol out of range";
                return false;
            }
            switch (r.type) {
            case ps4elf::R_X86_64_RELATIVE:
                *where = uint64_t(m->base()) + r.addend;
                break;
            case ps4elf::R_X86_64_64:
                *where = resolve(r.symbol).addr + r.addend;
                break;
            case ps4elf::R_X86_64_GLOB_DAT:
            case ps4elf::R_X86_64_JUMP_SLOT:
                *where = resolve(r.symbol).addr;
                break;
            case R_X86_64_DTPMOD64: {
                Module* def = r.symbol ? resolve(r.symbol).module : m;
                *where = def ? def->tlsIndex : 0;
                break;
            }
            case R_X86_64_DTPOFF64:
                *where = (r.symbol ? syms[r.symbol].value : 0) + r.addend;
                break;
            case R_X86_64_TPOFF64: {
                Module* def = m;
                uint64_t value = 0;
                if (r.symbol) {
                    const auto& s = syms[r.symbol];
                    if (s.isImport()) {
                        const Target t = resolve(r.symbol);
                        def = t.module;
                        value = def ? t.addr - uint64_t(def->base()) : 0; // export value, TLS-relative
                    } else {
                        value = s.value;
                    }
                }
                if (!def || !def->tlsIndex) {
                    if (error) *error = m->name() + ": TPOFF64 against a module without TLS";
                    return false;
                }
                *where = value + r.addend - def->tlsOffset;
                break;
            }
            default:
                if (error) *error = m->name() + ": unsupported relocation type " + std::to_string(r.type);
                return false;
            }
            ++st.relocations;
        }
        (void)STT_TLS;
    }
    if (stats) *stats = st;
    return true;
}

} // namespace host
