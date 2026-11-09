// Tests for the mods folder's index (common/mods.h): which file a game path opens.

#include "common/mods.h"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        if ((a) != (b)) {                                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            ++failures;                                                                             \
        }                                                                                           \
    } while (0)

// Same file, whatever the separators (Windows' iterator gives \\, the test writes /) and, as
// Windows' folders are not case sensitive (Data/ and data/ are one there), the case.
std::string lowered(const fs::path& p) {
    std::string s = p.lexically_normal().generic_string();
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
bool same(const std::string& a, const fs::path& b) { return !a.empty() && lowered(a) == lowered(b); }

void put(const fs::path& p) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << p.string();
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "lpn_mods_test";
    fs::remove_all(root);
    const fs::path mods = root / "mods";

    host::ModIndex none;
    none.scan((root / "missing").string());
    CHECK_EQ(none.mods().size(), size_t(0));
    CHECK_EQ(none.find("/data/2dpar/sprite_e.par"), std::string());

    put(mods / "10-low" / "data/2dpar/sprite_e.par");
    put(mods / "10-low" / "data/chara/kenshiro.gmd");
    put(mods / "20-high" / "Data/2dpar/Sprite_E.par");
    put(mods / "_off" / "data/sound/bgm.at9");
    put(mods / ".hidden" / "data/sound/bgm.at9");
    put(mods / "data/stage/desert.par");          // loose
    put(mods / "data/chara/kenshiro.gmd");        // loose, beaten by 10-low
    put(mods / "readme.txt");                     // not a game folder: ignored
    put(mods / "10-low" / "data/2dpar/ui_e.par/2d_jm_btn.dds");
    put(mods / "10-low" / "data/2dpar/ui_e.par/inner.par/x.dds");
    put(mods / "20-high" / "data/2dpar/UI_E.par/2d_jm_btn.dds");
    put(mods / "20-high" / "data/2dpar/ui_e.par/2d_jm_qte.dds");

    host::ModIndex ix;
    ix.scan(mods.string());
    CHECK_EQ(ix.mods().size(), size_t(3));
    CHECK_EQ(ix.mods()[0].name, std::string());
    CHECK_EQ(ix.mods()[1].name, std::string("10-low"));
    CHECK_EQ(ix.mods()[2].name, std::string("20-high"));
    CHECK_EQ(ix.size(), size_t(3));
    CHECK_EQ(ix.archives(), size_t(1));

    // Changes inside an archive: per file, the last mod wins; different files all apply.
    const host::ModIndex::Patches* pt = ix.patches("/data/2dpar/ui_e.par");
    CHECK_EQ(pt != nullptr, true);
    if (pt) {
        CHECK_EQ(pt->size(), size_t(3));
        CHECK_EQ(pt->count("2d_jm_btn.dds") && same(pt->at("2d_jm_btn.dds"), mods / "20-high" / "data/2dpar/UI_E.par/2d_jm_btn.dds"), true);
        CHECK_EQ(pt->count("2d_jm_qte.dds"), size_t(1));
        CHECK_EQ(pt->count("inner.par/x.dds"), size_t(1));
    }
    CHECK_EQ(ix.find("/data/2dpar/ui_e.par"), std::string()); // not a whole-file replacement
    CHECK_EQ(ix.patches("/data/2dpar/sprite_e.par") == nullptr, true);

    // The last mod by name wins, paths match in any case and with a leading / or dots.
    CHECK_EQ(same(ix.find("/data/2dpar/sprite_e.par"), (mods / "20-high" / "Data/2dpar/Sprite_E.par")), true);
    CHECK_EQ(same(ix.find("data/./x/../2dpar/SPRITE_E.PAR"), (mods / "20-high" / "Data/2dpar/Sprite_E.par")), true);
    CHECK_EQ(same(ix.find("/data/chara/kenshiro.gmd"), (mods / "10-low" / "data/chara/kenshiro.gmd")), true);
    CHECK_EQ(same(ix.find("/data/stage/desert.par"), (mods / "data/stage/desert.par")), true);
    CHECK_EQ(ix.find("/data/sound/bgm.at9"), std::string());
    CHECK_EQ(ix.find("/readme.txt"), std::string());
    CHECK_EQ(ix.find("/data/2dpar"), std::string()); // folders are never replaced
    CHECK_EQ(ix.find("/../mods/data/stage/desert.par"), std::string());

    const host::Mod* m = ix.modOf(ix.find("/data/2dpar/sprite_e.par"));
    CHECK_EQ(m ? m->name : std::string("?"), std::string("20-high"));
    m = ix.modOf(ix.find("/data/stage/desert.par"));
    CHECK_EQ(m ? m->name : std::string("?"), std::string());

    fs::remove_all(root);
    if (failures) std::fprintf(stderr, "%d failures\n", failures);
    else std::printf("mods_test: ok\n");
    return failures ? 1 : 0;
}
