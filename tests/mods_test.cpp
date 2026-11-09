// Tests for the mods folder's index (common/mods.h): which file a game path opens.

#include "common/mods.h"

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

    host::ModIndex ix;
    ix.scan(mods.string());
    CHECK_EQ(ix.mods().size(), size_t(3));
    CHECK_EQ(ix.mods()[0].name, std::string());
    CHECK_EQ(ix.mods()[1].name, std::string("10-low"));
    CHECK_EQ(ix.mods()[2].name, std::string("20-high"));
    CHECK_EQ(ix.size(), size_t(3));

    // The last mod by name wins, paths match in any case and with a leading / or dots.
    CHECK_EQ(ix.find("/data/2dpar/sprite_e.par"), (mods / "20-high" / "Data/2dpar/Sprite_E.par").string());
    CHECK_EQ(ix.find("data/./x/../2dpar/SPRITE_E.PAR"), (mods / "20-high" / "Data/2dpar/Sprite_E.par").string());
    CHECK_EQ(ix.find("/data/chara/kenshiro.gmd"), (mods / "10-low" / "data/chara/kenshiro.gmd").string());
    CHECK_EQ(ix.find("/data/stage/desert.par"), (mods / "data/stage/desert.par").string());
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
