# lost-paradise-native

Runs *Fist of the North Star: Lost Paradise* (PS4, CUSA12776) natively on Linux. A C++ host loads
the game's original x86-64 code, answers its PS4 system calls itself, and recompiles its GPU
shaders to Vulkan.

Did all my testing on CachyOS.
PC Specs:
4070 ti SUPER
i7 12700k
32GB RAM

**No game code or data is included.** You need your own dump.

The game boots, reaches the menus and plays. See [docs/STATUS.md](docs/STATUS.md) for progress
and known issues.

## Features

- **Display modes:** windowed, borderless fullscreen or fullscreen.
- **Resolutions:** 720p, 896p (PS4), 1080p (PS4 Pro) and native 1440p output.
- **Aspect ratio:** 16:9, 16:10 or 21:9. The game renders at that shape (2560x1080 for 1080p at
  21:9) with its 3D view widened to match (same vertical field of view, so 21:9 shows more to the
  sides and 16:10 a little less), and the HUD stays 16:9 in the middle. Not stretched.
- **Frame rate:** cap at 30, 60 or 90 fps, or uncapped, with the game still running at normal
  speed. The title screen, menus and real-time cutscenes go uncapped too. Optional FPS counter.
- **DLAA:** NVIDIA DLSS anti-aliasing in place of the game's SMAA (RTX cards, see Build).
- **PC Settings:** these options from the F11 menu, from a PC SETTINGS entry in the title
  and pause menus, or from the launcher.
- **Launcher:** an AppImage with Play, Settings, Skip intro logos and a .pkg unpacker.
- **Sharper image:** full texture mipmaps, anisotropic filtering and smooth scaling to the window.
- **Performance:** draws recorded on several threads, shader analysis ahead of the frame, and a
  pipeline cache that removes stutter on later runs.
- **Faster save loading:** loading screens run as fast as they draw.
- **Save/load menu:** pick a slot to save, load, overwrite or delete, with icon, chapter, date,
  play time and controller button hints.
- **Sound and controllers** through SDL.

## Build

Needs CMake, Ninja, a C++20 compiler, and the Vulkan, glslang, zlib and SDL2 development packages
(without SDL2 there is no window).

```sh
cmake -S . -B build -G Ninja
ninja -C build
scripts/fetch-nids.sh aerolib.csv     # import names, needed to run
```

Optional, for NVIDIA DLSS: run `scripts/fetch-dlss.sh`, then the two build commands again.

## Run

Lay out an install folder like this, with the dump (the folder with `eboot.bin` and
`sce_module/`) inside a `game` folder:

```
Lost Paradise/
  game/CUSA12776/          the dump
  lost-paradise/           this repo, built
  launch.sh                or Lost_Paradise-x86_64.AppImage
```

Saves go to `lpn-data/savedata/<title ID>` (one folder per version, e.g. `CUSA12776`, read from
the dump's `sce_sys/param.sfo`), the PC settings to `lpn-display.ini` and the log to `play.log`, all
in that folder. Saves from before the split are moved into `lpn-data/savedata/CUSA12776` on the
first run. `lpn-host` looks for the game in `game/` of the folder it runs in, or takes its path
first:

```sh
build/src/host/lpn-host /path/to/CUSA12776 --nids aerolib.csv
```

`--data DIR` changes the save folder; `lpn-host` with no arguments lists every option.

### Launcher

`scripts/make-appimage.sh "<install folder>"` packs the launcher, `lpn-host` and `aerolib.csv` into
`Lost_Paradise-x86_64.AppImage` (it downloads appimagetool the first time) and copies it there.
The launcher has **Play**, **Settings** (the in-game PC Settings: the same `lpn-display.ini`, so a
change in one shows in the other) and a **Skip intro logos** box, which runs the boot as fast as
it draws until the title screen.

**Check for updates** (top right) asks GitHub for the latest release. From the AppImage, a newer
one is downloaded over it and started; a build from source is told to pull and rebuild. 

The **gear** (bottom right) opens **Tools**. **Unpack a .pkg** extracts a PS4 package into the game
folder, as `game/<title ID>` (`-UPDATE` for a patch, `-DLC` for add-on content): pick it with the
desktop's file chooser (zenity or kdialog) or drop it on the window. It works with fake PKGs (fpkg,
made with the homebrew tools' published keys) and refuses a retail PKG, whose files are encrypted
with keys only a PS4 has. It never writes over a folder that is already there. The same from a
terminal: `build/src/pkg/lpn-pkg extract <file.pkg> [game folder]` (`lpn-pkg info` shows what a
package is). The extractor (`src/pkg`) is written from the format's documentation, with no code
from GPL tools such as shadPS4's old extractor.

### launch.sh

A `launch.sh` in the install folder keeps it to one command and writes the log to `play.log`:

```sh
#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec lost-paradise/build/src/host/lpn-host --nids lost-paradise/aerolib.csv --data lpn-data "$@" 2> play.log
```

In the game window, **F11** opens the display menu (window mode, resolution, FPS cap) and
**F12** saves a debug capture to `lpn-captures/`.

### Tuning

Defaults suit most machines. These environment variables override them:

| Variable | Default | Effect |
| --- | --- | --- |
| `LPN_RECORD_THREADS` | 6 on 16+ hardware threads, else 0 | Threads that record draws in parallel. `0` turns them off. |
| `LPN_ANALYSIS_WORKERS` | 6 on 16+ threads, 3 on 8+, else 1 | Threads that analyse shaders and draws. |
| `LPN_PROFILE=1` | off | Prints per-frame timings. |
| `LPN_HUD_STRETCH=1` | off | With a 16:10 or 21:9 aspect ratio, stretch the HUD over the whole picture instead of keeping it 16:9. |
| `LPN_HUD_LOG=1` | off | Logs which draws are treated as HUD (`[hud]` lines), for reporting HUD problems. |

Example: `LPN_RECORD_THREADS=0 ./launch.sh`

## Tools

Command-line tools for studying the game files, built with the rest:

| Tool | Does |
| --- | --- |
| `lpn-par` | Lists and extracts `.par` archives (`--recursive` for nested ones) |
| `lpn-gmd` | Reads GMD models and exports OBJ, optionally posed by an animation |
| `lpn-gmt` | Reads GMT skeletal animations |
| `lpn-inspect` | Names the eboot's imports and finds function starts for Ghidra |
| `lpn-gcn` | Disassembles GCN shaders |

```sh
build/tools/par/lpn-par extract CUSA12776/data/chara/ps4/adv/c_ca_ken/mesh.par out/mesh
build/tools/gmd/lpn-gmd obj out/mesh/c_ca_ken.gmd out/mesh/ken.obj
```

File formats are written up in [docs/formats.md](docs/formats.md), and Ghidra import scripts are
in [ghidra/](ghidra/). Run the tests with `ctest --test-dir build`.

## Layout

- `src/host/`: the host (`lpn-host`): loader, system calls, Vulkan renderer
- `src/gcn/`: GCN shader decoding and translation to SPIR-V
- `src/ps4elf/`: PS4 ELF and dynamic linking
- `src/formats/`: game file formats (PARC, SLLZ, GMD, GMT)
- `tools/`: the command-line tools
- `tests/`: unit tests
- `docs/`: findings about the game and how the host works

## License

MIT, see [LICENSE](LICENSE). This covers the code in this repository only, not the game, which
you must own.

Third-party code and art keep their own licenses:

- `external/stb`: MIT or public domain (`external/stb/LICENSE`).
- `external/nanosvg`: zlib (notice at the top of each header).
- `external/controllerimage`: button art is public domain, see its `DATA-LICENSE.txt` and
  `credits.txt`.
- NVIDIA DLSS SDK (optional, fetched by `scripts/fetch-dlss.sh`, not in this repository):
  NVIDIA RTX SDKs license. Release builds that bundle the DLSS runtime are subject to it.
- `aerolib.csv` (fetched by `scripts/fetch-nids.sh`, not in this repository): from
  [zecoxao/sce_symbols](https://github.com/zecoxao/sce_symbols).

This project is not affiliated with or endorsed by Sega, Sony or NVIDIA.
