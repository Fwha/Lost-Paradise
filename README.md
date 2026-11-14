# lost-paradise-native

Runs *Fist of the North Star: Lost Paradise* (PS4, CUSA12776 or CUSA12781) natively on Linux and Windows. A C++ host loads
the game's original x86-64 code, answers its PS4 system calls itself, and recompiles its GPU
shaders to Vulkan.

Did all my testing on CachyOS and Windows 11.
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
- **Sound and controllers** through SDL, with motion controls from a DualShock 4, DualSense or Switch
  Pro controller's sensors. Waggling the mouse over the window shakes with any controller, for the
  bartender minigame; without sensors the right stick (IJKL) tilts.

## Build

Needs CMake, Ninja, a C++20 compiler, and the Vulkan, glslang, zlib and SDL2 development packages
(without SDL2 there is no window).

```sh
cmake -S . -B build -G Ninja
ninja -C build
scripts/fetch-nids.sh aerolib.csv     # import names, needed to run
```

Optional, for NVIDIA DLSS: run `scripts/fetch-dlss.sh`, then the two build commands again.

### Windows (experimental)

Each push builds a Windows zip: open the commit's **Windows** workflow run on GitHub and download
`Lost_Paradise-windows` under Artifacts. Unzip it, put the dump in the `game` folder next to
`Lost Paradise.exe` (the launcher makes the folder on its first start) and press Play; the
game's log goes to `play.log` in that folder. Releases carry the same zip. Needs Windows 10 1803
or later and a Vulkan driver. DLSS is Linux only for now.

To build it yourself, in an [MSYS2](https://www.msys2.org) UCRT64 shell:

```sh
pacman -S --needed zip mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,SDL2,vulkan-headers,vulkan-loader,glslang,zlib}
cmake -S . -B build -G Ninja && ninja -C build
scripts/fetch-nids.sh aerolib.csv
scripts/make-windows-zip.sh           # build/Lost_Paradise-windows.zip
```

Or cross-compile from Linux with MinGW-w64: `cmake/mingw-w64-x86_64.cmake` is the toolchain file.
How the port works: the game's code calls ours through System V thunks (`src/host/guest_abi.h`),
guest memory is built on placeholders (`src/host/mman_win.cpp`), and the thread pointer is a
TEB slot read through `%gs` (`src/host/tls.cpp`).

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

### Mods

Put mods in a `mods` folder in the install folder (the launcher makes it). Each folder in `mods`
is one mod, laid out like the game folder; its files are opened instead of the game's:

```
Lost Paradise/
  mods/
    Red Kenshiro/data/2dpar/sprite_e.par     replaces game/CUSA12776/data/2dpar/sprite_e.par
    HD Buttons/data/2dpar/ui_e.par/2d_jm_btn.dds
                                             replaces only 2d_jm_btn.dds inside ui_e.par
    20-HD Desert/data/stage/...
    _Old Mod/...                             off: the name starts with _ (or .)
```

A folder named like an archive (`ui_e.par/`) holds files to change inside that archive, by their
path in it, so a mod needs only the textures it changes. Archives inside archives work the same
way (`ui_e.par/inner.par/x.dds`). The changed copy is made at the first start and kept in
`lpn-data/mod-archives`, and made again when a mod's files change. Only files already in an
archive can be changed; the log names any that are not there.

Files laid out right in `mods` (`mods/data/...`) work too. When two mods replace the same file,
the one whose folder name sorts last wins, so number them to set the order; mods that change
different files inside one archive both apply. The log lists the mods found at start, and
`--trace-files` names the mod each file came from; `--no-mods` plays without them and
`--mods DIR` reads another folder.

### Launcher

`scripts/make-appimage.sh "<install folder>"` packs the launcher, `lpn-host` and `aerolib.csv` into
`Lost_Paradise-x86_64.AppImage` (it downloads appimagetool the first time) and copies it there.
The launcher has **Play**, **Settings** (the in-game PC Settings: the same `lpn-display.ini`, so a
change in one shows in the other) and a **Skip intro logos** box, which runs the boot as fast as
it draws until the title screen.

Settings opens a separate window of the desktop's own controls, in tabs (Display, Graphics,
Controller, Extras), with **Set To Default**, **Ok** and **Cancel**: Win32 controls on Windows, GTK 3
on Linux (opened at run time, so neither the build nor the AppImage needs it). From a controller, or
on Linux without GTK 3, Settings shows the launcher's own screen instead.

**Check for updates** (top right) asks GitHub for the latest release. From the AppImage, a newer
one is downloaded over it and started; a build from source is told to pull and rebuild. 

The **gear** (bottom right) opens **Tools**:
- **Unpack a .pkg** extracts a PS4 package into the game folder as `game/<title ID>`. Pick it with
  the desktop's file chooser or drop it on the window.
- **Apply an update or DLC**, from a .pkg or from a folder that is already unpacked, adds it to the
  game with the same title ID. An update makes a new version, `game/<title ID>-v<version>`: a copy
  of the game made of hard links (no extra room for the files the update leaves alone) with the
  update's files over it, so the game as it was stays too. DLC goes into
  `game/<title ID>-DLC/<label>` (lpn-host does not load DLC yet).

With more than one version in `game/`, **Play** asks which one to start.

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
