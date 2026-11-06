# lost-paradise-native

Runs *Fist of the North Star: Lost Paradise* (PS4, CUSA12776) natively on Linux. A C++ host loads
the game's original x86-64 code, answers its PS4 system calls itself, and recompiles its GPU
shaders to Vulkan.

**No game code or data is included.** You need your own decrypted dump.

The game boots, reaches the menus and plays. See [docs/STATUS.md](docs/STATUS.md) for progress
and known issues.

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

Put the decrypted dump (the folder with `eboot.bin` and `sce_module/`) somewhere, then:

```sh
build/src/host/lpn-host /path/to/CUSA12776 --nids aerolib.csv
```

Saves go to `./lpn-data` (change with `--data DIR`). `lpn-host` with no arguments lists every
option.

A `launch.sh` beside the repo and the dump keeps this to one command and writes the log to
`play.log`:

```sh
#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec lost-paradise/build/src/host/lpn-host CUSA12776 --nids lost-paradise/aerolib.csv --data lpn-data "$@" 2> play.log
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
