# lost-paradise-native

A native Linux reimplementation of *Fist of the North Star: Lost Paradise* (PS4, CUSA12776),
in the spirit of [Severed Chains](https://github.com/Legend-of-Dragoon-Modding/Severed-Chains),
written in C++.

**No game code or data is included.** You need your own decrypted dump.

## Plan

1. **Map the game.** Recover functions, imports and asset formats from `eboot.bin`.
2. **C++ host.** Load the original x86-64 code natively on Linux and route its 372 PS4 system
   calls to our own implementations (files, threads, SDL input/audio, Vulkan renderer).
3. **Replace piece by piece.** Rewrite each subsystem in C++ until no original code remains.
4. **Extras.** Uncapped framerate, resolution, widescreen, mods.

## Status

Step 1. `lpn-inspect` parses the decrypted eboot, resolves imports and recovers ~60k
function boundaries from `.eh_frame` for Ghidra.

## Build

```sh
cmake -S . -B build -G Ninja
ninja -C build
```

## Usage

```sh
scripts/fetch-nids.sh aerolib.csv
build/tools/inspect/lpn-inspect /path/to/CUSA12776/eboot.bin --nids aerolib.csv --out analysis/
```

Then in Ghidra, load `eboot.bin` (the [GhidraOrbis](https://github.com/astrelsky/GhidraOrbis)
loader is recommended), run auto-analysis, and run `ghidra/ImportLpnAnalysis.py` on `analysis/`.

## Layout

- `src/ps4elf/`: PS4 ELF, dynamic linking and `.eh_frame` parsing
- `tools/inspect/`: `lpn-inspect`
- `ghidra/`: Ghidra scripts
- `docs/`: findings about the game
