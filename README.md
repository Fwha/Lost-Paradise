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

Step 1.
- `lpn-inspect` parses the decrypted eboot, resolves imports and recovers ~60k function
  boundaries from `.eh_frame` for Ghidra.
- `lpn-par` lists and extracts `.par` archives, including SLLZ decompression and nested
  archives. See [docs/formats.md](docs/formats.md).
- `lpn-gmd` reads GMD models (meshes, skeleton, material bindings) and exports OBJ for viewing,
  optionally posed by an animation.
- `lpn-gmt` reads GMT skeletal animations.

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

Unpack an archive (add `--recursive` to unpack archives nested inside it):

```sh
build/tools/par/lpn-par list    /path/to/CUSA12776/data/hact/c10_0102.par --recursive
build/tools/par/lpn-par extract /path/to/CUSA12776/data/hact/c10_0102.par out/c10_0102 --recursive
build/tools/par/lpn-par extract /path/to/CUSA12776/data/chara/ps4/adv/c_ca_ken/mesh.par out/mesh
build/tools/gmd/lpn-gmd info out/mesh/c_ca_ken.gmd
build/tools/gmd/lpn-gmd obj  out/mesh/c_ca_ken.gmd out/mesh/ken.obj   # textures resolve from the same folder
build/tools/gmt/lpn-gmt info out/c10_0102/000.par/c10_000_11.gmt
build/tools/gmd/lpn-gmd obj  out/mesh/c_ca_ken.gmd out/mesh/posed.obj \
    --anim out/c10_0102/000.par/c10_000_11.gmt --frame 80
ctest --test-dir build
```

Then in Ghidra, load `eboot.bin` (the [GhidraOrbis](https://github.com/astrelsky/GhidraOrbis)
loader is recommended; Ghidra's built-in ELF loader also maps the segments correctly), run
auto-analysis, and run `ghidra/ImportLpnAnalysis.py` on `analysis/`. Headless (Ghidra 12 needs
the bundled Jython extension installed and JDK 21+):

```sh
analyzeHeadless ghidra-project LostParadise -import /path/to/eboot.bin \
    -scriptPath ghidra -postScript ImportLpnAnalysis.py "$PWD/analysis"
```

If no loader accepts the file, import it as Raw Binary (`x86:LE:64:default`, `gcc`, base 0) and
run `ghidra/MapLpnSegments.py` on `analysis/` before analysis to map the segments.

## Layout

- `src/ps4elf/`: PS4 ELF, dynamic linking and `.eh_frame` parsing
- `src/formats/`: game file formats (PARC, SLLZ, GMD, GMT) and skinning
- `tools/inspect/`: `lpn-inspect`
- `tools/par/`: `lpn-par`
- `tools/gmd/`: `lpn-gmd`
- `tools/gmt/`: `lpn-gmt`
- `tests/`: unit tests on synthetic data
- `ghidra/`: Ghidra scripts
- `docs/`: findings about the game
