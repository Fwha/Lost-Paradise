# Status and handoff notes

Last updated 2026-10-13.

## Plan

1. **Map the game** (mostly done): code, imports and asset formats.
2. **C++ host** (in progress): load the original x86-64 eboot natively on Linux and route its 372
   PS4 system calls to our own code (files, threads, SDL input/audio, Vulkan renderer).
3. **Replace piece by piece**: rewrite subsystems in C++ until no original code remains.
4. **Extras**: uncapped framerate, resolution, widescreen, mods.

## Done

- `lpn-inspect`: parses the decrypted eboot (CUSA12776), names all 372 imports by NID, and
  recovers 60,499 function starts from `.eh_frame`. See `docs/recon.md`.
- Ghidra project on the owner's PC: `~/Documents/Fist of the North Star/ghidra-project/LostParadise`
  (Ghidra 12.1.2, built-in ELF loader, 61,467 functions, 791/792 import slots labelled).
  Ghidra 12 needs Java 21+ (system default set to java-27-openjdk) and the Jython extension.
- `.par` archives and SLLZ v1 compression (PR #1).
- Textures are plain linear BC1/BC3 DDS: no conversion needed.
- GMD models (PR #2), verified on Kenshiro (`chara/ps4/adv/c_ca_ken/mesh.par`).
- GMT animations and skinning (PR #3), verified with `hact/c10_0102.par` clips.
- Format write-ups: `docs/formats.md`.
- `lpn-host` (src/host): loads the eboot plus the game's own `libc.prx` and `libSceFios2.prx`
  natively, applies all 126,563 relocations, and redirects the eboot's 5,037 `mov reg, fs:[0]`
  reads to a per-thread guest TCB. libkernel basics are implemented in C++ (memory, threads,
  mutexes, condvars, semaphores, files, time). Every other import goes to a stub that logs the
  call and returns 0; a fault prints registers as `module+offset` for Ghidra.
- First run: libc starts, the game prints "test2", starts its file threads, sets up memory
  pools, opens video out, builds GNM command buffers and submits them, then spins waiting for
  the first flip (no VideoOut/GPU yet).

## Open questions / unverified

- GMD vertex flag → field mapping is inferred from 3 layouts; unknown bits are rejected.
- Rigid (non-skinned) meshes are untested; Kenshiro has none.
- GMT curve formats 28/29 (hand/face patterns) are not decoded.
- PARC offset high-32-bits field and SLLZ versions other than 1 are unseen.

## Next steps

1. Event queues + VideoOut: a window (SDL) and real flip events so the main loop advances.
2. Memory pools (`sceKernelMemoryPool*`), UserService, Pad, SaveData, AudioOut stubs with
   sensible results instead of 0.
3. GNM: parse the submitted PM4 command buffers, then GCN shaders → SPIR-V → Vulkan.

## Working setup

- Dump on the owner's PC: `~/Documents/Fist of the North Star/` (`CUSA12776/`, `sys_modules/`).
  Never commit game files; `.gitignore` excludes them.
- Build: `cmake -S . -B build -G Ninja && ninja -C build && ctest --test-dir build`.
- Run: `build/src/host/lpn-host <path>/CUSA12776 --nids aerolib.csv` (`--strict` stops at the
  first unimplemented call, `--list-unimplemented` lists what is still stubbed).
