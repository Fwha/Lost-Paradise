# Status and handoff notes

Last updated 2026-10-12.

## Plan

1. **Map the game** (mostly done): code, imports and asset formats.
2. **C++ host** (next): load the original x86-64 eboot natively on Linux and route its 372
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

## Open questions / unverified

- GMD vertex flag → field mapping is inferred from 3 layouts; unknown bits are rejected.
- Rigid (non-skinned) meshes are untested; Kenshiro has none.
- GMT curve formats 28/29 (hand/face patterns) are not decoded.
- PARC offset high-32-bits field and SLLZ versions other than 1 are unseen.

## Next steps

1. Step 2 groundwork: an ELF loader in the host that maps the eboot segments at their
   addresses, applies relocations, and binds each import slot to a C++ stub that logs its
   name. Then run the entry point until the first unimplemented call.
2. Implement libkernel/libc basics first (memory, threads, mutexes, files mapped to the dump).
3. Graphics last within step 2: GNM command buffers (PM4) and GCN shaders → Vulkan.

## Working setup

- Dump on the owner's PC: `~/Documents/Fist of the North Star/` (`CUSA12776/`, `sys_modules/`).
  Never commit game files; `.gitignore` excludes them.
- Build: `cmake -S . -B build -G Ninja && ninja -C build && ctest --test-dir build`.
