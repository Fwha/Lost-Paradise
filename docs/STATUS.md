# Status and handoff notes

Last updated 2026-10-16.

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
- VideoOut, event queues, memory pools, a PM4 walker that runs the game's GPU fences and
  end-of-pipe interrupts, and offline answers for every other system service (user, pad,
  save data, dialogs, trophies, PlayGo, network, Rtc, zlib; audio is paced but silent).
  Guest memory sits below 1 TiB because PM4 packets hold 40-bit addresses.
- The game now runs its frame loop steadily (~50 fps, ~8,300 GPU packets per frame) in
  English (USER_DEFINED_PARAM_4 from param.sfo), loading shaders, fonts, UI, sound banks and
  scenario data. Nothing is visible yet: frames are GPU-rendered and the GPU isn't emulated.
- GPU groundwork: GNM draw/shader helpers emit real PM4, the command processor tracks
  registers and reports each draw/dispatch, `--dump-frame N` writes a frame report and its
  shaders, and `lpn-gcn` disassembles GCN (tested against a real shader).
- GPU rendering (Vulkan): GCN shaders are recompiled to GLSL/SPIR-V at draw time and every
  draw and dispatch of the frame is replayed. The boot now shows the game's own loading
  screen (black, with the Lost Paradise logo). See `docs/gpu.md` for how it works.
- Gameplay renders lit (PRs #10, #11): the white-out came from ignored HTILE/CMASK fast
  clears; the black environment from a PC-relative constant table the shader analysis lost
  (`s_addc_u32`), which zeroed ambient occlusion. Cube-map sampling is translated, and so are
  `ds_swizzle_b32` (quad mode), `v_mad_u64_u32` and `v_cvt_off_f32_i4`.
- Performance (RTX 4070 Ti SUPER, `LPN_PROFILE=1`, headless): gameplay in Shin's Castle
  (~1,500 draws) runs at 40 fps, up from 15 (shader analyses are replayed instead of redone,
  texture checks hashed in parallel); menus reach the game's 30 fps cap, the opening cutscene
  50-60 fps. In the first fight (Rush Combos tutorial, ~2,800 draws) about 28 fps, up from 19,
  after sharing render passes between draws; first-time effects no longer hitch on later runs
  (SPIR-V cache).
- Boot reaches the title, main menu and difficulty select, and starts a new game (the
  opening cutscene `auth/a00_0100_before.par` loads in Southern Town). Scripted presses
  (`--press`) drive menus in headless runs.
- `--no-gpu` runs ignore the title press: the game reads back compute results. Use
  `--compute-only` (runs dispatches, skips draws, ~20 fps on lavapipe) for progress runs.
- Earlier first run: libc starts, the game prints "test2", starts its file threads, sets up memory
  pools, opens video out, builds GNM command buffers and submits them, then spins waiting for
  the first flip (no VideoOut/GPU yet).

## Open questions / unverified

- GMD vertex flag → field mapping is inferred from 3 layouts; unknown bits are rejected.
- Rigid (non-skinned) meshes are untested; Kenshiro has none.
- GMT curve formats 28/29 (hand/face patterns) are not decoded.
- PARC offset high-32-bits field and SLLZ versions other than 1 are unseen.

## Next steps

1. Performance in crowded areas (Eden plaza: ~3,600 draws, 12.5 fps before the 2026-10-17 work).
   In the first fight (2,800 draws) the frame is now ~25 ms: ~8 ms shader-analysis replays,
   ~8 ms waiting for the GPU (~900 render passes, a full barrier before each), ~3 ms binding.
   Static textures are re-checked every 8th frame; recently changed ones every frame.
2. Slow loading after Continue (~45 s): the game advances a loading screen once per frame (~890
   frames for a save, the same at 20 or 90 fps) and asks for 20 fps there (flip rate 2). The host
   now runs those screens as fast as they draw (`LPN_LOADING_HZ`, default 500; `LPN_FAST_LOADING=0`
   turns it off). `LPN_PROFILE=1` adds [wait] lines (what each game thread is blocked in, with
   guest addresses) and [io] lines.
3. Fight glitches: a one-frame black wedge over the HP bar (gone with `LPN_GPU_SYNC=1`, so likely
   a race between the game rewriting HUD data and our asynchronous GPU); the kick trail and torch
   flames sometimes draw green (particle colour tables; parked, see the 2026-10-17 session notes).
4. Rendering: 2D-tiled texture detiling (tile mode 14), indirect draws, a real swapchain.
   `docs/gpu.md` lists what is known to be missing.
5. Audio through SDL.

## Working setup

- Dump on the owner's PC: `~/Documents/Fist of the North Star/` (`CUSA12776/`, `sys_modules/`).
  Never commit game files; `.gitignore` excludes them.
- Build: `cmake -S . -B build -G Ninja && ninja -C build && ctest --test-dir build`.
- Run: `build/src/host/lpn-host <path>/CUSA12776 --nids aerolib.csv` (`--strict` stops at the
  first unimplemented call, `--list-unimplemented` lists what is still stubbed,
  `--trace-files` logs every file opened, `--headless` skips the window,
  `--dump-frame N` writes ./gpu-dump; Ctrl+C prints the stub summary).
- Progress runs: `--compute-only --trace-files --press cross@1500,...` (frame numbers; on a real GPU
  `cross@1500,cross@1800,cross@2100,cross@2400` reaches gameplay in Shin's Castle by frame
  ~3300-4500, depending on how fast the opening cutscene plays; use a fresh `--data` folder, as a
  save file changes the menus);
  `--trace-hle N` logs each system call's first use, and first use after frame N;
  `--screenshot-every N` saves every Nth frame (needs full rendering, ~1 fps on lavapipe).
- `lpn-gcn shader.bin` disassembles; `lpn-gcn --stats gpu-dump/shaders/*` counts instructions;
  `lpn-gcn --glsl shader.bin` shows the translated GLSL.
- Rendering needs Vulkan 1.3 with VK_EXT_external_memory_host (any current Linux driver;
  lavapipe works for testing) and glslang to build (`libvulkan-dev glslang-dev`).
  `--no-gpu` skips it, `--screenshot N` saves the Nth frame as screenshot.ppm,
  `LPN_PROFILE=1` prints, once a second and stamped `t=<seconds>`, fps with the worst frame and
  where the renderer's time goes (`[prof]`), the command processor's waits (`[pm4]`) and each
  thread's CPU use (`[cpu]`, host threads are named `lpn-*`), plus each frame over 50 ms
  (`[prof] slow frame`); F10 writes `[mark] N at t=...` to tag a spot where the fps drops,
  `LPN_GPU_SYNC=1` waits for the GPU after every submission (to rule out async issues),
  `LPN_GPU_TRACE=<frame>` logs every draw of that frame and dumps its render targets,
  adding `LPN_GPU_SNAP=1` saves the target after each draw (or `LPN_GPU_SNAP=2750-2760,2886` for
  just those draws); 32-bit float targets also get their min/max/mean logged.
  `LPN_GPU_GLSL=1` writes every translated shader as gpu-glsl-<stage>-<entry>-<id>.glsl.
  `LPN_DBG_PS=<entry>[,<entry>...] LPN_DBG_OUT="<vec4 args>"` replaces those pixel shaders' color
  output (e.g. `"iA4.xyz * 4.0, 1.0"`); `dbgT<n>` holds image n's last sample and
  `LPN_DBG_CAP="<GLSL line text>|<expr>"` captures `vec4(expr)` into `dbgC` after that line.
  `LPN_PROBE=<ps entry hex>:<pc hex>:<v>,<v>,<v>` shows three VGPRs of that pixel shader, as they
  are just before the instruction at pc, on its colour target 0 (magenta where pc is never reached);
  `LPN_NO_KILL=<ps entry hex>` stops that pixel shader discarding.
  `LPN_PM4_TRACE=<frame>` logs that frame's DMA_DATA and EVENT_WRITE packets;
  `LPN_GPU_STATS=<n>` prints PM4 opcode counts for the 300 submissions ending at submission n.
- Captures while playing: F12 saves the displayed frame and a GPU trace of the next one (every
  draw, its shaders' resources, all render targets) to `lpn-captures/capture-<n>/`;
  `LPN_CAPTURE_AT=<flip>` does the same in scripted runs. Run with `LPN_PROFILE=1 ... 2> play.log`
  to also log every frame over 50 ms.
- Controls: arrows = d-pad, WASD/IJKL = sticks, Enter = Cross, Backspace = Circle,
  Z = Square, X = Triangle, Q/E = L1/R1, 1/3 = L2/R2, Esc = Options, Tab = touchpad, F12 = capture;
  any SDL game controller works too.
- Building the window needs SDL2 (`pacman -S sdl2`); without it the host runs headless.
