# Native renderer

The graphics layer is being rewritten the way [GR2port](https://github.com/junminlee2004/GR2port)
rewrote Gravity Rush 2's: the game's frames are drawn by our own Vulkan renderer, pass by pass,
with handwritten GLSL that recreates what the game's GCN shaders draw. The game's PS4 command
buffers, the PM4 command processor's draw replay and the GCN-to-GLSL recompiler go away at the
end. This file is the plan and the log of what has moved over.

## What GR2port does

- Hooks the game's frame graph (one function its draw thread calls each frame) and, at its entry,
  reads the frame's scene straight from the engine's CPU-side data: draw lists, meshes, materials,
  bone palettes, lights, cameras. That scene goes into a ring of packets.
- A render thread of its own draws each packet with its own passes (world, characters, shadows,
  fog, water, particles, bloom, tonemap, UI), its own pipelines (all built at boot and cached) and
  its own swapchain.
- The game's per-item draw loops return at once, so it never builds GPU commands for the world.
  Nothing executes command buffers: the GPU writes the game's CPU waits on (frame counters,
  labels) are made at the moment the game asks for them, and a flip goes straight to the
  presenter.
- The screen drawing (UI) is still built by the game and captured from its packets.

## Why Lost Paradise moves over in stages

Lost Paradise's engine (the `pxd` engine with the `devil` shader set: about 3,000 shaders in
`data/shader/sh_devil_ps4.par`, plus `mars_shader_ps4.par` and the particle sets) has far more
passes than one change can rebuild: cascaded and per-character shadow maps, the G-buffer,
deferred lights, SSAO, reflections, glow and glare, focus blur, haze, the sea, crowds, toon
edges and hatching, particles, the UI, and minigame targets (karaoke, mahjong, golf, batting,
the print club). GR2port's renderer is about 43,000 lines. So passes move one at a time, and
every step leaves a game that plays:

- The current renderer (PM4 replay with recompiled shaders) stays as the fallback. A draw that no
  native pass claims is drawn by it, as today.
- A native pass claims the game's draws of one pass (found by the names of their shaders) and
  draws that pass itself into the same render target images, so passes before and after it
  don't notice.
- PC Settings > Renderer (`renderer = native | translated` in `lpn-display.ini`) switches every
  native pass off for A/B comparison.
- Each pass is checked against the translated one on the same frame.

## Stages

0. **Groundwork** (first change):
   - Shader names. At start the host indexes the game's four shader archives and names every
     shader the game binds (`ps_aa_filter2`, `csdw_cas0_c0_skin[pt]`...) by its code. The frame
     trace and the pass map use these names.
   - Pass map (`LPN_PASS_MAP=1`): every 10 seconds, one frame's passes in order, each with its
     targets (address, size, format), draw count and shader names, written to
     `lpn-captures/passmap-<frame>.txt`. This is the map the later stages work from.
   - The native pass framework (`src/host/native/`): passes, the registry that matches draws to
     them, a pipeline and GLSL toolkit of its own, and the setting.
   - First native passes: the self-contained full-screen passes at the end of the frame, written
     from the disassembly of the game's shaders.
1. **Draw state from the engine.** Hook the engine's state commit (0xeabdb0, the Gnmx-style
   constant update before every draw) and its draw calls (0xeac430, the auto-index draws) so a
   native pass gets textures, buffers and constants from the engine's state block, not from PM4
   and scalar-code replay.
2. **Scene capture.** Find the frame's render entry and the draw-command lists behind about 60% of
   all draws (0xea3f40), and capture meshes, materials, matrices and bone palettes into a scene
   ring at the frame's start, as GR2port does. Native shadow maps and G-buffer with handwritten
   material shaders per shader family (the archive's names group them: `csdw_*` shadow casters,
   ...).
3. **Lighting and post.** Deferred lights, SSAO, reflections, glow, glare, focus blur, haze; SMAA
   gives way to DLSS/DLAA natively.
4. **Characters, effects, UI.** Skinning, toon edges and hatching, particles, the sea, crowds;
   the UI captured from the game's packets.
5. **No more GPU commands.** The game's draw loops return; fences and flips are answered at once
   (the PM4 walker keeps only those); pipelines are all built at boot; the recompiler stays only
   for whatever minigame pass has not been rebuilt.

## Log

### Stage 0 (groundwork)

- `native/shader_names`: at start the host indexes the four archives (2,867 distinct programs in
  3,881 entries) on a thread of its own. A program's key is an FNV-1a hash of its code from the
  `ff 03 eb be` start up to and including the `OrbShdr` footer; the same bytes are read from
  guest memory at a bound shader's address, cached by address and the first 64 bytes. Checked
  offline: the renderer's SMAA hashes are `ps_aa_filter_edge2`, `ps_aa_filter_scan_edge2` and
  `ps_aa_filter2`. F12 traces now name each draw's shaders.
- `native/pass_map`: `LPN_PASS_MAP=1` (every 10 s, or `=<seconds>`) and every F12 capture write
  `passmap-<frame>.txt`: passes split where the render targets change, each with its targets and
  runs of draws by shader pair (`12x vs_x.vso | ps_y.pso`) and dispatches.
- `native/native.h`: `Pass` (claims pixel shaders by name, draws a `Draw` with a `Gpu`, may
  decline), `Draw` (targets, state, the pixel shader's textures by the pc of their first sample and
  its scalar loads by pc) and `Gpu` (device, shaders from GLSL, recording in order). The adapter is
  `nativeClaimed`/`nativeDraw` in `gpu/renderer.cpp`: claimed draws are never put on recorder
  threads, the pass records between two full barriers outside any render pass, and the renderer
  then forgets what it had bound and marks the targets written.
- `native/aa.cpp`: the game's MLAA (edges, blend weights, blend) in handwritten GLSL from the
  disassembly. It only runs with the Upscaler off (with one on, DLSS/FSR replace those draws).
  It declines a draw with blending, depth or stencil, more than one target, anything but one
  3-vertex rectangle, or a viewport other than the whole picture, and logs why once. Open: the
  rectangle's own vertices are not checked (the upscaler path assumes the same full-screen draw);
  the game's fp16 round-toward-zero export is not reproduced.
- PC Settings > Native Renderer (`renderer = native | translated`), applied at the next frame.
- The game's MLAA writes the edge pixels into the stencil (DB_DEPTH_CONTROL 0x771, ref 0x40)
  and runs weights and blend only there (0x571); the blend writes over the scene in place. The
  native passes attach the depth/stencil buffer and keep that state.
- 2026-11-19: confirmed in play at 2560x1440 (Upscaler Off): all three drawn natively, the
  picture as before.

### Stage 1 (draw origins)

- Every draw carries the engine code that drew it (`native/draw_origin.h`: three return addresses
  above the Gnm draw wrappers, written as a NOP into the space the game reserves ahead of each
  draw packet); pass maps list them. `docs/frame-map.md` is the first frame mapped with them.

### Stage 2 (shadow maps, first part)

- `native/shadow.cpp`: the casters of the sun's cascades and the near maps
  (`vs_shadowmap_normal[_pt]` + `ps_shadowmap[_pt][fp32][bias]`, about 500 draws a frame) in
  handwritten GLSL: vertices pulled from the game's buffers through the analysis's fetch slots,
  the matrix chain folded into one matrix, the game's slope bias and alpha test. Works with the
  Upscaler on too.
- Native draws now share render passes (`Gpu::beginRendering`) and take part in the translated
  path's barrier tracking. Still drawn from the game's draws; switching the game's shadow code
  off and drawing from captured scene data comes next.

### Stage 2 (shadow maps from the engine)

- Skinned casters (`vs_shadowmap_skin`) natively: four bones from the palette, the floor-relative
  translation blend.
- `native/engine_capture`: a hook on the engine's drawIndex (0xeac430) reads each stage shadow
  caster draw of 0xa51100 from the Gnm context's state block (vertex buffer +0x288, stride in
  +0x298, capacity +0x294, index buffer +0x688/+0x690, view constants' V# +0x20c8). The casters'
  list, mesh ranges and batching are in `docs/frame-map.md`'s engine notes and the commit history.
  `LPN_ENGINE_CHECK=1` compares every capture with the draw the game sends (3000 of 3000 matched in
  Fwha's runs; offsets 0, the camera always subtracted, 3-float positions).
- The stage's opaque casters (sites 0xa5177a, 0xa518fe) are now drawn from the captures alone:
  the packet helper writes a NOP with the capture's id where the draw packet was, and the renderer
  draws it there (`onEngineDraw`). The first draws with no PS4 draw packet. On with Native
  Renderer; `LPN_ENGINE_SHADOWS=0` turns it off. The game's state commit still runs; the
  alpha-tested class (sites 0xa51b79, 0xa51d37) needs the materials' textures and is next.

### Stage 2 (skipping the engine's per-caster state work: measured, dropped)

- Codex's reading of 0xe61ae0, 0xeabdb0 and 0xe574c0: not pure deferred emission. 0xeabdb0 issues
  synchronization when ctx+0x32c bit 2 is set and prepares shaders (0xeba8c0); 0xe61ae0 emits the
  cull state itself. The native draws also still read depth, cull and viewport from the PM4 state.
- `LPN_ENGINE_TIMING=1` times the caster loop (0xa51d90), its commits (0xe61ae0 from 0xa51100) and
  its drawIndex calls. Fwha's run, busiest 10 s: loop 24.3 ms/s, commits 1.9 ms/s, drawIndex
  3.0 ms/s (327k casters). Skipping both would save under 0.1 ms a frame, so the commits stay.

### Stage 2 (the depth and normal prepass)

- There is no G-buffer: pass 9 is a depth and normal prepass (~2,700 draws) that SSAO and the
  material passes read. Its draws come from the engine's draw-command lists: list entries
  (0x36d0's command) call 0xea3f40, which sets the vertex and index buffers in the state block,
  commits (0xe61ae0) and tail-calls drawIndex (0xeac430), so their origin is 0x3700.
- `native/prepass.cpp`: vs_normal_rigid[_pt] with ps_normal[_pt] natively. The position is
  computed with the game's operations in its order, each rounded, as the translated shader does
  (the material passes depth-test against it), including the translator's clip z mapping. The
  pixel shader's half rounding before the export is kept too. Declined with an upscaler on (the
  scene is jittered and the camera read from these draws), positions in pixels or a polygon
  offset. 240 bytes of push constants.
