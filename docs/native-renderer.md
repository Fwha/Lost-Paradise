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
- `LPN_NATIVE_COMPARE=prepass` (Fwha's runs): depth exact for both variants; the opaque variant's
  normals exact after the decode factor was corrected to the shader's 0x40008102 (255/127). The
  _pt variant's normals stay one 8-bit step off on a share of pixels, with the varyings, the
  encoding's rounding and the decode all as the game's: a known, invisible difference, not chased
  further.

### Stage 2 (the material pass, first pair)

- The forward material pass (~3,350 draws): two vertex shaders (`r_b1dst_a(vr)2d`, `r_b1dst`) cover
  ~2,500 draws across about ten pixel shader families, each full forward lighting (~850
  instructions, 13 textures).
- `native/material.cpp`: `r_b1dst_a(vr)2d` with `r_o1dst_m(vr)2ds` (~570 draws). Constants come from
  the shaders' own loads by their offsets, uploaded per draw (`Gpu::upload`). The position uses the
  prepass's operations. Codex's reading is in `docs/shaders/`.
- `LPN_NATIVE_COMPARE=material` (Fwha's run): ~57,000 draws compared, no pixel more than one 8-bit
  step off in either target, depth exact, none declined. Looked right in play.
- Three more pixel shaders of the family as variants of the same port (`r_o1dst_m(vr)2dst`,
  `r_o1dst_b(vr)2dst`, `r_o1dst_b(vr)2ds`: a detail normal map, and detail textures blended by
  their alpha), ~1,380 of the material pass's draws in all. Their names carry shader_names' " +1".
  Compare run: all four within one 8-bit step, depth exact.
- The `r_b1dst` family as `SIMPLE_MATERIAL` variants: `r_o1dst`, `r_o1dut` (specular map's green
  alone), `r_p1dut` (alpha-tested). The material pass now covers ~2,100 of its ~3,350 draws.
  Compare run: within one step, depth exact; once, a few thousand pixels of the edge mask (target 1)
  differed by up to 65/255, since that mask is a threshold that rounding can flip.

### Native draws at the old path's speed

Fwha measured ~45 fps with Native Renderer on against ~88 off at a busy town spot. With
`LPN_PROFILE` (now with a "native draws" line and native pass counts) and a perf profile:

- Native draws were recorded on the main recorder only, each draining the recorder threads: they
  now record in the threads' segments, engine-captured shadow casters too (`DrawCall::kEngine`).
- Native and translated draws on the same targets restarted the pass with a full barrier (~1,100
  passes a frame against ~115): they now share the open pass, a native draw counts only what it
  writes (a depth buffer it only tests is not written), and barriers follow the translated path's
  rules.
- One lock around every native draw: each pass now guards only its own state.
- Per draw, ~60 map lookups for the material's constants (now resolved once per shader pair) and
  two copied load maps in the analysis (now built once per cached program and shared).

Result: ~85–95 fps there with it on, GPU time ~10 ms against ~12 off.

### Reflective materials, and posters behind the walls

- The `[ref]` family: `r_o1dst[ref]`, `r_o1dut[ref]`, `r_o1du[ref]`, `r_p1dut[ref]`, with their
  environment cube map at its own binding and the `r_b1dst[ref]` / `r_b1dut[ref]` vertex shaders.
- Indoors, posters on the walls went black with Native Renderer on: they were drawn behind the
  walls. The simple and `[ref]` vertex shaders load the camera three times (for the position, for
  fog, for the reflection), each in its own branch, and the last listed load won. With vertex fog
  off, the fog branch's load is not taken and reads 0, so the camera sat at the origin. A load that
  reads 0 no longer replaces a value another load of the same draw set (the raster-to-uv scale of
  the pixel shaders has the same pattern, in the edge mask's branch).
- Compare mode now logs, per role (each on its own line, not roles past 6 together) and target,
  the first pixel differing by 8/255 or more: both values and the draw's constants, state and
  vertex layout. In compare mode the material pass also checks its cached load offsets.
- Native draws bind the index buffer as the translated draw does (from the buffer's start, with
  the first index in the draw), so both read the same copy of it.

### Characters: skin and the plain character material

- `native/character.cpp`, a pass of its own: characters light differently from the stage
  (ambient polynomial, Fresnel lobes, skin ramps, an extra lobe, material occlusion and grading).
  `s_b1dzt[skin]` + `sm_o1dzt[skin]` and `s_b1dst` + `sm_o1dzt` (~110 draws a frame). Both
  vertex shaders skin four bones from the palette in their own operation order, and move
  vertices by the deformation planes. Draws with more than one plane are declined, because the
  analysis keeps only one entry's loads. Codex's reading is in `docs/shaders/sm_o1dzt[skin].md`,
  and Codex drafted the port from it.
- Compare run (`LPN_NATIVE_COMPARE=character`), ~25,000 draws: colour within 1/255 and depth
  exact. A few thousand pixels of the edge mask (target 1) differ by up to ~115/255: as with the
  stage, it is a threshold and a sampled-offset test that rounding can move. Draws of
  `sm_o1dzt` with other vertex shaders (~4%) are still left to the translated path.

### Characters: hair

- The character pass draws hair: `s_b1dt[hair]` with `sm_b1ht[hair][offset][dedit]` and
  `sm_b1dzt[hair][cmask]` (~100 draws a frame). Two anisotropic lookup lobes take the place of
  the reflected-vector specular. The hair vertex shader has its own constant bindings and passes
  the specular shift as interpolants. The alpha test is done with demote.
- Every hair draw blends or marks stencil. Native draws now carry their targets' blend state and
  blend constants, as the renderer converts them for the translated pipeline
  (`native::Draw::blendState`, `blendConstants`). The character pass draws with those, and with the
  translated draw's stencil state.
- Compare run: `sm_b1ht` within 1/255. `sm_b1dzt[hair][cmask]` is within 4/255 on a few thousand
  pixels of a frame's blended layers.

### Alpha-tested characters, the reflective two-uv blend

- Character pass: `sm_p1dzt` (~38 draws a frame), which is `sm_o1dzt` with an alpha test. The
  test inserts 0x1c bytes at 0x11b0, so later PCs shift, and two loads next to it are reordered.
- Material pass: `r_o1dut_a(vr)2d[ref]` with `r_o1dut_m(vr)2d[ref]` (~70 draws a frame, role 15).
  It keeps the primary normal, uses specular green and its red for the reflection, and applies the
  detail texture to the diffuse only (`DIFFUSE_DETAIL_ONLY`, in the shader's own operation order).
  Codex drafted it against `r_o1dst_m(vr)2ds` and `r_o1dut[ref]`.
- Compare runs: both within 1/255, depth exact.

### Alpha-tested hair, reflective characters, eyes, a plain stage material

- Character pass, roles 6 to 8:
  - `sm_p1dzt[hair][cmask]`
  - `s_o1dzt[ref]` (an environment cube, sampled at (-x, y, z))
  - `s_o1dzt[eye][aref]` (its own skinning vertex shader with the eye's view terms)

  The cmask hair also multiplies its squared base colour by PSV[0x30] 0x110. Role 4 was missing
  that too, which made hair ~12% too bright where the tint was 0.78.
- Material pass, role 16: `r_b1d` with `r_o1d[ag]`, a material with no normal map or specular.
- Compare mode: each role now has its own draw budget, 8 a frame, so roles drawn late in the
  frame are compared too. There is room for 24 roles.
- Compare runs:
  - Eyes are exact.
  - Hair is within 2/255.
  - Role 16 is within 1/255.
  - `s_o1dzt[ref]` and the material role 15 are within 1/255, except on a few bright
    reflection pixels: up to ~40/255 and ~12/255, from the cube's sampling.
- Codex drafted these. Its full-file drafts are merged keeping the existing code. One draft
  rewrote a verified table, wrongly, and was caught by checking that the older variants'
  preprocessed GLSL is unchanged.

### Seven more scene shaders; the stage material blends

- Census (`lpn-captures/capture-5/passmap-7997.txt`, the rt1 passes): before this stage, native
  passes drew 2,947 of a town frame's 3,354 material draws.
- Material roles 17 to 22:
  - `r_b1dst`'s own pixel shader
  - `r_o1dst_b(vr)2dst[nte]`
  - `r_o1dut_b(vr)2dut[ref]`
  - `r_o1dut_m(vr)2du[ref]`
  - `r_o1d`
  - `r_o1dut_a(vr)2d`
- Character role 9: `sm_p1ht[hair][offset][dedit]`.
- The material pass blends and uses stencil as the translated pipeline does, as the character
  pass already did.
- Two pixel shaders read a register after a later fetch had replaced its contents, and the port
  had kept the older value:
  - `r_o1dst[ref]` (role 11): 0x870 loads the specular texture's alpha over the point light's red.
    The reflection is weighted by that alpha and multiplied by the specular colour. With no point
    lights, the port had drawn no reflection at all, up to 138/255 off.
  - `r_o1dut_a(vr)2d` (role 22): 0x818 loads the detail alpha over the indirect light. The
    added term is detail alpha times the squared detail.
- `r_o1dut_b(vr)2dut[ref]` (role 19) is still darker than the game, up to ~60/255. It is drawn only
  in compare mode until it matches.
- Compare runs: every other role within 1 to 5/255, except for a few cube-reflection highlight
  pixels (roles 15 and 20) and the usual edge-mask thresholds in target 1.
