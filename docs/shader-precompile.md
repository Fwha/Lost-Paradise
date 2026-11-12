# Compiling every shader before play

Goal: no shader translation, SPIR-V compile or pipeline build ever happens because a draw met
something new. Everything the game can draw is compiled at startup, behind the progress screen
(`Compile Shaders at Startup`, see gpu.md).

Today that only covers shaders seen in an earlier run, because a translation is specialised to
the draw that launched it. Removing each specialisation is a stage below. When all are gone, a
GCN program's code alone decides its SPIR-V, and the game's own shader tables list every program.

## What a translation depends on today

| Input | Where it comes from | Stage |
|---|---|---|
| Reachable code (scalar branches pruned) | CPU walk with the draw's user data | 1 |
| Image declaration: 2D/3D/Cube/Array, float/int/uint, shadow | T# type and NUM_FORMAT, sampler compare | 1 |
| Image DST_SEL swizzle (storage images) | T# | 1 |
| Buffer element format and DST_SEL (formatted loads) | V# or instruction | 1 |
| PS colour export: MRT swap, component count, int/uint type | bound render targets (CB_COLOR_INFO) | 1 |
| PS inputs (PS_INPUT_CNTL: VS output slot, flat, default) | paired VS (context registers) | 1 |
| Depth export, clip-space z | context registers | 1 |
| Fetch shader (VS calls it, vertex formats baked in) | user data pointer, per mesh | 2 |
| Pipeline state: blend, depth/stencil, topology, MRT formats | context registers | 3 |

Everything else (code bytes, RSRC1/2, PS_INPUT_ENA, SPI_SHADER_COL_FORMAT, compute local size)
is part of the shader binary's own registers and so known without a draw.

## Stages

1. **Draw-independent translation.** Keep every scalar branch as a real branch in GLSL (the
   scalar ALU is already translated; only the walk prunes). Resources become slots with a
   per-draw descriptor word in the draw data: images are declared once per dimension class they
   can take (aliased bindings) and chosen at run time; swizzles, buffer formats, MRT swaps and PS
   input mapping are read from the draw data. Output types follow SPI_SHADER_COL_FORMAT (the
   shader's own register): 16-bit int exports declare integer outputs, everything else float.
2. **Vertex pulling.** Translate the fetch shader's typed loads as a generic formatted load whose
   format comes from the draw data, so one VS translation serves every mesh layout. Done (see
   gpu.md, generic translation): the translation depends only on how many slots of four VGPRs the
   fetch fills, which the VS's own input semantics fix, so stage 4 can lay out a VS without a mesh.
3. **No per-state pipelines.** VK_EXT_shader_object (or graphics pipeline libraries) with
   blend, depth/stencil, topology and colour write state dynamic, so a shader pair needs no
   pipeline build per state. Done with fast-linked pipeline libraries (gpu.md): shader objects
   would need one descriptor layout for every shader, and the renderer pushes descriptors, of
   which a layout can push only one set; libraries with independent sets take one set per stage.
4. **Enumerate at boot.** Once the game has loaded its shader archives, read the engine's
   shader tables (PS 0x2698148, VS 0x2698168, CS 0x26981e8, counts beside them; see
   render-decomp.md) and compile every entry behind the progress screen.

### What a draw still decided, now draw data

At boot a shader has no draw, so whatever the translation took from the draw's registers rather
than the shader's own must reach the shader as draw data instead (generic translation, cache
folder `generic3`):

- **Vertex to pixel shader linkage.** SPI_PS_INPUT_CNTL says which vertex shader export each pixel
  shader input reads, or that it takes a default. The pixel shader now reads input n at location
  n; the vertex shader writes each export to every location whose input reads it, by a byte map in
  its draw data (`data::kPsInputMap`, from the draw's SPI_PS_INPUT_CNTL); a default comes in the
  pixel shader's draw data (`data::kPsDefaults`). Flat shading comes from the code
  (v_interp_mov); `lpn-gcn --boot` counts inputs the draw makes flat that the code interpolates.
- **Clip z** (PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF) is a word in the vertex shader's data
  (`data::kClipZ`).

Scalar load results start after these words (`data::smrd()`, 32 in the generic translation).

### The shader tables (measured on the user's PC, 2026-11-12)

All of the game's shaders are registered by the title menu and the tables do not change in play
(dumped at frame 1500 and in gameplay at 5500 with `LPN_DUMP_SHADER_TABLES=<flip>`). Each table
header at eboot+addr is: array start, array end, 1, capacity (0x2000), a small id, 0x80; entries are
pointers to engine objects until the first null. The names in render-decomp.md are swapped:

| eboot+ | Holds | Entries | Gnmx shader at | Register block (from that pointer) |
|---|---|---|---|---|
| 0x2698148 | vertex shaders | 818 | *(obj+0x48) | +0x08 PGM_LO, +0x0c PGM_HI, +0x10 RSRC1, +0x14 RSRC2, +0x18 VS_OUT_CONFIG, +0x1c POS_FORMAT, +0x20 PA_CL_VS_OUT_CNTL, +0x24 counts, semantics from +0x28 |
| 0x2698168 | pixel shaders | 1221 | *(obj+0x68) | +0x08 PGM_LO, +0x0c PGM_HI, +0x10 RSRC1, +0x14 RSRC2, +0x18 Z_FORMAT, +0x1c COL_FORMAT, +0x20 PS_INPUT_ENA, +0x24 PS_INPUT_ADDR, +0x28 PS_IN_CONTROL, +0x2c BARYC_CNTL, +0x30 DB_SHADER_CONTROL, +0x34 CB_SHADER_MASK, semantics from +0x38 |
| 0x2698188 | multi-stage sets (unconfirmed) | 10 (20 programs) | *(obj+0x68), more at obj+0x44..0x58 | not mapped; never run so far |
| 0x26981e8 | compute | 2 | *(obj+0x30) | +0x10 RSRC1, +0x14 RSRC2 |

2061 distinct programs in all; the user's cache (all play so far) has run 652, 648 of them from the
tables (the other 4 are engine-internal clears and copies). RSRC1, RSRC2, PS_INPUT_ENA, COL_FORMAT
and NUM_INTERP in the tables match every launch seen. A vertex shader's RSRC1 differs at launch in
its VGPR count and VGPR_COMP_CNT, which the engine raises for the fetch shader.

Each stage ships on its own and stays behind `LPN_GENERIC_SHADERS=1` until the stages after it have
been played through, so a regression can be compared against the specialised translation.

## Costs and risks

- Shaders get slower: run-time swizzles and format decode cost a few ALU ops per access, and
  unpruned branches add uniform branches. To be measured per stage (`LPN_PROFILE`).
- The specialised path has many game-specific fixes; each must keep working in the generic one.
- Stage 3 needs VK_EXT_shader_object or VK_EXT_graphics_pipeline_library and
  extended_dynamic_state3 (NVIDIA: yes). Without them the pipeline step stays as today.
