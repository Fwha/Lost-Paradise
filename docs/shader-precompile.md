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
   format comes from the draw data, so one VS translation serves every mesh layout.
3. **No per-state pipelines.** VK_EXT_shader_object (or graphics pipeline libraries) with
   blend, depth/stencil, topology and colour write state dynamic, so a shader pair needs no
   pipeline build per state.
4. **Enumerate at boot.** Once the game has loaded its shader archives, read the engine's
   shader tables (PS 0x2698148, VS 0x2698168, CS 0x26981e8, counts beside them; see
   render-decomp.md) and compile every entry behind the progress screen.

Each stage ships on its own and stays behind `LPN_GENERIC_SHADERS=1` until the stages after it have
been played through, so a regression can be compared against the specialised translation.

## Costs and risks

- Shaders get slower: run-time swizzles and format decode cost a few ALU ops per access, and
  unpruned branches add uniform branches. To be measured per stage (`LPN_PROFILE`).
- The specialised path has many game-specific fixes; each must keep working in the generic one.
- Stage 3 needs VK_EXT_shader_object or VK_EXT_graphics_pipeline_library and
  extended_dynamic_state3 (NVIDIA: yes). Without them the pipeline step stays as today.
