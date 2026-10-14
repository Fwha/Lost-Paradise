# GPU notes

What the game sends the PS4 GPU, as seen by `lpn-host --dump-frame N` (frame 600, the boot
screen loop). Addresses are guest addresses and change between runs.

## Command stream

- Register writes are emitted inline by the game's statically linked Gnmx; draws, dispatches
  and shader binds go through `sceGnm*` helpers, which `hle_gnm_packets.cpp` turns into the
  same PM4 the console driver writes.
- Per frame: ~8,300 packets, almost all `SET_CONTEXT_REG`; 43 draws (all `DRAW_INDEX_AUTO`,
  mostly 3-vertex RECTLIST full-screen passes) and 15 compute dispatches.
- Rendering is deferred: lighting and post passes at an internal 1664x896, shadow-like passes
  into 1024x1024 targets with depth, a downsample chain (1024x512 to 128x64), and a final
  composite into the 1920x1080 display buffer (tile mode 10) followed by a UI quad.
- Resource descriptors (T#, S#, V#) are passed directly in shader user-data registers.

## Shaders

- 31 distinct shaders in the frame (VS, PS, CS). Code is followed by an `OrbShdr` info
  block, which bounds the dump.
- Vertex shaders call a fetch subroutine with `s_swappc_b64`.
- `lpn-gcn --stats` over them: 63 distinct instructions, all decoded. The bulk is float ALU
  (`v_mac/v_mul/v_add/v_mad_f32`), interpolation, `image_sample[_c|_o]`, scalar buffer loads,
  and a few compares/branches. One compute shader uses `image_store` and
  `buffer_store_format_xyzw`.

## Renderer

`src/gcn` (translator) and `src/host/gpu/renderer.cpp` (Vulkan).

- **Shaders.** At each draw the scalar half of the program is evaluated on the CPU with the
  draw's user SGPRs (`gcn::analyze`): that finds the reachable code (including the fetch
  shader a VS calls), the value of every scalar memory load, and every T#/S#/V#. The vector
  half becomes GLSL (`gcn::translate`), one invocation per lane, with exec/vcc kept as the
  lane's own bit and branches as a loop over a switch on the PC. Scalar load results and
  buffer offsets reach the shader in a per-draw storage buffer; translations are cached by
  code and resource shape.
- **Memory.** Guest memory is imported into Vulkan (VK_EXT_external_memory_host), so shader
  buffer reads and writes hit the game's memory directly.
- **Surfaces.** Render targets and depth buffers are Vulkan images keyed by guest address; a
  texture at the same address samples that image (scaled when the T# is smaller than the
  padded target, e.g. 1600x896 inside 1664x896). Other textures are uploaded from guest
  memory, detiled if needed, and re-uploaded when their bytes change.
- **Draws.** RECTLIST and QUADLIST go through a generated geometry shader; GNM's
  decompress/eliminate passes are skipped; screen-space draws (viewport transform off) are
  mapped by the vertex shader. Each draw is its own render pass with a full barrier, and the
  queue is drained at the end of each submission (correct first, fast later).
- **Display.** At flip the display buffer's image is read back and shown by the existing SDL
  window.

- **Fast clears.** The game clears depth buffers and render targets by filling their HTILE or
  CMASK from a compute shader. Draws remember each surface's metadata address and clear value
  (DB_DEPTH_CLEAR, CB_COLOR_CLEAR_WORD0/1); a later fill of that address clears the surface.
- **CPU readbacks.** A compute shader storing into a linear (tile mode 8) image is followed by a
  copy of the image back to guest memory: the game reads the measured scene luminance this way
  for auto exposure. 1D images are treated as 2D images of one row, in shaders and views.

- **Cube maps.** `v_cubeid/sc/tc/ma` are translated, and a cube sample's (s, t, face) coordinates
  are turned back into a direction for `samplerCube`. Character shaders use them.
- **Constants inside shader code.** Some shaders build a V# from `s_getpc_b64` plus an offset to
  read a table stored after their code (the AO upsample's bilateral weights). The analysis
  tracks `s_add_u32`/`s_addc_u32` so that address stays known; when it was lost, the AO term read
  zero and every environment material in Shin's Castle shaded black.

Known gaps: tile mode 14 (2D thin) textures are read as linear (in gameplay only render targets,
which are sampled from their images, use it); draws into linear targets are not copied back;
indirect draws are skipped; cube arrays, MSAA and some formats are not handled;
`ds_swizzle_b32`, `v_mad_u64_u32` and `v_cvt_off_f32_i4` are not translated yet.

## Next

2D-tiled detiling, a swapchain, and batching draws per render pass.
