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
- **Analysis reuse.** Most draws launch a known shader with new user data (per-draw pointers),
  so analysing every launch from scratch cost ~38 ms per gameplay frame. Each analysis records
  its walk (work items and the instructions that touch scalar state); a later launch of the same
  code replays that walk with its own user data and memory (`gcn::refresh`, ~1.5 us) and keeps
  the analysis if every branch goes the same way and every resource keeps its format, else it is
  analysed again. Variants are grouped by walk (up to 8 branch patterns per shader), each with
  up to 32 resource-format shapes (one material shader with BC1 or BC3 textures), so a launch
  replays one trace and then only compares formats. Reuse is checked against the hash of the
  exact code bytes, not whole pages: a shader's last page often holds data that changes.
- **SPIR-V cache.** Compiling a shader's GLSL takes ~10 ms and a new attack effect brings dozens,
  so SPIR-V is kept in `~/.cache/lpn-host/spirv` (`LPN_SHADER_CACHE=<dir>`, or `off`), named by
  a hash of the GLSL. Only the first time an effect is seen anywhere hitches.
- **Startup compile.** With Compile Shaders at Startup (launcher setting, `precompile_shaders` in
  lpn-display.ini, `LPN_PRECOMPILE=0/1` wins) the game waits behind a progress screen while every
  saved shader (each `.gcn` in the cache, compute shaders too) is translated with today's
  translator and its SPIR-V compiled if missing, on every core, and then every pipeline in
  `pipelines.list` is built by all pipeline workers. Esc, Enter or a pad button starts the game at
  once; what is left builds in the background on a quarter of the workers, as with the setting off.
  Shaders never seen in any earlier run are still compiled when first drawn.
- **Memory.** Guest memory is imported into Vulkan (VK_EXT_external_memory_host), so shader
  buffer reads and writes hit the game's memory directly.
- **Surfaces.** Render targets and depth buffers are Vulkan images keyed by guest address; a
  texture at the same address samples that image (scaled when the T# is smaller than the
  padded target, e.g. 1600x896 inside 1664x896). Other textures are uploaded from guest
  memory with their whole mip chain (levels 0 to the T#'s LAST_LEVEL, laid out as GCN does for
  linear-aligned and 1D-thin modes), detiled if needed, and re-uploaded when their bytes change.
  Samplers take the S#'s LOD bias and anisotropy. `LPN_NO_MIPS=1` uploads level 0 only;
  `LPN_ANISO=N` forces Nx anisotropic filtering on filtered samplers. Each texture is hashed
  once per frame, in 128 KiB chunks on worker threads (~90 MB per gameplay frame).
- **Draws.** RECTLIST and QUADLIST go through a generated geometry shader; GNM's
  decompress/eliminate passes are skipped; screen-space draws (viewport transform off) are
  mapped by the vertex shader. Consecutive draws into the same targets share a render pass; a
  new pass (behind a full barrier) starts when the targets change, a draw samples one of the
  open pass's attachments, after a draw that stores to memory, or when anything else (a copy,
  clear, dispatch) is recorded. The command processor's DMA_DATA/WRITE_DATA copies between draws
  invalidate the renderer's cached guest pages.
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
- **Cross-lane and wide ops.** `ds_swizzle_b32` in quad mode becomes `subgroupQuadBroadcast`
  (GCN and Vulkan order a quad's pixels alike); the shadow-map pixel shader uses it for its
  slope-scaled depth bias. `v_mad_u64_u32` (integer division sequences in cutscene vertex
  shaders) and `v_cvt_off_f32_i4` are translated too.
- **Image stores apply DST_SEL.** A T#'s DST_SEL swizzles stores inversely, as it does loads;
  storage image views cannot swizzle, so the translator applies it. The game copies the scene
  colour with a compute shader through BGRA-swizzled T#s; without this the copy swapped red and
  blue, and the anti-aliasing pass that reads it turned every edge (outlines, ink lines) blue.
- **Constants inside shader code.** Some shaders build a V# from `s_getpc_b64` plus an offset to
  read a table stored after their code (the AO upsample's bilateral weights). The analysis
  tracks `s_add_u32`/`s_addc_u32` so that address stays known; when it was lost, the AO term read
  zero and every environment material in Shin's Castle shaded black.

Known gaps: tile mode 14 (2D thin) textures are read as linear (in gameplay only render targets,
which are sampled from their images, use it); draws into linear targets are not copied back;
indirect draws are skipped; cube arrays, MSAA and some formats are not handled; LDS/GDS memory
is not emulated, and `ds_swizzle_b32` only in its quad mode.

## Next

2D-tiled detiling, a swapchain, and batching draws per render pass.

## DLSS (DLAA)

`scripts/fetch-dlss.sh` puts NVIDIA's DLSS SDK in `third_party/dlss`; cmake then links NGX and
copies the DLSS runtime beside `lpn-host`. PC Settings > DLSS (Off/DLAA, `dlss` in
lpn-display.ini) replaces the game's SMAA (three full-screen draws, found by a hash of their
pixel shaders' code) with DLAA (`gpu/dlss.cpp`, `dlssSmaa` in the renderer):

- Depth: the depth buffer bound to the SMAA blend (the scene's).
- Motion vectors: a compute pass reprojects that depth with this and last frame's camera. The
  scene vertex shaders read it from the constant buffer whose V# s[12:13] points at: projection
  rows at 0x0, view rotation rows at 0x40/0x50/0x60, camera position at 0x220 (positions are
  camera-relative). Only camera motion: moving characters rely on DLSS.
- Jitter: every draw with the scene depth bound before SMAA (full-screen lighting passes too, or
  shadows flicker) gets a Halton sub-pixel viewport offset, passed to DLSS as is. So do 3D draws
  after SMAA that depth-test against it (Ken's aura redraws his body additively; unjittered it
  passed or failed its depth test by frame and flickered).
- `[dlss] last 300 frames` lines count history resets; `LPN_DLSS_NO_JITTER=1`,
  `LPN_DLSS_JITTER_SIGN=x,y` for debugging. Upscaling (render the scene smaller) is not done.
