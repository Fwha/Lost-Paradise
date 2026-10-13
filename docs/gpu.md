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

## Next

GCN → SPIR-V for these 63 instructions, a Vulkan backend for RECTLIST/full-screen draws,
texture detiling (tile modes 10 and 14), then the composite into the display buffer.
