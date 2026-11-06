# Render path decomp

Goal: lift FPS at heavy scenes by decompiling the game's draw-building code and replacing it,
so the host gets each draw's state directly instead of parsing PM4 and replaying shader scalar
code to find resources. The rest of the game (60,499 functions) already runs natively and is
not part of this.

## Layers in the eboot

| Layer | Where | Size |
|---|---|---|
| Gnm command buffer (static lib): wraps the 25 libSceGnmDriver packet imports, writes PM4 | 0xf9fd30-0xfa6e20 | 204 functions, 27 KB |
| Engine graphics: direct callers of that layer (draw wrappers, per-material draw functions) | 0xe566e0-0xf56290 | 127 functions, 104 KB |
| One more level up (scene/mesh submission, mixed with game code) | spread | 276 functions, 778 KB |

Key entries:
- 0xfa09b0 drawIndexAuto (only caller of sceGnmDrawIndexAuto), 0xfa0a10 drawIndex,
  0xfa0a80 dispatchDirect.
- 0xeac430: engine drawIndex wrapper, called from game code all over.
- 0xeb3a00, 0xeb4420, 0xeb4bb0, 0xeb6cc0, 0xeb7530, 0xeb7c20, 0xeb8270, 0xeb89f0: large
  (1.6-2.6 KB) functions that each end in drawIndexAuto: likely per-shader-type draw paths
  that fill descriptor tables. First candidates to decompile.

## First run (2026-11-06, LPN_DRAW_SITES)

Peak 8,500-9,100 draws a frame, nearly all through drawIndex (dispatches ~16/frame).
At >= 5,000 draws:

| Share | Path |
|---|---|
| ~60% | draw-command objects (0x36d0, run through virtual calls 0x3710/0xa70/0xbb0, nested lists) -> 0xea3f40 (440 B) |
| ~20% | 0xa51100 (3.2 KB, two draw sites), from 0xa51d90 <- 0x55fb20 |
| ~4% | 0x4cff00 (4 KB) |
| ~6% | 0x5a7ec0 / 0x5a7670 (2.1-2.3 KB) |

Every one of these goes through the same three engine functions:
0xeabc90 (108 B, one Gnm call), **0xe61ae0 (5.1 KB: binds shaders and resources)**, then
0xeac430 (draw, tail-calls 0xfa0a10). So 0xe61ae0 and its callees are the decomp target:
replacing that one binder covers almost every draw at the heavy spots.

## Milestones

1. **Find the hot paths (this branch).** `LPN_DRAW_SITES=1` logs per-second draw counts per
   call chain (4 return addresses up). Run it at the city overlook, crowds and the steps.
2. **Decompile 0xe61ae0 and its callees** (was: the top 1-3 chains) in Ghidra (`ghidra/` scripts load the eboot): learn how
   they lay out descriptor tables (the per-draw s2/s12 pointers) and what state they set.
3. **Replace them**: hook those functions, still emit the game's PM4 (keeps everything else
   working), and pass a side record (shaders, buffers, textures, samplers) to the renderer so
   it skips analysis/replay for those draws. Check against the existing path with an env
   switch and frame captures.
4. **Batch repeats**: crowds draw the same mesh thousands of times; with the draw state known
   up front, merge them into instanced Vulkan draws.

Analysis scripts (capstone over eboot.bin + analysis/functions.csv) live outside the repo;
never commit game files.

## Decompiling without full analysis

Full Ghidra auto-analysis of 34 MB takes hours; for single functions it isn't needed.
1. Flatten the eboot into a raw image (code at 0, relro at 0x2294000, data at 0x2398000).
2. `analyzeHeadless <proj> lpn -import image.bin -loader BinaryLoader -processor x86:LE:64:default
   -cspec gcc -noanalysis -scriptPath ghidra -postScript DecompileList.java <dir> e61ae0,eac430`
   where `<dir>` holds `funcs.txt` (`<hex start> <size>` per line, from functions.csv),
   `names.txt` (PLT stub names) and optionally `labels.txt` (our names). Output: `<dir>/out.c`.
   Later runs with `-process image.bin` instead of `-import` reuse the project.

## 0xe61ae0 first pass

It is the engine's dirty-state flush before a draw: the first qword of its context is a dirty
mask (bits 0x10 PS, 0x20 VS, 0x40, 0x100, 0x200, 0xc...); shaders come from tables at
0x2698148 / 0x2698168 / 0x26981e8 indexed by 16-bit ids at ctx+0xe80/0xe82, and it writes into
a Gnmx-like state block at `*(ctx+0x18)+800`. Because it only re-emits what changed, hooking it
gives us exactly the per-draw deltas, which is what the renderer currently reconstructs.
