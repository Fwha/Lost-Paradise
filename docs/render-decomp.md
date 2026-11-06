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

## The draw commit (0xeac430 -> 0xeabdb0)

0xeac430 (engine drawIndex) calls 0xeabdb0 (preDraw), then 0xe574c0, then sceGnmDrawIndex via
the 0xfa0a10 wrapper. 0xeabdb0 is a Gnmx ConstantUpdateEngine-style commit over a state block
at `ctx+800` (`param_1[200]`). For each active stage (mask at +0xdc) it walks dirty bitmasks and
calls one setter per changed slot:

| Slots | Array (per stage) | Stride | Setter |
|---|---|---|---|
| textures / buffers (T#, V#) | +0x1d38 + stage*0x210 | 0x20 | 0xe583a0 |
| constant buffers | +0x6c8 + stage*0x220 | 8 | 0xe584d0 / 0xe58680 |
| samplers (S#) | +0x1378 + stage*0x90 | 8 | 0xe58830 |
| RW buffers / textures | +0x16d8 + stage*0x110 | 0x10 | 0xe587c0 / 0xe58610 |

Vertex streams come from +0x288 (0x20 per stream: base, offset, count, stride). Samplers are
built and cached by 0xe9cce0 (key -> S#, binary search under a lock).

So at 0xeac430 the state block already holds every resource of the draw, by slot. Plan for
milestone 3: hook 0xeac430 and the auto-index draw, snapshot the changed slots (the dirty masks
tell us which) into a side record tagged with the draw's PM4 position, and have the renderer map
slots to shader inputs with the shader binary's own input-usage table (static, once per shader)
instead of replaying scalar code per draw.

## Reassessment after reading the renderer (2026-11-06)

Since PR #47 the per-draw analysis/replay runs on worker threads and is off the critical path;
the limit at heavy spots is the single recorder thread (90-93% busy) and then GPU time. Most heavy
draws already take `fastDraw`, whose per-draw cost is a data upload, a push-descriptor write and
a vkCmdDrawIndexed. So the state-block hook would mostly save worker time, not frame time.

The bigger lever is milestone 4: merge runs of fast draws that differ only in draw data into one
multi-draw (vkCmdDrawIndexedIndirect, shaders indexing their draw data by gl_DrawID). The engine
side supports this: ~60% of heavy draws come from one draw-command list through the same binder.
`LPN_PROFILE=1` now reports `mergeable fast draws N in M runs` per frame to size it first.

## Second run (2026-11-06, LPN_PROFILE)

At the spots tested (~2,900 draws, 36-59 fps) the recorder takes ~4 ms and the GPU ~5 ms, but the
recorder waits 8-10 ms a frame for the analysis workers (analyze ~28 ms summed over 3 workers).
So there the replay is the limit after all, and the state-block hook (milestone 3) targets it.
Merging is not worth it: only 1-9 fast draws a frame change draw data alone (~600 rebind a
buffer, ~700 switch shaders).

## Slot setters (0xe583a0, 0xe584d0/0xe58680, 0xe58830)

Gnmx-style input-usage mapping. Per stage, `*(ctx+0x78+stage*0x40)` points at the bound shader's
slot table: +0x1c constant buffers, +0x9c, +0xbc samplers, +0xdc resources, one u16 per slot:
- 0xffff: the shader doesn't use the slot (nothing written).
- bit 15 set: a user SGPR (`& 0x3fff`); written straight into the PM4 as SET_SH_REG.
- otherwise: a dword offset into the stage's flat table at `*(ctx+0x68) + stage*0x1800`, which
  is copied to GPU memory and its address passed in a user SGPR (the per-draw s2/s12 pointers).
Each setter also marks the stage dirty in `ctx+0x23c`.

So the descriptor for every resource a shader reads sits either in user data or at a fixed
offset in that table, and the offset only depends on the shader. That gives two designs:
1. Per shader, learn once (from the first full analysis) which table offset or user SGPR each
   resource comes from; later draws read those descriptors directly instead of replaying.
   Needs no hook, only that the trace's control flow doesn't depend on the data.
2. Hook 0xeabdb0 and capture the flat tables as the engine writes them (dirty stages only).
A perf capture at an analysis-bound spot decides: if replay time is the interpreter, (1) wins;
if it is the memory reads (PageCache), (2) does.

## Tried: skipping dead replay steps (reverted)

A per-shader plan that dropped trace steps whose only effect (marking SGPRs unknown) nothing
reads again kept 98-100% of steps on the owner's run (~60 steps per replay, up to 6,200 replays a
frame): every step in these traces is live. Check mode also logged 8 replays of shader
0x53c3fd200 whose two back-to-back replays of the same trace gave different output, which
points at guest memory changing while it is being read (the game writing ahead into its rings).

## Worker count (2026-11-06)

LPN_ANALYSIS_WORKERS=6 on the owner's i7-12700K: mid-size spots (1,500-3,700 draws) went from
36-59 to 90-140 fps with analysis-wait ~1 ms; the default is now 6 on 16+ hardware threads. The
~8,400-draw overlook stays ~56 fps with no analysis wait: the recorder (~16 ms) is its limit.

## Fast replay check (2026-11-06)

LPN_FAST_REPLAY=check past the overlook: 37-54% of trace steps run directly; 4 differences in
~85 s, all shader 0x53c3fd200, the same one that differed when the plan ran every step (so not
the fast path; most likely its memory changing between the two reads). Left off by default:
with 6 workers analysis no longer limits fps, so its gain is spare CPU.
