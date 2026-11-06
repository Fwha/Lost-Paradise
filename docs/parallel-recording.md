# Multi-threaded recording (design)

At the city overlook the frame has ~9,000 draws and the single recorder thread (`lpn-record`)
takes ~15 ms, so it caps fps at ~65 while the GPU needs ~10 ms. The recorder's cost is spread
thin (cache misses on data the analysis workers wrote, prefetchAhead, switchDraw's slot search,
descriptor pushes, CmdChunk allocation), so no single fix halves it. Recording on several
threads does.

## Shape

1. **Front stage (sequential, cheap).** Walks the frame's draws in order and makes every decision
   that depends on order: render pass boundaries, barriers and layout transitions, feedback
   (sampling a target written earlier in the frame), VRAM copy and texture upload scheduling.
   It cuts the frame into segments: whole render passes, and large passes split into runs of
   draws.
2. **Segment recorders (parallel).** Each segment is recorded into its own secondary command
   buffer by one of N recorder threads. A split pass uses dynamic rendering with
   `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` and inheritance info, so the pieces of
   one pass run in order inside the same rendering scope.
3. **Submit (sequential).** The primary buffer executes the secondaries in frame order,
   with the front stage's barriers between them.

## Per-thread state

Moves from the renderer into a recorder context: `fast_`, `fastSlots_`, `prevDraw_`, `bound_`,
`pass_`, `drawWrites_`, its own sub-range of the upload ring, its own CmdChunk pool and
profiler counters. Each segment starts from a known state (no carried bindings), costing one
full bind at its first draw.

## Shared caches

Already locked: `vram_` (cacheMutex_), `layouts_` (layoutMutex_), pipelines/`built_`
(buildMutex_), `shaders_` (shadersMutex_). Need locking or a read-mostly scheme: surfaces,
images/`imageMemos_`, imports, samplers, views, `warned_`. Anything that creates a resource
mid-segment (a VRAM copy, a texture upload) must be done by the front stage, or queued and
recorded before the segment's secondary runs.

## Risk and the measure first

If one pass holds most of the draws, splitting by pass gives nothing; the pass must be split
into pieces. `LPN_PROFILE=1` now prints `passes by draws` (counts and draws for passes of
1–15, 16–127, 128–1023 and 1024+ draws, and the largest pass) to size this before building.

## Stages

1. Pass-size counter (done).
2. Front stage split out of `onDraw`, still recording on one thread (no fps change; checks the
   split is right).
3. Per-thread recorder contexts and secondaries, behind `LPN_RECORD_THREADS=N`.
4. Intra-pass splitting, if the counter shows big passes.
