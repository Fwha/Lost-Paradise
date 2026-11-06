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

## What was built (stage 2, LPN_RECORD_THREADS=N)

The overlook run (2026-11-06) showed 74 passes a frame, ~15 of them holding 95% of the draws, the
largest 1,000 (the 1,000-draw flush cuts them). So runs are cut by count, not by pass:

- The main recorder (`lpn-record`) only collects draws into segments of up to
  `LPN_RECORD_SEGMENT` (128) draws, with a copy of the context registers whenever they change
  (`GpuRegs::ctxSerial`), and queues each segment's execution in its batch in order.
- N threads (`lpn-record-w`) each record a segment into a secondary command buffer of its own
  (own pool and 4 MB draw-data ring), from a fresh `Rec`: no pass, nothing bound, no fast-draw
  states, a full barrier before its first pass. Vulkan calls are made at once on that thread.
- The command thread executes a segment (`vkCmdExecuteCommands`) once it is recorded.
- Shared caches stay under `m_`: a thread holds it for a whole full draw, and around a fast or
  switch draw's lookups (images, index buffer, moved buffers). See `WLock`.
- Uploads and copies made on a thread (texture uploads, VRAM copies, a new image's first layout
  and clear) go into `pro_`, secondaries executed before the next segment the command thread
  executes, so no segment recorded earlier misses them. See `Prologue`.
- What a thread frees waits for the next flush of the main batch (every segment recorded by
  then is in it or an earlier one).
- Dispatches, DMA, memory writes, SMAA under DLSS, frame ends and traced frames wait for every
  segment so far (`drainSegments`) and are recorded in order on the main recorder as before.
- The main recorder stays at most ~2 batches ahead of the threads, and gives up `m_` while it
  waits for a batch they may still have to record (`begin`, `waitIdle`).

`LPN_PROFILE=1` adds a `[prof] ... recorder threads` line: segments, the threads' recording time
summed, waits for them (drains) and waits to let them catch up, per frame.

### Drains (2026-11-06)

With fast draws sharing m_ the overlook reached 58 fps (from ~55 single-threaded), but the main
recorder still waited ~12 ms a frame in 20 drains. GPU-made DMA copies no longer drain: the copy is
queued after the segments before it (same GPU order) and nothing those segments read on the CPU
changes; CPU-made copies still drain. The profile line now splits drains by cause.

### Lock (2026-11-06)

With 4, 6 and 8 threads the overlook stayed at ~60 fps while waits for m_ grew 17, 35, 51 ms
(summed): the exclusive hold of each full draw (~1,600 a frame) serialised the threads. A thread's
full draw now looks the caches up holding m_ shared; any lookup that would change a shared cache
(a new image, view, pipeline, import, VRAM copy or check, a surface marked written, ...) flags it
(`sharedMiss`) and the draw starts over exclusively. Fast and switch draws whose moved buffers hit
such a case bind them again exclusively. The profile line counts both.
