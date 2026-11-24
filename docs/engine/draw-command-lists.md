# The engine's draw-command lists: a reading

Codex's reading of the eboot code that builds and runs the scene's draw-command lists (2026-11-24),
from disassembly. Guesses are marked as such.

The useful capture boundary is an **ordered program of state changes and draws**, not an array of self-contained mesh/material/object records. `0x3710`, `0xa70`, and `0xbb0` execute that program; `0x36d0` is its indexed-mesh draw command (`0x3730–0x3742`, `0xa90–0xaa2`, `0xbd0–0xbe2`, `0x36d8–0x36fb`).
Two corrections matter:
- **The second virtual slot is not generally a destructor.** `0x1b80` calls `[vptr+8]` with the proposed next draw’s arguments and tests its Boolean result (`0x1bce–0x1be8`). `0x1b40` implements the matching adjacent-draw merge operation (`0x1b40–0x1b75`).
- **The supplied files do not establish a complete shader/material decoder.** In particular, `0x3830` forwards to the absent `0x3840`; other state-command invoke bodies and the vtable data are also absent. Their constructors establish payload layouts, but not all payload semantics.
Addresses below are eboot-relative. “Proven” means visible in the supplied instructions; interpretations and missing links are identified explicitly.
**1. Command objects and invocation**
Use these distinct names:
| Name | Meaning | Evidence |
|---|---|---|
| `B` | Mutable command builder | Pending count `B+0x260`, pointer array `B+0x980`; `0xb0–0xbf` |
| `Q` | Individual polymorphic command | Invoke through `(*Q->vptr)(Q,C)`; `0x3730–0x3739` |
| `C` | Execution context passed unchanged to children | Saved from `rsi` at `0x371d`, forwarded at `0x3733` |
| `G` | Lower Gnm context, `*(C+0x18)` | `0xea3f74`, `0xea40d8` |
| `S` | Gnm state block, `*(G+0x320)` | `0xea3fe8`, `0xea4084` |
| `S′` | State pointer obtained directly from `*(C+0x38)` | `0x3774`, `0x37d4` |
`S′ == S` is a plausible interpretation, but the supplied instructions do not prove that alias. Also, **builder `B+0x38` and execution-context `C+0x38` have different roles**; do not decode them using one structure.
Command constructors store the address point `vtable_base+0x10` at `Q+0`. Thus the draw constructor stores `0x2294010`, not `0x2294000` (`0x1c05–0x1c1a`). The associations with the supplied invokes below use your stated vtable mapping; the actual vtable bytes are not included.
**Indexed mesh command: invoke `0x36d0`, address point `0x2294010`**
Allocated as `0x30` bytes by `0x1b80` (`0x1bfe–0x1c30`):
| Offset | Type | Meaning | Evidence |
|---|---|---|---|
| `+0x00` | pointer | Virtual address point | `0x1c16–0x1c1a` |
| `+0x08` | pointer | Mesh/buffer-binding object `M` | `0x1c1d`; loaded at `0x36e0` |
| `+0x10` | `u32` | Mode passed to `0xeabc90` when changed | `0x1c21`, `0x36d8–0x36f3`, `0xea3f62–0xea3f78` |
| `+0x14` | `u32` | Index count | `0x1c25`, `0x36f3`, `0xea40df` |
| `+0x18` | `u32` | First index | `0x1c28`, `0x36dc`, `0xea40e2` |
| `+0x1c` | `u32` | Base vertex | `0x1c2c`, `0x36f7`, `0xea40dc` |
| `+0x20` | `u32` | Additional captured argument; semantics unresolved | `0x1c30`, `0x36e4–0x36ed` |
`0x36d0` calls:
```cpp
ea3f40(M, C, mode, indexCount, firstIndex, baseVertex, extra);
```
The packed qword loads merely load adjacent `u32` fields efficiently (`0x36d8–0x36f7`). **The shown `0xea3f40` does not consume the seventh, stack argument.** Do not assign `Q+0x20` an instancing or draw-count meaning from this listing.
The mode is plausibly a primitive/topology selector, but its actual enumeration requires `0xeabc90`, which is absent (`0xea3f62–0xea3f78`).
**Mesh and backing-buffer reads**
`0xea3f40` reads the following, rather than finding raw GPU addresses directly in `Q`:
| Source | Observed role | Evidence |
|---|---|---|
| `M+0x10` | Vertex-binding identity/cache key | `0xea3f7d–0xea3f98` |
| `M+0x18` | Vertex backing-buffer descriptor pointer `V` | `0xea3faa` |
| `M+0x2c` | Vertex stride | `0xea3fb2`, `0xea4009–0xea403c` |
| `M+0x70` | Index-binding object `I` | `0xea3f5e` |
| `I+0x10` | Index-binding identity/cache key | `0xea4050–0xea4067` |
| `I+0x18` | Index backing-buffer descriptor pointer `D` | `0xea4079` |
Both backing-buffer paths start with `descriptor+0` as the address. When `(descriptor[+0x28] & 3)==2`, they may substitute the current-context backing allocation from `*(descriptor+0x20)+0x10+8*G[+0x328]`. A null replacement is rejected; for slot zero, backing-table flag bit zero can prevent substitution (`0xea3fb5–0xea3fe1`, `0xea4081–0xea40b3`).
Consequently, **saving `M` alone is insufficient for an asynchronous renderer**: the selected backing allocation depends on execution-time `G+0x328` and the backing table.
Vertex state written when the binding cache requires it:
- `S+0x288`: resolved vertex base; `S+0x290=0` (`0xea3ff4–0xea3ffb`).
- `S+0x294`: low dword of `V+8`, divided by stride, or zero for zero stride (`0xea3fe4–0xea4013`).
- `S+0x298`: preserves its high 12 bits, inserts stride into bits 4–19, and inserts `(V[+0x10]>>44)&0xf` into the low nibble (`0xea4019–0xea403c`).
- `S+0x29c`: high dword of `V+8`; vertex dirty flag `S+0x280 |= 1` (`0xea4005`, `0xea4042–0xea4049`).
Index state:
- `S+0x688`: resolved index base.
- `S+0x694`: dword from `D+0xc`.
- `S+0x690=0`.
These writes occur at `0xea40b6–0xea40c6`. **This particular path explicitly clears `S+0x690`; do not blindly apply a previously captured binding’s index-width bit.** Your existing capture interprets that field for general `0xeac430` calls, but the meaning of `D+0xc` and any subsequent commit-time effects need the missing downstream code.
The helper then commits through `0xe61ae0` and tail-calls `0xeac430(G,count,first,baseVertex)` (`0xea40d0–0xea40f3`). The draw command itself contains **no explicit material, shader pair, transform matrix, or bone-palette pointer** (`0x1c1a–0x1c30`).
**Adjacent-draw merging**
Before allocating a draw, `0x1b80` flushes pending builder state with `0xcb0`, then asks the last command’s second virtual method to absorb the new draw (`0x1ba4`, `0x1bb3–0x1be8`).
`0x1b40` accepts when:
```cpp
old.mesh == new.mesh
old.mode == new.mode
old.baseVertex == new.baseVertex
old.firstIndex + old.indexCount == new.firstIndex
```
It then increments the old index count (`0x1b40–0x1b67`). It does **not** compare `Q+0x20`. Matching this code to the draw vtable’s second slot is strongly supported by the call signature and behavior, although the vtable bytes would provide direct confirmation.
**Nested sequence command: invoke `0x3710`, address point `0x2294030`**
| Offset | Meaning | Evidence |
|---|---|---|
| `+0` | Virtual address point | `0xff`, `0x592` |
| `+8` | `u32` child count | `0x10d`, `0x5a0` |
| `+0x10` | Inline array of child pointers | `0x130–0x138`, `0x5d0–0x5d8` |
It invokes children in array order, preserving `C`, with no sorting or state reset between children (`0x3720–0x3742`). It is a sequence node, not a mesh draw.
**Masked state command: invoke `0x3760`, address point `0x2294050`**
Allocated as `0x20` bytes at `0x522–0x54f`:
| Offset | Meaning | Constructor source |
|---|---|---|
| `+8` | Desired bits for first state word | Low dword of `B+0x28` |
| `+0xc` | Update mask for first state word | Low dword of `B+0x20` |
| `+0x10` | Desired bits for second state word | High dword of `B+0x28` |
| `+0x14` | Update mask for second state word | `B+0x24` |
The invoke updates selected bits of:
- `S′+0x420`, marking `S′+0xe |= 4` and writing `0x1010d` to `S′+0x80` when a bit changes (`0x3770–0x37ab`).
- `S′+0x430`, marking `S′+0xc |= 4` and writing `0x1001021d` to `S′+0x100` when a bit changes (`0x37d0–0x380a`).
Equivalent value operations are:
```cpp
word420 = (word420 & ~mask420) | (desired420 & mask420);
word430 = (word430 & ~mask430) | (desired430 & mask430);
```
Calling these specifically depth, blend, or stencil controls would require register/state decoding beyond these instructions.
**Pointer-payload command: invoke `0x3830`, address point `0x2294070`**
Allocated as `0x20` bytes, with a qword payload at `Q+0x10` (`0x674–0x68c`, `0xcf1–0xd0d`). At finalization, the payload comes from `0x55e9c0(B[+0x270])`; during ordinary state flushing it comes from `B+0x18` (`0x657–0x690`, `0xcdc–0xd0d`).
Its complete supplied behavior is:
```cpp
return function_3840(C, &Q->payload_at_10);
```
That is **the address of the captured field**, not simply its pointer value (`0x3830–0x383a`).
**Guess:** this is a program/state binding operation. The listing does not prove whether the payload identifies a shader pair, a constant source, or another binding object because `0x3840` and `0x55e9c0` are absent.
**Other state-command layouts recoverable from constructors**
These commands are necessary to interpret a whole sequence. Their invoke bodies are absent, so the following describes serialization, not confirmed GPU semantics.
| Address point | Captured layout | Evidence |
|---|---|---|
| `0x2294090` | `u32` at `+8`, sourced from builder `+0x680` or a manager value | `0x7c5–0x7f2`; `0x548d47–0x548d73` |
| `0x22940b0` | `+8/+0xc/+0x10/+0x14`: desired/mask pairs; `+0x18`: builder token `B+0x38`; `+0x20`: resolved pointer | `0xe52–0xe8c` |
| `0x22940d0` | `+8`: operation count; `+0xc`: payload-record count; `+0x10`: packed nibble operations; `+0x20`: inline `0x30`-byte records | `0x10f4–0x117a` |
| `0x22940f0` | `+8`: count; `+0xc`: packed pairs `{u16 value,u16 slot}` | `0x12a0–0x12c6`, `0x12d3–0x1307`, `0x131b–0x13ce` |
| `0x2294110` | Four dwords at `+8…+0x14`, copied from `B+0x690…+0x69c` | `0x14ca–0x152c` |
| `0x2294130` | `+8`: input `edx`; `+0xc`: input `ecx`; `+0x18`: copied `edx*ecx` bytes | `0x164e–0x16a4` |
| `0x2294150` | `+8`: `u16` input `edx`; `+0xa`: `u16` input `r9d`; `+0xc`: input `r8d`; `+0x10`: pointer to copied data; `+0x18`: copied `u16` array | `0x17d1–0x1867` |
| `0x2294170` | Same serialization as the preceding row, different command type | `0x1991–0x1a27` |
For `0x2294150/0x2294170`, the `u16` array has `r8d` elements; the following data begins at `Q+0x18+2*align_up(r8d,8)` and contains `edx*r9d` bytes (`0x1802–0x1809`, `0x1836–0x1867`; corresponding `0x19c2–0x1a27`).
These last types are **not automatically bone-palette uploads**. For example, `0x22b380` constructs CPU vertex records and submits them through `0x17c0` or `0x1980`, with record sizes `0x20`, `0x18`, `0x14`, `0x10`, or `0xc` depending on the branch (`0x22b7a3–0x22b7bc`, `0x22bc31–0x22bc52`, `0x22b914–0x22b930`, `0x22babf–0x22bb72`). Their payloads can therefore include generated geometry.
**Material, constants, transforms, and palettes**
The visible builder uses **state deltas**, not a complete material attached to every draw:
- `0xcb0` compares desired pointer `B+0x18` with cached pointer `B+0x10` and emits a `0x2294070` command only when different (`0xcdc–0xd20`).
- It combines pending masked state with a token from `B+0x38` and an auxiliary pointer into the `0x22940b0` command (`0xdfa–0xe8c`).
- If `B+0x50` is null, that auxiliary pointer can be resolved using the low 16 bits of `B+0x38`, a global table at `0x26944c0+0x3c88`, and the selected entry’s `+0x18` (`0xdfa–0xe34`). **Guess:** this is a resource/program-handle resolution path; its resource type is unproven.
- Eight `0x30`-byte state slots are tracked using dirty mask `B+0x2b8`, active pointers `B+0x2c0`, desired records `B+0x350`, previous pointers `B+0x300`, and cached records `B+0x4d0` (`0xf8e–0x10ee`). Changed records are copied into commands; removed slots produce nibble operations without a new record (`0x105c–0x10d9`).
- Another dirty mask, `B+0x44`, serializes slot/value pairs from `B+0x78` and updates cached values at `B+0x58` (`0x127a–0x12d1`).
The record provenance is partly visible. `0x542210` copies `0x30` bytes from its source `+0x30` and `+0x60` into builder slots zero and one; `0x5423d0` uses source `+0x90` and `+0xc0` instead. Both also prepare slots six/seven at `B+0x470/B+0x4a0` (`0x542210–0x5422a6`, `0x5423e7–0x542482`, `0x5422ad–0x542395`, `0x5424c5–0x5425b0`).
**Guess:** these records package resource or render-state descriptions. Their byte size and copying alone do not prove they are textures, constant buffers, or matrices.
There are identifiable object-related call boundaries:
- A stage object’s `+0x148` leads to an object whose `+0x80` is passed to missing `0x2410` before its draws (`0xab3d10–0xab3d61`; also `0xa6dae0–0xa6db3c`). **Guess:** this contributes object state, potentially transforms.
- Before a subset of draws, code passes a vector at record `+0`, a pointer to record `+0x5c`, and a byte count/control at `+0x68` to missing `0x544ba0` (`0xab3f40–0xab3f55`). Another subset calls missing `0x545520` (`0xab3e87–0xab3e94`). **Guess:** this distinguishes object/skinning constant preparation; the palette layout is not established.
- Missing `0x2fc0`, `0x3260`, and `0x33c0` receive slot numbers, counts, and data pointers, including stack-built data (`0x4c3fd7–0x4c4011`, `0x4c4108–0x4c4160`). They are important candidates for constant-data serialization, but their layouts and shader-stage assignments cannot be recovered here.
A native decoder must therefore treat transform/palette ownership and shader-slot meanings as unresolved, rather than assigning them to arbitrary draw-command fields.
**2. Allocation, containers, ordering, and phase lists**
**Allocation and append**
Commands use a shared bump pointer at `0x258e980`; `lock xadd` reserves bytes and returns the allocation address (`0xf6`, `0x52e`, `0x680`, `0x1c11`). Fixed commands reserve `0x10`, `0x20`, or `0x30` bytes; variable allocations round their computed sizes to 16-byte boundaries (`0x7da–0x7ef`, `0x522–0x53f`, `0x1c0c–0x1c1a`, `0x167e–0x168c`).
`0xb0(B,Q)` appends:
```cpp
B->commands[B->count++] = Q; // count +0x260, array +0x980
```
When the count reaches `0x800`, it copies all 2,048 pointers into a `0x2294030` sequence command, then replaces the pending array with that one sequence and sets the pending count to one (`0xb0–0x10d`, `0x130–0x1a0`).
This can produce **nested sequences** after repeated spills. A decoder must recurse; the builder does not simply flush or discard the first 2,048 entries (`0x199–0x1a0`, `0x3710–0x3742`).
**Finalized root container**
`0x4f0` finishes pending state and returns a different, **non-polymorphic** root:
```cpp
struct Root {
    uint32_t phase;       // +0x00
    uint32_t specialN;    // +0x04
    uint32_t ordinaryN;   // +0x08
    // +0x0c unspecified here
    Command* commands[]; // +0x10
};
```
The constructor initializes `+0=-1`, `+4=0`, `+8=N`, copies the command pointers, then replaces `+0` with `B+0x270` and clears the builder count (`0x905–0x946`, `0x9f0–0xa09`, `0xa56`).
**Do not read `Root+0` as a vtable pointer.**
When phase is `0xf`, finalizer argument `esi` is nonzero, and global `0x2486080+0x26bc` is nonzero, finalization moves the count from `+8` to `+4` and submits the root through `0xe3ff0` or `0xe4010` (`0xa0c–0xa51`). Their implementations are absent.
**Execution and cleanup**
| Walker | Count | Children | Additional behavior |
|---|---|---|---|
| `0x3710` | `Q+8` | `Q+0x10` | Child invokes only; `0x3720–0x3742` |
| `0xa70` | `Root+8` | `Root+0x10` | Post-list state/cache cleanup; `0xa80–0xba3` |
| `0xbb0` | `Root+4` | `Root+0x10` | Related post-list cleanup; `0xbc0–0xca3` |
Both root walkers can restore bit three of `S′+0x430` from thread-dependent manager state and invalidate a manager cache at `manager + threadIndex*0x430 + 0x588` (`0xad3–0xb5e`, `0xc13–0xc98`). They also conditionally write `-1.0f` into a TLS-selected tracking record (`0xaa4–0xacb`, `0xbe4–0xc0b`).
`0xa70` additionally replaces lane three of `S′+0x3760` with the value stored at `0xfa8870`, marking dirty state when changed (`0xb6a–0xb9f`). Its precise graphics meaning is unresolved.
**Sorting**
There is **no demonstrated sort of the finalized command pointers** in `0xb0`, `0xcb0`, `0x4f0`, or the three walkers. Their visible ordering is append order, nested-copy order, and sequential execution (`0xb0–0xbf`, `0x9f0–0xa00`, `0x3730–0x3742`).
Ordering may already be decided by the missing traversal helpers `0xa4d700` and `0xa4fff0`, or by the source arrays they traverse. The supplied callers only prove that they receive callback addresses and a builder—for example, `0xa54350–0xa54360` and `0xa54a80–0xa54a97`. A material-sort key, depth-sort key, or stable-sort rule would be a guess.
**Which roots exist**
The stage wrappers select a builder using `0x2486080+0x1a8+8*(input_edx&0xf)`, initialize it through missing `0x1b0`, traverse with a chosen callback, finalize it, and store the root on the wrapper object (`0xa54325–0xa5436f`). These are separate callback/root routes, not proven names for individual render passes:
| Builder wrapper | Traversal callback | Root field | Finalize mode / evidence |
|---|---|---|---|
| `0xa542f0` | `0xa6d910` | `+0xd8` | `0`; `0xa54350–0xa5436f` |
| `0xa54460` | `0xab3c20` | `+0xd8` | `0`; `0xa544c0–0xa544df` |
| `0xa545d0` | `0xaae800` | `+0xd8` | `2`; `0xa54630–0xa54652` |
| `0xa54740` | `0xab1bc0` | `+0xd8` | `0`; `0xa547a0–0xa547bf` |
| `0xa548b0` | `0xab4370` | `+0xd8` | `0`; `0xa54910–0xa5492f` |
| `0xa54a20` | `0xa7a5c0` | `+0xe0` | `0`; `0xa54a87–0xa54aa6` |
| `0xa54ba0` | `0xa72ae0` | `+0xe0` | `0`; `0xa54c07–0xa54c26` |
| `0xa54d20` | `0xabf380` | `+0xd8` | `0`; `0xa54d80–0xa54d9f` |
| `0xa54e90` | `0xabf7b0` | `+0xd8` | `0`; `0xa54ef0–0xa54f0f` |
| `0xa55000` | `0xabfd30`—absent | `+0xd8` | `0`; `0xa55060–0xa5507f` |
| `0xa55170` | `0xac0300`—absent | `+0xd8` | `0`; `0xa551d0–0xa551ef` |
| `0xa552e0` | `0xa75cb0` | `+0xd8` | `0`; `0xa55340–0xa5535f` |
| `0xa55470` | Conditional `0xa6c510`—absent | `+0xd8` | `1`; `0xa554df–0xa55531` |
| `0xa56f40` | `0xa8acf0`—absent | `+0xd8` | `0`; `0xa56fa0–0xa56fbf` |
| `0xa57150` | `0xa9d8b0`—absent | `+0xd8` | `0`; `0xa571b0–0xa571cf` |
| `0xa57360` | `0xaab4b0` | `+0xe0` | `0`; `0xa573c7–0xa573e6` |
The `+0xd8` root is executed through `0xa70` by the supplied wrappers `0xa542a0` and `0xa54410` (`0xa542b6–0xa542e0`, `0xa54416–0xa54450`). The corresponding consumers of every other stored field are not supplied.
Other observable list organizations:
- `0x9887d0` can create **12 root slots** at object `+0x100+8*i`, filtered by masks from `0xfcf070`, global `0x258ea80`, and object `+8/+0x20`; each uses object virtual method `+0x28` to populate the builder (`0x988823–0x98889d`).
- `0xe4030` executes roots through `0xbb0`: first the pointer array at owner `+0x278c00`, count `+0x278bf8`, then the array at `+0x276bf8`, count `+0x276bf4` (`0xe4046–0xe4057`, `0xe40b0–0xe40f2`). The entire operation is skipped when the second count is zero.
- `0x55fb20` selects phase `p` from its object `+8`, walks an object-pointer array at manager `+0x276988+8*p` with count at `+0x13b910+4*p`, and invokes each selected object’s virtual `+0x10` method (`0x55fd25–0x55fd86`, `0x55fef1–0x55ff0c`). Its phase configuration table is `0x1063a10`, stride `0x2c` (`0x55fd45–0x55fd6f`).
Your prepass/forward identification is runtime evidence from `frame-map.md` and the stated origins. **The walkers themselves do not identify the pass:** `0x373b` can occur in any phase using nested sequences. No supplied table contents connect every numeric `Root.phase` to a named pass.
**3. Capturing a complete list for native rendering**
**Recommended boundary: root entry plus ordered interpretation**
Hook `0xa70(Root,C)` and `0xbb0(Root,C)`, recording:
- Root phase, the appropriate count, and the pointer array (`0xa80–0xa99`, `0xbc0–0xbd9`).
- Execution context `C`, lower context `G=*(C+0x18)`, and the current state pointers (`0xea3f74`, `0xea3fe8`, `0x3774`).
- Every command’s actual invoke address, plus its payload and recursively nested sequences (`0x3730–0x3739`).
Hooking only `0x3710` captures a subtree. Hooking the root walkers captures the complete root and its cleanup boundary; suppress duplicate subtree captures when a root capture is active (`0x3710–0x374e`, `0xa70–0xbad`, `0xbb0–0xcad`).
At root entry, the context represents **inherited state before the first child runs**, not the state for all draws. Commands can change it before each mesh submission (`0x3760–0x3820`), and `0xea3f40` commits immediately before drawing (`0xea40d0–0xea40f3`).
A decoder can replace the GPU draw submissions only after it can interpret **every encountered state command in order**. A single entry-state snapshot plus all `0x36d0` payloads is insufficient.
**Useful secondary boundaries**
| Hook point | What it provides | Limitation |
|---|---|---|
| Return from `0x4f0` | Completed root and serialized commands; `0xa02–0xa61` | No execution context or final per-draw state |
| `0xe4030` immediately before `0xe40c6` / `0xe40e6` | Batched-root ordering and selected execution context | Entry `0xe4030` is before its virtual setup call at `0xe4088` |
| `0xea3f40` entry | Mesh pointer, execution context, exact draw arguments | Mesh bindings have not yet been refreshed or committed |
| `0xea40d8`, after commit | State immediately before the draw tail-call | Still a per-draw capture point |
| Existing `0xeac430` hook | Committed draw state and arguments | Existing `engineSite()` filters out scene-list draws |
The last limitation is concrete: `engine_capture.cpp:27` and `engineSite()` accept only the four shadow sites. List draws through the shown helper return to `0x3700` because `0xea3f40` tail-calls drawIndex; `0x373b`, `0xa9b`, and `0xbdb` belong to outer list-invoke frames (`0x36fb–0x3700`, `0xea40f3`, `0x3739`, `0xa99`, `0xbd9`).
**How to obtain shader pairs and constant data without PM4**
There are two levels of implementation:
1. **Capture while the engine still executes state commands.** At each committed draw, read CPU-side shader/binding state and copy the pointed-to constants/resources. This avoids parsing the GPU command stream, while retaining the engine’s state preparation (`0xea40d0–0xea40f3`).
2. **Decode the whole list independently.** Start with inherited state, interpret all binding/upload/state commands, and produce a self-contained native draw record whenever encountering `0x36d0`. This additionally requires the missing invoke bodies and binding-table semantics.
The supplied code supports parts of the first route:
| Data | Existing CPU-side access | Evidence |
|---|---|---|
| Vertex constant-slot pointers | `S+0x6c8+8*i`, currently only four slots | `engine_capture.cpp:206` |
| View descriptor | Qword at `S+0x20c8` | `engine_capture.cpp:207` |
| Pixel shader-associated metadata | `H=*(G+0xb8)` | `engine_capture.cpp:141` |
| Pixel resource-slot mapping | `u16 H[+0x1c+2*j]` | `engine_capture.cpp:144` |
| Pixel resource pointer slots | `*(S+0x8e8+8*j)` | `engine_capture.cpp:146` |
| Vertex fetch metadata | `F=*(S+0x698)`; entries at `*(F+0x10)`, count `F+0x18` | `engine_capture.cpp:111–120` |
The existing alpha capture selects a mapping with bit `0x8000` set, low 14 bits zero, and value other than `0xffff`, then copies a 32-byte texture descriptor from the corresponding resource pointer (`engine_capture.cpp:141–151`). **That selects one known shadow texture usage; it is not a general material resource decoder.** Generalization requires all relevant mapping entries, their resource types, samplers, and both shader stages.
Likewise, saving four constant pointers does not establish buffer sizes, all slots, or the meaning of the bytes (`engine_capture.cpp:206`). The view decoding—base masked to 40 bits, projection at `+0`, rotation at `+0x40`, camera at `+0x220`—is existing shadow-specific knowledge in the hook (`engine_capture.cpp:242–244`), not a universal layout proven for the forward shaders.
**There is no proven VS-program pointer offset or complete VS/PS-pair extraction path in the supplied listings.** `G+0xb8` is evidence for the pixel metadata path used by your hook; it does not establish the corresponding VS field or where either stage’s executable code resides.
For independent list decoding, the next decisive functions/data are:
- Vtable contents beginning at `0x2294000`, to enumerate actual invokes and merge methods.
- `0x3840` and the invokes behind `0x22940b0/0x22940d0/0x22940f0`, to resolve the program token, resource records, and slot/value bindings (`0x383a`, `0xe75–0xe8c`, `0x1133–0x113e`, `0x1304–0x1307`).
- `0x2410`, `0x2fc0`, `0x3260`, `0x33c0`, and their emitted command invokes, to locate object and constant-data serialization.
- `0xe61ae0` and its shader preparation callees, to connect engine binding state to final shader code and user-data layout (`0xea40d3`).
Until those are decoded, identify shaders at draw time using validated CPU-side binding metadata/code identity, rather than assuming `M`, `B+0x38`, or `Q+0x10` directly contains a VS/PS pair.
**Additional state the native draw records need**
Preserve effective render targets/depth attachment, viewport/scissor, rasterization/culling, depth/stencil, blending/write masks, topology, fetch layout, shader pair, constants, textures, and samplers.
This is a requirement for the proposed renderer, **not a claim that all those field offsets have been recovered here**. The supplied mesh path proves the vertex/index offsets; `0x3760` proves two additional mutable state words; the existing hook proves selected resource accesses. Exact render-target, depth-state, and complete shader-state offsets remain outside the supplied evidence.
Pass state also changes outside these lists: `0x55fb20` performs context setup before its object loop and changes several execution-context fields before each virtual render invocation (`0x55fb92`, `0x55fbaf`, `0x55fdb8–0x55fea9`). A root capture must retain that surrounding pass setup.
**4. Risks and omissions**
| Risk | Why it matters / evidence |
|---|---|
| Treating commands as independent draws | Material/constants/state are inherited and emitted only on changes; `0xcdc–0xd20`, `0xe38–0xe4c`, `0x1008–0x1056` |
| Dropping unknown command types | Constructors outside `0x2294000…` append commands too: `0x22b2b78`, `0x22b2b98`, `0x230c7a0`, and others; `0x21c10b–0x21c141`, `0x5423ac–0x5423c2`, `0x4c4036–0x4c405a` |
| Flattening by type or material | Walkers preserve order; state commands can bracket subsets of geometry; `0x3730–0x3742`, `0x21c166–0x21c211` |
| Capturing backing-buffer pointers too early | Vertex/index addresses can change with `G+0x328`; `0xea3fc3–0xea3fe1`, `0xea4096–0xea40b3` |
| Capturing pointers without their data lifetime | Some commands own inline copies, others retain external pointers; compare `0x16a4`, `0x1858–0x1867` with `0x68c`, `0xe8c` |
| Assuming one command equals one source mesh | Adjacent indexed draws merge by extending count; `0x1b40–0x1b67` |
| Missing generated geometry | `0x22b380` repacks CPU records and submits copied vertex/index data; `0x22b5c0–0x22b7bc`, `0x22bb20–0x22bb72` |
| Assuming all skinning inputs are in the mesh | Object-state and palette-like preparation occurs in separate, missing helpers before draw construction; `0xab3d61`, `0xab3e94`, `0xab3f55` |
| Skipping commits | `0xea3f40` explicitly commits before every submitted draw; your notes additionally report synchronization/shader preparation inside the commit path, whose bodies are absent here |
| Skipping root cleanup | `0xa70/0xbb0` restore state and invalidate caches after children; `0xaa4–0xba3`, `0xbe4–0xca3` |
| Losing deferred-root routing | Phase `0xf` roots can use `+4` rather than `+8` and enter separate queues; `0xa0c–0xa51`, `0xe40b0–0xe40f2` |
| Reusing a root-entry shader snapshot for every draw | Binding commands execute between draws; `0x3730–0x3742`, `0x3830–0x383a` |
The concrete implementation path supported by this evidence is to **capture root boundaries now, preserve ordered commands, and correlate each `0x36d0` with its committed CPU-side state while the engine still executes the list**. That produces reviewable native draw records without PM4 parsing. Eliminating the engine’s per-command execution requires decoding the missing state/binding invokes; the supplied mesh layout alone does not recover the scene’s materials, transforms, or shader constants.
