# Fixed-function draw state: a reading

Reading dated 2026-11-25. This traces the engine’s CPU state to the register packets used by indexed list draws. All addresses are eboot-relative x86 PCs unless identified as register indices. The investigation was read-only.

**The principal result:** attachments and selected restoration words are retained in `S`, but **`G/S` are not a complete register mirror**. Complete reconstruction of ordinary list draws also needs the engine execution context `C`, its committed state keys, and the pass defaults established earlier. Some registers retain values from previous writes.

Here, **proven** means visible instructions, **conditional** means valid for the identified engine path, and **unresolved** means the investigated code does not establish a universal draw-time value.

## 1. Pointer domains and capture boundary

| Symbol | Definition |
|---|---|
| `C` | Engine execution context passed to `0xe61ae0` |
| `G` | `u64(C+0x18)` |
| `S` | `u64(G+0x320)` |
| `T[i]` | Colour-target descriptor pointer, `u64(S+8*i)` |
| `D` | Depth-target descriptor pointer, `u64(S+0x40)` |
| `P` | Current PS header, `u64(G+0xc0)` |

The ordinary list-draw sequence is:

```text
0x36d0
  → 0xea3f40
      set topology if changed
      update vertex/index bindings
      0xe61ae0(C)
          commit raster/blend/depth state if dirty
          tail-call 0xe61310(C)
      → 0xeac430(G, count, firstIndex, baseVertex)
          0xeabdb0(...)
          calculate index address
          0xe574c0(G)       // includes lazy PS register emission
          → 0xfa0a10       // indexed draw wrapper
```

Evidence: `0xea3f62–0xea3f78`, `0xea40b6–0xea40f3`, `0xe62d0c–0xe62ed0`, `0xeac430–0xeac498`.

Consequently:

- **Raster, blend and depth/stencil commit has already happened at `0xeac430` entry.**
- **PS register emission may still happen afterward.** In particular, reconstruct `CB_SHADER_MASK` from the selected PS header rather than treating previously emitted shader state as current.
- Attachment binding and viewport/scissor setup happen earlier; commit does not rebind them before each draw.

No inspected instruction establishes a back-pointer from `G/S` to `C`. Retain `C` while observing `0xea3f40` or `0xe61ae0`; do not infer it from proximity to `G`.

## 2. What `S` actually retains

| Location | Proven contents | Installation evidence |
|---|---|---|
| `S+0x00+8*i`, `i=0…7` | Original colour-target descriptor pointers | `0xeaafb0–0xeaafb7`, unused entries cleared at `0xeab000` |
| `S+0x40` | Depth-target descriptor pointer | `0xeab03e` |
| `u8(S+0x48)` | Colour-target count | `0xeab01d–0xeab03a` |
| `u16(S+0x4c/+0x4e)` | Attachment width/height used for viewport/scissor setup | `0xeab044–0xeab05c`, depth-only case `0xeab0a9–0xeab0e2` |
| `S+0x50+0x10*i` | Additional 16-byte colour-target records | `0xeaafc9–0xeaafce` |
| `S+0x298c` | Cached unmasked `DB_DEPTH_CONTROL` constructor word | `0xeae5aa–0xeae5ac` |
| `S+0x2990` | Cached unmasked `DB_STENCIL_CONTROL` constructor word | `0xeae5b2–0xeae5b5` |
| `S+0x2994` | Cached unmasked **slot-zero** blend-control constructor word | `0xea6e07–0xea6e15`, `0xea6fa4–0xea6fb1` |
| `u16(S+0x2998/+0x299a/+0x299c/+0x299e)` | First viewport’s scissor `{left,top,right,bottom}` | `0xea9f5b–0xea9f87` |
| `S+0x688` | Index-buffer base | `0xea40b9` |
| `S+0x690` | Packed index byte offset and address-calculation width bit | `0xeac462–0xeac479` |

**`S+0x2994` is not an eight-entry blend array.** Its next dword is the beginning of the scissor cache.

The restoration purpose is directly visible: utility code reloads depth/stencil and scissor from these locations, for example `0xeb4271–0xeb42ce`; another path reloads blend slot zero from `S+0x2994` at `0xeb88d2–0xeb88e5`.

These are selected restoration fields, not a contiguous hardware-register image.

## 3. Colour render targets

### Pass-begin setters

The main attachment-binding routines are:

| Function | Operation |
|---|---|
| `0xeaa360(G,colour,depth)` | Bind one colour target and depth; disable colour slots 1–7 |
| `0xeab0f0(G,colours,count,depth)` | Bind multiple colour targets and depth; disable remaining colour slots |
| `0xeaa630(G,slot,T)` | Prepare a colour descriptor, sometimes modifying a temporary copy |
| `0xfa0520(G,slot,T)` | Write colour-target register packets |
| `0xfa0610(G,D)` | Write depth-target register packets |
| `0xeaaf90(S,...)` | Retain the original attachment pointers and dimensions |

The engine pass setter `0xe66220` reaches `0xeaa360` at `0xe66407`, or `0xeab0f0` at `0xe66452`.

The ordering matters: binding occurs first, then `S` is reset and populated with attachment pointers, then pass defaults are emitted. See `0xeaa3b4–0xeaa475` and `0xeab1bd–0xeab243`.

### Colour descriptor layout and packet

For a descriptor accepted by `0xfa0520`, register block `i` begins at:

```cpp
contextIndex = 0x318 + 0xf*i;
```

| Descriptor offset | Register |
|---:|---|
| `T+0x00` | `CB_COLORi_BASE` |
| `T+0x04` | `CB_COLORi_PITCH` |
| `T+0x08` | `CB_COLORi_SLICE` |
| `T+0x0c` | `CB_COLORi_VIEW` |
| `T+0x10` | `CB_COLORi_INFO` |
| `T+0x14` | `CB_COLORi_ATTRIB` |

**Packet evidence:** `0xfa058b–0xfa059c` calculates `0x318+0xf*i` and writes header `0xc00e6900`; `0xfa059f–0xfa05af` copies fourteen descriptor dwords, `T+0x00…0x34`, into the packet.

The overlapping vector stores still produce a contiguous fourteen-dword copy. Descriptor `T+0x3c` is placed in a separate NOP payload at `0xfa05b8–0xfa05c3`; it is not another colour register.

For a null or rejected descriptor, this helper writes **only**:

```cpp
CB_COLORi_INFO = 0;
```

Evidence: `0xfa05d0–0xfa05e8`, register index `0x31c+0xf*i`.

Thus inactive targets have disabled `INFO`, but their old base/pitch/slice/view/attrib registers need not be zero.

### Original descriptor versus effective descriptor

**Do not always copy the first six words from `u64(S+8*i)` unchanged.**

`0xeaa630` passes the original descriptor directly except when:

```cpp
(u32(T+0x10) & 0x2000) != 0
&& u32(T+0x1c) != 0
&& function_fa4bd0(T) == 0
```

On that path it copies 64 bytes to the stack and modifies the copy:

```cpp
tmp.INFO &= ~0x4000;

a = tmp.ATTRIB;
tmp.ATTRIB = (a & 0xfffffc1f) | ((a & 0x1f) << 5);

tmp.wordAt28 =
    (tmp.wordAt28 & 0xffc00000)
    | (tmp.SLICE & 0x003fffff);

p = tmp.PITCH;
tmp.PITCH =
    (p & 0x800fffff)
    | ((p << 20) & 0x7ff00000);
```

It then runs:

```text
base = 0xfa4940(&tmp)
0xfa47d0(&tmp, base)
0xfa0520(G,slot,&tmp)
```

Evidence: condition `0xeaa665–0xeaa683`; copy and transformations `0xeaa68a–0xeaa6e8`; address adjustment `0xeaa6eb–0xeaa6f5`; emission `0xeaa703`.

`0xfa47d0` updates descriptor `+0x24`, preserving address-layout bits as determined by the descriptor’s format/tiling calculations (`0xfa47f6–0xfa48d2`).

**Proven consequence:** `S` retains the original pointer, while the effective packet can contain changed `PITCH`, `INFO`, `ATTRIB`, and auxiliary surface words. A memory decoder must reproduce this preparation path.

The `0xfa4940`, `0xfa4bd0`, and `0xfa47d0` calculations are CPU descriptor operations. Reimplement them for capture; invoking the guest setters is unnecessary.

## 4. Depth target and HTILE

Let:

```cpp
D = u64(S+0x40);
```

For an accepted depth descriptor, `0xfa0610` emits:

| Descriptor offset | Register index | Register |
|---:|---:|---|
| `D+0x00` | `0x010` | `DB_Z_INFO` |
| `D+0x04` | `0x011` | `DB_STENCIL_INFO` |
| `D+0x08` | `0x012` | `DB_Z_READ_BASE` |
| `D+0x0c` | `0x013` | `DB_STENCIL_READ_BASE` |
| `D+0x10` | `0x014` | `DB_Z_WRITE_BASE` |
| `D+0x14` | `0x015` | `DB_STENCIL_WRITE_BASE` |
| `D+0x18` | `0x016` | `DB_DEPTH_SIZE` |
| `D+0x1c` | `0x017` | `DB_DEPTH_SLICE` |
| `D+0x20` | `0x002` | `DB_DEPTH_VIEW` |
| `D+0x24` | `0x005` | `DB_HTILE_DATA_BASE` |
| `D+0x28` | `0x2af` | `DB_HTILE_SURFACE` |
| `D+0x2c` | `0x00f` | `DB_DEPTH_INFO` |

Packet-writing PCs:

- `0xfa069b–0xfa06ac`: header `0xc0086900`, index `0x010`, eight dwords copied from `D`.
- `0xfa06b5–0xfa06c6`: `D+0x2c → 0x00f`.
- `0xfa06c9–0xfa06da`: `D+0x20 → 0x002`.
- `0xfa06dd–0xfa06ee`: `D+0x24 → 0x005`.
- `0xfa06f1–0xfa0702`: `D+0x28 → 0x2af`.

Descriptor `D+0x30` goes into a NOP payload (`0xfa0705–0xfa070f`).

Acceptance tests use `0xf9f670(D)` and `0xf9f9a0(D)`; at least one must return nonzero (`0xfa0672–0xfa0691`). These helpers calculate usable depth/stencil addresses with layout bits removed; they are not simply tests of the first descriptor word.

For null/rejected depth, `0xfa0718–0xfa072f` writes:

```cpp
DB_Z_INFO       = 0;
DB_STENCIL_INFO = 0;
```

The vector constant at `0x207eb80` supplies the first packet. **Other depth/HTILE registers remain inherited.**

**Draw-time validity:** these are pass-begin bindings, not lazy work inside `0xe61ae0`. They describe ordinary subsequent list draws provided no intervening setter or descriptor mutation changes the binding.

Utility routines temporarily bind other depth descriptors without changing `S+0x40`; examples include `0xeb3eb6`, `0xeb404c`, and `0xeb86c8`. Their internal draws therefore cannot universally use the retained pass descriptor.

## 5. Blend controls and colour masks

### Lazy commit and retained blend object

Commit tests engine dirty bit `2`, compares the requested blend key with its committed cache, and emits on change:

```text
key = u64(C+0x50)
compare with u64(C+0x840)
cache key at C+0x840
B = 0xec2420(key)
A = u64(B+0x18)
0xea6d20(A,G,0xffffffffffffffff)
```

Evidence: `0xe62de4–0xe62e18`.

`B+0x10` retains the key; `B+0x18` points to the blend description (`0xec2950–0xec2955`, `0xec2b8e`).

`A` has:

| Offset | Contents |
|---:|---|
| `A+0x00` | Flags; bit 1 controls the additional alpha-to-mask setup |
| `A+0x04` | Description count |
| `A+0x10+4*i` | Packed blend description and write-mask nibble |

### `CB_BLENDi_CONTROL`

`0xea6d20` constructs a register word from packed word `p`. The visible construction is:

```cpp
uint32_t blendWord(uint32_t p) {
    if (!(p & 1))
        return 0;

    return 0x60000000
        | (((p >>  1) & 31)      )
        | (((p >> 11) &  7) <<  5)
        | (((p >>  6) & 31) <<  8)
        | (((p >> 14) & 31) << 16)
        | (((p >> 24) &  7) << 21)
        | (((p >> 19) & 31) << 24);
}
```

Evidence: extraction/calls at `0xea6d7b–0xea6dee`, and constructor helpers `0xfa2c90`, `0xfa2cb0`, `0xfa2cd0`, `0xfa2d00`.

The packet helper applies:

```cpp
CB_BLENDi_CONTROL = blendWord(p) & 0xffff1fff;
```

**Packet:** `0xf9fe8e–0xf9fea6`, header `0xc0016900`, index `0x1e0+i`.

Selection rules:

- If `u32(A+4)==1`, description zero is broadcast to **all eight** controls (`0xea6ef3–0xea6f9f`).
- Otherwise, only controls `0…count-1` are emitted (`0xea6d70–0xea6e3b`).
- Controls outside that range retain their earlier values.
- Only control zero’s constructor word is cached at `S+0x2994`.

Therefore `S+0x2994` suffices for slot zero after masking, and for all eight only when the broadcast case is established.

### `CB_TARGET_MASK`

Each description contains:

```cpp
writeNibble = (p >> 27) & 0xf;
```

For the multi-description path:

```cpp
targetMask = OR_i(writeNibble[i] << (4*i));
```

For count one, its nibble is replicated across all eight targets.

Evidence: `0xea6e1b–0xea6e30`, `0xea6ed8–0xea6fae`.

**Packet:** `0xf9fdc8–0xf9fdd5`, register `0x08e`; called by `0xea6fc0`.

There is no demonstrated `G/S` mirror of the complete current target mask. Reconstruct it from the committed blend description.

Before the first blend commit, pass setup emits a mask covering its colour targets at `0xea9e34–0xea9e63`.

### `CB_COLOR_CONTROL`

This is **not emitted by `0xea6d20`**.

`0xfa0070(G,mode,rop)` writes:

```cpp
CB_COLOR_CONTROL =
    ((mode << 4) & 0x70)
    | ((rop << 16) & 0x00ff0000);
```

**Packet:** `0xfa00ae–0xfa00cf`, register `0x202`.

Ordinary pass setup supplies `mode=1`, `rop=0xcc` (`0xea9da4–0xea9db1`):

```cpp
CB_COLOR_CONTROL = 0x00cc0010;
```

That is a **pass default**, not a value read from the blend cache. Utility code temporarily changes it; for example `0xeb8553–0xeb855d`, followed by restoration at `0xeb86fd–0xeb870a`.

### Blend constants

`0xf9fec0` emits four float words to `0x105…0x108`:

- Header/index at `0xf9ff1f–0xf9ff25`.
- Payload at `0xf9ff2c–0xf9ff3b`.

Its recorded direct caller is `0xea9c40`, which supplies four zeros at `0xea9db6–0xea9dc9`.

**Conditional result for the investigated pass path:** blend constants are `{0,0,0,0}`. No complete `G/S` mirror was found, and `0xea6d20` does not update them.

### Two easy misidentifications

- `0xf9fdf0`, called at `0xea7020`, writes index `0x2dc`: **`DB_ALPHA_TO_MASK`**, not `CB_COLOR_CONTROL`.
- `0xfa0750`, called at `0xea6fcc`, writes indices `0x30e/0x30f`: sample masks. Commit supplies all ones.

## 6. `CB_SHADER_MASK`: selected PS state, emitted after entry

For the ordinary PS:

```cpp
P = u64(G+0xc0);
CB_SHADER_MASK = u32(P+0x34);   // when P != 0
```

The memory provenance is established by:

- PS installation in commit: `0xe621b6–0xe6222d`, `0xe58055–0xe5805c`.
- Selected header read: `0xe57514`.
- Header comparison including `P+0x34`: `0xe575f0–0xe57654`.
- Setter argument `P+8`: `0xe57656`.

Emission routes are:

| Route | Guest call sites |
|---|---|
| Set PS | `0xe57661 → 0xfa0df0`, import call `0xfa0e30 → 0xfa8360` |
| Update PS | `0xe576d4 → 0xfa0e50`, import call `0xfa0e90 → 0xfa81e0` |

These imports are `sceGnmSetPsShader350` and `sceGnmUpdatePsShader350`.

The local host implementation independently defines the PS register array’s final word as `cbShaderMask` and writes it to `0x08f`: `src/host/hle_gnm_packets.cpp`, `PsStageRegisters` and `sceGnmSetPsShader350`.

**Evidence boundary:** the guest disassembly proves the selected pointer and imported call. The `CB_SHADER_MASK` field interpretation and imported packet behavior are corroborated by the host implementation; there is no guest packet-store PC inside that import thunk.

At `drawIndex` entry, `G+0xd8` can still identify the previously emitted PS. Use **current `G+0xc0`**, because `0xe574c0` subsequently installs it at `G+0xd8` (`0xe57669`).

With PS disabled, the host setter writes shader mask zero.

## 7. Depth/stencil state

### Lazy commit

Commit handles dirty bits `4/8` and compares:

- Packed state key `C+0x58` against committed `C+0x848`.
- Dynamic reference `C+0x818` against committed `C+0xe70`.

The resolved depth-state object is retained at:

```cpp
E = u64(C+0xe78);
Z = E+0x10;
```

Evidence: `0xe62e24–0xe62ea6`; retained pointer at `0xe62e87`.

`0xeae560(Z,G,reference)` emits:

| Register | Effective value | Packet-writing PCs |
|---|---|---|
| `DB_DEPTH_CONTROL`, `0x200` | `u32(Z) & 0xc07007ff` | `0xfa0128–0xfa013c` |
| `DB_STENCIL_CONTROL`, `0x10b` | `u32(Z+4) & 0x00ffffff` | `0xfa0188–0xfa019c` |
| `DB_STENCILREFMASK`, `0x10c` | `m`, below | `0xf9ff88–0xf9ff95` |
| `DB_STENCILREFMASK_BF`, `0x10d` | Same `m` | `0xf9ff99` |

The reference/mask word is:

```cpp
m = (u32(Z+8) & 0x00ffff00)
    | uint8_t(reference)
    | 0x01000000;
```

Evidence: `0xeae588–0xeae59e`.

Thus:

```cpp
stencilReference = uint8_t(u32(C+0xe70));
stencilReadMask  = (u32(Z+8) >> 8)  & 0xff;
stencilWriteMask = (u32(Z+8) >> 16) & 0xff;
```

The setter writes identical front/back reference-mask words. It caches only the first two constructor words in `S+0x298c/+0x2990` (`0xeae5a3–0xeae5b5`).

**At ordinary list-draw entry, these committed values are already current.** No additional fixed-function depth/stencil commit is performed by `0xeac430`.

For a `G/S`-only capture, depth/stencil controls are available after masking. **Stencil reference and masks additionally require `C` and its retained depth object, or setter tracking.**

### `DB_RENDER_CONTROL`

`0xfa1e40` writes:

```cpp
DB_RENDER_CONTROL = argument & 0x1fff;
```

**Packet:** `0xfa1e78–0xfa1e8c`, index `0x000`.

Ordinary pass setup constructs zero through `0xfa2e10` and emits it at `0xea9c72–0xea9c7d`.

This register is not updated by `0xeae560` and has no demonstrated `G/S` mirror. Utility paths change it temporarily and restore zero; examples are `0xeb3b52` and `0xeb4252`, or `0xeb854e` and `0xeb886a`.

**Conditional result:** ordinary list draws after the normal pass setup use zero. Internal utility draws require their actual setter state.

## 8. Raster state and polygon offset

### Lazy commit and source object

Commit dirty bit `1` selects a raster object:

```text
key = u64(C+0x40)
compare with committed u64(C+0x830)
R = 0xe9b590(key)
cache key at C+0x830
```

Evidence: `0xe62d0c–0xe62d43`.

The returned object contains:

| Object offset | Contents consumed by commit |
|---:|---|
| `R+0x10` | `PA_SU_SC_MODE_CNTL` constructor word |
| `R+0x14` | Float multiplied by `16.0` for polygon-offset scale |
| `R+0x18` | Polygon-offset constant |
| `R+0x1c` | Polygon-offset clamp |
| `R+0x28` | Raster lookup key |

Object construction/key retention: `0xe9b728–0xe9b768`, `0xeb9dd3`.

The lookup pointer is not retained directly in `G/S` or in the shown commit cache; retain it when observing commit, or perform a read-only lookup using the committed key.

### Effective register values

| Register | Value | Packet-writing PCs |
|---|---|---|
| `PA_SU_SC_MODE_CNTL`, `0x205` | `u32(R+0x10) & 0x00393fff` | `0xfa20e8–0xfa20fc` |
| `PA_SU_POLY_OFFSET_FRONT_SCALE`, `0x2e0` | `16.0f * f32(R+0x14)` | `0xfa2368–0xfa2375` |
| `PA_SU_POLY_OFFSET_FRONT_OFFSET`, `0x2e1` | `f32(R+0x18)` | `0xfa237a` |
| `PA_SU_POLY_OFFSET_BACK_SCALE`, `0x2e2` | Same scale | `0xfa23d8–0xfa23e5` |
| `PA_SU_POLY_OFFSET_BACK_OFFSET`, `0x2e3` | Same constant | `0xfa23ea` |
| `PA_SU_POLY_OFFSET_CLAMP`, `0x2df` | `f32(R+0x1c)` | `0xfa228e–0xfa229b` |

Commit argument preparation and calls: `0xe62d68–0xe62dd8`. The multiplier at `0x2067304` is `16.0f`.

The mode word carries culling, winding, polygon mode and polygon-offset enables. Its constructors visibly manipulate:

- Cull bits `0…1`: `0xfa2d30`.
- Front-face bit `2`: `0xfa2d40`.
- Polygon-mode fields: `0xfa2d50`.
- Front/back offset-enable bits `11/12`: `0xfa2d90`.

Preserve the complete masked word rather than rebuilding only culling.

**Draw-time validity:** these packets are emitted during engine commit, before `0xeac430`. `G/S` contain no demonstrated complete raster mirror.

Pass initialization also emits raster defaults at `0xea97a6–0xea97f8`; normal pass setup invalidates the engine raster cache, forcing later committed state through the above path (`0xe9baa0–0xe9bad6`).

## 9. Reading committed raster/blend objects without calling setters

Both object caches are reachable through the manager at `0x26944c0`:

| Cache | Root pointer | Object key | Desired committed key |
|---|---|---|---|
| Raster | `u64(0x26944c0+0x38d0)` | `u64(object+0x28)` | `u64(C+0x830)` |
| Blend | `u64(0x26944c0+0x38d8)` | `u64(object+0x10)` | `u64(C+0x840)` |

The search representation is visible at `0xe9b600–0xe9b659` and `0xec2490–0xec24e9`:

```cpp
object = node + int16_t(u64(node+8) >> 48);

if (objectKey == wanted)
    return object;

edge = objectKey > wanted ? node+8 : node;
node = signExtend48(u64(edge));
```

For blend, the description passed to `0xea6d20` is then:

```cpp
A = u64(object+0x18);
```

Use a coherent guest-memory snapshot. These lookup functions also contain allocation/mutation paths, so reproducing their read-only search is preferable to executing them.

## 10. Viewport and scissor

### Persistent engine storage

`0xe66560` retains viewport/scissor inputs in `C`:

| Location | Contents |
|---|---|
| `u32(C+0x1dc8)` | Viewport count |
| `C+0x1dcc+0x10*i` | Viewport rectangle `{left,top,right,bottom}`, four signed dwords |
| `C+0x1e0c+8*i` | Two float viewport offsets |
| `C+0x1e40+0x10*i` | Clipped scissor rectangle |
| `C+0x1db8/+0x1dbc` | Attachment dimensions used for clipping |

Evidence: rectangle copy `0xe66590–0xe66599`; offsets `0xe665a5–0xe665ba`; clipped scissors `0xe66665–0xe6667d`; count `0xe66845`.

The function emits viewports through `0xeaa010` at `0xe66850`, then emits the separately clipped scissors through `0xea9ed0` at `0xe66863`.

**This is earlier setter work, not lazy draw commit.**

### Viewport register construction

For the normal explicit-rectangle path:

```cpp
halfWidth  = float(right-left) * 0.5f;
halfHeight = float(bottom-top) * 0.5f;

xScale  =  halfWidth;
xOffset = float(left) + halfWidth + offsetX;
yScale  = -halfHeight;
yOffset = float(top)  + halfHeight + offsetY;
zScale  = 1.0f;
zOffset = 0.0f;

zMin = 0.0f;
zMax = 1.0f;
```

Evidence: `0xeaa090–0xeaa131`; the no-offset branch has the same rectangle transformation at `0xeaa250–0xeaa2dc`.

`0xfa1bb0` writes:

| Register range | Payload | Evidence |
|---|---|---|
| `0x0b4+2*i … +1` | `zMin,zMax` | `0xfa1c0a–0xfa1c39` |
| `0x10f+6*i … +5` | `xScale,xOffset,yScale,yOffset,zScale,zOffset` | `0xfa1c4a–0xfa1c70` |

Pass begin initially installs a full-attachment viewport zero using `S+0x48` dimensions (`0xeaa4e5–0xeaa568`, or `0xeab2b3–0xeab356`).

**Limitation:** `G/S` do not retain the complete viewport array. The retained `C` rectangles and offsets reconstruct the `0xe66560` path; its count describes that engine setter, not proof that all higher hardware viewport registers were cleared.

### Scissor registers

`0xfa1c90` emits viewport scissor `i`:

```cpp
TL = (left & 0x7fff)
   | ((top & 0x7fff) << 16)
   | (disableWindowOffset ? 0x80000000 : 0);

BR = (right & 0x7fff)
   | ((bottom & 0x7fff) << 16);
```

The investigated engine calls pass the final argument as zero, which makes this helper set bit 31.

**Packet:** register pair `0x094+2*i`, header at `0xfa1cef`, index/payload at `0xfa1d00–0xfa1d2a`.

`0xea9ed0` retains only rectangle zero as four u16s at `S+0x2998…0x299e` (`0xea9f5b–0xea9f87`). Additional scissor rectangles require the `C` array.

After viewport scissors, it sets the screen scissor to:

```cpp
TL = 0;
BR = uint16_t(width) | (uint32_t(uint16_t(height)) << 16);
```

Evidence: `0xea9feb–0xeaa006`; packet writer `0xfa1da8–0xfa1dbf`, indices `0x00c/0x00d`.

Window offset is initialized to zero at `0xea97fd–0xea9804`; packet writer `0xfa04f6–0xfa0506`, index `0x080`.

Viewport transform control is initialized through `0xfa2eb0` to `0x43f` and emitted by `0xfa1de0` to `0x206` (`0xea9b5b–0xea9b6a`, `0xfa1e18–0xfa1e2c`).

**Scope:** the recovered arrays are viewport and screen scissor state for these engine setters. They do not establish a complete mirror of every window/generic scissor register.

## 11. Primitive type and index type

### Primitive type

`0xea3f40` compares requested topology with `u32(C+0xe74)`:

```cpp
if (u32(C+0xe74) != requested) {
    u32(C+0xe74) = requested;
    function_eabc90(G,requested);
}
```

Evidence: `0xea3f62–0xea3f78`.

`0xeabc90` immediately calls `0xfa08b0`, which emits:

```cpp
VGT_PRIMITIVE_TYPE = requested & 0x3f;
```

**Packet:** `0xfa0935–0xfa093e`, `SET_UCONFIG_REG` header `0xc0017900`, destination word `0x242` or `0x10000242` depending on the library’s mode.

The value is already emitted before commit and draw entry. Recover it from `C+0xe74` on this path.

`S+0x29a0` retains a topology-related classification bit, not the complete primitive type (`0xeabcae–0xeabcf1`).

### Index type

The library’s explicit index-type writer is `0xfa0d20`.

**Packet:** `0xfa0d6c` writes `0xc0002a00`; `0xfa0d8b` writes its payload.

Its recorded direct caller is `0xea9740`, supplying:

```cpp
indexTypeArgument = 0;
secondArgument    = 2;
```

Evidence: `0xea999d–0xea99a7`. These arguments produce payload zero: the ordinary 16-bit index setting.

The list mesh path also initializes:

```cpp
u32(S+0x690) = 0;
```

at `0xea40c6`.

`0xeac430` computes the index address as:

```cpp
packed = u32(S+0x690);
shift  = 1 + (packed >> 31);

address =
    u64(S+0x688)
    + (packed & 0x7fffffff)
    + uint32_t(firstIndex << shift);
```

Evidence: `0xeac462–0xeac479`.

**Distinction:** `S+0x690` controls address arithmetic; the draw does not use it to emit a new index-type packet. Its high bit alone is therefore not proof of the effective GPU index type.

For the identified ordinary pass/list path, both setup and addressing establish 16-bit indices. No per-draw index-type update occurs in `0xeac430`.

## 12. Draw-time reconstruction contract

For an ordinary list draw after the normal pass setup, capture:

```cpp
G = drawArgument;
S = u64(G+0x320);

// Retain C from the enclosing engine call:
assert(u64(C+0x18) == G);

colourPointers[0..7] = u64(S+0x00 ... S+0x38);
depthPointer        = u64(S+0x40);
attachmentSummary   = u64(S+0x48);

depthControlSource   = u32(S+0x298c);
stencilControlSource = u32(S+0x2990);
blend0Source         = u32(S+0x2994);
scissor0             = bytes8(S+0x2998);

rasterKey    = u64(C+0x830);
blendKey     = u64(C+0x840);
depthObject  = u64(C+0xe78);
stencilRef   = u32(C+0xe70);
primitive    = u32(C+0xe74) & 0x3f;

viewportCount   = u32(C+0x1dc8);
viewportRects   = C+0x1dcc;
viewportOffsets = C+0x1e0c;
scissorRects    = C+0x1e40;

psHeader = u64(G+0xc0);
```

Then:

1. Reproduce colour-descriptor preparation before copying its register words.
2. Decode depth-target words and HTILE directly from the retained accepted descriptor.
3. Resolve the committed blend description and reconstruct its emitted controls and target mask.
4. Apply the depth/stencil writer masks and reconstruct both reference-mask words.
5. Resolve the committed raster object and reproduce polygon-offset float operations.
6. Reconstruct viewport/scissor packets from retained `C` state.
7. Use the selected PS header’s shader mask, including the emission that follows draw entry.
8. Supply the established ordinary-pass defaults: `CB_COLOR_CONTROL=0x00cc0010`, zero blend constants, `DB_RENDER_CONTROL=0`, and 16-bit indices.

These defaults must carry provenance. They are established by earlier setters, not discovered in a complete CPU register bank.

| Requested state | Recoverable from `G/S` alone? | Additional requirement |
|---|---|---|
| Bound colour/depth descriptors | Yes, conditionally | Reproduce descriptor preparation; exclude temporary utility bindings |
| Exact disabled attachment register contents | No | Earlier register history; only disabling info words are written |
| All eight blend controls | No | Committed blend object; untouched controls remain inherited |
| `CB_TARGET_MASK` | No complete mirror | Committed blend description or setter tracking |
| `CB_SHADER_MASK` | Yes for selected ordinary PS | `u32(u64(G+0xc0)+0x34)` and imported-setter semantics |
| `CB_COLOR_CONTROL`, blend constants | Pass defaults | Track exceptions for broader coverage |
| Depth/stencil controls | Yes | Apply writer masks |
| Stencil reference/read/write masks | No | `C+0xe70` and retained depth object |
| `DB_RENDER_CONTROL` | Pass default | Track temporary overrides |
| Raster/polygon offset | No | `C`, committed key and raster object |
| Complete viewport/scissor arrays | No | Retained `C` arrays |
| Primitive type | No complete mirror | `C+0xe74` on the list path |
| Index type | Ordinary-path setting established | Earlier index-type setter, separately from address arithmetic |

**What is proven versus still conditional:** the offsets, transformations, packet destinations and dirty-commit boundaries above are concrete. A universal claim that arbitrary `0xeac430` calls can be decoded from only `G/S` is not supported. The practical PM4-free implementation is to retain `C`, decode committed objects, and maintain a small setter-state record for inherited registers and temporary bindings. That closes the identified gaps without parsing the PM4 stream.
