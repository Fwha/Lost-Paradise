# The engine's draw state: a reading

Codex's second reading (2026-11-24): the list commands' state writes, where the bound shaders,
constants, textures and fetch state are at drawIndex, and what the commit (0xe61ae0) finalizes.

The new code establishes the binding banks and commit sequence, but **does not yet establish a complete shader/user-data decoder**. Several decisive callees are absent from this directory, including `0x3ad0`, `0x3f80`, `0xe67b00`, `0xeba8c0`, and the resource-binding helpers called by `0xeabdb0`.
Two corrections to `reading1.md` matter immediately:
- **`0x3830 → 0x3840` selects packed pipeline/pass state, not a shader pair.** It consumes individual bytes of `Q+0x10`, updating `C+0x40/+0x48/+0x58/+0x60` and calling `0xe84160` (`0x3840–0x38d9`).
- **`S+0x6c8` is not established as a VS constant-buffer table.** It is the resource-pointer bank for physical bank zero. Ordinary PS resources are at `S+0x8e8`; ordinary VS resources are at `S+0xb08`. Dedicated constant-buffer V# records use different banks (`0xeac179–0xeac256`, `0xeac0ac–0xeac179`).
Below, **proven** means visible instructions; **inference** means a semantic interpretation of those instructions; **unknown** identifies a missing link.
**1. Pointer domains and capture boundary**
| Symbol | Definition | Evidence |
|---|---|---|
| `Q` | Invoked command | `0x3730–0x3739` |
| `C` | Engine execution context | Preserved and forwarded at `0x371d`, `0x3733` |
| `G` | `u64(C+0x18)` | `0xea3f74`, `0xea40d8` |
| `S` | `u64(G+0x320)` | `0xea3fe8`, `0xea4084` |
| `S′` | `u64(C+0x38)` | `0x3774`, `0x41f0`, `0xe61330` |
**Do not assume `S′ == S`.** The code uses them for different operations: `0xe61310` copies upload data from `S′+0x420+offset` and installs resulting buffer descriptors into `S` (`0xe613c6–0xe61597`). No supplied initialization proves their relationship.
There are three preparation boundaries:
```text
ea3f40:
    refresh mesh bindings in S
    e61ae0(C)                  // engine commit, including e61310(C)
    tail-call eac430(G,...)
eac430:
    eabdb0(G,...,baseVertex,...) // attribute V# construction and stage bindings
    compute index address
    e574c0(G)                  // further shader preparation/emission
    tail-call fa0a10           // indexed draw submission
```
Evidence: `0xea40b6–0xea40f3`, `0xeac43a–0xeac498`.
Thus **entry to `0xeac430` is after engine commit, but before all lower-level draw preparation** (`0xeac453`, `0xeac480`).
**2. Every invoke in the supplied vtable range**
The address points below come directly from `vtables_2294000.txt`. Their second slot is `0x3750` except the indexed draw, whose second slot is `0x1b40`. `0x3750` returns false; it is not a destructor body (`0x3750–0x3752`).
| Address point → invoke | Proven operation and writes |
|---|---|
| `0x2294010 → 0x36d0` | Calls `0xea3f40(M,C,mode,count,first,baseVertex,extra)` using `M=Q+8`, mode/count at `Q+0x10/+0x14`, first/base at `Q+0x18/+0x1c`, extra at `Q+0x20`. Return address is exactly `0x3700` (`0x36d8–0x3700`). |
| `0x2294030 → 0x3710` | Invokes `u32(Q+8)` children from `Q+0x10`, preserving order and `C`; no direct state writes (`0x371a–0x3742`). |
| `0x2294050 → 0x3760` | Masked writes to `S′+0x420` and `S′+0x430`, detailed below (`0x3760–0x3820`). |
| `0x2294070 → 0x3830` | Calls `0x3840(C,&Q[0x10])`; packed state selection, detailed below (`0x3830–0x383a`). |
| `0x2294090 → 0x38e0` | Tail-calls `0xe7d6f0(u32(Q+8),C)`. **Callee absent; downstream writes unknown** (`0x38e0–0x38e3`). |
| `0x22940b0 → 0x38f0` | Sets shader handles/auxiliary binding pointer in `C`, then performs the same masked `S′` updates as `0x3760` (`0x38f0–0x39e0`). |
| `0x22940d0 → 0x39f0` | Interprets packed nibble operations over eight `0x30`-byte engine records at `C+0x1f10+0x30*i`. Set/update calls missing `0x3ad0`; removal calls missing `0x3f80` for slots `0…5` (`0x3a20–0x3ab8`). |
| `0x22940f0 → 0x4100` | Resolves resource handles and binds PS resource pointers at `S+0x8e8+8*i`; details below (`0x4120–0x41d3`). |
| `0x2294110 → 0x41f0` | Compares/copies 16 bytes from `Q+8` to `S′+0x5480`; on change sets `byte(S′+0xf)|=0x10`, `u64(S′+0x30)=0x4f40130303` (`0x41f0–0x4220`). |
| `0x2294130 → 0x4230` | Ensures topology/mode `6`: writes `C+0xe74=6`, calls `0xeabc90(G,6)` if changed; tail-calls missing `0xea2a70(C,Q+0x18,u32(Q+8),u32(Q+0xc))` (`0x4241–0x4287`). |
| `0x2294150 → 0x4290` | Ensures mode `4`; tail-calls missing `0xea2ba0(C,u64(Q+0x10),u16(Q+8),Q+0x18,u32(Q+0xc),u16(Q+0xa))` (`0x42a4–0x4308`). |
| `0x2294170 → 0x4310` | Same argument layout as `0x4290`, mode `6` (`0x4324–0x4388`). |
| `0x2294190 → 0x4390` | Tail-calls missing `0xe916b0(u64(Q+8),C)`; downstream writes unknown (`0x4390–0x4394`). |
| `0x22941b0 → 0x43a0` | Calls missing `0xe90b40(C)`, compares/copies 32 bytes from `Q+0x10` to `S′+0x3740`; on change sets `byte(S′+0xf)|=4`, `u64(S′+0x40)=0x3320020305` (`0x43a7–0x4433`). |
| `0x22941d0 → 0x4440` | Tail-calls missing `0xe90940(C)`; downstream writes unknown (`0x4440–0x4443`). |
| `0x22941f0 → 0x4450` | Tail-calls missing `0xe67b00(C,Q+0x20)`. Constructor `0x2410` supplies a transposed three-vector transform payload (`0x4450–0x445a`, `0x246c–0x24e2`). |
For `0x39f0`, each operation is:
```cpp
slot = nibble & 7;
if (nibble & 8) {
    function_3ad0(C, slot, nextRecord);
    nextRecord += 0x30;
} else if (u32(C + 0x1f10 + 0x30*slot) != 0x7fffffff) {
    u32(C + 0x1f10 + 0x30*slot) = 0x7fffffff;
    if (slot <= 5)
        function_3f80(recordAddress, tlsContextOrFallback, slot);
}
```
Evidence: `0x3a20–0x3ab2`. **The missing callees prevent identifying these records’ ultimate GPU bindings.**
**Masked words: `0x3760`, also `0x38f0`**
For desired/mask pairs at `Q+8/+0xc` and `Q+0x10/+0x14`:
```cpp
new420 = (old420 & ~mask420) | (desired420 & mask420);
new430 = (old430 & ~mask430) | (desired430 & mask430);
```
| Changed destination | Additional writes | Evidence |
|---|---|---|
| `u32(S′+0x420)` | `byte(S′+0xe)|=4`; `u64(S′+0x80)=0x1010d` | `0x3774–0x37ab` |
| `u32(S′+0x430)` | `byte(S′+0xc)|=4`; `u64(S′+0x100)=0x1001021d` | `0x37d4–0x380a` |
**Their identification as depth/blend/raster words is unproven.** They lie in the upload-source area consumed by `0xe61310`, whose source starts at `S′+0x420` (`0xe613c6–0xe613e3`). Preserve them as engine upload state until the destination descriptors are decoded.
**Packed state selection: `0x3830 → 0x3840`**
Let `p=&Q[0x10]`:
| Input/state | Effect | Evidence |
|---|---|---|
| `p[0]` | Stores selector in `C+0x48`; replaces bits `2…3` of `C+0x40` using `u32(table + 16*u32(C+0x4c) + 4*p[0]) & 3`, table pointer at `0x238e378` | `0x3840–0x386c` |
| — | `u64(C)|=5` | `0x384a–0x3874` |
| `p[1]`, `C+0x1dc0` | Computes `C+0x60`; starts with `3`, conditionally replaces it with `p[1] | ((C[0x1dc0]>>20)&1)` | `0x3870–0x389d` |
| `C+0x60` | Replaces low six bits of `C+0x58` from byte table `0x20672b0` | `0x38a0–0x38d2` |
| `p[4]` | Calls `0xe84160(C,p[4])`; updates `C+0x58/+0x5c/+0x5e/+0x818` through table `0x2898be0`, and marks dirty bits `4/8` | `0x38d5–0x38d9`, `0xe84160–0xe841f6` |
`0x55e9c0(phase)` constructs this packed value from phase-table fields:
```cpp
0x01000000
+ ((u32(0x1063a10 + phase*0x2c + 0x10) & 0xff) << 16)
+ ((u32(0x1063a10 + phase*0x2c + 0x14) & 0xff) << 8)
```
Evidence: `0x55e9c0–0x55e9ef`. This is an integer bitfield, despite earlier pointer-like serialization.
**Shader selection: `0x38f0`**
| Destination | Value | Evidence |
|---|---|---|
| `u16(C+0xe80)` | Low 16 bits of `u64(Q+0x18)` | `0x38f0–0x3900` |
| `u16(C+0xe82)` | Bits `16…31` of that value | `0x3907–0x391d` |
| `u16(C+0xe86)` | `0` | `0x390a` |
| `u64(C+0xe98)` | `u64(Q+0x20)` | `0x38f9–0x3913` |
| `u64(C)` | Previous value OR `0xf0` | `0x38f4–0x391a` |
**Inference:** `C+0xe80` is the engine VS handle and `C+0xe82` the PS handle, based on their commit destinations and the PS binding call (`0xe61b50–0xe61baa`, `0xe6212c–0xe6222d`).
**Resource-handle binding: `0x4100`**
Each packed dword at `Q+0xc+4*k` contains `{u16 handle,u16 slot}` (`0x4120–0x412a`).
For changed slot `i`:
- Cache handle at `C+0x18a8+4*i`.
- Resolve `handle-1` through pointer table `u64(0x26944c0+0x3c68)`, bounded by `u32(0x26944c0+0x3c7c)`.
- Store resolved object at `C+0x1a68+8*i`.
- If nonnull, bind `u64(object+0x28)+0x20` at `S+0x8e8+8*i`, set bit `i` in `S+0x8c8` and `S+0x8d8`, and bit `16+i` in `u64(C)`.
- If invalid/null, clear bit `i` of `S+0x8d8`; **the old pointer is not cleared**.
Evidence: `0x4137–0x41d3`. Consequently, pointer nonnullness alone does not establish a current valid binding.
**3. Shader identity after `0xe61ae0`**
| Location | Proven contents after the corresponding dirty update | Evidence |
|---|---|---|
| `C+0xeb0` | Engine object resolved from VS handle `C+0xe80` via manager table `+0x3c88`, count `+0x3c9c` | `0xe61b50–0xe61b94` |
| `S+0x100` | Pointer `u64(C+0xeb0)+0x38` | `0xe61b94–0xe61baa` |
| `C+0xe98` | Auxiliary binding object; if null, filled from `u64(VSObject+0x18)` | `0xe62043–0xe6205a` |
| `C+0xea0/+0xea4` | Two dwords forming the VS-variant lookup key | `0xe62061–0xe6209a`, `0xe61de0–0xe61dea` |
| `C+0xeb8` | Resolved VS variant returned by `0xe79920` | `0xe6209a–0xe620a9` |
| `S+0x698` | `VSVariant+0x30`, or null | `0xe620b4–0xe62101` |
| `S+0x6a0` | `u32(VSVariant+0x58)`, or `0xffffffff` | `0xe620e1–0xe6210c` |
| `C+0xec0` | Engine object resolved from PS handle `C+0xe82` via manager table `+0x3ca8`, count `+0x3cbc` | `0xe6212c–0xe62180` |
| `S+0xf0` | `PSObject+0x58` | `0xe621a5–0xe621a9` |
| `G+0xc0` | `H=u64(PSObject+0x68)` | `0xe621bb–0xe6222d`, `0xe5805c` |
| `G+0xb8` | `H+4*v`, where `v=(u32(PSObject+0x74)>>4)&0xffff` | `0xe621b6–0xe621cd`, `0xe58055` |
`G+0xb8` is the **selected PS mapping metadata pointer** used by the existing hook; it is not established as shader code (`0xe58055`, `engine_capture.cpp`, `readAlphaInputsNow`).
For ordinary VS preparation, `0xe574c0` consumes the object at `G+0x100`, its register-related fields at `+0x18/+0x1c/+0x20`, and metadata at `G+0xf8` (`0xe5751b–0xe575ba`, `0xe57686–0xe576cb`). **The supplied `0xe61ae0` does not itself demonstrate refreshing that ordinary VS binding through `0xe57fc0`**; `0xeabdb0` subsequently calls missing `0xeba8c0` with the fetch-associated binding (`0xeabfb1–0xeabfc2`).
Therefore:
- **Engine shader identity at drawIndex entry:** capture `C+0xeb0`, `C+0xeb8`, `C+0xec0`, plus handles and variant key (`0xe61b94`, `0xe620a9`, `0xe62180`).
- **PS header/mapping identity:** capture `G+0xc0` and `G+0xb8` (`0xe58055–0xe5805c`).
- **VS code pointer / PS code pointer / OrbShdr pointer:** **not precisely recoverable from the supplied bodies**. The missing binding-resolution/emission code and shader-object contents prevent assigning those labels to a particular pointer.
**4. Resource, sampler, and constant-buffer banks**
`0xeabdb0` distinguishes the logical stage number passed to helpers from the **physical bank number** used for state access (`0xeac050–0xeac0ac`).
For the ordinary configuration without its special remapping branches:
| Stage | Logical stage / physical bank |
|---|---|
| PS | `1 / 1` |
| VS | `2 / 2` |
These assignments are supported by the PS/VS paths passing `esi=1/2` in `0xe574c0`, and the default `bank=stage` path in `0xeabdb0` (`0xe57a50–0xe57b32`, `0xe577ec–0xe578d6`, `0xeac09d–0xeac0ac`).
**Special pipelines remap banks.** Reproduce `0xeac01d–0xeac0a2` using `S+0xdc`; do not hard-code bank two for every vertex-stage configuration.
For physical bank `b`:
| Category | Usage mask | Binding storage | Other state | Evidence |
|---|---|---|---|---|
| Resource pointers | `u64(S+0x160+0x20*b)` | `u64(S+0x6c8+0x220*b+8*i)` | Type selector bits at `S+0x6a8+0x220*b`; dirty bits at `S+0x6b8+0x220*b` | `0xeac179–0xeac25f` |
| Sampler pointers | `u32(S+0x16c+0x20*b)` | `u64(S+0x1378+0x90*b+8*i)` | Dirty mask at `S+0x136c+0x90*b` | `0xeac256–0xeac31c` |
| Constant-buffer V# records | `u32(S+0x170+0x20*b)` | Record at `S+0x1d38+0x210*b+0x20*i` | Dirty mask at `S+0x1d2c+0x210*b` | `0xeac0ac–0xeac185` |
| Additional resource class | `u32(S+0x168+0x20*b)` | Record at `S+0x16d8+0x110*b+0x10*i`; first qword is passed onward | Type bits at `S+0x16c8+0x110*b`; dirty mask at `S+0x16d0+0x110*b` | `0xeac31e–0xeac3eb` |
**Inference:** the sampler classification follows commit’s conversion of packed sampling parameters through `0xe9cce0` and installation into these banks (`0xe629a0–0xe62a5f`, `0xe62bc0–0xe62c72`). The absent `0xe58830` prevents proving its complete descriptor transfer semantics.
Expanded ordinary-stage offsets:
| Field | PS, bank `1` | VS, bank `2` |
|---|---:|---:|
| Resource usage mask | `S+0x180` | `S+0x1a0` |
| Resource pointers | `S+0x8e8+8*i` | `S+0xb08+8*i` |
| Resource type bits | `S+0x8c8` | `S+0xae8` |
| Resource dirty bits | `S+0x8d8` | `S+0xaf8` |
| Sampler usage mask | `S+0x18c` | `S+0x1ac` |
| Sampler pointers | `S+0x1408+8*i` | `S+0x1498+8*i` |
| CB usage mask | `S+0x190` | `S+0x1b0` |
| CB records | `S+0x1f48+0x20*i` | `S+0x2158+0x20*i` |
| Additional-resource usage | `S+0x188` | `S+0x1a8` |
| Additional-resource records | `S+0x17e8+0x10*i` | `S+0x18f8+0x10*i` |
All offsets are arithmetic expansions of the instructions cited in the preceding table.
**Dedicated constant-buffer record layout**
For the VS records produced by `0xe61310`:
| Record offset | Contents |
|---|---|
| `+0x00…+0x0f` | Four descriptor dwords, copied from backing descriptor `D+0x10/+0x18`, with the address in the first qword replaced by the resolved backing base plus allocation offset |
| `+0x10` | Allocation offset |
| `+0x14` | `u32(D+0xc)` |
| `+0x18…+0x1f` | Not assigned by this shown update |
Evidence: `0xe614d4–0xe61597`. PS uses the same construction at `S+0x1f48+0x20*i` (`0xe615c9–0xe61694`).
The address replacement preserves the high 20 bits of the first qword and replaces its low 44 bits:
```cpp
descriptorQword0 =
    (u64(D+0x10) & 0xfffff00000000000)
    | ((resolvedBackingBase + allocationOffset) & 0xfffffffffff);
```
Evidence: `0xe61530–0xe61587`, `0xe6162d–0xe61684`.
**Do not interpret record `+0x10/+0x14` as V# words 2/3.** The V# occupies the first 16 bytes; those are additional bookkeeping fields (`0xe6156d–0xe61597`).
The resource-pointer category also supports buffer-like resources, not just textures: its type bit selects `0xe584d0` versus `0xe58680` (`0xeac1c3–0xeac21c`). **Those helpers are absent**, so neither descriptor size nor complete type semantics can be proven for every pointer.
**5. User data and resource mapping**
The supplied code exposes CPU backing tables used to prepare user data:
| Location | Observed use | Evidence |
|---|---|---|
| `T=u64(G+0x68)` | CPU-side backing table | `0xe57741`, `0xe57913` |
| `T+0x1800` | PS staging data copied according to `u16(PSMetadata)` | `0xe579dd–0xe57a3c` |
| `T+0x3000` | Ordinary VS staging data copied according to `u16(VSMetadata)` | `0xe57780–0xe577dc`, `0xe57903–0xe57943` |
| `G+0x60` | Moving allocation pointer for those copies | `0xe57798–0xe577aa`, `0xe579f4–0xe57a06` |
| Metadata bytes `+6…+0xe`, words `+0x12/+0x14/+0x16/+0x18/+0x1a` | Register destinations and offsets used when passing pointers/scalars to missing `0xfa1950/0xfa19e0` | `0xe577de–0xe578d6`, `0xe57947–0xe579c8`, `0xe57a3e–0xe57b32` |
These are **not proven flat arrays containing every final SGPR**. Some values are formed during `0xe574c0`, and their final setter implementations are absent (`0xe578c1–0xe578d6`, `0xe5799d–0xe579c8`).
The only supplied concrete texture mapping decoder is the existing shadow hook:
```cpp
H = u64(G+0xb8);
m = u16(H+0x1c+2*j);
p = u64(S+0x8e8+8*j);
```
It selects `m != 0xffff`, `m&0x8000`, `(m&0x3fff)==0`, then copies 32 bytes from `p` (`engine_capture.cpp`, `readAlphaInputsNow`). **This proves the hook’s selected texture access, not a general mapping interpretation for all slots or both stages.**
**6. Vertex fetch and index state**
Fetch metadata after engine commit:
```cpp
F = u64(S+0x698);
binding = u64(F);
entries = u64(F+0x10);
attributeCount = u32(F+0x18);
bindingControl = u32(F+0x1c);
```
Evidence: installation at `0xe620bb–0xe620eb`; consumption at `0xeabeb2–0xeabfc2`.
For each eight-byte entry `{a,word3}`:
| Bits/field | Meaning in the actual construction | Evidence |
|---|---|---|
| `a & 0xff` | Attribute byte offset divided by four | `0xeabefc–0xeabf10` |
| `(a>>8)&0xff` | Vertex stream | `0xeabf07–0xeabf24` |
| `a & 0xff0000` | Selects the alternate indexing path | `0xeabf3b–0xeabf56` |
| `a>>24` | Record-count value when shifted stride is zero | `0xeabf60–0xeabf66` |
| Entry dword `+4` | Final attribute V# word 3 | `0xeabf87–0xeabfa1` |
Stream `j` occupies:
| Offset | Contents |
|---|---|
| `S+0x288+0x20*j` | Base address |
| `S+0x290+0x20*j` | Additional byte offset |
| `S+0x294+0x20*j` | Available record count |
| `S+0x298+0x20*j` | Packed stride/format |
| `S+0x29c+0x20*j` | Additional source descriptor data |
Evidence: `0xeabf0c–0xeabf3b`; stream-zero writes at `0xea3ff4–0xea4049`.
For ordinary non-instanced attributes, the exact construction is:
```cpp
fmt = u32(S+0x298+0x20*j);
strideForArithmetic = fmt >> 4; // literal instruction; not a 16-bit mask
address =
    u64(S+0x288+0x20*j)
    + u32(S+0x290+0x20*j)
    + 4*(a&0xff)
    + uint32_t(strideForArithmetic * uint32_t(baseVertex));
vsharp.qword0 = (uint64_t(fmt) << 44) | (address & 0xfffffffffff);
vsharp.word2 = min(u32(S+0x294+0x20*j), 0x00ffffff);
vsharp.word3 = entry.word3;
```
Evidence: `0xeac441–0xeac453`, `0xeabf00–0xeabfa1`. The multiplication is performed in a 32-bit register (`0xeabf70`); preserve that behavior.
For entries with `a&0xff0000`, reproduce the alternate argument/count path rather than applying base vertex (`0xeabf3b–0xeabf56`).
**Index state and arguments**
At drawIndex entry:
```cpp
G          = rdi;
indexCount = uint32_t(esi);
firstIndex = uint32_t(edx);
baseVertex = uint32_t(ecx);
ib = u32(S+0x690);
indexAddress =
    u64(S+0x688)
    + uint32_t((firstIndex << (1+(ib>>31))) + (ib&0x7fffffff));
```
Evidence: `0xeac43a–0xeac477`.
`0xea3f40` installs `S+0x688=resolvedBase`, `S+0x694=u32(D+0xc)`, `S+0x690=0` when refreshing its index binding (`0xea40b6–0xea40c6`).
The actual emitter separately consumes `u32(G+0x28)&1` (`0xfa0a4e–0xfa0a67`). **The supplied code does not prove that this bit always equals `S+0x690>>31`; capture both.**
Topology is supplied through `0xeabc90 → 0xfa08b0`, which emits `mode&0x3f` to the packet’s register selector `0x242`; its engine cache is `C+0xe74` (`0xea3f62–0xea3f78`, `0xfa08e7–0xfa0946`). `prepass.cpp` identifies its accepted triangle-list value as `4`.
**7. What commit finalizes**
`0xe61ae0(C)` consumes and clears the engine dirty qword at `C+0`, resolves changed bindings, and always tail-calls `0xe61310(C)` (`0xe61b06–0xe61b1c`, `0xe62ebc–0xe62ed0`).
| Commit work | Finalized state / emission | Evidence |
|---|---|---|
| VS handle resolution | `C+0xeb0`, `S+0x100`; sampling defaults from VS object | `0xe61b50–0xe61de7` |
| VS variant resolution | `C+0xeb8`, fetch description `S+0x698/+0x6a0`; optional register emission through `0xfa0af0` | `0xe6206b–0xe6210c` |
| PS handle resolution | `C+0xec0`, `S+0xf0/+0xf8`, usage masks, `G+0xb8/+0xc0` | `0xe6212c–0xe6222d` |
| Sampling-state conversion | Builds/caches descriptors via `0xe9cce0`, installs VS pointers at `S+0x1498`, PS at `S+0x1408` | `0xe628e5–0xe62cf3` |
| Additional shader stages | Resolves `C+0xe86` into `C+0xec8`, prepares/clears additional stages | `0xe626a3–0xe628d6`, `0xe62ee0` |
| Raster-related state | Resolves packed `C+0x40`, caches object at `C+0x830`; emits control and bias-related values | `0xe62d0c–0xe62dd8` |
| Blend-related state | Resolves packed `C+0x50`, caches at `C+0x840`, calls `0xea6d20` | `0xe62de4–0xe62e18` |
| Depth/stencil-related state | Resolves packed `C+0x58`, caches at `C+0x848`, object at `C+0xe78`; calls `0xeae560` | `0xe62e24–0xe62ea6` |
| Constant uploads | Copies dirty upload ranges out of `S′`, resolves current backing allocations, installs CB descriptors into `S` | `0xe61330–0xe61ab9` |
The render-state names in the last three categories are **inferences from their emitted controls**, not recovered symbols.
Concrete render-state values worth capturing:
| State | CPU location / emitted value | Evidence |
|---|---|---|
| Raster control | `R=u64(C+0x830)`; `u32(R+0x10)&0x393fff` emitted to selector `0x205` | `0xe62d35–0xe62d72`, `0xfa20e8–0xfa20fc` |
| Bias-related values | Floats at `R+0x14/+0x18/+0x1c`; first is scaled before emission; selectors `0x2e0/+1`, `0x2e2/+1`, `0x2df` | `0xe62d77–0xe62dd8`, `0xfa2320`, `0xfa2390`, `0xfa2250` |
| Blend source | Packed key `u64(C+0x840)`; resolved source passed to `0xea6d20` | `0xe62de9–0xe62e18` |
| Blend target-zero control | `u32(S+0x2994)` | `0xea6e07–0xea6e15`, `0xea6fa4–0xea6fb1` |
| Depth/stencil source | Resolved object `u64(C+0xe78)`; state payload begins at object `+0x10`; dynamic parameter `C+0x818` | `0xe62e77–0xe62ea6` |
| Two emitted depth/stencil-related words | Mirrors at `S+0x298c`, `S+0x2990` | `0xeae573–0xeae5b5` |
**Complete per-target blend words and every raster/depth register mirror are not established:** the setters below `0xea6d20` and `0xeae560` are not supplied.
**8. Constant-producing helpers**
These additions clarify provenance, but several invoke bodies remain missing:
| Helper | Proven behavior | Implication |
|---|---|---|
| `0x2410` | Compares a 4×4 matrix against builder `B+0x1f0`; transposes it, retaining three vectors at `B+0x230/+0x240/+0x250`; emits `0x22941f0` command with those vectors at `Q+0x20/+0x30/+0x40` | Transform upload provenance proven; destination requires missing `0xe67b00` (`0x2410–0x24e2`). |
| `0x2fc0` | Copies `16*count` bytes inline at `Q+0x20`; descriptor at `Q+8` is `0x36702002190000 | (slot<<4)` | Uses **address point `0x2294230`, outside supplied vtable range**; invoke unavailable (`0x2fc0–0x301a`). |
| `0x3260` | Same copying scheme, descriptor `0x49805c02120000 | (slot<<4)` | Also uses missing `0x2294230` mapping/invoke (`0x3260–0x32ba`). |
| `0x33c0` | Copies `64*count` bytes using the latter descriptor; transposes each 4×4 matrix unless input flag bit zero is set | Same missing invoke (`0x33d8–0x349e`, `0x353c–0x3583`). |
| `0x544ba0` | Resolves IDs through manager `+0x1da8`, selects pointers into builder `+0x660…`, optionally ranks candidates by distance/angular tests, then calls `0x543c60` | **Not proven to be bone-palette preparation** (`0x544bdd–0x544cc5`, `0x544777–0x5448cb`). |
| `0x543c60` | Copies four selected `0x30`-byte records into builder slots `2…5`, marks bits `0x3c` of `B+0x2b8` | Ultimately consumed by `0x39f0`; semantics require `0x3ad0` (`0x543c60–0x543da6`). |
| `0x545520` | Clears those four selected pointers and active records; marks `B+0x2b8|=0x3c`, clears `B+0x658` | Removes those records, not a demonstrated skinning reset (`0x545520–0x545543`). |
**9. Concrete capture recipe**
A useful record can be built now, but a **complete, proven native replay record still requires the missing shader/resource decoders** listed below.
1. **Retain `C` across the lower-level call.** Hook `0xea3f40(M,C,...)` or maintain a thread-local association while `0xe61ae0(C)` executes. At `0xeac430`, only `G` is an argument; no supplied `G→C` back-pointer is established (`0xea3f55`, `0xea40d0–0xea40f3`).
2. **Filter the direct scene-list origin by return address `0x3700`.** It survives `0xea3f40`’s tail-call; `0x373b` is an outer sequence frame (`0x36fb–0x3700`, `0xea40f3`, `0x3739–0x373b`).
3. **At drawIndex entry, capture arguments and engine shader identity:**
   ```cpp
   S = u64(G+0x320);
   vsHandle = u16(C+0xe80);
   psHandle = u16(C+0xe82);
   vsObject = u64(C+0xeb0);
   vsVariant = u64(C+0xeb8);
   psObject = u64(C+0xec0);
   variantKey = u64(C+0xea0);
   psHeader = u64(G+0xc0);
   psMapping = u64(G+0xb8);
   fetch = u64(S+0x698);
   ```
   Evidence: `0xe61b50–0xe61baa`, `0xe6206b–0xe6210c`, `0xe6212c–0xe6222d`, `0xe58055–0xe5805c`.
4. **Enumerate dedicated CB V#s using the usage masks, not dirty masks.** For ordinary PS/VS, inspect bits of `S+0x190/+0x1b0`, and copy the first 16 bytes of each corresponding record at `S+0x1f48/+0x2158 + 0x20*i`. Also preserve its allocation bookkeeping at `+0x10/+0x14` (`0xeac0b7–0xeac13c`, `0xe6156d–0xe61597`, `0xe6166a–0xe61694`).
   **Limitation:** this covers the dedicated CB category. Buffer descriptors reachable through resource pointers or shader-indirect tables require those paths too (`0xeac1c3–0xeac21c`).
5. **Capture used resource and sampler entries.** Use the table in section 4, retaining type bits and copying the pointed-to descriptors while valid. The existing 32-byte texture copy is valid evidence for its known shadow texture case; general descriptor sizes/types remain unresolved without `0xe584d0/0xe58680/0xe58830` (`0xeac1f2–0xeac21c`, `0xeac2cc–0xeac2d4`).
6. **Capture fetch entries and referenced streams; reconstruct attribute V#s with base vertex.** Copy `8*attributeCount` bytes from `u64(F+0x10)`, then the stream records referenced by their stream bytes. This reproduces the work performed inside drawIndex rather than assuming it is already materialized in `S` (`0xeabeca–0xeabfa1`).
7. **Capture index state and the exact computed index address.** Preserve `S+0x688/+0x690/+0x694`, `G+0x28`, count, first and base vertex (`0xeac43a–0xeac479`, `0xfa0a4e–0xfa0a67`).
8. **Capture committed render-state sources/mirrors from section 7.** A snapshot limited to `S[0…0x4000)` misses `S′+0x5480`, and copying `S` alone misses the separate upload state at `S′` (`0x41f0–0x4212`, `0xe613c6–0xe613e3`).
9. **Preserve draw preparation when suppressing submission.** Your current hook continues through original `0xeac430`, which runs `0xeabdb0` and `0xe574c0`. Returning immediately from the entry hook would skip both. Suppression at the final draw submission preserves this preparation (`engine_capture.cpp`, `drawIndex`; `0xeac453`, `0xeac480–0xeac498`).
For asynchronous native execution, copy the selected constant bytes and required resource contents while their execution-time backing allocations are valid; commit explicitly selects allocations using `G+0x328` (`0xe614ff–0xe6151d`, `0xe615fc–0xe6161a`).
**10. Exact remaining gaps**
| Missing code/data | What it prevents proving |
|---|---|
| `0x3ad0`, `0x3f80` | Full semantics and state writes of the eight `0x30`-byte command records (`0x3a86`, `0x3aa9`) |
| `0xe67b00` | Destination of the object-transform command (`0x445a`) |
| Vtable/invoke at address point `0x2294230` | Destinations and stage assignments of `0x2fc0/0x3260/0x33c0` uploads (`0x2fff–0x300b`, `0x329f–0x32ab`) |
| `0xeba8c0`, variant-resolution path below `0xe79920` | Complete current VS binding/header/code relationship (`0xeabfc2`, `0xe7993f`) |
| `0xe583a0`, `0xe584d0`, `0xe58680`, `0xe58830`, `0xe587c0`, `0xe58610` | Complete CB/resource/sampler transfer and mapping semantics (`0xeac13c`, `0xeac1fe`, `0xeac21c`, `0xeac2d4`, `0xeac3a4`, `0xeac3bc`) |
| Shader object/header contents or their constructors | Executable-code and OrbShdr offsets; all shader-read buffer ranges |
| `0xfa1950`, `0xfa19e0` | Final user-data register installation (`0xe57a5c`, `0xe57b32`) |
| `0xe7d6f0`, `0xe916b0`, `0xe90b40`, `0xe90940`, `0xea2a70`, `0xea2ba0` | Remaining invoke side effects, including generated-geometry paths |
For `prepass.cpp`, the offsets in `kOpaqueLoads/kAlphaLoads` are **instruction locations used by `d.vsLoad`**, not offsets into these engine state banks. Mapping those loads to captured V#s and buffer byte ranges still requires the shader instructions or a proven header/user-data decoder (`prepass.cpp`, `Loads`, `Prepass::draw`).
The new code establishes the binding banks and commit sequence, but **does not yet establish a complete shader/user-data decoder**. Several decisive callees are absent from this directory, including `0x3ad0`, `0x3f80`, `0xe67b00`, `0xeba8c0`, and the resource-binding helpers called by `0xeabdb0`.
Two corrections to `reading1.md` matter immediately:
- **`0x3830 → 0x3840` selects packed pipeline/pass state, not a shader pair.** It consumes individual bytes of `Q+0x10`, updating `C+0x40/+0x48/+0x58/+0x60` and calling `0xe84160` (`0x3840–0x38d9`).
- **`S+0x6c8` is not established as a VS constant-buffer table.** It is the resource-pointer bank for physical bank zero. Ordinary PS resources are at `S+0x8e8`; ordinary VS resources are at `S+0xb08`. Dedicated constant-buffer V# records use different banks (`0xeac179–0xeac256`, `0xeac0ac–0xeac179`).
