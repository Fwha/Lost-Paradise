# How the engine fills each shader's user data: a reading

Codex's third reading (2026-11-24): the shader metadata that places constant buffers, resources
and samplers in user-data SGPRs or in the descriptor table, and how to rebuild it at drawIndex.

The new routines establish a metadata-driven decoder. **CB masks select slots; they do not determine SGPR order.** Each slot’s metadata entry independently selects either direct user-data registers or a dword offset in a shared descriptor table.
The remaining limitation is concrete: neither `probe_results.txt` nor the supplied listings contain the **values** of `vs_normal_rigid`’s mapping metadata. Consequently, the decoder below is recoverable, but the particular permutation of CB slots `{9,10,12,13}` cannot be proven from mask `0x3600` alone.
Addresses below are x86 addresses unless explicitly identified as shader PCs. `u8/u16/u32/u64` denote guest-memory reads.
**Shader records and metadata**
Keep these pointer domains separate:
| Quantity | Ordinary PS | Ordinary VS |
|---|---|---|
| CPU shader record `B` | `u64(S+0xf0)` | `u64(S+0x100)` |
| Program/header pointer `P` | `u64(B+0x10)` | `u64(B+0x10)` |
| Mapping variant `v` | `(u32(B+0x1c)>>4)&0xffff` | Same |
| Mapping metadata `M` | `P+4*v` | Same |
| Installed metadata | `u64(G+0xb8)` | `u64(G+0xf8)` |
| Installed program/header | `u64(G+0xc0)` | `u64(G+0x100)` |
The PS reconstruction follows `0xe621a5–0xe621cd` and installation at `0xe58055–0xe5805c`. The ordinary VS reconstruction follows `0xeba8d9–0xeba93b`, with installation through `0xeba9ed → 0xe57fc0`, specifically `0xe58010–0xe58025`.
The probe’s `ptr:vs_normal_rigid.vso` means that reading the first qword **at** `B` produces a recognized code pointer: see `probeNames()` in `engine_capture.cpp`. Thus `B`, its first-qword code address, `P`, and `M` are distinct objects.
**At `drawIndex` entry, reconstruct VS metadata from `S`; do not blindly trust `G+0xf8`.** `0xeac453` has yet to call `0xeabdb0`, whose `0xeba8c0` call can install the current VS metadata. PS installation already occurs in commit at `0xe6222d`. (`0xeac430–0xeac480`, `0xeabfb1–0xeabfc2`.)
**The slot-to-destination encoding**
All the descriptor setters implement:
```cpp
void place(uint16_t m, Descriptor d) {
    if (m == 0xffff)
        return;
    unsigned destination = m & 0x3fff;
    if (m & 0x8000)
        userData[destination ...] = d.dwords;
    else
        rootTable[destination ...] = d.dwords;
}
```
`destination` is a **dword index**, not a descriptor index or byte offset. Bit `0x4000` does not affect placement in these setters: they mask with `0x3fff` and branch on the sign bit. (`0xe583bf–0xe58401`, `0xe58571–0xe58600`, `0xe58721–0xe587b0`, `0xe587df–0xe58821`, `0xe5884f–0xe58891`.)
For indirect placement, the actual CPU destination is:
```cpp
u64(G+0x68) + 0x1800*logicalStage + 4*destination
```
The stage multiplier is derived by `stage<<9`, multiplying that result by three, then scaling by four. (`0xe583dc–0xe583f1`; equivalent arithmetic in the other setters.)
For direct placement, `0xfa19e0` writes the supplied descriptor dwords into a register-setting packet. Its destination register address is:
```cpp
u16(0x2080760 + 2*logicalStage) + destination
```
The packet stores that address minus `0x2c00`. (`0xfa1a2a–0xfa1a74`.) `0xfa1950` uses the same addressing calculation and writes a pointer’s low and high dwords. (`0xfa1992–0xfa19c7`.)
**Hardware interpretation:** these stage-relative user-data destinations are `s[destination]`, with successive descriptor words occupying successive SGPRs. The contents of the register-base lookup table at `0x2080760` are absent from the supplied files; the code nevertheless exposes the exact stage-relative index and the formula for checking its absolute hardware register address.
**Descriptor sources and mapping fields**
Let `l` be the logical stage and `b` its physical binding bank. For ordinary PS, `l=b=1`; for ordinary VS, `l=b=2`.
| Class | Mapping entry | Descriptor source | Descriptor size |
|---|---|---|---:|
| Constant buffer `i` | `u16(M+0xdc+2*i)` | Inline at `S+0x1d38+0x210*b+0x20*i` | 4 dwords |
| Ordinary resource `i` | `u16(M+0x1c+2*i)` | At `u64(S+0x6c8+0x220*b+8*i)` | 4 or 8 dwords |
| Sampler `i` | `u16(M+0xbc+2*i)` | At `u64(S+0x1378+0x90*b+8*i)` | 4 dwords |
| Additional resource `i` | `u16(M+0x9c+2*i)` | At the pointer `u64(S+0x16d8+0x110*b+0x10*i)` | 8 or 4 dwords |
| Generated vertex attribute `i` | `u16(M+0x104+2*i)` | Generated attribute V# array | 4 dwords |
Evidence:
- CB source and setter: `0xeac110–0xeac13c`, `0xe583a0–0xe58401`.
- Ordinary resources: `0xeac1c3–0xeac21c`, `0xe584d0`, `0xe58680`.
- Samplers: `0xeac2b0–0xeac2d4`, `0xe58830`.
- Additional resources: `0xeac35a–0xeac3bc`, `0xe587c0`, `0xe58610`.
- Attributes: `0xeabef0–0xeabfa4`, `0xeabfcc–0xeabfe1`, `0xe58460–0xe584af`.
The ordinary-resource type bit is:
```cpp
(u64(S+0x6a8+0x220*b) >> i) & 1
```
Zero selects the 16-byte buffer-descriptor helper `0xe584d0`; one selects the 32-byte texture-descriptor helper `0xe58680`. (`0xeac1c3–0xeac21c`.)
For the additional class:
```cpp
(u32(S+0x16c8+0x110*b) >> i) & 1
```
Zero selects **32 bytes**, through `0xe587c0`; one selects **16 bytes**, through `0xe58610`. Do not apply the ordinary-resource polarity to this class. (`0xeac35a–0xeac3bc`.)
The ordinary-resource helpers can modify descriptor format before placement. When `0xf9e030` or `0xfa6d10` returns `0x6d`, the helper copies the descriptor and changes the format using argument `0x10` to `0xf9dcb0` or `0xfa6cd0`. An exact decoder must reproduce those bit operations. (`0xe5852a–0xe5855a`, `0xe586da–0xe5870a`; bit operations in the four named callees.)
The ordinary-stage expansions are:
| Source | PS | VS |
|---|---|---|
| CB V# | `S+0x1f48+0x20*i` | `S+0x2158+0x20*i` |
| Ordinary resource pointer | `S+0x8e8+8*i` | `S+0xb08+8*i` |
| Sampler pointer | `S+0x1408+8*i` | `S+0x1498+8*i` |
| Additional-resource record | `S+0x17e8+0x10*i` | `S+0x18f8+0x10*i` |
These are arithmetic expansions of the source instructions above. A CB record’s descriptor is its **first 16 bytes**; record `+0x10/+0x14` are allocation bookkeeping, as shown by `0xe6156d–0xe61597` and `0xe6166a–0xe61694`.
**Usage, dirty state, and physical-bank remapping**
The usage masks are indexed by **logical stage**:
```cpp
resourceUsage = u64(S+0x160+0x20*l);
extraUsage    = u32(S+0x168+0x20*l);
samplerUsage  = u32(S+0x16c+0x20*l);
cbUsage       = u32(S+0x170+0x20*l);
```
The source bank is separately selected before the descriptor accesses. (`0xeac0a2–0xeac0e6`, `0xeac179–0xeac1a2`, `0xeac262–0xeac281`, `0xeac31e–0xeac339`.)
Reproduce the remapping exactly:
```cpp
unsigned physicalBank(unsigned l, uint32_t active) {
    bool stage3 = active & (1u << 3);
    bool stage5 = active & (1u << 5);
    if (!stage5 && stage3 && l == 4) return 2;
    if ( stage5 && l == 6)          return 2;
    if ( stage5 && !stage3 && l==2) return 4;
    return l;
}
```
Here `active=u32(S+0xdc)`; the code enumerates its set bits as logical stages. (`0xeac01d–0xeac0a2`.)
The actual engine processes `usage & dirty`, then clears the processed dirty bits. For reconstruction, enumerate **all used bindings**, because a clean direct descriptor may exist only in inherited GPU register state, while its current source descriptor remains in `S`. (`0xeac0e3–0xeac185`, `0xeac19f–0xeac25f`, `0xeac27e–0xeac31c`, `0xeac336–0xeac3eb`.)
**How the descriptor table is built**
There is one staging region per logical stage:
```cpp
stageStaging = u64(G+0x68) + 0x1800*l;
N            = u16(M);                    // dwords
```
Indirect descriptor setters scatter descriptors into this region at their metadata-selected dword offsets. When preparing the draw, the engine copies the region to transient memory beginning at `u64(G+0x60)` and advances that allocation pointer by `4*N`. The copy loop transfers `ceil(N/8)*32` bytes when `N!=0`. (`0xe57793–0xe577dc` for VS; `0xe579ef–0xe57a3c` for PS.)
**The table is not constructed by concatenating used CBs.** It is constructed by metadata-directed writes into the staging region, followed by a copy.
Furthermore, several SGPR pairs can point into different regions of the same copied allocation. Let `R` be its base. When `u8(M+3)&2`, preparation emits these values; each destination byte `0xff` means absent:
| Metadata destination byte | Value written to that SGPR pair |
|---|---|
| `M+7` | `R + 4*u16(M+0x12)` |
| `M+8` | `R + 4*u16(M+0x16)` |
| `M+9` | `R + 4*u16(M+0x18)` |
| `M+0xa` | `R + 4*u16(M+0x1a)` |
| `M+0xb` | `u64(G+0x70)` |
| `M+6` | `R` |
VS evidence: `0xe577de–0xe578ae`. PS evidence: `0xe57a3e–0xe57b04`. Pointer low/high dwords are emitted by `0xfa19c3–0xfa19c7`.
Therefore, **do not assume `s[12:13]` points at `R`.** Find which row assigns destination `12`.
For example, if:
```cpp
u8(M+7) == 12;
o = u16(M+0x12);
```
then:
```cpp
s[12:13] == R + 4*o;
VSV[k]   == bytes at rootTable + 4*o + k;
```
A CB whose mapping entry is indirect with low bits `d` appears at:
```cpp
VSV[4*(d-o)]
```
This subtraction is essential when the shader’s pointer addresses a table subregion.
Other user-data writes are:
| Condition | Destination | Value | Evidence |
|---|---|---|---|
| `M[3]&2`, `M[0xd]!=0xff` | One SGPR at `M[0xd]` | `u32(G+0x94+0x40*l)` | VS `0xe578b3–0xe578d6`; PS `0xe57b10–0xe57b32` |
| VS `M[3]&4`, `M[4]!=0xff` | Pair at `M[4]` | `u64(G+0x108)` | `0xe57947–0xe57961` |
| VS `M[3]&4`, `M[5]!=0xff` | Pair at `M[5]` | `R+4*u16(M+0x14)` | `0xe57966–0xe57985` |
| VS `M[0xe]!=0xff` | One SGPR at `M[0xe]` | `u16(G+0x2c8)*u16(G+0x2ca)` | `0xe5798a–0xe579c8` |
The VS `M[3]&4` path also allocates/copies the root table if the `&2` path has not done so. (`0xe578e9–0xe57945`.)
The `G+0x108` value is installed from the fetch-binding argument forwarded through `0xeabfb5 → 0xeba9ed → 0xe5801e`. Its interpretation as callable fetch code is consistent with shader PCs `0x0–0x8`, where `vs_normal_rigid` calls through `s[0:1]`; assigning it specifically to that pair still requires `M[4]==0`.
One further descriptor write deserves special handling:
```cpp
if (u16(P+4) != 0)
    place(u16(M+0xfa), bytes16(M+0x144));
```
It occurs before the table copy, with the corresponding stage preparation flags set. (`0xe57678–0xe576e9` for VS; `0xe57701–0xe57764` for PS.) `M+0xfa` is also `M+0xdc+2*15`: this special path uses the CB-slot-15 mapping entry but sources the descriptor from **metadata `+0x144`**.
**A concrete reconstruction algorithm**
The following reconstructs the effective descriptor mapping without reading PM4. Retain descriptor provenance alongside every written dword.
```cpp
decodeStage(G, S, l, M, P):
    b = physicalBank(l, u32(S+0xdc))
    // Preserve unassigned/padding bytes, if a byte-exact table is wanted.
    root = copy(u64(G+0x68) + 0x1800*l, 4*u16(M))
    sgpr = unknown_dwords()
    place(m, descriptor, provenance):
        if m == 0xffff: return
        d = m & 0x3fff
        if m & 0x8000:
            write sgpr[d .. d+descriptor.dwordCount)
        else:
            write root[d .. d+descriptor.dwordCount)
        attach provenance
    for i in setBits(cbUsage(l)):
        place(u16(M+0xdc+2*i),
              bytes16(S+0x1d38+0x210*b+0x20*i),
              CB(b,i))
    for i in setBits(resourceUsage(l)):
        p = u64(S+0x6c8+0x220*b+8*i)
        texture = bit(u64(S+0x6a8+0x220*b), i)
        d = texture ? bytes32(p) : bytes16(p)
        apply the format conversion from e58680/e584d0
        place(u16(M+0x1c+2*i), d, Resource(b,i))
    for i in setBits(samplerUsage(l)):
        p = u64(S+0x1378+0x90*b+8*i)
        place(u16(M+0xbc+2*i), bytes16(p), Sampler(b,i))
    for i in setBits(extraUsage(l)):
        p = u64(S+0x16d8+0x110*b+0x10*i)
        buffer = bit(u32(S+0x16c8+0x110*b), i)
        d = buffer ? bytes16(p) : bytes32(p)
        place(u16(M+0x9c+2*i), d, ExtraResource(b,i))
    // Include generated attribute descriptors for the stage returned
    // by eba8c0, using M+0x104+2*i.
    if u16(P+4) != 0:
        place(u16(M+0xfa), bytes16(M+0x144), MetadataDescriptor)
    // Assign symbolic pointers into root using the pointer table above.
    // Apply pointer/scalar writes in e574c0 order.
```
The binding-class order is `CB → resources → samplers → additional resources`, following `0xeac0ac–0xeac3eb`. Generated attribute placement precedes those loops. (`0xeabfcc–0xeac00d`.) Preserve actual write order if malformed or unusual metadata creates overlapping destinations.
For `s[12:13]`, retain a symbolic pointer such as `{root, byteOffset}`. A subsequent shader `s_load_dwordx4` reads four dwords from that symbolic table address; those four words are the V# used by the later `s_buffer_load`. This resolves both direct `s[4:7]` descriptors and indirect `VSV[k]` descriptors without needing the transient allocation’s literal GPU address.
For attributes, reproduce `0xeabef0–0xeabfa4`: its source is the fetch layout at `u64(S+0x698)`, layout entries at `u64(F+0x10)`, and stream records beginning at `S+0x288` with stride `0x20`. The generated V# incorporates the draw’s base-vertex adjustment before placement. At `drawIndex`, the third draw argument is forwarded as that adjustment through `0xeac43a–0xeac453`.
This reconstructs **assigned descriptor/pointer/scalar locations**. Registers never assigned by the selected metadata remain unknown/inherited; these files do not provide a CPU mirror of every literal GPU SGPR.
**Other logical stages**
The common descriptor setters obtain metadata using:
```cpp
M = u64(G+0x78+0x40*l);
```
This gives `G+0xb8`, `+0xf8`, `+0x138`, `+0x178`, `+0x1b8`, and `+0x1f8` for stages `1…6`. (`0xe583b4–0xe583bf` and the corresponding instructions in each setter.)
The additional preparation routines use the same root-table and pointer scheme:
| Logical stage | Metadata field | Staging displacement | Evidence |
|---:|---|---:|---|
| 3 | `G+0x138` | `0x4800` | `0xe5730e–0xe57491` |
| 4 | `G+0x178` | `0x6000` | `0xe570e4–0xe572ff` |
| 5 | `G+0x1b8` | `0x7800` | `0xe56d6c–0xe56ec6` |
| 6 | `G+0x1f8` | `0x9000` | `0xe56b60–0xe56d5d` |
Stages 4 and 6 have the additional `M[3]&4` pointer path corresponding to the ordinary VS path; stage 3 also emits the `M[0xe]` product scalar. The cited ranges show their precise sources.
**Worked example: `vs_normal_rigid`, mask `0x3600`**
The mask gives exactly:
```text
CB slots = {9, 10, 12, 13}
```
For ordinary VS bank 2, the four descriptor sources and mapping words are:
| CB slot | V# record | Mapping word |
|---:|---|---|
| 9 | `S+0x2278` | `u16(M+0xee)` |
| 10 | `S+0x2298` | `u16(M+0xf0)` |
| 12 | `S+0x22d8` | `u16(M+0xf4)` |
| 13 | `S+0x22f8` | `u16(M+0xf6)` |
These follow directly from `S+0x2158+0x20*i` and `M+0xdc+2*i`, established at `0xeac129–0xeac13c` and `0xe583bf`.
The shader proves these uses:
| Shader location | Descriptor use |
|---|---|
| `s[4:7]` | Load at buffer offset `0x170`, shader PC `0x10` |
| `s[8:11]` | Loads at offsets `0` and `0x20`, shader PCs `0x14/0x18` |
| `s[12:13]+0` | View-like descriptor: camera at `0x220`, matrices at `0/0x40/0x60`; shader PCs `0x98–0xa0`, `0xb4`, `0xe0–0xe8` |
| `s[12:13]+0x10` | Flags descriptor, loaded at shader PCs `0xc/0x20` |
To identify each CB **exactly**, read the four mapping words above:
```cpp
slotForS4 = i whose mapping has
    (m & 0x8000) && (m & 0x3fff) == 4;
slotForS8 = i whose mapping has
    (m & 0x8000) && (m & 0x3fff) == 8;
```
Then determine the root-relative dword origin `o` of the pointer assigned to `s[12:13]`:
```cpp
slotForTable0 = i whose mapping is indirect and
    (m & 0x3fff) == o;
slotForTable10 = i whose mapping is indirect and
    (m & 0x3fff) == o+4;
```
For example, **if** `M[6]==12` and the four words are:
```text
M+0xee = 0x8004
M+0xf0 = 0x8008
M+0xf4 = 0x0000
M+0xf6 = 0x0004
```
then the result is:
```text
s[4:7]       = CB9
s[8:11]      = CB10
table+0x00   = CB12
table+0x10   = CB13
