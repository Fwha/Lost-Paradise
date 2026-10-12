# File formats

Verified against sample archives from the CUSA12776 dump (cloth, effect, auth, hact).

## PARC (.par)

Container for nearly all game data. Header fields are big-endian.

| Offset | Type | Meaning |
|---|---|---|
| 0x00 | char[4] | `PARC` |
| 0x04 | u8, u8, u16 | platform (2), endianness (1 = big), padding |
| 0x08 | u32 | version, `0x00020001` |
| 0x0C | u32 | data size (0) |
| 0x10 | u32, u32 | folder count, folder table offset |
| 0x18 | u32, u32 | file count, file table offset |
| 0x20 | char[64] × n | names: all folders, then all files. Folder 0 is the root `.` |

Folder entry (32 bytes): child folder count, first child folder index, file count, first file
index, attributes (0x10), then 12 reserved bytes.

File entry (32 bytes):

| Offset | Meaning |
|---|---|
| 0x00 | flags; bit 31 set means the data is SLLZ-compressed |
| 0x04 | size (decompressed) |
| 0x08 | stored size |
| 0x0C | offset, low 32 bits |
| 0x10 | attributes (0x20) |
| 0x14 | offset, high 32 bits (assumed; always 0 in the samples) |
| 0x18 | u64 timestamp (often 0) |

File data is 0x800-aligned when uncompressed and packed tightly otherwise. Archives nest:
`hact/*.par` and `auth/*.par` hold uncompressed `000.par`, `cmn.par` and so on.

## SLLZ

LZ77 compression for files inside PARC. Header (16 bytes): `SLLZ`, u8 endianness of the header
(0 = little), u8 version (1), u16 header size (0x10), u32 decompressed size, u32 compressed size
including the header.

Version 1 stream: a flag byte covers 8 tokens, most significant bit first. Bit 0 is a literal
byte. Bit 1 is a little-endian u16 `v` back-reference with distance `(v >> 4) + 1` and length
`(v & 0xF) + 3`. The next flag byte comes immediately after the token that used the previous
flag's last bit, before that token's payload.

Only version 1 appears in the samples so far.

## Textures (.dds)

Standard DDS with a DX10 header, BC1 (DXT1) or BC3 (DXT5), full mip chains, linear (not
PS4-tiled). They load directly in normal tools and can go straight to Vulkan as BC textures.
Suffixes: `_di` diffuse, `_tn` normal, `_mt` material/specular, `_am` multi/ambient mask.

## GMD models (GSGM)

Little-endian. Verified on `chara/ps4/adv/c_ca_ken/mesh.par` → `c_ca_ken.gmd` (Kenshiro).

Header: `GSGM` at 0, the model name at 0x10 (u16 checksum + 30 chars), then (offset, count)
pairs from 0x30:

| # | Table | Entry |
|---|---|---|
| 0 | nodes | 0x80 |
| 1 | objects | 0x40 |
| 2 | meshes | 0x40 |
| 3 | attributes (material bindings) | 0x80 |
| 4 | materials | 0x10 |
| 5 | matrices | |
| 6 | vertex buffer layouts | 0x20 |
| 7 | vertex data | bytes |
| 8 | texture names | 0x20 |
| 9 | shader names | 0x20 |
| 10 | node names | 0x20 |
| 11 | index data | u16 count |
| 12 | object draw lists | bytes |
| 13 | mesh matrix lists | bytes |

After these come 8 floats of model bounds.

**Node:** index, first child, next sibling, object, matrix, stack op, name index, type; then
vec4 position, quaternion rotation, vec4 scale, vec4 world position, vec4 anim axis, 4 flags.
Parents are implied by the child/sibling links. Mesh objects are root nodes named `[lN]name`,
N being the level of detail (`[l0]` full detail, `[l3]` lowest).

**Vertex buffer:** index, vertex count, u64 packing flags, data offset, data size, stride, pad.
Layout from the flags (inferred; the three samples' strides all check out):

| Bits | Field | Bytes |
|---|---|---|
| 0-2 = 3 | position, float32 xyz | 12 |
| 6-7 | bone weights u8×4, then bone indices u8×4 | 8 |
| 8-12 | normal, u8×4 mapped to [-1, 1] | 4 |
| 13-15 | tangent, u8×4 | 4 |
| 21-23 | unknown u8×4 | 4 |
| 24-26 | unknown u8×4 | 4 |
| 27-28 | uv0, float16×2 | 4 |

**Mesh:** index, attribute, vertex buffer, vertex count, triangle list (count, start), two
triangle-strip variants of the same geometry (count, start each), matrix list (count, byte
offset), node, object, unknown, first vertex. Indices are u16 and absolute within the vertex
buffer, so subtract the first vertex. A matrix list is a u8 count followed by that many u8
matrix indices; vertex bone indices refer to it.

**Attribute:** index, material, shader, 2 unknown, flags, 2 unknown, then 8 texture indices
(0xFFFF = unused). Slot 0 is diffuse, 1 multi (`_am`), 2 material (`_mt`), 5 normal (`_tn`).

## Inner formats seen

| Magic | Extension | What |
|---|---|---|
| `DDS ` | .dds | Textures (standard DirectDraw Surface) |
| `PIBX` | .pib | Particle effects |
| `GSGM` | .gmd | Models |
| `GSGT` | .gmt | Animation |
| `CMTP` | .cmt | Camera motion |
| `MEP_` | .mep | Effect placement |
| `hitc` | .hcb | Cloth collision |
