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

## Inner formats seen

| Magic | Extension | What |
|---|---|---|
| `DDS ` | .dds | Textures (standard DirectDraw Surface) |
| `PIBX` | .pib | Particle effects |
| `GSGT` | .gmt | Animation |
| `CMTP` | .cmt | Camera motion |
| `MEP_` | .mep | Effect placement |
| `hitc` | .hcb | Cloth collision |
