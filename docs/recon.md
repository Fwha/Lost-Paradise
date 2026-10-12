# Lost Paradise (CUSA12776): step 1 recon

## eboot.bin
- Decrypted ELF, 64-bit x86-64, type 0xFE10 (PS4 dynamic executable), 41.3 MB, entry 0xd6b3d0.
- Code segment: 0x0000000-0x2291180 (~34.6 MB). Data/BSS: 0x2398000, 0x51a850 in memory.
- Has a ~470 KB .eh_frame_hdr/.eh_frame, which is useful for recovering function boundaries.
- Stripped, but leaks some source paths (`D:\Project\Mars_e\...\src\{action,minigame,msg,hact}`); the internal codename is "Mars".
- 38 needed modules, **372 imported functions, all named** via the shadPS4 and sce_symbols NID databases.

## Import surface (what the C++ host must implement)
| Library | Funcs |
|---|---|
| libSceGnmDriver | 87 |
| libkernel | 74 |
| libc | 67 |
| libSceVideoOut | 23 |
| libSceGameLiveStreaming | 11 |
| libScePad | 11 |
| libScePlayGo | 8 |
| libSceSaveDataDialog | 8 |
| libSceSaveData | 7 |
| libSceUserService | 7 |
| libSceNpTrophy | 6 |
| libSceAudioOut | 5 |
| libSceImeDialog | 5 |
| libSceMsgDialog | 5 |
| libSceRtc | 5 |
| libSceSystemService | 5 |
| libSceZlib | 5 |
| libSceAppContent | 4 |
| libSceErrorDialog | 4 |
| libSceRudp | 3 |
| libSceScreenShot | 3 |
| libSceSharePlay | 3 |
| libSceSysmodule | 3 |
| libSceHttp | 2 |
| libSceNet | 2 |
| libSceNpWebApi | 2 |
| libSceRemoteplay | 2 |
| libSceSsl | 2 |
| libSceCommonDialog | 1 |
| libSceNpManager | 1 |
| libSceVideoRecording | 1 |

Regenerate the full list with `lpn-inspect --out`.

## Dump layout
- `CUSA12776/`: game (eboot.bin, sce_module/ with libc, Fios2, Face, Hand, Smart, S3DConversion), `data/` with ~620 .par archives, .hca/.cpk audio (CRI), .usm video (CRI), shaders, and stage/hact/motion data. The listing hit its 2000-entry cap, so this count is partial.
- `sys_modules/`: firmware .sprx libraries, which are useful as a reference when reimplementing (e.g. libSceGnmDriver, libkernel).

## Takeaways
- The OS surface is small: GNM (87), kernel (74), libc (67), VideoOut (23). Most of the rest (network, trophies, PSN, streaming) can be stubbed.
- The hard part is graphics. The game builds raw GCN PM4 command buffers and GCN shader binaries, so the host needs a PM4 parser and a GCN-to-SPIR-V translator.
- The asset formats (.par, .gmd, .gmt) follow the Yakuza/RGG Studio family, for which community tools already exist.


## Function boundaries
`.eh_frame` yields 60,499 functions; the entry point (0xd6b3d0, 81 bytes) is among them.
Leaf functions without unwind info won't appear and are left for Ghidra's analysis to find.
