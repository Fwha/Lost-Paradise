# One gameplay frame, pass by pass

From the pass map of frame 5913 (Fwha's play, 2560x1440, town, 323 passes, ~8,000 draws). The
"@" column of a pass map is the engine code behind each draw (eboot offsets of the return
addresses above the Gnm draw wrapper, innermost first: `native/draw_origin.h`); the outermost of
the three is given here. Shader names are archive entries (`native/shader_names.h`).

| Passes | Targets | Draws | What (shaders) | Engine code |
|---|---|---|---|---|
| 1-3 | depth 1024² x3 | ~510 | sun shadow cascades: `vs_shadowmap_normal[_pt]` / `ps_shadowmap[_pt][fp32][bias]` (`_pt`: alpha-tested) | 0x55fefb (via 0xa51fc3, 0xa50abd), 0x5634bc |
| 4 | 128x64 | 2 | stage shadow factor `ps_calc_in_stg_sdw_3` + compute | 0x5634bc |
| 5-8 | depth 1024² x4 | ~50 | more shadow maps (characters, near) | 0x4dc83c, 0x4c7af8, 0x9e9a8c |
| 9 | rt 2560 + depth | ~2,740 | depth and normal prepass: `ps_normal`, `ps_normal_mask_n` | 0x373b, 0xa9b (draw-command lists) |
| 11, 21 | rt + depth | ~260 | character shadow receiving `ps_sdw_cas3_char1[_pt]`, hair normals | 0x4f4bb0, 0x4dc801 |
| 12-17 | 1280x768 | 6 | SSAO at half size, temporal filter | 0x56369b |
| 23-263 | rt0+rt1 + depth | ~3,500 | forward material pass (`r_o1dst_m(vr)2ds`, `r_b1dst`, ...), then hair (`sm_b1ht[hair]`), particles (`ps_ptc_*`), interleaved with depth-only passes | 0x373b, 0xbdb, 0x55fefb, 0x4dc801, 0x8fd63d (particles) |
| 262-264 | rt + depth | ~180 | translucent / reflections (`r_o1dut`, `r_o1d[ag]`) | 0x55fefb, 0xa8615c |
| 266-293 | 256² - 512x256 | ~30 | star glare and haze (`fx_star_glare`, `fx_haze_pass0`) | 0x3216ab |
| 294-303 | 1024x512 - 128x64 | ~11 | glow: luminance, 9 blur levels, combine (`ps_glow_pass0_lumi`, `_pass1_ca`, `_pass2_5`) | 0x316459 |
| 307-309 | 2560 + stencil | 3 | MLAA (native since stage 0) | 0xd68be |
| 311-322 | 2560 | ~55 | hand-drawn ink finish, toon edge FXAA, dither, haze | 0x31694d, 0x55fefb, 0x569c34 |
| 323 | display buffer | 1 | `ps_render_fb` to the display | 0x7d26b5 |

There is no deferred lighting: a normal/depth prepass feeds SSAO and the material shaders
light forward. 0x55fefb (inside 0x55fb20) is the frame's render loop that most passes come back
to.
