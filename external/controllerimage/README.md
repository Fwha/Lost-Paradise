# ControllerImage art (subset)

Face-button SVGs from https://github.com/icculus/ControllerImage (commit 0460d85),
`art/standard/gamepad/{ps5,xboxone,joyconpair}/{s,e,w,n}.svg`: south, east, west and north.
The art is public domain (DATA-LICENSE.txt, credits.txt); the library itself is zlib
(LICENSE.txt). Only the art is used: the library needs SDL3 and we are on SDL2, so
src/host/pad_glyphs.cpp rasterises these with nanosvg (external/nanosvg, zlib) itself.

After changing a file here, run `python3 scripts/embed-pad-glyphs.py` to regenerate
src/host/pad_glyphs_data.h.
