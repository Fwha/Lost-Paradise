#!/bin/sh
# Packs the Windows build into build/Lost_Paradise-windows.zip: one "Lost Paradise" folder with
# Lost Paradise.exe (the launcher), lpn-host.exe, the MSYS2 DLLs they use, aerolib.csv and, when
# fetched, NVIDIA Streamline's DLLs for DLSS.
# Run in an MSYS2 UCRT64 shell after building (README), from the repo:
#   scripts/make-windows-zip.sh
# The folder is the install folder: game\CUSA12776, lpn-data, lpn-display.ini and play.log go
# in it. No game files go into the zip.
set -eu
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
[ -f "$BUILD/src/host/lpn-host.exe" ] && [ -f "$BUILD/src/launcher/Lost Paradise.exe" ] ||
    { echo "build first: cmake -S . -B $BUILD -G Ninja && ninja -C $BUILD" >&2; exit 1; }
[ -f aerolib.csv ] || { echo "aerolib.csv missing: scripts/fetch-nids.sh aerolib.csv" >&2; exit 1; }

DIR="$BUILD/zip/Lost Paradise"
rm -rf "$BUILD/zip"
mkdir -p "$DIR"
cp "$BUILD/src/launcher/Lost Paradise.exe" "$BUILD/src/host/lpn-host.exe" aerolib.csv "$DIR/"
cp packaging/windows/README.txt "$DIR/"
# DLSS (NVIDIA Streamline, scripts/fetch-streamline.sh): its DLLs go beside lpn-host.exe, with
# their licenses.
if [ -f third_party/streamline/bin/sl.interposer.dll ]; then
    cp third_party/streamline/bin/* "$DIR/"
    cp third_party/streamline/license.txt "$DIR/streamline.license.txt"
fi
# The DLLs from the MSYS2 environment (not Windows' own, nor vulkan-1.dll, which comes with the
# graphics driver).
for exe in "$DIR/Lost Paradise.exe" "$DIR/lpn-host.exe"; do
    ldd "$exe" | awk '{print $3}' | grep -i "^${MINGW_PREFIX:-/ucrt64}/" || true
done | sort -u | while read -r dll; do
    cp "$dll" "$DIR/"
done
ls "$DIR"
OUT="$BUILD/Lost_Paradise-windows.zip"
rm -f "$OUT"
(cd "$BUILD/zip" && zip -qr9 "../$(basename "$OUT")" "Lost Paradise")
echo "made $OUT"
