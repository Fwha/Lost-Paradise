#!/bin/sh
# Packs lpn-launcher, lpn-host, aerolib.csv and the DLSS runtime (when built with it) into
# build/Lost_Paradise-x86_64.AppImage. Build first (README), then:
#   scripts/make-appimage.sh [install folder to copy it to]
# The AppImage runs in the folder it is put in: game/CUSA12776, lpn-data, lpn-display.ini and
# play.log live there, as for launch.sh. No game files go into it.
set -eu
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
[ -x "$BUILD/src/host/lpn-host" ] && [ -x "$BUILD/src/launcher/lpn-launcher" ] ||
    { echo "build lpn-host and lpn-launcher first: cmake -S . -B $BUILD -G Ninja && ninja -C $BUILD" >&2; exit 1; }
[ -f aerolib.csv ] || { echo "aerolib.csv missing: scripts/fetch-nids.sh aerolib.csv" >&2; exit 1; }

APPDIR="$BUILD/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/lost-paradise"
cp "$BUILD/src/launcher/lpn-launcher" "$BUILD/src/host/lpn-host" "$APPDIR/usr/bin/"
# NGX looks for the DLSS runtime beside lpn-host.
for f in "$BUILD"/src/host/libnvidia-ngx-dlss.so.*; do
    if [ -e "$f" ]; then cp "$f" "$APPDIR/usr/bin/"; fi
done
cp aerolib.csv "$APPDIR/usr/share/lost-paradise/"
cp packaging/AppRun packaging/lost-paradise.desktop packaging/lost-paradise.svg "$APPDIR/"
ln -sf lost-paradise.svg "$APPDIR/.DirIcon"

TOOL="$BUILD/appimagetool-x86_64.AppImage"
if [ ! -x "$TOOL" ]; then
    curl -fL -o "$TOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi
OUT="$BUILD/Lost_Paradise-x86_64.AppImage"
# --appimage-extract-and-run: works without FUSE too.
ARCH=x86_64 "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$OUT"
echo "made $OUT"
if [ $# -gt 0 ]; then
    cp "$OUT" "$1/"
    echo "copied to $1"
fi
