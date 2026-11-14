#!/usr/bin/env sh
# Windows DLSS: download NVIDIA Streamline (github.com/NVIDIA-RTX/Streamline; headers MIT, the
# signed DLLs and the DLSS runtime under NVIDIA's licenses, see their license files) into
# third_party/streamline, where the build finds it (LPN_HAVE_STREAMLINE). Only its headers and the
# four DLLs DLSS needs are kept: sl.interposer, sl.common, sl.dlss and nvngx_dlss.
set -eu
version=v2.9.0
dir="$(cd "$(dirname "$0")/.." && pwd)/third_party/streamline"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -sSfL -o "$tmp/sl.zip" "https://github.com/NVIDIA-RTX/Streamline/releases/download/$version/streamline-sdk-$version.zip"
rm -rf "$dir"
mkdir -p "$dir/bin"
python3 - "$tmp/sl.zip" "$dir" <<'PY'
import sys, zipfile, os
z, out = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
dlls = {"sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "nvngx_dlss.dll", "nvngx_dlss.license.txt"}
for n in z.namelist():
    if n.endswith("/"):
        continue
    if n.startswith("include/") or n == "license.txt":
        dest = os.path.join(out, n)
    elif n.startswith("bin/x64/") and n.count("/") == 2 and os.path.basename(n) in dlls:
        dest = os.path.join(out, "bin", os.path.basename(n))
    else:
        continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(z.read(n))
PY
echo "Streamline $version in $dir: re-run cmake on the build directory to use it"
