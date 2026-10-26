#!/usr/bin/env sh
# Download NVIDIA's DLSS SDK (headers, libnvsdk_ngx.a and the DLSS runtime) into third_party/dlss,
# where the build finds it. Not redistributed in this repo; see its LICENSE.txt.
set -eu
dir="$(dirname "$0")/../third_party/dlss"
if [ -d "$dir/.git" ]; then
    git -C "$dir" pull --ff-only
else
    git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/DLSS.git "$dir"
fi
git -C "$dir" sparse-checkout set include lib/Linux_x86_64
echo "DLSS SDK in $dir: re-run cmake on the build directory to use it"
