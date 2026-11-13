#!/usr/bin/env sh
# Download FSR-Vulkan (github.com/FireBurn/FSR-Vulkan, MIT): AMD's FidelityFX SDK 1.1.4 FSR 3.1
# upscaler with its Vulkan backend and shaders already compiled, behind a small C API. Into
# third_party/fsr-vulkan, where the build finds it (LPN_HAVE_FSR). Pinned to a known commit.
set -eu
commit=d38393c2cbaf33dce366bd4ec6927a4951cfbed2
dir="$(dirname "$0")/../third_party/fsr-vulkan"
if [ ! -d "$dir/.git" ]; then
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin https://github.com/FireBurn/FSR-Vulkan.git
fi
git -C "$dir" fetch -q --depth 1 origin "$commit"
git -C "$dir" checkout -q --force FETCH_HEAD
echo "FSR-Vulkan in $dir: re-run cmake on the build directory to use it"
