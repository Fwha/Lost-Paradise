#!/usr/bin/env sh
# Download a public PS4 NID -> name database (not redistributed in this repo).
set -eu
out="${1:-aerolib.csv}"
curl -sSfL -o "$out" https://raw.githubusercontent.com/zecoxao/sce_symbols/main/aerolib.csv
echo "wrote $out ($(wc -l < "$out") entries)"
