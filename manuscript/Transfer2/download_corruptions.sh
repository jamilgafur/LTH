#!/usr/bin/env bash
# Stage corruption benchmark data for Transfer2 --analysis full.
# RUN ON A LOGIN NODE (compute nodes have no internet access).
#
# Usage: ./download_corruptions.sh [target_dir]   (default: ./corruption_data)
#
# Then pass:  python main.py ... --analysis full --corruptions-dir <target_dir>
set -euo pipefail

TARGET_DIR="${1:-./corruption_data}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

fetch () {
    local url="$1" out="$2"
    if [ -d "${out%.tar}" ]; then
        echo "[skip] ${out%.tar} already extracted"
        return
    fi
    echo "[fetch] $url"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$out" "$url"
    else
        curl -L -o "$out" "$url"
    fi
    tar xf "$out"
    rm -f "$out"
}

# Hendrycks & Dietterich corruption benchmark (Zenodo)
fetch https://zenodo.org/records/2535556/files/CIFAR-10-C.tar   CIFAR-10-C.tar
fetch https://zenodo.org/records/3555552/files/CIFAR-100-C.tar  CIFAR-100-C.tar
# Uncomment if TinyImageNet-C support is added:
# fetch https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar Tiny-ImageNet-C.tar

echo "[done] Corruption data staged in $(pwd)"
