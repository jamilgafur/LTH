#!/bin/bash
# Stage CIFAR-10-C + CIFAR-100-C locally (Zenodo). Compute nodes have no
# internet: run this ONCE on a login node, then point submissions at the dir:
#   ./download_corruptions.sh ./corruption_data
#   CORRUPTIONS_DIR=$PWD/corruption_data ./run_postprocess.sh
set -euo pipefail
DEST="${1:-./corruption_data}"
mkdir -p "$DEST"

# Zenodo records for the Hendrycks & Dietterich corruption benchmarks.
for entry in "2535967 CIFAR-10-C" "3555552 CIFAR-100-C"; do
    read -r record name <<< "$entry"
    if [ -d "$DEST/$name" ]; then
        echo "[skip] $name already staged in $DEST"
        continue
    fi
    echo "[get] $name (zenodo $record)..."
    curl -fL --retry 3 -o "$DEST/$name.tar" \
        "https://zenodo.org/records/$record/files/$name.tar?download=1"
    tar -xf "$DEST/$name.tar" -C "$DEST"
    rm -f "$DEST/$name.tar"
    echo "[ok] $DEST/$name"
done
echo "Done. Use: CORRUPTIONS_DIR=$PWD/$DEST ./run_postprocess.sh"
