#!/bin/bash

# Stage 3: Submit adversarial attack generation jobs
# Usage:
#   bash temp3.sh <discovery_epochs> <pretrain_epochs> [model] [dataset] [attack] [kind]
# Examples:
#   bash temp3.sh 100 300
#   bash temp3.sh 100 300 InceptionNet Cifar10 PGD Finetuned

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <discovery_epochs> <pretrain_epochs> [model] [dataset] [attack] [kind]"
    echo "Example (all):     $0 100 300"
    echo "Example (single):  $0 100 300 InceptionNet Cifar10 PGD Finetuned"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2
MODEL_FILTER=${3:-ALL}
DATASET_FILTER=${4:-ALL}
ATTACK_FILTER=${5:-ALL}
KIND_FILTER=${6:-ALL}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="adversarial_results_ep${EPOCHS}_pre${PRETRAIN}"

echo "===================================================================="
echo "Stage 3: Adversarial Generate"
echo "Discovery Epochs: $EPOCHS"
echo "Pretrain Epochs:  $PRETRAIN"
echo "Output Dir:       $OUTPUT_DIR"
echo "Model Filter:     $MODEL_FILTER"
echo "Dataset Filter:   $DATASET_FILTER"
echo "Attack Filter:    $ATTACK_FILTER"
echo "Kind Filter:      $KIND_FILTER"
echo "===================================================================="

cd "$SCRIPT_DIR"

bash ./adversarial_hpc_orchestrate.sh generate "$OUTPUT_DIR" "$MODEL_FILTER" "$DATASET_FILTER" "$ATTACK_FILTER" "$KIND_FILTER"

echo "[DONE] Stage 3 submissions complete. Monitor with: qstat"
