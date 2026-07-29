#!/bin/bash

# Stage 3: Submit adversarial attack generation jobs
# Usage:
#   bash temp3.sh <discovery_epochs> <pretrain_epochs> [phase] [model] [dataset] [attack] [kind]
# Examples:
#   bash temp3.sh 100 300
#   bash temp3.sh 100 300 generate InceptionNet Cifar10 PGD Finetuned
#   bash temp3.sh 100 300 gradient_sim
#   bash temp3.sh 100 300 epsilon_sweep
#   bash temp3.sh 100 300 statistics
#   bash temp3.sh 100 300 cka

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <discovery_epochs> <pretrain_epochs> [phase] [model] [dataset] [attack] [kind]"
    echo "Example (generate all):    $0 100 300"
    echo "Example (generate single): $0 100 300 generate InceptionNet Cifar10 PGD Finetuned"
    echo "Example (gradient sim):    $0 100 300 gradient_sim"
    echo "Example (epsilon sweep):   $0 100 300 epsilon_sweep"
    echo "Example (statistics):      $0 100 300 statistics"
    echo "Example (cka):             $0 100 300 cka"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2
PHASE=${3:-generate}
MODEL_FILTER=${4:-ALL}
DATASET_FILTER=${5:-ALL}
ATTACK_FILTER=${6:-ALL}
KIND_FILTER=${7:-ALL}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="adversarial_results_ep${EPOCHS}_pre${PRETRAIN}"

echo "===================================================================="
echo "Stage 3: Adversarial – $PHASE"
echo "Discovery Epochs: $EPOCHS"
echo "Pretrain Epochs:  $PRETRAIN"
echo "Output Dir:       $OUTPUT_DIR"
echo "Phase:            $PHASE"
echo "Model Filter:     $MODEL_FILTER"
echo "Dataset Filter:   $DATASET_FILTER"
echo "Attack Filter:    $ATTACK_FILTER"
echo "Kind Filter:      $KIND_FILTER"
echo "===================================================================="

cd "$SCRIPT_DIR"

bash ./adversarial_hpc_orchestrate.sh "$PHASE" "$OUTPUT_DIR" "$MODEL_FILTER" "$DATASET_FILTER" "$ATTACK_FILTER" "$KIND_FILTER"

echo "[DONE] Stage 3 ($PHASE) submissions complete. Monitor with: qstat"
