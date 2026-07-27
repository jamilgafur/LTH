#!/bin/bash

# Stage 4: Submit adversarial transferability analysis, then plotting
# Usage:
#   bash temp4.sh <discovery_epochs> <pretrain_epochs> [model] [dataset] [attack] [kind]
# Examples:
#   bash temp4.sh 100 300
#   bash temp4.sh 100 300 InceptionNet Cifar10 PGD Finetuned

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

cd "$SCRIPT_DIR"

if ! command -v qsub >/dev/null 2>&1; then
    echo "[ERROR] qsub command not found in PATH."
    exit 1
fi

echo "===================================================================="
echo "Stage 4: Adversarial Analyze + Plot"
echo "Discovery Epochs: $EPOCHS"
echo "Pretrain Epochs:  $PRETRAIN"
echo "Output Dir:       $OUTPUT_DIR"
echo "Model Filter:     $MODEL_FILTER"
echo "Dataset Filter:   $DATASET_FILTER"
echo "Attack Filter:    $ATTACK_FILTER"
echo "Kind Filter:      $KIND_FILTER"
echo "===================================================================="

ANALYZE_CMD=(
    qsub -q all.q -l ngpus=1
    -v "MODEL=$MODEL_FILTER,DATASET=$DATASET_FILTER,ATTACK=$ATTACK_FILTER,KIND=$KIND_FILTER,PHASE=analyze,OUTPUT_DIR=$OUTPUT_DIR"
    adversarial_hpc_submit.pbs
)

echo "Submitting analyze job..."
ANALYZE_JOBID=$("${ANALYZE_CMD[@]}")
ANALYZE_JOBID=$(echo "$ANALYZE_JOBID" | awk '{print $1}')

echo "Analyze Job ID: $ANALYZE_JOBID"

echo "Submitting plot job with dependency after analyze success..."
PLOT_CMD=(
    qsub -q all.q -l ngpus=1
    -v "MODEL=$MODEL_FILTER,DATASET=$DATASET_FILTER,ATTACK=$ATTACK_FILTER,KIND=$KIND_FILTER,PHASE=plot,OUTPUT_DIR=$OUTPUT_DIR"
    adversarial_hpc_submit.pbs
)
PLOT_JOBID=$("${PLOT_CMD[@]}")
PLOT_JOBID=$(echo "$PLOT_JOBID" | awk '{print $1}')

echo "Plot Job ID: $PLOT_JOBID"
echo "[DONE] Stage 4 submissions complete (plot waits for analyze). Monitor with: qstat"
