#!/bin/bash

# Stage 5: Submit a single adversarial post-processing phase on HPC.
# This is useful to rerun one phase or to continue from a specific job dependency.
# Usage:
#   bash temp5.sh <discovery_epochs> <pretrain_epochs> <phase> [model] [dataset] [attack] [kind] [depend_jobid]
# Examples:
#   bash temp5.sh 100 300 plot
#   bash temp5.sh 100 300 compute_tradeoff ALL Cifar10 ALL ALL
#   bash temp5.sh 100 300 correlations ALL ALL ALL ALL 123456.server

set -euo pipefail

# Allow the <phase> argument to be optional – default to "full" (run all phases)
if [ "$#" -lt 2 ] || [ "$#" -gt 9 ]; then
    echo "Usage: $0 <discovery_epochs> <pretrain_epochs> [phase] [model] [dataset] [attack] [kind] [depend_jobid]"
    echo "  phase defaults to 'full' (run the complete pipeline)"
    echo "Examples:"
    echo "  $0 100 300               # runs full pipeline"
    echo "  $0 100 300 plot          # runs only the plot phase"
    echo "  $0 100 300 compute_tradeoff ALL Cifar10 ALL ALL"
    echo "  $0 100 300 correlations ALL ALL ALL ALL 123456.server"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2
# If the third argument is missing, default to "full"
PHASE=${3:-full}
MODEL_FILTER=${4:-ALL}
DATASET_FILTER=${5:-ALL}
ATTACK_FILTER=${6:-ALL}
KIND_FILTER=${7:-ALL}
DEPEND_JOBID=${8:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="adversarial_results_ep${EPOCHS}_pre${PRETRAIN}"

cd "$SCRIPT_DIR"

# The script will later check for qsub and fall back to local execution if unavailable.

echo "===================================================================="
echo "Stage 5: Adversarial Single-Phase Submission"
echo "Discovery Epochs: $EPOCHS"
echo "Pretrain Epochs:  $PRETRAIN"
echo "Output Dir:       $OUTPUT_DIR"
echo "Phase:            $PHASE"
echo "Model Filter:     $MODEL_FILTER"
echo "Dataset Filter:   $DATASET_FILTER"
echo "Attack Filter:    $ATTACK_FILTER"
echo "Kind Filter:      $KIND_FILTER"
if [ -n "$DEPEND_JOBID" ]; then
    echo "Depends On:       $DEPEND_JOBID"
fi
echo "===================================================================="

case "$PHASE" in
    full|analyze|plot|compare|gradient_sim|epsilon_sweep|statistics|cka|compute_tradeoff|correlations)
        ;;
    *)
        echo "[ERROR] Unsupported phase: $PHASE"
        echo "Valid phases: full, analyze, plot, compare, gradient_sim, epsilon_sweep, statistics, cka, compute_tradeoff, correlations"
        exit 1
        ;;
esac

echo "Submitting $PHASE job..."

# If qsub is not available (e.g., running on a local workstation), fall back to a direct Python call.
if ! command -v qsub >/dev/null 2>&1; then
    echo "[WARN] qsub not found – falling back to local Python execution."
    PY_CMD=(python -m pyPrune.adversarial.analysis --mode "$PHASE" --output-dir "$OUTPUT_DIR")
    # Apply optional filters only when they are not the default "ALL"
    if [ "$MODEL_FILTER" != "ALL" ]; then PY_CMD+=(--model "$MODEL_FILTER"); fi
    if [ "$DATASET_FILTER" != "ALL" ]; then PY_CMD+=(--dataset "$DATASET_FILTER"); fi
    if [ "$ATTACK_FILTER" != "ALL" ]; then PY_CMD+=(--attack "$ATTACK_FILTER"); fi
    if [ "$KIND_FILTER" != "ALL" ]; then PY_CMD+=(--kind "$KIND_FILTER"); fi
    echo "Running: ${PY_CMD[@]}"
    "${PY_CMD[@]}"
    EXIT_CODE=$?
    echo "[DONE] Local execution finished with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

CMD=(qsub -q all.q -l ngpus=1)
if [ -n "$DEPEND_JOBID" ]; then
    CMD+=( -W "depend=afterok:${DEPEND_JOBID}" )
fi
CMD+=(
    -v "MODEL=$MODEL_FILTER,DATASET=$DATASET_FILTER,ATTACK=$ATTACK_FILTER,KIND=$KIND_FILTER,PHASE=$PHASE,OUTPUT_DIR=$OUTPUT_DIR"
    adversarial_hpc_submit.pbs
)

JOBID=$("${CMD[@]}")
JOBID=$(echo "$JOBID" | awk '{print $1}')

echo "$PHASE Job ID: $JOBID"
echo "[DONE] Submission complete. Monitor with: qstat"
