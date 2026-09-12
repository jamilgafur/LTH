#!/bin/bash

# Stage 3: Submit adversarial pipeline jobs
# Usage:
#   bash temp3.sh <discovery_epochs> <pretrain_epochs> [phase] [model] [dataset] [attack] [kind]
# Examples:
#   bash temp3.sh 100 300
#   bash temp3.sh 100 300 generate InceptionNet Cifar10 PGD Dynamic_Region_All_Combined
#   bash temp3.sh 100 300 compare
#   bash temp3.sh 100 300 gradient_sim
#   bash temp3.sh 100 300 epsilon_sweep
#   bash temp3.sh 100 300 statistics
#   bash temp3.sh 100 300 cka
#   bash temp3.sh 100 300 compute_tradeoff
#   bash temp3.sh 100 300 correlations

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
if [ "$#" -lt 2 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <discovery_epochs> <pretrain_epochs> [phase] [model] [dataset] [attack] [kind]"
    echo "If <phase> is omitted or set to 'full', all stages are run sequentially."
    echo "Examples:"
    echo "  $0 100 300                     # run full pipeline"
    echo "  $0 100 300 generate            # run only generate phase"
    echo "  $0 100 300 analyze            # run only analyze phase"
    echo "  $0 100 300 compute_tradeoff    # run only compute_tradeoff phase"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2
# Default to "full" which triggers all phases in order.
PHASE=${3:-full}
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
# If the user requested the full pipeline, iterate over all phases in the desired order.
if [ "$PHASE" = "full" ]; then
    echo "--- Submitting phase: generate ---"
    bash ./adversarial_hpc_orchestrate.sh generate "$OUTPUT_DIR" "$MODEL_FILTER" "$DATASET_FILTER" "$ATTACK_FILTER" "$KIND_FILTER"

    # The generate phase writes submitted qsub IDs here; use them so analyze
    # waits for all generate jobs before starting the dependent post phases.
    GEN_IDS_FILE="$OUTPUT_DIR/generate_job_ids.txt"
    if [ ! -f "$GEN_IDS_FILE" ] || [ ! -s "$GEN_IDS_FILE" ]; then
        echo "[WARN] Generate job-id file missing or empty: $GEN_IDS_FILE"
        echo "[WARN] Submitting post phases without explicit generate dependency."
        bash ./temp4.sh "$EPOCHS" "$PRETRAIN" full "$MODEL_FILTER" "$DATASET_FILTER" "$ATTACK_FILTER" "$KIND_FILTER"
    else
        HOLD_IDS=$(paste -sd, "$GEN_IDS_FILE")
        echo "--- Submitting dependent post-processing phases (held on generate jobs) ---"
        UPSTREAM_HOLD_JID="$HOLD_IDS" \
            bash ./temp4.sh "$EPOCHS" "$PRETRAIN" full "$MODEL_FILTER" "$DATASET_FILTER" "$ATTACK_FILTER" "$KIND_FILTER"
    fi
else
    bash ./adversarial_hpc_orchestrate.sh "$PHASE" "$OUTPUT_DIR" "$MODEL_FILTER" "$DATASET_FILTER" "$ATTACK_FILTER" "$KIND_FILTER"
fi

echo "[DONE] Stage 3 ($PHASE) submissions complete. Monitor with: qstat"
