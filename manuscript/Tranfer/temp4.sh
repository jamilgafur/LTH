#!/bin/bash

# Stage 4: Submit full adversarial post-generation pipeline on HPC.
# This submits dependent jobs in order:
#   analyze -> plot -> compare -> gradient_sim -> epsilon_sweep -> statistics -> cka -> compute_tradeoff -> correlations
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
echo "Stage 4: Adversarial Full Post-Processing Pipeline"
echo "Discovery Epochs: $EPOCHS"
echo "Pretrain Epochs:  $PRETRAIN"
echo "Output Dir:       $OUTPUT_DIR"
echo "Model Filter:     $MODEL_FILTER"
echo "Dataset Filter:   $DATASET_FILTER"
echo "Attack Filter:    $ATTACK_FILTER"
echo "Kind Filter:      $KIND_FILTER"
echo "===================================================================="

submit_phase() {
    local phase=$1
    local dep_job=${2:-}
    local -a cmd

    cmd=(qsub -q all.q -l ngpus=1)
    # Use PBS/Torque dependency flag if supported. Some qsub implementations (e.g., SGE) use -hold_jid instead.
    if [ -n "$dep_job" ]; then
        # Use dependency flag compatible with common qsub implementations.
        # PBS/Torque uses "-W depend=afterok:<jobid>" while SGE uses "-hold_jid <jobid>".
        # We'll try the PBS style first; if the scheduler rejects it, the user can replace the flag manually.
        cmd+=( -W depend=afterok:${dep_job} )
            cmd+=( -hold_jid ${dep_job} )
    fi
    cmd+=(
        -v "MODEL=$MODEL_FILTER,DATASET=$DATASET_FILTER,ATTACK=$ATTACK_FILTER,KIND=$KIND_FILTER,PHASE=$phase,OUTPUT_DIR=$OUTPUT_DIR"
        adversarial_hpc_submit.pbs
    )

    local out
    out=$("${cmd[@]}")
    echo "$out" | awk '{print $1}'
}

echo "Submitting dependent post-generation pipeline..."

ANALYZE_JOBID=$(submit_phase analyze)
echo "Analyze Job ID:          $ANALYZE_JOBID"

PLOT_JOBID=$(submit_phase plot "$ANALYZE_JOBID")
echo "Plot Job ID:             $PLOT_JOBID"

COMPARE_JOBID=$(submit_phase compare "$PLOT_JOBID")
echo "Compare Job ID:          $COMPARE_JOBID"

GRADSIM_JOBID=$(submit_phase gradient_sim "$COMPARE_JOBID")
echo "Gradient Sim Job ID:     $GRADSIM_JOBID"

EPS_SWEEP_JOBID=$(submit_phase epsilon_sweep "$GRADSIM_JOBID")
echo "Epsilon Sweep Job ID:    $EPS_SWEEP_JOBID"

STATS_JOBID=$(submit_phase statistics "$EPS_SWEEP_JOBID")
echo "Statistics Job ID:       $STATS_JOBID"

CKA_JOBID=$(submit_phase cka "$STATS_JOBID")
echo "CKA Job ID:              $CKA_JOBID"

COST_JOBID=$(submit_phase compute_tradeoff "$CKA_JOBID")
echo "Compute Tradeoff Job ID: $COST_JOBID"

CORR_JOBID=$(submit_phase correlations "$COST_JOBID")
echo "Correlations Job ID:     $CORR_JOBID"

echo "[DONE] Full adversarial HPC pipeline submitted with dependencies."
echo "Final Job ID: $CORR_JOBID"
echo "Monitor with: qstat"
