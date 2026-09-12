#!/bin/bash

# Stage 4: Submit full adversarial post-generation pipeline on HPC.
# This submits dependent jobs in order:
#   analyze -> plot -> compare -> gradient_sim -> epsilon_sweep -> statistics -> cka -> compute_tradeoff -> correlations
# Optional env var:
#   UPSTREAM_HOLD_JID="id1,id2,..." to force the analyze phase to wait for
#   upstream jobs (e.g., all generate jobs) before starting.
# Usage:
#   bash temp4.sh <discovery_epochs> <pretrain_epochs> [model] [dataset] [attack] [kind]
# Examples:
#   bash temp4.sh 100 300
#   bash temp4.sh 100 300 full InceptionNet Cifar10 PGD Dynamic_Region_All_Combined

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument handling – optional <phase> argument.
# ---------------------------------------------------------------------------
if [ "$#" -lt 2 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <discovery_epochs> <pretrain_epochs> [phase] [model] [dataset] [attack] [kind]"
    echo "If <phase> is omitted or set to 'full', the complete dependent pipeline is submitted."
    echo "Examples:"
    echo "  $0 100 300                     # full pipeline (default)"
    echo "  $0 100 300 analyze            # submit only the analyze phase"
    echo "  $0 100 300 plot               # submit only the plot phase"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2
PHASE=${3:-full}
MODEL_FILTER=${4:-ALL}
DATASET_FILTER=${5:-ALL}
ATTACK_FILTER=${6:-ALL}
KIND_FILTER=${7:-ALL}

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
    local upstream_dep=${3:-}
    local -a cmd

    cmd=(qsub -q all.q -l ngpus=1)
    # Use dependency flag compatible with the scheduler.
    # This environment uses SGE, which supports "-hold_jid <jobid>".
    # If you switch to a PBS/Torque scheduler, replace the line below with:
    #   cmd+=( -W depend=afterok:${dep_job} )
    hold_targets=""
    if [ -n "$dep_job" ]; then
        hold_targets="$dep_job"
    fi
    if [ -n "$upstream_dep" ]; then
        if [ -n "$hold_targets" ]; then
            hold_targets="$hold_targets,$upstream_dep"
        else
            hold_targets="$upstream_dep"
        fi
    fi
    if [ -n "$hold_targets" ]; then
        # Use SGE-style dependency flag which is supported on this cluster.
        cmd+=( -hold_jid "$hold_targets" )
    fi
    cmd+=(
        -v "MODEL=$MODEL_FILTER,DATASET=$DATASET_FILTER,ATTACK=$ATTACK_FILTER,KIND=$KIND_FILTER,PHASE=$phase,OUTPUT_DIR=$OUTPUT_DIR,FORCE_RERUN=${FORCE_RERUN:-0}"
        adversarial_hpc_submit.pbs
    )

    local out
    out=$("${cmd[@]}")
    echo "$out" | awk '{print $1}'
}

if [ "$PHASE" = "full" ]; then
    echo "Submitting dependent post-generation pipeline..."

    ANALYZE_JOBID=$(submit_phase analyze "" "${UPSTREAM_HOLD_JID:-}")
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
else
    # Submit a single phase without dependencies.
    JOBID=$(submit_phase "$PHASE")
    echo "Submitted $PHASE job ID: $JOBID"
fi
echo "Monitor with: qstat"
