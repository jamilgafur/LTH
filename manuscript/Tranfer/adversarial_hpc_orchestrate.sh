#!/bin/bash

set -u
set -o pipefail

# ============================================================================
# Adversarial Analysis HPC Orchestration Script
# ============================================================================
#
# This script submits parallelized adversarial analysis jobs to an HPC cluster.
#
# Usage:
#   ./adversarial_hpc_orchestrate.sh <phase> [output_dir] [model] [dataset] [attack] [kind]
#
# Examples:
#   ./adversarial_hpc_orchestrate.sh generate adversarial_results
#   ./adversarial_hpc_orchestrate.sh generate adv_single InceptionNet Cifar10 PGD Finetuned
#   ./adversarial_hpc_orchestrate.sh analyze adversarial_results
#   ./adversarial_hpc_orchestrate.sh plot adversarial_results
#

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <phase> [output_dir] [model] [dataset] [attack] [kind]"
    echo ""
    echo "Phases:"
    echo "  generate  - Parallelize attack generation across models/datasets/attacks"
    echo "  analyze   - Compute transferability (single job; requires generated attacks)"
    echo "  plot      - Generate visualizations (single job; requires summary.csv)"
    echo ""
    echo "Optional source filters (for generate/analyze):"
    echo "  model   = VGG16 | RegNetX_400MF | InceptionNet | MobileNet | XceptionNet | ConvNeXt"
    echo "  dataset = Cifar10 | Cifar100"
    echo "  attack  = PGD | FGSM | IFGSM | BIM | APGD | CW | DeepFool"
    echo "  kind    = Original | Finetuned"
    echo ""
    echo "Examples:"
    echo "  $0 generate adversarial_results"
    echo "  $0 generate adv_single InceptionNet Cifar10 PGD Finetuned"
    echo "  $0 analyze  adv_single InceptionNet Cifar10 PGD Finetuned"
    echo "  $0 analyze adversarial_results"
    echo "  $0 plot adversarial_results"
    exit 1
fi

PHASE=$1
OUTPUT_DIR=${2:-adversarial_results}
MODEL_FILTER=${3:-ALL}
DATASET_FILTER=${4:-ALL}
ATTACK_FILTER=${5:-ALL}
KIND_FILTER=${6:-ALL}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/adversarial_orchestrate_${PHASE}_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$RUN_LOG"
}

fail() {
    log "[ERROR] $*"
    log "[ERROR] Aborting. See log file: $RUN_LOG"
    exit 1
}

log "Starting orchestration script"
log "Script path: $SCRIPT_DIR"
log "Current working directory: $(pwd)"
log "Phase: $PHASE"
log "Output directory: $OUTPUT_DIR"
log "Model filter: $MODEL_FILTER"
log "Dataset filter: $DATASET_FILTER"
log "Attack filter: $ATTACK_FILTER"
log "Kind filter: $KIND_FILTER"
log "Run log: $RUN_LOG"

if ! command -v qsub >/dev/null 2>&1; then
    fail "qsub command not found in PATH. Ensure you are on the HPC login node and scheduler tools are loaded."
fi
log "qsub command found: $(command -v qsub)"

cd "$SCRIPT_DIR" || fail "Failed to cd into script directory: $SCRIPT_DIR"

if [ ! -f "adversarial_hpc_submit.pbs" ]; then
    fail "Required file not found: $SCRIPT_DIR/adversarial_hpc_submit.pbs"
fi
log "Found submission file: $SCRIPT_DIR/adversarial_hpc_submit.pbs"

# Models and datasets
all_models=("VGG16" "RegNetX_400MF" "InceptionNet" "MobileNet" "XceptionNet" "ConvNeXt")
all_datasets=("Cifar10" "Cifar100")
all_attacks=("PGD" "FGSM" "IFGSM" "BIM" "APGD" "CW" "DeepFool")

models=("${all_models[@]}")
datasets=("${all_datasets[@]}")
attacks=("${all_attacks[@]}")

if [ "$MODEL_FILTER" != "ALL" ]; then
    models=("$MODEL_FILTER")
fi

if [ "$DATASET_FILTER" != "ALL" ]; then
    datasets=("$DATASET_FILTER")
fi

if [ "$ATTACK_FILTER" != "ALL" ]; then
    attacks=("$ATTACK_FILTER")
fi

log "============================================================================"
log "Adversarial Analysis HPC Orchestration"
log "============================================================================"
log "Phase:       $PHASE"
log "Output Dir:  $OUTPUT_DIR"
log "Models:      ${#models[@]} architectures"
log "Datasets:    ${#datasets[@]} datasets"
if [ "$PHASE" = "generate" ]; then
    log "Attacks:     ${#attacks[@]} attack types"
    log "Total Jobs:  ~$((${#models[@]} * ${#datasets[@]} * ${#attacks[@]}))"
fi
log "============================================================================"

mkdir -p logs

log "Run outputs will be written as txt files in manuscript/Tranfer/"
log "  - generate: <MODEL>_<DATASET>_<ATTACK>_generate_run.txt"
log "  - analyze:  adversarial_analyze_run.txt"
log "  - plot:     adversarial_plot_run.txt"

submit_and_log() {
    local cmd="$1"
    local label="$2"
    log "[SUBMIT] $label"
    log "[CMD] $cmd"
    local out
    out=$(eval "$cmd" 2>&1)
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "[FAIL] $label"
        log "[FAIL] qsub exit code: $rc"
        log "[FAIL] qsub output: $out"
        return $rc
    fi
    log "[OK] $label"
    log "[OK] qsub output: $out"
    return 0
}

case "$PHASE" in
    generate)
        log "[PHASE: GENERATE] Submitting attack generation jobs..."
        job_count=0
        for model in "${models[@]}"; do
            for dataset in "${datasets[@]}"; do
                for attack in "${attacks[@]}"; do
                    cmd="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",ATTACK=\"$attack\",KIND=\"$KIND_FILTER\",PHASE=\"generate\",OUTPUT_DIR=\"$OUTPUT_DIR\" adversarial_hpc_submit.pbs </dev/null"
                    submit_and_log "$cmd" "$model/$dataset/$attack/$KIND_FILTER" || fail "Submission failed for $model/$dataset/$attack/$KIND_FILTER"
                    ((job_count++))
                    # Optional: Add delay to avoid overwhelming scheduler
                    sleep 0.1
                done
            done
        done
        log "[SUCCESS] Submitted $job_count jobs"
        log "Monitor progress with: qstat"
        ;;
    
    analyze)
        log "[PHASE: ANALYZE] Submitting transferability analysis job..."
        log "WARNING: This requires all attacks from --generate to be completed first."
        log "Waiting for generate jobs to complete is recommended."
        cmd="qsub -q all.q -l ngpus=1 -v MODEL=\"$MODEL_FILTER\",DATASET=\"$DATASET_FILTER\",ATTACK=\"$ATTACK_FILTER\",KIND=\"$KIND_FILTER\",PHASE=\"analyze\",OUTPUT_DIR=\"$OUTPUT_DIR\" adversarial_hpc_submit.pbs </dev/null"
        submit_and_log "$cmd" "analyze" || fail "Failed to submit analyze phase"
        log "[SUCCESS] Transferability analysis job submitted"
        ;;
    
    plot)
        log "[PHASE: PLOT] Submitting visualization job..."
        log "WARNING: This requires summary.csv and transferability.csv to exist."
        cmd="qsub -q all.q -l ngpus=1 -v MODEL=\"$MODEL_FILTER\",DATASET=\"$DATASET_FILTER\",ATTACK=\"$ATTACK_FILTER\",KIND=\"$KIND_FILTER\",PHASE=\"plot\",OUTPUT_DIR=\"$OUTPUT_DIR\" adversarial_hpc_submit.pbs </dev/null"
        submit_and_log "$cmd" "plot" || fail "Failed to submit plot phase"
        log "[SUCCESS] Visualization job submitted"
        ;;
    
    *)
        log "ERROR: Unknown phase '$PHASE'"
        log "Valid phases: generate, analyze, plot"
        exit 1
        ;;
esac

    log "============================================================================"
    log "Job submission complete. Check logs/ for details."
    log "Run log saved to: $RUN_LOG"
    log "============================================================================"
