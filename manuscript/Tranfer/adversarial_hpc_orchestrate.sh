#!/bin/bash

# ============================================================================
# Adversarial Analysis HPC Orchestration Script
# ============================================================================
#
# This script submits parallelized adversarial analysis jobs to an HPC cluster.
#
# Usage:
#   ./adversarial_hpc_submit.sh <phase> [output_dir]
#
# Examples:
#   ./adversarial_hpc_submit.sh generate adversarial_results
#   ./adversarial_hpc_submit.sh analyze adversarial_results
#   ./adversarial_hpc_submit.sh plot adversarial_results
#

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <phase> [output_dir]"
    echo ""
    echo "Phases:"
    echo "  generate  - Parallelize attack generation across models/datasets/attacks"
    echo "  analyze   - Compute transferability (single job; requires generated attacks)"
    echo "  plot      - Generate visualizations (single job; requires summary.csv)"
    echo ""
    echo "Examples:"
    echo "  $0 generate adversarial_results"
    echo "  $0 analyze adversarial_results"
    echo "  $0 plot adversarial_results"
    exit 1
fi

PHASE=$1
OUTPUT_DIR=${2:-adversarial_results}

# Models and datasets
models=("VGG16" "RegNetX_400MF" "InceptionNet" "MobileNet" "XceptionNet" "ConvNeXt")
datasets=("Cifar10" "Cifar100")
attacks=("PGD" "FGSM" "IFGSM" "BIM" "APGD" "CW" "DeepFool")

echo "============================================================================"
echo "Adversarial Analysis HPC Orchestration"
echo "============================================================================"
echo "Phase:       $PHASE"
echo "Output Dir:  $OUTPUT_DIR"
echo "Models:      ${#models[@]} architectures"
echo "Datasets:    ${#datasets[@]} datasets"
if [ "$PHASE" = "generate" ]; then
    echo "Attacks:     ${#attacks[@]} attack types"
    echo "Total Jobs:  ~$((${#models[@]} * ${#datasets[@]} * ${#attacks[@]}))"
fi
echo "============================================================================"

# Create logs directory
mkdir -p logs

case "$PHASE" in
    generate)
        echo "[PHASE: GENERATE] Submitting attack generation jobs..."
        job_count=0
        for model in "${models[@]}"; do
            for dataset in "${datasets[@]}"; do
                for attack in "${attacks[@]}"; do
                    cmd="qsub -v MODEL=\"$model\",DATASET=\"$dataset\",ATTACK=\"$attack\",PHASE=\"generate\",OUTPUT_DIR=\"$OUTPUT_DIR\" adversarial_hpc_submit.pbs"
                    echo "[SUBMIT] $model/$dataset/$attack"
                    eval "$cmd"
                    ((job_count++))
                    # Optional: Add delay to avoid overwhelming scheduler
                    sleep 0.1
                done
            done
        done
        echo "[SUCCESS] Submitted $job_count jobs"
        echo "Monitor progress with: qstat"
        ;;
    
    analyze)
        echo "[PHASE: ANALYZE] Submitting transferability analysis job..."
        echo "WARNING: This requires all attacks from --generate to be completed first."
        echo "Waiting for generate jobs to complete is recommended."
        cmd="qsub -v MODEL=\"NONE\",DATASET=\"NONE\",ATTACK=\"NONE\",PHASE=\"analyze\",OUTPUT_DIR=\"$OUTPUT_DIR\" adversarial_hpc_submit.pbs"
        eval "$cmd"
        echo "[SUCCESS] Transferability analysis job submitted"
        ;;
    
    plot)
        echo "[PHASE: PLOT] Submitting visualization job..."
        echo "WARNING: This requires summary.csv and transferability.csv to exist."
        cmd="qsub -v MODEL=\"NONE\",DATASET=\"NONE\",ATTACK=\"NONE\",PHASE=\"plot\",OUTPUT_DIR=\"$OUTPUT_DIR\" adversarial_hpc_submit.pbs"
        eval "$cmd"
        echo "[SUCCESS] Visualization job submitted"
        ;;
    
    *)
        echo "ERROR: Unknown phase '$PHASE'"
        echo "Valid phases: generate, analyze, plot"
        exit 1
        ;;
esac

echo "============================================================================"
echo "Job submission complete. Check logs/ for details."
echo "============================================================================"
