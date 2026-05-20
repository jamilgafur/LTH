#!/bin/bash

# --- Input Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <discovery_epochs> <hpc_compute_epochs>"
    echo "Example: $0 50 100"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2

# 1. Clean up previous submissions
rm ~/submit* 2>/dev/null

models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("tinyimagenet" "Cifar10")
quant=("False")

echo "=== Submitting Stage 1: Discovery Jobs ==="
echo "    Discovery Budget (Epochs): $EPOCHS | HPC Target (Pretrain): $PRETRAIN"

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for quant_flag in "${quant[@]}"; do
            for flag in "JF"; do
                
                # Submit discovery job. Will train Control if needed, then dump JSON
                command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"discover\",FLAG=\"$flag\",QUANT=\"$quant_flag\",EPOCHS=\"$EPOCHS\",PRETRAIN=\"$PRETRAIN\" submit_job.pbs"
                
                echo "Executing: $command"
                eval "$command"
                
            done
        done
    done
done