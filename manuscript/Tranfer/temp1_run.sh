#!/bin/bash
# 1. Clean up previous submissions
rm ~/submit* 2>/dev/null

models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("Cifar10" "tinyimagenet" "Cifar100")
quant=("False")

echo "=== Submitting Stage 1: Discovery Jobs ==="

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for quant_flag in "${quant[@]}"; do
            for flag in "JF"; do
                
                # Submit discovery job. Will train if needed, then dump JSON
                command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"discover\",FLAG=\"$flag\",QUANT=\"$quant_flag\" submit_job.pbs"
                
                echo "Executing: $command"
                eval "$command"
                
            done
        done
    done
done

echo "Done. Wait for these jobs to finish and generate the JSON files before running Stage 2."
