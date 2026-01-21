#!/bin/bash

# Define models and datasets
models=("VGG16" "RegNetX_400MF")
datasets=("Cifar10" "Cifar100" "tinyimagenet")
epochs=50
break_group=(6)

# Loop through each combination of model and dataset and qsub the job
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for bg in "${break_group[@]}"; do
            command="qsub -q all.q -l ngpus=1 -v MODEL=${model},DATASET=${dataset},EPOCHS=${epochs},BREAK_GROUP=${bg} main_1.pbs"
            echo "Submitting job with command: $command"
            eval "$command"
        done
    done
done
