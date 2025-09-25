#!/bin/bash

# Define datasets and experiments for each
declare -A DATASET_EXPERIMENTS
DATASET_EXPERIMENTS["Cifar10"]="Original Model|Last 2|Stage 5|Stage 4|Stage 3|Stage 4-5|Stage 3-5|Stage 2-5"
DATASET_EXPERIMENTS["Cifar100"]="Original Model|Last 2|Stage 5|Stage 4|Stage 4-5"
# DATASET_EXPERIMENTS["TinyImageNet"]="Original Model|All Conv Layers"
# DATASET_EXPERIMENTS["ImageNet"]="Original Model|last 2|Stage 5|Stage 4|All Conv Layers"

for dataset in "${!DATASET_EXPERIMENTS[@]}"; do
    IFS='|' read -ra EXPERIMENTS <<< "${DATASET_EXPERIMENTS[$dataset]}"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "Submitting job for dataset: $dataset | experiment: $exp"
        sbatch run_experiment_job.sh "$dataset" "$exp"
    done
done
