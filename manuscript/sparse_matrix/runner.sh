#!/bin/bash

# Models to evaluate
models=("ResNet20" "Vgg16_" "Vgg16ImageNet")

# Batch sizes
batch_sizes=(1 64)

# Sparsity thresholds (comma-separated list)
thresholds=("0.0" "0.5" "0.7" "1.0")

# Path to the experiment script
experiment_script="experiment.sh"

# Loop through each combination of model and batch size
for model in "${models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        # Loop through each threshold individually
        for threshold in "${thresholds[@]}"; do
            echo "Running experiment for Model: $model, Batch Size: $batch_size, Threshold: $threshold"
            
            # Run the experiment.sh script for one threshold at a time
            sbatch $experiment_script "$model" "$batch_size" "$threshold"
        done
    done
done
