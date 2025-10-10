#!/bin/bash

# Define models and datasets
models=("RegNetX_400MF" "VGG16")
datasets=("Cifar10" "Cifar100" "TinyImageNet")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for experiment in "${experiment_names[@]}"; do
      # If experiment is VGG16, set specific experiment names
      if [[ "$model" == "VGG16" ]]; then
        experiment_names=("Original Model" "Last 2" "Stage 5" "Stage 4" "Stage 3" "Stage 4-5" "Stage 3-5" "Stage 2-5")
      else
        # If model is RegNetX_400MF, add the stage combinations and other experiments
        experiment_names=("Original Model" "Last 2" "Stage 5" "Stage 4" "Stage 3" "Stage 2" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 1-5" \
                          "Stage 5 Last 2" "Stage 4 Last 2" "Stage 3 Last 2" "Stage 2 Last 2" \
                          "Stage 5 First 2" "Stage 4 First 2" "Stage 3 First 2" "Stage 2 First 2")
      fi

      # Loop through flags
      for flag in "JF" "Kevin"; do
        # Properly quote the variables with spaces and commas
        command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\" submit_job.pbs"
        # Echo the command for debugging
        echo "Executing: $command"

        # Run the command
        eval $command
      done
    done
  done
done
