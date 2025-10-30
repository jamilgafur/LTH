#!/bin/bash

# Define models and datasets
models=("VGG16" "RegNetX_400MF")
datasets=("Cifar10" "Cifar100" )

# Loop through models
for model in "${models[@]}"; do
  # Define the experiment names inside the model loop
  if [[ "$model" == "VGG16" ]]; then
    experiment_names=("Original Model"  "Last 2" "Stage 4" "Stage 3" "Stage 4-5" "Stage 3-5" "Stage 2-5")
  else
    # If model is RegNetX_400MF, define valid experiments only
    experiment_names=(
      "Original Model" "Last 2"
      "Stage 4" "Stage 3"
      "Stage 2" 
      "Stage 4-5" "Stage 3-5" "Stage 2-5" 
      "Stage 4 last 2 conv" "Stage 3 last 2 conv" "Stage 2 last 2 conv" 
      "Stage 4 first 2 conv" "Stage 3 first 2 conv" "Stage 2 first 2 conv"
    )
  fi

  # Loop through datasets
  for dataset in "${datasets[@]}"; do
    # Loop through experiment names
    for experiment in "${experiment_names[@]}"; do
      # Loop through flags
      for flag in "JF" "Kevin"; do
        # Properly quote the variables with spaces and commas
        command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\" submit_job.pbs"
        
        # Echo the command for debugging
        echo "Executing: $command"
        
        # Run the command
        eval "$command"
      done
    done
  done
done
