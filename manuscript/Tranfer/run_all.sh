#!/bin/bash

# Define models and datasets
models=( "RegNetX_400MF" "VGG16")
datasets=("Cifar100" "Cifar10")

# Define the experiments for each model and dataset directly as arrays

# VGG16 Experiments
VGG16_Cifar10=("Original Model" "Last 2" "Stage 5" "Stage 4" "Stage 3" "Stage 4-5" "Stage 3-5" "Stage 2-5")
VGG16_Cifar100=("Original Model" "Last 2" "Stage 5" "Stage 4" "Stage 3" "Stage 4-5" "Stage 3-5" "Stage 2-5")
VGG16_Imagenet=("Original Model" "Last 2" "Stage 5" "Stage 4" "Stage 3" "Stage 4-5" "Stage 3-5" "Stage 2-5")
VGG16_Tinyimagenet=("Original Model" "Last 2" "Stage 5" "Stage 4" "Stage 3" "Stage 4-5" "Stage 3-5" "Stage 2-5")

# RegNetX_400MF Experiments
RegNetX_400MF_Cifar10=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 1-4" "Stage 2-4" "Stage 3-4"
"Stage 4 last 2 conv" "Stage 3 last 2 conv" "Stage 2 last 2 conv" "Stage 4 first 2 conv" "Stage 3 first 2 conv" "Stage 2 first 2 conv")

RegNetX_400MF_Cifar100=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 4-5" "Stage 3-5" "Stage 2-5"
"Stage 4 last 2 conv" "Stage 3 last 2 conv" "Stage 2 last 2 conv" "Stage 4 first 2 conv" "Stage 3 first 2 conv" "Stage 2 first 2 conv")

RegNetX_400MF_Imagenet=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 1-4" "Stage 2-4" "Stage 3-4"
"Stage 4 last 2 conv" "Stage 3 last 2 conv" "Stage 2 last 2 conv" "Stage 4 first 2 conv" "Stage 3 first 2 conv" "Stage 2 first 2 conv")

RegNetX_400MF_Tinyimagenet=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 1-4" "Stage 2-4" "Stage 3-4"
"Stage 4 last 2 conv" "Stage 3 last 2 conv" "Stage 2 last 2 conv" "Stage 4 first 2 conv" "Stage 3 first 2 conv" "Stage 2 first 2 conv")

rm ~/submit*
# Loop through models
for model in "${models[@]}"; do
  # Loop through datasets
  for dataset in "${datasets[@]}"; do
    # Construct the experiment key (model and dataset combination)
    experiment_key="${model}_${dataset}"

    # Select the experiment array based on the current model and dataset
    if [[ "$experiment_key" == "VGG16_Cifar10" ]]; then
      experiment_names=("${VGG16_Cifar10[@]}")
    elif [[ "$experiment_key" == "VGG16_Cifar100" ]]; then
      experiment_names=("${VGG16_Cifar100[@]}")
    elif [[ "$experiment_key" == "VGG16_Imagenet" ]]; then
      experiment_names=("${VGG16_Imagenet[@]}")
    elif [[ "$experiment_key" == "VGG16_Tinyimagenet" ]]; then
      experiment_names=("${VGG16_Tinyimagenet[@]}") 
    elif [[ "$experiment_key" == "RegNetX_400MF_Cifar10" ]]; then
      experiment_names=("${RegNetX_400MF_Cifar10[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_Cifar100" ]]; then
      experiment_names=("${RegNetX_400MF_Cifar100[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_Imagenet" ]]; then
      experiment_names=("${RegNetX_400MF_Imagenet[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_Tinyimagenet" ]]; then
      experiment_names=("${RegNetX_400MF_Tinyimagenet[@]}")
    else
      echo "Unknown model-dataset combination: $experiment_key"
      continue
    fi

    # Loop through experiment names
    for experiment in "${experiment_names[@]}"; do
      # Loop through flags
      for flag in "JF" "Kevin"; do
        # Construct the command with appropriate variables
        command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\" submit_job.pbs"
        
        # Echo the command for debugging
        echo "Executing: $command"
        
        # Run the command
        eval "$command"
      done
    done
  done
done
