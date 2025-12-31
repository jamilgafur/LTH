#!/bin/bash

# Define models and datasets
models=("InceptionNet" "XceptionNet" "MobileNet" "VGG16" "RegNetX_400MF" )
datasets=( "Cifar10" "Cifar100" )

# Define the experiments for each model and dataset directly as arrays

# VGG16 Experiments
VGG16_Cifar10=("Original Model", "Last 2", "Stage 5", "Stage 4-5", "Stage 3-5" )
VGG16_Cifar100=("Original Model", "Last 2", "Stage 5", "Stage 4-5", "Stage 3-5" )
VGG16_Imagenet=("Original Model")
VGG16_Tinyimagenet=("Original Model")

# RegNetX_400MF Experiments
RegNetX_400MF_Cifar10=("Original Model", "Last 2", "Stage 4", "Stage 3", "Stage 2", "Stage 1" )

RegNetX_400MF_Cifar100=("Original Model", "Last 2", "Stage 4", "Stage 3", "Stage 2", "Stage 1" )

RegNetX_400MF_Imagenet=("Original Model")

RegNetX_400MF_Tinyimagenet=("Original Model" )

# InceptionNet Experiments
InceptionNet_Cifar10=("Original Model" )
InceptionNet_Cifar100=("Original Model" )
InceptionNet_Imagenet=("Original Model" )
InceptionNet_Tinyimagenet=("Original Model" )

# XceptionNet Experiments
XceptionNet_Cifar10=("Original Model" )
XceptionNet_Cifar100=("Original Model" )
XceptionNet_Imagenet=("Original Model" )
XceptionNet_Tinyimagenet=("Original Model" )
# MobileNet Experiments
MobileNet_Cifar10=("Original Model" )
MobileNet_Cifar100=("Original Model" )
MobileNet_Imagenet=("Original Model" )
MobileNet_Tinyimagenet=("Original Model" )

# Quant = list of true and false
quant=("False", "True")
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
    elif [[ "$experiment_key" == "VGG16_imagenet" ]]; then
      experiment_names=("${VGG16_Imagenet[@]}")
    elif [[ "$experiment_key" == "VGG16_tinyimagenet" ]]; then
      experiment_names=("${VGG16_Tinyimagenet[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_Cifar10" ]]; then
      experiment_names=("${RegNetX_400MF_Cifar10[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_Cifar100" ]]; then
      experiment_names=("${RegNetX_400MF_Cifar100[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_imagenet" ]]; then
      experiment_names=("${RegNetX_400MF_Imagenet[@]}")
    elif [[ "$experiment_key" == "RegNetX_400MF_tinyimagenet" ]]; then
      experiment_names=("${RegNetX_400MF_Tinyimagenet[@]}")
    elif [[ "$experiment_key" == "InceptionNet_Cifar10" ]]; then
      experiment_names=("${InceptionNet_Cifar10[@]}")
    elif [[ "$experiment_key" == "InceptionNet_Cifar100" ]]; then
      experiment_names=("${InceptionNet_Cifar100[@]}")
    elif [[ "$experiment_key" == "InceptionNet_imagenet" ]]; then
      experiment_names=("${InceptionNet_Imagenet[@]}")
    elif [[ "$experiment_key" == "InceptionNet_tinyimagenet" ]]; then
      experiment_names=("${InceptionNet_Tinyimagenet[@]}")
    elif [[ "$experiment_key" == "XceptionNet_Cifar10" ]]; then
      experiment_names=("${XceptionNet_Cifar10[@]}")
    elif [[ "$experiment_key" == "XceptionNet_Cifar100" ]]; then
      experiment_names=("${XceptionNet_Cifar100[@]}")
    elif [[ "$experiment_key" == "XceptionNet_imagenet" ]]; then
      experiment_names=("${XceptionNet_Imagenet[@]}")
    elif [[ "$experiment_key" == "XceptionNet_tinyimagenet" ]]; then
      experiment_names=("${XceptionNet_Tinyimagenet[@]}")
    elif [[ "$experiment_key" == "MobileNet_Cifar10" ]]; then
      experiment_names=("${MobileNet_Cifar10[@]}")
    elif [[ "$experiment_key" == "MobileNet_Cifar100" ]]; then
      experiment_names=("${MobileNet_Cifar100[@]}")
    elif [[ "$experiment_key" == "MobileNet_imagenet" ]]; then
      experiment_names=("${MobileNet_Imagenet[@]}")
    elif [[ "$experiment_key" == "MobileNet_tinyimagenet" ]]; then
      experiment_names=("${MobileNet_Tinyimagenet[@]}")
    else
      echo "Unknown model-dataset combination: $experiment_key"
      continue
    fi

    # Loop through experiment names
    for experiment in "${experiment_names[@]}"; do
      # Loop through quant values (True/False)
      for quant_flag in "${quant[@]}"; do
        # Loop through flags
        for flag in "JF" "Kevin"; do
          # Construct the command with appropriate variables
          command="qsub -q UI-GPU -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\",QUANT=\"$quant_flag\" submit_job.pbs"
          
          # Echo the command for debugging
          echo "Executing: $command"
          
          # Run the command
          eval "$command"
        done
      done
    done
  done
done
