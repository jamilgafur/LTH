#!/bin/bash

# Define models and datasets
models=("ConvNeXt" "VGG16" "RegNetX_400MF")
datasets=( "Cifar10" "Cifar100" "tinyimagenet" )

# Define the experiments for each model and dataset directly as arrays

# VGG16 Experiments
VGG16_Cifar10=("Original Model" "Last 2" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" )
VGG16_Cifar100=("Original Model" "Last 2" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" )
VGG16_Imagenet=("Original Model" "Last 2" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" )
VGG16_Tinyimagenet=("Original Model" "Last 2" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" )

# RegNetX_400MF Experiments
RegNetX_400MF_Cifar10=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 3-4" "Stage 2-4" "Stage 1-4" )
RegNetX_400MF_Cifar100=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 3-4" "Stage 2-4" "Stage 1-4" )
RegNetX_400MF_Imagenet=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 3-4" "Stage 2-4" "Stage 1-4" )
RegNetX_400MF_Tinyimagenet=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 2" "Stage 1" "Stage 3-4" "Stage 2-4" "Stage 1-4" )

# ConvNeXt Experiments
ConvNeXt_Cifar10=("Original Model" "Stage 4" "Stage 3" "Stage 2" "Stage 1")
ConvNeXt_Cifar100=("Original Model" "Stage 4" "Stage 3" "Stage 2" "Stage 1")
ConvNeXt_Imagenet=("Original Model" "Stage 4" "Stage 3" "Stage 2" "Stage 1")
ConvNeXt_Tinyimagenet=("Original Model" "Stage 4" "Stage 3" "Stage 2" "Stage 1")

# InceptionNet Experiments
InceptionNet_Cifar10=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
InceptionNet_Cifar100=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
InceptionNet_Imagenet=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
InceptionNet_Tinyimagenet=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )

# XceptionNet Experiments
XceptionNet_Cifar10=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
XceptionNet_Cifar100=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
XceptionNet_Imagenet=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
XceptionNet_Tinyimagenet=("Original Model" "Stage 5" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 4" "Stage 3" "Stage 2" "Last 2" )
# MobileNet Experiments
MobileNet_Cifar10=("Original Model" "Stage 7" "Stage 6-7" "Stage 5-7" "Stage 4-7" "Stage 3-7" "Stage 2-7" "Stage 1-7" "Stage 6" "Stage 5" "Stage 4" "Stage 3" "Stage 2" "Stage 1" )
MobileNet_Cifar100=("Original Model" "Stage 7" "Stage 6-7" "Stage 5-7" "Stage 4-7" "Stage 3-7" "Stage 2-7" "Stage 1-7" "Stage 6" "Stage 5" "Stage 4" "Stage 3" "Stage 2" "Stage 1" )
MobileNet_Imagenet=("Original Model" "Stage 7" "Stage 6-7" "Stage 5-7" "Stage 4-7" "Stage 3-7" "Stage 2-7" "Stage 1-7" "Stage 6" "Stage 5" "Stage 4" "Stage 3" "Stage 2" "Stage 1" )
MobileNet_Tinyimagenet=("Original Model" "Stage 7" "Stage 6-7" "Stage 5-7" "Stage 4-7" "Stage 3-7" "Stage 2-7" "Stage 1-7" "Stage 6" "Stage 5" "Stage 4" "Stage 3" "Stage 2" "Stage 1" )

# Quant = list of true and false
quant=("False")
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
    elif [[ "$experiment_key" == "ConvNeXt_Cifar10" ]]; then
      experiment_names=("${ConvNeXt_Cifar10[@]}")
    elif [[ "$experiment_key" == "ConvNeXt_Cifar100" ]]; then
      experiment_names=("${ConvNeXt_Cifar100[@]}")
    elif [[ "$experiment_key" == "ConvNeXt_imagenet" ]]; then
      experiment_names=("${ConvNeXt_Imagenet[@]}")
    elif [[ "$experiment_key" == "ConvNeXt_tinyimagenet" ]]; then
      experiment_names=("${ConvNeXt_Tinyimagenet[@]}")
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
          command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\",QUANT=\"$quant_flag\" submit_job.pbs"
          
          # Echo the command for debugging
          echo "Executing: $command"
          
          # Run the command
          eval "$command"
        done
      done
    done
  done
done
