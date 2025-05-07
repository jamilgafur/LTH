#!/bin/bash

rm -f *.out

# Set number of pruning steps
steps=21
early_stopping=3

# model=$1
# pretrain_epochs=$2
# patience=$3
# finetune_epochs=$4
# steps=$5
# strategy=$6
# batch_size=$7  # Optional (only passed for ImageNet models)


# Total epochs = pretrain + finetune = 10
# === CIFAR MODELS ===
# Test 2 models: LeNet and ResNet20
# Strategy: magnitude and brain-damage


# LeNet - Magnitude
sbatch prune_job.sh "LeNet" 0 "$early_stopping" 30 "$steps" "magnitude"  64
# LeNet - Brain-damage
# sbatch prune_job.sh "LeNet" 1 "$early_stopping" 1 "$steps" "brain-damage" 128

# ResNet20 - Magnitude
sbatch prune_job.sh "ResNet20" 10 "$early_stopping" 150 "$steps" "magnitude" 128
# ResNet20 - Brain-damage
# sbatch prune_job.sh "ResNet20" 1 "$early_stopping" 1 "$steps" "brain-damage" 128

# Vgg16 

sbatch prune_job.sh "Vgg16" 10 "$early_stopping" 150 "$steps" "magnitude" 128
# ResNet20 - Brain-damage
# sbatch prune_job.sh "Vgg16" 1 "$early_stopping" 1 "$steps" "brain-damage" 128

# === ImageNet MODELS ===
# Test 2 models: RegNetX and EfficientNet
# Assume batch size is needed for these models
batch_size=32

# Vgg16ImageNet - Magnitude
sbatch prune_job.sh "Vgg16ImageNet" 10 "$early_stopping" 80 "$steps" "magnitude" "$batch_size"
# Vgg16ImageNet - Brain-damage
# sbatch prune_job.sh "Vgg16ImageNet" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

# RegNetX - Magnitude
sbatch prune_job.sh "RegNetX" 10 "$early_stopping" 80 "$steps" "magnitude" "$batch_size"
# RegNetX - Brain-damage
# sbatch prune_job.sh "RegNetX" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

# ResNet50
sbatch prune_job.sh "ResNet50" 10 "$early_stopping" 80 "$steps" "magnitude" "$batch_size"
# ResNet50 - Brain-damage
# sbatch prune_job.sh "ResNet50" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

# EfficientNet - Magnitude
sbatch prune_job.sh "EfficientNet" 10 "$early_stopping" 80 "$steps" "magnitude" "$batch_size"
# EfficientNet - Brain-damage
# sbatch prune_job.sh "EfficientNet" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

