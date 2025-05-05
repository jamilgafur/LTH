#!/bin/bash

rm -f *.out

# Set number of pruning steps
steps=2
early_stopping=3

# Total epochs = pretrain + finetune = 10
# === CIFAR MODELS ===
# Test 2 models: LeNet and ResNet20
# Strategy: magnitude and brain-damage

# LeNet - Magnitude
sbatch prune_job.sh "LeNet" 1 "$early_stopping" 1 "$steps" "magnitude"  128
# LeNet - Brain-damage
sbatch prune_job.sh "LeNet" 1 "$early_stopping" 1 "$steps" "brain-damage" 128

# ResNet20 - Magnitude
sbatch prune_job.sh "ResNet20" 1 "$early_stopping" 1 "$steps" "magnitude" 128
# ResNet20 - Brain-damage
sbatch prune_job.sh "ResNet20" 1 "$early_stopping" 1 "$steps" "brain-damage" 128

# Vgg16 

sbatch prune_job.sh "Vgg16" 1 "$early_stopping" 1 "$steps" "magnitude" 128
# ResNet20 - Brain-damage
sbatch prune_job.sh "Vgg16" 1 "$early_stopping" 1 "$steps" "brain-damage" 128

# === ImageNet MODELS ===
# Test 2 models: RegNetX and EfficientNet
# Assume batch size is needed for these models
batch_size=32

# Vgg16ImageNet - Magnitude
sbatch prune_job.sh "Vgg16ImageNet" 1 "$early_stopping" 1 "$steps" "magnitude" "$batch_size"
# Vgg16ImageNet - Brain-damage
sbatch prune_job.sh "Vgg16ImageNet" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

# RegNetX - Magnitude
sbatch prune_job.sh "RegNetX" 1 "$early_stopping" 1 "$steps" "magnitude" "$batch_size"
# RegNetX - Brain-damage
sbatch prune_job.sh "RegNetX" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

# ResNet50
sbatch prune_job.sh "ResNet50" 1 "$early_stopping" 1 "$steps" "magnitude" "$batch_size"
# ResNet50 - Brain-damage
sbatch prune_job.sh "ResNet50" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

# EfficientNet - Magnitude
sbatch prune_job.sh "EfficientNet" 1 "$early_stopping" 1 "$steps" "magnitude" "$batch_size"
# EfficientNet - Brain-damage
sbatch prune_job.sh "EfficientNet" 1 "$early_stopping" 1 "$steps" "brain-damage" "$batch_size"

