#!/bin/bash
# choices=['cifar10', 'imagenet', 'cifar100', 'stl10', 'caltech101', 'fashionmnist', 'mnist'],
datasets=("cifar10" "cifar100" "stl10" "caltech101" "fashionmnist" "mnist")

for dataset in "${datasets[@]}"; do
  echo "Running experiment for dataset: $dataset"
  sudo docker run --rm --gpus all --runtime=nvidia -v "$(pwd)":/workspace pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    sh -c "python setup.py develop ; cd manuscript/structured_study/ && python3 main_experiment.py --dataset $dataset" 
done
