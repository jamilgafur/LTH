#!/bin/bash

rm *.out

# Model and hyperparameter configuration
models=("LeNet" "ResNet20" "Vgg16")
pretrain_epochs_list=(5 10 20 50)
finetune_epochs_LeNet=(1 5 10)
early_stopping=3
steps=21

# Loop over each model
for model in "${models[@]}"; do
    for pretrain_epoch in "${pretrain_epochs_list[@]}"; do

        # Set finetune epochs based on model type
        if [ "$model" == "LeNet" ]; then
            finetune_epochs=("${finetune_epochs_LeNet[@]}")
        else
            finetune_epoch=$((160 - pretrain_epoch))
            if [ "$finetune_epoch" -lt 0 ]; then
                # Skip invalid fine-tune settings
                continue
            fi
            finetune_epochs=("$finetune_epoch")
        fi

        # Submit job for each finetune epoch
        for finetune_epoch in "${finetune_epochs[@]}"; do
            sbatch prune_job.sh "$model" "$pretrain_epoch" "$early_stopping" "$finetune_epoch" "$steps"
        done

    done
done
