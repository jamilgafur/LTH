#!/bin/bash

rm *.out

# Define arrays for models, pretrain epochs, early stopping, finetune epochs, and steps
models=("LeNet" "ResNet20" "Vgg16")
pretrain_epochs=(5 20 20)
early_stopping=(3 3 3)
finetune_epochs_LeNet=(1 5 10)
finetune_epochs_rest=(3 5 10)
steps=21

# Loop over models
for i in "${!models[@]}"; do
    model=${models[$i]}
    pretrain_epoch=${pretrain_epochs[$i]}
    es=${early_stopping[$i]}

    # Decide which finetune epochs to use based on the model
    if [ "$model" == "LeNet" ]; then
        finetune_epochs=("${finetune_epochs_LeNet[@]}")
    else
        finetune_epochs=("${finetune_epochs_rest[@]}")
    fi

    # Loop over finetune epochs
    for finetune_epoch in "${finetune_epochs[@]}"; do
        sbatch prune_job.sh "$model" "$pretrain_epoch" "$es" "$finetune_epoch" "$steps"
    done
done