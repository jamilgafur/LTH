#!/bin/bash

rm -f *.out

# Strategies to test
strategy=("magnitude" "brain-damage")

# Model lists
models_cifar=("LeNet" "ResNet20" "Vgg16")
models_imagenet=("RegNetX" "EfficientNet" "ResNet50")

# Hyperparameters
pretrain_epochs_list_cifar=(5 10 20 50)
pretrain_epochs_list_imagenet=(100 125 150)
finetune_epochs_LeNet=(1 5 10)
early_stopping=3
steps=2

# Define total training epochs per model
declare -A total_epochs_map
total_epochs_map=(
    ["LeNet"]=160
    ["ResNet20"]=160
    ["Vgg16"]=160
    ["RegNetX"]=150
    ["EfficientNet"]=200
    ["ResNet50"]=150
)

# === Run CIFAR Models ===
for strat in "${strategy[@]}"; do
    for model in "${models_cifar[@]}"; do
        total_epochs=${total_epochs_map[$model]}

        for pretrain_epoch in "${pretrain_epochs_list_cifar[@]}"; do
            if [ "$model" == "LeNet" ]; then
                finetune_epochs=("${finetune_epochs_LeNet[@]}")
            else
                finetune_epoch=$((total_epochs - pretrain_epoch))
                if [ "$finetune_epoch" -lt 0 ]; then
                    continue
                fi
                finetune_epochs=("$finetune_epoch")
            fi

            for finetune_epoch in "${finetune_epochs[@]}"; do
                echo "Submitting CIFAR job: $model, Pretrain: $pretrain_epoch, Finetune: $finetune_epoch"
                sbatch prune_job.sh "$model" "$pretrain_epoch" "$early_stopping" "$finetune_epoch" "$steps" "$strat" 
            done
        done
    done

    # === Run TinyImageNet Models ===
    for model in "${models_imagenet[@]}"; do
        total_epochs=${total_epochs_map[$model]}

        for pretrain_epoch in "${pretrain_epochs_list_imagenet[@]}"; do
            finetune_epoch=$((total_epochs - pretrain_epoch))
            if [ "$finetune_epoch" -lt 0 ]; then
                continue
            fi

            batch_size=2046
            echo "Submitting ImageNet job: $model, Pretrain: $pretrain_epoch, Finetune: $finetune_epoch, Batch size: $batch_size"
            sbatch prune_job.sh "$model" "$pretrain_epoch" "$early_stopping" "$finetune_epoch" "$steps" "$strat" "$batch_size"
        done
    done
done
