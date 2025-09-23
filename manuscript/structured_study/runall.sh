#!/bin/bash

rm -f *.out

models=("ResNet20" "ResNet50" "Vgg16" "Vgg16ImageNet" "RegNetX")
strategies=("magnitude")
steps_list=(5 10 21 30)
pretrain_epochs_list=(1 5 10)
finetune_epochs_list=(30 80 150)
early_stopping=3

for model in "${models[@]}"; do
    for strategy in "${strategies[@]}"; do
        for steps in "${steps_list[@]}"; do
            for pretrain_epochs in "${pretrain_epochs_list[@]}"; do
                for finetune_epochs in "${finetune_epochs_list[@]}"; do
                    
                    # Use larger batch size for ImageNet models
                    if [[ "$model" == "RegNetX" || "$model" == "ResNet50" || "$model" == "Vgg16ImageNet" ]]; then
                        batch_size=256
                    else
                        batch_size=128
                    fi
                    
                    echo "Submitting: $model | Strategy: $strategy | Pretrain: $pretrain_epochs | Finetune: $finetune_epochs | Steps: $steps | Batch: $batch_size"
                    sbatch prune_job.sh "$model" "$pretrain_epochs" "$early_stopping" "$finetune_epochs" "$steps" "$strategy" "$batch_size"

                done
            done
        done
    done
done
