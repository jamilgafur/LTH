#!/bin/bash

# Define the models and datasets to process
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("tinyimagenet" "Cifar100" "Cifar10")

echo "=== Submitting CKA Post-Processing Jobs ==="

# Epoch configs as "EPOCHS PRETRAIN" pairs (matches temp1/temp2 arg order)
configs=("100 300" "200 200" "300 100")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for quant in "False" "True"; do
            for cfg in "${configs[@]}"; do
                read -r EPOCHS PRETRAIN <<< "$cfg"

                # Submit the post-processing job using SGE GPU flags
                command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",QUANT=\"$quant\",EPOCHS=\"$EPOCHS\",PRETRAIN=\"$PRETRAIN\" submit_postprocess.pbs"

                echo "Executing: $command"
                eval "$command"

            done
        done
    done
done

echo "All jobs submitted to the queue."
