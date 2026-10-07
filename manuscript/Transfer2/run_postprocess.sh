#!/bin/bash

# Define the models and datasets to process
models=("VGG16")
#"RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("Cifar10")
#tinyimagenet" "Cifar100" "Cifar10")

echo "=== Submitting CKA Post-Processing Jobs ==="

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        
        # Submit the post-processing job
        command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\" submit_postprocess.pbs"
        
        echo "Executing: $command"
        eval "$command"
        
    done
done

echo "All jobs submitted to the queue."