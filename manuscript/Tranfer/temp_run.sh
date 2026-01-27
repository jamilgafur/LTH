#!/bin/bash

# 1. Clean up previous submissions
rm ~/submit* 2>/dev/null

# 2. Define Models and Datasets
# Note: Ensure these names match the prefixes of the config arrays below
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("Cifar100" "tinyimagenet")
quant=("True")

# 3. Define Experiment Configurations (Once per Model)
# We use a standard suffix (e.g., _experiments) to make lookup easy
declare -a VGG16_experiments=("Original Model" "Last 2" "Stage 5"  "Stage 3" "Stage 2")
declare -a RegNetX_400MF_experiments=("Original Model" "Last 2" "Stage 4" "Stage 3" "Stage 1-4")
declare -a ConvNeXt_experiments=("Original Model" "Stage 4" "Stage 3" "Stage 2" "Stage 1")
declare -a InceptionNet_experiments=("Original Model" "Stage 5"  "Stage 2" "Last 2")
declare -a XceptionNet_experiments=("Original Model" "Stage 5"  "Stage 2-5"  "Stage 2" "Last 2")
declare -a MobileNet_experiments=("Original Model" "Stage 7" "Stage 2" "Stage 1")

# 4. Main Execution Loop
for model in "${models[@]}"; do
    
    # --- DYNAMIC REFERENCE MAGIC ---
    # Construct the variable name for the current model's experiments (e.g., "VGG16_experiments")
    # declare -n creates a "nameref" (pointer) so $current_experiments actually reads the specific model array
    declare -n current_experiments="${model}_experiments"

    for dataset in "${datasets[@]}"; do
        for experiment in "${current_experiments[@]}"; do
            for quant_flag in "${quant[@]}"; do
                for flag in "JF" "Kevin"; do
                    
                    # Construct the command
                    # We escape the quotes around EXPERIMENT to handle spaces safely (e.g., "Original Model")
                    command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\",QUANT=\"$quant_flag\" submit_job.pbs"

                    # Debug output
                    echo "Executing: $command"
                    
                    # Run the command
                    eval "$command"

                done
            done
        done
    done
done