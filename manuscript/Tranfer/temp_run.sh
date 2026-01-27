#!/bin/bash

# 1. Clean up previous submissions
rm ~/submit* 2>/dev/null

# 2. Define Models and Datasets
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("Cifar100" "tinyimagenet")
quant=("True")

# 3. Define Experiment Configurations (Once per Model)
# Note: These arrays define the experiments for each specific model
VGG16_experiments=("Original Model" "Last 2" "Stage 5"  "Stage 2-5"  "Stage 2")
RegNetX_400MF_experiments=("Original Model" "Last 2" "Stage 4" "Stage 1"  "Stage 1-4")
ConvNeXt_experiments=("Original Model" "Stage 4"  "Stage 1")
InceptionNet_experiments=("Original Model" "Stage 5" "Stage 2-5" "Stage 3" "Stage 2" "Last 2")
XceptionNet_experiments=("Original Model" "Stage 5" "Stage 4-5"  "Stage 3" "Stage 2" "Last 2")
MobileNet_experiments=("Original Model" "Stage 7" "Stage 4-7" "Stage 3-7" "Stage 6"  "Stage 2" "Stage 1")

# 4. Main Execution Loop
for model in "${models[@]}"; do

    # --- COMPATIBILITY FIX ---
    # Construct the variable name string pointing to the array (e.g. "VGG16_experiments[@]")
    exp_var_name="${model}_experiments[@]"

    # Use the exclamation mark (!) to expand the variable name indirectly
    for experiment in "${!exp_var_name}"; do
        
        for dataset in "${datasets[@]}"; do
            for quant_flag in "${quant[@]}"; do
                for flag in "JF" "Kevin"; do
                    
                    # Construct the command
                    # Quotes are escaped around variables to handle spaces safely
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