#!/bin/bash

# 1. Clean up previous submissions
rm ~/submit* 2>/dev/null

# 2. Define Models and Datasets
# Note: Ensure "RegNetX_400MF" matches your python model class name if used there
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("tinyimagenet") # Focused on TinyImageNet based on previous turns
quant=("True" "False")

# 3. Define Experiment Configurations
# ==============================================================================

# VGG16: Testing the "Stage 3 Crash" vs "Stage 1 Safety"
VGG16_experiments=(
    "Original Model"
    "Stage 5 (Full)" "Stage 4 (Full)" "Stage 3 (Full)" "Stage 2 (Full)" "Stage 1 (Full)"
    "Stage 3 Conv 1 Only" "Stage 3 Conv 2 Only" "Stage 3 Conv 3 Only"
    "Stage 1 Conv 1 Only" "Stage 1 Conv 2 Only"
    "Last 2" "Stage 4-5" "Stage 3-5" "Stage 2-5" "Stage 1-5"
)

# RegNetX: Massive granular search (Efficiency Wall)
RegNetX_400MF_experiments=(
    "Original Model"
    "Last 2" "Stage 4 (Full)" "Stage 3 (Full)" "Stage 2 (Full)" "Stage 1 (Full)"
    # Granular Stage 3
    "Stage 3 Block 0 Only" "Stage 3 Block 1 Only" "Stage 3 Block 2 Only" "Stage 3 Block 3 Only"
    # Granular Stage 4
    "Stage 4 Block 0 Only" "Stage 4 Block 1 Only" "Stage 4 Block 2 Only" "Stage 4 Block 3 Only" 
    "Stage 4 Block 4 Only" "Stage 4 Block 5 Only" "Stage 4 Block 6 Only"
    # Low Variance Control
    "Stage 1 Block 0 Only"
    # Multi-Stage
    "Stage 3-4" "Stage 2-4" "Stage 1-4"
    # Partial Collapses
    "Stage 1 first 2 conv" "Stage 2 first 2 conv" "Stage 3 first 2 conv" "Stage 4 first 2 conv"
    "Stage 1 last 2 conv" "Stage 2 last 2 conv" "Stage 3 last 2 conv" "Stage 4 last 2 conv"
)

# MobileNet: Verifying "Isomorphic" safety everywhere
MobileNet_experiments=(
    "Original Model"
    "Stage 7" "Stage 6" "Stage 5" "Stage 4" "Stage 3" "Stage 2" "Stage 1"
    # Granular Checks
    "Block 7 Only" "Block 6 Only" "Block 5 Only" "Block 4 Only" "Block 3 Only" "Block 2 Only" "Block 1 Only"
    # Combinations
    "Stage 5-7" "Stage 4-7" "Stage 6-7" "Stage 3-7" "Stage 2-7" "Stage 1-7" "Last 2"
)

# XceptionNet: Control group for MobileNet behavior
XceptionNet_experiments=(
    "Original Model"
    "Stage 5 (Full)" "Stage 4 (Full)" "Stage 3 (Full)" "Stage 2 (Full)" "Stage 1 (Full)"
    "Block 5 Only" "Block 4 Only" "Block 3 Only" "Block 2 Only" "Block 1 Only"
    "Stage 3-5" "Stage 2-5" "Stage 1-5"
)

# InceptionNet: Probing the Stage 3a "Trap"
InceptionNet_experiments=(
    "Original Model"
    "Stage 5 (Full)" "Stage 4 (Full)" "Stage 3 (Full)" "Stage 2 (Full)"

    "Stage 2-5" "Stage 3-5" "Stage 4-5" "Last 2"
)

# ConvNeXt: Probing redundant depth vs interface
ConvNeXt_experiments=(
    "Original Model"
    "Stage 4 (Full)" "Stage 3 (Full)" "Stage 2 (Full)" "Stage 1 (Full)"
    "Stage 3 Block 1 Only" "Stage 3 Block 2 Only" "Stage 3 Block 3 Only"
    "Stage 1 Block 1 Only"
    "Stage 3 Inner (Block 2)"
)

# 4. Main Execution Loop
# ==============================================================================
for model in "${models[@]}"; do

    # Construct the variable name string pointing to the array
    exp_var_name="${model}_experiments[@]"

    # Use indirect expansion (!) to loop through the array named by exp_var_name
    for experiment in "${!exp_var_name}"; do
        
        for dataset in "${datasets[@]}"; do
            for quant_flag in "${quant[@]}"; do
                # Running both pruning/collapse flags (JF = Post-Collapse, Kevin = No-Collapse Baseline)
                for flag in "JF" "Kevin"; do
                    # only run the original model
                    if [[ "$experiment" == "Original Model" ]]; then
                        continue
                    fi
                    # Submit job
                    command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\",QUANT=\"$quant_flag\" submit_job.pbs"
                    
                    echo "Executing: $command"
                    eval "$command"

                done
            done
        done
    done
done
