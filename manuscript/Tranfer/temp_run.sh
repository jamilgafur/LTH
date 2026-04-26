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

# VGG16: Probing Safe Early Stages vs High Variance Deep Stages
VGG16_experiments=(
    "Original Model"
    "Stage 1 (Full)" "Stage 2 (Full)" "Stage 3 (Full)" "Stage 4 (Full)" "Stage 5 (Full)"
    "Conv 1 Only" "Conv 2 Only" "Conv 3 Only"
    "Conv 9 Only" "Conv 10 Only" "Conv 13 Only"
    "Stages 1 and 2" "Stages 3 and 4" "Stages 4 and 5"
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

# MobileNet: Probing Pointwise Variance Spikes vs Safe Early Blocks
MobileNet_experiments=(
    "Original Model"
    "Early Features (Full)" "Middle Features (Full)" "Late Features (Full)"
    "Block 0 Only" "Block 1 Only" "Block 2 Only"
    "Block 8 Only" "Block 10 Only" "Block 11 Only"
    "Early and Middle" "Middle and Late" "Almost All (1-11)"
)

# XceptionNet: Control group for MobileNet behavior
XceptionNet_experiments=(
   "Original Model", "Entry Flow (Full)", "Middle Flow (Full)", "Exit Flow (Full)", "Block 1 Only", "Block 2 Only", "Block 3 Only", "Conv 3 and 4 Only", "Middle Flow Block 4 Only", "Middle Flow Block 7 Only", "Block 4 Only", "Entry and Middle Flow", "Middle and Exit Flow",   
)

# InceptionNet: Probing flat early stages vs massive Stage 5 spikes
InceptionNet_experiments=(
    "Original Model"
    "Stage 2 (Full)" "Stage 3 (Full)" "Stage 4 (Full)" "Stage 5 (Full)"
    "Stage 3a Only" "Stage 4a Only"
    "Stage 5a Only" "Stage 5b Only"
    "Stages 2 and 3" "Stages 3 and 4"
)

# ConvNeXt: Probing redundant early stages vs high-variance deep stages
ConvNeXt_experiments=(
    "Original Model"
    "Stage 1 (Full)" "Stage 2 (Full)" "Stage 3 (Full)" "Stage 4 (Full)"
    "Stage 1 Block 0 Only" "Stage 1 Block 2 Only"
    "Stage 3 Block 8 Only" "Stage 4 Block 0 Only" "Stage 4 Block 2 Only"
    "Stages 1 and 2" "Stages 3 and 4"
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
                    if [[ "$experiment" != "Original Model" ]]; then
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

