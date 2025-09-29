#!/bin/bash

# Define datasets and experiments for each dataset
declare -A DATASET_EXPERIMENTS
DATASET_EXPERIMENTS["Cifar10"]="Original Model|Last 2|Stage 5|Stage 4|Stage 3|Stage 4-5|Stage 3-5|Stage 2-5"
DATASET_EXPERIMENTS["Cifar100"]="Original Model|Last 2|Stage 5|Stage 4|Stage 4-5"
DATASET_EXPERIMENTS["TinyImageNet"]="Original Model|Last 2|Stage 5|Stage 4|Stage 4-5|All Conv Layers"
DATASET_EXPERIMENTS["ImageNet"]="Original Model|Last 2|Stage 5|Stage 4|All Conv Layers"

# List of experiment functions
EXPERIMENT_FUNCTIONS=("run_jf_experiment" "run_kevin_experiment" "run_nick_experiment")

# Loop through datasets
for dataset in "${!DATASET_EXPERIMENTS[@]}"; do
    # Split the dataset's experiments into an array
    IFS='|' read -ra EXPERIMENTS <<< "${DATASET_EXPERIMENTS[$dataset]}"

    # Loop through each experiment
    for experiment in "${EXPERIMENTS[@]}"; do
        # Loop through each experiment function
        for func in "${EXPERIMENT_FUNCTIONS[@]}"; do
            # Based on the function name, set the experiment flag (this is how we decide between --JF, --Kevin, or --Nick)
            if [[ "$func" == "run_jf_experiment" ]]; then
                JF_FLAG="--JF"
            elif [[ "$func" == "run_kevin_experiment" ]]; then
                JF_FLAG="--Kevin"
            elif [[ "$func" == "run_nick_experiment" ]]; then
                JF_FLAG="--Nick"
            fi

            echo "Submitting job for dataset: $dataset | experiment: $experiment | function: $func"
            sbatch run_experiment_job.sh "$dataset" "$experiment" "$JF_FLAG"  # passing JF_FLAG which determines the experiment function
        done
    done
done
``
