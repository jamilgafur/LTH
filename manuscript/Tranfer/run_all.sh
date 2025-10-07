#!/bin/bash

# Define models and datasets
models=("VGG16")
datasets=("Cifar100" )
experiment_names=("Original Model")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for experiment in "${experiment_names[@]}"; do
      for flag in "JF" "Kevin"; do
        # Properly quote the variables with spaces and commas
	command="qsub -q all.q -l ngpus=1 -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"$experiment\",FLAG=\"$flag\" submit_job.pbs"
        # Echo the command for debugging
        echo "Executing: $command"

        # Run the command
        eval $command
      done
    done
  done
done

