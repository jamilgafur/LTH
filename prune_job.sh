#!/bin/bash
#SBATCH --job-name=prune_runner       # Job name, based on the model argument
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=1-12:00:00                   # Max runtime (4 hours)
#SBATCH --mem=16GB                       # Memory allocation
#SBATCH --gpus=1                         # Number of GPUs
#SBATCH --account=modularai
#SBATCH --output=/scratch/jgafur/LTH_output/prune_runner_%j.out
#SBATCH --error=/scratch/jgafur/LTH_output/prune_runner_%j.err

# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

# Set model from the first command-line argument
model=$1
pruner_percent=$2

# Define available models
available_models=("LeNet" "ResNet20" "Vgg16")

# Check if a model argument is provided
if [ -z "$2" ]; then
    echo "Error: No model specified."
    echo "Usage: $0 <model_name> <pruner_percent>"
    echo "Available models: ${available_models[@]}"
    exit 1
fi


# Check if the provided model is valid
if [[ ! " ${available_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model specified."
    echo "Usage: $0 <model_name>"
    echo "Available models: ${available_models[@]}"
    exit 1
fi

# Run the pruning job for the specified model logging
echo "Current Time: $(date)"
echo "Running pruning job for ${model} with command: python main_experiment.py --model ${model} --save_dir /scratch/jgafur/LTH_output --experiment None"
python main_experiment.py --model ${model} --save_dir /scratch/jgafur/LTH_output --experiment None 

# wait 10 seconds to ensure the pruning job completes
sleep 10

# Submit experiment jobs after pruning completes
echo "Submitting experiment jobs for model: ${model} for job id: ${SLURM_JOB_ID}"
sbatch experiment_job.sh  ${model}  /scratch/jgafur/LTH_output  NeuronSimilarity  ${pruner_percent} ${SLURM_JOB_ID} 
sbatch experiment_job.sh  ${model}  /scratch/jgafur/LTH_output  NeuronZeroing  ${pruner_percent} ${SLURM_JOB_ID}
sbatch experiment_job.sh  ${model}  /scratch/jgafur/LTH_output  WeightZeroing   ${pruner_percent} ${SLURM_JOB_ID}
echo "End Time: $(date)"
