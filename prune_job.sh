#!/bin/bash
#SBATCH --job-name=prune_${model}           # Job name
#SBATCH --output=prune_${model}.out         # Output file
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --time=4:00:00                    # Max runtime (24 hours)
#SBATCH --mem=16GB                         # Memory allocation
#SBATCH --gpus=1                           # Number of GPUs
#SBATCH --account=modularai

# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

# Loop through the models
for model in "LeNet", "ResNet20", "Vgg16"
do
    # Run the pruning job for the current model
    python main_experiment.py --model ${model} --save_dir /scratch/jgafur/LTH_output --experiment None 

    # Submit experiment jobs after pruning completes
    sbatch experiment_job.sh --model ${model} --save_dir /scratch/jgafur/LTH_output --experiment NeuronSimilarity
    sbatch experiment_job.sh --model ${model} --save_dir /scratch/jgafur/LTH_output --experiment NeuronZeroing
    sbatch experiment_job.sh --model ${model} --save_dir /scratch/jgafur/LTH_output --experiment WeightZeroing
done
