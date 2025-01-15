#!/bin/bash
#SBATCH --job-name=experiment_$MODEL_$EXPERIMENT  # Job name
#SBATCH --output=experiment_$MODEL_$EXPERIMENT.out  # Output file
#SBATCH --ntasks=1                                    # Number of tasks
#SBATCH --time=4:00:00                                # Max runtime (4 hours)
#SBATCH --mem=16GB                                    # Memory allocation
#SBATCH --gpus=1                                     # Number of GPUs
#SBATCH --account=modularai
# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env


# Run the experiment job based on the pruning result
echo "Running experiment for $MODEL with $EXPERIMENT"
echo "Command: python main_experiment.py --model $MODEL --experiment $EXPERIMENT --save_dir $SAVE_DIR"
python main_experiment.py --model $MODEL --experiment $EXPERIMENT --save_dir $SAVE_DIR
