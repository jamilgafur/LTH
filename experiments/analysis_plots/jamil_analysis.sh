#!/bin/bash
#SBATCH --job-name=jamil_analysis      # Job name, based on the model argument
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=1-12:00:00                   # Max runtime (4 hours)
#SBATCH --mem=32GB                       # Memory allocation
#SBATCH --gpus=1                         # Number of GPUs
#SBATCH --account=modularai

# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

python jamil_analysis.py 