#!/bin/bash
#SBATCH --job-name=prune_runner
#SBATCH --ntasks=1
#SBATCH --time=11:00:00   # Max runtime (11 hour)
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH --account=modularai
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /kfs2/projects/modularai/jgafur/LTH/pyprune_conda

cd /projects/modularai/jgafur/LTH/manuscript/Tranfer/

# Parameters passed from command line
DATASET="$1"
EXPERIMENT="$2"

echo "Running dataset: $DATASET"
echo "Running experiment: $EXPERIMENT"

python transfer.py --dataset "$DATASET" --experiment "$EXPERIMENT" --post_compress
