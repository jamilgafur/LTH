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

# Change directory to the project folder
cd /projects/modularai/jgafur/LTH/manuscript/Tranfer/

# Parameters passed from command line (dataset, experiment, and experiment function)
DATASET="$1"
EXPERIMENT="$2"
EXPERIMENT_FUNC="$3"  # This will be either --JF, --Kevin, or --Nick
IMP_FLAG="--imp"
POST_COMPRESS_FLAG="--post_compress"

echo "Running dataset: $DATASET"
echo "Running experiment: $EXPERIMENT"
echo "Using experiment function: $EXPERIMENT_FUNC"

# Run the experiment with the correct function
python transfer.py --dataset "$DATASET" --experiment "$EXPERIMENT" $EXPERIMENT_FUNC $IMP_FLAG $POST_COMPRESS_FLAG
