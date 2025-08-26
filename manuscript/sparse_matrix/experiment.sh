#!/bin/bash
#SBATCH --job-name=trinity
#SBATCH --ntasks=1
#SBATCH --time=02:00:00   # Max runtime (11 hour)
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --account=modularai
#SBATCH --output=out.out
#SBATCH --error=err.err

# Check if enough arguments are passed
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> <batch_size> <threshold>"
    exit 1
fi

# Assign arguments to variables
MODEL_NAME=$1
BATCH_SIZE=$2
THRESHOLD=$3

# Load necessary modules (e.g., for Python or CUDA, if using GPUs)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

# Run the Python script with the specified model, batch size, and threshold
echo "Running Python script for model: $MODEL_NAME, batch size: $BATCH_SIZE, threshold: $THRESHOLD"
python csr_model_analysis.py --models "$MODEL_NAME" --batch_sizes "$BATCH_SIZE" --thresholds "$THRESHOLD"

python plot.py

echo "Experiment completed for model: $MODEL_NAME, batch size: $BATCH_SIZE, threshold: $THRESHOLD"