#!/bin/bash
#SBATCH --job-name=prune_${MODEL}           # Job name
#SBATCH --output=prune_${MODEL}.out         # Output file
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --time=24:00:00                    # Max runtime (24 hours)
#SBATCH --mem=16GB                         # Memory allocation
#SBATCH --gpus=1                           # Number of GPUs

# Load necessary modules (e.g., Python, CUDA, etc.)
module load python/3.8
module load cuda/11.2

# Activate virtual environment or set up environment
source /path/to/your/virtualenv/bin/activate

# Run the pruning job (replace with your actual script path)
python main_experiment.py --model ${MODEL}

# Submit experiment jobs after pruning completes
sbatch experiment_job.sh --model ${MODEL} --experiment NeuronSimilarity
sbatch experiment_job.sh --model ${MODEL} --experiment NeuronZeroing
sbatch experiment_job.sh --model ${MODEL} --experiment WeightZeroing
