#!/bin/bash
#SBATCH --job-name=experiment_${MODEL}_${EXPERIMENT}  # Job name
#SBATCH --output=experiment_${MODEL}_${EXPERIMENT}.out  # Output file
#SBATCH --ntasks=1                                    # Number of tasks
#SBATCH --time=4:00:00                                # Max runtime (4 hours)
#SBATCH --mem=8GB                                    # Memory allocation
#SBATCH --gpus=1                                     # Number of GPUs

# Load necessary modules (e.g., Python, CUDA, etc.)
module load python/3.8
module load cuda/11.2

# Activate virtual environment or set up environment
source /path/to/your/virtualenv/bin/activate

# Run the experiment job based on the pruning result
python main_experiment.py --model ${MODEL} --experiment ${EXPERIMENT}
