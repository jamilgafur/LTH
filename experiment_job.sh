#!/bin/bash
#SBATCH --job-name=experiment_$1_$3  # Job name
#SBATCH --output=experiment_$1_$3.out  # Output file
#SBATCH --ntasks=1                                    # Number of tasks
#SBATCH --time=16:00:00                                # Max runtime (8 hours)
#SBATCH --mem=16GB                                    # Memory allocation
#SBATCH --gpus=1                                     # Number of GPUs
#SBATCH --account=modularai
#SBATCH --output=/scratch/jgafur/LTH_output/experiment_%j.out
#SBATCH --error=/scratch/jgafur/LTH_output/experiment_%j.err
# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

model=$1
save_dir=$2
experiment=$3
pruner_percent=$4
parent_job_id=$5

# Run the experiment job based on the pruning result
echo "Current Time: $(date)"
echo "Running experiment for ${model} with ${experiment} at pruner percent ${pruner_percent} in ${save_dir} with parent job id ${parent_job_id}"
echo "Command: python main_experiment.py --model $model --experiment ${experiment} --save_dir ${save_dir}" --sampling_fraction ${pruner_percent}
python main_experiment.py --model ${model} --experiment ${experiment} --save_dir ${save_dir} --sampling_fraction ${pruner_percent} >> experiment_${model}_${experiment}.out
echo "End Time: $(date)"