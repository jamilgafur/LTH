#!/bin/bash
#SBATCH --job-name=prune_runner
#SBATCH --ntasks=1
#SBATCH --time=11:00:00   # Max runtime (11 hour)
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --account=modularai
#SBATCH --output=/scratch/jgafur/LTH_output/prune_runner_%j.out
#SBATCH --error=/scratch/jgafur/LTH_output/prune_runner_%j.err

# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

model=$1
pretrain_epochs=$2
patience=$3
finetune_epochs=$4
steps=$5
strategy=$6
batch_size=$7  # Optional (only passed for ImageNet models)

# Base command
cmd="python main_experiment.py --model ${model} \
    --save_dir /scratch/jgafur/LTH_output \
    --experiment None \
    --finetune_epochs ${finetune_epochs} \
    --pretrain_epochs ${pretrain_epochs} \
    --steps ${steps} \
    --patience ${patience} \
    --strategy ${strategy}"

# Append batch size if provided
if [ -n "$batch_size" ]; then
    cmd="$cmd --batch_size ${batch_size}"
fi

echo "Current Time: $(date)"
echo "Running pruning job for ${model} with command:"
echo "$cmd"
eval $cmd

# Submit experiment jobs after pruning completes
echo "Submitting experiment jobs for model: ${model} for job id: ${SLURM_JOB_ID}"
sleep 10
sbatch experiment_job.sh ${model} ${pretrain_epochs} ${patience} ${finetune_epochs} ${steps} /scratch/jgafur/LTH_output NeuronSimilarity ${SLURM_JOB_ID} ${patience}
sleep 10
sbatch experiment_job.sh ${model} ${pretrain_epochs} ${patience} ${finetune_epochs} ${steps} /scratch/jgafur/LTH_output NeuronZeroing ${SLURM_JOB_ID} ${patience}
# sbatch experiment_job.sh ${model} ${pretrain_epochs} ${patience} ${finetune_epochs} ${steps} /scratch/jgafur/LTH_output WeightZeroing ${SLURM_JOB_ID} ${patience}
echo "End Time: $(date)"
