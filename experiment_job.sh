#!/bin/bash
#SBATCH --job-name=experiment_${1}_${3}  # Job name
#SBATCH --ntasks=1
#SBATCH --time=16:00:00                  # Max runtime (16 hours)
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH --account=modularai
#SBATCH --output=/scratch/jgafur/LTH_output/experiment_%j.out
#SBATCH --error=/scratch/jgafur/LTH_output/experiment_%j.err

# Load necessary modules (e.g., Python, CUDA, etc.)
module load conda
conda activate /scratch/jgafur/LTH_Conda_ENV/LTH_exp_env

model=$1
pretrain_epochs=$2
early_stopping=$3
finetune_epochs=$4
steps=$5
save_dir=$6
experiment=$7
parent_job_id=$8
patience=$9

echo "Parameters: $1 $2 $3 $4 $5 $6 $7 $8 $9"
echo "Current Time: $(date)"
echo "Running experiment for ${model} with ${experiment} in ${save_dir} with parent job id ${parent_job_id}"
echo "Command: python main_experiment.py --model ${model} --experiment ${experiment} --steps ${steps} --finetune_epochs ${finetune_epochs} --save_dir ${save_dir} --pretrain_epochs ${pretrain_epochs} --patience ${patience} >> experiment_${model}_${experiment}.out" 
sleep 10
python main_experiment.py --model ${model} --experiment ${experiment} --steps ${steps} --finetune_epochs ${finetune_epochs} --save_dir ${save_dir} --pretrain_epochs ${pretrain_epochs} --patience ${patience} >> experiment_${model}_${experiment}.out
echo "End Time: $(date)"