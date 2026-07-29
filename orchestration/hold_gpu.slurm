#!/bin/bash
#SBATCH --job-name=llm-reasoning
#SBATCH --account=kjun
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:volta:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Placeholder allocation: holds one V100 so experiment runs can be
# attached later with
#     srun --jobid=$SLURM_JOB_ID --overlap python ...
# Does no work itself; just keeps the allocation alive.
sleep infinity
