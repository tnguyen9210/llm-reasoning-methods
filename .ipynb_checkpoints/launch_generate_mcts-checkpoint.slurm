#!/bin/bash
#SBATCH --job-name=multi-gpu    # Job name
#SBATCH --account=windfall
#SBATCH --partition=gpu_windfall
#SBATCH --nodes=1
#SBATCH --ntasks=44
#SBATCH --mem-per-cpu=5gb
#SBATCH --gres=gpu:volta:1              # Number of GPUs (per node)
#SBATCH --output=logs/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=100:00:00                  # Time limit hrs:min:sec

'''Usage:
# Best-of-N on the MATH-500 dataset

sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
'''

source ~/.bashrc
set -x -e
micromamba activate py311

python generate_mcts_prm800k_v52.py --d 

