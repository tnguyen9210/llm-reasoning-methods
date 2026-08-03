#!/bin/bash
#SBATCH --job-name=llm-reasoning
#SBATCH --account=kjun
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Placeholder allocation: holds one GPU so experiment runs can be
# attached later with
#     srun --jobid=$SLURM_JOB_ID --overlap python ...
# Does no work itself; just keeps the allocation alive.
#
# gres is any-GPU (not volta-only) as of 2026-07-31. Note that the
# cluster's only non-Volta GPUs are MIG A100 slices
# (nvidia_a100_80gb_pcie_3g.40gb, 8 on r7u25n1), which vLLM cannot
# use -- it rejects MIG UUIDs (cf. job 23378067, prm800k-level5
# ledger). An allocation that lands on one will accept launches
# that die immediately.
sleep infinity
