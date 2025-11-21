#!/bin/bash
#SBATCH --job-name=distilbert
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=logs/distilbert-%j.out
#SBATCH --error=logs/distilbert-%j.err

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=1

module purge
module load cuda/11.4

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "conda command not found. Exiting."
  exit 1
fi
conda activate comp7015

echo "=========================================="
echo "DistilBert training test (RTX 4090)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "=========================================="

python distilbert.py

echo "=========================================="
echo "DistilBert training test completed!"
echo "=========================================="

