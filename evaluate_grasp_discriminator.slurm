#!/bin/bash
#SBATCH --job-name=gg_dis_eval
#SBATCH --output=tmp/gg_dis_eval_%j.out
#SBATCH --error=tmp/gg_dis_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=stud

source ~/miniconda3/etc/profile.d/conda.sh
conda activate GraspGen

set -euo pipefail

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"

cd ~/GraspGen

mkdir -p tmp outputs/discriminator_${SLURM_JOB_ID:-manual}

python -V
nvidia-smi

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python scripts/evaluate_grasp_discriminator.py
