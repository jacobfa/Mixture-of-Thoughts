#!/bin/bash
#SBATCH --job-name=mot-train
#SBATCH --partition=rashidinejad
#SBATCH --account=rashidin_1815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#
# Usage: sbatch slurm/train.sbatch configs/default.yaml [dotlist overrides...]
# Experts are placed round-robin over the allocated GPUs; the router/interaction layers sit on cuda:0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

module load python/3.11.9 cuda/12.6.3 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${1:-configs/default.yaml}"
shift || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python scripts/train.py --config "$CONFIG" "$@"
