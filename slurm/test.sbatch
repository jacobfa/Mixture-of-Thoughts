#!/bin/bash
#SBATCH --job-name=mot-test
#SBATCH --partition=rashidinejad
#SBATCH --account=rashidin_1815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#
# Runs the full test suite (CPU tests + GPU/FlashAttention-3 smoke tests) on a compute node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
module load python/3.11.9 cuda/12.6.3 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
srun python -m pytest tests -q -p no:warnings --tb=short
