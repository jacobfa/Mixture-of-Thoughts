#!/bin/bash
#SBATCH --job-name=mot-generate
#SBATCH --partition=rashidinejad
#SBATCH --account=rashidin_1815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Usage: sbatch slurm/generate.sbatch configs/default.yaml runs/default/final prompts.jsonl out.jsonl [overrides...]
# All experts on one H200 avoids inter-GPU hidden-state traffic in the decode loop; request more GPUs
# (and set model.devices) only if the expert pool does not fit.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

module load python/3.11.9 cuda/12.6.3 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CONFIG="$1"; CKPT="$2"; PROMPTS="$3"; OUT="$4"
shift 4
srun python scripts/generate.py --config "$CONFIG" --checkpoint "$CKPT" --prompts "$PROMPTS" --output "$OUT" "$@"
