#!/bin/bash
#SBATCH --job-name=mot-expert-generate
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
# Usage: sbatch slurm/expert_generate.sbatch configs/default.yaml prompts.jsonl out.jsonl [--experts 0,1 --n 4 ...]
# Runs each raw expert through vLLM (for expert_scores labelling / routing-only baselines).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

module load python/3.11.9 cuda/12.6.3 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CONFIG="$1"; PROMPTS="$2"; OUT="$3"
shift 3
srun python scripts/expert_generate.py --config "$CONFIG" --prompts "$PROMPTS" --output "$OUT" "$@"
