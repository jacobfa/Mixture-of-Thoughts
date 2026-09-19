#!/bin/bash
#SBATCH --job-name=mot-train
#SBATCH --partition=rashidinejad
#SBATCH --account=rashidin_1815
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#
# Usage: sbatch slurm/train.sbatch configs/default.yaml [dotlist overrides...]
#
# NPROC=1 (default): one process, experts spread round-robin over all GPUs (expert parallel).
# NPROC=N>1: torchrun data parallel; each rank gets GPUS/N GPUs and a full copy of the experts
#            (3 x 7-8B bf16 ~ 45 GB fits one H200). Add train.deepspeed=configs/deepspeed/zero0.json
#            to run the trainable modules under DeepSpeed instead of the built-in all-reduce.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

module load python/3.11.9 cuda/12.6.3 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / ${NPROC:-1} ))

CONFIG="${1:-configs/default.yaml}"
shift || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
  srun torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py --config "$CONFIG" "$@"
else
  srun python scripts/train.py --config "$CONFIG" "$@"
fi
