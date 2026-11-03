#!/usr/bin/env bash
# Avengers-protocol comparison of MoT against routing/ensembling baselines on a shared expert pool.
# POOL is `small` (four 3-4B experts) or `large` (three ~30B experts); see configs/<pool>/.
#
#   run.sh setup            patch the harness, install deps, download and split the data
#   run.sh serve POOL       vLLM endpoints for the experts + embedding model (leave running; --gpus 0,1,2)
#   run.sh bank POOL        single-expert runs on the calibration split -> bank, MoT data, rank artifacts, router data
#   run.sh routerdc POOL    train RouterDC on the bank
#   run.sh zooter POOL      train Zooter on the bank
#   run.sh loraretriever POOL   train the LoraRetriever encoder and expert embeddings on the bank
#   run.sh routers POOL     all three of the above
#   run.sh mot POOL         train MoT on the bank
#   run.sh serve-mot POOL   MoT endpoint (leave running)
#   run.sh eval POOL        every method on the held-out split, then results/POOL/summary.{md,json}
#   run.sh table POOL       rebuild the summary from existing results
# Every stage logs to wandb (project/entity/mode from configs/POOL/mot.yaml train.wandb; WANDB_MODE=offline|disabled
# overrides). The final summary is also logged as wandb tables.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXP=$ROOT/experiments
stage=${1:?usage: run.sh STAGE [POOL] [extra args]}
pool=${2:-small}
extra=("${@:3}")
set -- ${extra[@]+"${extra[@]}"}

case $stage in
  setup)
    if ! git -C "$EXP/baselines/avengers" apply --reverse --check ../../patches/avengers.patch 2>/dev/null; then
      git -C "$EXP/baselines/avengers" apply ../../patches/avengers.patch
    fi
    pip install -e "$ROOT[experiments]" -r "$EXP/baselines/avengers/requirements.txt"
    python "$EXP/split_data.py" ;;
  serve) python "$EXP/serve_experts.py" --pool "$pool" "$@" ;;
  bank)
    python "$EXP/harness.py" --pool "$pool" --split train "$@"
    python "$EXP/build_bank.py" --pool "$pool" ;;
  routerdc) python "$EXP/routerdc.py" --pool "$pool" "$@" ;;
  zooter) python "$EXP/zooter.py" --pool "$pool" "$@" ;;
  loraretriever) python "$EXP/loraretriever.py" --pool "$pool" "$@" ;;
  routers)
    bash "$0" routerdc "$pool"
    bash "$0" zooter "$pool"
    bash "$0" loraretriever "$pool" ;;
  mot) python "$ROOT/scripts/train.py" --config "$EXP/configs/$pool/mot.yaml" "$@" ;;
  serve-mot)
    python "$ROOT/scripts/serve.py" --config "$EXP/configs/$pool/mot.yaml" \
      --checkpoint "$EXP/results/$pool/mot/final" --port 8010 "$@" ;;
  eval)
    python "$EXP/harness.py" --pool "$pool" --split test "$@"
    python "$EXP/summarize.py" --pool "$pool" ;;
  table) python "$EXP/summarize.py" --pool "$pool" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
