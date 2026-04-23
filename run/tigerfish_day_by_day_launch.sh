#!/usr/bin/env bash
# Launch day-by-day sequential SFT on tigerfish (1× A100-40GB SXM, GPU 1).
#
# Prereqs (one-time on tigerfish — venv from previous Qwen3.5 session is reused):
#   cd /local/nlp/aij2115/ro_asi_ft/repo
#   git fetch origin && git checkout day-by-day
#   mkdir -p /local/nlp/aij2115/ro_asi_ft/hf_cache
#   source ../venv/bin/activate
#   python -m pipeline.train.prepare_data \
#       --output /local/nlp/aij2115/data/asi_day_by_day --per-language-splits
#
# Run:
#   cd /local/nlp/aij2115/ro_asi_ft/repo
#   tmux new -s dbd
#   bash run/tigerfish_day_by_day_launch.sh 2>&1 | tee /local/nlp/aij2115/runs/day_by_day/train.log

set -euo pipefail

BASE=/local/nlp/aij2115
PROJECT="$BASE/ro_asi_ft"
REPO="$PROJECT/repo"
RUNS_ROOT="$BASE/runs/day_by_day"

export HF_HOME="$PROJECT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME/torch"
export TRITON_CACHE_DIR="$HF_HOME/triton"
export WANDB_DIR="$RUNS_ROOT"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG=WARN
export NVIDIA_TF32_OVERRIDE=1
# Avoid the fragmentation OOM observed mid-day-1 (epoch ~1.5): a 1.5 GB
# allocation failed even though only ~3 GB was reserved-but-unallocated.
# expandable_segments lets PyTorch grow existing segments instead of needing
# a contiguous free block — directly recommended by the OOM error message.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$RUNS_ROOT"

cleanup() {
  echo "[cleanup] releasing GPU memory ..."
  python - <<'PY' || true
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY
}
trap cleanup EXIT INT TERM

echo "[info] nvidia-smi before launch:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv

# Abort if /local has <60 GB free — full run needs ~56 GB peak (16 GB base
# model in HF cache + 5 × ~8 GB day final/ checkpoints).
free_gb=$(df -BG /local | awk 'NR==2 {gsub("G",""); print $4}')
if (( free_gb < 60 )); then
  echo "[abort] /local has only ${free_gb}G free, need >=60G for the day-by-day run."
  exit 1
fi
echo "[info] /local free: ${free_gb}G"

cd "$REPO"
source "$PROJECT/venv/bin/activate"

LANGUAGE_ORDER="${LANGUAGE_ORDER:-en,es,fa,hi,ro}"
echo "[info] Language order: $LANGUAGE_ORDER"

python -m pipeline.train.train_day_by_day \
  --config pipeline/train/configs/qwen3_5_4b_day_by_day.yaml \
  --language-order "$LANGUAGE_ORDER"

echo "[info] disk usage after run:"
du -sh "$BASE"/{ro_asi_ft/hf_cache,runs,data} 2>/dev/null || true
