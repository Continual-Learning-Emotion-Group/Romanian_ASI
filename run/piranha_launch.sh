#!/usr/bin/env bash
# Launch Romanian ASI SFT Run #1 on piranha (4× A100-40GB, NVLink).
#
# Prereqs (one-time on piranha):
#   mkdir -p /local/nlp/aij2115/{code,venv,cache,data,runs/{checkpoints,final,logs}}
#   cd /local/nlp/aij2115/code
#   git clone git@github.com:lolismek/Romanian_ASI.git
#   python3 -m venv /local/nlp/aij2115/venv
#   source /local/nlp/aij2115/venv/bin/activate
#   pip install -r /local/nlp/aij2115/code/Romanian_ASI/requirements.txt
#   python -m pipeline.ft_qwen_mixed.prepare_data --output /local/nlp/aij2115/data/asi_multilingual
#
# Run:
#   cd /local/nlp/aij2115/code/Romanian_ASI
#   tmux new -s sft
#   bash run/piranha_launch.sh 2>&1 | tee /local/nlp/aij2115/runs/logs/train.log

set -euo pipefail

BASE=/local/nlp/aij2115
REPO="$BASE/code/Romanian_ASI"
LOG_DIR="$BASE/runs/logs"

export HF_HOME="$BASE/cache"
export TRANSFORMERS_CACHE="$BASE/cache"
export TORCH_HOME="$BASE/cache/torch"
export TRITON_CACHE_DIR="$BASE/cache/triton"  # avoid NAS-backed ~/.triton per DeepSpeed warning
export WANDB_DIR="$LOG_DIR"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN

# Fast TF32 on A100
export NVIDIA_TF32_OVERRIDE=1

mkdir -p "$BASE/cache" "$LOG_DIR"

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

# Abort if another user is using all 4 GPUs (>1 GiB used on all of them).
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024' | wc -l)
if [[ "$busy" -ge 4 ]]; then
  echo "[abort] all 4 GPUs look busy (>1 GiB used). Check nvidia-smi."
  exit 1
fi

cd "$REPO"
source "$BASE/venv/bin/activate"

torchrun --nproc_per_node=4 -m pipeline.ft_qwen_mixed.train \
  --config pipeline/ft_qwen_mixed/configs/qwen3_5_4b_full_ft.yaml

echo "[info] disk usage after run:"
du -sh "$BASE"/{cache,runs,data} 2>/dev/null || true
