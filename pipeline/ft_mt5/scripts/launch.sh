#!/bin/bash
# Full fine-tune run on seahorse. See pipeline/ft_mt5/README.md for setup
# (venv, HF cache env vars, GPU selection). Run under tmux so SSH drops
# don't kill it:
#
#   tmux new -d -s mt5ft "bash pipeline/ft_mt5/scripts/launch.sh 2>&1 | stdbuf -oL tee runs/mt5-large-ro-asi.log"
#
# Expected wall clock on 1 × RTX A6000, bf16, bs=16: ~2 h.

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- activate env (edit USER_BASE if your setup differs) --------------------
USER_BASE="${USER_BASE:-/local/nlp/$USER/ro_asi_ft}"
source "$USER_BASE/venv/bin/activate"

# --- SSD caches (lab GPU guide) ---------------------------------------------
export HF_HOME="$USER_BASE/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$USER_BASE/torch_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

# --- pick a free GPU (override at invocation: CUDA_VISIBLE_DEVICES=N bash launch.sh)
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "=== $(date) ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

cd "$(dirname "$0")/../../.."  # → repo root
python -m pipeline.ft_mt5.train \
    --output-dir "$USER_BASE/runs/mt5-large-ro-asi"

echo "=== DONE $(date) ==="
