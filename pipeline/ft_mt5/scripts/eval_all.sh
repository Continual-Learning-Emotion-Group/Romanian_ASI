#!/bin/bash
# Run all three evals that produced the RESULTS.md numbers:
#   1. Fine-tuned mT5 on val   (seen vocab)
#   2. Fine-tuned mT5 on test  (unseen vocab, with Sim@k)
#   3. Zero-shot mT5-large on test (unseen-vocab baseline)
#
# Uses single GPU (override with CUDA_VISIBLE_DEVICES). Each eval runs
# sequentially; total ~20 min on A6000.

set -eo pipefail
export PYTHONUNBUFFERED=1

USER_BASE="${USER_BASE:-/local/nlp/$USER/ro_asi_ft}"
MODEL_PATH="${MODEL_PATH:-$USER_BASE/runs/mt5-large-ro-asi/best}"

source "$USER_BASE/venv/bin/activate"
export HF_HOME="$USER_BASE/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

cd "$(dirname "$0")/../../.."

echo "=== 1/3  FT on val (seen vocab) ==="
python -m pipeline.eval.eval_generative \
    --model "$MODEL_PATH" \
    --split val --backend transformers --no-similarity

echo "=== 2/3  FT on test (unseen vocab, with similarity) ==="
python -m pipeline.eval.eval_generative \
    --model "$MODEL_PATH" \
    --split test --backend transformers

echo "=== 3/3  Zero-shot mT5-large on test (unseen baseline) ==="
python -m pipeline.eval.eval_generative \
    --model google/mt5-large \
    --split test --backend transformers --no-similarity

echo "=== ALL EVALS DONE $(date) ==="
