#!/usr/bin/env bash
# Rebuild the piranha training environment from scratch. Run once after
# cloning the repo to /local/nlp/aij2115/code/Romanian_ASI; produces a venv
# at /local/nlp/aij2115/venv matching what the v1 SFT run used.
#
# Key pins captured in run/requirements-piranha.txt:
#   Python 3.11.15 · torch 2.5.1+cu121 · transformers from git main
#   (qwen3_5 model_type not released in any pypi transformers as of this run)
#   · deepspeed 0.17.6 · flash_attn 2.7.4.post1 prebuilt wheel · hf_hub 1.11
#
# piranha's system Python is 3.9, which transformers 5.x will not load, so
# we bootstrap Python 3.11 via uv and use that to create the venv.

set -euo pipefail

ROOT=/local/nlp/aij2115
VENV=$ROOT/venv
REPO=$ROOT/code/Romanian_ASI

test -d "$REPO" || { echo "expected repo at $REPO — git clone it first"; exit 1; }

# 1. Bootstrap Python 3.11 via uv (one-time; skip if already present).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11

# 2. Fresh venv.
if [[ -d "$VENV" ]]; then
  echo "venv already exists at $VENV — delete it manually if you want a clean rebuild"
  exit 1
fi
uv venv "$VENV" --python 3.11
source "$VENV/bin/activate"
python -V

# 3. Install pinned environment. The frozen file has transformers @ git+URL
# and the prebuilt flash-attn wheel baked in, so this one command should
# reproduce the exact env.
pip install --upgrade pip
pip install -r "$REPO/run/requirements-piranha.txt"

# 4. Smoke test: Qwen3.5-4B loads and our sanity check passes.
cd "$REPO"
export HF_HOME=$ROOT/cache
python -m pipeline.ft_qwen_mixed.prompts --sanity

echo
echo "[setup] done. To train:"
echo "  bash $REPO/run/piranha_launch.sh"
echo "or pull the released checkpoint:"
echo "  huggingface-cli download alexjerpelea/qwen3.5-4B-mixed --local-dir $ROOT/runs/final"
