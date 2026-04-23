#!/usr/bin/env bash
# Launch Qwen3.5-4B SFT on tigerfish (single A100-40GB).
# Per the NLP Lab GPU Guide: project lives on /local SSD, not NFS home.

set -euo pipefail

PROJECT=/local/nlp/${USER}/ro_asi_ft
REPO=${PROJECT}/repo
RUN_DIR=${PROJECT}/runs/qwen3.5-4b-ro-asi
LOG=${PROJECT}/runs/qwen3.5-4b-ro-asi.log

export HF_HOME=${PROJECT}/hf_cache
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=${PROJECT}/torch_cache
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "$(dirname "${LOG}")"

# Avoid GPU 0 — another user has a job on it (verified 2026-04-23).
# Override at the call site if a different GPU is free: CUDA_VISIBLE_DEVICES=2 bash ...
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# Pre-flight: refuse to launch if /local is too tight.
free_gb=$(df -BG /local | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "${free_gb}" -lt 40 ]; then
    echo "ABORT: /local has only ${free_gb}G free; need >=40G. Clean caches first." >&2
    exit 1
fi

cd "${REPO}"
source "${PROJECT}/venv/bin/activate"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv | tee -a "${LOG}"

python -m pipeline.ft_qwen.train \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee -a "${LOG}"

echo "---"
du -sh "${PROJECT}" | tee -a "${LOG}"
