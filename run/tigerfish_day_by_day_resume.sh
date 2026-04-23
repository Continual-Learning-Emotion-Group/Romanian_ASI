#!/usr/bin/env bash
# Auto-resume wrapper for the day-by-day run. Detects which day's `final/`
# checkpoint exists under $RUNS_ROOT and either starts fresh (day 1) or
# resumes from the next day with --start-from N. Loops with backoff so a
# transient OOM / preemption / eval crash gets retried automatically.
#
# Usage (inside tmux, after the main launcher has crashed):
#   bash run/tigerfish_day_by_day_resume.sh
#
# Stops automatically once day 5's final/ exists, or after MAX_RETRIES.

set -uo pipefail

BASE=/local/nlp/aij2115
PROJECT="$BASE/ro_asi_ft"
REPO="$PROJECT/repo"
RUNS_ROOT="$BASE/runs/day_by_day"
LANGUAGE_ORDER="${LANGUAGE_ORDER:-en,es,fa,hi,ro}"
MAX_RETRIES="${MAX_RETRIES:-6}"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$RUNS_ROOT"

cd "$REPO"
source "$PROJECT/venv/bin/activate"

IFS=',' read -ra ORDER <<< "$LANGUAGE_ORDER"
TOTAL=${#ORDER[@]}

# Find the highest day N such that day{N}_{lang}/final/ exists.
last_done=0
for ((i=0; i<TOTAL; i++)); do
  day=$((i+1))
  lang="${ORDER[$i]}"
  if [[ -d "$RUNS_ROOT/day${day}_${lang}/final" ]]; then
    last_done=$day
  fi
done

retries=0
while (( last_done < TOTAL )); do
  if (( retries >= MAX_RETRIES )); then
    echo "[resume] max retries ($MAX_RETRIES) hit. Bailing — manual intervention needed."
    exit 1
  fi
  start_from=$((last_done + 1))
  echo "[resume] last completed day=$last_done, launching --start-from $start_from (try $((retries+1))/$MAX_RETRIES)"

  python -m pipeline.train.train_day_by_day \
    --config pipeline/train/configs/qwen3_5_4b_day_by_day.yaml \
    --language-order "$LANGUAGE_ORDER" \
    --start-from "$start_from" \
    2>&1 | tee -a "$RUNS_ROOT/train.log"
  rc=$?

  # Recompute last_done after the run.
  prev_last_done=$last_done
  for ((i=0; i<TOTAL; i++)); do
    day=$((i+1))
    lang="${ORDER[$i]}"
    if [[ -d "$RUNS_ROOT/day${day}_${lang}/final" ]]; then
      last_done=$day
    fi
  done

  if (( last_done == TOTAL )); then
    echo "[resume] all $TOTAL days complete (exit $rc)."
    exit 0
  fi

  if (( last_done == prev_last_done )); then
    retries=$((retries + 1))
    sleep_for=$(( 30 * retries ))
    echo "[resume] no progress this attempt (exit $rc). Sleeping ${sleep_for}s before retry."
    sleep $sleep_for
  else
    # Made progress; reset retry counter.
    retries=0
    echo "[resume] progressed to day $last_done (exit $rc). Continuing."
  fi
done
