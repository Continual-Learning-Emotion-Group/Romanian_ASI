# Day-by-day sequential multilingual SFT — run log

**Branch:** `day-by-day` · **Model:** `Qwen/Qwen3.5-4B` · **Run finished:** 2026-04-23

This is a continual-learning experiment: instead of `main`'s scrambled
all-at-once SFT, we train one language per "day" with the next day starting
from the previous day's checkpoint. After each day we evaluate on **all 5**
language test sets to measure forgetting and cross-lingual transfer.

## Setup

- 5 languages, in fixed order this run: **`en → es → fa → hi → ro`**
- 1000 train + 250 val + 1000 test rows per language (200 val for `hi`),
  produced by `pipeline/ft_qwen_day_by_day/prepare_data.py --per-language-splits`
- 3 epochs per day, effective batch size 4 (`per_device_train_batch_size=1`,
  `gradient_accumulation_steps=4`)
- Hyperparams from `pipeline/ft_qwen_day_by_day/configs/qwen3_5_4b_day_by_day.yaml`
  (LR `1e-5`, warmup 0.03, cosine schedule, sdpa attention, bf16, gradient
  checkpointing, `load_best_model_at_end: false`)
- Hardware: tigerfish, single A100-40GB SXM (GPU 1), no DeepSpeed/DDP
- Eval backend: `transformers` (HF generate) — not `vllm`, see "Lessons
  learned" below

## Results — `set_acc@1` 5×5 matrix

The headline metric. `set_acc@1` (introduced by main's `cea5726`) treats
`labels` as the unordered set of distinct affective expressions filling the
`[MASK]` positions, and matches the model's first generation against that
set as a word sequence. Replaces the broken legacy `acc@1` that auto-missed
every Farsi multi-word phrase label.

```
                test_en   test_es   test_fa   test_hi   test_ro
day1 (en)       0.187*    0.149     0.114     0.117     0.121
day2 (→es)      0.192     0.267*    0.112     0.088     0.090
day3 (→fa)      0.166     0.231     0.445*    0.068     0.111
day4 (→hi)      0.157     0.234     0.315     0.571*    0.174
day5 (→ro)      0.144     0.167     0.358     0.533     0.443*
```

`*` = the language trained on that day. Row = model state after that day's
training. Column = which language's test set we evaluated on.

### Key findings

1. **Just-trained boost is real and large.** Diagonal entries jump
   significantly above the same-column-prior-day numbers. ES gained
   `+0.118` (0.149 → 0.267) when trained, FA gained `+0.331` (0.114 →
   0.445), HI gained `+0.503` (0.068 → 0.571), RO gained `+0.269`
   (0.174 → 0.443). Hindi is the easiest to fit; English is the hardest
   (only `+0.000` improvement vs base, since the base is already strong on
   EN).

2. **Forgetting on EN is gradual.** test_en goes 0.187 → 0.192 → 0.166 →
   0.157 → 0.144. Notably, day 2 (after Spanish training) **improved**
   English by +0.005, suggesting positive cross-lingual transfer between
   English and Spanish. Forgetting kicks in only at day 3 with Farsi
   (a typologically distant language). Total drop EN(d1) → EN(d5) is
   −0.043 absolute (−23% relative).

3. **Negative transfer to Hindi pre-training.** test_hi: 0.117 → 0.088 →
   0.068 → 0.571 → 0.533. Training on en, es, fa each pushed Hindi DOWN
   before HI was trained. The model was specializing away from Hindi-style
   labels in ways that hurt generalization. Once HI was trained directly,
   it bounced strongly.

4. **Romanian shows mixed transfer.** 0.121 → 0.090 → 0.111 → 0.174 →
   0.443. ES (sibling Romance language) actually hurt RO by −0.031, which
   is counterintuitive. HI training (typologically distant!) helped RO by
   +0.063. May be a data artifact worth investigating.

5. **Hindi retention is best.** After being trained on day 4, test_hi
   only dropped from 0.571 to 0.533 by day 5 (−0.038). Compare with FA:
   trained day 3 at 0.445, dropped to 0.315 by day 4 (−0.130, a 29%
   relative loss in one day).

The legacy `acc@1` (first-emitted-word vs `labels[0]`) is also reported in
each per-language JSON under `legacy_first_label`. It agrees with
`set_acc@1` everywhere except FA, where it understates by ~2× because of
multi-word phrase labels — the bug `cea5726` was fixing.

## Files in this run

### Code (added/modified on the `day-by-day` branch)

| Path | Purpose |
|---|---|
| `pipeline/ft_qwen_day_by_day/prepare_data.py` | Added `--per-language-splits` flag for per-language `train_{lang}` / `val_{lang}` Datasets |
| `pipeline/ft_qwen_day_by_day/train.py` | Outer per-day loop: load prev day's `final/`, train one language for 3 epochs, save end-of-day model, eval on all 5 langs, repeat |
| `pipeline/ft_qwen_day_by_day/configs/qwen3_5_4b_day_by_day.yaml` | Single-GPU sister of `qwen3_5_4b_full_ft.yaml`. `load_best_model_at_end: false`, no DeepSpeed |
| `run/tigerfish_day_by_day_launch.sh` | Tigerfish env + invokes `train_day_by_day.py` for a fresh 5-day run |
| `run/tigerfish_day_by_day_resume.sh` | Auto-resume wrapper. Detects highest day with saved `final/` and relaunches with `--start-from N+1`, with retries+backoff. Use after a crash |

### Outputs (committed under `runs/day_by_day/` and `pipeline/data/eval_results/`)

- `pipeline/data/eval_results/sft_day{1..5}_{lang}_test_{lang}_metrics.json` — 30 per-day per-test-language metric files (5 days × 5 langs + 5 `*_all_metrics.json` summary files)
- `runs/day_by_day/train.log` — full HF Trainer log for the original 5-day run (days 1–5, with day 1's vllm-eval crash visible)
- `runs/day_by_day/train_day1redo.log` — log for the day-1-only redo done after the run completed (vllm crash on the original meant we had no day-1 eval JSONs and the cleanup logic had deleted day1_en/final by the time we noticed)
- `runs/day_by_day/wandb/` — original 5-day offline wandb runs (5 files, syncable with `wandb sync`)
- `runs/day_by_day/wandb_day1redo/` — day-1 redo's wandb run

### Not committed

- Model checkpoints (~8 GB each). Only `day5_ro/final/` and `day1_en/final/` (from the redo) currently survive on tigerfish at `/local/nlp/aij2115/runs/day_by_day{,_day1redo}/`. Other days' `final/` were deleted by the disk-hygiene cleanup chain (only the most recent day kept during the run). If you need a specific intermediate checkpoint, you'd have to re-run from day 1.

## How to reproduce

### On tigerfish (env preserved from previous Qwen3.5 session)

```bash
ssh aij2115@tigerfish.cs.columbia.edu
cd /local/nlp/aij2115/ro_asi_ft/repo
git checkout day-by-day && git pull
source ../venv/bin/activate

# Build per-language DatasetDict (only needed once)
python -m pipeline.ft_qwen_day_by_day.prepare_data \
    --output /local/nlp/aij2115/data/asi_day_by_day --per-language-splits

# Full 5-day run (~2.5–3 hr)
tmux new -s dbd
bash run/tigerfish_day_by_day_launch.sh 2>&1 | tee /local/nlp/aij2115/runs/day_by_day/train.log

# If anything crashes mid-run, resume from highest day with saved final/
bash run/tigerfish_day_by_day_resume.sh
```

Override the language permutation:

```bash
LANGUAGE_ORDER=ro,hi,fa,es,en bash run/tigerfish_day_by_day_launch.sh
```

### Local sanity checks (no GPU needed)

```bash
python -m pipeline.ft_qwen_day_by_day.prepare_data --output /tmp/dbd --per-language-splits
python -m pipeline.ft_qwen_day_by_day.prompts --sanity   # confirm chat template + loss masking
```

## Environment

### Important: `requirements.txt` at the repo root is wrong for this run

The root `requirements.txt` pins `torch==2.2.2` and `transformers==5.0.0`.
Both are incompatible with Qwen3.5 — transformers 5.0.0 raises
`Unrecognized model_type: qwen3_5`, and torch 2.2.2 is too old for the
`cu128` wheels we need.

The authoritative pin list from the actual tigerfish venv is committed at
`pipeline/ft_qwen_day_by_day/requirements-tigerfish.txt` (208 lines, captured from the
wandb offline-run requirements snapshot). Use that, not root
`requirements.txt`.

### Install order that works (Python 3.12, CUDA 12.x)

```bash
# 1. Fresh venv
python3.12 -m venv venv && source venv/bin/activate
pip install --upgrade pip wheel packaging setuptools

# 2. torch FIRST (cu128 wheel; other libs need it present for their builds)
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0

# 3. Main stack (transformers must be 5.6.0+ for qwen3_5; not 5.0.0)
pip install 'transformers>=5.6.0' accelerate==1.13.0 datasets==4.5.0 \
            wandb==0.26.0 pyyaml

# 4. Qwen3.5 fast-path libs (hybrid attention: every 4th layer full_attn,
#    rest linear_attn via fla + conv1d). Needs system CUDA for the
#    source build of causal-conv1d — SDPA fallback works without these
#    but is ~3–5× slower.
pip install flash-linear-attention==0.5.0 fla-core==0.5.0 einops==0.8.2

export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
pip install causal-conv1d==1.6.1 --no-build-isolation   # ~3-5 min build

# 5. (Optional) vllm for fast eval. We did NOT install it for this run
#    because it's finicky on torch 2.11+cu128; we fell back to the
#    transformers backend (~5× slower, self-contained).
```

### Critical env vars per launch

Set in `run/tigerfish_day_by_day_launch.sh` — keep them if you write a
different launcher:

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export CUDA_VISIBLE_DEVICES=1                         # pin to one GPU
export WANDB_MODE=offline                             # or set WANDB_API_KEY
export TOKENIZERS_PARALLELISM=false                   # silence forked-proc warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # prevents the day-1 OOM
export NVIDIA_TF32_OVERRIDE=1
```

### On-disk layout during a run

```
$BASE/
├── repo/                                   # git clone, checkout day-by-day
├── venv/                                   # the Python 3.12 env built above
├── hf_cache/                               # HF_HOME — ~16 GB after first Qwen3.5-4B download
├── data/asi_day_by_day/                    # prepare_data.py --per-language-splits output, ~40 MB
└── runs/day_by_day/                        # outputs: per-day dirs, train.log, wandb/
    ├── day5_ro/final/                      # ~8 GB per day, only last survives cleanup
    └── ...
```

Peak `/local` usage during a run: ~45 GB (base model cache + one day's
trainer state + two days' final/ dirs during the chain handoff). Leave
≥60 GB headroom.

## Lessons learned (the painful ones)

### 1. CUDA OOM at epoch 1.5 of day 1 (commit `d72b847`)

First attempt with `per_device_train_batch_size=2, grad_accum=2` OOM'd
mid-step in epoch 2 of day 1 with the error:

> `torch.OutOfMemoryError: Tried to allocate 1.50 GiB. GPU has 39.49 GiB
> total, 38.69 GiB in use (34.88 alloc + 3.30 reserved-but-unalloc), only
> 817 MiB free.`

3.3 GB of memory was reserved by PyTorch but fragmented into chunks too
small for the next allocation. Two fixes applied:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — directly recommended
  by the OOM message. Lets the allocator grow existing segments instead of
  needing a contiguous free block.
- `per_device_train_batch_size 2 → 1, grad_accum 2 → 4`. Effective batch
  size unchanged, but peak activation memory halved. Worst-case
  long-sequence batches now fit with margin.

After fix: peak GPU memory 36.6 GB (vs 38.7 GB pre-fix), no further OOMs
across all 5 days × 3 epochs.

### 2. vllm not installed → day 1 eval crashed → loss of day-1 checkpoint

After day 1 trained successfully, `train_day_by_day.py` invoked
`pipeline.ft_qwen_day_by_day.eval_sft --backend vllm`, which crashed with
`ModuleNotFoundError: No module named 'vllm'`. The original `run_eval` used
`subprocess.run(check=True)` — fatal on failure. The whole script died,
day 1 had no eval JSONs.

Two fixes applied (commits `1296ed8`, `a6ff284`):

- Switched eval backend to `transformers` (already implemented in
  `eval_sft.py`, ~5× slower but self-contained).
- Made `run_eval` non-fatal: catches `CalledProcessError`, logs a warning,
  returns False. Outer loop continues to next day. Cleanup is gated on
  eval success, so a failed eval preserves the checkpoint for later
  re-eval.

**Bug not yet fully fixed:** the cleanup logic checks the *current* day's
eval_ok, not the *previous* day's. So when day 2's transformers eval
succeeded after the resume, it triggered cleanup of day 1's directory —
even though day 1 itself had never been eval'd (it had been the vllm
crash). We had to redo day 1 entirely (35 min) just to get its eval
metrics. Future improvement: also check `prev_day` eval JSONs exist before
deleting `prev_day_dir`.

### 3. Speed reality

Original estimate was ~70 min for the full 5-day run (based on initial
1.4 it/s observation at bs=2). After the OOM fix forced bs=1, throughput
dropped to ~2.0 s/it (~35% slower). Plus transformers eval backend at
~10 min/day vs vllm's ~1.5 min. Realistic per-day cost: **~25 min train
+ ~10 min eval = ~35 min/day**. Full run: **~2.5–3 hr** end-to-end.

### 4. Disk hygiene

Per-day Qwen3.5-4B `final/` is **7.9 GB** (not 14 GB as I initially
worried). With the cleanup chain enabled, peak disk usage during a day is
~24 GB (current day's trainer/ + new final/ + previous day's final/
during the brief overlap). After cleanup runs, only one `final/` dir
persists. Tigerfish `/local` had 67 GB free at launch and never dropped
below 40 GB.

## Suggested follow-ups

- Re-run with reverse permutation (`ro,hi,fa,es,en`) to test
  order-symmetry of forgetting/transfer
- Re-run with `num_train_epochs=2` instead of 3 — day 1 train_loss
  dropped from 0.97 to 0.30 across epoch 3, but val regressed from 1.18
  → 1.40 (mild overfitting). 2 epochs may transfer better day-to-day
- Investigate the ES → RO negative transfer in the current matrix (was it
  data noise or a real effect?)
- Install vllm properly to speed up eval; or write a batched HF generate
  that's faster than the current transformers backend
- Fix the `prev_day_dir` cleanup race so a single-day eval failure doesn't
  cost a whole day's checkpoint
