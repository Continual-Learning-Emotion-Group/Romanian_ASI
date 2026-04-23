# mT5-Large fine-tuning on Romanian ASI

Replicates the MASIVE paper's generative fine-tuning recipe on our Romanian ASI
benchmark. Single-GPU training on `seahorse.cs.columbia.edu` (A6000 48GB).

## Overview

```
benchmark_ro_asi_clean.jsonl (70K raw → 53,154 deduped)
    │
    ▼  pipeline/ft_mt5/resplit.py  (already run, committed)
splits/{train,val,test}.jsonl      (45,181 / 2,658 / 5,315)
    │
    ▼  pipeline/ft_mt5/train.py    (run on seahorse)
runs/mt5-large-ro-asi/best/        (1.2B-param mT5 checkpoint)
    │
    ▼  pipeline/eval/eval_generative.py  --model <best>
eval_results/gen_mt5-large-ro-asi_test_ro_*.json
```

## Setup on seahorse

### 1. SSH in, claim a free GPU

```bash
ssh <cs_account>@seahorse.cs.columbia.edu
nvidia-smi                       # pick a GPU that's idle
export CUDA_VISIBLE_DEVICES=<idx>
```

Do **not** run on piranha/tigerfish (busy). kingcrab (V100) is second-best if
all A6000s are taken.

### 2. Project directory on local SSD

Per the NLP Lab GPU Guide — always use `/local/nlp/` (fast SSD), never NFS home.

```bash
mkdir -p /local/nlp/$USER/ro_asi_ft
cd       /local/nlp/$USER/ro_asi_ft
```

### 3. Clone this worktree branch

From your laptop (push the branch first):

```bash
# laptop, inside the worktree
cd /Users/alexjerpelea/Romanian_ASI_mt5_finetune
git push -u origin mt5-finetune
```

On seahorse:

```bash
cd /local/nlp/$USER/ro_asi_ft
git clone <repo_url> repo
cd repo
git checkout mt5-finetune
```

### 4. Python environment

```bash
# on seahorse
python3 -m venv /local/nlp/$USER/ro_asi_ft/venv
source /local/nlp/$USER/ro_asi_ft/venv/bin/activate
pip install --upgrade pip

# Core deps from the repo, plus fine-tuning extras
pip install -r requirements.txt
pip install "transformers>=4.40" accelerate sentencepiece nltk
# torch: install the build that matches the host CUDA (A6000 = CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 5. Cache on SSD (critical — lab guide)

Put these in `~/.bashrc` or the training launch script:

```bash
export HF_HOME=/local/nlp/$USER/ro_asi_ft/hf_cache
export HF_DATASETS_CACHE=/local/nlp/$USER/ro_asi_ft/hf_cache/datasets
export TRANSFORMERS_CACHE=/local/nlp/$USER/ro_asi_ft/hf_cache/transformers
export TORCH_HOME=/local/nlp/$USER/ro_asi_ft/torch_cache
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME
```

### 6. Smoke test (<5 min)

Uses mt5-small on 200 rows to verify data pipeline + Adafactor + bf16 + checkpointing:

```bash
cd /local/nlp/$USER/ro_asi_ft/repo
python -m pipeline.ft_mt5.train \
    --model google/mt5-small \
    --max-train-samples 200 --max-val-samples 40 \
    --num-train-epochs 1 \
    --per-device-train-batch-size 2 \
    --output-dir /tmp/mt5-smoke
```

Expected: training starts, prints loss, saves `/tmp/mt5-smoke/best/`, no errors.

### 7. Full training run

```bash
cd /local/nlp/$USER/ro_asi_ft/repo
python -m pipeline.ft_mt5.train \
    --output-dir /local/nlp/$USER/ro_asi_ft/runs/mt5-large-ro-asi \
    2>&1 | tee /local/nlp/$USER/ro_asi_ft/runs/mt5-large-ro-asi.log
```

Run it under `tmux` or `nohup` so an SSH drop doesn't kill it. Expected
wall-clock on one A6000: **~8–15 h**.

While it runs, check occasionally:
```bash
nvidia-smi                       # util >80%, mem <40 GB
tail -f /local/nlp/$USER/ro_asi_ft/runs/mt5-large-ro-asi.log
```

### 8. Evaluate

```bash
python -m pipeline.eval.eval_generative \
    --model /local/nlp/$USER/ro_asi_ft/runs/mt5-large-ro-asi/best \
    --split test --backend transformers --no-similarity
```

Output: `pipeline/data/eval_results/gen_mt5-large-ro-asi_test_ro_metrics.json`.

Note the test split is **seed-word-disjoint** from train (unseen vocabulary
evaluation). Acc@1 will be low; look at Acc@5 and similarity metrics for
meaningful signal. Compare against zero-shot mT5-large on the same new split
(rerun that baseline to get a clean delta — numbers in the README are from
the old seen-vocab test).

### 9. Clean up

Per the lab guide — delete caches when done:

```bash
rm -rf /local/nlp/$USER/ro_asi_ft/hf_cache
df -h /local/nlp                 # verify freed space
```

Keep `runs/mt5-large-ro-asi/best/` if you want to re-evaluate later; otherwise
rsync it to your laptop or HF Hub and delete.

## Hyperparameters

Set in `pipeline/ft_mt5/config.py`. All values match MASIVE Appendix D unless
noted:

| | Value |
|---|---|
| Model | `google/mt5-large` |
| Optimizer | Adafactor (scale_parameter=False, relative_step=False) |
| Learning rate | 1e-4, linear decay |
| Weight decay | 0.01 |
| Per-device batch size | 4 |
| Grad accumulation | 1 |
| Epochs | 3 |
| Max input tokens | 512 |
| Max target tokens | 32 |
| Precision | bf16 + TF32 matmul |
| Seed | 42 |

## Files

| | |
|---|---|
| `resplit.py` | Regenerate 85/5/10 splits (already run) |
| `truncate.py` | Sentence-level truncation to 512 mT5 tokens, preserves mask sentence |
| `build_training_data.py` | `[MASK]` → `<extra_id_0>`; target = `<extra_id_0> word <extra_id_1>` |
| `config.py` | `TrainConfig` dataclass with all hyperparameters |
| `train.py` | HF Seq2SeqTrainer loop |
