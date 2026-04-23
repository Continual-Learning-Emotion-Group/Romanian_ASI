# Qwen3.5-4B fine-tuning on Romanian ASI

Sibling of `pipeline/ft_mt5/`. Same data, same splits, same eval — different
model family. Runs on a single A100-40GB on `tigerfish.cs.columbia.edu`.

The plan and rationale live in `~/.claude/plans/understand-this-repo-i-lovely-grove.md`.

## Differences from `ft_mt5/`

| | `ft_mt5/` | `ft_qwen/` |
|---|---|---|
| Model | `google/mt5-large` (1.2B, encoder-decoder) | `Qwen/Qwen3.5-4B` (4B, decoder-only) |
| Mask token | `<extra_id_0>` (mT5 sentinel) | literal `[MASK]` (Qwen has no sentinel) |
| Prompt | input/target text pair | chat template (system + user + assistant), `enable_thinking=False` |
| Loss | seq2seq CE on full target | causal CE, prompt tokens masked to -100 |
| Optimizer | Adafactor | AdamW (fused) |
| LR | 1e-4 (linear, no warmup) | 2e-5 (cosine, 5% warmup) |
| Attention | default | Flash Attention 2 |
| Max input | 512 mT5 tokens | 1024 Qwen tokens |
| Machine | seahorse (A6000 48GB) | tigerfish (A100-SXM 40GB, GPU 1) |

Everything else (splits, seed, batch size, epochs, weight decay, eval cadence)
is identical for direct comparability.

## Setup on tigerfish

Identical to `ft_mt5/README.md` steps 1–5 with these substitutions:

* SSH to `tigerfish.cs.columbia.edu` instead of seahorse.
* `git checkout qwen-finetune` instead of `mt5-finetune`.
* `export CUDA_VISIBLE_DEVICES=1` (GPU 0 is taken by another user as of 2026-04-23).
* Add `flash-attn` to the pip install: `pip install flash-attn --no-build-isolation`.

`/local` on tigerfish is at 99% utilization (~85 GB free). Aggressive cache
cleanup is mandatory — see `launch_tigerfish.sh`, which aborts if <40 GB free.

## Local sanity check (laptop, no GPU)

Verify chat template + loss masking before pushing to tigerfish:

```bash
python -m pipeline.ft_qwen.prompts --sanity --model Qwen/Qwen3.5-0.5B
```

Prints one tokenized example with mask diagnostics; downloads only the
tokenizer (a few MB), not model weights.

## Smoke test on tigerfish (<5 min)

```bash
python -m pipeline.ft_qwen.train \
    --model Qwen/Qwen3.5-0.5B \
    --max-train-samples 50 --max-val-samples 10 \
    --num-train-epochs 1 --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 1 \
    --output-dir /tmp/qwen-smoke
```

## Full training run

```bash
tmux new -s qwen-sft
bash pipeline/ft_qwen/launch_tigerfish.sh
```

Expected wall-clock on a single A100-40GB SXM with BF16 + FA2 + grad
checkpointing: **~1.5 h training + ~30 min final eval**.

## Evaluate

```bash
# Baseline first: re-run zero-shot Qwen3.5-9B on the new (unseen-vocab) test split
python -m pipeline.eval.eval_generative --model Qwen/Qwen3.5-9B \
    --split test --backend vllm \
    --output pipeline/data/eval_results/zeroshot_qwen3.5-9b_v2_test_metrics.json

# Then the SFT checkpoint
python -m pipeline.eval.eval_generative \
    --model /local/nlp/$USER/ro_asi_ft/runs/qwen3.5-4b-ro-asi/best \
    --split test --backend vllm \
    --output pipeline/data/eval_results/sft_qwen3.5-4b_ro_v1_test_metrics.json
```

The published Qwen3.5-9B acc@1 = 0.220 in `pipeline/data/eval_results/` was
computed against `main`'s old (seen-vocab) test set, not the new
seed-word-disjoint split. The v2 baseline above is the apples-to-apples
comparison.
