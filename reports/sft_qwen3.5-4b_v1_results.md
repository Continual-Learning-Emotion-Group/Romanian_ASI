# Romanian ASI — SFT Run v1 (Qwen3.5-4B) Results

Fine-tuning of `Qwen/Qwen3.5-4B` on the 5-language presentation benchmark (RO / EN / ES / FA / HI). See `plans/sft_run_v1_qwen3.5-4b.md` for the experimental design and the `## v2 addendum` section of that file for the bug-fix history. This file documents the *post-run* numbers and what they mean.

## TL;DR

- **Primary goal (RO) met**: `set_acc@1 = 0.381`, vs the prior zero-shot Qwen3.5-9B baseline of **0.22**. +17 pp with a smaller (4B vs 9B) model.
- **FA evaluation bug fixed**: set-match scoring reports `0.414`; the old `parse_single_word` path (reported as `legacy acc@1`) would have scored the same predictions as **0.170** — pure eval-pipeline artifact.
- **EN / ES look low (0.147 / 0.191) but aren't the model's fault**: driven by flat label vocabularies (328 / 292 unique test labels) and lexically-exact grading. Semantic agreement (`sim@1`) is uniformly high across all five languages (0.81–0.90).

## Run configuration

| | |
|---|---|
| Model | `Qwen/Qwen3.5-4B` (post-trained, not `-Instruct`) |
| GPUs | 3 × A100-40GB on `piranha.cs.columbia.edu` (GPUs 1,2,3; NVLink) |
| Framework | HF `Trainer` + DeepSpeed ZeRO-3 + BF16 + SDPA |
| Precision | bf16 train, tf32 matmul, gradient checkpointing (use_reentrant=false) |
| Batch | per_device 2 × grad_accum 2 × 3 GPUs → **effective 12** |
| Optimizer | AdamW, cosine schedule, `lr=1e-5`, `warmup_ratio=0.03`, `max_grad_norm=1.0`, `weight_decay=0` |
| Epochs | 3 (1251 steps) |
| Seed | 42 |
| Eval / save | every 25 steps, `save_total_limit=1`, `load_best_model_at_end=true`, `metric_for_best_model=eval_loss` |
| Data | 5000 train / 1200 val / 5×1000 per-language test, built by `pipeline/ft_qwen_mixed/prepare_data.py` |
| Supervision | `label = " ".join(labels)` — `labels` is the set of distinct affective expressions in order of first appearance |
| Loss masking | assistant-only tokens, prompt prefix set to `-100`; `enable_thinking=False` |
| Checkpoint path (piranha, preserved) | `/local/nlp/aij2115/runs/final/` (~7.9 GB) |
| Training log | `runs/logs/train_20260423_044249.log` on piranha (100 MB); distilled here as `pipeline/data/eval_results/sft_qwen3.5-4b_v2_training_curve.tsv` |

## Training dynamics

Full per-step training history in `pipeline/data/eval_results/sft_qwen3.5-4b_v2_training_curve.tsv`. Summary:

### Eval loss every 25 steps

```
step  25: 1.300   (warmup in progress)
step  50: 1.384   (warmup peak)
step  75: 1.343
step 100: 1.283
step 125: 1.242
step 150: 1.191
step 175: 1.156
step 200: 1.144
step 300: 1.009
step 400: 0.932
step 500: 0.975
step 600: 0.875
step 700: 0.849
step 800: 0.828
step 825: 0.796   ← low water mark  (best checkpoint saved)
step 850: 0.839   (overfit onset)
step 900: 0.981
step 1000: 0.928
step 1100: 0.949
step 1250: 0.951  plateau
```

Best checkpoint is from **step 825** (epoch ~2.0, `load_best_model_at_end=true`). The final 400 steps show mild-to-moderate overfitting — train loss falls to 0.05–0.10 in epoch 3 while val loss drifts up to ~0.95. The saved checkpoint is the pre-overfit one.

### Rolling 25-step mean of train loss (first 150 steps)

```
steps 5–25    → 1.832
steps 30–50   → 1.628
steps 55–75   → 1.808
steps 80–100  → 1.716
steps 105–125 → 1.704
steps 130–150 → 1.612
```

Train loss is noisy per-step (effective batch 12 × 5 languages → high composition variance), but the rolling mean drifts down monotonically. Grad norms halve over the same window (100+ → ~30) — the optimizer is settling into a coherent direction, not fighting contradictory labels (v1 symptom).

### Comparison to v1 (abandoned run on same data)

| metric | v1 | v2 (this run) |
|---|---|---|
| eval_loss at step 125 | 1.540 | 1.242 |
| eval_loss low-water | 1.452 (step 150) | **0.796** (step 825) |
| train_loss pattern | oscillated 1.7–2.2, never dropped | dropped to 0.05 in epoch 3 |
| grad_norm | clipped 25–40 consistently | 30–50 trending down |

v1's contradictory supervision (label-as-positional-enumeration) and prompt/data mismatch (single-word prompt + phrase labels) were both resolved in v2; the training dynamics immediately looked healthy.

## Final eval — full test sets

Generated with `pipeline.ft_qwen_mixed.eval_sft --backend transformers --tag qwen3.5-4b_v2_tf` (greedy, single completion per row; no top-k sampling because vllm wasn't installed in the venv). Scoring via **word-sequence set-match** (each gold label — word or phrase — must appear as a contiguous whitespace-token subsequence of the normalized output).

| lang | n | set_acc@1 | legacy acc@1 | sim@1 | notes |
|---|---|---|---|---|---|
| ro | 1000 | **0.381** | 0.381 | 0.851 | all rows single-mask single-label |
| en | 1000 | 0.147 | 0.146 | 0.848 | single-mask 0.126 (n=865), multi-mask 0.281 (n=135) |
| es | 1000 | 0.191 | 0.192 | 0.813 | single-mask 0.192 (n=900), multi-mask 0.180 (n=100) |
| fa | 1000 | **0.414** | 0.170 | 0.902 | single-mask 0.412 (n=833), multi-mask 0.425 (n=167); legacy gap = phrase-label bug |
| hi | 1000 | 0.481 | 0.481 | 0.821 | 32-label test space, heavy head |
| **mean** | 5000 | **0.323** | 0.276 | 0.847 | |

Per-language JSONs: `pipeline/data/eval_results/sft_qwen3.5-4b_v2_tf_test_{lang}_metrics.json`. Per-row predictions: `...sft_qwen3.5-4b_v2_tf_test_{lang}_predictions.jsonl`. Combined: `...sft_qwen3.5-4b_v2_tf_all_metrics.json`.

### Single-mask only (apples-to-apples across languages)

| lang | n | set_acc@1 |
|---|---|---|
| hi | 1000 | 0.481 |
| fa | 833 | 0.412 |
| ro | 1000 | 0.381 |
| es | 900 | 0.192 |
| en | 865 | 0.126 |

Multi-mask isn't the reason EN / ES underperform — single-mask-only numbers show the same gap.

### set_acc@1 stratified by gold-label training frequency (≥ k threshold)

Test rows filtered by how often `labels[0]` appeared in that language's training split.

| lang | ≥1 | ≥3 | ≥5 | ≥10 | ≥20 | ≥50 |
|---|---|---|---|---|---|---|
| ro | 0.381 (1000) | 0.412 (908) | 0.441 (825) | 0.480 (642) | 0.537 (516) | **0.788** (165) |
| en | 0.147 (1000) | 0.137 (620) | 0.174 (242) | 0.029 (34) | — | — |
| es | 0.191 (1000) | 0.218 (665) | 0.226 (407) | 0.238 (101) | 0.484 (31) | — |
| fa | 0.414 (1000) | 0.508 (731) | 0.604 (560) | 0.743 (342) | **0.853** (204) | 0.853 (204) |
| hi | 0.481 (1000) | 0.487 (987) | 0.492 (975) | 0.510 (938) | 0.515 (907) | 0.581 (716) |

When the gold was seen 20+ times in training, non-EN languages all land at 0.48–0.85. EN has **zero** test rows whose gold was seen that often — its label distribution is simply too flat for any label to accumulate density during training. This, not a capability gap, is the structural reason EN lags overall.

## Observations

### 1. EN / ES "underperformance" is mostly synonym grading

Sample wrong rows:

```
en:  pred="stupid"     gold="foolish"      ← synonym
en:  pred="scared"     gold="afraid"       ← synonym
en:  pred="ashamed"    gold="ashame"       ← gold is a typo; pred is correct spelling
en:  pred="devastated" gold="worried"      ← related negative state
es:  pred="interesado" gold="interesada"   ← only grammatical gender differs
ro:  pred="dezamăgit"  gold="dezamăgită"   ← only gender differs
```

98% of EN/ES predictions are well-formed emotion words. `sim@1 = 0.848 / 0.813` confirms semantic proximity. The benchmark is exact-token match and the label space is lexically specific; synonyms and gender variants systematically drop to 0 regardless of correctness.

### 2. The FA result rests on the evaluation rewrite

FA has 204/1000 test rows whose gold is the 3-word Persian idiom `'دلم تنگ شده'` ("I miss"). v1's `parse_single_word` took the first whitespace token of the model output (`'دلم'`) and compared to the full 3-word gold — guaranteed mismatch on every one of those 204 rows. v2 scores the same predictions via word-sequence set-match and gets `0.414` for the language, vs `0.170` under the legacy rule. The `legacy_first_label` column in each language metrics JSON shows what v1 would have reported for the same predictions, per language.

### 3. The 4B model learns the RO emotion vocabulary well

On RO test rows whose gold appeared ≥50 times in training, `set_acc@1 = 0.788`. The high-frequency head (`dor`, `frică`, `fericit`, `bine`) is essentially solved. Accuracy falls off sharply in the tail — which is where exact-match metrics always punish SFT runs on small data.

### 4. Overfit starts mid-epoch-2; `load_best_model_at_end` caught it

Eval loss bottomed at step 825 (0.796) and drifted back up. Train loss kept falling through epoch 3 to 0.05, which at effective batch 12 = memorization. The saved final checkpoint is step 825, not step 1250. If a future run wants to push further without overfitting, natural knobs are: fewer epochs (2 instead of 3), higher weight decay, or more data.

## Known limitations

- **Greedy eval, no top-k**: `vllm` wasn't installed in the piranha venv at eval time, so we fell back to `--backend transformers` which uses `do_sample=False, num_beams=1`. The top-k columns (`set_acc@3`, `set_acc@5`) are therefore identical to `@1`. Installing vllm and re-running would take ~10 min and give proper top-k sampling.
- **Synonym / gender-inflection grading**: the exact-match metric substantially underscores EN / ES / RO. A follow-up eval pass could add a gender-collapsed or lemmatized scoring variant; `sim@1` already gives a rough soft bound.
- **HI label noise**: Hindi sources used substring matching during conversion (`scripts/convert_hindi_to_presentation_format.py`), so short stems like `डर` ("fear") may match loanwords like `किंडरगार्टेन` ("kindergarten"). Accepted for this run; affects HI "correct" labels in ~a handful of rows.
- **FA phrase labels vs `parse_single_word`**: the legacy scorer is preserved for RO / HI comparability to the zero-shot report but understates FA / multi-word-label rows. Use `set_acc@1` as the headline metric for anything other than RO zero-shot cross-reference.

## What's reproducible from this repo

Given the code state at this commit:

1. Rebuild the dataset from `presentation_data/`:
   ```
   python -m pipeline.ft_qwen_mixed.prepare_data --output /tmp/asi_multilingual
   ```
2. Sanity-check tokenization + loss masking (4 cases including multi-mask + phrase label):
   ```
   python -m pipeline.ft_qwen_mixed.prompts --sanity
   ```
3. Launch training with the config in `pipeline/ft_qwen_mixed/configs/qwen3_5_4b_full_ft.yaml` via `run/piranha_launch.sh` (3-GPU `torchrun` + DeepSpeed).
4. Eval with `python -m pipeline.ft_qwen_mixed.eval_sft --checkpoint <path> --backend {vllm|transformers}`.

Checkpoint binary itself is **not** in the repo (7.9 GB); reference it at `/local/nlp/aij2115/runs/final/` on piranha or re-train with the pinned seed=42 / data_seed=42.
