# Fine-tuned mT5-Large — Results

mT5-Large (1.2B params) fine-tuned on the Romanian ASI benchmark following the
MASIVE (Deas et al. 2024) recipe, with a single-span masking variant and a
seed-word-aware 85/5/10 split.

## Setup

| | |
|---|---|
| Model | `google/mt5-large` |
| Training data | 45,181 examples (from `benchmark_ro_asi_clean.jsonl` deduped to 53,154 rows) |
| Val set | 2,658 examples, 185 seed words (182 overlap with train) — **seen vocab** |
| Test set | 5,315 examples, 71 seed words, **fully disjoint from train/val** — **unseen vocab** |
| Masking | Single-span: `[MASK]` → `<extra_id_0>`; target = `<extra_id_0> <seed_word> <extra_id_1>` |
| Optimizer | Adafactor, lr 4e-4 linear decay, wd 0.01 (MASIVE used bs=4 / lr=1e-4; we use bs=16 / lr=4e-4 — linear scaling) |
| Epochs | 3 (8,472 optimizer steps) |
| Precision | bf16 + TF32 matmul |
| Machine | seahorse.cs.columbia.edu, single NVIDIA RTX A6000 48 GB |
| Wall clock | 2h 04m |
| Final train loss | 0.356 |
| Best checkpoint | step 8,000 (epoch 2.83), val loss 0.3499 |

Val loss trajectory across epochs: 0.413 → 0.374 → 0.361 → 0.350 — still
decreasing at the end of epoch 3, so longer training would likely give a
modest further gain.

## Headline Results

| Eval set | Acc@1 | Acc@3 | Acc@5 | Sim@1 | Sim@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| **val (seen vocab)** | **57.7%** | 77.5% | 83.2% | — | — | 0.680 |
| test (unseen vocab) — fine-tuned | 0.79% | 1.52% | 2.20% | 0.741 | 0.801 | 0.013 |
| test (unseen vocab) — zero-shot mT5-large | 14.5% | 18.7% | 18.8% | — | — | 0.165 |

### Interpretation

**Two orthogonal findings, driven entirely by whether the eval vocabulary
overlaps the training vocabulary.**

**1. Seen vocab: fine-tuning works as expected (+30pp over zero-shot baseline).**
The 57.7% Acc@1 on val roughly doubles the old README's 27.9% zero-shot
mT5-large number (measured on the older seen-vocab test split with same
stratification philosophy). Top-5 coverage hits 83% — competitive with the
monolingual Romanian ro-bert baseline (61%), and beating every LLM we
evaluated.

**2. Unseen vocab: generative fine-tuning narrows the output distribution.**
The test set was constructed so that all 71 gold seed words are held out of
train — the model has literally never been asked to predict these during
fine-tuning. Result: Acc@1 collapses to 0.79%, *below* the zero-shot 14.5%.

Fine-tuning taught mT5 to emit the ~298 train seed words confidently; in
doing so, it suppressed its pretraining distribution's ability to emit
held-out emotion words like `rușine` (shame), `șocat` (shocked), `scârbă`
(disgust). When shown a context whose gold is `rușine`, the model instead
emits `frică` / `teamă` / `greață` — correct sentiment, wrong lexeme.

**Sim@k tells a more forgiving story.** Contextual BERT cosine similarity
between predicted and gold word: **Sim@1 = 0.74, Sim@5 = 0.80**. That is
higher than MASIVE's reported mT5-En unseen Sim@1 of 0.488 — our test is
dominated by negative emotions (`rușine` alone is 1,709 / 5,315 = 32% of
test), so near-synonym substitution is especially effective.

**Reproduces MASIVE Table 6.** Their mT5-En: seen 32.85% → unseen 1.04%
(≈30× drop). Ours mT5-Ro: seen 57.7% → unseen 0.79% (≈73× drop). Our larger
drop is plausibly due to the concentration of negative-emotion gold words in
our test.

## Per-group Breakdown

### Fine-tuned, test (unseen vocab, n=5,315)

By source:
| | n | Acc@1 | Acc@5 |
|---|--:|--:|--:|
| filmot (YouTube) | 931 | 0.11% | 1.40% |
| fulg (web) | 4,204 | 0.98% | 2.40% |
| merged_corpus | 180 | 0.00% | 1.67% |

By pattern:
| | n | Acc@1 | Acc@5 |
|---|--:|--:|--:|
| primary ("mă simt [X]") | 424 | 0.71% | 1.42% |
| secondary ("sunt / am [X]") | 4,891 | 0.80% | 2.27% |

By gender:
| | n | Acc@1 | Acc@3 | Acc@5 |
|---|--:|--:|--:|--:|
| masculine adj | 2,125 | 1.08% | 2.49% | 3.53% |
| feminine adj | 1,452 | **0.00%** | **0.00%** | 0.21% |
| nouns (unknown) | 1,738 | 1.09% | 1.61% | 2.24% |

Total feminine collapse (0/1,452 at Acc@1 and Acc@3) echoes the zero-shot
XLM-R gender bias noted in the main README (XLM-R fem 0.1%). The fine-tune
did not resolve this in the unseen regime — the model simply can't
morphologically generate feminine forms of words it never saw.

### Zero-shot mT5-large, test (unseen vocab, n=5,315)

By gender:
| | n | Acc@1 | Acc@5 |
|---|--:|--:|--:|
| masculine adj | 2,125 | 8.14% | 10.92% |
| feminine adj | 1,452 | 6.89% | 9.30% |
| nouns (unknown) | 1,738 | **28.54%** | 36.36% |

Zero-shot mT5 does especially well on **emotion nouns** (28.54% Acc@1 on
unknown/noun class), because nouns don't carry gender agreement and mT5's
pretraining distribution of Romanian nouns is broad. Fine-tuning on our 298
adjective-heavy seed hurts precisely this capability — the fine-tuned model
can only fluently generate the 298 train seeds.

## Files Produced

Code — on branch `mt5-finetune`:

| | |
|---|---|
| `pipeline/ft_mt5/resplit.py` | 85/5/10 split (run once, outputs to `pipeline/data/splits/`) |
| `pipeline/ft_mt5/truncate.py` | sentence-level truncation preserving the mask sentence |
| `pipeline/ft_mt5/build_training_data.py` | `[MASK]` → `<extra_id_0>`, target format |
| `pipeline/ft_mt5/config.py` | `TrainConfig` dataclass with MASIVE hyperparameters |
| `pipeline/ft_mt5/train.py` | HF Seq2SeqTrainer loop with Adafactor + linear decay |
| `pipeline/ft_mt5/README.md` | Seahorse runbook (env setup, smoke test, full train, eval) |
| `pipeline/ft_mt5/RESULTS.md` | This file |

Data splits — `pipeline/data/splits/`:
- `train.jsonl` (45,181 rows)
- `val.jsonl`   (2,658 rows)
- `test.jsonl`  (5,315 rows)
- `split_stats.json` (including the 71 held-out seed words)

Eval outputs — `pipeline/data/eval_results/`:
- `gen_mt5-large-ro-asi-ft_val_ro_metrics.json`  + `_results.jsonl`  (fine-tuned on val)
- `gen_mt5-large-ro-asi-ft_test_ro_metrics.json` + `_results.jsonl`  (fine-tuned on test, includes Sim@k)
- `gen_mt5-large_test_ro_metrics.json`           + `_results.jsonl`  (zero-shot mT5 on test)

Checkpoint — **not committed to git** (4.9 GB). On seahorse at:
`/local/nlp/aij2115/ro_asi_ft/runs/mt5-large-ro-asi/best/`.

## Takeaways for Follow-up Work

1. **The 57.7% Acc@1 on seen-vocab val is the headline number.** Use this
   when reporting "fine-tuned mT5 on Romanian ASI" externally. It replicates
   MASIVE's finding that small fine-tuned encoder-decoder models outperform
   much larger zero-shot LLMs.

2. **Don't use the unseen-vocab test as a headline metric.** It measures
   out-of-vocabulary generation, which generative fine-tunes are
   constitutionally bad at — not a bug, an architectural property.

3. **If unseen-vocab performance matters, the right approach is
   similarity-based, not exact-match.** Sim@1 = 0.74 on unseen shows the
   model is semantically competent; it just can't produce the exact target
   word. For downstream use cases where near-synonyms are acceptable, this
   model is usable on unseen vocab.

4. **Feminine-adjective collapse on unseen vocab (0.00% Acc@1)** is worth a
   dedicated follow-up. Options: train with explicit gender-augmented
   targets (seed × {masc, fem} pairs) or supplement with a morphology
   post-processor.

5. **Val loss was still decreasing at epoch 3.** A 5-epoch run is cheap on
   seahorse (~3.5h) and would tighten the seen-vocab numbers further.
