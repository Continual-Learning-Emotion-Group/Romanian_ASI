# viz_hidden — emotion geometry in LLM hidden states

Do Qwen3.5-4B hidden states of Romanian emotion words arrange in Russell's
**circumplex** (a valence×arousal ring)? And does the broader ASI set fill that
circle or form a more complex shape? See `RESULTS.md` for Step-1 findings.

## Pipeline

```
                          ┌── select.py       category-family selection (Plutchik-8 via
                          │                    WN-Affect rollup in tiers.py). First pass;
benchmark_ro_asi_clean ───┤                    superseded by canonical.py (lexical confound).
                          │
                          └── canonical.py    PRIMARY: canonical basic-emotion ADJECTIVES
                                              only (uniform frame, synonyms/emotion, gender
                                              variants, per-emotion cap). → out/*_input.jsonl
                                                   │
                    extract.py  (on GPU box) ──────┘   feed unmasked text through Qwen3.5-4B,
                                                       mean-pool hidden states at the emotion-
                                                       word subword tokens, 13 layers.
                                                       → out/*_hidden.npz + .meta.jsonl
                                                   │
        ┌──────────────────────────────────────────┼───────────────────────────────┐
   analyze.py                  diagnose.py                 circle_scan.py / valence_arousal.py
   token-cloud PCA,            per-word + per-emotion       Procrustes circumplex fit per layer;
   layer sweep, valence        centroids, confound          find valence/arousal PC axes and view
   panel, headline fig         diagnostics                  the ring (headline result: L8, disp 0.29)
```

## Run (Romanian, Step 1)

```bash
# 1. select canonical adjectives-only Plutchik-8 (local, CPU)
python -m pipeline.viz_hidden.canonical --write --cap 600 \
    --out pipeline/viz_hidden/out/canon_ro_input.jsonl

# 2. extract hidden states on the GPU box (tigerfish: ~/ro_asi_viz/.venv, Qwen3.5-4B cached)
#    scp extract.py + canon_ro_input.jsonl, then:
python extract.py --input canon_ro_input.jsonl --out canon_ro_hidden \
    --batch-size 32 --num-save-layers 13         # ~3 min on one A100
#    scp canon_ro_hidden.npz + .meta.jsonl back to out/

# 3. analyze (local)
python -m pipeline.viz_hidden.analyze        --npz .../canon_ro_hidden.npz --meta .../canon_ro_hidden.meta.jsonl --lang ro_canon
python -m pipeline.viz_hidden.diagnose       --npz ... --meta ... --layer 16
python -m pipeline.viz_hidden.circle_scan    --npz ... --meta ...
python -m pipeline.viz_hidden.valence_arousal --npz ... --meta ... --layer 8
```

## Notes
- `out/*.npz` (hidden states) and `out/*_input.jsonl` are **git-ignored** — large and
  fully reproducible from the benchmark + these scripts. Figures in `out/figs/` are kept.
- Extraction runs on tigerfish (`~/ro_asi_viz/`), not committed; env is torch 2.10 +
  transformers 5.6 (needed for `qwen3_5`), SDPA path (no flash-linear-attention needed).
- Benchmark is first-person-**singular**, so only gender (m/f) variants exist, not plurals.
