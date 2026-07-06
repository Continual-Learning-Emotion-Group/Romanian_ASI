# Hidden-State Circumplex — Step 1 (Romanian, Plutchik-8)

**Question:** do Qwen3.5-4B hidden states of Romanian emotion words arrange in
Russell's circumplex (a valence×arousal ring)?

## Method
- **Data:** `benchmark_ro_asi_clean.jsonl` (70,289 rows) → **canonical basic-emotion
  adjectives only** (Plutchik-8), several classical synonyms per emotion, gender
  variants, singular (1st-person-sg benchmark has no plurals). Balanced, cap
  600/emotion → **3,839 rows** (`out/canon_ro_input.jsonl`).
  - Adjectives-only by design: uniform `sunt / mă simt [adj]` frame removes the
    noun-vs-adjective POS confound (`mi-e frică` etc.).
  - Multiple synonyms/emotion so a vertex is an *emotion*, not one lexeme.
- **Model:** `Qwen/Qwen3.5-4B` on tigerfish (A100-40GB). Feed the *unmasked* text,
  take hidden states at the emotion-word subword tokens (mean-pooled), 13 layers.
- **Analysis:** z-score per dim → PCA. Aggregate to per-emotion centroids.
  Circumplex fit = Procrustes disparity vs theoretical NRC-VAD/Russell coords
  (0 = perfect ring, <0.3 = good match).

## Findings
1. **Naive token-cloud PCA does not show a circle.** Top-2 PCs are dominated by
   non-affective variance; silhouette ≈ 0 in the first (category-family) run,
   where a lexical/domain confound ("dor" = 43% of sadness) formed a detached blob.
2. **Clean canonical adjectives fix separability.** Silhouette becomes positive
   (peak **+0.29 at layer 8**); valence emerges as a strong axis (|r| up to 0.55–0.74).
3. **The circumplex is real but not in PC1–PC2.** In unsupervised top-2 PCA the
   four negative emotions collapse (Procrustes ≈ 0.47). But arousal is actually the
   **dominant** direction: at **layer 8**, valence = PC2 (|r|=0.74), arousal = PC0
   (|r|=0.64), and in that valence×arousal plane the 8 emotions form Russell's
   circumplex — **Procrustes disparity = 0.288**.
   - Layout: joy → anticipation → surprise(top) → fear/disgust(left) → anger →
     sadness(bottom). Valence left–right, arousal up–down.

## Key figures (`out/figs/`)
- `fig_ro_canon_valence_arousal_L8.png` — **headline: the circumplex** (disparity 0.29)
- `fig_ro_canon_headline_layer16.png` — cloud + centroids + clean valence gradient
- `fig_ro_canon_centroids_layer16.png` — per-word + per-emotion centroids
- `fig_ro_canon_layer_sweep.png` — silhouette/valence by layer

## Caveats (state these)
- Arousal axis is *identified* by picking the PC best correlated with canonical
  per-emotion arousal (axis selection, not construction). Valence is more robust.
- `trust` has one lexeme (încrezător) → weakest vertex. `fear`≈`disgust` overlap;
  `anger`/`sadness` arousal slightly off theory.
- One model / one language / mid-layer. Arousal is the largest axis of variation.

## Next
- **Tier-2/3 fill-in** (original thesis): project broader ASI states onto this
  circle — do they fill the disk or form a higher-dim manifold?
- **English + Spanish** (MASIVE) with the same model → cross-lingual alignment.
