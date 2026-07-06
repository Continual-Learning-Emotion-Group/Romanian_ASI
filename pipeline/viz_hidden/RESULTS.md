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

# Step 2 — Fill-in: do the broader ASI states fill the circle, or a complex shape?

**Question (original thesis):** project the *full* affective-state vocabulary onto
the Step-1 circumplex. Do the broader states fill the 2D disk, or live on a
higher-dimensional shape?

## Method
- **Data:** full `benchmark_ro_asi_clean.jsonl` → **89 lexemes** (`fillin_select.py`):
  29 **anchors** (canonical basic-8 synonyms) + **60 broader "fill" states**
  (e.g. `entuziasmat, coplesit, recunoscator, vinovat, frustrat, disperat, calm`).
  - **Adjective-frame only** (`pattern_used` ∈ copula/`ma simt` set) → POS held
    constant, same discipline as Step 1. Noun frames (`mi-e frica`) dropped.
  - **Gender-merged** to a lexeme key (fericit/fericita → fericit).
  - **Independent labels** from RoEmoLex V3: family (argmax of 8 columns) + valence
    (Pozitivitate−Negativitate); benchmark `emotion_category` kept as cross-check.
  - 9,313 rows (cap 150/lexeme), same Qwen3.5-4B extraction (`run3`).
- **Train/test framing:** the valence×arousal plane is defined from **anchors
  only**; the 60 fill lexemes are **held out** and projected in.
- **Rigor:** intrinsic dimensionality is measured with a **split-half reliability**
  test (each lexeme's rows split in two) so only axes with reproducible
  between-lexeme signal are counted — raw effective dim is noise-inflated.

## Findings (layer-robust across layers 5–21)
1. **Valence generalizes to unseen states.** Held-out fill lexemes land along the
   anchor-defined valence axis consistent with independent RoEmoLex valence:
   **r = 0.49 (L8) → 0.59–0.62 (L13–16)**. The circumplex valence axis is real
   beyond the basic 8.
2. **They do NOT collapse onto the 2D disk.** The valence×arousal plane holds only
   **~20–25%** of between-lexeme centroid variance (uniform would be 6.7% → the
   plane is ~3× denser, but still a minority). The centroid spectrum is flat
   (PC1≈12%, 16 PCs for 80%).
3. **The affective geometry is genuinely high-dimensional.** Split-half reliability
   confirms **~50–60 reliable dimensions** (r>0.5), decaying to noise only past
   PC~60 — reliable **effective dim ≈ 14–17**. Each state has a stable, distinct
   position. → **the broader ASI states live on a complex, high-dimensional
   manifold; the circumplex is a real but small 2D projection of it.**
4. **A reliable 3rd axis beyond valence/arousal.** High: `furios, iritat, jignit,
   rusinat, jenat, ofensat` (anger + shame/embarrassment — hostile/self-conscious);
   low: `extraordinar, excelent, pasionat, entuziasmat, fermecat, curios`
   (excited-appetitive positive). An approach/appraisal-like axis (loosely PAD
   "dominance").

## Key figures (`out/figs/`)
- `fig_fillin_plane_L8.png` — 8 anchors (○) + 60 broader states (□) in the plane
- `fig_fillin_scree_L8.png` — flat centroid spectrum (intrinsic dimensionality)
- `metrics_fillin_L8.json` — generalization r, reliable dim, plane fraction

## Caveats
- RoEmoLex family argmax is noisy for some words (`disperat`→fear vs bench sadness,
  `iubit`→joy vs bench trust); valence poles are the robust independent signal.
- Arousal axis is anchor-defined only (RoEmoLex has no arousal) — fill words'
  arousal position is model-derived, not independently validated.
- `trust` is thin (3 lexemes). Plane figure is dense (60+ labels).

## Next
- **English + Spanish** (MASIVE) with the same model → cross-lingual alignment:
  is the same ~15-dim structure shared across languages?
- De-clutter the plane figure; optionally add a 3-D (valence, arousal, PC3) view.
