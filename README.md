# Romanian ASI · Mosaic-Emo

This repository was built as part of **Mosaic-Emo**, a native-speaker-informed
multilingual benchmark of affective states spanning eight languages (English,
Spanish, French, Romanian, Persian, Hindi, Mandarin, Indonesian), currently
under anonymous review. It contains two things:

1. **Data collection** — the full pipeline that built the Romanian side of the
   benchmark: seed lexicon construction, corpus collection, "I feel X" pattern
   extraction, LLM validation, human evaluation, and the zero-shot /
   fine-tuning evaluations. Documented in
   [`pipeline/README.md`](pipeline/README.md).
2. **Interpretability experiments** — the paper's intrinsic analysis of how a
   multilingual LLM represents affective states in its activation space,
   written up in the blog post [*The Emotion Circumplex in a Multilingual
   LLM's Activation Space*](https://alexjerpelea.com/circumplex.html).

This README focuses on the interpretability experiments, which live in
[`pipeline/affect_geometry/`](pipeline/affect_geometry/).

## The questions

An *affective state* is any term people use to describe their felt experience —
the open-vocabulary superset of the closed basic-emotion label sets used by
emotion detection datasets. We ground the analysis in Russell's (1980)
**circumplex theory of affect**, which arranges emotions on a circle in the
two-dimensional plane spanned by valence (pleasant ↔ unpleasant) and arousal
(activated ↔ deactivated). Circumplex-like geometry is known to emerge in LLM
activations, but for closed sets of standard emotions. We ask two questions:

1. Are the much broader, open-ended affective states of Mosaic-Emo explained
   by the same valence–arousal geometry?
2. Is that geometry shared between languages?

## Method

**Representations.** Qwen3-8B (bf16) is run over each occurrence of an
affective state in its original document context; hidden states are recorded
after the embedding layer and after each of the 36 transformer blocks (37
layers × 4096 dims). The hidden vectors at the state's token span are
mean-pooled per occurrence and averaged over all occurrences, giving one
centroid per state per layer. Only states with ≥ 30 occurrences are kept.

**Sets.** 𝒮 is a language's full set of affective-state centroids. 𝒜 ⊂ 𝒮 are
the *original emotions*: the language's strict, audited synonyms of Russell's
28 circumplex words (mapping in `anchors_russell.json`, full audit trail in
`results/russell_mapping_report.tsv`). Everything else, 𝒮 ∖ 𝒜, are the
*broader affective states*.

**Candidate planes.** Centroids of a fit set ℱ ∈ {𝒜, 𝒮 ∖ 𝒜, 𝒮} are
standardized and PCA-fit. We examine (1) the *leading plane* PC1+PC2, where
the circumplex appears only if it is the dominant structure of the fit set,
and (2) a *search* over pairs of the top-10 PCs and over layers for the
best-fitting plane, which can reveal a circumplex that is present but
entangled with stronger variation factors.

**Scoring.** The original emotions' centroids are projected onto the candidate
plane and compared against Russell's published coordinates via the Procrustes
disparity *D* (alignment is measured on the original emotions only — the
broader states have no prescribed position). Reported as PRE = 1 − *D*/*D*ₙᵤₗₗ,
the fraction of chance-level disparity eliminated. Significance uses the
PROTEST permutation test (5,000 shuffles); when an alignment benefits from a
plane/layer search, the null replays the identical search, so cherry-picking
the best plane cannot manufacture significance.

## Findings

- **The circumplex is the leading geometry of the broader affective states.**
  Removing every standard emotion from the fit set (ℱ = 𝒮 ∖ 𝒜) barely changes
  the leading-plane alignment in most languages — the model organizes even its
  broader affective vocabulary primarily along valence and arousal, on the
  first two principal components of the broader states' own variance.
- **Where the leading plane is weak, the circumplex is present but
  entangled.** For Romanian, Persian, and Hindi the searched planes recover it
  on non-leading components. Alignment is strongest in the middle of the
  network in every language.
- **The circumplex transfers across languages.** A plane fit and frozen on one
  language recovers every other language's circumplex (all 56 ordered pairs
  significant). Language-matched planes are rarely the best source — Mandarin,
  the largest vocabulary, is almost always the strongest.
- **The broader states obey the angular structure but leave the circle.**
  Their angles track meaning while their radii exceed the emotion circle —
  consistent with an extra degree of freedom in magnitude (offered as a
  hypothesis, not a conclusion).

A fine-tuning follow-up (multilingual ASI LoRA merged into the base model)
leaves the geometry intact and slightly strengthens the low-resource corner of
the transfer matrix — see `RESULTS_FINAL_DATA.md`.

## Code map (`pipeline/affect_geometry/`)

| Files | Purpose |
|---|---|
| `prepare.py`, `prepare_russell.py`, `prepare_new_languages.py`, `prepare_final_data.py` | Build per-language occurrence manifests from the benchmark data (frozen sampling config in `config.json`) |
| `extract.py` | Run the model over the manifests and save per-layer hidden states at the state's token span (`--adapter` for the LoRA variant) |
| `anchors_russell.json`, `results/russell_mapping_report.tsv` | Russell-word ↔ lemma anchor mapping and its 100%-coverage audit trail |
| `analyze_russell.py` | Anchor-basis fits (ℱ = 𝒜): confirmatory PC1+PC2 + corrected plane/layer search |
| `analyze_all_states.py`, `analyze_broader_only.py` | ℱ = 𝒮 and ℱ = 𝒮 ∖ 𝒜 fits (broader states carry the structure on their own) |
| `analyze_plane_share_russell.py`, `analyze_convexity_russell.py` | Controls: plane variance share, convex reconstruction of broader states from anchors |
| `basis_search_cross_language.py`, `transfer_*.py` | Cross-language transfer: frozen source planes, shared-label tiers, pairwise label intersections, fixed-layer and target-label variants |
| `plot_*.py`, `paper_style.py` | Figures; the `*_paper.py` / `*_main.py` scripts produce the paper/post figures |
| `run_final_suite.sh` | Runs the full analysis + figure suite for one extraction variant |

Results docs, in chronological order:

| Doc | Contents |
|---|---|
| `RESULTS.md` | v1: Plutchik-8 anchors, ro/en/es, Qwen3.5-4B |
| `RESULTS_RUSSELL.md` | Russell-28 redesign; six languages; anchor audit |
| `RESULTS_MODEL_COMPARISON.md` | Qwen3.5-4B vs Qwen3-8B end-to-end rerun |
| `RESULTS_FINAL_DATA.md` | Final 8-language data; base vs multilingual-SFT Qwen3-8B — the runs behind the paper and the post |

Numerical results and figures are committed under `results/{variant}/` and
`figures/{variant}/` (`qwen3-8b-final` and `qwen3-8b-joint-final` are the
final ones). Large manifests and centroid archives live under `artifacts/`
and are not committed.

## Reproducing

From the repo root, with the per-language judged data available:

```bash
# 1. Build manifests (frozen sampling config)
python -m pipeline.affect_geometry.prepare_final_data --help

# 2. Extract hidden states (GPU)
python -m pipeline.affect_geometry.extract \
  --manifest pipeline/affect_geometry/artifacts/manifests_final/en.jsonl \
  --output pipeline/affect_geometry/artifacts/hidden/qwen3-8b-final/en.npz \
  --model Qwen/Qwen3-8B --batch-size 32 --maximum-tokens 512

# 3. Full analysis + figure suite for one variant
./pipeline/affect_geometry/run_final_suite.sh qwen3-8b-final
```

## Status

Mosaic-Emo is joint work, currently under anonymous review; the paper is not
linked here yet. The blog post covers the intrinsic analysis only — the
benchmark construction, the data collection across the eight languages, the
native-speaker validation, and the extrinsic experiments are the bulk of the
paper. The Romanian data contribution is this repo's
[`pipeline/`](pipeline/README.md).
