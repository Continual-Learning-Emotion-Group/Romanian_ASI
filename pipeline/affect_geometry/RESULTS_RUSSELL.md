# Russell-Anchor Geometry (v2) Results

This document covers the Russell (1980) circumplex redesign of the affect
geometry experiment and the follow-up experiment suite. The v1 experiment
(Plutchik-8 anchors, hand-set valence/arousal coordinates, adjective-frame
Romanian manifest) is frozen and documented in `RESULTS.md`.

## 1. Anchor redesign

Targets are the 28 Russell circumplex words placed on the unit circle at
Russell's published angles (15 published directly; 13 derived from his Table 2
correlations via Ross circular scaling); see `anchors_russell.json`.

Per language, dataset lemmas (any part of speech, >= 30 occurrences) were
mapped to Russell labels under a strict-synonymy rule: include a lemma only
with high confidence of synonymy; exclude cross-label blends (suparat, upset),
adjacent-but-distinct concepts (worried != distressed), and polysemy risks
(pissed, dormido). The label "aroused" was dropped in all languages. The full
audit trail (tier, gloss, NRC-VAD two-gate angle check, exclusion reasons) is
in `results/russell_mapping_report.tsv`.

Final anchor sets: English 21 labels / 39 lemmas; Romanian 21 / 37;
Spanish 18 / 26. The 15-label intersection (the "shared" scope) is used
whenever languages are compared: happy, excited, astonished, angry, afraid,
annoyed, distressed, frustrated, miserable, sad, depressed, bored, tired,
calm, glad.

The Romanian manifest was rebuilt without the v1 adjective-frame filter
(`prepare_russell.py`): 113 eligible lemmas, 7,699 occurrences, re-extracted
with the frozen extraction config (`artifacts/hidden/ro_russell.npz`). The
English/Spanish v1 archives are reused; labels are assigned at analysis time.

## 2. Anchor-PCA analysis (analyze_russell.py)

Statistics are identical to v1: at each layer, StandardScaler + PCA fit on
anchor-lemma centroids only; confirmatory PC1+PC2 and exploratory 45-pair x
33-layer search; global permutation tests (5000 perms) that repeat all
selection inside the null.

| Language | Scope | PC1+PC2 best layer | Disparity | Corrected p |
|---|---|---:|---:|---:|
| Romanian | full (21) | 15 | 0.582 | 0.0006 |
| Romanian | shared (15) | 16 | 0.587 | 0.0118 |
| English | full (21) | 16 | 0.306 | 0.0002 |
| English | shared (15) | 18 | 0.245 | 0.0002 |
| Spanish | full (18) | 17 | 0.303 | 0.0002 |
| Spanish | shared (15) | 17 | 0.490 | 0.0020 |

With 18-21 anchors the null median disparity is ~0.80-0.87, so all fits are
far below chance. Against v1 (8 anchors): English went from non-significant
(0.452, p=0.10) to floor; the language ranking flipped (Romanian is now the
roughest fit). Searched-pair fits now also survive correction (none did in
v1); Romanian's best searched plane is PC3+PC7 at L10 (0.421, p=0.0002).

Robustness: removing Romanian noun anchors (frica, teama, groaza, liniste,
bucurie) changes the full-scope disparity from 0.582 to 0.527
(`metrics_russell_ro_adj_only.json`); the noun/adjective construction
confound is real (noun-vs-adjective displacement up to 1.17x anchor RMS
radius for bucurie/bucuros) but not decisive.

## 3. Convex-combination test (analyze_convexity_russell.py)

Question: are broader-state centroids convex combinations of the anchor label
centroids in the full 2560-dim standardized space? (The 2D version is
unfalsifiable: the disk is the convex hull of the circle.)

Per broader lemma, at the best PC1+PC2 layer: simplex-constrained least
squares against the anchor label centroids (convex R2), an unconstrained
affine-subspace projection (affine R2), and a null in which the anchor
simplex is randomly rotated within the centroid span (500 rotations).

| Language | mean convex R2 | null mean (max of 500) | hull retains of subspace fit | weight-angle circ. corr |
|---|---:|---:|---:|---:|
| Romanian (L15) | 0.207 | 0.125 | 80% | 0.06 |
| English (L16) | 0.137 | 0.039 | 79% | 0.72 |
| Spanish (L17) | 0.145 | 0.059 | 76% | 0.88 |

All p = 0.002 (floor at 500 rotations). Interpretation: absolute convex R2 is
modest because centroids carry lexical content orthogonal to affect, but the
component inside the anchor subspace is essentially convex (gap ~0.04-0.05),
and the mixture weights recover the lemma's circumplex angle (median angular
error ~17 deg in EN/ES; e.g. agobiado = distressed+tired, deranjat =
distressed+annoyed, bine = calm+happy+satisfied). The anchor leave-one-out
control fits BETTER (R2 ~0.27-0.30) than broader lemmas, so anchors are not
extreme vertices; the defensible claim is "broader states decompose into
adjacency-respecting mixtures", not "anchors are the extreme points".

Figure: `figures/russell_convexity.png`.

## 4. Plane-share comparison (analyze_plane_share_russell.py)

Question: does the anchor-PCA circumplex plane explain broader states as well
as it explains the classic emotion words themselves? Per lemma, plane share =
fraction of its squared displacement lying in the PC1+PC2 plane. Anchors are
scored held-out (leave-one-out refit of scaler+PCA); broader lemmas never
entered the PCA.

| Language | in-sample anchors (mean) | held-out anchors (mean) | broader (mean) | MW p |
|---|---:|---:|---:|---:|
| English (L16) | 24.1% | 14.4% | 4.2% | 6e-15 |
| Spanish (L17) | 27.0% | 13.2% | 4.4% | 5e-6 |
| Romanian (L15) | 23.2% | 14.5% | 8.5% | 0.002 |

The LOO correction halves the anchors' number (quantifying PCA overfitting),
but a 2-4x gap over broader states survives on the ANCHOR-FIT plane. The
lowest-plane-share broader lemmas are systematically the self-conscious and
social emotions (rusinat, umilit, dor; culpable, enamorado; misled, lied) -
exactly the states the psychological circumplex is known to miss. NOTE: the
all-states analysis (section 5) shows this gap is a property of the
anchor-fitted plane, not of the circumplex itself.

Figure: `figures/russell_plane_share_table.png`.

## 5. All-states PCA (analyze_all_states.py)

PCA fit on ALL state centroids (anchors + broader together; 503/113/251);
anchors only define the target shape. Pair search over the top-10 PCs (and
top-20 extended), selection-corrected as always.

| Language | best pair | layer | disparity | corrected p | anchor / broader mean plane share |
|---|---|---:|---:|---:|---|
| English | PC2+PC8 | 15 | 0.267 | 0.0002 | 7.8% / 8.2% |
| Romanian | PC4+PC10 | 25 | 0.330 | 0.0002 | 5.7% / 6.3% |
| Spanish | PC1+PC2 | 26 | 0.335 | 0.0002 | 11.8% / 8.9% |

Findings: (a) the circumplex is among the leading natural axes of the whole
affective vocabulary in every language - in Spanish it is literally PC1+PC2,
and the searched pair coincides with PC1+PC2 from layer ~6 through ~27;
(b) fits match or beat the anchor-only PCA (EN 0.267 vs 0.306); (c) on these
planes the anchor/broader plane-share gap from section 4 disappears - the
plane is equally "owned" by broader states, rehabilitating the shared-affect-
subspace reading.

Figures: `figures/russell_allstates_geometry_{en,ro,es}.png` (the layer-sweep
panel includes the anchor-only searched curve for comparison).

## 6. Cross-language transfer (transfer_cross_language.py)

All languages share the model, so subspaces are directly comparable. Tier (b),
frozen-plane transfer: per layer, source A's all-states PCA and its own
circumplex pair (selection entirely on A); target B is recentered by its own
mean (language offset removed), scaled with A's scaling, and projected onto
A's frozen pair; B's anchors are scored against B's Russell targets. Layer
selection on the transfer curve is permutation-corrected.

All six directions are significant at p = 0.0002 (null median ~0.83), with
transfer disparities 0.27-0.38. Best-vs-best comparison (each side at its own
best layer): native wins 4 of 6; both directions INTO Spanish are won by
foreign axes (EN->ES 0.269 and RO->ES 0.289 vs native 0.335). Variance
transfer ratios: EN<->ES ~0.81-0.82 (near-shared plane); RO-involving pairs
0.20-0.41. Principal angles between native and foreign planes are 45-79 deg
(vs ~90 deg for random planes in 2560 dims): the languages use correlated but
not identical planes drawn from a shared low-dimensional affect bundle.

Figures: `figures/russell_transfer_{en,ro,es}.png` (each target on its own
axes and on each foreign language's axes).

## 7. Cross-language basis search (basis_search_cross_language.py)

Tier (c), maximally permissive: for source basis B and target A, search ALL
45 pairs x 33 layers of B's basis for the plane that best fits A's anchors
(selection on the target side, mirroring the native searched analysis; the
permutation null repeats the full search). Two results:

**The search finds almost nothing beyond the source's own plane.** In 4 of 6
off-diagonal cells the exhaustive search returns the source's own circumplex
pair, and the best improvement over tier (b) is 0.02. Each basis contains
exactly one good circumplex plane for any target, and it is the plane the
source language uses for itself.

**Shared-15-label 3x3 table** (every cell fits the same 15-point shape; all
corrected p <= 0.0022, null medians ~0.60-0.62):

| target \ basis | EN | RO | ES |
|---|---:|---:|---:|
| English | **0.257** | 0.303 | 0.274 |
| Romanian | **0.295** | 0.346 | 0.324 |
| Spanish | **0.268** | 0.325 | 0.320 |

With equalized targets, the English basis is the best basis for ALL three
languages, including beating Romanian's and Spanish's native bases. (On the
unequal full-label sets, EN and RO appear to prefer their native bases - an
artifact of their extra labels.) The likely explanation is estimation
quality: the EN basis is fit on 503 centroids (2-4x the others) in the
model's dominant pretraining language. Pending checks before leaning on the
inversion: anchor bootstrap for cell error bars, and an EN-subsampled-to-251
control.

## Files

- Scripts: `prepare_russell.py`, `analyze_russell.py`,
  `analyze_convexity_russell.py` (+ `plot_convexity_russell.py`),
  `analyze_plane_share_russell.py` (+ `plot_plane_share_table.py`),
  `analyze_all_states.py` (+ `plot_all_states.py`),
  `transfer_cross_language.py`, `basis_search_cross_language.py`
- Anchors: `anchors_russell.json`, `anchors_russell_ro_adj_only.json`
- Results: `results/metrics_russell_*.json`, `results/projections_russell_*.json`,
  `results/selection_ro_russell.json`, `results/russell_mapping_report.tsv`,
  `results/convexity_russell.json`, `results/plane_share_russell_*.json`,
  `results/all_states_*.json`, `results/cross_language_transfer.json`,
  `results/cross_language_basis_search.json`
- Figures: `figures/russell_*.png`
