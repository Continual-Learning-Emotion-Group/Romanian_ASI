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

Final anchor sets (after the 2026-07-30 full-archive sweep, below): English
21 labels / 40 lemmas; Romanian 22 / 38; Spanish 18 / 26; Mandarin 25 / 67;
Persian 20 / 43; Hindi 19 / 39.
The "shared" scope (used whenever languages are compared) is the label
intersection over all six languages — 11 labels: happy, excited, astonished,
angry, afraid, distressed, sad, depressed, bored, tired, calm. (With the
original three languages it was 15 labels; adding zh/fa/hi dropped annoyed,
frustrated, miserable, glad. Angles remain well spread, 7.8-316.2 deg.)

The Romanian manifest was rebuilt without the v1 adjective-frame filter
(`prepare_russell.py`): 113 eligible lemmas, 7,699 occurrences, re-extracted
with the frozen extraction config (`artifacts/hidden/ro_russell.npz`). The
English/Spanish v1 archives are reused; labels are assigned at analysis time.

## 1b. Extension languages: Mandarin, Persian, Hindi (2026-07-30)

Source data (`dump_new_data/`, from the Continual-Learning-Emotion-Group
repos): Mandarin 905,502 MASIVE-format rows and Persian 271,479 masked-context
rows, both already LLM-filtered (only llm_score=3 kept); Hindi 68,101 rows
from `Hindi_all.csv`, which has no llm_score column and no language config in
the Experiments/llm_validation repo — Hindi was never LLM-filtered, so all of
it is used.

Manifests are built by `prepare_new_languages.py` with the same frozen
sampling config (>= 30 occurrences/lemma, <= 80 contexts/lemma, 350-char
windows, seed-stable ordering) and language-specific normalization: zh NFKC;
fa NFKC plus Arabic-letter unification (ye/kaf) and tatweel removal; hi NFC
only — the generic normalizer strips combining marks, which would destroy
Devanagari vowel signs. Persian multi-mask rows are filled from mask_labels
and matched against the state (somatic idiom templates like "delam ... gerefte"
are kept as template lemmas); rows with zero or ambiguous matches are dropped
(124 / 0). Hindi contexts use the `[[...]]`-marked window, falling back to the
bare sentence for the 13,145 v4_legacy rows with empty context; 4,757 rows
where only a derived form of the annotated word appears (e.g. thaka vs thakan)
are skipped; feminine forms in -ii are merged into the -aa lemma only when
both occur. Resulting manifests: zh 761 lemmas / 58,017 occurrences;
fa 247 / 15,053; hi 118 / 8,880.

Extraction uses the identical frozen config (Qwen3.5-4B rev 851bf6e, bf16,
mean over target tokens, 33 layers x 2560), run on a Columbia A100; all rows
valid in all three archives (`artifacts/hidden/{zh,fa,hi}.npz`).

Anchor mapping follows the same strict-synonymy rule via manual gloss checks
(NRC-VAD gate not available for these languages); the audit rows are appended
to `results/russell_mapping_report.tsv`. Notable calls: Persian idiom
templates were checked for conjunct homogeneity — the "gerefte" templates are
97-98% self-conjunct and enter as gloomy anchors, while "delam ... shode/shod"
are heterogeneous and stay broader-only; excluded blends mirror the earlier
languages (zh tongku = distressed/miserable blend, fa naarahat = the upset
case, hi bechain = restless, adjacent to but not distressed).

**Full-archive sweep (2026-07-30).** The initial mapping pass reviewed only
the frequent lemmas plus targeted per-label searches; a completeness audit
then swept every remaining archive lemma in all six languages (1,662 audit
rows added; the mapping TSV now covers 100% of all archives). Findings:
Romanian was missing the "relaxed" label entirely despite `relaxat` being in
the archive (ro 21 -> 22 labels); Mandarin gained 18 mostly literary/variant
synonyms of existing anchors; Persian gained 4 (two clitic variants, two
literary forms); English gained "spent" (tired); Spanish and Hindi gained
nothing. Crucially, the sweep confirmed the 11-label shared scope is a
property of the corpora, not an oversight: Hindi's archive has no strict
synonym for annoyed, frustrated, glad, or miserable, and Mandarin none for
miserable or droopy. All headline conclusions below are stable under the
expanded anchors; the one shift is that English and Mandarin become
statistically indistinguishable co-leaders in the confirmatory anchor fit
(previously Mandarin led).

## 2. Anchor-PCA analysis (analyze_russell.py)

Statistics are identical to v1: at each layer, StandardScaler + PCA fit on
anchor-lemma centroids only; confirmatory PC1+PC2 and exploratory 45-pair x
33-layer search; global permutation tests (5000 perms) that repeat all
selection inside the null.

Full-scope table for all six languages (searched = best of 45 PC pairs x 33
layers, selection-corrected):

| Language | Labels | PC1+PC2 | Layer | p | Searched | Layer | p |
|---|---:|---:|---:|---:|---|---:|---:|
| English | 21 | 0.299 | 16 | 0.0002 | PC1+PC2 0.299 | 16 | 0.0002 |
| Spanish | 18 | 0.303 | 17 | 0.0002 | PC1+PC2 0.303 | 17 | 0.0002 |
| Mandarin | 25 | 0.310 | 25 | 0.0002 | PC1+PC2 0.310 | 25 | 0.0002 |
| Romanian | 22 | 0.600 | 15 | 0.0008 | PC3+PC7 0.428 | 18 | 0.0002 |
| Persian | 20 | 0.592 | 12 | 0.0008 | PC2+PC4 0.376 | 14 | 0.0002 |
| Hindi | 19 | 0.748 | 11 | 0.0874 | PC3+PC6 0.432 | 15 | 0.0006 |

With 18-25 anchors the null median disparity is ~0.67-0.87, so all
significant fits are far below chance. Against v1 (8 anchors): English went
from non-significant (0.452, p=0.10) to floor; the language ranking flipped
(Romanian is no longer the best fit). Searched-pair fits now also survive
correction (none did in v1).

The six languages split into two clear profiles. English, Spanish, and
Mandarin carry the circumplex directly in PC1+PC2 (their searched pair IS
PC1+PC2, fits 0.30-0.31, all at floor; English and Mandarin are
statistically indistinguishable co-leaders). Romanian, Persian, and Hindi
carry it on non-leading PCs: their confirmatory PC1+PC2 planes are
weak-to-non-significant, but a searched pair fits well and survives full
selection correction (0.38-0.43, p <= 0.0006).

Shared-scope (11 labels) results are lower-powered — the null median drops to
~0.45-0.48, so only strong fits clear it: English stays at floor
(0.238 @L32, p=0.0002), Spanish p=0.0154, Mandarin searched p=0.0104
(confirmatory p=0.102), Romanian searched p=0.0164 (confirmatory ns),
Hindi searched p=0.0062 (confirmatory ns), Persian ns (p~0.15-0.29).
Interpret shared-scope non-significance as low power, not absence: every
full-scope searched fit is significant.

Robustness: removing Romanian noun anchors (frica, teama, groaza, liniste,
bucurie) changes the full-scope disparity from 0.582 to 0.527
(`metrics_russell_ro_adj_only.json`); the noun/adjective construction
confound is real (noun-vs-adjective displacement up to 1.17x anchor RMS
radius for bucurie/bucuros) but not decisive.

Figures: `figures/russell_anchor_geometry_{en,ro,es,zh,fa,hi}.png`
(3 panels per language: confirmatory plane, best searched plane, layer sweep
with null median).

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

PCA fit on ALL state centroids (anchors + broader together;
503/113/251/761/247/118 for en/ro/es/zh/fa/hi); anchors only define the
target shape. Pair search over the top-10 PCs (and top-20 extended),
selection-corrected as always.

| Language | best pair | layer | disparity | corrected p | anchor / broader mean plane share |
|---|---|---:|---:|---:|---|
| English | PC2+PC8 | 15 | 0.284 | 0.0002 | 7.7% / 8.2% |
| Romanian | PC4+PC10 | 25 | 0.320 | 0.0002 | 5.7% / 6.4% |
| Spanish | PC1+PC2 | 26 | 0.335 | 0.0002 | 11.8% / 8.9% |
| Mandarin | PC1+PC4 | 24 | 0.278 | 0.0002 | 14.3% / 6.9% |
| Persian | PC3+PC9 | 12 | 0.262 | 0.0002 | 8.6% / 6.1% |
| Hindi | PC5+PC9 | 17 | 0.433 | 0.0008 | 6.2% / 6.2% |

Findings: (a) the circumplex is among the leading natural axes of the whole
affective vocabulary in every language - in Spanish it is literally PC1+PC2
(and Mandarin's plain PC1+PC2 fits at 0.300, p=0.0002, before any search);
(b) fits match or beat the anchor-only searched PCA in every language
(Persian 0.262 vs 0.376 anchor-only is the largest improvement - with 247
centroids the plane is estimated far better than from 43 anchors);
(c) on these planes the anchor/broader plane-share gap from section 4
disappears - the plane is equally "owned" by broader states, rehabilitating
the shared-affect-subspace reading.

Figures: `figures/russell_allstates_geometry_{en,ro,es,zh,fa,hi}.png` (the
layer-sweep panel includes the anchor-only searched curve for comparison).

### 5b. Broader-only PCA control (analyze_broader_only.py)

Strict version of section 5: scaler + PCA fit on broader-state centroids ONLY
(anchors fully excluded from the fit — they influence neither standardization
nor axes), anchors projected in, same top-10 pair search + selection-corrected
permutation test. Anchors are strictly out-of-sample.

| Language | best pair | layer | disparity | corrected p | anchor / broader mean plane share |
|---|---|---:|---:|---:|---|
| English | PC3+PC7 | 32 | 0.294 | 0.0002 | 7.6% / 6.7% |
| Romanian | PC1+PC5 | 8 | 0.455 | 0.0008 | 6.5% / 11.8% |
| Spanish | PC2+PC3 | 11 | 0.263 | 0.0002 | 7.6% / 9.0% |
| Mandarin | PC1+PC2 | 12 | 0.306 | 0.0002 | 15.9% / 9.0% |
| Persian | PC3+PC9 | 13 | 0.374 | 0.0002 | 5.0% / 6.9% |
| Hindi | PC9+PC10 | 14 | 0.555 | 0.0468 | 2.2% / 5.2% |

Findings: (a) the circumplex survives with the anchors held out entirely —
five of six languages fit at 0.26–0.46 with p <= 0.0008, comparable to the
anchor-only searched fits; in Mandarin the circumplex is literally PC1+PC2 of
the broader states alone; (b) held-out anchor plane share is comparable to
(in-sample) broader share in every language, so the plane is a genuinely
shared subspace, not an anchor artifact; (c) Hindi degrades (0.555, p=0.047,
marginal) — the smallest vocabulary (79 broader lemmas) estimates the basis
worst; (d) English's best layer is 32 (final), unlike the mid-layer optima
elsewhere. Output: `results/broader_only_{lang}.json`.

## 6. Cross-language transfer (transfer_cross_language.py)

All languages share the model, so subspaces are directly comparable. Tier (b),
frozen-plane transfer: per layer, source A's all-states PCA and its own
circumplex pair (selection entirely on A); target B is recentered by its own
mean (language offset removed), scaled with A's scaling, and projected onto
A's frozen pair; B's anchors are scored against B's Russell targets. Layer
selection on the transfer curve is permutation-corrected.

With six languages there are 30 ordered directions, and ALL 30 are
significant (27 at the p=0.0002 floor; max p=0.0012; null medians
~0.79-0.87). Transfer disparities span 0.216-0.551; the single best
cross-language fit anywhere is Mandarin's plane on Spanish anchors (0.216).
Best-vs-best against each target's own all-states plane (section 5), the
best foreign plane beats the native plane for five of six targets:
ZH->EN 0.252 vs 0.284, FA->RO 0.261 vs 0.320, ZH->ES 0.216 vs 0.335,
ZH->FA 0.222 vs 0.262, FA->HI 0.350 vs 0.433; only Mandarin narrowly
retains its own plane (EN->ZH 0.285 vs native 0.278). Mandarin and Persian
are the strongest exporters (with three languages foreign-beats-native was
true only for Spanish as a target). Variance transfer ratios: EN->ES is the
most shared direction (0.81; ES->EN 0.48 after the sweep); other pairs move
0.05-0.67 of the variance. Principal angles between native and foreign planes are 45-90 deg
(vs ~90 deg for random planes in 2560 dims): correlated but not identical
planes drawn from a shared low-dimensional affect bundle.

Figures: `figures/russell_transfer_{en,ro,es,zh,fa,hi}.png` (each target on
its own axes and on each of the five foreign languages' axes).

## 6b. Frozen transfer on shared labels (transfer_shared_labels.py)

Paper-table version of the frozen transfer: same protocol as section 6, but
every cell (including the native diagonal) is scored on the 11 circumplex
labels shared by all six languages, so all 36 cells fit the identical
11-point shape and are comparable in every direction. Layer selection is
label-permutation corrected. Output: `results/transfer_shared_labels.json`.

29/30 off-diagonal cells significant (only ro->hi fails, D=0.535 p=0.083;
es->hi marginal, 0.474 p=0.042; null medians ~0.64-0.75). Off-diagonal D
range 0.174-0.535. Column winners: en target -> native en 0.197 (zh 0.198
a 0.001 tie); es -> en 0.261; zh -> en 0.174; ro -> zh 0.242; fa -> zh
0.186; hi -> fa 0.265. Foreign matches or beats native for all six targets
(beats for five, ties for English). Best-basis identity varies (en x2, zh
x2, fa x1, native x1), consistent with section 8: no basis ranking is
claimed. Hindi is the hardest target for every basis; English the easiest.
Native diagonals on shared labels: en 0.197 (L28), es 0.290 (L25), zh
0.280 (L30), ro 0.292 (L31), fa 0.218 (L12), hi 0.380 (L17). Note some
best layers shift late (28-32) on the 11-label score, so the main text
should not claim a mid-layer band for this table; the depth-alignment
argument (layer-l axes on layer-l centroids) still holds.

## 7. Cross-language basis search (basis_search_cross_language.py)

Tier (c), maximally permissive: for source basis B and target A, search ALL
45 pairs x 33 layers of B's basis for the plane that best fits A's anchors
(selection on the target side, mirroring the native searched analysis; the
permutation null repeats the full search). Two results:

**Full-label scope: every basis contains a circumplex plane for every
target.** All 36 cells are significant (the weakest is ES->HI at p=0.0118).
With three languages the free search rarely improved on the source's own
plane; with six it sometimes does (e.g. FA basis on EN anchors: 0.200
searched vs 0.295 frozen-plane transfer), though the searched pair is
usually the source's own circumplex pair or a layer-shifted variant of it.

**Shared-11-label 6x6 table** (every cell fits the same 11-point shape;
null medians ~0.45-0.50, so power is limited - cells marked + are not
significant at 0.05):

| target \ basis | EN | RO | ES | ZH | FA | HI |
|---|---:|---:|---:|---:|---:|---:|
| English | 0.156 | 0.250 | 0.197 | 0.171 | **0.147** | 0.225 |
| Romanian | 0.276 | 0.292 | 0.249 | 0.218 | **0.153** | 0.291 |
| Spanish | 0.236 | 0.310 | 0.290 | 0.265 | **0.174** | 0.264 |
| Mandarin | 0.174 | 0.329 | 0.235 | 0.272 | **0.165** | 0.250 |
| Persian | 0.250 | 0.394+ | 0.283 | **0.186** | 0.219 | 0.288 |
| Hindi | 0.244 | 0.413+ | 0.399+ | 0.269 | **0.218** | 0.379+ |

With equalized targets, the Persian basis is the best basis for every target
except Persian itself (where Mandarin wins) - overturning the three-language
finding that the English basis was best-for-all, and stable under the
full-archive anchor sweep (the pre-sweep table had the same six row
winners). The earlier EN-best result was explained by estimation quality
(EN: 503 centroids, dominant pretraining language), but Persian has only
247 centroids, so that explanation no longer carries. A plausible
alternative: the Persian centroids come exclusively from rigid masked
frames (conjunction patterns and somatic idioms), so its PCA has less
register/topic variance to absorb and concentrates affect in the leading
PCs - testable by rebuilding another language's basis from a single
register. Note the basis-quality ranking is NOT a resource ranking in
either direction: Hindi (smallest, unfiltered) is a mediocre basis, and as
targets the rows follow the expected resource ordering (English easiest,
Hindi hardest, three ns cells). Native bases win NO target, and RO is the
weakest basis for every target. Pending checks before leaning on the
Persian inversion: anchor bootstrap for cell error bars, and a
frame-uniformity / centroid-count control.

**UPDATE (section 8): the Persian-best-basis ranking does NOT survive the
frozen-plane protocol on the same shared labels — treat it as a
search-protocol artifact and do not use this table for basis-quality
claims in the paper.**

## 8. Why do Mandarin and Persian export so well? (diagnostics/)

Diagnostics run after skepticism about the Persian dominance in section 7
(`diagnostics/diagnose_exporter_quality.py`, `..._quality2.py`,
`diagnostics/subsample_basis_control.py`). Verdict: the frozen-transfer
table (section 6) is robust but mostly target-driven; the shared-11-label
free-search ranking (section 7) is fragile and its Persian sweep is an
artifact of searching 45 pairs x 33 layers against 11 anchor points.

**(a) Additive decomposition of the 6x6 frozen-transfer matrix.** Grand
mean D = 0.338; target (column) effects explain 50% of the variance,
source (row) effects 30%, residual 21%. Source effects: zh -0.068 (best
exporter), fa -0.049, en +0.013, hi +0.012, ro +0.024, es +0.069. Target
effects: es -0.073 (easiest), hi +0.119 (hardest). Largest residuals:
fa<->hi transfer unusually well in both directions (-0.06 each).

**(b) fa is NOT the consensus plane.** Mean squared-cosine affinity of each
language's searched plane to the other five (layers 8-28, random baseline
0.0008): en 0.099 > es 0.093 > ro 0.087 > zh 0.085 > fa 0.074 > hi 0.034.
en-es is the most mutually aligned pair (0.215). fa's own per-layer pair
choice is also the least stable (PC3+PC9 / PC4+PC10 / PC2+PC7...), and its
anchor within-label coherence gap is the lowest of the six (0.40 vs zh
0.57). So fa's exports are not explained by an unusually clean or central
native circumplex.

**(c) Frozen transfer on the shared 11 labels does NOT reproduce the
Persian sweep.** Freezing each source's own circumplex pair and scoring the
shared labels, column winners are: en target -> en 0.197 (zh 0.198), es ->
en 0.261, zh -> en 0.174 (!), ro -> zh 0.242, fa -> zh 0.186, hi -> fa
0.265. The section-7 fa-wins-everything pattern only appears under the
free search; with only 11 points the search inflation dwarfs the between-
basis differences. fa's spread-out, layer-unstable affect structure gives
the free search more distinct candidate planes to exploit, which is a
plausible mechanism for it winning specifically under that protocol.

**(d) Vocabulary size does not explain basis quality — corpus purity
does.** Subsampling en's broader vocabulary from 463 to fa's 204 (anchors
kept, 3 seeds) IMPROVES its transfers (en->fa 0.392 -> 0.295-0.320; en->zh
0.285 -> 0.255-0.308); zh subsampled from 694 to 204 stays at full
strength. en's top-10 PCs capture the smallest fraction of anchor-label
variance of all six languages (0.28 at L16 vs zh 0.41, fa 0.38, ro 0.48):
its large unfiltered vocabulary (including non-affect extraction artifacts:
got, made, been, put, done, taken, given, kept, used, seen, heard, found,
sure, right, most, less) feeds the PCA register/topic variance that dilutes
the leading components. zh and fa are the two LLM-filtered (score >= 3)
pattern-extracted corpora. The fa somatic-idiom templates are clean: the
wildcard absorbs intensifiers/clitics (delam [barat] tang shode), each
template is one coherent idiom, and only 2 of 43 fa anchors are templates.

**(e) No scaling artifact.** Transfer is insensitive to the standardization
ingredient: source scales vs target scales vs no scaling changes best-layer
D by <= 0.02 on every fa/zh/en source pair; per-dimension log-scale vectors
are 0.96-0.98 correlated across all language pairs. The PCA axes carry the
entire effect, which also strengthens the shared-plane reading.

Paper guidance: keep the frozen-transfer table and its zh/fa-best-per-
column statement (factually robust); attribute exporter quality to corpus
purity, not language identity or size; do NOT claim Persian is the best
basis (section 7 shared table stays out of the paper).

## Files

- Scripts: `prepare_russell.py`, `prepare_new_languages.py` (zh/fa/hi
  manifests), `analyze_russell.py` (+ `plot_russell_geometry.py`),
  `analyze_convexity_russell.py` (+ `plot_convexity_russell.py`),
  `analyze_plane_share_russell.py` (+ `plot_plane_share_table.py`),
  `analyze_all_states.py` (+ `plot_all_states.py`),
  `transfer_cross_language.py`, `basis_search_cross_language.py`,
  `diagnostics/` (exporter-quality and subsample controls, section 8)
- Anchors: `anchors_russell.json`, `anchors_russell_ro_adj_only.json`
- Results: `results/metrics_russell_*.json`, `results/projections_russell_*.json`,
  `results/selection_ro_russell.json`, `results/russell_mapping_report.tsv`,
  `results/convexity_russell.json`, `results/plane_share_russell_*.json`,
  `results/all_states_*.json`, `results/cross_language_transfer.json`,
  `results/cross_language_basis_search.json`
- Figures: `figures/russell_*.png`
