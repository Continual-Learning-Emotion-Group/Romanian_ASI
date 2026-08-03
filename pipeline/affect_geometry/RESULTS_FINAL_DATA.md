# Final Data refresh: 8-language affect geometry (base vs. multilingual-SFT Qwen3-8B)

Date: 2026-08-01/02. Extends RESULTS_RUSSELL.md (six languages, July data) with
(1) a refreshed, uniformly LLM-judged dataset, (2) two new languages (French,
Indonesian), and (3) a comparison between the base model and a LoRA
fine-tuned on the 8-language ASI task (`joint_all`).

## 1. Data refresh

Source: "Final Data" dump (2026-08-01) — per-language `{lang}_judged.csv.gz`
with a unified judged schema, plus an Indonesian 10k judged sample
(`id_judge_10k_v2.csv`). Filter everywhere: **llm_score == 3 AND match_ok**
(the threshold validated at 91.3% precision against human annotators on the
Romanian benchmark). This fixes the two known asymmetries of the July set:
en/es previously came from the original MASIVE corpus, and Hindi was entirely
unfiltered.

Judge models: Qwen3.5-9B for en/es/ro/zh/fr, Qwen3.5-27B for fa/hi/id
(same rubric; noted as a footnote-level caveat).

Builder: `prepare_final_data.py` → `artifacts/manifests_final/{lang}.jsonl`
(checksums in `checksums.sha256`). Same frozen sampling as all prior runs
(config.json: ≥30 occurrences/lemma, ≤80 contexts, 350-char windows, seed
20260720). Span policy: the judged term is located in its context with
word-boundary checks; rows where the term is absent or occurs more than once
are dropped (fa spans are exact via [MASK] reconstruction; id terms located
in `selftext`).

| lang | manifest occurrences | lemmas | note |
|---|---:|---:|---|
| en | 35,557 | 507 | fresh reddit/OSCAR corpora (no longer MASIVE) |
| ro | 8,202 | 122 | same sources as before (fulg/filmot), repackaged |
| es | 16,843 | 251 | fresh corpora |
| zh | 58,030 | 761 | |
| fa | 23,024 | 356 | template idioms preserved (e.g. دلم ... گرفته) |
| hi | 6,638 | 92 | now LLM-filtered (was unfiltered) |
| fr | 8,659 | 117 | **new language** |
| id | 1,694 | 29 | **new language**; only a 10k judged sample exists |

Anchor changes (all under the frozen strict-synonymy rule; audit rows
appended to `results/russell_mapping_report.tsv`):
- **French added: 14 labels / 18 lemmas** (happy: heureux, joyeux; delighted:
  ravi; astonished: surpris; angry: en colère, furieux, enragé; afraid:
  effrayé; distressed: angoissé; frustrated: frustré; sad: triste; depressed:
  déprimé; calm: apaisé; serene: paisible; relaxed: détendu; satisfied:
  satisfait, épanoui; pleased: contente). No anchors possible for excited /
  tense / bored / tired (below floor or absent at score 3).
- **Indonesian added: 9 labels / 12 lemmas** (happy: senang, bahagia; afraid:
  takut, ketakutan; sad: sedih; calm: tenang; bored: bosan; annoyed: kesal;
  satisfied: puas, kepuasan; at ease: nyaman; frustrated: frustasi).
  gembira (happy) fell below floor after ambiguity drops and was removed.
  No angry anchor (marah = 27 occurrences < 30 floor).
- New-data floor losses in old languages: ro annoyed (iritat), fa pleased
  (خشنود), hi at ease (सहज). The 11-label six-language shared set is intact.

## 2. Analysis changes

- All analysis scripts discover archives dynamically (`common.discover_archives`),
  so legacy 6-language dirs and the new 8-language dirs both work.
- `transfer_shared_labels.py` is now **tiered**: `six_lang` (11 shared labels)
  and `seven_lang_fr` (8 shared labels). Indonesian is excluded from tiers —
  a global 8-language intersection would collapse to 4 labels, which a 2D
  Procrustes cannot falsify (24 possible label permutations → p floor ≈ 0.04).
- **New: `transfer_pairwise.py`** — every ordered cell (A→B) is scored on the
  labels shared by just A and B (6–25 across pairs), and standardized against
  its own layer-search-corrected label-permutation null:
  `null_z = (mean(null) − observed) / sd(null)`. null_z is comparable across
  all 64 cells regardless of label count; `n_labels` is reported per cell.
- `analyze_russell.py --shared-languages` pins the native "shared" scope to
  the six-language label set.

Variants: `AFFECT_GEOMETRY_MODEL=qwen3-8b-final` (base
Qwen/Qwen3-8B@b968826) and `qwen3-8b-joint-final` (same base with the
`joint_all` LoRA merged: r=32/α=64 on all proj layers, trained 2 epochs on
the 40k 8-language ASI mask task, final val loss 0.4956). Extraction:
`extract.py` (now with `--adapter`), batch 32, ≤512 tokens, bf16, A100-40GB
(piranha), 0 invalid rows in all 16 runs.

## 3. Base model (qwen3-8b-final)

### 3a. Native circumplex fits (anchor-basis, full label set)

| lang | labels | best searched D | corrected p | PC1+PC2 p |
|---|---:|---:|---:|---:|
| en | 21 | 0.337 | 0.0002 | 0.0002 |
| ro | 21 | 0.442 | 0.0004 | 0.0028 |
| es | 18 | 0.300 | 0.0002 | 0.0002 |
| zh | 25 | 0.283 | 0.0002 | 0.0002 |
| fa | 19 | 0.478 | 0.0034 | 0.0338 |
| hi | 18 | 0.462 | 0.0054 | 0.1468 |
| **fr** | 14 | 0.312 | **0.0012** | 0.0002 |
| **id** | 9 | 0.197 | **0.0172** | 0.0290 |

All-states PCA protocol (width-10 search): en 0.289 (p=0.0002), ro 0.405
(0.0002), es 0.277 (0.0002), zh 0.266 (0.0002), fa 0.424 (0.0014), hi 0.429
(0.0012), fr 0.219 (0.0002), **id 0.317 (p=0.32, n.s.)** — with only 29
lemmas Indonesian's own all-states plane does not beat its null, even though
its anchor-basis fit does. See 3c: foreign planes DO organize Indonesian
significantly, i.e. cross-lingual transfer recovers structure that the
language's own thin sample cannot reveal.

### 3b. Shared-label tiers

- **six_lang (11 labels): all 36 cells p < 0.05** (worst: fa→fa D=0.451
  p=0.0202). On the July data three cells hovered at p≈0.05–0.06 (fa→hi,
  hi→fa, hi→hi); the refresh (esp. LLM-filtered Hindi) tightened all of them.
- **seven_lang_fr (8 labels): 43/49 cells p < 0.05.** Failures: en→hi
  (0.207), hi→fa (0.120), ro→fa (0.094), fa→hi (0.090) + two more in the
  hi/fa corridor. French transfers well in both directions (best cell:
  hi→fr D=0.144 p=0.0012).

### 3c. Pairwise-intersection standardized 8×8

61/64 cells significant (p < 0.05). Mean null_z as source (excl. native):
**zh 7.6 > en 6.0 > es 5.9 > ro 5.8 > fa 5.5 > hi 4.8 > fr 4.4 > id 2.3**.
Mandarin remains the best universal explainer (largest corpus: 761 lemmas).
Indonesian is the weakest but still transfers: en→id z=3.27 (p=0.0028),
zh→id z=4.23 (p=0.0002) on all 9 of id's labels. The only non-significant
cells are the n=6 pairings (hi→id p=0.60, fr→id p=0.051, id→fr p=0.10) plus
none others — exactly the thin-intersection cells the design flags via
n_labels.

Full tables: `results/qwen3-8b-final/transfer_pairwise.json` (`matrix` key),
`transfer_shared_labels.json`, `cross_language_transfer.json`.

## 4. Fine-tuned model (qwen3-8b-joint-final)

Same suite, hidden states from the base model with the `joint_all` LoRA
merged (0 invalid rows, identical manifests).

### 4a. Native circumplex fits (anchor-basis, full label set)

| lang | base D / p | joint D / p |
|---|---:|---:|
| en | 0.337 / 0.0002 | 0.358 / 0.0002 |
| ro | 0.442 / 0.0004 | **0.352 / 0.0002** |
| es | 0.300 / 0.0002 | 0.327 / 0.0002 |
| zh | 0.283 / 0.0002 | 0.315 / 0.0002 |
| fa | 0.478 / 0.0034 | 0.497 / 0.0074 |
| hi | 0.462 / 0.0054 | 0.488 / 0.0118 |
| fr | 0.312 / 0.0012 | 0.328 / 0.0022 |
| id | 0.197 / 0.0172 | 0.205 / 0.0212 |

All 8 languages remain significant. All-states PCA (w10): 7/8 significant in
both variants; Indonesian stays n.s. but improves (D 0.317→0.269,
p 0.32→0.14); ro (0.405→0.339) and hi (0.429→0.370, p→0.0002) improve,
en degrades (0.289→0.365, still p=0.0002).

### 4b. Shared-label tiers

- six_lang (11 labels): **36/36 significant** (worst ro→hi p=0.0126 —
  tighter than base's worst, fa→fa p=0.0202).
- seven_lang_fr (8 labels): **43/49 significant**, same count as base;
  the failures stay concentrated in the hi/fa corridor (worst ro→fa
  p=0.177). Mean disparity 0.225 (base 0.231).

### 4c. Pairwise-intersection standardized 8×8

**62/64 cells significant** (base: 61/64). The one flip is
**hi→id: z −0.29 (p=0.60) → z 2.22 (p=0.0112)** — the base model's only
outright transfer failure is repaired by fine-tuning. The remaining
non-significant cells in both variants are fr↔id (n=6 labels, the thinnest
intersection). Grand mean off-diagonal null_z: 5.29 (base) → 5.25 (joint).

## 5. Base vs. fine-tuned comparison

Headline: **8-language ASI fine-tuning leaves the affect circumplex intact
— no collapse, no reorganization — and slightly redistributes transfer
strength toward the low-resource corner of the matrix.**

- **Geometry is preserved.** Every significance verdict that held for the
  base model holds after SFT (native fits 8/8, six_lang tier 36/36,
  seven_lang_fr 43/49, pairwise 61→62/64). Task training on masked
  affective-state prediction did not overwrite the circumplex structure.
- **Low-resource corner improves.** Persian as a source gains the most
  (mean off-diag z 5.46→6.05, +0.60); Indonesian as a target gains +0.34
  mean z, including the hi→id repair and fa→id strengthening
  (z 2.46→3.11). Indonesian's own all-states fit halves its p (0.32→0.14)
  without reaching significance — the 29-lemma sample stays the binding
  constraint.
- **High-resource sources soften slightly.** en (−0.42), ro (−0.38),
  es (−0.25) mean source-z; native anchor disparities tick up for
  en/es/zh/fa/hi. Consistent with SFT slightly homogenizing
  representations across languages rather than sharpening each language's
  own plane. Romanian native fit is the exception, improving markedly
  (D 0.442→0.352).
- **Anchor plane-share advantage attenuates in some languages.** The LOO
  anchors-vs-broader Mann–Whitney stays strong for en/zh/fr/fa but drops
  to n.s. for es (p=0.59) and borderline for ro (p=0.046) — after SFT,
  broader states load onto the circumplex plane about as strongly as
  anchors in those languages, again pointing at homogenization rather
  than degradation (transfer from ro/es planes is unchanged).

Net: fine-tuning is safe for the geometry claims, and the modest movement
that does occur is in the direction predicted by cross-lingual alignment
(low-resource languages benefit, high-resource planes converge).

## 6. Loose anchor pass + extended Indonesian data (2026-08-02)

Two user-requested modifications; both variants fully re-run. The
strict-anchor results above are preserved under
`results|figures/{variant}_strict_backup_20260802/`.

### 6a. Extended Indonesian data

Four unjudged corpora (`threads/x/yt/kaggle_reddit.csv`, ~64.5k rows, same
MASIVE-style schema, no llm_score) added via a **calibration filter**: the
judged 10k sample yields a per-term score-3 precision; only terms with
rate ≥ 0.7 over ≥ 5 judged occurrences (134 terms) are mined from the
unjudged text. Negated/challenge rows skipped, rows overlapping the judged
sample deduped, same boundary/single-occurrence locate rules
(`prepare_final_data.py --id-extra-csvs`). Result: **id 1,694 → 4,240
occurrences, 29 → 68 lemmas**; `marah` (angry) and `gembira` (happy) rise
above the 30-floor.

### 6b. Loose anchor pass

All embedded lemmas in all 8 languages swept against the 28 Russell labels
(91 additions, audit rows `ACCEPT_LOOSE_PASS` in
`results/russell_mapping_report.tsv`; strict file backed up as
`anchors_russell.json.bak_strict_20260802`). Rules: each lemma to its
single best label; strong/near-strong synonyms added freely; loose matches
only to labels lacking a strong anchor; blends and distinct concepts
(guilty, proud, lonely, disappointed, worried…) stay excluded.
Label counts: en 21→27, ro 22→23, es 18→24, zh 25→26, fa 20→23, hi 19→20,
fr 14→17, **id 9→16**. Notable: en gains exact-match anchors that were
absent from the old MASIVE-era pool (satisfied, content, serene, pleased,
delighted); es gains encantado→delighted and alarmado→alarmed; fa gains a
tense label; id gains angry/astonished/tense/alarmed/serene.

### 6c. Results (both variants)

- **Native fits: 8/8 significant in both variants** (worst: base fa
  p=0.0060; joint fa p=0.0024). Disparities stable or better for en/ro/es;
  slightly higher for fa/hi/zh (looser synonyms add angle noise) with
  p-values unharmed. id: p 0.0172 → 0.0002 (16 labels).
- **All-states PCA: 8/8 significant in both variants.** The former sole
  negative result — Indonesian, p=0.32 on 29 lemmas — flips to
  **D 0.242, p=0.0002 on 68 lemmas** (joint: 0.269, p=0.0002),
  confirming it was a data-thinness artifact.
- **Pairwise 8×8: 64/64 cells significant in BOTH variants** (strict:
  base 61, joint 62). Thinnest intersection now 12 labels (was 6).
  Mean off-diag source-z (base): zh 9.9 > en 8.7 > es 8.1 > fa 7.9 >
  hi 7.2 > ro 7.1 > id 6.4 ≈ fr 6.4. Joint softens most sources slightly
  (grand mean 7.72 → 7.53), same pattern as the strict comparison.
- **Tiers: six_lang shared set 11 → 15 labels, 36/36 both variants;
  seven_lang_fr 8 → 12 labels, 49/49 both variants** (base strict was
  43/49 — the hi/fa corridor failures all resolve).
- Caveat: strict-vs-loose z values are not directly comparable (larger
  label sets → more permutation power); the correct reading is that the
  circumplex geometry is robust to a broader, noisier anchor vocabulary,
  and the added breadth converts to real statistical power.
- **NEW: global-intersection table** (`transfer_global_intersection.py`):
  the loose pass raises the all-8-language label intersection from 4
  (unfalsifiable) to **10** (afraid, angry, astonished, at ease, calm,
  depressed, distressed, happy, sad, satisfied), making a fixed-label 8×8
  meaningful for the first time — every cell uses the identical set, so
  raw disparities are directly comparable. Base model: **64/64 significant**
  (z 2.5–5.2). With power equalized, Indonesian's row is fully competitive
  (z 4.0–4.9, id→fr D=0.12) — its weakness in the per-pair table was
  label-count power, not geometry — and Spanish emerges as the sharpest
  source (es→ro z=5.2, D=0.13). Weakest cells involve Hindi/Persian as
  targets (en→hi and hi→hi z=2.5), echoing the tier results.

## 7. Fixed-layer, fixed-label protocol + "tired" floor exception (2026-08-02)

Simplified headline protocol replacing the layer-searched tables for the
main 8×8 (searched tables remain as "best achievable transfer" appendix
material). Two changes:

**7a. "tired" admitted via a documented floor exception.** The 10-label
global intersection had a 107° empty arc in the deactivation quadrant
(no bored/tired/sleepy), which distorted fixed-set rankings. The gap was
blocked only by fr and id: fr *fatigué* (25 clean manifest candidates) and
id *lelah* (24) both just missed the frozen 30-occurrence floor.
`ANCHOR_FLOOR_EXCEPTIONS` in `prepare_final_data.py` admits exactly these
two lemmas at ≥24 judged-clean occurrences (id *lelah* has judged score-3
rate 0.17, so the unjudged id corpora were NOT mined for it — its 24 rows
are all individually judged). Manifests rebuilt as strict supersets
(fr 8,659→8,684; id 4,240→4,264; old occurrence_ids unchanged), fr+id
re-embedded on both variants. Global intersection is now **11 labels**
spanning all four quadrants, max angular gap 68° (was 107°):
happy 8° → astonished 70° → angry 99° → afraid 119° → distressed 140° →
sad 208°/depressed 210° → **tired 268°** → calm 316°/at ease/satisfied 320°.

**7b. Protocol** (`transfer_fixed_layer.py`): each source's layer is frozen
ONCE at its native-best (its own full anchor set, no target seen); every
cell is a single Procrustes disparity on the same 11 labels, so raw D is
directly comparable across all 64 cells; significance is plain PROTEST
(Jackson 1995) label permutation at that single layer, 5000 draws, no
layer-search correction needed because nothing is searched.

**7c. Results** (`transfer_fixed_layer.json`; heatmaps on Desktop):
- Base: **64/64 significant** (worst ro→fa p=0.032). Joint: **64/64**
  (worst p=0.013). Without-tired robustness
  (`transfer_fixed_layer_no_tired.json`): 64/64 — significance does not
  hinge on the thin tired centroids.
- Source ranking (mean off-diagonal D, base): zh 0.241 < id 0.265 <
  en 0.297 < fa 0.341 < es 0.348 < fr 0.400 < ro 0.436 < hi 0.453.
  Mandarin best universal source; the ranking is stable with/without
  tired (zh 0.210 < id 0.245 < es/en ~0.27-0.30 on the 10-label set).
- Indonesian as second-best source persists across label sets under this
  protocol (caveat: its plane is built from the smallest corpus, 4,264
  occurrences; report with that caveat).
- Spanish's earlier "best source" showing on the arousal-poor 10-label
  searched table does not survive the tired label: es adds ~0.06-0.09 D
  when arousal-axis content matters (es mean 0.273→0.348 adding tired).
- Freezing layers is the honest price: ro and fr degrade sharply as
  sources (ro 0.436, fr 0.400) — their circumplex geometry is not
  concentrated at one transferable layer; in searched tables they
  compensated with per-target layer choice.
- Native cells are reference-only (frozen layer chosen on same data);
  natives are rarely column-best (e.g. best for fa is id 0.247, for hi is
  fa 0.320).

**7d. Fixed layer + per-pair labels** (`transfer_fixed_layer_pairwise.py`,
`transfer_fixed_layer_pairwise.json`) — the headline-protocol combination:
source plane frozen at its native-best layer (target never seen), cells
scored on each pair's own label intersection (13-27 labels), plain PROTEST
z at the frozen layer. Base: 64/64 significant (all p<=0.003). Source
ranking by mean off-diag z: zh 12.6 > en 11.4 > fa 10.0 > es 9.9 > ro 8.6 >
id 8.1 > fr 7.6 > hi 6.7 — zh column-best for 6/8 targets (zh->en z=17.8),
en best for hi. Joint: 64/64; same shape, en softens (11.4->9.7), hi source
strengthens (6.7->8.7), zh unchanged (12.3). Unlike the searched per-pair
table, needs no layer-search correction; unlike the global fixed-set table,
keeps full per-pair circle coverage.

Display metric: **skill = 1 − D / null_mean** ∈ [0,1] (fraction of
chance-level disparity eliminated) rather than null_z — z correlates with
label count mechanically (corr(n, z)=+0.83, driven by null_sd shrinking
with n, corr=−0.97), while skill cuts that to +0.37 (residual plausibly
genuine: raw D also falls with n). Base source ranking by mean skill:
zh 0.71 > en/id 0.64 > fa 0.62 > es 0.57 > fr 0.55 > ro 0.54 > hi 0.45.
Joint: zh 0.69 > id 0.63 > hi 0.58 > fa/es ~0.58 > en 0.54 > fr 0.48 >
ro 0.47 — SFT lifts Hindi as a source (+0.13, e.g. hi→es 0.46→0.66) and
drops English (−0.10, e.g. en→fa 0.60→0.33): resource homogenization.
Figure: fixed_layer_pairwise_skill_heatmap_base_vs_joint.png (Desktop).

**7e. Column-fair variant — target's full labels**
(`transfer_fixed_layer_target_labels.py`): cell (A→B) scores B's ENTIRE
label set on A's frozen plane, so every cell in a column answers the
identical exam and within-column source comparisons are exactly fair
(rows/cross-column are not). Base: 64/64 significant (worst ro→hi
p=0.008). zh wins 6/8 columns (skill 0.80 on en and es exams), fa takes
hi, fr its own. Mean within-column rank: zh 1.6 > en 3.0 = id 3.0 >
fa 4.4 ≈ es 4.5 > ro 6.0 > fr 6.4 > hi 7.1. Joint: zh 7/8 columns
(rank 1.1); en softens to 4.5, hi rises off the floor. Note en vs id:
this fair view ties them, while the matched en∩id∩target head-to-head
gives en a small edge (0.652 vs 0.627) — margins are within noise; the
robust claims are zh #1 and ro/fr/hi at the tail, with the en/id/es/fa
middle order unstable across fair views. Figure:
target_labels_transfer_heatmap_base_vs_joint.png (Desktop).

**7f. Searched companion to 7e**
(`transfer_searched_target_labels.py`): same target-full-label exams, but
A's layer swept per target (plane per layer still from A's anchors);
p and skill use the layer-search-corrected null. Base: 64/64 (all
p=0.0002 except ro→hi 0.0006); zh wins 5/8 columns, mean rank zh 1.9 >
en 3.1 > id 3.5. Biggest mover vs the frozen table: hi as source
(rank 7.1→4.4; hi→en 0.65→0.73 at L19 vs its native-best L9) — Hindi has
good transferable geometry, just not at its native-best layer; ro stays
last (7.4). Joint: zh 7/8 columns, mean rank 1.4. Figure:
searched_target_labels_heatmap_base_vs_joint.png (Desktop).

## Files

- Builder: `prepare_final_data.py`; manifests `artifacts/manifests_final/`
- Anchors: `anchors_russell.json` (fr/id merged 2026-08-01;
  backup `anchors_russell.json.bak_pre_fr_id`)
- Driver: `run_final_suite.sh <variant>`
- Results: `results/qwen3-8b-final/`, `results/qwen3-8b-joint-final/`
- Figures: `figures/qwen3-8b-final/`, `figures/qwen3-8b-joint-final/`
