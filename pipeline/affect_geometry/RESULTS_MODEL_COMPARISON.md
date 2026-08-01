# Model comparison: Qwen3.5-4B vs Qwen3-8B

All geometry experiments reported in the paper draft (Tables 7–10, Figures 3–4)
were rerun end-to-end with **Qwen3-8B** (`Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`, bf16), the same model used by the
paper's finetuning/benchmarking experiments. Everything upstream of the model
is byte-identical to the 4B run: same six manifests (sha256-verified against
`results/manifest_checksums.json`), same 30-occurrence threshold, ≤80 contexts
per lemma, 350-char windows, same anchor mapping (`anchors_russell.json`),
same analysis code, same seed and 5,000-permutation tests.

Extraction: piranha (3× A100-40GB), 2026-08-01, batch 32,
`logits_to_keep=1` added to `extract.py` (skips the unused vocab projection;
hidden states unaffected — needed to fit the 8B on 40 GB cards).
Qwen3-8B gives **37 hidden states (embedding + 36 layers) × 4096 dims** vs
33 × 2560 for the 4B; all analyses read shapes from the archives and adapt.
Per-lemma context counts are identical between the two extractions for every
language (same single invalid English row).

Layout: per-model artifacts live in `artifacts/hidden/<model>/`,
`results/<model>/`, `figures/<model>/` with `<model>` ∈
{`qwen3.5-4b`, `qwen3-8b`}; scripts select via `AFFECT_GEOMETRY_MODEL`
(default `qwen3.5-4b` until the paper flips). The same layout exists on
piranha under `~/affect_geometry_russell/`.

## Table 7 — anchor-only circumplex fits (F = A)

| Lang | 4B conf D@L (p) | 8B conf D@L (p) | 4B searched | 8B searched |
|---|---|---|---|---|
| English | 0.299@16 (0.0002) | 0.328@21 (0.0002) | PC1+PC2 0.299@16 (0.0002) | PC1+PC2 0.328@21 (0.0002) |
| Spanish | 0.303@17 (0.0002) | 0.333@21 (0.0002) | PC1+PC2 0.303@17 (0.0002) | PC1+PC3 0.287@22 (0.0002) |
| Mandarin | 0.310@25 (0.0002) | 0.315@19 (0.0002) | PC1+PC2 0.310@25 (0.0002) | PC2+PC3 0.265@17 (0.0002) |
| Romanian | 0.600@15 (0.0008) | 0.571@20 (0.0006) | PC3+PC7 0.428@18 (0.0002) | PC3+PC5 0.425@24 (0.0004) |
| Persian | 0.592@12 (0.0008) | 0.699@20 (0.0342) | PC2+PC4 0.376@14 (0.0002) | PC2+PC10 0.447@23 (0.0004) |
| Hindi | 0.748@11 (0.0874) | 0.771@22 (0.1466) | PC3+PC6 0.432@15 (0.0006) | PC1+PC3 0.461@18 (0.0022) |

**Reproduces.** The two-profile split is intact: English/Spanish/Mandarin fit
the confirmatory PC1+PC2 plane at the permutation floor (D = 0.32–0.33);
Romanian/Persian/Hindi need searched planes (D = 0.43–0.46, all significant;
Hindi confirmatory still n.s.). One nuance: on 8B the searched plane for
Spanish (PC1+PC3) and Mandarin (PC2+PC3) slightly beats PC1+PC2, so the crisp
"the searched plane IS the confirmatory plane" sentence now holds only for
English — but the confirmatory fits remain at floor significance for all
three, so the "leading-plane" story survives. Best layers shift from ~16/33
to ~21/37 (same relative depth). Persian's confirmatory p weakens to 0.034.

## Table 8 — all-states PCA (F = S, width 10)

| Lang | 4B pair D@L (p) | 8B pair D@L (p) | 4B share A/B | 8B share A/B |
|---|---|---|---|---|
| English | PC2+PC8 0.284@15 (0.0002) | PC6+PC9 0.279@26 (0.0002) | 7.7/8.2% | 3.4/4.3% |
| Spanish | PC1+PC2 0.335@26 (0.0002) | PC2+PC7 0.254@17 (0.0002) | 11.8/8.9% | 9.9/7.8% |
| Mandarin | PC1+PC4 0.278@24 (0.0002) | PC2+PC4 0.257@17 (0.0002) | 14.3/6.9% | 14.3/7.0% |
| Romanian | PC4+PC10 0.320@25 (0.0002) | PC3+PC5 0.343@22 (0.0002) | 5.7/6.4% | 9.9/9.3% |
| Persian | PC3+PC9 0.262@12 (0.0002) | PC4+PC7 0.430@16 (0.0004) | 8.6/6.1% | 8.5/5.0% |
| Hindi | PC5+PC9 0.433@17 (0.0008) | PC1+PC8 0.554@19 (0.0466) | 6.2/6.2% | 7.8/9.9% |

**Mostly reproduces, one claim weakens.** The circumplex still emerges from
the all-states PCA in all six languages (all p < 0.05), and anchor vs broader
plane shares stay comparable. But the strong 4B claim "F = S matches or beats
the anchor-only fit in every language (0.26 vs 0.38 in Persian)" does not
hold on 8B: Persian's all-states fit (0.430) only matches its anchor-searched
fit (0.447), and Hindi's degrades (0.554, p = 0.047, marginal). For
en/es/zh/ro the claim still holds (es notably improves to 0.254).

## Table 9 — broader-only fits (F = S \ A)

| Lang | 4B pair D@L (p) | 8B pair D@L (p) |
|---|---|---|
| English | PC3+PC7 0.294@32 (0.0002) | PC3+PC8 0.335@21 (0.0002) |
| Spanish | PC2+PC3 0.263@11 (0.0002) | PC2+PC7 0.358@18 (0.0002) |
| Mandarin | PC1+PC2 0.306@12 (0.0002) | PC2+PC5 0.275@8 (0.0002) |
| Romanian | PC1+PC5 0.455@8 (0.0008) | PC4+PC8 0.365@27 (0.0002) |
| Persian | PC3+PC9 0.374@13 (0.0002) | PC5+PC10 0.369@20 (0.0002) |
| Hindi | PC9+PC10 0.555@14 (0.0468) | PC3+PC5 0.573@20 (0.0698) |

**Reproduces.** With anchors fully held out the circumplex is recovered in
five of six languages at p ≤ 0.0002 (D = 0.27–0.37); Hindi is marginal in
both runs, and on 8B slips just above 0.05 (0.0698) — the paper's "Hindi,
with only 79 broader states, is marginal" phrasing still fits, but the †
footnote value changes.

## Table 10 — cross-lingual transfer of the circumplex plane (shared 11 labels)

Qwen3-8B (row = language the plane is fit/selected on, col = target):

| from\to | en | es | zh | ro | fa | hi |
|---|---|---|---|---|---|---|
| en | **0.146** | **0.203** | **0.216** | **0.195** | 0.282 | 0.292 |
| es | 0.201 | 0.260 | 0.260 | 0.230 | 0.284 | 0.372 |
| zh | 0.181 | 0.206 | 0.257 | 0.197 | **0.177** | **0.253** |
| ro | 0.303 | 0.339 | 0.279 | 0.304 | 0.393 | 0.344 |
| fa | 0.273 | 0.343 | 0.309 | 0.297 | 0.360 | 0.508† |
| hi | 0.273 | 0.322 | 0.343 | 0.268 | 0.485† | 0.470 |

† not significant (fa→hi p = 0.060, hi→fa p = 0.062); all other 28
off-diagonal cells p < 0.05. (4B: 29/30, the failure being ro→hi.)

**Reproduces, and the headline interpretation gets stronger.** The affect
plane transfers across languages, and every row winner comes from the two
largest vocabularies: English's plane is best for en/es/zh/ro, Mandarin's for
fa/hi. On 4B, Hindi's best foreign plane was Persian's — that exception
disappears on 8B. Differences worth noting: (a) English's native diagonal is
now clearly the best cell of its column (0.146 vs 0.181), no longer an
"almost tie"; (b) the failing pair moves from Romanian→Hindi to
Persian↔Hindi.

## Not in the paper: full-label basis search (results/qwen3-8b/cross_language_basis_search.json)

On 4B, Persian's basis won five of six target rows (full-label scope) — a
result we had flagged as statistically dubious. On 8B it does **not**
replicate: **Mandarin's basis wins all six rows** (D = 0.17–0.33, all 36
cells significant), including beating every native basis. Mandarin has the
largest vocabulary (761 lemmas), so the 8B basis-search result now agrees
with Table 10's resourcing story (biggest vocabulary → best estimate of the
shared plane) instead of contradicting it. This supports treating the 4B
"Persian wins" pattern as model-specific noise rather than a finding.

## Verdict

Reproduction ran end-to-end without issues; every table and figure has an 8B
counterpart. Qualitatively: the core findings (circumplex present in every
language, two-profile confirmatory/searched split, broader states sharing the
plane, anchor-free recovery, significant cross-lingual transfer dominated by
high-resource planes) all reproduce. Claims needing softening in the rewrite:
the Spanish/Mandarin searched-equals-confirmatory detail, the "all-states
matches or beats anchor-only everywhere" claim (Persian/Hindi), Hindi's
broader-only marginality (now just above 0.05), and the identity of the
non-transferring pair (Persian↔Hindi, not Romanian→Hindi). All exact numbers,
layers, and PC pairs change and every figure/table must be regenerated from
`results/qwen3-8b/` + `figures/qwen3-8b/`.
