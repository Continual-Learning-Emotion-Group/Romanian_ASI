# Embedding Similarity Strategy — Results Analysis

**Date:** 2026-02-17
**Branch:** `strategy/embedding-similarity`
**Model:** `intfloat/multilingual-e5-base` (278M params, 768-dim)
**GPU:** Modal T4
**Threshold:** 0.75 (all candidates pass; effective min similarity = 0.81)

## Summary

| Metric | Count |
|--------|-------|
| Input posts (RedditRoAP + PoPreRo) | 54,623 |
| Posts with "I feel" trigger sentences | 1,664 |
| Regex-confirmed (also found by regex) | 407 |
| **Novel (NOT found by regex)** | **1,257** |

**Net new ASI candidates: 1,257** — a 63% expansion over the 2,003 regex-only finds.

## Pipeline Configuration

**Trigger words (focused "I feel" patterns only):**
- `mă simt / te simți / se simte / ne simțim / vă simțiți / se simt` (present reflexive)
- `mă simțeam / se simțea / se simțeau` (imperfect reflexive)
- `m-am simțit / s-a simțit / ne-am simțit / s-au simțit` (perfect reflexive)
- `simt / simți / simte / simțim / simțiți / simțeam / simțea` (bare verb)
- `am simțit / ai simțit / a simțit / au simțit` (bare perfect)
- `mi-e / mi-i / ți-e / ți-i / i-e / i-i` (dative contractions)
- `îmi e / îți e / îi e / îmi este / îți este / îi este` (dative full)
- `îmi era / îți era / îi era / ne era` (dative imperfect)
- `îmi vine / îți vine / îi vine / îmi venea / îți venea` (dative "vine")
- `ne-e / ne este / ne era` (1st person plural dative)
- `simt că / simt ca / mă simt ca și cum / mă simt de parcă` (feel that/like)

**Excluded (too noisy):** `sunt, este, era, a fost, am fost, eram, ești, suntem, sunteți`

**Anchors:** 2,203 unique sentences from regex PatternMatcher on all 54,623 posts.

## Results by Source

| Source | Total | Novel | Regex-known |
|--------|-------|-------|-------------|
| RedditRoAP | 1,251 | 946 | 305 |
| PoPreRo | 413 | 311 | 102 |

## Novel Candidates by Trigger Word (top 15)

| Trigger | Count | % of novel |
|---------|-------|------------|
| simt | 358 | 28.5% |
| îmi e | 123 | 9.8% |
| mă simt | 104 | 8.3% |
| îmi este | 94 | 7.5% |
| îmi vine | 64 | 5.1% |
| se simte | 63 | 5.0% |
| simți | 57 | 4.5% |
| am simțit | 46 | 3.7% |
| vă simțiți | 28 | 2.2% |
| îi este | 28 | 2.2% |
| mi-e | 28 | 2.2% |
| te simți | 26 | 2.1% |
| simțiți | 24 | 1.9% |
| simțeam | 22 | 1.7% |
| simte | 21 | 1.7% |

## Inherited Emotion Distribution (from nearest anchor)

| Emotion | Count | % |
|---------|-------|---|
| anticipation | 558 | 44.4% |
| trust | 242 | 19.3% |
| fear | 206 | 16.4% |
| sadness | 194 | 15.4% |
| joy | 174 | 13.8% |
| anger | 74 | 5.9% |
| surprise | 55 | 4.4% |
| disgust | 12 | 1.0% |

Note: High "anticipation" count is because many anchors are Reddit curiosity posts ("sunt curios...") which get matched broadly.

## Similarity Distribution (novel candidates only)

| Percentile | Score |
|-----------|-------|
| Max | 0.9136 |
| P95 | 0.8913 |
| P90 | 0.8870 |
| Median | 0.8723 |
| P10 | 0.8565 |
| Min | 0.8089 |

| Band | Count |
|------|-------|
| [0.95, 1.00) | 0 |
| [0.90, 0.95) | 10 |
| [0.85, 0.90) | 1,191 |
| [0.80, 0.85) | 56 |

## Quality Assessment

**Estimated quality (heuristic — presence of known emotion words in matched sentence):**
- With emotion indicator word: 330 (26.3%)
- Without: 927 (73.7%)

This underestimates quality because many valid ASI expressions use colloquial/metaphorical language not in the indicator list.

### Good novel finds (examples):
- "Mă simt un pic copleșită și nu știu de unde să mă apuc să caut" (sim=0.91) — overwhelmed
- "nu mă mai simt capabilă să identific semnele unei intenții reale" (sim=0.88) — inadequacy
- "Cumva mereu am avut frica de a spune unei fete ce simt" (sim=0.88) — fear of vulnerability
- "mi-e teamă să am parte din nou de experiențe negative" (sim=0.88) — fear/anxiety
- "îmi e rușine că-s roman" (sim=0.88) — shame
- "Nu îmi vine să cred cât de bine arată orașul" (sim=0.88) — surprise

### Noisy cases (false positives):
- "îmi vine din o altă sursă de venit" — `îmi vine` = financial "comes to me", not a feeling
- "se simte cam învechit" — object described as "feeling old", not a person's emotion
- "îmi este aproape imposibil să găsesc ceapă" — `îmi este` = impersonal "it's impossible for me"
- "se simțea aroma" — literal sensory (smell), not emotional

### Noise sources:
1. **`simt`/`se simte` in sensory contexts** — physical sensation, not affective state
2. **`îmi vine`/`îmi este` in idiomatic use** — "it comes to me", "it is [impossible] for me"
3. **`anticipation` emotion over-representation** — many anchors are curiosity-type posts, inflating this category

## Comparison with Regex Baseline

| Method | Candidates |
|--------|-----------|
| Regex PatternMatcher | 2,003 |
| Embedding novel | 1,257 |
| **Combined (union)** | **~3,260** |

The embedding strategy finds **1,257 posts with "I feel" expressions that the regex pipeline misses** because:
1. The emotion word after the pattern isn't in the 511-word curated seed list
2. Non-standard spelling/phrasing that regex doesn't cover
3. Dative constructions (`mi-e`, `îmi e`) with words not in seed

## Output

- File: `data/embedding_asi_candidates.jsonl`
- Schema: see `CLAUDE.md` for field descriptions
- Each record includes `metadata.is_novel` boolean and `metadata.cosine_similarity` score
