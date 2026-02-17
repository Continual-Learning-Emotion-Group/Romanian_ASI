# LLM Filtering Results

## Summary

LLM-based validation of pattern-matched ASI candidates from RedditRoAP and PoPreRo sources.

| Metric | Value |
|--------|-------|
| Total candidates | 2,255 |
| **Kept (affective)** | **1,687 (74.8%)** |
| Filtered (not affective) | 568 (25.2%) |
| Errors | 0 |

## Configuration

- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Platform:** Modal (A10G GPU, vLLM 0.6.6.post1)
- **Temperature:** 0.0
- **Sources:** reddit_roap, poprero
- **Processing time:** 519s (~8.5 min), 4.3 candidates/sec

## Results by Source

| Source | Total | Kept | Filtered | Keep Rate |
|--------|-------|------|----------|-----------|
| reddit_roap | 1,788 | 1,358 | 430 | 76.0% |
| poprero | 467 | 329 | 138 | 70.4% |

## Results by Pattern

| Pattern | Category | Total | Kept | Filtered | Keep Rate |
|---------|----------|-------|------|----------|-----------|
| sunt_adj_present | secondary | 1,429 | 979 | 450 | 68.5% |
| eram_adj_imperfect | secondary | 205 | 172 | 33 | 83.9% |
| am_noun_present | secondary | 156 | 97 | 59 | 62.2% |
| mie_short | secondary | 127 | 127 | 0 | 100% |
| am_fost_adj_perfect | secondary | 113 | 94 | 19 | 83.2% |
| ma_simt_present | primary | 111 | 110 | 1 | 99.1% |
| imi_este_present | secondary | 49 | 49 | 0 | 100% |
| mam_simtit_perfect | primary | 11 | 11 | 0 | 100% |
| ma_simteam_imperfect | primary | 11 | 10 | 1 | 90.9% |
| imi_era_imperfect | secondary | 10 | 10 | 0 | 100% |
| Other (8 patterns) | mixed | 33 | 28 | 5 | 84.8% |

### Key Observations

- **Primary patterns** (`ma_simt`, `mam_simtit`, etc.) are highly precise: 99-100% keep rate. These directly express "I feel [state]" and rarely produce false positives.
- **`sunt_adj_present`** is the noisiest pattern (68.5% keep rate). The ambiguity of "sunt" (I am / they are) plus non-affective adjectives like "sigur" (sure) and "curios" (curious) drive most false positives.
- **`am_noun_present`** is also noisy (62.2%). "Am voie" (I have permission) and "am chef" (I feel like / I'm in the mood) are often non-affective.
- **`mie_short`** and **`imi_este`** are 100% precise — dative constructions like "mi-e frică" (I'm afraid) are unambiguous.

## Results by Emotion Category

| Emotion | Total | Kept | Filtered | Keep Rate |
|---------|-------|------|----------|-----------|
| fear | 292 | 289 | 3 | 99.0% |
| surprise | 118 | 116 | 2 | 98.3% |
| sadness | 284 | 267 | 17 | 94.0% |
| anger | 93 | 86 | 7 | 92.5% |
| anticipation | 967 | 759 | 208 | 78.5% |
| disgust | 27 | 21 | 6 | 77.8% |
| joy | 410 | 282 | 128 | 68.8% |
| trust | 501 | 203 | 298 | 40.5% |

### Key Observations

- **Fear, surprise, sadness, anger** have very high keep rates (92-99%). These are unambiguously emotional.
- **Trust** has the lowest keep rate (40.5%), driven almost entirely by "sunt sigur/sigură" (I'm sure) — which is epistemic certainty, not an affective state.
- **Joy** is moderate (68.8%) because "bine" (well/good) is often used non-affectively ("sunt bine" can mean "I'm fine" as a social response, not a genuine emotional expression).

## Top Filtered Seed Words

These words were most frequently judged as non-affective in context:

| Seed Word | Times Filtered | Typical False Positive |
|-----------|---------------|----------------------|
| sigur | 195 | "sunt sigur că..." (I'm sure that...) — epistemic |
| curios | 173 | "sunt curios ce..." (I'm curious what...) — intellectual interest |
| bine | 57 | "sunt bine" (I'm fine) — social/neutral |
| voie | 31 | "am voie să..." (I'm allowed to...) — permission |
| chef | 21 | "am chef de..." (I feel like...) — desire, not emotion |
| sigură | 13 | feminine form of "sigur" |
| acceptat | 12 | "am fost acceptat" (I was accepted) — event, not emotion |

## Output Files

- `data/llm_filtered_candidates.jsonl` — 1,687 validated ASI candidates
- `data/llm_filter_results.jsonl` — all 2,255 candidates with LLM judgments
- `data/llm_filter_stats.json` — machine-readable statistics
