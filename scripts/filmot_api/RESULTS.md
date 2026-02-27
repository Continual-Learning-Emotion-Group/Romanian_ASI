# Filmot API Extraction Results

**Date**: 2026-02-27
**Pipeline**: `scripts/filmot_api/` (2-phase: API collection → local pattern filtering)

## Overview

| Metric | Value |
|--------|-------|
| API calls used | 230 (1.5% of 150K monthly quota) |
| Raw subtitle hits collected | 37,267 |
| ASI candidates after filtering | 7,622 |
| Match rate (raw → candidate) | 20.5% |
| Unique seed words matched | 212 |
| Unique videos with candidates | 3,053 |
| Unique channels | 602 |
| Patterns matched | 16 of 18 |
| Run time (collection) | ~15 minutes |
| Run time (filtering) | ~3 seconds |

## Phase 1: API Collection

Used 10 primary trigger queries (broad verb phrases, no emotion words) against the Filmot RapidAPI `getsearchsubtitles` endpoint. Each API call returns up to 50 videos, each with multiple subtitle hits.

| Query | Pages | Raw Hits | Notes |
|-------|-------|----------|-------|
| `"mi-e"` | 20 | 13,411 | Most prolific — dative "mi-e frică/teamă/dor/etc." |
| `"mă simt"` | 50 | 10,759 | Core "I feel" pattern, ran full 50 pages |
| `"simt că"` | 20 | 3,072 | "I feel that..." |
| `"m-am simțit"` | 20 | 2,408 | Perfect tense "I felt" |
| `"îmi este"` | 20 | 1,482 | Formal dative "to me it is" |
| `"mă simțeam"` | 20 | 1,433 | Imperfect "I was feeling" |
| `"îmi era"` | 20 | 1,269 | Imperfect dative "to me it was" |
| `"ne simțim"` | 20 | 1,247 | Plural "we feel" |
| `"ne-am simțit"` | 20 | 1,123 | Plural perfect "we felt" |
| `"mă voi simți"` | 20 | 1,063 | Future "I will feel" |
| **Total** | **230** | **37,267** | |

Secondary queries (no-diacritic variants like `"ma simt"`, `"imi este"`) were skipped for this run. Queries like `"sunt"`, `"eram"`, `"am"` were deliberately excluded — too broad, millions of non-affective results.

## Phase 2: Pattern Filtering

Applied the 511-word curated emotion seed via `PatternMatcher` (18 regex patterns) to each raw hit's `full_context` (= `ctx_before` + `token` + `ctx_after`).

- **29,408 hits** had no seed word match (79.5%) — the trigger phrase appeared but wasn't followed by an emotion word
- **919 duplicates** skipped (same matched text seen in different hits)
- **7,622 candidates** kept

### By Pattern

| Pattern | Category | Candidates | % of Total |
|---------|----------|------------|------------|
| `ma_simt_present` | primary | 4,292 | 56.3% |
| `ne_simtim_present` | primary | 637 | 8.4% |
| `sunt_adj_present` | secondary | 626 | 8.2% |
| `imi_era_imperfect` | secondary | 517 | 6.8% |
| `ma_simteam_imperfect` | primary | 468 | 6.1% |
| `imi_este_present` | secondary | 390 | 5.1% |
| `ma_voi_simti_future` | primary | 319 | 4.2% |
| `am_noun_present` | secondary | 138 | 1.8% |
| `eram_adj_imperfect` | secondary | 88 | 1.2% |
| `am_fost_adj_perfect` | secondary | 55 | 0.7% |
| `mie_short` | secondary | 48 | 0.6% |
| `suntem_adj_present` | secondary | 17 | 0.2% |
| `aveam_noun_imperfect` | secondary | 15 | 0.2% |
| `simt_ca` | primary | 8 | 0.1% |
| `simteam_noun` | primary | 2 | <0.1% |
| `simt_noun` | primary | 2 | <0.1% |

Primary patterns: 5,728 (75.1%) — Secondary patterns: 1,894 (24.9%)

### By Emotion

| Emotion | Count | % of Total |
|---------|-------|------------|
| trust | 4,075 | — |
| joy | 3,982 | — |
| sadness | 2,152 | — |
| anger | 1,111 | — |
| fear | 1,019 | — |
| anticipation | 469 | — |
| surprise | 220 | — |
| disgust | 35 | — |

Note: candidates can map to multiple emotions, so counts sum to more than 7,622.

### Top 25 Seed Words

| Seed Word | Count | Emotions |
|-----------|-------|----------|
| bine (well/good) | 3,311 | joy, trust |
| prost (bad) | 591 | sadness, anger |
| frică (fear) | 344 | fear |
| dor (longing) | 340 | sadness, anticipation |
| rău (bad) | 293 | sadness, anger |
| confortabil (comfortable) | 149 | trust |
| vinovată (guilty, f.) | 146 | sadness, fear |
| vinovat (guilty, m.) | 134 | sadness, fear |
| singur (alone, m.) | 100 | sadness |
| în siguranță (safe) | 97 | trust |
| rușine (shame) | 95 | sadness, fear |
| teamă (dread) | 90 | fear |
| fericit (happy, m.) | 89 | joy |
| singură (alone, f.) | 79 | sadness |
| fericită (happy, f.) | 74 | joy |
| aiurea (weird/off) | 73 | surprise, anger |
| extraordinar (extraordinary) | 70 | joy, surprise |
| chef (mood/feel like) | 59 | joy, anticipation |
| milă (pity) | 59 | sadness |
| obosit (tired) | 58 | sadness |
| în largul meu (at ease) | 55 | trust |
| sigur (sure) | 53 | trust |
| încredere (trust) | 45 | trust |
| sigură (sure, f.) | 42 | trust |
| iubit (loved, m.) | 39 | joy, trust |

"bine" dominates because "mă simt bine" (I feel good) is the most common affective expression in Romanian speech.

## Top Channels

| Channel | Candidates | Type |
|---------|------------|------|
| Casa Iubirii | 2,258 | Reality show (Kanal D) |
| Kanal D Romania | 1,357 | TV network (talk shows, reality) |
| GALBEN | 206 | Documentary/interview |
| imoGen LIVE ! | 171 | Live streaming |
| Selly | 126 | YouTuber/vlogger |
| Acasa La Maruta | 105 | Talk show |
| Theo Zeciu | 90 | YouTuber |
| Fain & Simplu cu Mihai Morar | 85 | Podcast |
| SOLD OUT MEDIA | 80 | Media/entertainment |
| Big Man Romania | 61 | Music |
| Radu Tibulca | 57 | Journalism/interview |
| LucaLuk | 48 | YouTuber |
| Survivor Romania | 48 | Reality show |
| JORGE OFFICIAL | 46 | Podcast |
| SNIK | 45 | Entertainment |

Reality TV (Casa Iubirii, Survivor) and talk shows dominate because people express emotions frequently in those formats. The 602 unique channels provide good diversity beyond the top contributors.

## Data Files

| File | Description | Size |
|------|-------------|------|
| `data/filmot_api_raw_hits.jsonl` | All raw API hits (Phase 1) | 37,267 lines |
| `data/filmot_api_candidates.jsonl` | Pattern-matched ASI candidates (Phase 2) | 7,622 lines |
| `data/filmot_api_stats.json` | Collection run statistics | — |
| `data/filmot_api_candidates.stats.json` | Filtering statistics | — |
| `data/filmot_api_checkpoint.json` | Resume checkpoint (for future runs) | — |

## Phase 3: LLM Validation (Modal + vLLM)

Applied Qwen2.5-7B-Instruct via Modal (A10G GPU, vLLM batch inference) to validate whether each candidate is a genuine affective state expression. Pilot run on first 500 candidates (6.6% of total).

### Overview

| Metric | Value |
|--------|-------|
| Candidates processed | 500 |
| Affective (kept) | 487 (97.4%) |
| Not affective (filtered) | 13 (2.6%) |
| Parse errors | 0 |
| Processing time | ~2.5 min (3.2 candidates/sec) |
| Infrastructure | Modal A10G, vLLM 0.6.6.post1 |

### Acceptance Rate: 97.4% vs Expected 75%

The experiment-phase LLM filtering (on RedditRoAP + PoPreRo) kept 74.8% of candidates. The filmot pipeline keeps **97.4%** — dramatically higher. Key reasons:

1. **Pre-filtering quality**: The 2-phase filmot pipeline (API trigger queries → pattern matching) already provides high-precision candidates. The API queries target specific "I feel" verb phrases (`"mă simt"`, `"mi-e"`, etc.), and the pattern matcher further requires a curated emotion seed word. By the time candidates reach the LLM, most are genuine.
2. **YouTube speech patterns**: Spoken Romanian in YouTube subtitles uses "mă simt bine/prost" in genuinely affective contexts more often than written social media text, which has more figurative or idiomatic uses.
3. **Dominant pattern**: 92.8% of the pilot sample uses `ma_simt_present` — the highest-precision "I feel" pattern, which inherently carries affective meaning.

### Confidence Distribution

| Range | Count | % |
|-------|-------|---|
| 0.9–1.0 | 260 | 52.0% |
| 0.8–0.9 | 237 | 47.4% |
| 0.7–0.8 | 3 | 0.6% |
| < 0.7 | 0 | 0.0% |

All predictions are high-confidence (≥0.7). Mean confidence: 0.854 (affective), 0.777 (not-affective).

### Acceptance Rate by Pattern

| Pattern | Total | Kept | Rate |
|---------|-------|------|------|
| `ma_simt_present` | 464 | 454 | 97.8% |
| `sunt_adj_present` | 22 | 20 | 90.9% |
| `eram_adj_imperfect` | 6 | 5 | 83.3% |
| `ne_simtim_present` | 2 | 2 | 100% |
| `ma_simteam_imperfect` | 2 | 2 | 100% |
| Others (4 patterns) | 4 | 4 | 100% |

By category: primary patterns 97.9%, secondary patterns 90.6%. Secondary patterns (`sunt`, `eram`) are slightly noisier as expected — they're more ambiguous verb forms.

### Rejection Analysis

All 13 rejected candidates were correctly identified as non-affective:

- **10/13** were "bine" (good) used in non-emotional contexts: health descriptions ("stare de sănătate"), greetings ("Ce faci? Bine"), referring to someone else's state rather than the speaker's
- **1** was "sigur" (sure) expressing certainty, not emotion ("eram sigur" = "I was sure")
- **1** was "curios" (curious) expressing interest/inquiry, not emotional curiosity
- **1** was "bine" used as a discourse marker

The LLM correctly catches the most common false positive pattern: "mă simt bine" appearing in conversational exchanges where "bine" is a greeting or health status rather than an affective state.

### Emotion Distribution (Affective Only)

| Emotion | Count | % of 487 |
|---------|-------|----------|
| trust | 318 | 65.3% |
| joy | 291 | 59.8% |
| sadness | 132 | 27.1% |
| anger | 85 | 17.5% |
| fear | 45 | 9.2% |
| surprise | 17 | 3.5% |
| disgust | 2 | 0.4% |
| anticipation | 1 | 0.2% |

Trust+joy dominate due to "bine" (264 of 487 candidates). The pilot sample skews toward positive emotions; the full 7,622-candidate run should show more balanced distribution as it includes more `mi-e` (fear/dread) and `imi_era` (longing/shame) patterns.

### Top Seed Words (Affective)

| Seed Word | Count | Translation |
|-----------|-------|-------------|
| bine | 264 | good/well |
| prost | 48 | bad/stupid |
| vinovată | 23 | guilty (f.) |
| rău | 22 | bad |
| vinovat | 16 | guilty (m.) |
| confortabil | 15 | comfortable |
| aiurea | 11 | weird/off |
| în largul meu | 11 | at ease |
| fericită | 4 | happy (f.) |
| emoționată | 4 | moved/emotional (f.) |

### Source Diversity

- **118 unique videos** across **9 channels** in this 500-sample pilot
- Dominated by Casa Iubirii (450/487) — the first 500 candidates are sorted by input order, which clusters by API query result
- Full run will cover 3,053 videos across 602 channels (much more diverse)

### Affective State Vocabulary Coverage

The filmot pipeline discovers **248 unique affective states** out of the 511-word curated seed (**48.5% coverage**). The LLM-validated pilot (487 candidates) covers 55 unique AS.

| Frequency Tier | Words | Description |
|----------------|-------|-------------|
| 50+ occurrences | 22 | Well-attested core vocabulary |
| 10–49 occurrences | 41 | Solid evidence |
| 2–9 occurrences | 111 | Present but sparse |
| 1 occurrence (hapax) | 74 | May not survive LLM validation |

**Top 10 affective states** (all 7,622 candidates):

| Seed Word | Count | Translation |
|-----------|-------|-------------|
| bine | 3,281 | good/well |
| prost | 591 | bad/stupid |
| frică | 341 | fear |
| dor | 340 | longing |
| rău | 278 | bad |
| confortabil | 148 | comfortable |
| vinovată | 141 | guilty (f.) |
| vinovat | 134 | guilty (m.) |
| singur | 99 | alone (m.) |
| în siguranță | 97 | safe |

The remaining ~263 seed words (51.5%) were not found — these are rarer emotional vocabulary unlikely to appear in spoken YouTube Romanian, which tends toward a smaller, more colloquial register. "bine" alone accounts for 43% of all candidates, reflecting how dominant "mă simt bine" is in everyday speech.

### Projected Full Run

Based on the 97.4% acceptance rate:
- **7,622 input candidates** → estimated **~7,424 validated** candidates
- Processing time: ~40 min at 3.2 candidates/sec
- The high acceptance rate suggests the pattern-matching pipeline is already precise enough that LLM validation serves mainly as a quality assurance step rather than a major filter

## Data Files

| File | Description | Size |
|------|-------------|------|
| `data/filmot_api_raw_hits.jsonl` | All raw API hits (Phase 1) | 37,267 lines |
| `data/filmot_api_candidates.jsonl` | Pattern-matched ASI candidates (Phase 2) | 7,622 lines |
| `data/filmot_api_llm_validated.jsonl` | LLM-validated affective candidates (Phase 3) | 487 lines (pilot) |
| `data/filmot_api_llm_results.jsonl` | All candidates with LLM judgments (Phase 3) | 500 lines (pilot) |
| `data/filmot_api_stats.json` | Collection run statistics | — |
| `data/filmot_api_candidates.stats.json` | Filtering statistics | — |
| `data/filmot_api_checkpoint.json` | Resume checkpoint (for future runs) | — |

## Notes

- Only primary queries were used (no secondary/no-diacritic variants). Adding secondary queries could increase yield by ~30-50%.
- `"mă simt"` ran 50 pages while other queries ran 20 pages each. Running all queries to 50 pages would roughly double the dataset.
- The `"mi-e"` query produced the most raw hits (13K) but many are non-affective ("mi-e foame" = I'm hungry, not in seed). This is expected — broad queries maximize recall, pattern filtering ensures precision.
- LLM validation acceptance rate (97.4%) is much higher than the experiment-phase rate (74.8%), confirming the filmot 2-phase pipeline produces high-quality candidates before LLM validation.
- Next step: full LLM validation run on all 7,622 candidates (`modal run scripts/filmot_api/llm_validate.py`).
