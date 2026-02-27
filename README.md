# Romanian ASI Benchmark - Project Status

**Last Updated:** 2026-02-27

## Overview

The Romanian ASI (Affective State Identification) Benchmark extracts natural "I feel [state]" expressions from Romanian text corpora, following the MASIVE paper methodology. The goal is to create a dataset of how Romanian speakers naturally express their emotional states.

---

## Current Data Collection Status

### Summary

| Source | Records Processed | ASI Candidates | Status |
|--------|-------------------|----------------|--------|
| Small Datasets (6 sources) | 105,927 | 5,565 | ✅ Complete |
| FULG (web crawl) | 405,000 | 21,184 | ⏸️ Paused (resumable) |
| Filmot/YouTube | - | - | ❌ Blocked |
| **Total** | **510,927** | **26,749** | |

### 1. Small Datasets (Complete)

**Source:** 6 Romanian NLP datasets merged into `data/merged_corpus.jsonl`

| Dataset | Records | ASI Matches | Match Rate |
|---------|---------|-------------|------------|
| RoSent | 27,338 | 2,481 | 9.1% |
| RedditRoAP | 26,269 | 1,788 | 6.8% |
| LaRoSeDa | 14,982 | 1,017 | 6.8% |
| PoPreRo | 28,103 | 467 | 1.7% |
| RED v2 | 5,199 | 186 | 3.6% |
| RED v1 | 4,036 | 131 | 3.2% |

**Output:** `data/asi_candidates.jsonl` (5,565 samples)

**RedditRoAP** ([paper](https://arxiv.org/abs/2410.09907)): 26,517 Romanian Reddit posts from 100+ subreddits with authorship profiling annotations (subdialect, employment status, topic labels, personal inclination). Loaded from HuggingFace: `roship-profiling/reddit_authorship_profiling_romanian`.

### 2. FULG Dataset (Paused - Resumable)

**Source:** FULG web crawl (150B tokens, 289GB) via HuggingFace streaming

**Current Progress:**
- Records streamed: 405,000
- Candidates extracted: 21,184
- Match rate: 7.0%
- Unique domains: 2,792

**Resume command:**
```bash
python -m scripts.fulg.extract_candidates --resume
```

**Output:** `data/fulg_asi_candidates.jsonl` (21,184 samples)

### 3. Filmot/YouTube (Blocked)

**Status:** ❌ Blocked by Cloudflare bot protection

Filmot.com (YouTube subtitle search) uses aggressive bot detection that blocks Playwright browsers even with stealth mode. The pattern matching and transcript fetching components work, but search automation is blocked.

**Potential workarounds:**
1. Manual filmot search → export video IDs → use transcript phase
2. yt-dlp channel crawling for known Romanian YouTube channels
3. Different automation (undetected-chromedriver, Puppeteer)

---

## Extracted Data Analysis

### Emotion Distribution (Combined: 26,749 samples)

| Emotion | Small Datasets | FULG | Total | % |
|---------|----------------|------|-------|---|
| Joy | 2,050 | 7,967 | 10,017 | 31% |
| Anticipation | 1,282 | 3,896 | 5,178 | 16% |
| Trust | 1,261 | 7,170 | 8,431 | 26% |
| Sadness | 1,025 | 4,966 | 5,991 | 19% |
| Surprise | 697 | 1,285 | 1,982 | 6% |
| Fear | 529 | 2,957 | 3,486 | 11% |
| Anger | 289 | 1,201 | 1,490 | 5% |
| Disgust | 57 | 293 | 350 | 1% |

### Pattern Usage

| Pattern | Count | % | Example |
|---------|-------|---|---------|
| sunt_adj_present | 3,837 | 63% | "sunt fericit" |
| am_fost_adj_perfect | 1,022 | 17% | "am fost surprins" |
| eram_adj_imperfect | 337 | 6% | "eram trist" |
| mie_short | 250 | 4% | "mi-e frică" |
| am_noun_present | 219 | 4% | "am teamă" |
| ma_simt_present | 203 | 3% | "mă simt bine" |
| Other patterns | 202 | 3% | Various |

*Note: Pattern counts above are for small datasets only. FULG has similar distribution.*

**Note:** Primary "mă simt" patterns are ~9% of matches. The "sunt" (I am) pattern dominates, which is typical for written Romanian.

### Top Seed Words (FULG)

| Word | Count | Emotion |
|------|-------|---------|
| bine | 2,104 | joy, trust |
| sigur/sigură | 2,501 | trust |
| dor | 1,268 | sadness, anticipation |
| frică | 648 | fear |
| curios/curioasă | 1,030 | anticipation |
| fericit/fericită | 705 | joy |
| mulțumit/mulțumită | 782 | joy |

### Source Categories (FULG only)

| Category | Count | % |
|----------|-------|---|
| other | 11,311 | 58% |
| blog | 7,137 | 36% |
| wiki | 517 | 3% |
| news | 346 | 2% |
| forum | 200 | 1% |
| social | 51 | <1% |

### Context Quality (FULG)

| Metric | Value |
|--------|-------|
| Median context length | 585 chars |
| Average context length | 627 chars |
| Under 500 chars | 33.6% |
| 500-1000 chars | 57.0% |
| Over 1000 chars | 6.4% |

---

## Extraction Strategy Experiments

Five parallel strategies were evaluated on RedditRoAP + PoPreRo (54,623 texts) to improve extraction beyond basic pattern matching. See `EXPERIMENT_CONCLUSIONS.md` for full analysis.

| # | Strategy | Candidates | Novel Finds | Status |
|---|----------|-----------|-------------|--------|
| 1 | Pattern Matching (baseline) | 234 | — | ✅ Complete |
| 2 | MASIVE Bootstrapping | 428 | +4 new words | ✅ Complete |
| 3 | LLM Filtering (Qwen2.5-7B) | 1,687 validated / 2,255 input | Validates others | ✅ Complete |
| 4 | Embedding Similarity | 1,664 | 1,257 novel | ✅ Complete |
| 5 | Distributional Mining | 538 | 392 from new words | ✅ Complete |

**Combined: ~2,800 unique candidates, 1,687 LLM-validated**

### Key Findings from Experiments

1. **Corpus size is the bottleneck** — bootstrapping saturated at round 3 on 54K texts (MASIVE found 1,600 words from 6 seeds on massive English Reddit)
2. **Embeddings give the best ROI** — 5-7x more candidates than regex alone, capturing paraphrases the regex misses
3. **Fear is overrepresented** in Romanian informal text — `mi-e frică` is extremely common
4. **"Trust" is unreliable** — 59.5% false positive rate (words like `sigur` = "sure" used non-emotionally)
5. **LLM filtering provides actionable seed curation** — top false positives: sigur (208x), curios (183x), bine (57x)
6. **`sunt` pattern is double-edged** — productive (63% of matches) but noisy (68.5% precision vs `mă simt` at 99.1%)

### Experiment Data Files

| File | Size | Description |
|------|------|-------------|
| `reddit_baseline_candidates.jsonl` | 543 KB | 234 baseline candidates (6 Ekman emotions) |
| `bootstrapped_asi_candidates.jsonl` | 850 KB | 428 bootstrapping candidates |
| `llm_filtered_candidates.jsonl` | 3.0 MB | 1,687 LLM-validated candidates |
| `llm_filter_results.jsonl` | 4.0 MB | All 2,255 with LLM judgments |
| `embedding_asi_candidates.jsonl` | 3.8 MB | 1,664 embedding-based candidates |
| `distributional_asi_candidates.jsonl` | 1.1 MB | 538 distributional mining candidates |
| `distributional_expanded_seed.json` | 14 KB | 251 discovered emotion words |

---

## Project Structure

```
Romanian_ASI/
├── README.md                            # This file
├── CLAUDE.md                            # Development guide for Claude Code
├── EXPERIMENT_CONCLUSIONS.md            # Strategy comparison & findings
├── EXTRACTION_STRATEGIES.md             # Strategy overview
├── BOOTSTRAPPING_ANALYSIS.md            # Bootstrapping methodology
├── requirements.txt
│
├── data/
│   ├── merged_corpus.jsonl              # 106K records from 6 datasets
│   ├── asi_candidates.jsonl             # 5,565 samples (small datasets)
│   ├── fulg_asi_candidates.jsonl        # 21,184 samples (FULG)
│   ├── emotion_seed.json               # 511 curated affective words
│   ├── fulg_extraction_checkpoint.json  # Resume point
│   ├── fulg_extraction_analysis.json    # Detailed statistics
│   │
│   │   # Experiment outputs
│   ├── reddit_baseline_candidates.jsonl       # Baseline (234)
│   ├── bootstrapped_asi_candidates.jsonl      # Bootstrapping (428)
│   ├── llm_filtered_candidates.jsonl          # LLM-validated (1,687)
│   ├── llm_filter_results.jsonl               # All LLM judgments (2,255)
│   ├── embedding_asi_candidates.jsonl         # Embedding similarity (1,664)
│   ├── distributional_asi_candidates.jsonl    # Distributional mining (538)
│   ├── distributional_expanded_seed.json      # 251 discovered words
│   └── roemolex/                              # RoEmoLex V3 CSV files
│
├── scripts/
│   ├── ro_asi/                          # Core extraction pipeline
│   │   ├── pattern_matcher.py           # 18 Romanian "I feel" patterns
│   │   ├── curated_affective_states.py  # 511 emotion words
│   │   ├── extract_candidates.py        # Small dataset extraction
│   │   ├── merge_datasets.py            # Dataset merger
│   │   ├── emotion_seed.py              # Emotion seed generation
│   │   └── load_roemolex.py             # RoEmoLex lexicon loader
│   │
│   ├── fulg/                            # FULG streaming extraction
│   │   └── extract_candidates.py
│   │
│   ├── filmot/                          # YouTube extraction (blocked)
│   │   ├── config.py
│   │   ├── searcher.py
│   │   ├── transcript_fetcher.py
│   │   └── extract_candidates.py
│   │
│   ├── distributional_mining/           # Pattern-based word discovery
│   │   └── run.py
│   │
│   ├── explore_fulg_dataset.py
│   ├── sample_popplero.py
│   └── sample_reddit_roap.py
│
├── experiments/                         # Extraction strategy experiments
│   ├── baseline_pattern_matching/
│   │   ├── README.md
│   │   └── extract_baseline.py
│   ├── bootstrapping/
│   │   ├── RESULTS.md
│   │   └── bootstrap_candidates.py
│   ├── embedding_similarity/
│   │   ├── ANALYSIS.md
│   │   ├── embedding_candidates.py
│   │   └── modal_embeddings.py          # Modal GPU wrapper
│   └── llm_filtering/
│       ├── RESULTS.md
│       ├── config.py
│       ├── filter_candidates.py
│       └── modal_filter.py              # Modal GPU wrapper
│
├── references/                          # Research papers
│   ├── MASIVE_paper.pdf
│   ├── fulg_paper.pdf
│   └── roemolex_paper.pdf
│
└── small_datasets/                      # Source datasets
    ├── LaRoSeDa/
    ├── PoPreRo/
    ├── RED/
    ├── RedditRoAP/
    └── RoSent/
```

---

## Technical Details

### Pattern Matching

18 Romanian patterns organized into two categories:

**Primary (adjectives with "mă simt"):**
- `mă simt [adj]` - present (99.1% precision)
- `m-am simțit [adj]` - perfect
- `mă simțeam [adj]` - imperfect

**Secondary (nouns with "am", "mi-e"):**
- `sunt [adj]` - present (most common, 68.5% precision)
- `mi-e [noun]` - dative short form
- `am [noun]` - have + emotion noun
- `îmi este [noun]` - dative formal

### Emotion Lexicon

511 manually curated Romanian affective words:
- 348 adjectives (fericit/fericită, trist/tristă, etc.)
- 150 emotion nouns (frică, bucurie, tristețe, etc.)
- 15 state adverbs (bine, rău, groaznic, etc.)

Each word mapped to Plutchik's 8 basic emotions.

### Diacritics Handling

Romanian has 5 special characters (ă, â, î, ș, ț) that are often omitted in informal text. The pattern matcher normalizes both text and patterns to handle:
- `mă simt` = `ma simt`
- `frică` = `frica`
- `mulțumit` = `multumit`

---

## Commands

```bash
# Activate environment
source venv/bin/activate

# Small datasets extraction (complete)
python -m scripts.ro_asi.extract_candidates

# FULG extraction (resume from checkpoint)
python -m scripts.fulg.extract_candidates --resume

# FULG extraction (fresh start with limits)
python -m scripts.fulg.extract_candidates --max-samples 50000

# Test pattern matcher
python -m scripts.ro_asi.pattern_matcher

# Run experiments (on RedditRoAP + PoPreRo)
python -m experiments.baseline_pattern_matching.extract_baseline
python -m experiments.bootstrapping.bootstrap_candidates
python -m experiments.embedding_similarity.embedding_candidates
python -m experiments.llm_filtering.filter_candidates
python -m scripts.distributional_mining.run
```

---

## Next Steps

1. **Curate seed list** using LLM filtering findings (remove problematic words like sigur, curios)
2. **Resume FULG extraction** to reach 50K+ samples target
3. **Run bootstrapping + distributional mining on FULG** (larger corpus should yield more discoveries)
4. **Run LLM filtering as second pass** on all strategies' output
5. **Human annotation** on ~500 samples for real precision numbers
6. **Combine all strategies** into final benchmark with confidence scores
7. **Train/test split** for final benchmark release

---

## Files Summary

| File | Size | Description |
|------|------|-------------|
| `merged_corpus.jsonl` | 49 MB | 105,927 source records (6 datasets) |
| `asi_candidates.jsonl` | 7 MB | 5,565 small dataset samples |
| `fulg_asi_candidates.jsonl` | 37 MB | 21,184 FULG samples |
| `emotion_seed.json` | 40 KB | 511 curated emotion words |
| `fulg_extraction_analysis.json` | 12 KB | Detailed FULG statistics |
| `llm_filtered_candidates.jsonl` | 3 MB | 1,687 LLM-validated candidates |
| `embedding_asi_candidates.jsonl` | 4 MB | 1,664 embedding-based candidates |
| `distributional_expanded_seed.json` | 14 KB | 251 discovered emotion words |
