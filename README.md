# Romanian ASI Benchmark - Project Status

**Last Updated:** 2026-02-17

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

## Project Structure

```
Romanian_ASI/
├── README.md                    # This file (project status & documentation)
├── CLAUDE.md                    # Development guide for Claude Code
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── merged_corpus.jsonl              # 106K records from 6 datasets
│   ├── asi_candidates.jsonl             # 4,282 samples (small datasets)
│   ├── asi_candidates.stats.json
│   ├── fulg_asi_candidates.jsonl        # 21,184 samples (FULG)
│   ├── fulg_asi_candidates.stats.json
│   ├── fulg_extraction_checkpoint.json  # Resume point
│   ├── fulg_extraction_analysis.json    # Detailed statistics
│   ├── emotion_seed.json               # 511 curated affective words
│   └── roemolex/                        # RoEmoLex V3 CSV files
│
├── scripts/
│   ├── explore_fulg_dataset.py          # FULG dataset exploration utility
│   │
│   ├── ro_asi/                          # Core extraction pipeline
│   │   ├── pattern_matcher.py           # 18 Romanian "I feel" patterns
│   │   ├── curated_affective_states.py  # 511 emotion words
│   │   ├── extract_candidates.py        # Small dataset extraction
│   │   ├── merge_datasets.py            # Dataset merger
│   │   ├── emotion_seed.py              # Emotion seed generation
│   │   └── load_roemolex.py             # RoEmoLex lexicon loader
│   │
│   ├── fulg/                            # FULG streaming extraction
│   │   └── extract_candidates.py        # HuggingFace streaming + context extraction
│   │
│   └── filmot/                          # YouTube extraction (blocked)
│       ├── config.py                    # Extraction configuration
│       ├── searcher.py                  # Playwright-based search
│       ├── transcript_fetcher.py        # youtube-transcript-api wrapper
│       └── extract_candidates.py        # 3-phase pipeline
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
- `mă simt [adj]` - present
- `m-am simțit [adj]` - perfect
- `mă simțeam [adj]` - imperfect

**Secondary (nouns with "am", "mi-e"):**
- `sunt [adj]` - present (most common)
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
```

---

## Next Steps

1. **Resume FULG extraction** to reach 50K samples target
2. **Explore YouTube alternatives** (yt-dlp channel crawling, manual filmot export)
3. **Quality review** of extracted samples
4. **Annotation** for benchmark validation
5. **Train/test split** for final benchmark

---

## Files Summary

| File | Size | Description |
|------|------|-------------|
| `merged_corpus.jsonl` | 49 MB | 105,927 source records (6 datasets) |
| `asi_candidates.jsonl` | 7 MB | 5,565 small dataset samples |
| `fulg_asi_candidates.jsonl` | 37 MB | 21,184 FULG samples |
| `emotion_seed.json` | 40 KB | 511 curated emotion words |
| `fulg_extraction_analysis.json` | 12 KB | Detailed FULG statistics |
