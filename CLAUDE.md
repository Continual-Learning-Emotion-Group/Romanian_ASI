# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Romanian ASI (Affective State Identification) Benchmark - creates a Romanian language benchmark for identifying affective state expressions following the MASIVE paper methodology. The goal is to extract natural "I feel [state]" expressions from Romanian text corpora.

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Generate curated emotion seed
python scripts/ro_asi/curated_affective_states.py

# Merge all datasets into unified corpus
python -m scripts.ro_asi.merge_datasets

# Extract ASI candidates from corpus
python -m scripts.ro_asi.extract_candidates

# Extract with options
python -m scripts.ro_asi.extract_candidates --sample 20 --max-records 1000

# Run pattern matcher tests
python -m scripts.ro_asi.pattern_matcher

# Load RoEmoLex lexicon (if CSV files present in data/roemolex/)
python -m scripts.ro_asi.load_roemolex --force

# FULG Dataset Extraction (150B tokens, streaming)
# Basic usage - extract up to 50K samples with sentence-level context
python -m scripts.fulg.extract_candidates

# Custom limits
python -m scripts.fulg.extract_candidates --max-samples 100000 --max-records 5000000

# Control context window (sentences before/after match)
python -m scripts.fulg.extract_candidates --context-sentences 3 --max-context-length 1500

# Resume from checkpoint after interruption
python -m scripts.fulg.extract_candidates --resume

# Quick test run
python -m scripts.fulg.extract_candidates --max-records 10000 --max-samples 100

# Filmot/YouTube Transcript Extraction (⚠️ BLOCKED - see notes below)
# Install dependencies first:
pip install playwright playwright-stealth youtube-transcript-api yt-dlp
playwright install chromium

# Basic usage - extract up to 50K samples
python -m scripts.filmot.extract_candidates

# Custom limits
python -m scripts.filmot.extract_candidates --max-samples 10000 --max-videos 5000

# Resume from checkpoint after interruption
python -m scripts.filmot.extract_candidates --resume

# Run specific phase only
python -m scripts.filmot.extract_candidates --phase search       # Only search filmot (⚠️ BLOCKED)
python -m scripts.filmot.extract_candidates --phase transcripts  # Only download transcripts
python -m scripts.filmot.extract_candidates --phase extract      # Only extract from cached

# Quick test run
python -m scripts.filmot.extract_candidates --max-samples 100 --max-videos 50

# Filmot API Extraction (replaces blocked Playwright pipeline)
# Install dependencies first:
pip install python-dotenv filmot

# Phase 1: Collect raw subtitle hits from Filmot API (via RapidAPI)
python -m scripts.filmot_api.collect

# Resume from checkpoint after interruption
python -m scripts.filmot_api.collect --resume

# Limit pages per query (for testing)
python -m scripts.filmot_api.collect --max-pages-per-query 2

# Primary queries only (skip no-diacritic variants)
python -m scripts.filmot_api.collect --no-secondary

# Phase 2: Filter raw hits into ASI candidates (local, no API calls)
python -m scripts.filmot_api.filter_candidates

# Show more sample candidates
python -m scripts.filmot_api.filter_candidates --sample 20

# Phase 3: LLM validation of filmot API candidates (Modal + vLLM)
modal run scripts/filmot_api/llm_validate.py

# Quick test (500 candidates)
modal run scripts/filmot_api/llm_validate.py --max-candidates 500

# Resume from checkpoint after interruption
modal run scripts/filmot_api/llm_validate.py --resume

# Custom batch size
modal run scripts/filmot_api/llm_validate.py --batch-size 100

# Pipeline: Pattern-based extraction (extract_match/)
# Small datasets
python -m pipeline.extract_match.run
python -m pipeline.extract_match.run --max-records 1000 --sample 10

# Filmot API (requires RAPIDAPI_KEY in .env)
python -m pipeline.extract_match.filmot
python -m pipeline.extract_match.filmot --workers 8 --resume

# FULG (HuggingFace streaming, supports parallel shard workers)
python -m pipeline.extract_match.fulg --max-samples 100000 --workers 4
python -m pipeline.extract_match.fulg --resume
python -m pipeline.extract_match.fulg --max-records 10000 --max-samples 100  # quick test

# Extraction Strategy Experiments (on RedditRoAP + PoPreRo)
python -m experiments.baseline_pattern_matching.extract_baseline
python -m experiments.bootstrapping.bootstrap_candidates
python -m experiments.embedding_similarity.embedding_candidates
python -m experiments.llm_filtering.filter_candidates
python -m scripts.distributional_mining.run

# NOTE: Filmot search is currently blocked by Cloudflare. Alternatives:
# - Manually search filmot.com, save video IDs to data/filmot_video_ids.jsonl
# - Then run: python -m scripts.filmot.extract_candidates --phase transcripts
# - Then run: python -m scripts.filmot.extract_candidates --phase extract
```

## Architecture

### Pipeline Flow
```
6 Source Datasets → merge_datasets.py → merged_corpus.jsonl (106K records)
                                              ↓
curated_affective_states.py → emotion_seed.json
                                              ↓
                         pattern_matcher.py (regex matching)
                                              ↓
                    extract_candidates.py → asi_candidates.jsonl

FULG Dataset (150B tokens) → streaming extraction → fulg_asi_candidates.jsonl
     (uses same pattern_matcher.py and emotion_seed.json)

Filmot/YouTube → filmot search → transcript download → filmot_asi_candidates.jsonl
     (uses same pattern_matcher.py and emotion_seed.json)

Filmot API → paginated subtitle search → raw hits JSONL → pattern filter → filmot_api_candidates.jsonl
     (replaces blocked Playwright pipeline, uses filmot Python package via RapidAPI)
```

### Core Modules (`scripts/ro_asi/`) - Small Datasets

These modules handle the 6 smaller source datasets (106K records total):

- **curated_affective_states.py**: Manually curated list of 511 Romanian affective state words (348 adjectives, 150 nouns, 15 adverbs) with emotion mappings
- **pattern_matcher.py**: Regex-based pattern matching with 18 Romanian "I feel" patterns, handles diacritics normalization (ă→a, ș→s, ț→t)
- **extract_candidates.py**: Main extraction pipeline, outputs JSONL with matched sentences and emotion categories
- **merge_datasets.py**: Unifies LaRoSeDa, PoPreRo, RED v1/v2, RoSent, RedditRoAP into common schema
- **load_roemolex.py**: Parses RoEmoLex V3 CSV files (optional, falls back to curated list)

### FULG Extraction (`scripts/fulg/`)

Big-data extraction from the FULG dataset (150B tokens, 289GB):

- **extract_candidates.py**: Streaming extraction using HuggingFace datasets
  - **Sentence-level context extraction**: Extracts 2 sentences before/after match instead of full page (median 576 chars vs 21K)
  - **Soft domain categorization**: Tags sources (forum, social, blog, qa, news, wiki, reviews, other) for analysis without filtering
  - Pre-filters by language score (>0.8), text length, and trigger word presence
  - Checkpoint/resume support for long runs
  - Deduplication via MD5 hashes
  - Reuses `pattern_matcher.py` and `curated_affective_states.py` from `scripts/ro_asi/`

Output: `data/fulg_asi_candidates.jsonl`
```json
{
  "context": "2-3 sentences around the match",
  "context_before": "...",
  "context_after": "...",
  "matched_sentence": "Mă simt fericit",
  "source_category": "blog",  // soft tag for analysis
  "source_domain": "example.ro",
  "url": "...",
  "full_text_length": 15000
}
```

### Filmot Extraction (`scripts/filmot/`) - ⚠️ BLOCKED BY CLOUDFLARE

YouTube transcript extraction via filmot.com subtitle search:

- **config.py**: Extraction configuration dataclass with YouTube filters and rate limiting
- **searcher.py**: Playwright-based filmot search with stealth and cookie persistence
- **transcript_fetcher.py**: youtube-transcript-api wrapper with caching and rate limiting
- **extract_candidates.py**: Three-phase pipeline (search → transcripts → extract)
  - Phase 1: Search filmot.com for Romanian subtitle content
  - Phase 2: Download transcripts via youtube-transcript-api
  - Phase 3: Apply pattern matching (reuses `pattern_matcher.py`)
  - Checkpoint/resume support for long runs
  - Deduplication via MD5 hashes

**⚠️ CURRENT STATUS: BLOCKED**

Filmot.com uses Cloudflare bot protection that blocks Playwright browsers:
- Tried: playwright-stealth, cookie persistence, system Chrome (`channel="chrome"`)
- Issue: Cloudflare detects automation even with stealth mode
- Issue: Google blocks login on Playwright-controlled browsers (needed for Patreon auth)

**Potential alternatives to explore:**
1. **yt-dlp channel crawling**: Crawl known Romanian YouTube channels, download subtitles, search locally
2. **Manual filmot export**: Manually search filmot, export results, then process with pipeline
3. **Different browser automation**: Try undetected-chromedriver (Selenium) or Puppeteer
4. **API access**: Check if filmot offers API access for Patreon supporters

The pattern matching and transcript fetching components are tested and working - only the filmot search phase is blocked.

Output: `data/filmot_asi_candidates.jsonl` (same schema as `asi_candidates.jsonl` with YouTube-specific fields: `video_id`, `video_title`, `channel`, `views`, `duration_seconds`, `upload_date`)

### Filmot API Extraction (`scripts/filmot_api/`) - ACTIVE

Replacement for the blocked Playwright pipeline, using the `filmot` Python package (v0.0.4) via RapidAPI:

- **config.py**: API key setup (loads `.env`), trigger queries (broad verb phrases without emotion words), `FilmotAPIConfig` dataclass
- **collect.py**: Phase 1 — paginated `getsearchsubtitles` API calls, 50 results/page, checkpoint/resume per (query, page), dedup by video_id + hit_start
- **filter_candidates.py**: Phase 2 — local PatternMatcher filtering, zero API calls, dedup by matched text MD5

Two-phase pipeline (vs old 3-phase): API returns subtitle context directly (`ctx_before` + `token` + `ctx_after`), so transcript downloads are unnecessary.

Output files:
- `data/filmot_api_raw_hits.jsonl` — all API hits (Phase 1)
- `data/filmot_api_candidates.jsonl` — pattern-matched ASI candidates (Phase 2)

### Pattern Types

Two pattern categories differentiate adjective vs noun usage:
- **Adjective patterns** (`sunt`, `eram`, `mă simt`): Use curated adjectives
- **Noun patterns** (`am`, `mi-e`, `simt`): Use curated emotion nouns only

This prevents false positives like "sunt elev" (I am a student) being matched as affective state.

### Data Schema

**Merged corpus** (`data/merged_corpus.jsonl`):
```json
{"id": "source_123", "text": "...", "source": "laroseda", "split": "train", "original_labels": {...}}
```

**ASI candidates** (`data/asi_candidates.jsonl`):
```json
{"id": "...", "text": "...", "matched_sentence": "sunt fericit", "pattern_used": "sunt_adj_present", "seed_word": "fericit", "emotion_category": ["joy"], "source": "..."}
```

## Key Considerations

- Romanian has gendered adjectives (fericit/fericită) - both forms must be in the seed
- Diacritics are inconsistent in social media text - patterns match both normalized and original forms
- The "sunt" pattern is ambiguous (1st person "I am" vs 3rd person plural "they are") - some noise is acceptable
- RoEmoLex contains many non-affective words (professions, objects) - use curated list instead for precision

### Experiments (`experiments/`)

5 extraction strategies evaluated on RedditRoAP + PoPreRo (54,623 texts):

- **baseline_pattern_matching/**: 6 Ekman emotions only → 234 candidates (floor comparison)
- **bootstrapping/**: MASIVE-style `"sunt X și Y"` seed expansion → 428 candidates (+4 new words)
- **llm_filtering/**: Qwen2.5-7B-Instruct validates candidates → 1,687 kept / 2,255 input (74.8%)
- **embedding_similarity/**: `intfloat/multilingual-e5-base` semantic search → 1,664 candidates (1,257 novel)
- **distributional_mining/** (in `scripts/`): Pattern-based word discovery → 538 candidates, 235 new words

GPU experiments use Modal (A10G/T4). See `EXPERIMENT_CONCLUSIONS.md` for full analysis.

### Distributional Mining (`scripts/distributional_mining/`)

- **run.py**: Discovers emotion words via explicit labeling patterns (`un sentiment de X`, `plin de X`, etc.)
- Output: `data/distributional_asi_candidates.jsonl`, `data/distributional_expanded_seed.json`

## References

See `EXPERIMENT_CONCLUSIONS.md` for strategy comparison, `BOOTSTRAPPING_ANALYSIS.md` for methodology details, and `/references/` for MASIVE, RoEmoLex, and FULG papers.
