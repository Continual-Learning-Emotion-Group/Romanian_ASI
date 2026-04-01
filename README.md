# Romanian ASI Benchmark

**Affective State Identification for Romanian Language**

Last Updated: 2026-04-01

---

## Overview

This project creates a Romanian-language benchmark for Affective State Identification (ASI), following the MASIVE paper methodology. The goal is to extract natural "I feel [state]" expressions from Romanian text corpora — sentences where speakers express genuine emotional or affective states.

**Examples:**
- *"Mă simt fericit"* (I feel happy)
- *"Mi-e frică"* (I'm afraid)
- *"Sunt trist și obosit"* (I am sad and tired)

**Why this matters:**
- No Romanian ASI benchmark exists
- Romanian has unique linguistic challenges: gendered adjectives, diacritics inconsistency, ambiguous copular constructions (`sunt` = both "I am" and "they are")

---

## Data Collection Summary

### Three extraction pipelines

| Source | Records Processed | ASI Candidates | Status |
|--------|-------------------|----------------|--------|
| Small Datasets (6 sources) | 105,927 | 6,070 | Complete |
| FULG (web crawl, 150B tokens) | 405,000 | 21,184 | Paused (resumable) |
| Filmot API (YouTube subtitles) | 37,267 raw hits | 7,622 filtered / 487 LLM-validated | Complete (pilot LLM validation) |
| **Total** | | **~34,876** | |

### Five extraction strategy experiments (on RedditRoAP + PoPreRo, 54,623 texts)

| # | Strategy | Candidates | Novel Finds | Key Idea |
|---|----------|-----------|-------------|----------|
| 1 | Pattern Matching (baseline) | 234 | — | 6 Ekman emotions only |
| 2 | MASIVE Bootstrapping | 428 | +4 new words | Iterative `"sunt X și Y"` seed expansion |
| 3 | LLM Filtering (Qwen2.5-7B) | 1,687 kept / 2,255 input | Validates others' output | LLM judges each candidate |
| 4 | Embedding Similarity | 1,664 | 1,257 novel | Semantic search with multilingual embeddings |
| 5 | Distributional Mining | 538 | 235 candidate words | Discover words via `"sentiment de X"` patterns |

**Combined: ~2,800 unique candidates, 1,687 LLM-validated**

---

## Data Sources

### 1. Small Datasets (Complete)

Six Romanian NLP datasets merged into `data/merged_corpus.jsonl`:

| Dataset | Records | ASI Matches | Match Rate | Domain |
|---------|---------|-------------|------------|--------|
| RoSent | 27,338 | 2,481 | 9.1% | Reviews |
| RedditRoAP | 26,269 | 1,788 | 6.8% | Social media (Reddit) |
| LaRoSeDa | 14,982 | 1,017 | 6.8% | Reviews |
| PoPreRo | 28,103 | 467 | 1.7% | Social media (Reddit) |
| RED v2 | 5,199 | 186 | 3.6% | Emotion-annotated |
| RED v1 | 4,036 | 131 | 3.2% | Emotion-annotated |

**Output:** `data/asi_candidates.jsonl` (6,070 candidates)

**RedditRoAP** ([paper](https://arxiv.org/abs/2410.09907)): 26,517 Romanian Reddit posts from 100+ subreddits with authorship profiling annotations. Loaded from HuggingFace: `roship-profiling/reddit_authorship_profiling_romanian`.

### 2. FULG Dataset (Paused — Resumable)

Streaming extraction from the FULG web crawl (150B tokens, 289GB) via HuggingFace datasets.

| Metric | Value |
|--------|-------|
| Records streamed | 405,000 |
| Candidates extracted | 21,184 |
| Match rate | 7.0% |
| Unique domains | 2,792 |
| Median context length | 585 chars |

Uses sentence-level context extraction (2 sentences before/after match) instead of full pages. Soft domain categorization tags each candidate (blog, forum, news, wiki, social, etc.) without filtering.

**Output:** `data/fulg_asi_candidates.jsonl` (21,184 candidates)

```bash
# Resume from checkpoint
python -m scripts.fulg.extract_candidates --resume
```

### 3. Filmot API / YouTube Subtitles (Complete)

Replacement for the original Playwright-based filmot pipeline (blocked by Cloudflare). Uses the `filmot` Python package via RapidAPI to search YouTube subtitle content.

**Two-phase pipeline:**
1. **API Collection** — paginated `getsearchsubtitles` calls with 10 trigger queries (`"mă simt"`, `"mi-e"`, etc.), 230 API calls total (1.5% of 150K monthly quota)
2. **Pattern Filtering** — local PatternMatcher on raw hits, zero additional API calls

| Metric | Value |
|--------|-------|
| Raw subtitle hits | 37,267 |
| ASI candidates (after filtering) | 7,622 |
| Match rate (raw → candidate) | 20.5% |
| Unique videos | 3,053 |
| Unique channels | 602 |
| Unique seed words matched | 212 / 511 (41.5%) |

**Phase 3: LLM Validation (pilot)** — Qwen2.5-7B-Instruct via Modal validated 500 candidates: **97.4% accepted** (487 kept, 13 filtered). The high acceptance rate (vs 74.8% in experiments) reflects the pipeline's pre-filtering quality — API queries target specific "I feel" verb phrases, so candidates reaching the LLM are already high-precision.

Top channels by candidates: Casa Iubirii (2,258), Kanal D Romania (1,357), GALBEN (206), imoGen LIVE (171), Selly (126), Acasa La Maruta (105).

**Output:**
- `data/filmot_api_raw_hits.jsonl` — all API hits (Phase 1)
- `data/filmot_api_candidates.jsonl` — pattern-matched candidates (Phase 2)
- `data/filmot_api_llm_validated.jsonl` — LLM-validated candidates (Phase 3 pilot, 487)

---

## Extraction Strategy Experiments

All five experiments ran on the same controlled corpus: **RedditRoAP (26,517) + PoPreRo (28,106) = 54,623 texts**. See `EXPERIMENT_CONCLUSIONS.md` for full analysis.

### Strategy 1: Pattern Matching (Baseline)

Runs the 18-regex pipeline restricted to only 6 Ekman emotions (85 word forms).

- **234 candidates** (0.40% match rate)
- Only 28 of 85 seed words appeared
- Fear dominates (62%) — driven by `mi-e frică`
- Establishes the floor: our full 511-word seed gets 13x more candidates

**Output:** `data/reddit_baseline_candidates.jsonl`

### Strategy 2: MASIVE-Style Bootstrapping

Finds conjunction patterns `"sunt X și Y"` where X is a known emotion word, then harvests Y as a new seed.

- **428 candidates** (83% improvement over baseline)
- 4 new words discovered: *mâhnit* (grieving), *afectată* (affected), *nervos* (nervous), *transpirat* (sweaty/anxious)
- Saturated at round 3 — corpus too small for more

**Output:** `data/bootstrapped_asi_candidates.jsonl`

### Strategy 3: LLM-Assisted Filtering

Qwen2.5-7B-Instruct (Modal A10G, vLLM) judges each candidate: "Does this express a genuine affective state?"

- **1,687 kept / 2,255 input** (74.8% acceptance)
- Precision by pattern: `mă simt` 99.1%, `mi-e` 100%, `sunt` 68.5%
- Top false positives: *sigur* (208x, epistemic), *curios* (183x, intellectual), *bine* (57x, formulaic)
- "Trust" category is unreliable (59.5% false positive rate)

**Output:** `data/llm_filtered_candidates.jsonl`, `data/llm_filter_results.jsonl`

### Strategy 4: Embedding Similarity

`intfloat/multilingual-e5-base` embeddings (Modal T4) find sentences semantically similar to known ASI expressions.

- **1,664 candidates** (highest yield), **1,257 novel** (not found by regex)
- Similarity range: 0.81–0.91 (median 0.87)
- Catches colloquial phrasings, non-standard spellings, and expressions outside the curated seed

**Output:** `data/embedding_asi_candidates.jsonl`

### Strategy 5: Distributional Pattern Mining

Discovers new emotion words via explicit labeling patterns: `"un sentiment de X"`, `"stare de X"`, `"plin de X"`, `"cuprins de X"`.

- **538 candidates**, **235 unique words discovered**
- High-precision patterns (`sentiment de`, `cuprins de`) are excellent but rare
- Low-precision patterns (`plin de`) introduce noise (non-emotional nouns)
- Genuine discoveries: *siguranță, iubire, vină, nostalgie, afecțiune, neliniste*

**Output:** `data/distributional_asi_candidates.jsonl`, `data/distributional_expanded_seed.json`

### Cross-Strategy Comparison

```
                    High Precision
                         ^
                         |
         Baseline (234)  |  Bootstrapping (428)
                         |
                         |  LLM Filtering (1,687)
                         |
    Distributional (538) |  Embedding (1,664)
                         |
                         +-------------------------->
                                              High Recall
```

| Strategy | Est. Precision | Recall | Unique Contribution |
|----------|---------------|--------|---------------------|
| Pattern Matching | ~95% | Low | Clean floor, high-confidence samples |
| Bootstrapping | ~85–90% | Low–Medium | Discovers new adjective forms |
| LLM Filtering | 74.8% kept | Medium | Identifies unreliable seed words |
| Embedding Similarity | ~50–70% | High | Catches non-standard expressions |
| Distributional Mining | ~30–50% | Medium | Discovers new emotion nouns |

The strategies are **complementary**: embedding finds 1,257 candidates no other strategy catches; bootstrapping and distributional mining discover words from different signals; LLM filtering is a second pass that improves all others.

---

## Key Findings

1. **Corpus size is the bottleneck** — bootstrapping saturated at round 3 on 54K texts; MASIVE found 1,600 words from 6 seeds on massive English Reddit. Running on FULG would dramatically improve all strategies.
2. **Embeddings give the best ROI** — 5–7x more candidates than regex alone, capturing paraphrases and non-standard phrasings.
3. **Fear is overrepresented** in Romanian informal text — `mi-e frică` is extremely common (60%+ across strategies).
4. **"Trust" is an unreliable emotion category** — 59.5% false positive rate, driven by `sunt sigur` (epistemic certainty, not emotion).
5. **The `sunt` pattern is double-edged** — most productive (63% of matches) but noisiest (68.5% precision vs `mă simt` at 99.1%).
6. **LLM filtering provides actionable seed curation** — removing 5 problematic words (sigur, curios, bine, voie, chef) would eliminate most false positives.
7. **YouTube subtitles are high-quality** — the filmot pipeline's 97.4% LLM acceptance rate (vs 74.8% for Reddit) shows spoken Romanian uses emotion phrases more genuinely.

---

## Technical Details

### Emotion Lexicon

511 manually curated Romanian affective state words:
- 348 adjectives (fericit/fericită, trist/tristă, etc.)
- 150 emotion nouns (frică, bucurie, tristețe, etc.)
- 15 state adverbs (bine, rău, groaznic, etc.)

Each word is mapped to Plutchik's 8 basic emotions: Joy, Trust, Anticipation, Sadness, Surprise, Fear, Anger, Disgust.

### Pattern Matching

18 Romanian patterns in two categories:

**Primary (unambiguous, 99–100% precision):**
- `mă simt [adj]` — "I feel [adj]" (present reflexive)
- `m-am simțit [adj]` — "I felt [adj]" (perfect)
- `mă simțeam [adj]` — "I was feeling [adj]" (imperfect)
- `simt [noun]` — "I feel [noun]"

**Secondary (productive but noisier, 62–84% precision):**
- `sunt [adj]` — "I am [adj]" (most common, also noisiest)
- `mi-e [noun]` — dative short form (e.g., mi-e frică)
- `îmi este [noun]` — dative formal
- `eram [adj]`, `am fost [adj]`, `am [noun]`, etc.

### Diacritics Handling

Romanian has 5 special characters (ă, â, î, ș, ț) often omitted in informal text. The pattern matcher normalizes both text and patterns:
- `mă simt` = `ma simt`
- `frică` = `frica`
- `mulțumit` = `multumit`

### GPU Infrastructure

Experiments using GPU compute run on [Modal](https://modal.com):
- **LLM Filtering:** Qwen2.5-7B-Instruct on A10G via vLLM
- **Embedding Similarity:** `intfloat/multilingual-e5-base` on T4

---

## Emotion Distribution

### Small Datasets + FULG Combined (~27K candidates)

| Emotion | Small Datasets | FULG | Total | % |
|---------|----------------|------|-------|---|
| Joy | 2,050 | 7,967 | 10,017 | 31% |
| Trust | 1,261 | 7,170 | 8,431 | 26% |
| Sadness | 1,025 | 4,966 | 5,991 | 19% |
| Anticipation | 1,282 | 3,896 | 5,178 | 16% |
| Fear | 529 | 2,957 | 3,486 | 11% |
| Surprise | 697 | 1,285 | 1,982 | 6% |
| Anger | 289 | 1,201 | 1,490 | 5% |
| Disgust | 57 | 293 | 350 | 1% |

### Filmot API Candidates (7,622)

| Emotion | Count |
|---------|-------|
| Trust | 4,075 |
| Joy | 3,982 |
| Sadness | 2,152 |
| Anger | 1,111 |
| Fear | 1,019 |
| Anticipation | 469 |
| Surprise | 220 |
| Disgust | 35 |

*Note: Candidates can map to multiple emotions, so counts sum to more than the candidate total.*

### Top Seed Words

| Word | Translation | Count (FULG) | Count (Filmot) | Emotions |
|------|-------------|-------------|----------------|----------|
| bine | good/well | 2,104 | 3,311 | joy, trust |
| sigur/sigură | sure | 2,501 | 95 | trust |
| dor | longing | 1,268 | 340 | sadness, anticipation |
| frică | fear | 648 | 344 | fear |
| curios/curioasă | curious | 1,030 | — | anticipation |
| prost | bad | — | 591 | sadness, anger |
| fericit/fericită | happy | 705 | 163 | joy |
| mulțumit/mulțumită | content | 782 | — | joy |
| vinovat/vinovată | guilty | — | 280 | sadness, fear |

---

## Project Structure

```
Romanian_ASI/
├── README.md                            # This file
├── CLAUDE.md                            # Development guide for Claude Code
├── EXPERIMENT_CONCLUSIONS.md            # Full strategy comparison & findings
├── EXTRACTION_STRATEGIES.md             # Strategy overview & common schema
├── BOOTSTRAPPING_ANALYSIS.md            # Bootstrapping methodology deep dive
├── PRESENTATION_SLIDES.md              # Slide deck (markdown format)
├── requirements.txt
│
├── data/
│   ├── merged_corpus.jsonl              # 106K records from 6 datasets (49 MB)
│   ├── asi_candidates.jsonl             # 6,070 small dataset candidates (7 MB)
│   ├── fulg_asi_candidates.jsonl        # 21,184 FULG candidates (37 MB)
│   ├── emotion_seed.json               # 511 curated affective words
│   ├── fulg_extraction_checkpoint.json  # FULG resume point
│   ├── fulg_extraction_analysis.json    # Detailed FULG statistics
│   │
│   │   # Filmot API outputs
│   ├── filmot_api_raw_hits.jsonl        # 37,267 raw API hits (34 MB)
│   ├── filmot_api_candidates.jsonl      # 7,622 pattern-filtered candidates (7 MB)
│   ├── filmot_api_llm_validated.jsonl   # 487 LLM-validated (pilot)
│   ├── filmot_api_llm_results.jsonl     # 500 with LLM judgments (pilot)
│   ├── filmot_api_stats.json            # Collection statistics
│   ├── filmot_api_checkpoint.json       # API resume point
│   │
│   │   # Experiment outputs
│   ├── reddit_baseline_candidates.jsonl       # Baseline (234)
│   ├── bootstrapped_asi_candidates.jsonl      # Bootstrapping (428)
│   ├── bootstrap_expanded_seed.json           # Bootstrapping expanded seed
│   ├── bootstrap_provenance.json              # Bootstrapping chain provenance
│   ├── llm_filtered_candidates.jsonl          # LLM-validated (1,687)
│   ├── llm_filter_results.jsonl               # All LLM judgments (2,255)
│   ├── llm_filter_stats.json                  # LLM filtering statistics
│   ├── embedding_asi_candidates.jsonl         # Embedding similarity (1,664)
│   ├── distributional_asi_candidates.jsonl    # Distributional mining (538)
│   ├── distributional_expanded_seed.json      # 251 discovered words
│   ├── distributional_discovered_words.json   # Word discovery details
│   ├── distributional_stats.json              # Mining statistics
│   └── roemolex/                              # RoEmoLex V3 CSV files (optional)
│
├── scripts/
│   ├── ro_asi/                          # Core extraction pipeline
│   │   ├── curated_affective_states.py  # 511 emotion words with Plutchik mappings
│   │   ├── pattern_matcher.py           # 18 Romanian "I feel" regex patterns
│   │   ├── extract_candidates.py        # Small dataset extraction
│   │   ├── merge_datasets.py            # Unifies 6 datasets into common schema
│   │   ├── emotion_seed.py              # Emotion seed generation
│   │   └── load_roemolex.py             # RoEmoLex V3 lexicon loader
│   │
│   ├── fulg/                            # FULG streaming extraction
│   │   └── extract_candidates.py        # HuggingFace streaming, checkpoint/resume
│   │
│   ├── filmot/                          # YouTube extraction via Playwright (BLOCKED)
│   │   ├── config.py                    # Extraction configuration
│   │   ├── searcher.py                  # Playwright-based filmot search
│   │   ├── transcript_fetcher.py        # youtube-transcript-api wrapper
│   │   └── extract_candidates.py        # Three-phase pipeline
│   │
│   ├── filmot_api/                      # YouTube extraction via RapidAPI (ACTIVE)
│   │   ├── config.py                    # API key, trigger queries, settings
│   │   ├── collect.py                   # Phase 1: paginated API collection
│   │   ├── filter_candidates.py         # Phase 2: local pattern filtering
│   │   ├── llm_validate.py             # Phase 3: Modal + vLLM validation
│   │   └── RESULTS.md                   # Detailed results & analysis
│   │
│   ├── distributional_mining/           # Pattern-based word discovery
│   │   └── run.py                       # Discovers words via "sentiment de X" etc.
│   │
│   ├── explore_fulg_dataset.py          # FULG dataset exploration utility
│   ├── sample_popplero.py               # PoPreRo sampling utility
│   └── sample_reddit_roap.py            # RedditRoAP sampling utility
│
├── experiments/                         # Extraction strategy experiments (Reddit-only)
│   ├── baseline_pattern_matching/
│   │   ├── extract_baseline.py          # 6 Ekman emotions baseline
│   │   └── README.md
│   ├── bootstrapping/
│   │   ├── bootstrap_candidates.py      # MASIVE-style seed expansion
│   │   └── RESULTS.md
│   ├── embedding_similarity/
│   │   ├── embedding_candidates.py      # Semantic similarity extraction
│   │   ├── modal_embeddings.py          # Modal GPU wrapper
│   │   └── ANALYSIS.md
│   ├── llm_filtering/
│   │   ├── filter_candidates.py         # LLM validation pipeline
│   │   ├── modal_filter.py              # Modal GPU wrapper
│   │   ├── config.py                    # LLM prompt & settings
│   │   └── RESULTS.md
│   └── __init__.py
│
├── references/                          # Research papers
│   ├── MASIVE_paper.pdf
│   ├── fulg_paper.pdf
│   ├── roemolex_paper.pdf
│   └── Working Emotoin Vocab (1).pdf
│
└── small_datasets/                      # Source datasets
    ├── LaRoSeDa/
    ├── PoPreRo/
    ├── RED/
    ├── RedditRoAP/
    └── RoSent/
```

---

## Commands

```bash
# Activate environment
source venv/bin/activate

# --- Core Pipeline ---

# Generate curated emotion seed
python scripts/ro_asi/curated_affective_states.py

# Merge all datasets into unified corpus
python -m scripts.ro_asi.merge_datasets

# Extract ASI candidates from small datasets
python -m scripts.ro_asi.extract_candidates

# Test pattern matcher
python -m scripts.ro_asi.pattern_matcher

# Load RoEmoLex lexicon (optional, falls back to curated list)
python -m scripts.ro_asi.load_roemolex --force

# --- FULG Extraction ---

# Fresh start with limits
python -m scripts.fulg.extract_candidates --max-samples 50000

# Resume from checkpoint
python -m scripts.fulg.extract_candidates --resume

# Quick test
python -m scripts.fulg.extract_candidates --max-records 10000 --max-samples 100

# --- Filmot API Extraction ---

# Phase 1: Collect raw subtitle hits
python -m scripts.filmot_api.collect
python -m scripts.filmot_api.collect --resume                   # resume
python -m scripts.filmot_api.collect --max-pages-per-query 2    # test run

# Phase 2: Filter into ASI candidates (local, no API)
python -m scripts.filmot_api.filter_candidates

# Phase 3: LLM validation (Modal + vLLM)
modal run scripts/filmot_api/llm_validate.py
modal run scripts/filmot_api/llm_validate.py --max-candidates 500   # pilot
modal run scripts/filmot_api/llm_validate.py --resume               # resume

# --- Strategy Experiments (RedditRoAP + PoPreRo) ---

python -m experiments.baseline_pattern_matching.extract_baseline
python -m experiments.bootstrapping.bootstrap_candidates
python -m experiments.embedding_similarity.embedding_candidates
python -m experiments.llm_filtering.filter_candidates
python -m scripts.distributional_mining.run
```

---

## Data Schema

### Merged Corpus (`data/merged_corpus.jsonl`)
```json
{"id": "source_123", "text": "...", "source": "laroseda", "split": "train", "original_labels": {...}}
```

### ASI Candidates (`data/asi_candidates.jsonl`)
```json
{"id": "...", "text": "...", "matched_sentence": "sunt fericit", "pattern_used": "sunt_adj_present", "seed_word": "fericit", "emotion_category": ["joy"], "source": "..."}
```

### FULG Candidates (`data/fulg_asi_candidates.jsonl`)
```json
{"context": "2-3 sentences around the match", "context_before": "...", "context_after": "...", "matched_sentence": "Mă simt fericit", "source_category": "blog", "source_domain": "example.ro", "url": "...", "full_text_length": 15000}
```

### Filmot API Candidates (`data/filmot_api_candidates.jsonl`)
```json
{"video_id": "...", "video_title": "...", "channel": "...", "matched_sentence": "mă simt bine", "pattern_used": "ma_simt_present", "seed_word": "bine", "emotion_category": ["joy", "trust"], "full_context": "..."}
```

### Experiment Candidates (common schema)
```json
{"id": "...", "text": "...", "matched_sentence": "...", "extraction_strategy": "pattern_matching|bootstrapping|llm_filtering|embedding_similarity|distributional_mining", "confidence": 0.95, "seed_word": "...", "emotion_category": ["joy"], "source": "reddit_roap", "metadata": {}}
```

---

## Next Steps

1. **Curate seed list** using LLM filtering findings — remove or flag sigur, curios, voie, chef; keep bine with context-dependent validation
2. **Complete filmot LLM validation** — run full 7,622 candidates through Qwen2.5-7B (projected ~7,424 validated)
3. **Resume FULG extraction** to reach 50K+ samples target
4. **Run bootstrapping + distributional mining on FULG** — larger corpus should yield hundreds of new words
5. **Run LLM filtering as second pass** on all strategies' output
6. **Human annotation** on ~500 samples across strategies for real precision numbers
7. **Combine all strategies** into final benchmark with confidence scores
8. **Train/test split** for final benchmark release

### Projected Final Yield

| Source | Conservative | Optimistic |
|--------|-------------|------------|
| Small datasets (all strategies) | ~3,000 | ~3,500 |
| FULG (expanded seeds + embeddings) | ~50,000 | ~100,000+ |
| Filmot API (full LLM validation) | ~7,400 | ~7,600 |
| **Total** | **~60,400** | **~111,100+** |

---

## References

- **MASIVE** — Deas et al. (2024). Affective State Identification benchmark (English). See `references/MASIVE_paper.pdf`.
- **FULG** — Romanian web crawl corpus (150B tokens). See `references/fulg_paper.pdf`.
- **RoEmoLex** — Romanian Emotion Lexicon V3. See `references/roemolex_paper.pdf`.
- **RedditRoAP** — [Romanian Reddit Authorship Profiling](https://arxiv.org/abs/2410.09907). HuggingFace: `roship-profiling/reddit_authorship_profiling_romanian`.

See also: `EXPERIMENT_CONCLUSIONS.md`, `BOOTSTRAPPING_ANALYSIS.md`, `EXTRACTION_STRATEGIES.md` for detailed analysis.
