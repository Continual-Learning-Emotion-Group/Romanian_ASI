# Pipeline

Unified pipeline for constructing the Romanian ASI benchmark.

```
pipeline/
├── README.md
├── data/                              # Pipeline outputs
│   ├── merged_corpus.jsonl            # Unified corpus (106K records)
│   ├── enriched_seed.json             # Small-dataset enriched seed (377 words)
│   ├── enriched_seed_merged.json      # Final merged enriched seed (524 words)
│   ├── fulg_enrichment_filtered.json  # FULG enrichment: 135 new nouns (manually filtered)
│   ├── filmot_enrichment_filtered.json # Filmot enrichment: 4 new nouns (manually filtered)
│   ├── filmot_raw.jsonl               # Raw filmot subtitle hits (88K records)
│   ├── pattern_candidates_small.jsonl  # Extraction from small datasets (5.2K candidates)
│   ├── pattern_candidates_filmot.jsonl # Extraction from Filmot API (29K candidates)
│   ├── pattern_candidates_filmot_pp.jsonl    # Post-processed: Stanza 1st-person trimming
│   ├── pattern_candidates_filmot_light.jsonl # Post-processed: light (>>, periods, fragments)
│   ├── embedding_asi_candidates.jsonl # Embedding extraction raw output (all hits >= 0.75)
│   ├── embedding_asi_candidates_filtered.jsonl # Filtered (bad noun anchors removed)
│   └── ...                            # Provenance files, checkpoints, stats
├── utils/                             # Shared utilities
│   ├── text_utils.py                  # Diacritics normalization, sentence splitting
│   ├── inflect.py                     # Lemma → inflected forms (MULTEXT-East)
│   ├── pattern_matcher.py             # 20 "I feel" patterns (1st person sg), PatternMatcher
│   ├── corpus_reader.py               # Unified JSONL reader + FULG streamer
│   └── stoplists.py                   # Minimal stoplist, gender inference
├── seed/                              # Seed construction
│   ├── bridge.py                      # Bridge-only seed (229 words)
│   ├── merged.py                      # Bridge + old seed merged (375 words)
│   ├── enriched.py                    # Loader for enriched seed
│   ├── test_wn_affect_bridge.py       # Bridge script (RoEmoLex × WN-Affect)
│   ├── wn_affect_bridge_results.json  # Raw bridge output (398 words)
│   ├── wn-affect-1.1/                 # WordNet-Affect data
│   ├── wn-mappings/                   # WN 1.6→3.0 offset mappings
│   └── multext-east/                  # Romanian morphological lexicon
├── collect/                           # Data collection
│   ├── merge_small.py                 # Merge 6 small datasets
│   ├── stream_fulg.py                 # Stream from FULG (trigger-filtered)
│   ├── stream_filmot.py               # Stream from Filmot API (parallel workers)
│   └── small_datasets/                # Raw source data
├── extract_match/                     # Pattern-based ASI candidate extraction
│   ├── run.py                         # Small datasets → pattern_candidates_small.jsonl
│   ├── filmot.py                      # Filmot API collect+filter → pattern_candidates_filmot.jsonl
│   ├── postprocess_filmot.py          # Stanza-based: trim to 1st-person context (_pp.jsonl)
│   └── postprocess_filmot_light.py    # Light: split >>, add periods, trim fragments (_light.jsonl)
├── extract_embed/                     # Embedding similarity ASI extraction
│   ├── run.py                         # Main pipeline (Modal GPU + E5-base)
│   ├── modal_embeddings.py            # Modal GPU embedder class
│   ├── filter_results.py              # Post-processing: remove bad-anchor hits
│   └── ANALYSIS.md                    # Experiment results & conclusions
└── seed_enrichment/                   # Seed enrichment
    ├── run.py                         # CLI: runs both methods on any source
    ├── bootstrapping.py               # MASIVE-style "I feel X and Y"
    ├── distributional.py              # "un sentiment de X" discovery
    ├── merge_results.py               # Combine bootstrap + distributional per source
    └── merge_all_sources.py           # Merge original seed + all filtered results
```

## Seed Construction (`seed/`)

The seed is a curated list of Romanian lemmas (masculine singular adjectives,
base-form nouns, adverbs) that describe **affective states** — how someone
feels internally. These words are used downstream by the pattern matcher to
find "I feel [X]" expressions in Romanian text.

### Method

The seed was built in two stages, then merged.

**Stage 1: RoEmoLex × WordNet-Affect bridge** (`seed/bridge.py`, 229 words)

RoEmoLex V3 is a Romanian emotion lexicon with ~9K words, but most entries are
emotion-*associated* (e.g., "accident" tagged fear/sadness) rather than
emotion-*describing*. To filter it down to genuine affective states, we bridged
it with WordNet-Affect 1.1:

1. WordNet-Affect labels ~798 WordNet synsets as affective (emotions, moods,
   traits, etc.), but uses WN 1.6 offsets
2. RoEmoLex links each word to a WN 3.0 synset ID
3. UPC/TALP mapping files (`seed/wn-mappings/`) convert WN 1.6 → 3.0 offsets
4. Join: if a RoEmoLex word's WN 3.0 synset appears in WordNet-Affect → keep

This produced 398 candidate words, which were then manually reviewed:
- 199 passed (genuine affective states)
- 156 failed (causative adjectives like "enervant", external qualities like
  "admirabil", event nouns like "tortură", social concepts like "fraternitate")
- 43 questionable
- 33 salvaged from fails by converting causative forms to felt-state
  participles (e.g., "enervant" → "enervat", "cutremurător" → "cutremurat")

Bridge script: `seed/test_wn_affect_bridge.py`
Bridge results: `seed/wn_affect_bridge_results.json`

**Stage 2: Merge with old curated seed** (`seed/merged.py`, 375 words)

The previous hand-curated seed (`scripts/ro_asi/curated_affective_states.py`,
511 entries including gender/diacritics variants) covered many common words
that the bridge missed — either because they aren't in RoEmoLex at all
(e.g., "stresat", "obosit") or have no valid WN synset ID (e.g., "dezamăgit",
"plictisit" marked as DEX).

To merge:
1. Deduplicated old seed to base lemmas (removed feminine forms, diacritics
   variants)
2. Removed words that failed quality review (sigur, voie, chef, violent,
   agresiv, beligerant, confident, etc.)
3. Added remaining 148 words that weren't already in the bridge seed

### Output files

| File | Words | Description |
|------|-------|-------------|
| `seed/bridge.py` | 229 | Bridge-only seed (WN-Affect validated) |
| `seed/merged.py` | 375 | Bridge + old curated seed (recommended) |

Run with `python -m pipeline.seed.bridge` or `python -m pipeline.seed.merged`
to export to JSON.

### Quality criteria

Every word in both seeds was checked against:

1. **Pattern fit** — must work grammatically in Romanian "I feel" patterns:
   - Adjectives: "mă simt [X]", "sunt [X]"
   - Nouns: "mi-e [X]", "am [X]", "simt [X]"
   - Adverbs: "mă simt [X]"

2. **Affective state** — must describe internal felt experience:
   - Yes: emotions, moods, bodily-felt states, psychological states
   - No: epistemic states, personality traits, external qualities,
     causative/stimulus adjectives, social roles, events/situations

## Data Collection (`collect/`)

### Small Datasets (`collect/merge_small.py`)

Merges 6 Romanian NLP datasets into `data/merged_corpus.jsonl` (106K records)
with a unified schema. Deduplicates by MD5 text hash.

| Dataset | Records | Type |
|---------|---------|------|
| LaRoSeDa | 14,982 | Product reviews (sentiment) |
| PoPreRo | 28,103 | News articles (popularity) |
| RED v1 | 4,036 | Tweets, single-label emotion (5 classes) |
| RED v2 | 5,199 | Tweets, multi-label emotion (7 classes) |
| RoSent | 27,338 | Reviews (sentiment, binary) |
| RedditRoAP | 26,269 | Reddit posts (authorship profiling) |

Run with `python -m pipeline.collect.merge_small`.

Output schema:
```json
{"id": "source_123", "text": "...", "source": "laroseda", "split": "train", "original_labels": {...}}
```

Raw datasets live in `collect/small_datasets/` (not checked into git).

### FULG (`collect/stream_fulg.py`)

Streams records from the FULG dataset (150B tokens, 289GB Romanian web text)
via HuggingFace Datasets in streaming mode. By default, filters by trigger words
from the pattern matcher to keep only records likely to contain "I feel" patterns.

Run with `python -m pipeline.collect.stream_fulg --max-records 50000`.

Options: `--min-language-score 0.8`, `--min-text-length 100`, `--no-trigger-filter`.

### Filmot API (`collect/stream_filmot.py`)

Streams raw subtitle hits from the Filmot API (RapidAPI) by searching for
Romanian "I feel" trigger phrases in YouTube subtitles. Queries are sourced
from `utils/pattern_matcher.get_filmot_queries()` (single source of truth).
No pattern matching — saves raw subtitle context for downstream filtering.

Supports checkpoint/resume for long collection runs.

Requires: `pip install filmot python-dotenv` and `RAPIDAPI_KEY` in `.env`.

```bash
# Collect filmot data (run before enrichment)
python -m pipeline.collect.stream_filmot --max-hits 100000
python -m pipeline.collect.stream_filmot --max-hits 100000 --resume          # resume interrupted run
python -m pipeline.collect.stream_filmot --max-pages-per-query 200           # deeper pagination
python -m pipeline.collect.stream_filmot --workers 8                         # parallel query workers
python -m pipeline.collect.stream_filmot --no-secondary                      # skip no-diacritic variants
```

## Shared Utilities (`utils/`)

### Pattern Matcher (`utils/pattern_matcher.py`)

20 Romanian "I feel" regex patterns (first person singular only). Auto-expands
lemma seeds to masc/fem singular + diacritic variants via MULTEXT-East.

New patterns added: `o să mă simt` (colloquial future), `o să fiu` (colloquial
future of "to be"), `m-aș simți` (conditional), `să mă simt` (subjunctive),
`mă fac` (reflexive "I become").

Also exports `get_trigger_words()` and `get_filmot_queries()` for seed
enrichment collection, and `get_filmot_queries_all()` for extraction (adds
"sunt", "eram", "am fost", "mă fac", "aveam" triggers).

### Inflection (`utils/inflect.py`)

Expands lemmas to masculine + feminine singular forms for "I feel" patterns
using MULTEXT-East (428K entries). E.g., `fericit` → `{fericit, fericită, fericita}`.
No plurals, articles, oblique/vocative — only forms that appear after "mă simt".

### Corpus Reader (`utils/corpus_reader.py`)

Unified JSONL reader: `iter_corpus(data_dir)` yields `(id, text, source)` from
all `*.jsonl` files in `pipeline/data/`. Handles different text field names
(`text`, `full_context`). Optional trigger word pre-filter.

## Seed Enrichment (`seed_enrichment/`)

Discovers new seed words from text data via two methods, then manually filters
results. Supports three data sources: small datasets (JSONL), FULG (streaming),
and Filmot (JSONL from API).

```bash
# Run enrichment on a specific source
python -m pipeline.seed_enrichment.run                          # small datasets only
python -m pipeline.seed_enrichment.run --source filmot          # filmot JSONL only
python -m pipeline.seed_enrichment.run --source fulg            # FULG streaming only
python -m pipeline.seed_enrichment.run --source all             # small + FULG + filmot
python -m pipeline.seed_enrichment.run --source fulg --fulg-max-records 500000
python -m pipeline.seed_enrichment.run --source filmot --filmot-path /path/to/file.jsonl

# After manually filtering results into *_enrichment_filtered.json files,
# merge everything into the final enriched seed:
python -m pipeline.seed_enrichment.merge_all_sources
```

### Method 1: Bootstrapping (`bootstrapping.py`)

MASIVE-style conjunction mining: finds "mă simt X și Y" patterns where X is a
known seed word and Y is a candidate. Uses only unambiguous "simt" verb forms
(no "sunt"/"eram" — too ambiguous with 3rd person). Iterative (4 rounds) for
small datasets, single-pass for streaming/JSONL sources. Starts from the
375-word merged seed. Validates by co-occurrence threshold, gender agreement,
and stopword filtering.

**Results:** Bootstrapping produced no usable words across all three sources.
On small datasets (106K records): only 1 word accepted. On FULG (500K records):
10 accepted, all garbage (conjunctions, verbs: "încerc", "iar", "deși").
On filmot (88K records): 28 accepted, all garbage ("simt", "știi", "mulțumesc").
The method finds many conjunction matches but the Y words are rarely affective
states — co-occurring words tend to be filler (adverbs, verbs, conjunctions).

### Method 2: Distributional Mining (`distributional.py`)

Discovers emotion words via explicit labeling patterns (no seed needed):
"un sentiment de X", "sentimentul de X", "emoție de X", "o senzație de X",
"copleșit de X". Primarily finds nouns. Excluded patterns that produced too
much noise: "stare de" (matches "starea de urgență"), "plin de" (matches
"plin de gauri"), "cuprins de" (matches "cuprinsă de flăcări").

**Results:** Distributional mining was the productive method. FULG (500K records)
found 570 raw candidates, manually filtered to 135 genuine affective nouns.
Filmot found 13 candidates, filtered to 4 new nouns. Small datasets found 20
candidates, filtered to 2.

### Filtering

Raw enrichment results contain significant noise: articulated forms (moartea,
dragostea), plural forms (emoțiile, frici), garbled diacritics (fricã, aparteneþã),
foreign words (doom, bukkake), wrong parts of speech, and non-affective concepts.
Each source's results are manually reviewed and saved to `*_enrichment_filtered.json`
files with full documentation of what was removed and why.

### Merging (`merge_all_sources.py`)

After filtering, `merge_all_sources.py` unions the original seed with all
filtered results. First source to introduce a word wins. No cross-source
validation.

```bash
python -m pipeline.seed_enrichment.merge_all_sources
```

### Output files

| File | Description |
|------|-------------|
| `data/enriched_seed.json` | Small-dataset enrichment only (377 words) |
| `data/fulg_enrichment_filtered.json` | FULG: 135 new nouns (manually filtered, with metadata) |
| `data/filmot_enrichment_filtered.json` | Filmot: 4 new nouns (manually filtered, with metadata) |
| `data/enriched_seed_merged.json` | **Final merged seed: 524 words** (193 adj + 304 nouns + 27 adv) |
| `data/bootstrap_*_provenance.json` | Raw bootstrapping results per source |
| `data/distributional_*_discovered.json` | Raw distributional results per source |

The final enriched seed is loadable via `pipeline.seed.enriched.build_enriched_seed()`.

### Results summary

| Source | Records | Bootstrapping | Distributional | After filtering |
|--------|---------|---------------|----------------|-----------------|
| Small datasets | 106K | 1 word | 20 candidates → 2 new | +2 nouns |
| FULG | 500K | 0 usable | 570 candidates → 135 new | +134 nouns |
| Filmot | 88K | 0 usable | 13 candidates → 4 new | +3 nouns |
| **Total** | | | | **375 → 524 words** |

Note: new words are stored without diacritics (normalized form) because the
distributional mining operates on normalized text. This is intentional — the
pattern matcher normalizes during matching anyway, so diacritic-free storage
avoids errors from incorrect diacritic restoration.

## Pattern Extraction (`extract_match/`)

Extracts ASI candidates from text corpora using the enriched seed (524 words)
and 20 "I feel" regex patterns. Each hit = one pattern match (trigger + seed
word) in one text. Texts can produce multiple candidates if different patterns
fire. Deduplication by MD5 text hash.

### Small Datasets (`extract_match/run.py`)

Reads `data/merged_corpus.jsonl` (106K records). Full text preserved (reviews,
tweets, and Reddit posts are short enough).

```bash
python -m pipeline.extract_match.run                          # full extraction
python -m pipeline.extract_match.run --max-records 1000       # quick test
python -m pipeline.extract_match.run --sample 10              # show samples
```

Output: `data/pattern_candidates_small.jsonl`

**Results** (106K records):
- 5,227 candidates from 4,788 texts (4.5% hit rate)
- 237 unique seed words matched
- Top patterns: `sunt_adj_present` (3,090), `am_fost_adj_perfect` (1,001)
- Top emotions: contentment (1,105), anticipation (928), sadness (791), joy (624)
- Dominated by product review "sunt mulțumit/dezamăgit" from LaRoSeDa/RoSent

### Filmot API (`extract_match/filmot.py`)

Queries **all** trigger phrases (27 queries: existing "simt"-family + new
"sunt", "eram", "am fost", "mă fac", "aveam") via the Filmot subtitle search
API. Each hit is immediately filtered through PatternMatcher — only candidates
where a seed word follows the trigger are saved. Combined collect+extract in
one step.

Query list defined in `utils/pattern_matcher.get_filmot_queries_all()`. Parallel
workers, checkpoint/resume. The API hits a server-side pagination limit at ~102
pages per query (500 error), which the script handles gracefully.

Requires: `pip install filmot python-dotenv` and `RAPIDAPI_KEY` in `.env`.

```bash
python -m pipeline.extract_match.filmot                                     # default
python -m pipeline.extract_match.filmot --workers 8 --max-hits 1000000      # full run
python -m pipeline.extract_match.filmot --resume                            # resume
python -m pipeline.extract_match.filmot --max-pages-per-query 2             # quick test
```

Output: `data/pattern_candidates_filmot.jsonl`

**Results** (200 pages/query, 8 workers):
- 331K hits scanned, 29,452 candidates saved (8.9% hit rate)
- 364 unique seed words matched
- Top patterns: `ma_simt_present` (10,071), `sunt_adj_present` (5,656),
  `eram_adj_imperfect` (2,845)
- Top emotions: joy (9,791), sadness (7,282), negative-fear (1,847)
- Text is ~250 chars of YouTube auto-caption context (no periods, speaker
  changes marked with `>>`, ~96% auto-generated subtitles)

### Filmot Post-Processing

YouTube auto-captions lack punctuation and may contain multiple speakers. Two
post-processing scripts clean up the filmot candidates. Both preserve the
original text and write to separate output files.

**Heavy: Stanza-based 1st-person trimming** (`postprocess_filmot.py`)

Uses Stanza NLP (tokenizer + POS + dependency parser) to detect the grammatical
person of each sentence's root verb. Finds the anchor sentence (containing the
match), expands outward keeping 1st-person and unknown sentences, stops at 2nd
or 3rd person boundaries.

```bash
python -m pipeline.extract_match.postprocess_filmot                    # full run (~16 min)
python -m pipeline.extract_match.postprocess_filmot --max-records 200  # quick test
```

Output: `data/pattern_candidates_filmot_pp.jsonl`
- Each record has `text_pp` (post-processed) and `text_original`
- 71% of records get trimmed (avg 5.0 → 2.7 sentences, 220 → 148 chars)
- Requires: `pip install stanza` (downloads ~200MB Romanian model on first run)

**Light: rule-based cleanup** (`postprocess_filmot_light.py`)

No ML — instant processing:
1. Remove `#tags#` (music/sound markers like `#Muzică#`)
2. Split on `>>` (speaker change markers) — keep segment with match
3. Add periods at sentence boundaries (capitalization-based, skips proper noun
   sequences and words after prepositions)
4. Remove leading fragment if first word is not capitalized (incomplete sentence
   from API context window)

```bash
python -m pipeline.extract_match.postprocess_filmot_light                    # full run (<1 sec)
python -m pipeline.extract_match.postprocess_filmot_light --max-records 200  # quick test
```

Output: `data/pattern_candidates_filmot_light.jsonl`
- Each record has `text_light` (post-processed) and `text_original`

### Output schema

All extractors share a common base schema:

```json
{
  "id": "source_123",
  "text": "...",
  "matched_sentence": "Mă simt fericit",
  "pattern_used": "ma_simt_present",
  "pattern_category": "primary",
  "seed_word": "fericit",
  "seed_word_normalized": "fericit",
  "emotion_category": ["happiness"],
  "source": "filmot|laroseda|..."
}
```

Filmot adds: `video_id`, `video_title`, `channel_name`, `youtube_url`,
`view_count`.

## Embedding Extraction (`extract_embed/`) — EXPERIMENTAL

Attempted to find ASI expressions via semantic similarity to synthetic anchor
sentences (all combinations of 20 "I feel" verb templates × 524 seed words).
Used `intfloat/multilingual-e5-base` (278M params, 768-dim) on Modal T4.

```bash
python -m pipeline.extract_embed.run --dry-run                    # test without GPU
python -m pipeline.extract_embed.run --threshold 0.75 --sample 10 # full run
python -m pipeline.extract_embed.filter_results                   # remove bad noun anchors
python -m pipeline.extract_embed.filter_results --min-confidence 0.90  # + threshold filter
```

### Pipeline

1. Load all 106K posts from `merged_corpus.jsonl`
2. Split into sentences (409K total) — no pre-filter, embed everything
3. Generate 3,912 synthetic anchors ("mă simt fericit", "sunt trist", etc.)
4. Embed anchors (`query:` prefix) + all sentences (`passage:` prefix) on Modal GPU
5. Cosine similarity: each sentence vs all anchors, keep max
6. Group hits by post, output one row per post with `hits` list

### Results & Conclusions

**The synthetic anchor approach did not work well.** Short anchors produce a very
narrow similarity band (median ~0.836) — 99.9% of sentences pass the 0.75
threshold. Only above 0.90 (342 hits) is there meaningful signal, and those
are mostly near-exact matches to what regex already catches.

See `extract_embed/ANALYSIS.md` for full analysis. Key findings:

**New patterns discovered** (to consider adding to `pattern_matcher.py`):
- `am rămas [adj]` / `rămân [adj]` — "Am rămas perplex"
- `aș fi [adj]` — "Aș fi foarte recunoscător" (conditional of "a fi")
- `să fiu [adj]` — "Să fiu mulțumită" (subjunctive of "a fi")
- `îmi pare rău` — common regret/sympathy expression
- `mă ia cu [noun]` — "Mă ia cu rău" (felt-state idiom)

**New seed words discovered**: `special`, `rupt` (colloquial "shattered")

**Recommendation**: For regex-matched corpora, pattern matching with the enriched
seed is sufficient. If embeddings are revisited, corpus-mined anchors (real
matched sentences) provide much better discrimination than synthetic ones.

### Output schema

One row per post, all qualifying sentences grouped in `hits`:

```json
{
  "id": "reddit_roap_12345",
  "text": "full post text",
  "source": "merged_corpus",
  "extraction_method": "embedding_similarity",
  "hits": [
    {
      "sentence": "Mă simt foarte trist.",
      "confidence": 0.8723,
      "emotion_category": ["sadness"],
      "nearest_anchor": "mă simt trist",
      "nearest_anchor_pattern": "ma_simt_present",
      "is_novel": true
    }
  ]
}
```

### Post-processing filter (`filter_results.py`)

Removes hits where a noun-only pattern (e.g., `imi_este`, `am_noun`) was paired
with a noun not in the curated `EMOTION_NOUNS_ONLY` set. Also supports
`--min-confidence` for threshold filtering without re-running the GPU pipeline.

## External Data (`seed/`)

### `seed/wn-affect-1.1/`

WordNet-Affect 1.1 — labels ~798 WordNet 1.6 synsets as affective.

- `a-synsets.xml` — synset-to-category mappings (280 nouns with category
  labels; adjectives/verbs/adverbs reference a noun-id)
- `a-hierarchy.xml` — category taxonomy (root → mental-state → affective-state
  → emotion → positive-emotion → joy → happiness, etc.)

Source: https://github.com/larsmans/wordnet-domains-sentiwords

### `seed/wn-mappings/`

UPC/TALP WordNet 1.6 → 3.0 offset mapping files. Each line maps one WN 1.6
offset to one or more WN 3.0 offsets with confidence scores.

Files: `wn16-30.noun`, `wn16-30.adj`, `wn16-30.verb`, `wn16-30.adv`

Source: https://github.com/getalp/UFSAC

### `seed/multext-east/`

MULTEXT-East Romanian morphological lexicon — 428K entries mapping inflected
forms to lemmas with morphosyntactic descriptors.

File: `wfl-ro.txt` (tab-separated: form, lemma, MSD tag)

Used by `utils/inflect.py` to expand lemma seeds to all gender/number/diacritics
forms for the pattern matcher.

Source: https://www.clarin.si/repository/xmlui/handle/11356/1041
