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
│   └── ...                            # Provenance files (*_provenance.json, *_discovered.json)
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

Also exports `get_trigger_words()` and `get_filmot_queries()` as single source
of truth for FULG/Filmot collection scripts.

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
