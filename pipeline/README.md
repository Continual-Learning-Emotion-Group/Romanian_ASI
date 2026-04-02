# Pipeline

Unified pipeline for constructing the Romanian ASI benchmark.

```
pipeline/
├── README.md
├── data/                              # Pipeline outputs
│   ├── merged_corpus.jsonl            # Unified corpus (106K records)
│   └── enriched_seed.json             # Enriched seed (after seed enrichment)
├── utils/                             # Shared utilities
│   ├── text_utils.py                  # Diacritics normalization, sentence splitting
│   ├── inflect.py                     # Lemma → inflected forms (MULTEXT-East)
│   ├── pattern_matcher.py             # 23 "I feel" patterns, PatternMatcher
│   ├── corpus_reader.py               # Unified JSONL reader
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
│   ├── stream_filmot.py               # Stream from Filmot API
│   └── small_datasets/                # Raw source data
└── seed_enrichment/                   # Seed enrichment
    ├── run.py                         # CLI: runs both methods, merges
    ├── bootstrapping.py               # MASIVE-style "I feel X and Y"
    ├── distributional.py              # "un sentiment de X" discovery
    └── merge_results.py               # Combine outputs → enriched seed
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
Romanian "I feel" trigger phrases in YouTube subtitles. No pattern matching —
saves raw subtitle context for downstream filtering.

Requires: `pip install filmot python-dotenv` and `RAPIDAPI_KEY` in `.env`.

Run with `python -m pipeline.collect.stream_filmot --max-hits 50000`.

## Shared Utilities (`utils/`)

### Pattern Matcher (`utils/pattern_matcher.py`)

23 Romanian "I feel" regex patterns (18 original + 5 new colloquial/conditional/
subjunctive forms). Auto-expands lemma seeds to all gender/number/diacritics
forms via MULTEXT-East.

New patterns added: `o să mă simt` (colloquial future), `o să fiu` (colloquial
future of "to be"), `m-aș simți` (conditional), `să mă simt` (subjunctive),
`mă fac` (reflexive "I become").

Also exports `get_trigger_words()` and `get_filmot_queries()` as single source
of truth for FULG/Filmot collection scripts.

### Inflection (`utils/inflect.py`)

Expands lemmas to all inflected forms using MULTEXT-East (428K entries).
E.g., `fericit` → `{fericit, fericită, fericite, fericiți, fericita, ...}`.

### Corpus Reader (`utils/corpus_reader.py`)

Unified JSONL reader: `iter_corpus(data_dir)` yields `(id, text, source)` from
all `*.jsonl` files in `pipeline/data/`. Handles different text field names
(`text`, `full_context`). Optional trigger word pre-filter.

## Seed Enrichment (`seed_enrichment/`)

Discovers new seed words from text data. Reads from all JSONL files in
`pipeline/data/` (merged small datasets + optional FULG/Filmot dumps).

Run with `python -m pipeline.seed_enrichment.run`.

### Method 1: Bootstrapping (`bootstrapping.py`)

MASIVE-style conjunction mining: finds "I feel X and Y" patterns where X is a
known seed word and Y is a candidate. Iterative (4 rounds by default). Starts
from the 375-word merged seed. Validates candidates by co-occurrence threshold,
gender agreement, and stopword filtering.

### Method 2: Distributional Mining (`distributional.py`)

Discovers emotion words via explicit labeling patterns (no seed needed):
"un sentiment de X", "o stare de X", "plin de X", "cuprins de X", etc.
Primarily finds nouns.

### Output

Both methods' results are merged and deduplicated into `data/enriched_seed.json`,
loadable via `pipeline.seed.enriched.build_enriched_seed()`.

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
