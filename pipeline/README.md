# Pipeline

Unified pipeline for constructing the Romanian ASI benchmark.

```
pipeline/
├── README.md
├── data/                              # Pipeline outputs
│   └── merged_corpus.jsonl            # Unified corpus (106K records)
├── seed/                              # Seed construction
│   ├── bridge.py                      # Bridge-only seed (229 words)
│   ├── merged.py                      # Bridge + old seed merged (375 words)
│   ├── test_wn_affect_bridge.py       # Bridge script (RoEmoLex × WN-Affect)
│   ├── wn_affect_bridge_results.json  # Raw bridge output (398 words)
│   ├── wn-affect-1.1/                 # WordNet-Affect data
│   ├── wn-mappings/                   # WN 1.6→3.0 offset mappings
│   └── multext-east/                  # Romanian morphological lexicon
└── collect/                           # Data collection
    ├── merge_small.py                 # Merge 6 small datasets
    └── small_datasets/                # Raw source data
        ├── LaRoSeDa/                  # Product reviews (sentiment)
        ├── PoPreRo/                   # News popularity prediction
        ├── RED/                       # Emotion tweets (v1 + v2)
        ├── RoSent/                    # Sentiment analysis
        └── RedditRoAP/               # Romanian Reddit posts
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

Used for: resolving inflected forms to lemmas, comparing seed lists at the
lemma level, and (future) auto-expanding lemma seeds to all gender/number
forms for the pattern matcher.

Source: https://www.clarin.si/repository/xmlui/handle/11356/1041
