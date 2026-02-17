# Bootstrapping Methods for Romanian ASI Data Collection

## How the Current Dataset Was Obtained

The current pipeline is a **static, one-pass regex extraction** approach:

1. **Manually curated seed list** (`scripts/ro_asi/curated_affective_states.py`): 511 Romanian affective state words (348 adjectives, 150 nouns, 15 adverbs) hand-picked from RoEmoLex V3, each mapped to Plutchik emotion categories.

2. **Pattern matcher** (`scripts/ro_asi/pattern_matcher.py`): 18 regex patterns covering Romanian "I feel" constructions:
   - Primary: `mă simt [X]`, `m-am simțit [X]`, `mă simțeam [X]`, `simt [noun]`, etc.
   - Secondary: `sunt [adj]`, `eram [adj]`, `îmi este [noun]`, `mi-e [noun]`, `am [noun]`, etc.
   - Handles diacritics normalization and optional adverb modifiers (`foarte`, `mai`, `puțin`...)

3. **Three extraction pipelines** over different corpora:
   - **Small datasets** (79K records): LaRoSeDa, PoPreRo, RED v1/v2, RoSent → merged into `merged_corpus.jsonl` → regex extraction → `asi_candidates.jsonl`
   - **FULG** (150B tokens): Streaming extraction from HuggingFace with sentence-level context
   - **Filmot/YouTube** (blocked by Cloudflare)

**The key limitation**: the seed list is **fixed**. We start with 511 words and only ever find those 511 words. No new affective states are ever discovered.

---

## How MASIVE's Bootstrapping Works

MASIVE's approach (Deas et al., 2024) is fundamentally different — it **automatically discovers new affective states** through iterative corpus mining:

1. **Seed with Ekman emotions** (6 adjectives): happy, sad, angry, scared, disgusted, surprised

2. **Query with conjunction patterns**: Search Reddit for `"I feel happy and..."`, `"I am feeling happy and..."`, `"I don't feel happy and..."`

3. **Extract new terms**: From `"I feel happy and proud"`, extract `proud` as a new affective state (take adjective conjuncts after "and")

4. **Expand the seed**: Add `proud` to the query set. Now search for `"I feel proud and..."`

5. **Repeat for 4 rounds** — each round discovers new affective states from conjuncts of previously discovered ones.

This yielded **1,600 unique affective states in English** and **1,000 in Spanish**, starting from just 6 seeds. Human validation showed 88% (En) and 72% (Es) of automatically discovered terms were genuine affective states.

### Key assumption
Any adjective conjunct of a known affective state (joined by "and") is also an affective state. E.g., in "I feel happy and light", "light" is used as an affective state regardless of its typical meaning.

---

## Implementing MASIVE-Style Bootstrapping for Romanian

### Why it would help
- Our current 511 words are hand-curated. MASIVE found 1,600 from just 6 seeds. We're missing many valid affective states, especially informal/figurative ones.
- Bootstrapping discovers words **as people actually use them** — it's descriptive rather than prescriptive.
- It captures states that no lexicon lists: slang, metaphorical usage, intensity variants, context-dependent expressions.

### Romanian query templates

```
Round 0 seed: {fericit, trist, furios, speriat, dezgustat, surprins}

Query templates:
  "mă simt {seed} și ..."
  "sunt {seed} și ..."
  "m-am simțit {seed} și ..."
  "mă simțeam {seed} și ..."
  + negated versions: "nu mă simt {seed} și ..."

Extract: the adjective/word after "și" (and)

Repeat 3-4 rounds.
```

### Implementation sketch

```python
def bootstrap_round(current_seeds, corpus_iterator):
    new_seeds = set()
    samples = []

    for text in corpus_iterator:
        normalized = normalize_text(text)
        for seed in current_seeds:
            # Match "mă simt {seed} și {NEW_WORD}"
            pattern = rf'\b(?:ma simt|sunt|m-?am simtit)\s+{re.escape(seed)}\s+si\s+(\w+)\b'
            for match in re.finditer(pattern, normalized):
                candidate = match.group(1)
                # Filter: must be adjective (use spaCy/stanza POS tagger)
                if is_adjective(candidate, text):
                    new_seeds.add(candidate)
                    samples.append(extract_context(text, match))

    return new_seeds, samples
```

### Key considerations for Romanian
- **POS filtering is essential**: Use a Romanian NLP pipeline (spaCy `ro_core_news_lg` or Stanza) to verify that conjuncts are adjectives, not verbs or nouns. In Romanian, "sunt fericit și vreau..." — "vreau" is a verb, not an affective state.
- **Gendered forms**: When you discover "fericit", automatically also search for "fericită", "fericiți", "fericite".
- **Diacritics**: Match both forms (with and without), as the current pipeline already does.
- **Corpus choice**: Romanian Reddit is small. FULG (150B tokens) covering forums, blogs, social media is the best corpus for this.

---

## Other Intelligent Extraction Methods

Beyond MASIVE-style bootstrapping, here are several approaches ordered roughly by impact:

### 1. LLM-Assisted Seed Expansion (highest impact, easiest)

Instead of bootstrapping from corpus, use an LLM to expand the seed:

```python
prompt = """Given the Romanian affective state "fericit" (happy),
list 20 related affective states in Romanian that someone might
use in "mă simt [X]", ranging from similar intensity to very
different intensity. Include informal/colloquial terms."""
```

Then validate each LLM suggestion against the corpus (does it actually appear in "mă simt [X]" patterns?). This is fast, cheap, and captures long-tail states that corpus bootstrapping might miss due to sparsity.

### 2. Embedding-Based Discovery

- Take the current 511 seed words and embed them (using a Romanian language model like `dumitrescustefan/bert-base-romanian-cased-v1`)
- Find the nearest neighbors in the embedding space
- Filter candidates that appear in affective patterns in the corpus
- This discovers morphological variants and semantically similar words

### 3. Dependency Parse Extraction

Instead of regex, use dependency parsing to find affective states structurally:

```
"Mă simt [X]" → find any word that is a predicative complement of "simți" with subject "eu"
```

This handles complex cases regex misses: "Mă simt, sincer vorbind, destul de trist" or "Trist mă simt" (inverted word order).

### 4. Distributional Pattern Mining (Hearst Patterns for Emotions)

Search for explicit emotion labeling patterns:
- "o stare de [X]" (a state of X)
- "sentimentul de [X]" (the feeling of X)
- "emoție ca [X]" (emotion like X)
- "mă cuprinde [X]" (X overcomes me)

These discover emotion nouns automatically.

### 5. Cross-lingual Transfer from MASIVE

Since MASIVE already has 1,600 English and 1,000 Spanish affective states:
- Translate them to Romanian using a good MT model
- Validate each against the corpus (does it appear naturally?)
- The MASIVE paper warns that MT degrades performance (Takeaway #6), but for **seed expansion** (not training data), it's a good starting point

### 6. Active Learning / Human-in-the-Loop

- Run the extractor, then sample borderline cases (sentences matching patterns but with unknown words)
- Have a native speaker quickly label: affective state or not?
- Add validated words to seed
- Repeat

---

## Recommended Hybrid Approach

A **hybrid approach** would give the best results:

1. **LLM seed expansion** (quick win): Use Claude/GPT to generate 500+ candidate affective states, validate against FULG corpus. Probably doubles the seed list in a day.

2. **MASIVE-style bootstrapping on FULG** (medium effort, high value): The "și" (and) conjunction trick adapted for Romanian, run over FULG's 150B tokens. This is the biggest corpus and will discover the most natural expressions.

3. **Use the expanded seed for re-extraction**: Run the existing pattern matcher with the new, much larger seed list.

This would likely grow the affective state vocabulary from ~511 to 1,500+ and correspondingly increase the number and diversity of extracted candidates.

---

## References

- Deas, N., Turcan, E., Pérez Mejía, I., & McKeown, K. (2024). MASIVE: Open-Ended Affective State Identification in English and Spanish. arXiv:2407.12196v2.
