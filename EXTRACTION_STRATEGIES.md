# ASI Extraction Strategies

This document describes 5 parallel extraction strategies being developed on separate branches. Each strategy targets Romanian Reddit data (RedditRoAP + PoPreRo) and can also be applied to the full merged corpus (105,927 records) and FULG (150B tokens), but this will be done later.

All strategies output to a common schema (see below) so results can be compared and combined.

---

## Strategies Overview

| # | Strategy | Branch | Description |
|---|----------|--------|-------------|
| 1 | Pattern Matching | `strategy/pattern-matching` | Existing 18-regex pipeline (baseline) |
| 2 | MASIVE Bootstrapping | `strategy/bootstrapping` | "X and Y" co-occurrence expansion with filtering |
| 3 | LLM Filtering | `strategy/llm-filtering` | LLM-assisted validation of pattern-matched candidates |
| 4 | Embedding Similarity | `strategy/embedding-similarity` | Semantic similarity to known ASI expressions |
| 5 | Distributional Mining | `strategy/distributional-mining` | Discover new emotion patterns from corpus statistics |

---

## Strategy 1: Pattern Matching (Baseline)

**Branch:** `strategy/pattern-matching`

The existing pipeline. 18 Romanian "I feel" regex patterns + 511 curated seed words. Already extracts 1,788 candidates from RedditRoAP (6.8% match rate).

**Goal:** Establish baseline numbers on Reddit data specifically. No new code needed — just run existing extraction and report Reddit-specific statistics.

**Script:** `scripts/ro_asi/extract_candidates.py`

---

## Strategy 2: MASIVE-Style Bootstrapping

**Branch:** `strategy/bootstrapping`

Follow the MASIVE paper's bootstrapping approach: find "I feel X and Y" patterns where X is a known ASI word, then harvest Y as a new candidate ASI word.

**Key challenge:** Not everything after "and" is an ASI. "Sunt fericit și muncitor" — "muncitor" (hardworking) is not an affective state. Need filtering:
- Y must not be in a stoplist of non-affective adjectives
- Y should appear in ASI patterns independently elsewhere in the corpus
- Optionally: Y should have semantic similarity to known ASI words (embeddings)
- Frequency thresholds: Y must co-occur with multiple known ASI words

**Patterns to mine:**
- `sunt X și Y` / `sunt X dar Y`
- `mă simt X și Y`
- `eram X și Y`
- `am fost X și Y`

**Output:** Expanded seed word list + new candidates found using expanded seeds.

**Script:** `scripts/reddit/bootstrap_candidates.py`

---

## Strategy 3: LLM-Assisted Filtering

**Branch:** `strategy/llm-filtering`

Take candidates that already match "I feel" patterns (from Strategy 1) and use an LLM to validate whether they are genuine affective state expressions.

**Approach:**
- Feed each matched sentence + surrounding context to an LLM
- Ask: "Does this sentence express the speaker's emotional/affective state?"
- Filter out false positives (e.g., "sunt sigur" = "I am sure" used epistemically, not affectively)
- Optionally: ask LLM to classify the emotion category

**Benefits:** Higher precision than pure regex. Can catch subtle disambiguation (e.g., "sunt rău" = "I am bad/evil" vs "I feel bad").

**Script:** `scripts/reddit/llm_filter_candidates.py`

---

## Strategy 4: Embedding Similarity

**Branch:** `strategy/embedding-similarity`

Use sentence embeddings to find texts that are semantically similar to known ASI expressions, even if they don't match any regex pattern.

**Approach:**
- Embed known good ASI sentences (from pattern matching with high confidence)
- Embed all Reddit sentences
- Find nearest neighbors that express affective states but use non-standard phrasing
- Cluster to discover natural groupings of emotional expressions

**Benefits:** Catches expressions the regex misses entirely — metaphors, colloquial speech, code-switching.

**Script:** `scripts/reddit/embedding_candidates.py`

---

## Strategy 5: Distributional Pattern Mining

**Branch:** `strategy/distributional-mining`

Instead of hand-crafted patterns, discover new emotion-expression patterns from corpus statistics.

**Approach:**
- Search for explicit labeling patterns: "un sentiment de [X]", "o stare de [X]", "emoția de [X]"
- Mine frequent n-grams around known emotion words to discover new frame patterns
- Use PMI (pointwise mutual information) to find words that strongly co-occur with emotion contexts
- Extract candidates from newly discovered patterns

**Benefits:** Finds patterns we didn't think to write regexes for. Data-driven discovery.

**Script:** `scripts/reddit/distributional_candidates.py`

---

## Common Output Schema

All strategies should output JSONL with this schema:

```json
{
  "id": "reddit_roap_12345",
  "text": "Full original post text...",
  "matched_sentence": "Mă simt fericit",
  "extraction_strategy": "pattern_matching|bootstrapping|llm_filtering|embedding_similarity|distributional_mining",
  "confidence": 0.95,
  "seed_word": "fericit",
  "emotion_category": ["joy"],
  "source": "reddit_roap",
  "metadata": {}
}
```

**Fields:**
- `extraction_strategy`: Which strategy found this candidate
- `confidence`: Strategy-specific confidence score (0-1)
- `metadata`: Strategy-specific extra info (e.g., LLM reasoning, cosine similarity, bootstrap chain)

---

## How to Work on a Strategy

```bash
# Each strategy lives in a git worktree
cd ../Romanian_ASI_<strategy_name>

# Activate environment
source ../Romanian_ASI/venv/bin/activate  # or set up local venv

# Shared code is on main — merge if needed
git merge main

# When done, merge back
cd ../Romanian_ASI
git merge strategy/<branch-name>
```

---

## Comparison Plan

After all strategies produce results, compare:
1. **Yield:** How many unique candidates does each find?
2. **Precision:** Sample 100 from each, manually check — what % are true ASI?
3. **Overlap:** Venn diagram of candidates found by each strategy
4. **Novel finds:** What does each strategy find that others miss?
