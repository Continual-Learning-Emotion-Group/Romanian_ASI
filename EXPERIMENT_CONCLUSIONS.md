# ASI Extraction Strategies — Experiment Conclusions

**Date:** 2026-02-17
**Corpus:** RedditRoAP (26,517) + PoPreRo (28,106) = 54,623 Romanian texts
**Baseline seed:** 6 Ekman emotions (happy, sad, angry, afraid, disgusted, surprised)

---

## Results at a Glance

| # | Strategy | Candidates | Unique Texts | Novel Finds | Seed Size | Status |
|---|----------|-----------|--------------|-------------|-----------|--------|
| 1 | Pattern Matching (baseline) | 234 | 219 | — | 85 (6 emotions) | Complete |
| 2 | MASIVE Bootstrapping | 428 | 391 | +4 new words | 28 (24 Ekman + 4 discovered) | Complete |
| 3 | LLM Filtering | — | — | — | — | **In Progress** |
| 4 | Embedding Similarity | 1,664 | ~1,664 | 1,257 novel | N/A (uses anchors) | Complete |
| 5 | Distributional Mining | 538 | ~500 | 392 from new words | 251 (18 + 233 discovered) | Complete |

**Combined unique candidates (strategies 1, 2, 4, 5): ~2,800+**

---

## Strategy 1: Pattern Matching (Baseline)

**Branch:** `strategy/pattern-matching`
**Output:** `data/reddit_baseline_candidates.jsonl` (234 candidates)

### What it does
Runs the existing 18-regex pipeline but restricted to only 6 Ekman basic emotions (85 word forms including gender/plural/diacritics variants), matching MASIVE's starting point.

### Key results
- **Match rate:** 0.40% (234/54,623)
- **28 of 85 seed words** actually appeared in the corpus
- **Fear dominates** (144/234 = 62%) — driven by `mi-e frică` being extremely common in Romanian
- **Top patterns:** `mi-e [noun]` (87), `sunt [adj]` (58), `îmi este [noun]` (37)

### Takeaway
With only 6 emotions, the yield is very low (0.4%). This confirms the need for seed expansion — our full 511-word curated seed gets 5,565 candidates from the same-sized corpus (5.3%), a 13x improvement. The baseline establishes the floor for comparison.

---

## Strategy 2: MASIVE-Style Bootstrapping

**Branch:** `strategy/bootstrapping`
**Output:** `data/bootstrapped_asi_candidates.jsonl` (428 candidates)

### What it does
Replicates MASIVE's core idea: find `"sunt X și Y"` patterns where X is a known emotion word, then harvest Y as a new seed. Runs 3 rounds of iterative expansion with validation filters (min length, function word exclusion, gender agreement).

### Key results
- **Seed growth:** 24 Ekman forms → 28 (+4 new words)
- **New discoveries:** mâhnit (grieving), afectată (affected), nervos (nervous), transpirat (sweaty/anxious)
- **Saturated at round 3** — no new words found
- **428 total candidates** (vs 234 baseline = 83% improvement)
- **Fear still dominates** (160/428 = 37%) but more balanced than baseline

### Round-by-round
| Round | New Words | Examples |
|-------|-----------|---------|
| 1 | 3 | mâhnit, afectată, nervos |
| 2 | 1 | transpirat (chained from nervos) |
| 3 | 0 | Saturated |

### Limitations
- **Corpus too small:** MASIVE used massive Reddit data and found 1,600 words from 6 seeds. With 54K texts, conjunction patterns are too sparse (only 13-16 matches per round).
- **Co-occurrence threshold dilemma:** Threshold=2 finds nothing; threshold=1 accepts single-evidence words.
- **Noun patterns inflate numbers:** `am [noun]` and `mi-e [noun]` use the full noun lexicon (~81 words), not just the bootstrapped adjectives.

### Takeaway
Bootstrapping works in principle — the 4 discovered words are genuinely valid ASI terms. But the corpus is too small for the method to shine. Running this on FULG (150B tokens) would likely yield hundreds of new words, much closer to MASIVE's English results.

---

## Strategy 3: LLM-Assisted Filtering

**Branch:** `strategy/llm-filtering`
**Status:** **In Progress**

### Planned approach
Take candidates that already match "I feel" patterns and use an LLM to:
- Validate whether each is a genuine affective state expression
- Filter false positives (e.g., "sunt sigur" used epistemically vs affectively)
- Optionally classify emotion category

### Expected contribution
Higher **precision** rather than higher recall. Should help clean the output of all other strategies, particularly the noisier ones (distributional mining, embedding similarity).

*Results will be added when complete.*

---

## Strategy 4: Embedding Similarity

**Branch:** `strategy/embedding-similarity`
**Output:** `data/embedding_asi_candidates.jsonl` (1,664 candidates)

### What it does
Uses `intfloat/multilingual-e5-base` embeddings (768-dim, run on Modal T4 GPU) to find sentences semantically similar to known ASI expressions. Pre-filters posts by "I feel" trigger words, embeds them, then ranks by cosine similarity to 2,203 regex-confirmed anchor sentences.

### Key results
- **1,664 total candidates** (highest yield of all strategies)
- **1,257 novel** (not found by regex) — a **63% expansion**
- **407 overlap** with regex baseline
- **Similarity range:** 0.81–0.91 (median 0.87)
- **Top novel trigger:** bare `simt` (28.5%), `îmi e` (9.8%), `mă simt` (8.3%)

### Emotion distribution (novel candidates)
| Emotion | % |
|---------|---|
| Anticipation | 44.4% |
| Trust | 19.3% |
| Fear | 16.4% |
| Sadness | 15.4% |
| Joy | 13.8% |

Note: Anticipation is inflated because many anchors are "sunt curios" (curiosity) posts that match broadly.

### Quality assessment
- **26.3%** of novel candidates contain a known emotion indicator word
- Good finds: "mă simt copleșită" (overwhelmed), "îmi e rușine" (shame), colloquial emotion expressions
- Noise sources: sensory `se simte` ("it smells/feels"), idiomatic `îmi vine` ("it comes to me"), impersonal constructions

### Takeaway
Embeddings are the **highest-yield strategy** and genuinely complementary to regex. They catch non-standard phrasings, words outside the seed list, and colloquial expressions. However, the ~74% of candidates without known emotion words need validation (this is where Strategy 3 / LLM filtering would help).

---

## Strategy 5: Distributional Pattern Mining

**Branch:** `strategy/distributional-mining`
**Output:** `data/distributional_asi_candidates.jsonl` (538 candidates)

### What it does
Discovers new emotion words by mining explicit emotion-labeling patterns in the corpus: "un sentiment de [X]", "stare de [X]", "plin de [X]", "cuprins de [X]", etc. Then uses discovered words as an expanded seed for the standard pattern matcher.

### Key results
- **235 unique words discovered** from 288 pattern matches across 8 patterns
- **Seed expanded:** 18 Ekman → 251 words
- **538 ASI candidates** extracted (73% from newly discovered words)
- **Only 15/235 (6.4%)** of discovered words were confirmed emotions — rest are noisy

### Pattern effectiveness
| Pattern | Matches | Precision | Notes |
|---------|---------|-----------|-------|
| `plin de X` | 190 | Low | Too broad — captures physical nouns (câini, mașini) |
| `stare de X` | 19 | Medium | Mixed — "stare de spirit" good, "stare de urgență" noise |
| `un sentiment de X` | 12 | High | Almost all genuine emotions |
| `sentimentul de X` | 7 | High | Almost all genuine emotions |
| `cuprins de X` | 4 | High | Good quality |
| `o senzație de X` | 3 | High | Good quality |
| `copleșit de X` | 3 | High | Good quality |

### Top discovered words (genuine)
siguranță, iubire, vină, nostalgie, afecțiune, regrete, emoții, neliniste, durere, dezamăgire

### Top noise
"persoane" (72 false hits via "sunt persoane"), "chestii", "jocuri", "urși", "autobuze", "parcuri"

### Takeaway
The **high-precision patterns** (`sentiment de`, `sentimentul de`, `cuprins de`) are excellent for discovering genuine emotion words but have very low yield on a 54K corpus. The **high-recall patterns** (`plin de`) introduce too much noise. A two-stage approach would work best: use high-precision patterns for discovery, then validate with frequency thresholds or LLM filtering. Again, FULG (150B tokens) would dramatically improve yield.

---

## Cross-Strategy Comparison

### New Affective States Discovered

| Strategy | Starting Seed | New Words Found | Confirmed ASI | Examples |
|----------|--------------|-----------------|---------------|---------|
| Pattern Matching | 85 (6 emotions) | 0 | — | *(baseline, no discovery)* |
| Bootstrapping | 24 (6 emotions) | 4 | 4 (100%) | mâhnit, afectată, nervos, transpirat |
| LLM Filtering | — | — | — | *In progress* |
| Embedding Similarity | N/A | ~1,257 novel expressions | ~330 (26%) with known emotion words | copleșită, rușine, capabilă, non-standard phrasings |
| Distributional Mining | 18 (6 emotions) | 235 candidate words | 15 (6.4%) | siguranță, iubire, vină, nostalgie, afecțiune, neliniste |

**Notes:**
- **Bootstrapping** discovers new **adjective** forms (words usable with "sunt X", "mă simt X"). All 4 are genuine, but the small corpus limits yield. MASIVE found 1,600 from 6 seeds on English Reddit.
- **Embedding similarity** doesn't expand the seed list — it finds novel **expressions** that use emotion words outside the curated 511. The 1,257 novel candidates include non-standard spellings, words not in any seed, and colloquial phrasings.
- **Distributional mining** discovers new **noun** forms (words usable with "am X", "mi-e X", "sentiment de X"). High raw count (235) but low precision — most discoveries are non-emotional nouns captured by the noisy `plin de` pattern. The 15 confirmed words come from high-precision patterns (`sentiment de`, `cuprins de`).

### Yield vs. Precision Tradeoff

```
                    High Precision
                         ^
                         |
         Baseline (234)  |  Bootstrapping (428)
         ████████████    |  ████████████████
                         |
                         |         LLM Filtering (TBD)
                         |         ██████████████████
                         |
    Distributional (538) |  Embedding (1,664)
    █████████████████    |  ██████████████████████████
                         |
                         +-------------------------->
                                              High Recall
```

| Strategy | Estimated Precision | Recall | Best For |
|----------|-------------------|--------|----------|
| Pattern Matching | ~95% | Low | Clean, high-confidence samples |
| Bootstrapping | ~85-90% | Low-Medium | Discovering new seed words |
| LLM Filtering | ~95%+ (expected) | Medium | Cleaning other strategies' output |
| Embedding Similarity | ~50-70% | High | Finding non-standard expressions |
| Distributional Mining | ~30-50% (mixed) | Medium | Discovering emotion vocabulary |

### What Each Strategy Uniquely Contributes

1. **Pattern Matching:** The clean floor — everything it finds is almost certainly an ASI expression
2. **Bootstrapping:** Discovers new adjective-form ASI words through natural co-occurrence (mâhnit, nervos)
3. **Embedding Similarity:** Catches colloquial, metaphorical, and non-standard ASI expressions that no regex could match
4. **Distributional Mining:** Discovers new emotion **nouns** through labeling patterns (sentiment de X, stare de X)

### Overlap Analysis

The strategies are largely **complementary**, not redundant:
- Baseline ⊂ Bootstrapping (bootstrapping is a superset with expanded seed)
- Embedding finds 1,257 candidates **no other strategy catches** (novel)
- Distributional mining discovers words from a completely different signal (labeling patterns vs. co-occurrence vs. similarity)

---

## Key Findings

### 1. Corpus size is the bottleneck
All strategies suffer from the same limitation: 54K texts is small. MASIVE used orders-of-magnitude more Reddit data. Both bootstrapping (saturates at round 3) and distributional mining (only 288 pattern matches) would benefit enormously from running on FULG (150B tokens).

### 2. Embeddings give the best ROI
The embedding strategy produced 5-7x more candidates than any other strategy, with reasonable quality. It's the only approach that doesn't require a predefined seed list, making it inherently more flexible.

### 3. Fear is overrepresented in Romanian informal text
Across all strategies, `mi-e frică` (I'm afraid) is disproportionately common. This is partly linguistic — `mi-e frică` is a very natural Romanian construction — and partly cultural (Reddit posts tend toward anxiety/concern topics).

### 4. High-precision emotion labeling patterns exist but are rare
Patterns like "un sentiment de X" and "cuprins de X" almost always yield genuine emotion words, but they appear very rarely in informal text. They're better suited for large-scale corpus mining.

### 5. The "sunt" pattern is a double-edged sword
`sunt [adj]` is the most productive pattern (captures "sunt fericit", "sunt trist", etc.) but also the noisiest ("sunt student", "sunt persoane"). Every strategy has to deal with this ambiguity.

---

## Recommendations

### Immediate next steps
1. **Complete LLM filtering** (Strategy 3) to validate candidates from all other strategies
2. **Run bootstrapping on FULG** — the 150B token corpus should yield hundreds of new emotion words instead of 4
3. **Run distributional mining on FULG** with only high-precision patterns (`sentiment de`, `cuprins de`)

### For the final benchmark
4. **Combine all strategies** into a union set with per-strategy confidence scores
5. **Human annotation** on a sample of ~500 candidates across strategies to get real precision numbers
6. **Use LLM filtering as a second pass** on the combined set to produce a clean final dataset

### Estimated final yield (projection)
| Source | Conservative | Optimistic |
|--------|-------------|------------|
| Small datasets (all strategies) | ~3,000 | ~3,500 |
| FULG (with expanded seeds + embeddings) | ~50,000 | ~100,000+ |
| **Total** | **~53,000** | **~103,500+** |

---

## Output Files

| Strategy | Candidates File | Stats/Analysis |
|----------|----------------|----------------|
| Pattern Matching | `data/reddit_baseline_candidates.jsonl` | `experiments/baseline_pattern_matching/README.md` |
| Bootstrapping | `data/bootstrapped_asi_candidates.jsonl` | `experiments/bootstrapping/RESULTS.md`, `data/bootstrap_provenance.json` |
| LLM Filtering | *In progress* | *In progress* |
| Embedding Similarity | `data/embedding_asi_candidates.jsonl` | `experiments/embedding_similarity/ANALYSIS.md` |
| Distributional Mining | `data/distributional_asi_candidates.jsonl` | `scripts/distributional_mining/RESULTS.md`, `data/distributional_stats.json` |
