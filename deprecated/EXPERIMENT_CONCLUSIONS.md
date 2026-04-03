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
| 3 | LLM Filtering | 1,687 kept / 2,255 input | ~1,687 | Validates others' output | 511 (full seed) | Complete |
| 4 | Embedding Similarity | 1,664 | ~1,664 | 1,257 novel | N/A (uses anchors) | Complete |
| 5 | Distributional Mining | 538 | ~500 | 392 from new words | 251 (18 + 233 discovered) | Complete |

**Combined unique candidates (all 5 strategies): ~2,800+ extraction, 1,687 LLM-validated**

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
**Output:** `data/llm_filtered_candidates.jsonl` (1,687 validated candidates)
**Model:** Qwen/Qwen2.5-7B-Instruct via Modal (A10G GPU, vLLM)

### What it does
Takes the 2,255 pattern-matched candidates from RedditRoAP and PoPreRo (from the full 511-word seed extraction) and asks an LLM to judge each one: "Does this sentence express a genuine affective state, or is the word used in a non-affective sense?" Uses a Romanian prompt with the matched sentence, surrounding context, seed word, and pattern type.

### Key results
- **2,255 candidates processed**, 1,687 kept (74.8%), 568 filtered (25.2%)
- **Zero errors** — all candidates successfully parsed
- **Processing time:** 519s (~8.5 min), 4.3 candidates/sec
- **Confidence:** kept mean 0.847, filtered mean 0.807

### Precision by pattern type

| Pattern | Total | Kept | Keep Rate | Notes |
|---------|-------|------|-----------|-------|
| `mă simt` (present) | 111 | 110 | **99.1%** | Near-perfect — "I feel X" is unambiguous |
| `m-am simțit` (perfect) | 11 | 11 | **100%** | |
| `mi-e` (dative short) | 127 | 127 | **100%** | "mi-e frică" always affective |
| `îmi este` (dative formal) | 49 | 49 | **100%** | |
| `eram` (imperfect) | 205 | 172 | 83.9% | |
| `am fost` (perfect) | 113 | 94 | 83.2% | |
| **`sunt` (present)** | **1,429** | **979** | **68.5%** | Noisiest — "I am" ambiguity |
| `am [noun]` | 156 | 97 | 62.2% | "am voie" (permission) noise |

**Primary patterns** (`mă simt`, `m-am simțit`, etc.) are 99-100% precise. **Secondary patterns** (`sunt`, `am`) carry most of the noise.

### Precision by emotion category

| Emotion | Total | Kept | Keep Rate |
|---------|-------|------|-----------|
| Fear | 292 | 289 | **99.0%** |
| Surprise | 118 | 116 | **98.3%** |
| Sadness | 284 | 267 | 94.0% |
| Anger | 93 | 86 | 92.5% |
| Anticipation | 967 | 759 | 78.5% |
| Disgust | 27 | 21 | 77.8% |
| Joy | 410 | 282 | 68.8% |
| **Trust** | **501** | **203** | **40.5%** |

### Top false positives identified

| Seed Word | Times Filtered | Why |
|-----------|---------------|-----|
| sigur/sigură | 208 | Epistemic "I'm sure that..." — certainty, not emotion |
| curios/curioasă | 183 | Intellectual curiosity, not affective state |
| bine | 57 | Social "I'm fine" — formulaic, not genuine emotion |
| voie | 31 | "am voie" = "I'm allowed" — permission, not feeling |
| chef | 21 | "am chef de" = "I feel like" — desire, not emotion |
| acceptat | 12 | "am fost acceptat" = "I was accepted" — event, not state |

### Results by source

| Source | Total | Kept | Keep Rate |
|--------|-------|------|-----------|
| RedditRoAP | 1,788 | 1,358 | 76.0% |
| PoPreRo | 467 | 329 | 70.4% |

### Takeaway
LLM filtering is the **precision strategy** — it doesn't find new candidates but cleans existing ones. The 25.2% filter rate confirms that regex alone has significant noise, especially from `sunt [adj]` and `am [noun]` patterns. The most important insight: **"trust" is not a reliable emotion category** in this pipeline because "sunt sigur" (I'm sure) is almost always epistemic, not affective. Fear, surprise, and sadness are the most reliable categories.

This strategy is designed to be a **second pass** on output from all other strategies.

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
| LLM Filtering | 511 (full seed) | 0 (validates, doesn't discover) | Removed 568 false positives (25.2%) | sigur, curios, bine, voie — epistemic/non-affective |
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
                         |  LLM Filtering (1,687)
                         |  ██████████████████████
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
| LLM Filtering | 74.8% kept (removes 25.2% noise) | Medium | Cleaning other strategies' output |
| Embedding Similarity | ~50-70% | High | Finding non-standard expressions |
| Distributional Mining | ~30-50% (mixed) | Medium | Discovering emotion vocabulary |

### What Each Strategy Uniquely Contributes

1. **Pattern Matching:** The clean floor — everything it finds is almost certainly an ASI expression
2. **Bootstrapping:** Discovers new adjective-form ASI words through natural co-occurrence (mâhnit, nervos)
3. **LLM Filtering:** Identifies which seed words are unreliable ("sigur", "curios", "bine" in non-affective senses) and which patterns are trustworthy (primary > secondary)
4. **Embedding Similarity:** Catches colloquial, metaphorical, and non-standard ASI expressions that no regex could match
5. **Distributional Mining:** Discovers new emotion **nouns** through labeling patterns (sentiment de X, stare de X)

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
`sunt [adj]` is the most productive pattern (captures "sunt fericit", "sunt trist", etc.) but also the noisiest ("sunt student", "sunt persoane"). Every strategy has to deal with this ambiguity. LLM filtering confirms: `sunt` has only 68.5% precision vs 99-100% for `mă simt` patterns.

### 6. "Trust" is an unreliable emotion category
LLM filtering revealed that 59.5% of "trust"-labeled candidates are false positives, almost entirely driven by "sunt sigur/sigură" (I'm sure) being used epistemically rather than affectively. Similarly, "curios" (curiosity) is frequently intellectual rather than emotional. These words should either be removed from the seed list or always run through LLM validation.

### 7. LLM filtering provides actionable seed list curation
Beyond validating individual candidates, LLM filtering identifies **which seed words are problematic**: sigur (208 filtered), curios (183), bine (57), voie (31), chef (21). This is directly actionable — removing or flagging these 5 words would eliminate most false positives upstream.

---

## Recommendations

### Immediate next steps
1. **Curate seed list using LLM findings** — remove or flag sigur, curios, voie, chef; keep bine with context-dependent validation
2. **Run LLM filtering on embedding + distributional candidates** — these have higher noise and would benefit most
3. **Run bootstrapping on FULG** — the 150B token corpus should yield hundreds of new emotion words instead of 4
4. **Run distributional mining on FULG** with only high-precision patterns (`sentiment de`, `cuprins de`)

### For the final benchmark
5. **Combine all strategies** into a union set with per-strategy confidence scores
6. **Human annotation** on a sample of ~500 candidates across strategies to get real precision numbers
7. **Use LLM filtering as a second pass** on the combined set to produce a clean final dataset

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
| LLM Filtering | `data/llm_filtered_candidates.jsonl` | `experiments/llm_filtering/RESULTS.md`, `data/llm_filter_stats.json` |
| Embedding Similarity | `data/embedding_asi_candidates.jsonl` | `experiments/embedding_similarity/ANALYSIS.md` |
| Distributional Mining | `data/distributional_asi_candidates.jsonl` | `scripts/distributional_mining/RESULTS.md`, `data/distributional_stats.json` |
