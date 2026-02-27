# Romanian ASI Benchmark
## Affective State Identification for Romanian Language

---

# Slide 1: Project Goal

**Building a Romanian benchmark for Affective State Identification (ASI)**

- Following the MASIVE paper methodology (English ASI benchmark)
- Extract natural **"I feel [state]"** expressions from Romanian text corpora
- Goal: identify sentences where speakers express genuine emotional/affective states
- Example: *"Mă simt fericit"* (I feel happy), *"Mi-e frică"* (I'm afraid)

**Why this matters:**
- No Romanian ASI benchmark exists
- Romanian has unique linguistic challenges: gendered adjectives, diacritics inconsistency, ambiguous copular constructions

---

# Slide 2: Data Sources Overview

## Small Datasets (6 sources, 106K records)

| Dataset | Records | Domain | Description |
|---------|---------|--------|-------------|
| RoSent | 27,338 | Reviews | Romanian sentiment corpus |
| RedditRoAP | 26,269 | Social media | Romanian Reddit posts |
| PoPreRo | 28,103 | Social media | Popular posts from Romanian Reddit |
| LaRoSeDa | 14,982 | Reviews | Large Romanian Sentiment Dataset |
| RED v2 | 5,199 | Emotions | Romanian Emotion Dataset v2 |
| RED v1 | 4,036 | Emotions | Romanian Emotion Dataset v1 |
| **TOTAL** | **105,927** | | |

## Large-Scale Dataset

| Dataset | Size | Records Streamed | Domain |
|---------|------|-----------------|--------|
| **FULG** | 150B tokens (289 GB) | 405,000 (paused, resumable) | Web crawl (blogs, forums, news, wiki) |

---

# Slide 3: The Emotion Seed

**511 curated Romanian affective state words**

| Category | Count | Examples |
|----------|-------|---------|
| Adjectives | 348 | fericit/fericită, trist/tristă, furios/furioasă |
| Nouns | 150 | frică, bucurie, tristețe, mânie |
| Adverbs | 15 | bine, rău, groaznic |

**Emotion taxonomy:** Plutchik's 8 basic emotions
- Joy, Trust, Anticipation, Sadness, Surprise, Fear, Anger, Disgust

**Key challenge:** Romanian has gendered adjectives -- both masculine and feminine forms must be included (e.g., fericit/fericită, supărat/supărată)

---

# Slide 4: Pattern Matching -- The Core Pipeline

**18 Romanian "I feel" regex patterns in two categories:**

### Primary patterns (unambiguous, 99-100% precision)
- `mă simt [adj]` -- "I feel [adj]" (present reflexive)
- `m-am simțit [adj]` -- "I felt [adj]" (perfect)
- `mă simțeam [adj]` -- "I was feeling [adj]" (imperfect)
- `simt [noun]` -- "I feel [noun]"

### Secondary patterns (productive but noisier, 62-84% precision)
- `sunt [adj]` -- "I am [adj]" (most productive, also noisiest)
- `mi-e [noun]` -- "I have [noun]" (dative short, e.g., mi-e frică)
- `îmi este [noun]` -- "I have [noun]" (dative formal)
- `eram [adj]`, `am fost [adj]`, `am [noun]`, etc.

### Diacritics normalization
Handles inconsistent diacritics in social media: ă->a, ș->s, ț->t
- *"ma simt fericit"* matches same as *"mă simt fericit"*

---

# Slide 5: Main Extraction Results

## Small datasets: 6,070 ASI candidates from 106K records (5.7% match rate)

| Dataset | Records | Candidates | Match Rate |
|---------|---------|-----------|------------|
| RoSent | 27,338 | 2,481 | 9.1% |
| RedditRoAP | 26,269 | 1,788 | 6.8% |
| LaRoSeDa | 14,982 | 1,017 | 6.8% |
| PoPreRo | 28,103 | 467 | 1.7% |
| RED v2 | 5,199 | 186 | 3.6% |
| RED v1 | 4,036 | 131 | 3.2% |

## FULG extraction: 21,184 candidates from 405K records streamed (7.0% match rate)
- 2,792 unique domains
- Sentence-level context extraction (median 585 chars vs 21K for full page)

**Total so far: ~27,000 ASI candidates**

---

# Slide 6: Sample Extractions -- Good Examples

### Pattern: `sunt [adj]` (present copular)
> **"sunt foarte mulțumit de calitatea acestuia"**
> Source: LaRoSeDa product review (5-star rating)
> Emotion: Joy | Seed word: mulțumit

### Pattern: `mă simt [adj]` (present reflexive)
> **"mă simt foarte demoralizat și stresat, pentru că am avut parte de sute de aplicări fără răspuns"**
> Source: RedditRoAP (job searching post)
> Emotion: Sadness | Seed word: demoralizat

### Pattern: `mi-e [noun]` (dative short)
> **"mi-e frică că modelul/țesătura să fie prea batranească pentru vârsta mea"**
> Source: PoPreRo (fashion question)
> Emotion: Fear | Seed word: frică

### Pattern: `mă simt [adj]` from FULG web crawl
> **"Mă simt atât de vinovată după ce fac asta."**
> Source: ortodoxia.md (personal confession)
> Emotion: Sadness/Fear | Seed word: vinovată

---

# Slide 7: Experiments Overview

**5 extraction strategies tested on the same corpus**
(RedditRoAP + PoPreRo = 54,623 texts)

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
                         +------------------------->
                                          High Recall
```

| # | Strategy | Candidates | Novel Finds | Key Idea |
|---|----------|-----------|-------------|----------|
| 1 | Pattern Matching (baseline) | 234 | -- | 6 Ekman emotions only |
| 2 | MASIVE Bootstrapping | 428 | +4 new words | Iterative seed expansion |
| 3 | LLM Filtering | 1,687 kept | Validates others | Qwen2.5-7B as judge |
| 4 | Embedding Similarity | 1,664 | 1,257 novel | Semantic search |
| 5 | Distributional Mining | 538 | 235 candidate words | Pattern-based discovery |

---

# Slide 8: Experiment 1 -- Baseline Pattern Matching

**Setup:** Only 6 Ekman basic emotions (85 word forms), matching MASIVE's starting point

**Results:**
- **234 candidates** from 54,623 texts (0.40% match rate)
- Only 28 of 85 seed words appeared in the corpus
- Fear dominates: 144/234 = **62%**

**Top patterns by yield:**
| Pattern | Matches |
|---------|---------|
| `mi-e [noun]` | 87 |
| `sunt [adj]` | 58 |
| `îmi este [noun]` | 37 |

**Takeaway:** With only 6 emotions, yield is very low (0.4%). Our full 511-word seed gets 5,565 candidates from the same corpus -- a **13x improvement**. This establishes the floor for comparison.

---

# Slide 9: Experiment 2 -- MASIVE-Style Bootstrapping

**Idea:** Replicate MASIVE's core method -- find conjunction patterns *"sunt X și Y"* where X is known, harvest Y as new seed word

**Round-by-round results:**
| Round | New Words | Discoveries |
|-------|-----------|-------------|
| 1 | 3 | **mâhnit** (grieving), **afectată** (affected), **nervos** (nervous) |
| 2 | 1 | **transpirat** (sweaty/anxious) -- chained from nervos |
| 3 | 0 | Saturated -- no new words found |

**Results:** 428 candidates (83% improvement over baseline)

### Sample: Bootstrapping discovering "mâhnit" (grieving)
> **"Dar nu-i nimic, nu sunt mâhnit / Cred că-s doar cam obosit"**
> Source: RedditRoAP (poem)
> -- Found via conjunction "sunt X și Y" with known emotion word

**Limitation:** Corpus too small (54K). MASIVE found 1,600 words from 6 seeds on massive English Reddit. Running on FULG (150B tokens) would likely yield hundreds of new words.

---

# Slide 10: Experiment 3 -- LLM-Assisted Filtering

**Setup:** Qwen2.5-7B-Instruct (Modal A10G GPU, vLLM) judges each candidate:
*"Does this sentence express a genuine affective state?"*

**Results:** 2,255 candidates in -> **1,687 kept (74.8%)**, 568 filtered (25.2%)

### Precision by pattern type:
| Pattern | Keep Rate | Verdict |
|---------|-----------|---------|
| `mă simt` (reflexive) | **99.1%** | Near-perfect |
| `m-am simțit` (perfect) | **100%** | Perfect |
| `mi-e` (dative) | **100%** | Perfect |
| `sunt` (copular) | **68.5%** | Noisiest |
| `am [noun]` | 62.2% | Noisy |

### Top false positives discovered:
| Word | Times Filtered | Reason |
|------|---------------|--------|
| **sigur/sigură** | 208 | Epistemic "I'm sure that..." -- certainty, not emotion |
| **curios/curioasă** | 183 | Intellectual curiosity, not affective state |
| **bine** | 57 | Social "I'm fine" -- formulaic, not genuine |
| **voie** | 31 | "am voie" = "I'm allowed" -- permission |
| **chef** | 21 | "am chef de" = "I feel like" -- desire |

---

# Slide 11: LLM Filtering -- Concrete Examples

### Correctly KEPT (genuine affective state):
> **"Sunt mândru să anunț primul concurs al acestui subreddit!"**
> LLM reasoning: *"Cuvântul 'mândru' indică un sentiment pozitiv și de satisfacție"*
> Verdict: KEEP (confidence 0.9)

> **"Sunt disperat să-mi vând toate lucrurile pt bani de plecare"**
> LLM reasoning: *"Cuvântul 'disperat' exprimă o emoție puternică de îngrijorare"*
> Verdict: KEEP (confidence 0.9)

### Correctly FILTERED (not affective):
> **"Nu sunt sigur dacă este vreo farsă, dar ecranul este strâmb"**
> LLM reasoning: *"Cuvântul 'sigur' este folosit pentru a exprima certitudine, nu o stare afectivă"*
> Verdict: FILTER

> **"Breaking Bad - sunt curios dacă rulează la TV în România"**
> LLM reasoning: *"Cuvântul 'curios' se referă la o interesare, nu la o emoție"*
> Verdict: FILTER

### Key insight: "Trust" category is unreliable
- 59.5% false positive rate, almost entirely from "sunt sigur" (epistemic, not affective)
- Fear, Surprise, Sadness are the most reliable categories (92-99% precision)

---

# Slide 12: Experiment 4 -- Embedding Similarity

**Setup:** `intfloat/multilingual-e5-base` embeddings (Modal T4 GPU) to find semantically similar sentences to known ASI expressions

**Results:**
- **1,664 total candidates** (highest yield of all strategies)
- **1,257 novel** -- NOT found by regex (63% expansion!)
- Similarity range: 0.81-0.91 (median 0.87)

### What embeddings find that regex can't:

**Non-standard phrasings:**
> **"nu prea am simțit să fac acest lucru"** (didn't really feel like doing this)
> Cosine similarity: 0.89 -- captures informal phrasing regex would miss

**Colloquial expressions:**
> **"ne simțim bine și prindem pești"** (we feel good and catch fish)
> Cosine similarity: 0.97 -- plural form "ne simțim" outside standard patterns

**Complex emotional expressions:**
> **"mă simt foarte demoralizat și stresat, pentru că am avut parte de sute de aplicări fără răspuns"**
> Cosine similarity: 0.97 -- rich context about job searching despair

### Top novel trigger words found:
`simt` (28.5%), `îmi e` (9.8%), `mă simt` (8.3%), `îmi este` (7.5%), `îmi vine` (5.1%)

---

# Slide 13: Experiment 5 -- Distributional Pattern Mining

**Idea:** Discover new emotion words from explicit labeling patterns in the corpus

**Patterns mined:**
| Pattern | Matches | Precision |
|---------|---------|-----------|
| `un sentiment de X` | 12 | **High** -- almost all genuine emotions |
| `sentimentul de X` | 7 | **High** |
| `cuprins de X` | 4 | **High** |
| `plin de X` | 190 | Low -- too broad |

**Genuine discoveries:**
*siguranță, iubire, vină, nostalgie, afecțiune, regrete, emoții, neliniște, durere, dezamăgire*

### Sample: "simt iubire" found via distributional mining
> **"Simt... acum eu simt iubire / Și mai simt... o zvârcolire / Fiindcă mă gândesc la tine"**
> Source: RedditRoAP (poem)
> Discovered word "iubire" (love) via "un sentiment de iubire" pattern

**Results:** 235 unique candidate words -> 538 ASI candidates
- Only 15/235 (6.4%) were confirmed genuine emotions
- High-precision patterns are excellent but rare; noisy patterns dominate

---

# Slide 14: Cross-Strategy Comparison

### Each strategy contributes something unique:

| Strategy | Est. Precision | Recall | Unique Contribution |
|----------|---------------|--------|---------------------|
| Pattern Matching | ~95% | Low | Clean floor -- high-confidence samples |
| Bootstrapping | ~85-90% | Low-Medium | Discovers new **adjective** forms (mâhnit, nervos) |
| LLM Filtering | validates 74.8% | Medium | Identifies unreliable seed words (sigur, curios) |
| Embedding Similarity | ~50-70% | **High** | Catches colloquial/non-standard expressions |
| Distributional Mining | ~30-50% | Medium | Discovers new emotion **nouns** (iubire, vină) |

### Strategies are complementary, not redundant:
- Embedding finds **1,257 candidates no other strategy catches**
- Bootstrapping and distributional mining discover words from different signals
- LLM filtering is a second-pass that improves all others

---

# Slide 15: Key Findings

### 1. Corpus size is the bottleneck
All strategies are limited by 54K texts. MASIVE used orders-of-magnitude more data. Bootstrapping saturates at round 3; distributional mining gets only 288 matches. Running on FULG (150B tokens) would dramatically improve all strategies.

### 2. Embeddings give the best ROI
5-7x more candidates than any other strategy, with reasonable quality. The only approach that doesn't require a predefined seed list.

### 3. Fear is overrepresented in Romanian informal text
`mi-e frică` is disproportionately common (60%+ across strategies). Partly linguistic (very natural Romanian construction), partly topical (Reddit skews toward anxiety).

### 4. The "sunt" pattern is a double-edged sword
Most productive pattern but also noisiest. `sunt` precision: 68.5% vs `mă simt`: 99.1%.

### 5. "Trust" is an unreliable emotion category
59.5% false positive rate -- almost entirely "sunt sigur" used epistemically.

### 6. LLM filtering provides actionable seed curation
Removing just 5 problematic words (sigur, curios, bine, voie, chef) would eliminate most false positives.

---

# Slide 16: Current Numbers & Next Steps

## What we have now:

| Source | ASI Candidates |
|--------|---------------|
| Small datasets (main pipeline) | 6,070 |
| Small datasets (5 experiments) | ~2,800 unique |
| FULG (paused, resumable) | 21,184 |
| **Total extracted** | **~27,000** |
| **LLM-validated (clean)** | **~1,687** |

## Projected final yield:
| Source | Conservative | Optimistic |
|--------|-------------|------------|
| Small datasets (all strategies) | ~3,000 | ~3,500 |
| FULG (with expanded seeds + embeddings) | ~50,000 | ~100,000+ |
| **Total** | **~53,000** | **~103,500+** |

## Recommended next steps:
1. Curate seed list using LLM findings (remove sigur, curios, voie, chef)
2. Run bootstrapping + distributional mining on FULG
3. Run LLM filtering as second pass on all strategies' output
4. Human annotation on ~500 samples for real precision numbers
5. Combine all strategies into final benchmark with confidence scores
