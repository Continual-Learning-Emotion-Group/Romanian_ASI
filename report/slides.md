# Romanian ASI Benchmark — Slides

---

## Slide 1 — Project Overview

**Goal:** Build the first Romanian **Affective State Identification (ASI)** benchmark, following the MASIVE methodology — extract natural *"I feel [state]"* expressions from Romanian text, then evaluate how well existing models can predict the masked emotion word.

**Pipeline at a glance:**

| Stage | Input | Output | Size |
|-------|-------|--------|------|
| Seed construction | RoEmoLex × WN-Affect bridge + curated merge | Base lemma seed | 375 words |
| Seed enrichment | Bootstrapping + distributional mining on 3 corpora | Enriched seed | **524 words** |
| Pattern extraction | 20 "I feel" regex patterns × 3 corpora | Unified candidates | 129,700 |
| LLM validation | Qwen 3.5-9B, MASIVE-style 0–3 Likert | Validated candidates | 129,700 |
| Human evaluation | 2 Romanian native annotators (MTurk) | Agreement + threshold | n=105 |
| Benchmark build | Keep candidates with LLM score = 3 | **Final benchmark** | **73,427** |
| Zero-shot eval | MLMs, mT5, Qwen, RoGemma, RoLlama, XLM-R | Acc@k, MRR | 2,658 test |

**Key design choices:**
- Romanian-specific: gendered adjectives (*fericit / fericită*), inconsistent diacritics, social-media noise
- Precision-over-recall: benchmark uses **LLM ≥ 3 only** (91.3% human precision)
- Three-source collection for genre diversity: reviews/tweets + YouTube subtitles + web text

---

## Slide 2 — Mining Affective State Labels (methods explored)

We compared four families of methods for discovering ASI expressions. Each has different strengths in recall, precision, and grammatical control.

| Method | How it works | What it's good for | What we observed |
|--------|--------------|--------------------|------------------|
| **Template / pattern matching** (primary approach) | 20 hand-written Romanian regex patterns for 1st-person singular: *mă simt X, sunt X, am fost X, mi-e X, eram X, o să mă simt X, …*. Seed of 524 lemmas auto-expanded to masc/fem + diacritic variants via MULTEXT-East. | High precision, interpretable, controls for gender and verb person. Dominant source of benchmark candidates. | 129K candidates across 3 corpora (3.6–8.9% hit rate). `mă simt` patterns hit 91% LLM acceptance, `sunt` is noisiest (77%). |
| **Bootstrapping (conjunction mining)** | MASIVE-style: find *"mă simt X **și** Y"* where X is a known seed and Y is a candidate new seed. Iterative rounds, gender-agreement and stopword filters. | In theory grows the seed from data for free. | **Failed in practice for Romanian.** On 500K FULG records only 10 accepted words, all garbage (conjunctions, verbs like *încerc*, *iar*, *deși*). Conjunct slots rarely hold affective states. |
| **Distributional mining** | Explicit labelling templates independent of seed: *"un sentiment de X", "sentimentul de X", "o senzație de X", "copleșit de X", "emoție de X"*. Candidates extracted, then manually filtered. | Finds **affective nouns** that no seed lexicon contained. | Productive: 570 FULG raw → 135 new nouns; +3 from Filmot, +2 from small datasets. Took seed from 375 → 524. |
| **Embedding similarity (experimental)** | Synthetic anchors (20 verb templates × 524 seeds = 3,912 sentences) vs. 409K corpus sentences, cosine with `multilingual-e5-base` on GPU. | Should catch paraphrases the regex misses. | Poor discrimination: median similarity 0.836, 99.9% of sentences pass 0.75. Only > 0.90 was signal, and those overlapped with regex hits. |
| **LLM validation** (used as verifier, not extractor) | Qwen 3.5-9B with MASIVE verification prompt (0–3 scale, 7 few-shot examples). Does not generate candidates; scores them. | Cheap, high-recall filter that mimics a human annotator. | See Slide 4. |

**Takeaway:** the extraction-time winner was **template matching with an enriched seed**; bootstrapping underperformed for Romanian, distributional mining was useful for growing the seed, and embedding similarity with synthetic anchors did not beat regex.

---

## Slide 3 — Romanian Data Collection

Three complementary corpora were streamed and filtered into a common schema, then deduplicated by MD5 text hash.

| Corpus | Genre | Raw size | Collection method | Yielded candidates |
|--------|-------|---------:|-------------------|-------------------:|
| **Small datasets (merged_corpus)** | Reviews, tweets, Reddit, news — 6 public Romanian NLP datasets (LaRoSeDa, PoPreRo, RED v1/v2, RoSent, RedditRoAP) | 106K records | Local merge + dedup (`collect/merge_small.py`) | 4,905 |
| **Filmot (YouTube subtitles)** | Spoken Romanian auto-captions from YouTube, retrieved via Filmot RapidAPI with 27 trigger-phrase queries | 331K subtitle hits scanned | API streaming with parallel workers + checkpoint/resume | 28,314 |
| **FULG (web text)** | 150B-token, 289 GB Romanian Common-Crawl–derived corpus | 2.8M records scanned (streamed) | HuggingFace streaming + trigger-word pre-filter | 96,481 |

**Examples of extracted candidates (one per source):**

| Source | `matched_sentence` | Pattern | Seed | Emotion |
|--------|--------------------|---------|------|---------|
| merged_corpus (LaRoSeDa) | *"sunt multumit."* | `sunt_adj_present` | mulțumit | contentment |
| Filmot | *"mă voi simți foarte bine pentru că voi avea posibilitatea să cunosc un nou băiat …"* | `ma_voi_simti_future` | bine | joy |
| FULG | *"Sunt încântat că am șansa de a fi prezent pentru a doua oară la Idelier Concept Store …"* | `sunt_adj_present` | încântat | joy |

**Source-specific cleaning:** Filmot subtitles lack punctuation and mix speakers — a light rule-based pass strips `#Muzică#` tags, splits on `>>` speaker-change markers, and re-inserts periods using a capitalization heuristic before LLM validation.

---

## Slide 4 — LLM Validation (Qwen 3.5-9B on Modal)

Each of the 129,700 candidates is scored on a **0–3 Likert scale** with a MASIVE-style verification prompt (Romanian, 7 in-context examples). vLLM on A100-80GB, temperature 0, parse failures = 0.

**Scale:**

| Score | Label | Meaning |
|-------|-------|---------|
| 0 | Nu este o stare afectivă | Not an affective state |
| 1 | Improbabil o stare afectivă | Unlikely affective |
| 2 | Probabil o stare afectivă | Likely affective |
| 3 | Categoric o stare afectivă | Definitely affective |

**Score distribution (all 129,700):**

| Score | Count | % |
|------:|------:|--:|
| 3 | 73,427 | 56.6% |
| 2 | 30,373 | 23.4% |
| 1 | 16,348 | 12.6% |
| 0 | 9,552 | 7.4% |

**Acceptance rate (LLM ≥ 2) by source and pattern:**

| Source | n | Score ≥ 2 | Rate |
|--------|--:|----------:|-----:|
| Filmot | 28,314 | 24,148 | **85.3%** |
| FULG | 96,481 | 75,952 | 78.7% |
| Small datasets | 4,905 | 3,700 | 75.4% |

| Pattern | Score ≥ 2 |
|---------|----------:|
| `ma_simt_present` | 91.0% |
| `mie_short` | 91.0% |
| `sunt_adj_present` | 77.5% |
| `am_noun_present` | 34.7% |

The LLM correctly identifies `mă simt` as a high-precision pattern and `am + noun` as the noisiest — matching our a-priori expectations and confirming the regex-level quality ordering.

---

## Slide 5 — Human Evaluation & Benchmark Threshold

Pilot human eval with **2 Romanian native annotators** on MTurk Developer Sandbox, stratified sample of 200 candidates (50 per LLM score bin), n=105 completed by both annotators. Same 0–3 scale as the LLM.

**Agreement & correlation:**

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Cohen's κ (quadratic weighted) | **0.649** | Substantial agreement |
| Cohen's κ (unweighted) | 0.295 | Fair (expected for 4-point ordinal) |
| Binary κ (0–1 vs 2–3) | 0.564 | Moderate |
| Binary % agreement | 78.1% | — |
| Spearman's ρ (mean-human vs LLM) | **0.701** | Strong (p < 0.0001) |
| Human validation rate @ LLM ≥ 2 | 71.7% | Comparable to MASIVE Spanish (72%) |

**Human confirmation rate per LLM score:**

| LLM score | n | % confirmed by humans (mean ≥ 2.0) |
|----------:|--:|-----------------------------------:|
| 0 | 25 | 4.0% |
| 1 | 27 | 29.6% |
| 2 | 30 | 46.7% |
| 3 | 23 | **91.3%** |

**Threshold decision for the final benchmark:**

| Threshold | Precision | Recall | F1 | Est. benchmark size |
|-----------|----------:|-------:|---:|--------------------:|
| LLM ≥ 2 | 71.7% | 77.6% | 74.5% | ~104K |
| **LLM ≥ 3** (chosen) | **91.3%** | 42.9% | 58.3% | **73K** |

We pick **LLM ≥ 3**: a cleaner benchmark is more valuable than a larger noisy one for a first-of-its-kind Romanian resource. Final: **73,427 candidates, 910 unique seed words, 87% secondary / 13% primary patterns**.

---

## Slide 6 — Zero-Shot Evaluation

How well do existing models predict the masked emotion word **without any task-specific training**? Test split: 2,658 samples, 212 unique seed words.

**Results on Romanian (native):**

| Model | Type | Params | Acc@1 | Acc@3 | Acc@5 | MRR |
|-------|------|-------:|------:|------:|------:|----:|
| **ro-bert** | MLM | 124M | **35.6%** | **54.0%** | **61.1%** | **0.469** |
| mT5-large | Gen (enc-dec) | 1.2B | 27.9% | 34.6% | 34.6% | 0.311 |
| RoBERT-base | MLM | 125M | 23.4% | 32.7% | 36.0% | 0.287 |
| Qwen3.5-9B | Gen (LLM) | 9B | 22.0% | 33.4% | 34.5% | 0.275 |
| RoGemma2-9b | Gen (LLM) | 9B | 19.4% | 28.9% | 29.8% | 0.239 |
| XLM-R-large | MLM | 550M | 17.2% | 20.5% | 21.9% | 0.193 |
| RoLlama3.1-8b | Gen (LLM) | 8B | 7.7% | 14.3% | 15.6% | 0.110 |

**Translate-test (RO → EN via NLLB-200):**

| Model | Acc@1 (RO) | Acc@1 (EN) | Δ |
|-------|-----------:|-----------:|--:|
| XLM-R-large | 17.2% | 14.5% | −16% |
| Qwen3.5-9B | 22.0% | 12.9% | −41% |

**Gender bias — Acc@1 by seed-word gender:**

| Model | Masc (n=1272) | Fem (n=1013) | Nouns (n=373) |
|-------|--------------:|-------------:|--------------:|
| ro-bert | 35.4% | 30.3% | 50.9% |
| Qwen3.5-9B | 18.7% | 22.5% | 31.9% |
| RoGemma2-9b | 25.6% | 7.5% | 30.6% |
| **XLM-R-large** | 25.8% | **0.1%** | 34.3% |

**Key findings:**
1. A small **Romanian monolingual MLM (ro-bert, 124M)** beats all 9B-parameter LLMs — including Romanian-tuned ones.
2. **Native Romanian beats translated English** for both MLM and LLM — machine translation is not a shortcut to cross-lingual ASI, confirming MASIVE Takeaway #6.
3. **Romanian instruction-tuned LLMs underperform** multilingual Qwen, despite being trained specifically on Romanian.
4. **XLM-R has a severe masculine-default bias** (0.1% feminine accuracy); ro-bert and Qwen do not.
5. **MLMs prefer primary patterns** (`mă simt [X]`), **LLMs prefer secondary patterns** (`sunt [X]`) that reward contextual reasoning.
