# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Romanian ASI (Affective State Identification) Benchmark — creates a Romanian language benchmark for identifying affective state expressions following the MASIVE paper methodology. The goal is to extract natural "I feel [state]" expressions from Romanian text corpora.

All active code lives in `pipeline/`. Old experiments and scripts are archived in `deprecated/`.

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# --- Seed ---
python -m pipeline.seed.bridge        # Bridge-only seed (229 words)
python -m pipeline.seed.merged        # Bridge + old curated seed (375 words)

# --- Data Collection ---
python -m pipeline.collect.merge_small                           # Merge 6 small datasets → merged_corpus.jsonl
python -m pipeline.collect.stream_fulg --max-records 50000       # Stream from FULG (trigger-filtered)
python -m pipeline.collect.stream_filmot --max-hits 100000       # Stream from Filmot API

# --- Seed Enrichment ---
python -m pipeline.seed_enrichment.run                           # Small datasets only
python -m pipeline.seed_enrichment.run --source fulg             # FULG streaming
python -m pipeline.seed_enrichment.run --source filmot           # Filmot JSONL
python -m pipeline.seed_enrichment.merge_all_sources             # Merge all → enriched_seed_merged.json (524 words)

# --- Pattern Extraction ---
python -m pipeline.extract_match.run                             # Small datasets → pattern_candidates_small.jsonl
python -m pipeline.extract_match.filmot --workers 8 --resume     # Filmot API → pattern_candidates_filmot.jsonl
python -m pipeline.extract_match.fulg --max-samples 100000       # FULG → pattern_candidates_fulg.jsonl
python -m pipeline.extract_match.postprocess_filmot_light        # Light cleanup for filmot
python -m pipeline.extract_match.unify                           # Merge all → candidates_unified.jsonl (130K)

# --- LLM Validation (Modal GPU) ---
modal run pipeline/llm_validation/modal_validate.py                              # Full run (130K)
modal run pipeline/llm_validation/modal_validate.py --max-candidates 200 --shuffle  # Quick test
modal run pipeline/llm_validation/modal_validate.py --resume --shuffle           # Resume

# --- Human Evaluation ---
python -m pipeline.human_eval.sample                             # Stratified sample (200)
python -m pipeline.human_eval.prepare_csv                        # Convert to MTurk CSV
python -m pipeline.human_eval.agreement data1.csv data2.csv      # Compute IAA
```

## Architecture

```
pipeline/
├── data/                     # All pipeline outputs (JSONL, stats, checkpoints)
├── seed/                     # Seed construction (WN-Affect bridge + curated merge)
├── collect/                  # Data collection (small datasets, FULG, Filmot)
├── seed_enrichment/          # Bootstrapping + distributional mining → enriched seed
├── utils/                    # Pattern matcher (20 patterns), inflection, text utils
├── extract_match/            # Pattern extraction + unification
├── extract_embed/            # Embedding extraction (experimental, not used)
├── llm_validation/           # LLM verification (Qwen 3.5-9B on Modal A100-80GB)
└── human_eval/               # Human annotation + agreement analysis
```

### Pipeline Flow
```
Seed (524 words) + 20 "I feel" patterns
    ↓
3 Sources: Small datasets (5K) + Filmot (29K) + FULG (100K)
    ↓
candidates_unified.jsonl (130K)
    ↓
LLM Validation (Qwen 3.5-9B, 0-3 Likert scale)
    ↓
candidates_validated.jsonl (130K with llm_affect_score)
    ↓
Human Evaluation (pilot: 105 samples, 2 annotators)
```

## Key Considerations

- Romanian has gendered adjectives (fericit/fericită) — both forms in seed, expanded via MULTEXT-East
- Diacritics are inconsistent in social media — patterns match both normalized and original forms
- "sunt" pattern is ambiguous (1st person "I am" vs 3rd person plural "they are") — LLM validation filters noise
- Filmot YouTube auto-captions lack punctuation — light post-processing adds periods via capitalization heuristic
- LLM validation uses MASIVE-style verification prompt (0-3 Likert scale, 7 few-shot examples)

## References

See `references/` for MASIVE, RoEmoLex, FULG, and Working Emotion Vocab papers.
Old experiments and analysis docs are in `deprecated/`.
