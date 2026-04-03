#!/usr/bin/env python3
"""
Run seed enrichment: bootstrapping + distributional mining.

Usage:
    python -m pipeline.seed_enrichment.run                          # small datasets only
    python -m pipeline.seed_enrichment.run --source fulg            # FULG only
    python -m pipeline.seed_enrichment.run --source filmot          # Filmot JSONL only
    python -m pipeline.seed_enrichment.run --source all             # small + FULG + filmot
    python -m pipeline.seed_enrichment.run --source fulg --fulg-max-records 100000
    python -m pipeline.seed_enrichment.run --method bootstrap       # bootstrapping only
    python -m pipeline.seed_enrichment.run --method distributional  # distributional only

Filmot data: run `python -m pipeline.collect.stream_filmot` first to collect data,
then use --source filmot (or --source all) to include it in enrichment.
"""

import argparse
import json
from pathlib import Path

from pipeline.seed.merged import build_seed, ADJECTIVES, NOUNS, ADVERBS
from pipeline.utils.text_utils import normalize_text
from pipeline.utils.stoplists import infer_gender

DATA_DIR = Path(__file__).parent.parent / "data"


def _seed_to_bootstrap_format() -> dict:
    """Convert merged seed dicts to bootstrapping format (word → {emotions, gender})."""
    result = {}
    for word, categ in ADJECTIVES.items():
        result[word] = {"emotions": [categ], "gender": infer_gender(word) or "m"}
    for word, categ in NOUNS.items():
        result[word] = {"emotions": [categ], "gender": infer_gender(word) or "m"}
    for word, categ in ADVERBS.items():
        result[word] = {"emotions": [categ], "gender": infer_gender(word) or "m"}
    return result


def main():
    parser = argparse.ArgumentParser(description="Seed enrichment pipeline")
    parser.add_argument(
        "--source", choices=["small", "fulg", "filmot", "all"],
        default="small",
        help="Data source: small (JSONL files), fulg (stream), filmot (JSONL), all (default: small)",
    )
    parser.add_argument(
        "--method", choices=["bootstrap", "distributional", "both"],
        default="both", help="Which method(s) to run (default: both)",
    )
    parser.add_argument(
        "--bootstrap-rounds", type=int, default=4,
        help="Number of bootstrapping rounds for small datasets (default: 4)",
    )
    parser.add_argument(
        "--co-occurrence-threshold", type=int, default=2,
        help="Min distinct X seeds for bootstrapping (default: 2)",
    )
    parser.add_argument(
        "--min-freq", type=int, default=2,
        help="Min frequency for distributional mining (default: 2)",
    )
    parser.add_argument(
        "--fulg-max-records", type=int, default=0,
        help="Max FULG records to stream, 0 = unlimited (default: 0)",
    )
    parser.add_argument(
        "--fulg-min-language-score", type=float, default=0.8,
        help="Min Romanian language score for FULG (default: 0.8)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data directory with JSONL files (for small source)",
    )
    parser.add_argument(
        "--filmot-path", type=Path, default=None,
        help="Path to filmot JSONL file (default: pipeline/data/filmot_raw.jsonl, "
             "falls back to data/filmot_api_raw_hits.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for enriched seed JSON",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    data_dir = args.data_dir or DATA_DIR
    verbose = not args.quiet

    run_small = args.source in ("small", "all")
    run_fulg = args.source in ("fulg", "all")
    run_filmot = args.source in ("filmot", "all")

    # Default output: enriched_seed.json for combined, separate files for single-source
    if args.output:
        output_path = args.output
    elif args.source == "fulg":
        output_path = DATA_DIR / "fulg_enrichment_results.json"
    elif args.source == "filmot":
        output_path = DATA_DIR / "filmot_enrichment_results.json"
    else:
        output_path = DATA_DIR / "enriched_seed.json"

    # Resolve filmot data path
    filmot_path = args.filmot_path
    if filmot_path is None:
        # Try pipeline/data first, fall back to project root data/
        filmot_path = DATA_DIR / "filmot_raw.jsonl"
        if not filmot_path.exists():
            project_root = Path(__file__).parent.parent.parent
            filmot_path = project_root / "data" / "filmot_api_raw_hits.jsonl"

    # Load starting seed
    print("Loading merged seed (375 words)...")
    seed = build_seed()
    seed_normalized = {normalize_text(w) for w in seed["all_words"]}
    bootstrap_seed = _seed_to_bootstrap_format()

    print(f"  Adjectives: {len(seed['adjectives'])}")
    print(f"  Nouns: {len(seed['nouns'])}")
    print(f"  Adverbs: {len(seed['adverbs'])}")

    bootstrap_result = {"new_words": {}}
    distrib_result = {"new_words": {}}

    # ------------------------------------------------------------------
    # Small datasets (multi-round, loaded into memory)
    # ------------------------------------------------------------------
    if run_small:
        print(f"\n{'=' * 60}")
        print("Source: small datasets (JSONL)")
        print(f"{'=' * 60}")

        if args.method in ("bootstrap", "both"):
            from pipeline.seed_enrichment.bootstrapping import run_bootstrapping
            bootstrap_result = run_bootstrapping(
                data_dir=data_dir,
                seed=bootstrap_seed,
                rounds=args.bootstrap_rounds,
                co_occurrence_threshold=args.co_occurrence_threshold,
                verbose=verbose,
            )
            _save_json(DATA_DIR / "bootstrap_provenance.json", bootstrap_result["provenance"], verbose)

        if args.method in ("distributional", "both"):
            from pipeline.seed_enrichment.distributional import run_distributional_mining
            distrib_result = run_distributional_mining(
                data_dir=data_dir,
                seed_normalized=seed_normalized,
                min_freq=args.min_freq,
                verbose=verbose,
            )
            _save_json(DATA_DIR / "distributional_discovered.json", distrib_result["discovered"], verbose)

    # ------------------------------------------------------------------
    # FULG (single-pass streaming — both methods in one scan)
    # ------------------------------------------------------------------
    if run_fulg:
        print(f"\n{'=' * 60}")
        print("Source: FULG (streaming)")
        print(f"{'=' * 60}")

        from pipeline.utils.corpus_reader import stream_fulg
        from pipeline.utils.pattern_matcher import get_trigger_words

        fulg_bootstrap_result, fulg_distrib_result = _run_fulg_single_pass(
            bootstrap_seed=bootstrap_seed,
            seed_normalized=seed_normalized,
            trigger_words=get_trigger_words(),
            max_records=args.fulg_max_records,
            min_language_score=args.fulg_min_language_score,
            co_occurrence_threshold=args.co_occurrence_threshold,
            min_freq=args.min_freq,
            method=args.method,
            verbose=verbose,
        )

        # Save provenance
        if fulg_bootstrap_result:
            _save_json(DATA_DIR / "bootstrap_fulg_provenance.json",
                       fulg_bootstrap_result["provenance"], verbose)
            for word, info in fulg_bootstrap_result["new_words"].items():
                if word not in bootstrap_result["new_words"]:
                    bootstrap_result["new_words"][word] = info

        if fulg_distrib_result:
            _save_json(DATA_DIR / "distributional_fulg_discovered.json",
                       fulg_distrib_result["discovered"], verbose)
            for word, info in fulg_distrib_result["new_words"].items():
                if word not in distrib_result["new_words"]:
                    distrib_result["new_words"][word] = info

    # ------------------------------------------------------------------
    # Filmot (JSONL file — same approach as small datasets)
    # ------------------------------------------------------------------
    if run_filmot:
        print(f"\n{'=' * 60}")
        print("Source: Filmot (JSONL)")
        print(f"{'=' * 60}")

        if not filmot_path.exists():
            print(f"  WARNING: Filmot data not found at {filmot_path}")
            print(f"  Run `python -m pipeline.collect.stream_filmot` first to collect data.")
        else:
            filmot_bootstrap_result, filmot_distrib_result = _run_filmot(
                filmot_path=filmot_path,
                bootstrap_seed=bootstrap_seed,
                seed_normalized=seed_normalized,
                co_occurrence_threshold=args.co_occurrence_threshold,
                min_freq=args.min_freq,
                method=args.method,
                verbose=verbose,
            )

            if filmot_bootstrap_result:
                _save_json(DATA_DIR / "bootstrap_filmot_provenance.json",
                           filmot_bootstrap_result["provenance"], verbose)
                for word, info in filmot_bootstrap_result["new_words"].items():
                    if word not in bootstrap_result["new_words"]:
                        bootstrap_result["new_words"][word] = info

            if filmot_distrib_result:
                _save_json(DATA_DIR / "distributional_filmot_discovered.json",
                           filmot_distrib_result["discovered"], verbose)
                for word, info in filmot_distrib_result["new_words"].items():
                    if word not in distrib_result["new_words"]:
                        distrib_result["new_words"][word] = info

    # ------------------------------------------------------------------
    # Merge all results
    # ------------------------------------------------------------------
    from pipeline.seed_enrichment.merge_results import (
        merge_enrichment_results, build_enriched_seed, save_enriched_seed,
    )

    merged = merge_enrichment_results(
        bootstrap_result, distrib_result, seed_normalized, verbose=verbose,
    )

    original_seed_dicts = {
        "adjectives": dict(ADJECTIVES),
        "nouns": dict(NOUNS),
        "adverbs": dict(ADVERBS),
    }
    enriched = build_enriched_seed(original_seed_dicts, merged["new_words"])
    save_enriched_seed(enriched, output_path)


def _run_fulg_single_pass(
    bootstrap_seed: dict,
    seed_normalized: set,
    trigger_words: set,
    max_records: int,
    min_language_score: float,
    co_occurrence_threshold: int,
    min_freq: int,
    method: str,
    verbose: bool,
):
    """
    Single-pass FULG streaming: runs both bootstrapping and distributional
    patterns on each page, with a progress bar showing discovered words.
    """
    import re
    from collections import Counter, defaultdict
    from pipeline.utils.corpus_reader import stream_fulg
    from pipeline.utils.text_utils import normalize_text
    from pipeline.utils.pattern_matcher import extract_sentence, NOUN_EMOTION_MAP
    from pipeline.utils.stoplists import STOPWORDS

    run_bootstrap = method in ("bootstrap", "both")
    run_distrib = method in ("distributional", "both")

    # --- Bootstrap setup ---
    bootstrap_candidates = {}
    bootstrap_match_count = 0
    norm_seed_lookup = {}
    conj_patterns = []

    if run_bootstrap:
        from pipeline.seed_enrichment.bootstrapping import (
            build_conjunction_patterns, CandidateEvidence, _build_seed_lookup,
        )
        conj_patterns = build_conjunction_patterns(list(bootstrap_seed.keys()))
        norm_seed_lookup = _build_seed_lookup(bootstrap_seed)

    # --- Distributional setup ---
    distrib_word_data = defaultdict(lambda: {"frequency": 0, "patterns": set(), "examples": []})
    distrib_match_count = 0

    if run_distrib:
        from pipeline.seed_enrichment.distributional import COMPILED_DISCOVERY

    # --- Progress bar ---
    pbar = None
    if max_records > 0:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_records, desc="FULG", unit="rec")
        except ImportError:
            pass

    # Track only NEW words (not already in seed) for progress bar
    bootstrap_new = set()
    distrib_new = set()
    original_seed_normalized = {normalize_text(w) for w in bootstrap_seed}

    fulg_iter = stream_fulg(
        max_records=max_records,
        min_language_score=min_language_score,
        trigger_words=trigger_words,
        progress=False,  # we handle our own progress bar
    )

    for record_id, text, source in fulg_iter:
        norm_text = normalize_text(text)

        # --- Bootstrapping scan ---
        if run_bootstrap:
            for pat in conj_patterns:
                for m in pat.finditer(norm_text):
                    groups = m.groups()
                    if len(groups) < 2:
                        continue
                    x_word = groups[-2].lower()
                    y_word_norm = groups[-1].lower()

                    x_info = norm_seed_lookup.get(x_word)
                    if not x_info:
                        continue

                    bootstrap_match_count += 1

                    # Only track if not in seed
                    if y_word_norm not in original_seed_normalized:
                        bootstrap_new.add(y_word_norm)

                    y_orig = y_word_norm
                    for w in re.findall(r"\b\w+\b", text, re.UNICODE):
                        if normalize_text(w) == y_word_norm:
                            y_orig = w
                            break

                    sentence = extract_sentence(text, m.start(), m.end())

                    if y_word_norm not in bootstrap_candidates:
                        bootstrap_candidates[y_word_norm] = CandidateEvidence(
                            word=y_orig, normalized=y_word_norm,
                        )
                    bootstrap_candidates[y_word_norm].add_evidence(
                        x_seed=x_word,
                        x_emotions=x_info["emotions"],
                        x_gender=x_info["gender"],
                        verb_form=m.group(0)[:30],
                        sentence=sentence,
                    )

        # --- Distributional scan ---
        if run_distrib:
            for pattern_name, pattern in COMPILED_DISCOVERY:
                for match in pattern.finditer(norm_text):
                    word = match.group(1).strip().lower()
                    if len(word) < 3 or len(word) > 20:
                        continue
                    if word in STOPWORDS:
                        continue
                    if any(c.isdigit() for c in word):
                        continue

                    distrib_match_count += 1

                    # Only track if not in seed
                    if word not in seed_normalized:
                        distrib_new.add(word)

                    entry = distrib_word_data[word]
                    entry["frequency"] += 1
                    entry["patterns"].add(pattern_name)
                    if len(entry["examples"]) < 3:
                        start = max(0, match.start() - 40)
                        end = min(len(norm_text), match.end() + 40)
                        entry["examples"].append({
                            "context": norm_text[start:end].strip(),
                            "source": source,
                            "record_id": record_id,
                        })

        if pbar:
            pbar.update(1)
            postfix = {}
            if run_bootstrap:
                postfix["boot_new"] = len(bootstrap_new)
            if run_distrib:
                postfix["dist_new"] = len(distrib_new)
            pbar.set_postfix(postfix)

    if pbar:
        pbar.close()

    # --- Process bootstrap results ---
    fulg_bootstrap_result = None
    if run_bootstrap:
        from pipeline.seed_enrichment.bootstrapping import (
            validate_candidate, compute_confidence, infer_emotions,
        )
        from pipeline.utils.stoplists import infer_gender

        accepted = []
        rejected_reasons = Counter()
        for y_norm, evidence in bootstrap_candidates.items():
            valid, reason = validate_candidate(
                evidence, original_seed_normalized, co_occurrence_threshold,
            )
            if valid:
                confidence = compute_confidence(evidence)
                emotions = infer_emotions(evidence)
                accepted.append({
                    "word": evidence.word,
                    "normalized": y_norm,
                    "emotions": emotions,
                    "gender": infer_gender(evidence.word) or "m",
                    "confidence": round(confidence, 3),
                    "co_occurring_seeds": sorted(evidence.co_occurring_seeds),
                    "source_count": evidence.source_count,
                    "sample_sentences": evidence.source_sentences[:3],
                })
            else:
                rejected_reasons[reason] += 1

        new_words = {
            a["word"]: {"emotions": a["emotions"], "gender": a["gender"]}
            for a in accepted
            if a["normalized"] not in original_seed_normalized
        }

        if verbose:
            print(f"\nBootstrap (FULG): {bootstrap_match_count} matches, "
                  f"{len(bootstrap_new)} new unique Y, {len(accepted)} accepted")
            print(f"  Rejected: {dict(rejected_reasons)}")
            if accepted:
                top = sorted(accepted, key=lambda x: -x["confidence"])[:10]
                print(f"  Top: {[a['word'] for a in top]}")

        fulg_bootstrap_result = {
            "new_words": new_words,
            "provenance": {
                "mode": "streaming_fulg",
                "conjunction_matches": bootstrap_match_count,
                "unique_new_candidates": len(bootstrap_new),
                "accepted": len(accepted),
                "new_words_count": len(new_words),
                "rejected_reasons": dict(rejected_reasons),
                "accepted_words": [
                    {"word": a["word"], "emotions": a["emotions"], "confidence": a["confidence"]}
                    for a in sorted(accepted, key=lambda x: -x["confidence"])
                ],
            },
        }

    # --- Process distributional results ---
    fulg_distrib_result = None
    if run_distrib:
        from pipeline.seed_enrichment.distributional import expand_seed_with_discoveries

        discovered = {}
        for word, data in sorted(distrib_word_data.items(), key=lambda x: -x[1]["frequency"]):
            if data["frequency"] < min_freq:
                continue
            discovered[word] = {
                "frequency": data["frequency"],
                "patterns": sorted(data["patterns"]),
                "examples": data["examples"],
            }

        new_words = expand_seed_with_discoveries(discovered, seed_normalized, verbose=verbose)

        if verbose:
            print(f"\nDistributional (FULG): {distrib_match_count} matches, "
                  f"{len(distrib_new)} new unique, {len(discovered)} (freq>={min_freq})")

        fulg_distrib_result = {
            "new_words": new_words,
            "discovered": discovered,
            "stats": {"total_discovered": len(discovered), "new_words": len(new_words)},
        }

    return fulg_bootstrap_result, fulg_distrib_result


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _run_filmot(
    filmot_path: Path,
    bootstrap_seed: dict,
    seed_normalized: set,
    co_occurrence_threshold: int,
    min_freq: int,
    method: str,
    verbose: bool,
):
    """
    Single-pass filmot enrichment with progress bar.

    Reads the JSONL once, runs both bootstrapping and distributional on each
    record (same approach as _run_fulg_single_pass).
    """
    import re
    from collections import Counter, defaultdict
    from pipeline.utils.text_utils import normalize_text
    from pipeline.utils.pattern_matcher import extract_sentence
    from pipeline.utils.stoplists import STOPWORDS

    run_bootstrap = method in ("bootstrap", "both")
    run_distrib = method in ("distributional", "both")

    # --- Bootstrap setup ---
    bootstrap_candidates = {}
    bootstrap_match_count = 0
    norm_seed_lookup = {}
    conj_patterns = []

    if run_bootstrap:
        from pipeline.seed_enrichment.bootstrapping import (
            build_conjunction_patterns, CandidateEvidence, _build_seed_lookup,
        )
        conj_patterns = build_conjunction_patterns(list(bootstrap_seed.keys()))
        norm_seed_lookup = _build_seed_lookup(bootstrap_seed)

    # --- Distributional setup ---
    distrib_word_data = defaultdict(lambda: {"frequency": 0, "patterns": set(), "examples": []})
    distrib_match_count = 0

    if run_distrib:
        from pipeline.seed_enrichment.distributional import COMPILED_DISCOVERY

    # Track only NEW words for progress bar
    bootstrap_new = set()
    distrib_new = set()
    original_seed_normalized = {normalize_text(w) for w in bootstrap_seed}

    # --- Progress bar ---
    total_lines = _count_jsonl_lines(filmot_path)
    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_lines, desc="Filmot", unit="rec")
    except ImportError:
        pass

    # --- Single pass ---
    import json as _json
    count = 0
    with open(filmot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = _json.loads(line)
            text = record.get("full_context", "")
            if not text:
                if pbar:
                    pbar.update(1)
                continue

            record_id = f"filmot_{record.get('video_id', count)}_{record.get('hit_start', 0)}"
            source = "filmot"
            count += 1
            norm_text = normalize_text(text)

            # --- Bootstrapping scan ---
            if run_bootstrap:
                for pat in conj_patterns:
                    for m in pat.finditer(norm_text):
                        groups = m.groups()
                        if len(groups) < 2:
                            continue
                        x_word = groups[-2].lower()
                        y_word_norm = groups[-1].lower()

                        x_info = norm_seed_lookup.get(x_word)
                        if not x_info:
                            continue

                        bootstrap_match_count += 1

                        if y_word_norm not in original_seed_normalized:
                            bootstrap_new.add(y_word_norm)

                        y_orig = y_word_norm
                        for w in re.findall(r"\b\w+\b", text, re.UNICODE):
                            if normalize_text(w) == y_word_norm:
                                y_orig = w
                                break

                        sentence = extract_sentence(text, m.start(), m.end())

                        if y_word_norm not in bootstrap_candidates:
                            bootstrap_candidates[y_word_norm] = CandidateEvidence(
                                word=y_orig, normalized=y_word_norm,
                            )
                        bootstrap_candidates[y_word_norm].add_evidence(
                            x_seed=x_word,
                            x_emotions=x_info["emotions"],
                            x_gender=x_info["gender"],
                            verb_form=m.group(0)[:30],
                            sentence=sentence,
                        )

            # --- Distributional scan ---
            if run_distrib:
                for pattern_name, pattern in COMPILED_DISCOVERY:
                    for match in pattern.finditer(norm_text):
                        word = match.group(1).strip().lower()
                        if len(word) < 3 or len(word) > 20:
                            continue
                        if word in STOPWORDS:
                            continue
                        if any(c.isdigit() for c in word):
                            continue

                        distrib_match_count += 1

                        if word not in seed_normalized:
                            distrib_new.add(word)

                        entry = distrib_word_data[word]
                        entry["frequency"] += 1
                        entry["patterns"].add(pattern_name)
                        if len(entry["examples"]) < 3:
                            start = max(0, match.start() - 40)
                            end = min(len(norm_text), match.end() + 40)
                            entry["examples"].append({
                                "context": norm_text[start:end].strip(),
                                "source": source,
                                "record_id": record_id,
                            })

            if pbar:
                pbar.update(1)
                postfix = {}
                if run_bootstrap:
                    postfix["boot_new"] = len(bootstrap_new)
                if run_distrib:
                    postfix["dist_new"] = len(distrib_new)
                pbar.set_postfix(postfix)

    if pbar:
        pbar.close()

    # --- Process bootstrap results ---
    filmot_bootstrap_result = None
    if run_bootstrap:
        from pipeline.seed_enrichment.bootstrapping import (
            validate_candidate, compute_confidence, infer_emotions,
        )
        from pipeline.utils.stoplists import infer_gender

        accepted = []
        rejected_reasons = Counter()
        for y_norm, evidence in bootstrap_candidates.items():
            valid, reason = validate_candidate(
                evidence, original_seed_normalized, co_occurrence_threshold,
            )
            if valid:
                confidence = compute_confidence(evidence)
                emotions = infer_emotions(evidence)
                accepted.append({
                    "word": evidence.word,
                    "normalized": y_norm,
                    "emotions": emotions,
                    "gender": infer_gender(evidence.word) or "m",
                    "confidence": round(confidence, 3),
                    "co_occurring_seeds": sorted(evidence.co_occurring_seeds),
                    "source_count": evidence.source_count,
                    "sample_sentences": evidence.source_sentences[:3],
                })
            else:
                rejected_reasons[reason] += 1

        new_words = {
            a["word"]: {"emotions": a["emotions"], "gender": a["gender"]}
            for a in accepted
            if a["normalized"] not in original_seed_normalized
        }

        if verbose:
            print(f"\nBootstrap (Filmot): {bootstrap_match_count} matches, "
                  f"{len(bootstrap_new)} new unique Y, {len(accepted)} accepted")
            print(f"  Rejected: {dict(rejected_reasons)}")
            if accepted:
                top = sorted(accepted, key=lambda x: -x["confidence"])[:10]
                print(f"  Top: {[a['word'] for a in top]}")

        filmot_bootstrap_result = {
            "new_words": new_words,
            "provenance": {
                "mode": "filmot_jsonl",
                "conjunction_matches": bootstrap_match_count,
                "unique_new_candidates": len(bootstrap_new),
                "accepted": len(accepted),
                "new_words_count": len(new_words),
                "rejected_reasons": dict(rejected_reasons),
                "accepted_words": [
                    {"word": a["word"], "emotions": a["emotions"], "confidence": a["confidence"]}
                    for a in sorted(accepted, key=lambda x: -x["confidence"])
                ],
            },
        }

    # --- Process distributional results ---
    filmot_distrib_result = None
    if run_distrib:
        from pipeline.seed_enrichment.distributional import expand_seed_with_discoveries

        discovered = {}
        for word, data in sorted(distrib_word_data.items(), key=lambda x: -x[1]["frequency"]):
            if data["frequency"] < min_freq:
                continue
            discovered[word] = {
                "frequency": data["frequency"],
                "patterns": sorted(data["patterns"]),
                "examples": data["examples"],
            }

        new_words = expand_seed_with_discoveries(discovered, seed_normalized, verbose=verbose)

        if verbose:
            print(f"\nDistributional (Filmot): {distrib_match_count} matches, "
                  f"{len(distrib_new)} new unique, {len(discovered)} (freq>={min_freq})")

        filmot_distrib_result = {
            "new_words": new_words,
            "discovered": discovered,
            "stats": {"total_discovered": len(discovered), "new_words": len(new_words)},
        }

    return filmot_bootstrap_result, filmot_distrib_result


def _save_json(path: Path, data, verbose: bool):
    """Save JSON with optional verbose message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
