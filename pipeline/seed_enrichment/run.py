#!/usr/bin/env python3
"""
Run seed enrichment: bootstrapping + distributional mining.

Usage:
    python -m pipeline.seed_enrichment.run
    python -m pipeline.seed_enrichment.run --method bootstrap
    python -m pipeline.seed_enrichment.run --method distributional
    python -m pipeline.seed_enrichment.run --bootstrap-rounds 6 --min-freq 3
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
        "--method", choices=["bootstrap", "distributional", "both"],
        default="both", help="Which method(s) to run (default: both)",
    )
    parser.add_argument(
        "--bootstrap-rounds", type=int, default=4,
        help="Number of bootstrapping rounds (default: 4)",
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
        "--data-dir", type=Path, default=None,
        help="Data directory with JSONL files",
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
    output_path = args.output or DATA_DIR / "enriched_seed.json"
    verbose = not args.quiet

    # Load starting seed
    print("Loading merged seed (375 words)...")
    seed = build_seed()
    seed_normalized = {normalize_text(w) for w in seed["all_words"]}

    print(f"  Adjectives: {len(seed['adjectives'])}")
    print(f"  Nouns: {len(seed['nouns'])}")
    print(f"  Adverbs: {len(seed['adverbs'])}")

    bootstrap_result = {"new_words": {}}
    distrib_result = {"new_words": {}}

    # Run bootstrapping
    if args.method in ("bootstrap", "both"):
        from pipeline.seed_enrichment.bootstrapping import run_bootstrapping

        bootstrap_seed = _seed_to_bootstrap_format()
        bootstrap_result = run_bootstrapping(
            data_dir=data_dir,
            seed=bootstrap_seed,
            rounds=args.bootstrap_rounds,
            co_occurrence_threshold=args.co_occurrence_threshold,
            verbose=verbose,
        )

        # Save provenance
        prov_path = DATA_DIR / "bootstrap_provenance.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(bootstrap_result["provenance"], f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\nBootstrap provenance saved to {prov_path}")

    # Run distributional mining
    if args.method in ("distributional", "both"):
        from pipeline.seed_enrichment.distributional import run_distributional_mining

        distrib_result = run_distributional_mining(
            data_dir=data_dir,
            seed_normalized=seed_normalized,
            min_freq=args.min_freq,
            verbose=verbose,
        )

        # Save discovered words
        disc_path = DATA_DIR / "distributional_discovered.json"
        disc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(disc_path, "w", encoding="utf-8") as f:
            json.dump(distrib_result["discovered"], f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\nDiscovered words saved to {disc_path}")

    # Merge results
    from pipeline.seed_enrichment.merge_results import (
        merge_enrichment_results, build_enriched_seed, save_enriched_seed,
    )

    merged = merge_enrichment_results(
        bootstrap_result, distrib_result, seed_normalized, verbose=verbose,
    )

    # Pass the actual dicts (seed["adjectives"] is a list, we need the dicts)
    original_seed_dicts = {
        "adjectives": dict(ADJECTIVES),
        "nouns": dict(NOUNS),
        "adverbs": dict(ADVERBS),
    }
    enriched = build_enriched_seed(original_seed_dicts, merged["new_words"])
    save_enriched_seed(enriched, output_path)


if __name__ == "__main__":
    main()
