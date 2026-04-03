#!/usr/bin/env python3
"""
Merge the original seed with filtered enrichment results from all sources.

Reads:
    - pipeline/seed/merged.py (original 375-word seed)
    - pipeline/data/enriched_seed.json (small dataset enrichment)
    - pipeline/data/fulg_enrichment_filtered.json (FULG filtered)
    - pipeline/data/filmot_enrichment_filtered.json (filmot filtered)

Writes:
    - pipeline/data/enriched_seed_merged.json

Each filtered file contains new words that were manually reviewed.
This script simply unions them, skipping duplicates (first source wins).

Usage:
    python -m pipeline.seed_enrichment.merge_all_sources
    python -m pipeline.seed_enrichment.merge_all_sources --output path/to/output.json
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def merge_all_sources(
    output_path: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Merge original seed + all filtered enrichment sources.

    Sources are applied in order. For each new word, the first source
    to introduce it wins (no overwriting).

    Returns the merged enriched seed dict.
    """
    from pipeline.seed.merged import ADJECTIVES, NOUNS, ADVERBS

    if output_path is None:
        output_path = DATA_DIR / "enriched_seed_merged.json"

    # Start with original seed
    adjectives = dict(ADJECTIVES)
    nouns = dict(NOUNS)
    adverbs = dict(ADVERBS)

    original_total = len(adjectives) + len(nouns) + len(adverbs)
    if verbose:
        print(f"Original seed: {len(adjectives)} adj + {len(nouns)} nouns + {len(adverbs)} adv")

    sources_log = []

    # ------------------------------------------------------------------
    # Source 1: Small dataset enrichment (enriched_seed.json)
    # ------------------------------------------------------------------
    small_path = DATA_DIR / "enriched_seed.json"
    if small_path.exists():
        with open(small_path) as f:
            small = json.load(f)

        added = _add_from_enriched_seed(adjectives, nouns, adverbs, small)
        sources_log.append({
            "name": "small_datasets",
            "file": str(small_path),
            "new_adjectives": added["adj"],
            "new_nouns": added["nouns"],
            "new_adverbs": added["adv"],
        })
        if verbose:
            print(f"+ Small datasets: {added['adj']} adj, {added['nouns']} nouns, {added['adv']} adv")
    elif verbose:
        print(f"  Skipping small datasets (not found: {small_path})")

    # ------------------------------------------------------------------
    # Source 2: FULG filtered
    # ------------------------------------------------------------------
    fulg_path = DATA_DIR / "fulg_enrichment_filtered.json"
    if fulg_path.exists():
        with open(fulg_path) as f:
            fulg = json.load(f)

        added = _add_from_filtered(adjectives, nouns, adverbs, fulg)
        sources_log.append({
            "name": "fulg",
            "file": str(fulg_path),
            "new_adjectives": added["adj"],
            "new_nouns": added["nouns"],
            "new_adverbs": added["adv"],
        })
        if verbose:
            print(f"+ FULG filtered: {added['adj']} adj, {added['nouns']} nouns, {added['adv']} adv")
    elif verbose:
        print(f"  Skipping FULG (not found: {fulg_path})")

    # ------------------------------------------------------------------
    # Source 3: Filmot filtered
    # ------------------------------------------------------------------
    filmot_path = DATA_DIR / "filmot_enrichment_filtered.json"
    if filmot_path.exists():
        with open(filmot_path) as f:
            filmot = json.load(f)

        added = _add_from_filtered(adjectives, nouns, adverbs, filmot)
        sources_log.append({
            "name": "filmot",
            "file": str(filmot_path),
            "new_adjectives": added["adj"],
            "new_nouns": added["nouns"],
            "new_adverbs": added["adv"],
        })
        if verbose:
            print(f"+ Filmot filtered: {added['adj']} adj, {added['nouns']} nouns, {added['adv']} adv")
    elif verbose:
        print(f"  Skipping filmot (not found: {filmot_path})")

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    total = len(adjectives) + len(nouns) + len(adverbs)

    result = {
        "_metadata": {
            "description": "Enriched seed: original merged seed + manually filtered enrichment from all sources.",
            "sources": sources_log,
            "merge_strategy": "Union of all sources. First source to introduce a word wins. No cross-source validation.",
        },
        "adjectives": adjectives,
        "nouns": nouns,
        "adverbs": adverbs,
        "stats": {
            "adjectives": len(adjectives),
            "nouns": len(nouns),
            "adverbs": len(adverbs),
            "total": total,
            "original": original_total,
            "new": total - original_total,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nEnriched seed: {len(adjectives)} adj + {len(nouns)} nouns + {len(adverbs)} adv = {total}")
        print(f"New words added: {total - original_total}")
        print(f"Saved to {output_path}")

    return result


def _add_from_enriched_seed(adjectives, nouns, adverbs, enriched):
    """Add words from an enriched_seed.json (same format as build_seed output)."""
    added = {"adj": 0, "nouns": 0, "adv": 0}
    for w, c in enriched.get("adjectives", {}).items():
        if w not in adjectives:
            adjectives[w] = c
            added["adj"] += 1
    for w, c in enriched.get("nouns", {}).items():
        if w not in nouns:
            nouns[w] = c
            added["nouns"] += 1
    for w, c in enriched.get("adverbs", {}).items():
        if w not in adverbs:
            adverbs[w] = c
            added["adv"] += 1
    return added


def _add_from_filtered(adjectives, nouns, adverbs, filtered):
    """Add words from a *_enrichment_filtered.json file."""
    added = {"adj": 0, "nouns": 0, "adv": 0}
    for w, c in filtered.get("new_adjectives", {}).items():
        if w not in adjectives:
            adjectives[w] = c
            added["adj"] += 1
    for w, c in filtered.get("new_nouns", {}).items():
        if w not in nouns:
            nouns[w] = c
            added["nouns"] += 1
    for w, c in filtered.get("new_adverbs", {}).items():
        if w not in adverbs:
            adverbs[w] = c
            added["adv"] += 1
    return added


def main():
    parser = argparse.ArgumentParser(
        description="Merge original seed with all filtered enrichment sources"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: pipeline/data/enriched_seed_merged.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    merge_all_sources(
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
