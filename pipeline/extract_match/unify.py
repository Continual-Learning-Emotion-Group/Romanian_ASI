#!/usr/bin/env python3
"""
Unify extracted candidates from all sources into a single dataset.

Reads pattern candidates from small datasets, filmot, and FULG,
normalizes to a common schema, deduplicates, and writes a single JSONL.

Does NOT overwrite any existing data file.

Usage:
    python -m pipeline.extract_match.unify
    python -m pipeline.extract_match.unify --filmot-variant pp
    python -m pipeline.extract_match.unify --no-fulg
    python -m pipeline.extract_match.unify --max-per-source 10000
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"

DEFAULT_SMALL = DATA_DIR / "pattern_candidates_small.jsonl"
DEFAULT_FILMOT_LIGHT = DATA_DIR / "pattern_candidates_filmot_light.jsonl"
DEFAULT_FILMOT_PP = DATA_DIR / "pattern_candidates_filmot_pp.jsonl"
DEFAULT_FULG = DATA_DIR / "pattern_candidates_fulg.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "candidates_unified.jsonl"

# Fields to drop during normalization
_DROP_FIELDS = {"text_light", "text_original", "text_pp", "_text_hash", "_domain_sent_hash",
                "pp_sentences_before", "pp_sentences_after"}


def _dedup_key(matched_sentence: str) -> str:
    return hashlib.md5(matched_sentence.strip().lower().encode("utf-8")).hexdigest()


def load_jsonl(path: Path, max_records: int = 0) -> list:
    records = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                bad += 1
    if max_records > 0:
        records = records[:max_records]
    if bad:
        print(f"  ({bad} bad lines skipped in {path.name})")
    return records


def normalize_filmot(records: list, variant: str) -> list:
    """Normalize filmot records: promote text_light/text_pp to text."""
    text_field = "text_light" if variant == "light" else "text_pp"
    out = []
    for rec in records:
        if text_field in rec:
            rec["text"] = rec[text_field]
        for key in _DROP_FIELDS:
            rec.pop(key, None)
        out.append(rec)
    return out


def normalize_fulg(records: list) -> list:
    """Normalize FULG records: drop internal hash fields."""
    for rec in records:
        for key in _DROP_FIELDS:
            rec.pop(key, None)
    return records


def unify(
    small_path: Path = None,
    filmot_path: Path = None,
    fulg_path: Path = None,
    output_path: Path = None,
    filmot_variant: str = "light",
    include_fulg: bool = True,
    max_per_source: int = 0,
    verbose: bool = True,
) -> dict:
    if small_path is None:
        small_path = DEFAULT_SMALL
    if filmot_path is None:
        filmot_path = DEFAULT_FILMOT_LIGHT if filmot_variant == "light" else DEFAULT_FILMOT_PP
    if fulg_path is None:
        fulg_path = DEFAULT_FULG
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    # --- Load sources (priority order: small > filmot > fulg) ---
    sources = []

    if small_path.exists():
        if verbose:
            print(f"Loading small datasets from {small_path.name}...")
        recs = load_jsonl(small_path, max_per_source)
        sources.append(("small", recs))
        if verbose:
            print(f"  {len(recs)} records")
    else:
        print(f"  Skipping small datasets ({small_path} not found)")

    if filmot_path.exists():
        if verbose:
            print(f"Loading filmot ({filmot_variant}) from {filmot_path.name}...")
        recs = load_jsonl(filmot_path, max_per_source)
        recs = normalize_filmot(recs, filmot_variant)
        sources.append(("filmot", recs))
        if verbose:
            print(f"  {len(recs)} records")
    else:
        print(f"  Skipping filmot ({filmot_path} not found)")

    if include_fulg:
        if fulg_path.exists():
            if verbose:
                print(f"Loading FULG from {fulg_path.name}...")
            recs = load_jsonl(fulg_path, max_per_source)
            recs = normalize_fulg(recs)
            sources.append(("fulg", recs))
            if verbose:
                print(f"  {len(recs)} records")
        else:
            print(f"  Skipping FULG ({fulg_path} not found)")

    # --- Deduplicate by matched_sentence ---
    seen = set()
    kept = []
    stats = {
        "total_input": 0,
        "total_output": 0,
        "duplicates_removed": 0,
        "by_source_input": {},
        "by_source_output": {},
        "by_pattern": defaultdict(int),
        "by_emotion": defaultdict(int),
    }

    for source_name, records in sources:
        source_kept = 0
        source_dupes = 0
        stats["by_source_input"][source_name] = len(records)
        stats["total_input"] += len(records)

        iter_recs = tqdm(records, desc=f"Dedup {source_name}", unit="rec") if verbose else records
        for rec in iter_recs:
            ms = rec.get("matched_sentence", "")
            key = _dedup_key(ms)
            if key in seen:
                source_dupes += 1
                stats["duplicates_removed"] += 1
                continue
            seen.add(key)
            kept.append(rec)
            source_kept += 1

            stats["by_pattern"][rec.get("pattern_used", "unknown")] += 1
            for emo in rec.get("emotion_category", []):
                stats["by_emotion"][emo] += 1

        stats["by_source_output"][source_name] = source_kept
        if verbose and source_dupes:
            print(f"  {source_name}: {source_dupes} duplicates removed")

    stats["total_output"] = len(kept)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Stats ---
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])

    if verbose:
        n = stats["total_output"]
        print(f"\n{'='*60}")
        print(f"Unification complete")
        print(f"{'='*60}")
        print(f"Total input:  {stats['total_input']:,}")
        print(f"Total output: {n:,}")
        print(f"Duplicates:   {stats['duplicates_removed']:,}")
        print()
        for src, cnt in stats["by_source_output"].items():
            inp = stats["by_source_input"][src]
            print(f"  {src}: {inp:,} → {cnt:,} ({inp - cnt} dupes)")
        print(f"\nTop patterns:")
        for pat, cnt in sorted(stats["by_pattern"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {pat}: {cnt:,} ({cnt/max(n,1)*100:.1f}%)")
        print(f"\nTop emotions:")
        for emo, cnt in sorted(stats["by_emotion"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {emo}: {cnt:,} ({cnt/max(n,1)*100:.1f}%)")
        print(f"\nOutput: {output_path}")

    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Unify extracted candidates from all sources"
    )
    parser.add_argument(
        "--small", type=Path, default=None,
        help=f"Small datasets JSONL (default: {DEFAULT_SMALL})",
    )
    parser.add_argument(
        "--filmot", type=Path, default=None,
        help="Filmot JSONL (default: auto from --filmot-variant)",
    )
    parser.add_argument(
        "--fulg", type=Path, default=None,
        help=f"FULG JSONL (default: {DEFAULT_FULG})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--filmot-variant", choices=["light", "pp"], default="light",
        help="Which filmot post-processing to use (default: light)",
    )
    parser.add_argument(
        "--no-fulg", action="store_true",
        help="Exclude FULG candidates",
    )
    parser.add_argument(
        "--max-per-source", type=int, default=0,
        help="Max records per source (0 = all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    unify(
        small_path=args.small,
        filmot_path=args.filmot,
        fulg_path=args.fulg,
        output_path=args.output,
        filmot_variant=args.filmot_variant,
        include_fulg=not args.no_fulg,
        max_per_source=args.max_per_source,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
