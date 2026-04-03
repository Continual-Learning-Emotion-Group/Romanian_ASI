#!/usr/bin/env python3
"""
Post-processing filter for embedding extraction results.

Removes hits whose nearest anchor used a noun-only pattern with a noun
that isn't in the curated EMOTION_NOUNS_ONLY set. These are invalid
anchors that should not have been generated (e.g., "îmi este mulțumire",
"aveam curiozitate").

Also supports threshold filtering to avoid re-running the full pipeline.

Usage:
    python -m pipeline.extract_embed.filter_results
    python -m pipeline.extract_embed.filter_results --min-confidence 0.90
    python -m pipeline.extract_embed.filter_results --output filtered.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.pattern_matcher import EMOTION_NOUNS_ONLY
from pipeline.utils.text_utils import normalize_text

INPUT_PATH = Path(__file__).parent.parent / "data" / "embedding_asi_candidates.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "embedding_asi_candidates_filtered.jsonl"

# Noun-only pattern names (these should only pair with EMOTION_NOUNS_ONLY)
NOUN_ONLY_PATTERNS = {
    "simt_noun", "simteam_noun",
    "imi_este_present", "imi_era_imperfect", "mie_short",
    "am_noun_present", "aveam_noun_imperfect",
}

# Pre-compute normalized valid nouns
VALID_NOUNS_NORMALIZED = {normalize_text(n) for n in EMOTION_NOUNS_ONLY}


def is_valid_anchor(hit: dict) -> bool:
    """Check if a hit's nearest anchor is a valid pattern × word combination."""
    pattern = hit.get("nearest_anchor_pattern", "")
    if pattern not in NOUN_ONLY_PATTERNS:
        return True

    # Extract the noun from the anchor sentence (last word)
    anchor = hit.get("nearest_anchor", "")
    if not anchor:
        return False
    noun = normalize_text(anchor.split()[-1])
    return noun in VALID_NOUNS_NORMALIZED


def filter_results(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    min_confidence: float = 0.0,
):
    """Filter embedding results: remove bad anchors and apply threshold."""
    total_posts_in = 0
    total_posts_out = 0
    total_hits_in = 0
    total_hits_out = 0
    removed_bad_anchor = 0
    removed_low_conf = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)
            total_posts_in += 1
            total_hits_in += len(row["hits"])

            filtered_hits = []
            for hit in row["hits"]:
                if hit["confidence"] < min_confidence:
                    removed_low_conf += 1
                    continue
                if not is_valid_anchor(hit):
                    removed_bad_anchor += 1
                    continue
                filtered_hits.append(hit)

            if filtered_hits:
                row["hits"] = filtered_hits
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_posts_out += 1
                total_hits_out += len(filtered_hits)

    print(f"Input:  {total_posts_in:,} posts, {total_hits_in:,} hits")
    print(f"Output: {total_posts_out:,} posts, {total_hits_out:,} hits")
    print(f"")
    print(f"Removed:")
    print(f"  Bad anchor (noun-only + wrong noun): {removed_bad_anchor:,}")
    print(f"  Below confidence {min_confidence:.2f}: {removed_low_conf:,}")
    print(f"  Total removed: {removed_bad_anchor + removed_low_conf:,}")
    print(f"")
    print(f"Posts dropped (no remaining hits): {total_posts_in - total_posts_out:,}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter embedding extraction results")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence threshold (default: 0.0, no filtering)")
    args = parser.parse_args()

    filter_results(
        input_path=Path(args.input),
        output_path=Path(args.output),
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
