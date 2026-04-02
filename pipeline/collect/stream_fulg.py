#!/usr/bin/env python3
"""
Stream raw records from the FULG dataset (150B tokens, 289GB).

Saves JSONL records from HuggingFace streaming. Optionally filters by trigger
words (verb stems from the pattern matcher) to keep only records likely to
contain "I feel" patterns.

Usage:
    python -m pipeline.collect.stream_fulg
    python -m pipeline.collect.stream_fulg --max-records 100000
    python -m pipeline.collect.stream_fulg --no-trigger-filter
    python -m pipeline.collect.stream_fulg --min-language-score 0.8
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Set

DATA_DIR = Path(__file__).parent.parent / "data"

DATASET_ID = "faur-ai/fulg"


def stream_fulg(
    output_path: Path,
    max_records: int = 50_000,
    min_language_score: float = 0.0,
    min_text_length: int = 0,
    trigger_words: Optional[Set[str]] = None,
    verbose: bool = True,
):
    """
    Stream FULG records and save as JSONL.

    Args:
        output_path: Where to write the JSONL output.
        max_records: Stop after this many saved records.
        min_language_score: Only keep records above this Romanian confidence.
        min_text_length: Only keep records with text longer than this.
        trigger_words: If given, only keep records containing at least one
                       of these words/phrases (case-insensitive).
        verbose: Print progress.
    """
    from datasets import load_dataset

    print(f"Streaming from {DATASET_ID}...")
    if trigger_words:
        print(f"  Trigger filter: {len(trigger_words)} words/phrases")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(ds):
            text = record.get("raw_content", "")
            lang_score = record.get("language_score", 0)

            if min_language_score > 0 and lang_score < min_language_score:
                skipped += 1
                continue

            if min_text_length > 0 and len(text) < min_text_length:
                skipped += 1
                continue

            # Trigger word filter
            if trigger_words:
                text_lower = text.lower()
                if not any(tw in text_lower for tw in trigger_words):
                    skipped += 1
                    continue

            row = {
                "id": f"fulg_{record.get('digest', '')[:12]}",
                "text": text,
                "source": "fulg",
                "url": record.get("url", ""),
                "title": record.get("title", ""),
                "source_domain": record.get("source_domain", ""),
                "language_score": lang_score,
                "length": record.get("length", 0),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            saved += 1

            if verbose and saved % 10_000 == 0:
                print(f"  Saved {saved:,} records (skipped {skipped:,}, scanned {i + 1:,})")

            if saved >= max_records:
                break

    print(f"\nDone: {saved:,} records saved, {skipped:,} skipped (scanned {i + 1:,})")
    print(f"Output: {output_path}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Stream FULG records with optional trigger word filter"
    )
    parser.add_argument(
        "--max-records", type=int, default=50_000,
        help="Number of records to save (default: 50000)",
    )
    parser.add_argument(
        "--min-language-score", type=float, default=0.0,
        help="Minimum Romanian language score, 0 = no filter (default: 0)",
    )
    parser.add_argument(
        "--min-text-length", type=int, default=0,
        help="Minimum text length in chars, 0 = no filter (default: 0)",
    )
    parser.add_argument(
        "--no-trigger-filter", action="store_true",
        help="Disable trigger word filter (saves all records)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    output = args.output or DATA_DIR / "fulg_raw.jsonl"

    # Load trigger words from pattern matcher (unless disabled)
    trigger_words = None
    if not args.no_trigger_filter:
        from pipeline.utils.pattern_matcher import get_trigger_words
        trigger_words = get_trigger_words()

    stream_fulg(
        output_path=output,
        max_records=args.max_records,
        min_language_score=args.min_language_score,
        min_text_length=args.min_text_length,
        trigger_words=trigger_words,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
