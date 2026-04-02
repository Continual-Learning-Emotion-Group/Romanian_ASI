#!/usr/bin/env python3
"""
Stream raw records from the FULG dataset (150B tokens, 289GB).

No filtering — just streams N records from HuggingFace and saves them as JSONL.
This is the raw collection step; filtering happens in a later pipeline stage.

Usage:
    python -m pipeline.collect.stream_fulg
    python -m pipeline.collect.stream_fulg --max-records 100000
    python -m pipeline.collect.stream_fulg --min-language-score 0.8
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

DATASET_ID = "faur-ai/fulg"


def stream_fulg(
    output_path: Path,
    max_records: int = 50_000,
    min_language_score: float = 0.0,
    min_text_length: int = 0,
    verbose: bool = True,
):
    """
    Stream FULG records and save as JSONL.

    Args:
        output_path: Where to write the JSONL output.
        max_records: Stop after this many saved records.
        min_language_score: Only keep records above this Romanian confidence.
        min_text_length: Only keep records with text longer than this.
        verbose: Print progress.
    """
    from datasets import load_dataset

    print(f"Streaming from {DATASET_ID}...")
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

    print(f"\nDone: {saved:,} records saved, {skipped:,} skipped")
    print(f"Output: {output_path}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Stream raw FULG records (no filtering)"
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
        "--output", type=Path, default=None,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    output = args.output or DATA_DIR / "fulg_raw.jsonl"

    stream_fulg(
        output_path=output,
        max_records=args.max_records,
        min_language_score=args.min_language_score,
        min_text_length=args.min_text_length,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
