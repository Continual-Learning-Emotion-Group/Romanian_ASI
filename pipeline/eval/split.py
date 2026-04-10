"""Create train / test splits for evaluation.

Deduplicates the benchmark first (same text + same seed word = duplicate),
then does a stratified train/test split so test is representative of train
(same seed words appear in both, proportional source distribution).

Usage:
    python -m pipeline.eval.split
    python -m pipeline.eval.split --test-frac 0.05 --seed 42
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "benchmark_ro_asi_clean.jsonl"
SPLITS_DIR = DATA_DIR / "splits"


def load_and_dedup(path: Path) -> list[dict]:
    """Load records and deduplicate by (id, seed_word_normalized)."""
    records = []
    seen = set()
    dupes = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["id"], r["seed_word_normalized"])
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            records.append(r)
    if dupes:
        print(f"  Removed {dupes} duplicates")
    return records


def write_split(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test splits for ASI evaluation"
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_and_dedup(args.input)
    print(f"  Records after dedup: {len(records)}")

    # Stratify by source to keep distribution representative
    sources = [r["source"] for r in records]
    train, test = train_test_split(
        records, test_size=args.test_frac,
        random_state=args.seed, stratify=sources,
    )

    write_split(train, args.output_dir / "train.jsonl")
    write_split(test, args.output_dir / "test.jsonl")

    # Stats
    train_words = set(r["seed_word_normalized"] for r in train)
    test_words = set(r["seed_word_normalized"] for r in test)
    stats = {
        "total_raw": 70289,
        "total_deduped": len(records),
        "train": len(train),
        "test": len(test),
        "test_frac": args.test_frac,
        "seed": args.seed,
        "unique_seed_words": {
            "train": len(train_words),
            "test": len(test_words),
            "shared": len(train_words & test_words),
        },
        "source_distribution": {
            name: dict(Counter(r["source"] for r in split))
            for name, split in [("train", train), ("test", test)]
        },
    }
    with open(args.output_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== Split Summary ===")
    print(f"Train:  {len(train):>6}  ({len(train_words)} seed words)")
    print(f"Test:   {len(test):>6}  ({len(test_words)} seed words)")
    print(f"Shared seed words: {len(train_words & test_words)}")
    print(f"\nSource distribution:")
    for name, dist in stats["source_distribution"].items():
        print(f"  {name}: {dist}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
