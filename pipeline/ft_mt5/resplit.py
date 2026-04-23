"""Regenerate train/val/test splits for mT5 fine-tuning.

Split strategy (per PI suggestion):
- Test (~10%): sample whole seed_word_normalized groups until cumulative size >= 10%.
  No seed word in test appears in train or val -> "unseen vocabulary" evaluation.
- Val (5%): uniformly sampled at the row level from what remains. Seed words may
  overlap with train; used for checkpoint selection.
- Train (~85%): everything else.

Input:  pipeline/data/benchmark_ro_asi_clean.jsonl (deduped by (id, seed_word_normalized))
Output: pipeline/data/splits/{train,val,test}.jsonl  (overwrites existing files)
        pipeline/data/splits/split_stats.json

Usage:
    python -m pipeline.ft_mt5.resplit
    python -m pipeline.ft_mt5.resplit --seed 42 --test-frac 0.10 --val-frac 0.05
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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


def sample_test_by_seed_word(
    records: list[dict], target_frac: float, rng: random.Random
) -> tuple[list[dict], list[dict], set[str]]:
    """Pick whole seed_word_normalized groups until test >= target_frac of records.

    Returns (test_records, remaining_records, test_seed_words).
    """
    target_size = int(round(target_frac * len(records)))

    # Group by normalized seed word
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["seed_word_normalized"]].append(r)

    # Shuffle the list of seed words, then consume until we hit target
    seed_words = list(groups.keys())
    rng.shuffle(seed_words)

    test_records: list[dict] = []
    test_seed_words: set[str] = set()
    for sw in seed_words:
        if len(test_records) >= target_size:
            break
        test_records.extend(groups[sw])
        test_seed_words.add(sw)

    remaining = [r for r in records if r["seed_word_normalized"] not in test_seed_words]
    return test_records, remaining, test_seed_words


def main():
    parser = argparse.ArgumentParser(description="Regenerate train/val/test splits")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of ORIGINAL deduped size to use for val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading {args.input} ...")
    records = load_and_dedup(args.input)
    total = len(records)
    print(f"  Records after dedup: {total}")

    # 1. Test: whole seed-word groups until >= test_frac
    test, remaining, test_sws = sample_test_by_seed_word(records, args.test_frac, rng)
    print(f"  Test: {len(test)} rows ({len(test)/total:.1%}), "
          f"{len(test_sws)} seed words held out")

    # 2. Val: uniform row-level sample of val_frac * total from remaining
    val_size = int(round(args.val_frac * total))
    rng.shuffle(remaining)
    val = remaining[:val_size]
    train = remaining[val_size:]
    print(f"  Val:   {len(val)} rows ({len(val)/total:.1%})")
    print(f"  Train: {len(train)} rows ({len(train)/total:.1%})")

    # Write splits (deterministic order within each split: id)
    train.sort(key=lambda r: (r["id"], r["seed_word_normalized"]))
    val.sort(key=lambda r: (r["id"], r["seed_word_normalized"]))
    test.sort(key=lambda r: (r["id"], r["seed_word_normalized"]))

    write_split(train, args.output_dir / "train.jsonl")
    write_split(val,   args.output_dir / "val.jsonl")
    write_split(test,  args.output_dir / "test.jsonl")

    # Stats
    train_sws = {r["seed_word_normalized"] for r in train}
    val_sws   = {r["seed_word_normalized"] for r in val}
    assert not (train_sws & test_sws), "Train/test seed-word overlap - bug"
    assert not (val_sws   & test_sws), "Val/test seed-word overlap - bug"

    stats = {
        "seed": args.seed,
        "total_deduped": total,
        "train": {
            "rows": len(train),
            "frac": len(train) / total,
            "unique_seed_words": len(train_sws),
            "source_distribution": dict(Counter(r["source"] for r in train)),
        },
        "val": {
            "rows": len(val),
            "frac": len(val) / total,
            "unique_seed_words": len(val_sws),
            "source_distribution": dict(Counter(r["source"] for r in val)),
        },
        "test": {
            "rows": len(test),
            "frac": len(test) / total,
            "unique_seed_words": len(test_sws),
            "source_distribution": dict(Counter(r["source"] for r in test)),
        },
        "test_seed_words": sorted(test_sws),
        "val_seed_words_overlap_train": len(val_sws & train_sws),
    }
    with open(args.output_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== Split Summary ===")
    print(f"Train: {len(train):>6} rows, {len(train_sws)} seed words")
    print(f"Val:   {len(val):>6} rows, {len(val_sws)} seed words "
          f"({len(val_sws & train_sws)} overlap with train)")
    print(f"Test:  {len(test):>6} rows, {len(test_sws)} seed words (disjoint from train/val)")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
