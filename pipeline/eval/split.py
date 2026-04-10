"""Create train / test / unseen-challenge splits for evaluation.

Following MASIVE methodology:
  - Hold out ~15 seed words entirely -> unseen challenge set
  - Remaining: 90/10 train/test stratified by source

Usage:
    python -m pipeline.eval.split
    python -m pipeline.eval.split --n-unseen 15 --test-frac 0.1 --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "benchmark_ro_asi_clean.jsonl"
SPLITS_DIR = DATA_DIR / "splits"


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def select_unseen_words(
    word_freq: dict[str, int],
    word_emotions: dict[str, set[str]],
    n_unseen: int = 15,
    seed: int = 42,
    min_freq: int = 5,
) -> list[str]:
    """Select n_unseen words to hold out, covering diverse emotion categories
    and frequency bands.

    Strategy:
    1. Bin words into frequency bands: low (5-50), medium (51-500), high (501+)
    2. Aim for ~5 words per band
    3. Within each band, greedily select words that maximise emotion category
       coverage (words whose emotion set adds the most new categories)
    """
    rng = random.Random(seed)

    eligible = {w for w, f in word_freq.items() if f >= min_freq}

    # Bin by frequency
    bins: dict[str, list[str]] = {"low": [], "medium": [], "high": []}
    for w in eligible:
        f = word_freq[w]
        if f <= 50:
            bins["low"].append(w)
        elif f <= 500:
            bins["medium"].append(w)
        else:
            bins["high"].append(w)

    # Shuffle within each bin for reproducibility
    for b in bins.values():
        rng.shuffle(b)

    per_band = n_unseen // 3
    remainder = n_unseen - per_band * 3

    selected = []
    for i, (band_name, candidates) in enumerate(bins.items()):
        target = per_band + (1 if i < remainder else 0)
        covered_emotions: set[str] = set()
        band_selected: list[str] = []

        # Greedy: pick word that adds the most new emotion categories
        remaining = list(candidates)
        while len(band_selected) < target and remaining:
            best_word = None
            best_new = -1
            for w in remaining:
                emos = word_emotions.get(w, set())
                new_count = len(emos - covered_emotions)
                if new_count > best_new:
                    best_new = new_count
                    best_word = w

            if best_word is None:
                break

            band_selected.append(best_word)
            covered_emotions |= word_emotions.get(best_word, set())
            remaining.remove(best_word)

        selected.extend(band_selected)

    return sorted(selected)


def split_data(
    records: list[dict],
    unseen_words: list[str],
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into (train, test, unseen)."""
    unseen_set = set(unseen_words)

    unseen = [r for r in records if r["seed_word_normalized"] in unseen_set]
    remaining = [r for r in records if r["seed_word_normalized"] not in unseen_set]

    sources = [r["source"] for r in remaining]
    train, test = train_test_split(
        remaining, test_size=test_frac, random_state=seed, stratify=sources,
    )

    return train, test, unseen


def write_split(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test/unseen splits for ASI evaluation"
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--n-unseen", type=int, default=15)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_records(args.input)
    print(f"  Total records: {len(records)}")

    # Compute word statistics
    word_freq: dict[str, int] = Counter(
        r["seed_word_normalized"] for r in records
    )
    word_emotions: dict[str, set[str]] = defaultdict(set)
    for r in records:
        for emo in r.get("emotion_category", []):
            word_emotions[r["seed_word_normalized"]].add(emo)

    # Select unseen words
    unseen_words = select_unseen_words(
        word_freq, word_emotions,
        n_unseen=args.n_unseen, seed=args.seed,
    )

    # Split
    train, test, unseen = split_data(
        records, unseen_words,
        test_frac=args.test_frac, seed=args.seed,
    )

    # Write
    write_split(train, args.output_dir / "train.jsonl")
    write_split(test, args.output_dir / "test.jsonl")
    write_split(unseen, args.output_dir / "unseen.jsonl")

    # Stats
    stats = {
        "total": len(records),
        "train": len(train),
        "test": len(test),
        "unseen": len(unseen),
        "unseen_words": unseen_words,
        "unseen_word_freqs": {w: word_freq[w] for w in unseen_words},
        "unseen_emotion_coverage": sorted(
            set().union(*(word_emotions.get(w, set()) for w in unseen_words))
        ),
        "source_distribution": {
            split_name: dict(Counter(r["source"] for r in split_records))
            for split_name, split_records in [
                ("train", train), ("test", test), ("unseen", unseen),
            ]
        },
        "unique_seed_words": {
            "train": len(set(r["seed_word_normalized"] for r in train)),
            "test": len(set(r["seed_word_normalized"] for r in test)),
            "unseen": len(set(r["seed_word_normalized"] for r in unseen)),
        },
    }
    stats_path = args.output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n=== Split Summary ===")
    print(f"Train:  {len(train):>6}")
    print(f"Test:   {len(test):>6}")
    print(f"Unseen: {len(unseen):>6}")
    print(f"Total:  {len(train) + len(test) + len(unseen):>6}")
    print(f"\nUnseen words ({len(unseen_words)}):")
    for w in unseen_words:
        emos = sorted(word_emotions.get(w, set()))
        print(f"  {w:20s}  freq={word_freq[w]:>5}  emotions={emos}")
    print(f"\nEmotion coverage in unseen: {stats['unseen_emotion_coverage']}")
    print(f"\nSource distribution:")
    for split_name, dist in stats["source_distribution"].items():
        print(f"  {split_name}: {dist}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
