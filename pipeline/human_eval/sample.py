"""Stratified sampling from validated candidates for human evaluation."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "candidates_validated_partial.jsonl"
OUTPUT_FILE = DATA_DIR / "human_eval_sample.jsonl"

TOTAL_SAMPLES = 200
SAMPLES_PER_SCORE = 50  # 50 per LLM score bin (0, 1, 2, 3)


def load_candidates(path: Path) -> list[dict]:
    candidates = []
    with open(path) as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates


def stratified_sample(candidates: list[dict], seed: int = 42) -> list[dict]:
    """Sample SAMPLES_PER_SCORE candidates per LLM score bin.

    Within each bin, sample proportionally by source.
    """
    rng = random.Random(seed)

    # Group by LLM score
    by_score: dict[int, list[dict]] = defaultdict(list)
    for c in candidates:
        score = c.get("llm_affect_score")
        if score is not None:
            by_score[score].append(c)

    sampled = []
    for score in sorted(by_score):
        pool = by_score[score]
        n = min(SAMPLES_PER_SCORE, len(pool))

        # Secondary stratification by source (proportional)
        by_source: dict[str, list[dict]] = defaultdict(list)
        for c in pool:
            by_source[c.get("source", "unknown")].append(c)

        # Compute proportional allocation per source
        source_counts = {}
        remaining = n
        sources = sorted(by_source.keys())
        for i, src in enumerate(sources):
            if i == len(sources) - 1:
                source_counts[src] = remaining
            else:
                alloc = round(n * len(by_source[src]) / len(pool))
                alloc = min(alloc, len(by_source[src]), remaining)
                alloc = max(alloc, 1) if remaining > 0 else 0
                source_counts[src] = alloc
                remaining -= alloc

        score_sampled = []
        for src in sources:
            k = min(source_counts[src], len(by_source[src]))
            score_sampled.extend(rng.sample(by_source[src], k))

        sampled.extend(score_sampled)

    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Sample candidates for human eval")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLES)
    args = parser.parse_args()

    global SAMPLES_PER_SCORE
    SAMPLES_PER_SCORE = args.total // 4

    candidates = load_candidates(args.input)
    print(f"Loaded {len(candidates)} candidates from {args.input}")

    sampled = stratified_sample(candidates, seed=args.seed)
    print(f"Sampled {len(sampled)} candidates")

    # Print distribution
    from collections import Counter
    score_dist = Counter(c["llm_affect_score"] for c in sampled)
    source_dist = Counter(c["source"] for c in sampled)
    pattern_dist = Counter(c["pattern_category"] for c in sampled)
    print(f"\nScore distribution: {dict(sorted(score_dist.items()))}")
    print(f"Source distribution: {dict(sorted(source_dist.items()))}")
    print(f"Pattern distribution: {dict(sorted(pattern_dist.items()))}")

    with open(args.output, "w") as f:
        for c in sampled:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
