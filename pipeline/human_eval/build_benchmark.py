"""Build the final Romanian ASI benchmark from LLM-validated candidates.

Filters candidates_validated.jsonl to only those with llm_affect_score >= 3,
based on human evaluation results:
  - LLM >= 3 gives 91.3% precision against human annotations (mean >= 2.0)
  - Comparable to MASIVE methodology quality thresholds
  - Produces ~73K high-confidence affective state candidates

Does NOT overwrite any existing files — writes to a new benchmark file.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "candidates_validated.jsonl"
OUTPUT_FILE = DATA_DIR / "benchmark_ro_asi.jsonl"

LLM_THRESHOLD = 3


def build_benchmark(input_path: Path, output_path: Path, threshold: int):
    if output_path.exists():
        print(f"ERROR: {output_path} already exists. Will not overwrite.")
        print("Delete it manually or choose a different output path with --output.")
        return

    kept = 0
    skipped = 0
    score_dist = Counter()
    source_dist = Counter()
    pattern_dist = Counter()
    seed_words = Counter()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            candidate = json.loads(line)
            score = candidate.get("llm_affect_score")

            if score is not None and score >= threshold:
                fout.write(line)
                kept += 1
                score_dist[score] += 1
                source_dist[candidate.get("source", "unknown")] += 1
                pattern_dist[candidate.get("pattern_category", "unknown")] += 1
                seed_words[candidate.get("seed_word", "unknown")] += 1
            else:
                skipped += 1

    # Write stats alongside
    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "llm_threshold": threshold,
        "total_input": kept + skipped,
        "kept": kept,
        "skipped": skipped,
        "score_distribution": dict(sorted(score_dist.items())),
        "source_distribution": dict(sorted(source_dist.items())),
        "pattern_distribution": dict(sorted(pattern_dist.items())),
        "unique_seed_words": len(seed_words),
        "top_seed_words": dict(seed_words.most_common(20)),
    }

    stats_path = output_path.with_suffix(".stats.json")
    if not stats_path.exists():
        with open(stats_path, "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Stats written to {stats_path}")

    print(f"\n=== Benchmark Construction ===")
    print(f"Input: {input_path} ({kept + skipped} candidates)")
    print(f"Threshold: llm_affect_score >= {threshold}")
    print(f"Kept: {kept} ({kept/(kept+skipped):.1%})")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_path}")
    print(f"\nSource distribution:")
    for src, n in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")
    print(f"\nPattern distribution:")
    for pat, n in sorted(pattern_dist.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {n}")
    print(f"\nUnique seed words: {len(seed_words)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build final Romanian ASI benchmark (LLM score >= threshold)"
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE,
                        help="Validated candidates JSONL")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Output benchmark JSONL (must not exist)")
    parser.add_argument("--threshold", type=int, default=LLM_THRESHOLD,
                        help="Minimum LLM affect score to include (default: 3)")
    args = parser.parse_args()

    build_benchmark(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
