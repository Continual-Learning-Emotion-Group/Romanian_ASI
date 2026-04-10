"""Post-process benchmark into experiment-ready format.

Re-runs PatternMatcher on each candidate to extract the exact seed word
span via the original regex (trigger + seed word), rather than naive string
matching. Produces a clean JSONL with:
  - masked_text: text with seed word replaced by [MASK]
  - seed_word_start / seed_word_end: character offsets in text
  - Verified flag: whether the pattern re-fired on the stored text

Does NOT overwrite any existing files.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from pipeline.utils.text_utils import normalize_text
from pipeline.utils.pattern_matcher import (
    PatternMatcher, PATTERNS, MODIFIER_PATTERN, EMOTION_NOUNS_ONLY,
)
from pipeline.seed.enriched import build_enriched_seed

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "benchmark_ro_asi.jsonl"
OUTPUT_FILE = DATA_DIR / "benchmark_ro_asi_clean.jsonl"


def find_seed_span_in_original(text: str, normalized_text: str,
                               match_obj: re.Match) -> tuple[int, int] | None:
    """Find the seed word's character span in the original (non-normalized) text.

    The regex match is on normalized text. We use the seed word group's
    start/end positions on the normalized text to find the corresponding
    span in the original text. Since normalize_text preserves string length
    (ă→a, ș→s, ț→t, â→a, î→i are all 1:1 char replacements), the offsets
    transfer directly.
    """
    # Group 2 is the seed word in all patterns
    try:
        seed_start = match_obj.start(2)
        seed_end = match_obj.end(2)
    except IndexError:
        return None

    # Verify normalization preserves length
    if len(normalized_text) == len(text):
        return seed_start, seed_end

    # Fallback: if lengths differ (shouldn't happen with current normalize_text),
    # search near the expected position
    return None


def verify_length_preserving():
    """Check that normalize_text preserves string length."""
    tests = [
        "Mă simt fericit și recunoscător.",
        "Mi-e frică de întuneric.",
        "Sunt mulțumită de rezultat.",
    ]
    for t in tests:
        n = normalize_text(t)
        assert len(t) == len(n), f"normalize_text changes length: {len(t)} -> {len(n)} for: {t}"


def build_compiled_patterns(matcher: PatternMatcher) -> list:
    """Get compiled patterns from the matcher for re-matching."""
    return matcher.compiled_patterns


def process_candidate(candidate: dict, matcher: PatternMatcher) -> dict | None:
    """Re-match the candidate and extract precise seed word span.

    Returns a cleaned record or None if verification fails.
    """
    text = candidate["text"]
    pattern_name = candidate["pattern_used"]
    seed_norm = candidate["seed_word_normalized"]
    normalized = normalize_text(text)

    # Find the specific pattern that originally matched
    target_pattern = None
    for pname, category, compiled, noun_only in matcher.compiled_patterns:
        if pname == pattern_name:
            target_pattern = (pname, category, compiled, noun_only)
            break

    if target_pattern is None:
        return None

    pname, category, compiled, noun_only = target_pattern

    # Re-run the regex on normalized text
    for match in compiled.finditer(normalized):
        groups = match.groups()
        if len(groups) < 2:
            continue

        matched_seed = groups[1].lower()
        if matched_seed != seed_norm:
            continue

        # Found our match — extract seed word span
        span = find_seed_span_in_original(text, normalized, match)
        if span is None:
            continue

        seed_start, seed_end = span
        original_seed_word = text[seed_start:seed_end]

        # Build masked text
        masked_text = text[:seed_start] + "[MASK]" + text[seed_end:]

        return {
            "id": candidate["id"],
            "text": text,
            "masked_text": masked_text,
            "seed_word": original_seed_word,
            "seed_word_normalized": seed_norm,
            "seed_word_start": seed_start,
            "seed_word_end": seed_end,
            "pattern_used": pattern_name,
            "pattern_category": category,
            "emotion_category": candidate.get("emotion_category", []),
            "source": candidate.get("source", ""),
            "llm_affect_score": candidate.get("llm_affect_score"),
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Clean benchmark into experiment-ready format with verified spans"
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--sample", type=int, default=0,
                        help="Print N random samples after processing")
    args = parser.parse_args()

    if args.output.exists():
        print(f"ERROR: {args.output} already exists. Will not overwrite.")
        print("Delete it manually or choose a different output path with --output.")
        return

    # Verify normalization is length-preserving
    verify_length_preserving()

    # Build matcher with enriched seed
    seed_data = build_enriched_seed()
    # PatternMatcher expects {word: [list of emotions]}, but enriched seed
    # may return {word: "emotion_string"} — normalize to lists
    word_to_emotions = {}
    for word, emo in seed_data["word_to_affect_categ"].items():
        if isinstance(emo, str):
            word_to_emotions[word] = [emo]
        else:
            word_to_emotions[word] = list(emo)
    matcher = PatternMatcher(word_to_emotions)

    verified = 0
    failed = 0
    results = []

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            candidate = json.loads(line)
            cleaned = process_candidate(candidate, matcher)

            if cleaned is not None:
                fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                results.append(cleaned)
                verified += 1
            else:
                failed += 1

    total = verified + failed
    print(f"\n=== Benchmark Cleaning ===")
    print(f"Input: {args.input} ({total} candidates)")
    print(f"Verified (pattern re-fired): {verified} ({verified/total:.1%})")
    print(f"Failed (could not re-match): {failed} ({failed/total:.1%})")
    print(f"Output: {args.output}")

    # Write stats
    stats_path = args.output.with_suffix(".stats.json")
    if not stats_path.exists():
        source_dist = Counter(r["source"] for r in results)
        pattern_dist = Counter(r["pattern_category"] for r in results)
        seed_words = Counter(r["seed_word_normalized"] for r in results)
        stats = {
            "total_input": total,
            "verified": verified,
            "failed": failed,
            "verification_rate": round(verified / total, 4) if total else 0,
            "source_distribution": dict(sorted(source_dist.items())),
            "pattern_distribution": dict(sorted(pattern_dist.items())),
            "unique_seed_words": len(seed_words),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Stats: {stats_path}")

    # Show samples
    if args.sample and results:
        rng = random.Random(42)
        samples = rng.sample(results, min(args.sample, len(results)))
        print(f"\n=== {len(samples)} Random Samples ===")
        for i, s in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"  ID: {s['id']}")
            print(f"  Seed: \"{s['seed_word']}\" [{s['seed_word_start']}:{s['seed_word_end']}]")
            print(f"  Pattern: {s['pattern_used']}")
            print(f"  Emotion: {s['emotion_category']}")
            print(f"  Text: ...{s['text'][max(0,s['seed_word_start']-40):s['seed_word_end']+40]}...")
            print(f"  Masked: ...{s['masked_text'][max(0,s['seed_word_start']-40):s['seed_word_start']+45]}...")


if __name__ == "__main__":
    main()
