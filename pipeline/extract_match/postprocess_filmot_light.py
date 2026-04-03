#!/usr/bin/env python3
"""
Light post-processing for filmot candidates.

No ML models — just text cleanup:
  1. Remove #tags# (music/sound markers)
  2. Split on >> (speaker change markers) — keep segment with match
  3. Add periods at sentence boundaries (capitalization-based)
  4. Remove leading fragment if first word is not capitalized
     (incomplete sentence from filmot context window)

Writes to a separate file (does NOT overwrite the original).

Usage:
    python -m pipeline.extract_match.postprocess_filmot_light
    python -m pipeline.extract_match.postprocess_filmot_light --max-records 500
"""

import argparse
import json
import re
from pathlib import Path
from typing import List

from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "pattern_candidates_filmot.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "pattern_candidates_filmot_light.jsonl"


def find_match_segment(segments: List[str], seed_word: str) -> int:
    """Find which segment contains the seed word."""
    seed_lower = seed_word.lower()
    for i, seg in enumerate(segments):
        if seed_lower in seg.lower():
            return i
    return 0


def add_periods(text: str) -> str:
    """
    Insert periods at sentence boundaries detected by capitalization.

    Only splits when the preceding word starts with a lowercase letter
    and the next word starts with uppercase. This avoids breaking
    proper noun sequences like "Duke Denis" or "Noul Coldrex Hotrem".
    """
    words = text.split()
    if not words:
        return text

    result = [words[0]]
    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]

        # Previous word starts lowercase, current word starts uppercase
        # → likely a sentence boundary, UNLESS previous word is a
        # preposition/article/conjunction (commonly precedes proper nouns)
        _NO_SPLIT_AFTER = {
            'la', 'în', 'din', 'de', 'pe', 'cu', 'și', 'sau',
            'lui', 'al', 'a', 'ai', 'ale',
            'prin', 'spre', 'sub', 'despre', 'între', 'peste',
            'lângă', 'contra', 'către', 'după', 'fără',
            'the', 'and', 'of', 'from', 'for', 'with', 'by', 'at',
        }
        prev_starts_lower = prev_word[0].islower() if prev_word else False
        curr_starts_upper = curr_word[0].isupper() if curr_word else False
        prev_is_prep = prev_word.lower().rstrip('.,!?') in _NO_SPLIT_AFTER

        if prev_starts_lower and curr_starts_upper and not prev_is_prep:
            # End previous sentence with period
            if not result[-1].endswith(('.', '?', '!')):
                result[-1] = result[-1].rstrip('., ') + '.'
            result.append(curr_word)
        else:
            result.append(curr_word)

    text = ' '.join(result)
    if not text.rstrip().endswith(('.', '?', '!')):
        text = text.rstrip() + '.'
    return text


def process_one(text: str, seed_word: str) -> str:
    """Apply light post-processing to one filmot text."""
    # 1. Remove #tags#
    text = re.sub(r'#[^#]+#', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Split on >> — keep segment with the match
    if '>>' in text:
        segments = [s.strip() for s in text.split('>>') if s.strip()]
        if segments:
            idx = find_match_segment(segments, seed_word)
            text = segments[idx]

    # 3. Add periods at capitalization boundaries
    text = add_periods(text)

    # 4. Remove leading fragment if first word is not capitalized
    #    (incomplete sentence carried over from filmot context window)
    sentences = [s.strip() for s in re.split(r'(?<=\.)\s+', text) if s.strip()]
    if sentences and sentences[0] and not sentences[0][0].isupper():
        sentences = sentences[1:]

    if not sentences:
        # Fallback: return the text with periods added
        return text

    result = ' '.join(sentences)
    if not result.endswith(('.', '?', '!')):
        result += '.'
    return result


def postprocess_light(
    input_path: Path = None,
    output_path: Path = None,
    max_records: int = 0,
    verbose: bool = True,
) -> dict:
    if input_path is None:
        input_path = DEFAULT_INPUT
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    candidates = []
    bad_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                candidates.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                bad_lines += 1

    if max_records > 0:
        candidates = candidates[:max_records]

    if verbose:
        print(f"Loaded {len(candidates)} candidates from {input_path}")
        if bad_lines:
            print(f"  ({bad_lines} bad lines skipped)")

    stats = {
        "total": len(candidates),
        "had_speaker_change": 0,
        "had_leading_fragment": 0,
        "avg_chars_before": 0,
        "avg_chars_after": 0,
    }

    total_chars_before = 0
    total_chars_after = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        pbar = tqdm(candidates, desc="Light post-processing", unit="rec") if verbose else candidates
        for cand in pbar:
            original_text = cand["text"]
            seed_word = cand["seed_word"]

            if '>>' in original_text:
                stats["had_speaker_change"] += 1

            first_char = original_text.strip()[0] if original_text.strip() else ''
            if first_char and not first_char.isupper():
                stats["had_leading_fragment"] += 1

            processed = process_one(original_text, seed_word)

            cand["text_light"] = processed
            cand["text_original"] = original_text

            total_chars_before += len(original_text)
            total_chars_after += len(processed)

            out_f.write(json.dumps(cand, ensure_ascii=False) + "\n")

    n = len(candidates)
    stats["avg_chars_before"] = round(total_chars_before / max(n, 1), 0)
    stats["avg_chars_after"] = round(total_chars_after / max(n, 1), 0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Light post-processing complete")
        print(f"{'='*60}")
        print(f"Total: {stats['total']}")
        print(f"Had >> (speaker change): {stats['had_speaker_change']} ({stats['had_speaker_change']/max(n,1)*100:.1f}%)")
        print(f"Had leading fragment: {stats['had_leading_fragment']} ({stats['had_leading_fragment']/max(n,1)*100:.1f}%)")
        print(f"Avg chars: {stats['avg_chars_before']:.0f} → {stats['avg_chars_after']:.0f}")
        print(f"Output: {output_path}")

    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Light post-processing for filmot candidates (no ML)"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help=f"Input JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Process only first N records (0 = all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    postprocess_light(
        input_path=args.input,
        output_path=args.output,
        max_records=args.max_records,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
