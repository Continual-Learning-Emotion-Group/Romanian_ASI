#!/usr/bin/env python3
"""
Phase 2: Filter raw Filmot API hits into ASI candidates.

Reads raw hits from Phase 1 (filmot_api_raw_hits.jsonl), applies
PatternMatcher from scripts/ro_asi/pattern_matcher.py, and outputs
validated ASI candidates.

Zero API calls — all processing is local.

Usage:
    python -m scripts.filmot_api.filter_candidates
    python -m scripts.filmot_api.filter_candidates --sample 20
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from scripts.ro_asi.load_roemolex import load_or_create_emotion_seed
from scripts.ro_asi.pattern_matcher import PatternMatcher

from .config import FilmotAPIConfig


def filter_raw_hits(
    config: FilmotAPIConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Filter raw hits with PatternMatcher and output ASI candidates.

    Args:
        config: Configuration with file paths.
        verbose: Print progress.

    Returns:
        Statistics dictionary.
    """
    # Load emotion seed
    print("Loading emotion seed...")
    emotion_seed = load_or_create_emotion_seed()

    word_to_emotions = emotion_seed.get("word_to_emotions", {})
    if not word_to_emotions:
        raise ValueError("No word_to_emotions in emotion seed")

    noun_words = emotion_seed.get("nouns", None)
    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words)

    # Check input file
    if not config.output_raw_path.exists():
        print(f"No raw hits file found at {config.output_raw_path}")
        print("Run Phase 1 first: python -m scripts.filmot_api.collect")
        return {}

    # Count input lines
    total_raw = 0
    with open(config.output_raw_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_raw += 1

    print(f"\nFilmot API Candidate Filtering")
    print("=" * 70)
    print(f"Input: {config.output_raw_path} ({total_raw:,} raw hits)")
    print(f"Output: {config.output_candidates_path}")
    print()

    # Stats
    stats = {
        "total_raw_hits": total_raw,
        "total_candidates": 0,
        "duplicates_skipped": 0,
        "no_match": 0,
        "by_pattern": defaultdict(int),
        "by_emotion": defaultdict(int),
        "by_category": defaultdict(int),
        "unique_seed_words": set(),
        "started_at": datetime.now().isoformat(),
    }

    seen_hashes = set()

    # Process raw hits
    config.output_candidates_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config.output_raw_path, "r", encoding="utf-8") as in_f, \
         open(config.output_candidates_path, "w", encoding="utf-8") as out_f:

        for i, line in enumerate(in_f):
            try:
                hit = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            full_context = hit.get("full_context", "")
            if not full_context:
                continue

            # Apply pattern matching
            matches = matcher.find_matches(
                full_context, extract_sentences=True, max_matches=10
            )

            if not matches:
                stats["no_match"] += 1
                continue

            for match in matches:
                # Dedup by matched text
                text_hash = hashlib.md5(match.matched_text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    stats["duplicates_skipped"] += 1
                    continue
                seen_hashes.add(text_hash)

                # Format candidate (compatible with filmot_asi_candidates.jsonl)
                video_id = hit.get("video_id", "")
                hit_start = hit.get("hit_start", 0)

                candidate = {
                    "id": f"filmot_api_{video_id}_{hit_start}",
                    "text": full_context,
                    "matched_sentence": match.matched_text,
                    "pattern_used": match.pattern_name,
                    "pattern_category": match.pattern_category,
                    "seed_word": match.seed_word,
                    "seed_word_normalized": match.seed_word_normalized,
                    "emotion_category": match.emotions,
                    "source": "filmot_api",
                    "video_id": video_id,
                    "video_title": hit.get("video_title", ""),
                    "channel": hit.get("channel_name", ""),
                    "views": hit.get("view_count", 0),
                    "duration_seconds": hit.get("duration", 0),
                    "upload_date": hit.get("upload_date", ""),
                    "youtube_url": hit.get("youtube_url", ""),
                }

                out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

                # Update stats
                stats["total_candidates"] += 1
                stats["by_pattern"][match.pattern_name] += 1
                stats["by_category"][match.pattern_category] += 1
                stats["unique_seed_words"].add(match.seed_word_normalized)

                for emotion in match.emotions:
                    stats["by_emotion"][emotion] += 1

            if verbose and (i + 1) % 10_000 == 0:
                print(f"  Processed {i + 1:,}/{total_raw:,} | "
                      f"Candidates: {stats['total_candidates']:,}")

    # Finalize stats
    stats["finished_at"] = datetime.now().isoformat()
    stats["unique_seed_words_count"] = len(stats["unique_seed_words"])
    stats["unique_seed_words"] = sorted(list(stats["unique_seed_words"]))
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])
    stats["by_category"] = dict(stats["by_category"])
    stats["match_rate"] = (
        stats["total_candidates"] / total_raw * 100 if total_raw > 0 else 0
    )

    # Save stats
    stats_path = config.output_candidates_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def print_stats(stats: Dict[str, Any]):
    """Print filtering statistics."""
    print(f"\n{'=' * 70}")
    print("Filmot API Filtering Statistics")
    print("=" * 70)
    print(f"Raw hits processed: {stats['total_raw_hits']:,}")
    print(f"ASI candidates: {stats['total_candidates']:,}")
    print(f"No pattern match: {stats['no_match']:,}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']:,}")
    print(f"Match rate: {stats['match_rate']:.1f}%")
    print(f"Unique seed words: {stats['unique_seed_words_count']}")

    print("\nCandidates by pattern:")
    for pattern, count in sorted(stats["by_pattern"].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count:,}")

    print("\nCandidates by emotion:")
    for emotion, count in sorted(stats["by_emotion"].items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count:,}")

    print("\nCandidates by category:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")

    if stats.get("unique_seed_words"):
        sample = stats["unique_seed_words"][:20]
        print(f"\nSample seed words: {', '.join(sample)}")


def sample_candidates(output_path: Path, n: int = 10):
    """Print sample candidates for verification."""
    if not output_path.exists():
        print(f"\nNo output file found at {output_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"Sample Candidates (first {n})")
    print("=" * 70)

    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            candidate = json.loads(line)
            print(f"\n[{i + 1}] ID: {candidate['id']}")
            print(f"    Video: {candidate.get('video_id', 'N/A')}")
            print(f"    Channel: {candidate.get('channel', 'N/A')}")
            print(f"    Pattern: {candidate['pattern_used']} ({candidate['pattern_category']})")
            print(f"    Seed word: {candidate['seed_word']} → {candidate['emotion_category']}")
            matched = candidate["matched_sentence"]
            if len(matched) > 100:
                print(f'    Matched: "{matched[:100]}..."')
            else:
                print(f'    Matched: "{matched}"')
            context = candidate.get("text", "")
            if len(context) > 120:
                print(f'    Context: "{context[:120]}..."')
            else:
                print(f'    Context: "{context}"')


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw Filmot API hits into ASI candidates"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample candidates to print (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    config = FilmotAPIConfig()

    stats = filter_raw_hits(config=config, verbose=not args.quiet)

    if not stats:
        return 1

    print_stats(stats)

    if args.sample > 0 and stats["total_candidates"] > 0:
        sample_candidates(config.output_candidates_path, args.sample)

    print(f"\nOutput: {config.output_candidates_path}")
    return 0


if __name__ == "__main__":
    exit(main())
