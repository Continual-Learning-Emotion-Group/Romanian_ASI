#!/usr/bin/env python3
"""
Pattern-based ASI candidate extraction.

Uses the enriched seed (524 words) + 20 "I feel" regex patterns to extract
affective state candidates from Romanian text corpora.

Each candidate = one pattern match in one text. A single text can produce
multiple candidates if different patterns fire.

Usage:
    python -m pipeline.extract_match.run                          # small datasets
    python -m pipeline.extract_match.run --max-records 1000       # quick test
    python -m pipeline.extract_match.run --sample 10              # show sample matches
    python -m pipeline.extract_match.run --source filmot_raw      # specific JSONL source
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from pipeline.utils.pattern_matcher import PatternMatcher
from pipeline.utils.corpus_reader import iter_corpus

DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_SEED_PATH = DATA_DIR / "enriched_seed_merged.json"
DEFAULT_OUTPUT = DATA_DIR / "pattern_candidates_small.jsonl"


def load_enriched_seed(path: Path = None) -> dict:
    """
    Load enriched seed and convert to PatternMatcher-compatible format.

    The seed file maps word -> emotion_string. PatternMatcher expects
    word -> [list of emotions]. We also need a separate noun list.
    """
    if path is None:
        path = ENRICHED_SEED_PATH

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Build word_to_emotions: word -> [emotion_string]
    word_to_emotions = {}
    for word, emotion in raw.get("adjectives", {}).items():
        word_to_emotions[word] = [emotion] if isinstance(emotion, str) else emotion
    for word, emotion in raw.get("nouns", {}).items():
        word_to_emotions[word] = [emotion] if isinstance(emotion, str) else emotion
    for word, emotion in raw.get("adverbs", {}).items():
        word_to_emotions[word] = [emotion] if isinstance(emotion, str) else emotion

    # Noun list for noun-only patterns
    noun_words = list(raw.get("nouns", {}).keys())

    stats = raw.get("stats", {})
    print(f"Enriched seed loaded: {stats.get('adjectives', '?')} adj + "
          f"{stats.get('nouns', '?')} nouns + {stats.get('adverbs', '?')} adv "
          f"= {stats.get('total', len(word_to_emotions))} words")

    return word_to_emotions, noun_words


def count_jsonl_lines(path: Path) -> int:
    """Count lines in a JSONL file for progress bar."""
    count = 0
    with open(path, "r") as f:
        for _ in f:
            count += 1
    return count


def extract_candidates(
    sources: list[str] = None,
    output_path: Path = None,
    seed_path: Path = None,
    max_records: int = 0,
    sample: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Extract pattern-matched ASI candidates from JSONL corpus files.

    Args:
        sources: JSONL file stems to read (default: ["merged_corpus"]).
        output_path: Where to write candidates JSONL.
        seed_path: Path to enriched seed JSON.
        max_records: Stop after this many records (0 = unlimited).
        sample: Print this many sample candidates at the end.
        verbose: Print progress info.

    Returns:
        Stats dict.
    """
    if sources is None:
        sources = ["merged_corpus"]
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    # Load seed and init matcher
    word_to_emotions, noun_words = load_enriched_seed(seed_path)
    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words, expand_forms=True)

    # Count total lines for progress bar
    total_lines = 0
    for source in sources:
        fpath = DATA_DIR / f"{source}.jsonl"
        if fpath.exists():
            total_lines += count_jsonl_lines(fpath)
    if max_records > 0:
        total_lines = min(total_lines, max_records)

    # Stats
    seen_hashes = set()
    total_processed = 0
    total_dupes = 0
    total_matches = 0
    texts_with_matches = 0
    by_source = defaultdict(int)
    by_pattern = defaultdict(int)
    by_emotion = defaultdict(int)
    unique_seed_words = set()
    sample_candidates = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=total_lines, desc="Extracting", unit="rec") if verbose else None

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record_id, text, source_name in iter_corpus(DATA_DIR, sources=sources):
            if max_records > 0 and total_processed >= max_records:
                break

            # Deduplicate by text hash
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                total_dupes += 1
                if pbar:
                    pbar.update(1)
                continue
            seen_hashes.add(text_hash)
            total_processed += 1

            if pbar:
                pbar.update(1)
                pbar.set_postfix(matches=total_matches, texts_hit=texts_with_matches)

            # Find matches
            matches = matcher.find_matches(text, extract_sentences=True)
            if not matches:
                continue

            texts_with_matches += 1

            for match in matches:
                candidate = {
                    "id": record_id,
                    "text": text,
                    "matched_sentence": match.matched_text,
                    "pattern_used": match.pattern_name,
                    "pattern_category": match.pattern_category,
                    "seed_word": match.seed_word,
                    "seed_word_normalized": match.seed_word_normalized,
                    "emotion_category": match.emotions,
                    "source": source_name,
                }

                out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                total_matches += 1

                # Track stats
                by_source[source_name] += 1
                by_pattern[match.pattern_name] += 1
                for e in match.emotions:
                    by_emotion[e] += 1
                unique_seed_words.add(match.seed_word_normalized)

                if len(sample_candidates) < sample:
                    sample_candidates.append(candidate)

    if pbar:
        pbar.close()

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Extraction complete")
        print(f"{'='*60}")
        print(f"Records processed: {total_processed:,} (dupes skipped: {total_dupes:,})")
        print(f"Texts with matches: {texts_with_matches:,} ({texts_with_matches/max(total_processed,1)*100:.1f}%)")
        print(f"Total candidates: {total_matches:,}")
        print(f"Unique seed words matched: {len(unique_seed_words)}")
        print(f"Output: {output_path}")

        print(f"\nBy source:")
        for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {src}: {cnt:,}")

        print(f"\nBy pattern (top 10):")
        for pat, cnt in sorted(by_pattern.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pat}: {cnt:,}")

        print(f"\nBy emotion (top 10):")
        for emo, cnt in sorted(by_emotion.items(), key=lambda x: -x[1])[:10]:
            print(f"  {emo}: {cnt:,}")

        if sample_candidates:
            print(f"\nSample candidates ({len(sample_candidates)}):")
            for i, c in enumerate(sample_candidates, 1):
                print(f"\n  [{i}] {c['pattern_used']} → {c['seed_word']} ({c['emotion_category']})")
                print(f"      Sentence: {c['matched_sentence'][:120]}")
                print(f"      Source: {c['source']} | ID: {c['id']}")

    stats = {
        "total_processed": total_processed,
        "total_dupes": total_dupes,
        "total_matches": total_matches,
        "texts_with_matches": texts_with_matches,
        "unique_seed_words": len(unique_seed_words),
        "by_source": dict(by_source),
        "by_pattern": dict(by_pattern),
        "by_emotion": dict(by_emotion),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pattern-based ASI candidate extraction"
    )
    parser.add_argument(
        "--source", type=str, nargs="+", default=["merged_corpus"],
        help="JSONL file stem(s) to read from pipeline/data/ (default: merged_corpus)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed", type=Path, default=None,
        help=f"Enriched seed path (default: {ENRICHED_SEED_PATH})",
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Max records to process (0 = unlimited)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Print this many sample candidates",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    extract_candidates(
        sources=args.source,
        output_path=args.output,
        seed_path=args.seed,
        max_records=args.max_records,
        sample=args.sample,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
