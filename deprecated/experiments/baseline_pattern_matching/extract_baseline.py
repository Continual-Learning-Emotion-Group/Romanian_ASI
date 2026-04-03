#!/usr/bin/env python3
"""
Baseline extraction using minimal MASIVE seed (6 emotions) on Reddit datasets.

Uses only the 6 basic MASIVE emotions {happy, sad, angry, afraid, disgusted, surprised}
mapped to Romanian forms (with gender/plural/diacritics variants). No bootstrapping.

This establishes the starting point that other strategies will build upon.

Datasets: RedditRoAP (parquet) + PoPreRo (CSV)
Output: data/reddit_baseline_candidates.jsonl

Usage:
    python -m experiments.baseline_pattern_matching.extract_baseline
    python -m experiments.baseline_pattern_matching.extract_baseline --sample 20
"""

import json
import csv
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Generator, Any, Set
from collections import defaultdict
from datetime import datetime

from scripts.ro_asi.pattern_matcher import PatternMatcher, PatternMatch, remove_diacritics


# ============================================================
# Minimal MASIVE seed: 6 emotions → Romanian forms
# ============================================================

def _expand_diacritics(words_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Add diacritics-stripped variants for every word."""
    expanded = {}
    for word, emotions in words_dict.items():
        expanded[word] = emotions
        stripped = remove_diacritics(word)
        if stripped != word and stripped not in expanded:
            expanded[stripped] = emotions
    return expanded


# Adjectives: used with "mă simt [X]", "sunt [X]", etc.
MASIVE_SEED_ADJECTIVES = _expand_diacritics({
    # happy → fericit (m/f/m-pl/f-pl)
    "fericit": ["joy"],
    "fericită": ["joy"],
    "fericiți": ["joy"],
    "fericite": ["joy"],
    # sad → trist
    "trist": ["sadness"],
    "tristă": ["sadness"],
    "triști": ["sadness"],
    "triste": ["sadness"],
    # angry → furios + supărat
    "furios": ["anger"],
    "furioasă": ["anger"],
    "furioși": ["anger"],
    "furioase": ["anger"],
    "supărat": ["anger"],
    "supărată": ["anger"],
    "supărați": ["anger"],
    "supărate": ["anger"],
    # afraid → speriat + înfricoșat
    "speriat": ["fear"],
    "speriată": ["fear"],
    "speriați": ["fear"],
    "speriate": ["fear"],
    "înfricoșat": ["fear"],
    "înfricoșată": ["fear"],
    "înfricoșați": ["fear"],
    "înfricoșate": ["fear"],
    # disgusted → dezgustat + scârbit
    "dezgustat": ["disgust"],
    "dezgustată": ["disgust"],
    "dezgustați": ["disgust"],
    "dezgustate": ["disgust"],
    "scârbit": ["disgust"],
    "scârbită": ["disgust"],
    "scârbiți": ["disgust"],
    "scârbite": ["disgust"],
    # surprised → surprins + uimit
    "surprins": ["surprise"],
    "surprinsă": ["surprise"],
    "surprinși": ["surprise"],
    "surprinse": ["surprise"],
    "uimit": ["surprise"],
    "uimită": ["surprise"],
    "uimiți": ["surprise"],
    "uimite": ["surprise"],
})

# Nouns: used with "simt [X]", "am [X]", "mi-e [X]", "îmi este [X]"
MASIVE_SEED_NOUNS = _expand_diacritics({
    # happy
    "fericire": ["joy"],
    "bucurie": ["joy"],
    # sad
    "tristețe": ["sadness"],
    # angry
    "furie": ["anger"],
    "mânie": ["anger"],
    # afraid
    "frică": ["fear"],
    "teamă": ["fear"],
    "spaimă": ["fear"],
    # disgusted
    "dezgust": ["disgust"],
    "scârbă": ["disgust"],
    # surprised
    "surpriză": ["surprise"],
    "uimire": ["surprise"],
})


def build_masive_seed() -> Dict[str, Any]:
    """Build seed dict in the same format as curated_affective_states.build_curated_seed()."""
    word_to_emotions = {}
    word_to_emotions.update(MASIVE_SEED_ADJECTIVES)
    word_to_emotions.update(MASIVE_SEED_NOUNS)

    return {
        "source": "masive_baseline_6_emotions",
        "word_to_emotions": word_to_emotions,
        "all_words": sorted(word_to_emotions.keys()),
        "nouns": list(MASIVE_SEED_NOUNS.keys()),
        "adjectives": list(MASIVE_SEED_ADJECTIVES.keys()),
        "statistics": {
            "total_adjectives": len(MASIVE_SEED_ADJECTIVES),
            "total_nouns": len(MASIVE_SEED_NOUNS),
            "total_words": len(word_to_emotions),
        },
    }


# ============================================================
# Dataset loaders
# ============================================================

def load_reddit_roap(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Load RedditRoAP dataset from parquet."""
    file_path = base_path / "small_datasets" / "RedditRoAP" / "train.parquet"
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return

    import pandas as pd
    df = pd.read_parquet(file_path)

    for idx, row in df.iterrows():
        text = str(row.get("TEXT", "")).strip()
        if not text:
            continue

        labels = {}
        if pd.notna(row.get("SUBDIALECT")):
            labels["subdialect"] = row["SUBDIALECT"]
        if pd.notna(row.get("STATUS")):
            labels["status"] = row["STATUS"]
        if pd.notna(row.get("LABELS")):
            labels["topic_labels"] = row["LABELS"]
        if pd.notna(row.get("PERSONAL INCLINATION")):
            labels["personal_inclination"] = row["PERSONAL INCLINATION"]

        yield {
            "id": f"reddit_roap_{idx}",
            "text": text,
            "source": "reddit_roap",
            "split": "train",
            "original_labels": labels,
        }


def load_poprero(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Load PoPreRo dataset from CSV splits."""
    for split in ["train", "validation", "test"]:
        file_path = base_path / "small_datasets" / "PoPreRo" / "Dataset" / f"{split}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("full_text", "").strip()
                if not text:
                    continue

                yield {
                    "id": f"poprero_{row.get('index', '')}",
                    "text": text,
                    "source": "poprero",
                    "split": split,
                    "original_labels": {
                        "popularity_label": row.get("label"),
                    },
                }


# ============================================================
# Candidate formatting
# ============================================================

def format_candidate(record: Dict[str, Any], match: PatternMatch) -> Dict[str, Any]:
    """Format a matched record as ASI candidate."""
    return {
        "id": record["id"],
        "text": record["text"],
        "matched_sentence": match.matched_text,
        "pattern_used": match.pattern_name,
        "pattern_category": match.pattern_category,
        "seed_word": match.seed_word,
        "seed_word_normalized": match.seed_word_normalized,
        "emotion_category": match.emotions,
        "source": record.get("source", "unknown"),
        "split": record.get("split", "unknown"),
        "original_labels": record.get("original_labels", {}),
        "extraction_strategy": "pattern_matching_baseline",
    }


# ============================================================
# Main extraction
# ============================================================

def extract_candidates(
    base_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run baseline extraction on RedditRoAP + PoPreRo."""
    stats = {
        "total_processed": 0,
        "total_matches": 0,
        "unique_texts_matched": 0,
        "by_source": defaultdict(int),
        "by_pattern": defaultdict(int),
        "by_emotion": defaultdict(int),
        "unique_seed_words": set(),
        "started_at": datetime.now().isoformat(),
        "extraction_strategy": "pattern_matching_baseline",
        "seed": "masive_6_emotions",
    }

    # Build seed and matcher
    seed = build_masive_seed()
    word_to_emotions = seed["word_to_emotions"]
    noun_words = seed.get("nouns", None)

    print(f"\nMASIVE baseline seed:")
    print(f"  Adjectives: {seed['statistics']['total_adjectives']}")
    print(f"  Nouns: {seed['statistics']['total_nouns']}")
    print(f"  Total words: {seed['statistics']['total_words']}")
    print()

    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words)

    # Track unique texts
    seen_text_hashes: Set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process both datasets
    datasets = [
        ("reddit_roap", load_reddit_roap(base_path)),
        ("poprero", load_poprero(base_path)),
    ]

    source_totals: Dict[str, int] = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for dataset_name, records in datasets:
            print(f"\nProcessing {dataset_name}...")

            for record in records:
                stats["total_processed"] += 1
                source_totals[dataset_name] += 1

                if verbose and stats["total_processed"] % 5000 == 0:
                    print(f"  Processed {stats['total_processed']} records, "
                          f"found {stats['total_matches']} matches...")

                text = record.get("text", "")
                if not text:
                    continue

                # Deduplicate
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen_text_hashes:
                    continue

                # Find matches
                matches = matcher.find_matches(text, extract_sentences=True)

                if matches:
                    seen_text_hashes.add(text_hash)
                    stats["unique_texts_matched"] += 1

                    for match in matches:
                        candidate = format_candidate(record, match)
                        out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

                        stats["total_matches"] += 1
                        stats["by_source"][record.get("source", "unknown")] += 1
                        stats["by_pattern"][match.pattern_name] += 1
                        stats["unique_seed_words"].add(match.seed_word_normalized)

                        for emotion in match.emotions:
                            stats["by_emotion"][emotion] += 1

    # Finalize stats
    stats["source_totals"] = dict(source_totals)
    stats["unique_seed_words_count"] = len(stats["unique_seed_words"])
    stats["unique_seed_words"] = sorted(list(stats["unique_seed_words"]))
    stats["finished_at"] = datetime.now().isoformat()
    stats["by_source"] = dict(stats["by_source"])
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])

    return stats


def print_stats(stats: Dict[str, Any]):
    """Print extraction statistics."""
    print("\n" + "=" * 60)
    print("Baseline Extraction Statistics (MASIVE 6-emotion seed)")
    print("=" * 60)
    print(f"Total records processed: {stats['total_processed']}")
    print(f"Unique texts matched: {stats['unique_texts_matched']}")
    print(f"Total pattern matches: {stats['total_matches']}")

    if stats["total_processed"] > 0:
        rate = stats["unique_texts_matched"] / stats["total_processed"] * 100
        print(f"Match rate: {rate:.2f}%")

    print(f"\nUnique seed words found: {stats['unique_seed_words_count']}")

    print("\nRecords per dataset:")
    for source, count in sorted(stats.get("source_totals", {}).items()):
        matches = stats["by_source"].get(source, 0)
        pct = matches / count * 100 if count > 0 else 0
        print(f"  {source}: {count} records → {matches} matches ({pct:.2f}%)")

    print("\nMatches by pattern:")
    for pattern, count in sorted(stats["by_pattern"].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    print("\nMatches by emotion:")
    for emotion, count in sorted(stats["by_emotion"].items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count}")

    if stats["unique_seed_words"]:
        print(f"\nSeed words matched ({stats['unique_seed_words_count']}): "
              f"{', '.join(stats['unique_seed_words'])}")


def sample_candidates(output_path: Path, n: int = 10):
    """Print sample candidates for manual inspection."""
    print(f"\n{'=' * 60}")
    print(f"Sample Candidates (first {n})")
    print("=" * 60)

    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            candidate = json.loads(line)
            print(f"\n[{i+1}] ID: {candidate['id']}")
            print(f"    Source: {candidate['source']}")
            print(f"    Pattern: {candidate['pattern_used']} ({candidate['pattern_category']})")
            print(f"    Seed word: {candidate['seed_word']} → {candidate['emotion_category']}")
            matched = candidate["matched_sentence"]
            if len(matched) > 120:
                matched = matched[:120] + "..."
            print(f"    Matched: \"{matched}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline ASI extraction with MASIVE 6-emotion seed on Reddit datasets"
    )
    parser.add_argument(
        "--sample", type=int, default=10,
        help="Number of sample candidates to print (default: 10)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / "data" / "reddit_baseline_candidates.jsonl"

    print("=" * 60)
    print("Baseline Pattern Matching: MASIVE 6-emotion seed")
    print("Datasets: RedditRoAP + PoPreRo")
    print("=" * 60)

    # Run extraction
    stats = extract_candidates(
        base_path=base_path,
        output_path=output_path,
        verbose=not args.quiet,
    )

    # Print stats
    print_stats(stats)

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

    # Print samples
    if args.sample > 0 and stats["total_matches"] > 0:
        sample_candidates(output_path, args.sample)

    print(f"\nOutput saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
