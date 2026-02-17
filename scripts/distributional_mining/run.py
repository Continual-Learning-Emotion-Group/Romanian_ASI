#!/usr/bin/env python3
"""
Distributional Mining: Discover new ASI words from Reddit corpora and extract candidates.

Three-phase pipeline:
  Phase 1 (discover): Mine explicit labeling patterns to find new emotion words
  Phase 2 (expand):   Merge discovered words into the Ekman seed
  Phase 3 (extract):  Run the 18-pattern "I feel [ASI]" extraction with expanded seed

Usage:
    python -m scripts.distributional_mining.run
    python -m scripts.distributional_mining.run --min-freq 2 --sample 20
    python -m scripts.distributional_mining.run --phase discover
    python -m scripts.distributional_mining.run --phase extract
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Reuse existing modules
from scripts.ro_asi.pattern_matcher import (
    NOUN_EMOTION_MAP,
    PatternMatcher,
    extract_sentence,
    normalize_text,
)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

REDDIT_ROAP_PATH = PROJECT_ROOT / "small_datasets" / "RedditRoAP" / "train.parquet"
POPRERO_DIR = PROJECT_ROOT / "small_datasets" / "PoPreRo" / "Dataset"

DISCOVERED_WORDS_PATH = DATA_DIR / "distributional_discovered_words.json"
EXPANDED_SEED_PATH = DATA_DIR / "distributional_expanded_seed.json"
CANDIDATES_PATH = DATA_DIR / "distributional_asi_candidates.jsonl"
STATS_PATH = DATA_DIR / "distributional_stats.json"

# ── Ekman Seed (12 words) ───────────────────────────────────────────────────

EKMAN_ADJECTIVES = {
    "fericit": ["joy"], "fericită": ["joy"],
    "trist": ["sadness"], "tristă": ["sadness"],
    "speriat": ["fear"], "speriată": ["fear"],
    "furios": ["anger"], "furioasă": ["anger"],
    "dezgustat": ["disgust"], "dezgustată": ["disgust"],
    "surprins": ["surprise"], "surprinsă": ["surprise"],
}

EKMAN_NOUNS = {
    "fericire": ["joy"],
    "tristețe": ["sadness"],
    "frică": ["fear"],
    "furie": ["anger"],
    "dezgust": ["disgust"],
    "surpriză": ["surprise"],
}

# ── Discovery Patterns ───────────────────────────────────────────────────────

# These patterns discover new emotion words by context (no seed needed).
# Each captures a single word that appears in an emotion-labeling frame.
DISCOVERY_PATTERNS = [
    ("sentiment_de", r"\bun\s+sentiment\s+de\s+(\w+)"),
    ("sentimentul_de", r"\bsentimentul\s+de\s+(\w+)"),
    ("stare_de", r"\bo\s+stare\s+de\s+(\w+)"),
    ("starea_de", r"\bstarea\s+de\s+(\w+)"),
    ("emotie_de", r"\bemoti[ae]\s+de\s+(\w+)"),
    ("senzatie_de", r"\bo\s+senzatie\s+de\s+(\w+)"),
    ("plin_de", r"\bplin[aă]?\s+de\s+(\w+)"),
    ("cuprins_de", r"\bcuprins[aă]?\s+de\s+(\w+)"),
    ("coplesit_de", r"\bcoplesit[aă]?\s+de\s+(\w+)"),
]

# Compiled discovery patterns (work on normalized text)
COMPILED_DISCOVERY = [
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE))
    for name, pattern in DISCOVERY_PATTERNS
]

# Common non-emotion words to skip
STOPLIST = {
    # Determiners / pronouns / prepositions / conjunctions
    "un", "una", "doi", "doua", "trei", "din", "prin",
    "el", "ea", "ei", "ele", "noi", "voi", "lor",
    "ce", "care", "asta", "aia", "acesta", "aceasta", "acestea", "aceia",
    "mai", "nu", "da", "si", "sau", "cu", "la", "de", "pe", "in",
    "tot", "toti", "toate", "multi", "multe", "ceva", "nimic",
    "alt", "alta", "alte", "alti", "fiecare", "oricare",
    "asa", "foarte", "doar", "cam", "deja", "inca", "acum", "apoi",
    # Common nouns — physical objects, people, places
    "om", "oameni", "timp", "viata", "lucru", "lucruri", "loc", "mod",
    "parte", "fel", "zi", "noapte", "cap", "mine", "sine", "tine",
    "apa", "aer", "foc", "sange", "putere", "energie", "munca",
    "masini", "copii", "gandaci", "flori", "culori", "culoare",
    "noroi", "praf", "gunoi", "stele", "pietre", "iarba",
    "anunturi", "orase", "case", "drumuri", "camere",
    "bani", "haine", "carti", "idei", "probleme", "intrebari",
    "ani", "anul", "luni", "zile", "ore", "saptamani",
    "oameni", "prieteni", "romani", "copii", "femei", "barbati",
    # Physical / medical / non-emotional states
    "sanatate", "urgenta", "ebra", "vadita", "alcool",
    "somn", "oboseala", "foame", "sete",
    # Abstract non-emotional nouns
    "fapt", "spirit", "drept", "lege", "regula", "ordine",
    "treaba", "chestie", "rost", "sens", "scop", "motiv",
    "nevoie", "grija", "griji", "trebuinta",
    # Adjectives / states that aren't emotions
    "gol", "plin", "mare", "mic", "bun", "rau",
    "nou", "vechi", "lung", "scurt", "greu", "usor",
    "sigur", "gata", "clar", "liber", "singur",
    "drept", "drepte", "acord",
    # Verbs / participles that leak through
    "face", "fost", "avea", "facut", "spus", "dus", "dat", "luat",
    "fi", "este", "era", "sunt", "esti", "vor", "pot", "stiu",
    "vin", "vine", "mers", "ajuns", "plecat", "ramas",
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_reddit_posts() -> List[Tuple[str, str, str]]:
    """Load all Reddit posts from RedditRoAP and PoPreRo.

    Returns list of (record_id, text, source).
    """
    records = []

    # RedditRoAP
    if REDDIT_ROAP_PATH.exists():
        df = pd.read_parquet(REDDIT_ROAP_PATH)
        for idx, row in df.iterrows():
            text = str(row.get("TEXT", ""))
            if text.strip():
                records.append((f"reddit_roap_{idx}", text, "reddit_roap"))
        print(f"  RedditRoAP: {len(records)} posts")
    else:
        print(f"  Warning: {REDDIT_ROAP_PATH} not found")

    # PoPreRo (train + validation + test)
    poprero_count = 0
    for split_name in ["train", "validation", "test"]:
        csv_path = POPRERO_DIR / f"{split_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                text = str(row.get("full_text", ""))
                if text.strip():
                    records.append((f"poprero_{split_name}_{idx}", text, "poprero"))
                    poprero_count += 1
    print(f"  PoPreRo: {poprero_count} posts")

    print(f"  Total: {len(records)} posts loaded")
    return records


# ── Phase 1: Discovery ──────────────────────────────────────────────────────

def discover_words(records: List[Tuple[str, str, str]]) -> Dict:
    """Mine explicit labeling patterns to discover new emotion words.

    Returns dict with word -> {frequency, patterns, examples}.
    """
    print("\n═══ Phase 1: Discovery ═══")
    print(f"Scanning {len(records)} posts with {len(COMPILED_DISCOVERY)} discovery patterns...")

    word_data = defaultdict(lambda: {
        "frequency": 0,
        "patterns": set(),
        "examples": [],
    })

    total_matches = 0
    for record_id, text, source in records:
        normalized = normalize_text(text)
        for pattern_name, pattern in COMPILED_DISCOVERY:
            for match in pattern.finditer(normalized):
                word = match.group(1).strip().lower()

                # Filter
                if len(word) < 3 or len(word) > 20:
                    continue
                if word in STOPLIST:
                    continue
                # Skip if it contains digits
                if any(c.isdigit() for c in word):
                    continue

                entry = word_data[word]
                entry["frequency"] += 1
                entry["patterns"].add(pattern_name)
                # Keep up to 3 example contexts
                if len(entry["examples"]) < 3:
                    # Extract a short context around the match
                    start = max(0, match.start() - 40)
                    end = min(len(normalized), match.end() + 40)
                    context = normalized[start:end].strip()
                    entry["examples"].append({
                        "context": context,
                        "source": source,
                        "record_id": record_id,
                    })
                total_matches += 1

    # Convert sets to lists for JSON serialization
    result = {}
    for word, data in sorted(word_data.items(), key=lambda x: -x[1]["frequency"]):
        result[word] = {
            "frequency": data["frequency"],
            "patterns": sorted(data["patterns"]),
            "examples": data["examples"],
        }

    print(f"  Total pattern matches: {total_matches}")
    print(f"  Unique words discovered: {len(result)}")

    # Show top 20
    if result:
        print("\n  Top 20 discovered words:")
        for i, (word, data) in enumerate(list(result.items())[:20]):
            patterns_str = ", ".join(data["patterns"])
            print(f"    {i+1:3d}. {word:<20s}  freq={data['frequency']:3d}  patterns=[{patterns_str}]")

    return result


def save_discovered_words(discovered: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISCOVERED_WORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(discovered, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved {len(discovered)} discovered words to {DISCOVERED_WORDS_PATH}")


def load_discovered_words() -> Dict:
    with open(DISCOVERED_WORDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Phase 2: Expand Seed ────────────────────────────────────────────────────

def expand_seed(discovered: Dict, min_freq: int = 1) -> Dict:
    """Merge discovered words into the Ekman seed.

    Returns expanded seed dict with word_to_emotions and noun_words.
    """
    print("\n═══ Phase 2: Expand Seed ═══")

    # Start with Ekman seed
    word_to_emotions: Dict[str, List[str]] = {}
    adjective_words: Set[str] = set()
    noun_words: Set[str] = set()

    # Add Ekman adjectives
    for word, emotions in EKMAN_ADJECTIVES.items():
        normalized = normalize_text(word)
        word_to_emotions[normalized] = emotions
        adjective_words.add(normalized)

    # Add Ekman nouns
    for word, emotions in EKMAN_NOUNS.items():
        normalized = normalize_text(word)
        word_to_emotions[normalized] = emotions
        noun_words.add(normalized)

    ekman_count = len(word_to_emotions)
    print(f"  Ekman seed: {ekman_count} words ({len(adjective_words)} adj, {len(noun_words)} nouns)")

    # Add discovered words (as nouns — discovery patterns primarily find nouns)
    new_count = 0
    for word, data in discovered.items():
        if data["frequency"] < min_freq:
            continue
        normalized = normalize_text(word)
        if normalized in word_to_emotions:
            continue  # Already in seed

        # Try to assign emotion from NOUN_EMOTION_MAP
        emotions = NOUN_EMOTION_MAP.get(normalized, ["unknown"])
        word_to_emotions[normalized] = emotions
        noun_words.add(normalized)
        new_count += 1

    print(f"  New words added (min_freq={min_freq}): {new_count}")
    print(f"  Expanded seed total: {len(word_to_emotions)} words")
    print(f"    Adjectives: {len(adjective_words)}")
    print(f"    Nouns: {len(noun_words)}")

    # Build output
    expanded = {
        "word_to_emotions": word_to_emotions,
        "adjective_words": sorted(adjective_words),
        "noun_words": sorted(noun_words),
        "stats": {
            "ekman_words": ekman_count,
            "discovered_words": new_count,
            "total_words": len(word_to_emotions),
        },
    }

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPANDED_SEED_PATH, "w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)
    print(f"  Saved expanded seed to {EXPANDED_SEED_PATH}")

    return expanded


# ── Phase 3: Extract Candidates ─────────────────────────────────────────────

def _discovery_pattern_counts(discovered: Dict) -> Dict[str, int]:
    """Count how many words each discovery pattern found."""
    counts = defaultdict(int)
    for data in discovered.values():
        for p in data["patterns"]:
            counts[p] += 1
    return dict(counts)


def extract_candidates(
    records: List[Tuple[str, str, str]],
    expanded: Dict,
    ekman_normalized: Set[str],
    discovered_words: Optional[Dict] = None,
    min_freq: int = 1,
    sample: int = 0,
) -> Tuple[List[Dict], Dict]:
    """Extract ASI candidates using expanded seed.

    Args:
        records: List of (id, text, source) tuples
        expanded: Expanded seed dict from Phase 2
        ekman_normalized: Set of normalized Ekman words (for confidence scoring)
        discovered_words: Discovery phase output (for stats)
        min_freq: Min freq threshold used (for stats)
        sample: If > 0, print this many sample candidates

    Returns:
        (candidates_list, stats_dict)
    """
    print("\n═══ Phase 3: Extract Candidates ═══")

    word_to_emotions = expanded["word_to_emotions"]
    noun_words_list = expanded["noun_words"]

    # Create PatternMatcher with expanded seed
    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words_list)

    candidates = []
    seen_hashes = set()
    match_count_by_pattern = defaultdict(int)
    match_count_by_source = defaultdict(int)
    conf_counts = defaultdict(int)
    emo_counts = defaultdict(int)
    seed_words = defaultdict(int)
    new_word_matches = 0
    ekman_matches = 0

    print(f"\nScanning {len(records)} posts...")
    for record_id, text, source in records:
        matches = matcher.find_matches(text, extract_sentences=True)
        for m in matches:
            # Dedup by hash of matched sentence
            h = hashlib.md5(m.matched_text.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            is_new = m.seed_word_normalized not in ekman_normalized
            is_primary = m.pattern_category == "primary"

            # Confidence scoring
            if not is_new:
                confidence = 0.9
            elif is_primary:
                confidence = 0.7
            else:
                confidence = 0.5

            candidate = {
                "id": record_id,
                "text": text,
                "matched_sentence": m.matched_text,
                "extraction_strategy": "distributional_mining",
                "confidence": confidence,
                "seed_word": m.seed_word,
                "emotion_category": m.emotions,
                "source": source,
                "metadata": {
                    "pattern_used": m.pattern_name,
                    "pattern_category": m.pattern_category,
                    "is_new_discovery": is_new,
                },
            }
            candidates.append(candidate)

            # Stats
            match_count_by_pattern[m.pattern_name] += 1
            match_count_by_source[source] += 1
            conf_counts[confidence] += 1
            seed_words[m.seed_word] += 1
            for emo in m.emotions:
                emo_counts[emo] += 1
            if is_new:
                new_word_matches += 1
            else:
                ekman_matches += 1

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\n  Total candidates: {len(candidates)}")
    print(f"    From Ekman words: {ekman_matches}")
    print(f"    From new discoveries: {new_word_matches}")

    print(f"\n  By source:")
    for source, count in sorted(match_count_by_source.items()):
        print(f"    {source}: {count}")

    print(f"\n  By pattern:")
    for pattern, count in sorted(match_count_by_pattern.items(), key=lambda x: -x[1]):
        print(f"    {pattern}: {count}")

    # Sample output
    if sample > 0 and candidates:
        print(f"\n  Sample candidates (first {min(sample, len(candidates))}):")
        for c in candidates[:sample]:
            new_tag = " [NEW]" if c["metadata"]["is_new_discovery"] else ""
            print(f"    [{c['confidence']:.1f}] {c['matched_sentence'][:80]}")
            print(f"         word={c['seed_word']}, emotion={c['emotion_category']}, "
                  f"pattern={c['metadata']['pattern_used']}{new_tag}")

    print(f"\n  Saved to {CANDIDATES_PATH}")

    # Build stats dict
    # Unique seed words by origin
    new_seed_words_used = defaultdict(int)
    ekman_seed_words_used = defaultdict(int)
    for c in candidates:
        w = c["seed_word"]
        if c["metadata"]["is_new_discovery"]:
            new_seed_words_used[w] += 1
        else:
            ekman_seed_words_used[w] += 1

    stats = {
        "corpus": {
            "total_posts": len(records),
            "by_source": dict(sorted(
                {s: sum(1 for r in records if r[2] == s) for s in {r[2] for r in records}}.items()
            )),
        },
        "discovery": {
            "total_pattern_matches": sum(d["frequency"] for d in discovered_words.values()) if discovered_words else 0,
            "unique_words_discovered": len(discovered_words) if discovered_words else 0,
            "words_matching_known_emotions": sum(
                1 for w in (discovered_words or {})
                if normalize_text(w) in NOUN_EMOTION_MAP
            ),
            "by_pattern": dict(sorted(
                _discovery_pattern_counts(discovered_words).items(),
                key=lambda x: -x[1]
            )) if discovered_words else {},
        },
        "seed_expansion": {
            "ekman_words": expanded["stats"]["ekman_words"],
            "discovered_words_added": expanded["stats"]["discovered_words"],
            "total_seed_size": expanded["stats"]["total_words"],
            "min_freq_threshold": min_freq,
        },
        "extraction": {
            "total_candidates": len(candidates),
            "from_ekman_words": ekman_matches,
            "from_new_discoveries": new_word_matches,
            "unique_seed_words_matched": len(seed_words),
            "by_confidence": {str(k): v for k, v in sorted(conf_counts.items())},
            "by_source": dict(sorted(match_count_by_source.items())),
            "by_pattern": dict(sorted(match_count_by_pattern.items(), key=lambda x: -x[1])),
            "by_emotion": dict(sorted(emo_counts.items(), key=lambda x: -x[1])),
            "top_new_seed_words": dict(sorted(new_seed_words_used.items(), key=lambda x: -x[1])[:20]),
            "top_ekman_seed_words": dict(sorted(ekman_seed_words_used.items(), key=lambda x: -x[1])[:20]),
        },
    }

    return candidates, stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Distributional Mining: Discover new ASI words and extract candidates"
    )
    parser.add_argument(
        "--phase", choices=["discover", "expand", "extract"],
        help="Run only a specific phase (default: all phases)"
    )
    parser.add_argument(
        "--min-freq", type=int, default=1,
        help="Minimum frequency for discovered words to be included in seed (default: 1)"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Number of sample candidates to print (default: 0)"
    )
    args = parser.parse_args()

    # Build set of normalized Ekman words (for confidence scoring later)
    ekman_normalized = set()
    for word in list(EKMAN_ADJECTIVES.keys()) + list(EKMAN_NOUNS.keys()):
        ekman_normalized.add(normalize_text(word))

    print("Distributional Mining Pipeline")
    print("=" * 50)

    run_all = args.phase is None

    # Load data (needed for discover + extract phases)
    records = None
    if run_all or args.phase in ("discover", "extract"):
        print("\nLoading Reddit corpora...")
        records = load_reddit_posts()

    # Phase 1: Discover
    discovered = None
    if run_all or args.phase == "discover":
        discovered = discover_words(records)
        save_discovered_words(discovered)

    # Phase 2: Expand
    expanded = None
    if run_all or args.phase == "expand":
        if discovered is None:
            print("\nLoading previously discovered words...")
            discovered = load_discovered_words()
        expanded = expand_seed(discovered, min_freq=args.min_freq)

    # Phase 3: Extract
    if run_all or args.phase == "extract":
        if expanded is None:
            print("\nLoading expanded seed...")
            with open(EXPANDED_SEED_PATH, "r", encoding="utf-8") as f:
                expanded = json.load(f)
        if discovered is None:
            if DISCOVERED_WORDS_PATH.exists():
                discovered = load_discovered_words()
        if records is None:
            print("\nLoading Reddit corpora...")
            records = load_reddit_posts()
        candidates, stats = extract_candidates(
            records, expanded, ekman_normalized,
            discovered_words=discovered,
            min_freq=args.min_freq,
            sample=args.sample,
        )

        # Save stats
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved stats to {STATS_PATH}")

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
