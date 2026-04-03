"""
Distributional mining for seed enrichment.

Discovers new emotion words via explicit labeling patterns that don't need a
seed: "un sentiment de X", "sentimentul de X", "emoție de X", etc.

These patterns primarily discover nouns.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.utils.text_utils import normalize_text
from pipeline.utils.stoplists import STOPWORDS
from pipeline.utils.corpus_reader import iter_corpus
from pipeline.utils.pattern_matcher import NOUN_EMOTION_MAP

# ---------------------------------------------------------------------------
# Discovery patterns (no seed needed)
# ---------------------------------------------------------------------------

DISCOVERY_PATTERNS = [
    ("sentiment_de", r"\bun\s+sentiment\s+de\s+(\w+)"),
    ("sentimentul_de", r"\bsentimentul\s+de\s+(\w+)"),
    ("emotie_de", r"\bemoti[ae]\s+de\s+(\w+)"),
    ("senzatie_de", r"\bo\s+senzatie\s+de\s+(\w+)"),
    ("coplesit_de", r"\bcoplesit[aă]?\s+de\s+(\w+)"),
]

COMPILED_DISCOVERY = [
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE))
    for name, pattern in DISCOVERY_PATTERNS
]


# ---------------------------------------------------------------------------
# Core scanning (works with any text iterator)
# ---------------------------------------------------------------------------

def _scan_for_discoveries(
    text_iterator,
    verbose: bool = True,
) -> Tuple[Dict, int]:
    """
    Scan texts for discovery pattern matches.

    Args:
        text_iterator: Iterator yielding (record_id, text, source) tuples.
        verbose: Print progress.

    Returns:
        (word_data dict, total_matches count).
        word_data values have "frequency", "patterns" (set), "examples" (list).
    """
    word_data = defaultdict(lambda: {
        "frequency": 0,
        "patterns": set(),
        "examples": [],
    })
    total_matches = 0

    for record_id, text, source in text_iterator:
        normalized = normalize_text(text)

        for pattern_name, pattern in COMPILED_DISCOVERY:
            for match in pattern.finditer(normalized):
                word = match.group(1).strip().lower()

                if len(word) < 3 or len(word) > 20:
                    continue
                if word in STOPWORDS:
                    continue
                if any(c.isdigit() for c in word):
                    continue

                entry = word_data[word]
                entry["frequency"] += 1
                entry["patterns"].add(pattern_name)

                if len(entry["examples"]) < 3:
                    start = max(0, match.start() - 40)
                    end = min(len(normalized), match.end() + 40)
                    context = normalized[start:end].strip()
                    entry["examples"].append({
                        "context": context,
                        "source": source,
                        "record_id": record_id,
                    })
                total_matches += 1

    return word_data, total_matches


def _filter_and_format(
    word_data: Dict,
    min_freq: int,
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    """Filter by frequency, convert sets to lists, print summary."""
    result = {}
    for word, data in sorted(word_data.items(), key=lambda x: -x[1]["frequency"]):
        if data["frequency"] < min_freq:
            continue
        result[word] = {
            "frequency": data["frequency"],
            "patterns": sorted(data["patterns"]),
            "examples": data["examples"],
        }

    if verbose:
        print(f"  Unique words (freq >= {min_freq}): {len(result)}")
        if result:
            print(f"\n  Top 20 discovered words:")
            for i, (word, data) in enumerate(list(result.items())[:20]):
                pats = ", ".join(data["patterns"])
                print(f"    {i + 1:3d}. {word:<20s}  freq={data['frequency']:3d}  [{pats}]")

    return result


# ---------------------------------------------------------------------------
# Discovery entry points
# ---------------------------------------------------------------------------

def discover_words(
    data_dir: Path,
    min_freq: int = 2,
    sources: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Discover words from JSONL files in data_dir."""
    if verbose:
        print(f"\nDistributional mining: scanning with {len(COMPILED_DISCOVERY)} patterns")

    text_iter = iter_corpus(data_dir, sources=sources)
    word_data, total_matches = _scan_for_discoveries(text_iter, verbose)

    if verbose:
        print(f"  Total pattern matches: {total_matches}")

    return _filter_and_format(word_data, min_freq, verbose)


def discover_words_streaming(
    text_iterator,
    min_freq: int = 2,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Discover words from a streaming text source (e.g., FULG)."""
    if verbose:
        print(f"\nDistributional mining (streaming): {len(COMPILED_DISCOVERY)} patterns")

    word_data, total_matches = _scan_for_discoveries(text_iterator, verbose)

    if verbose:
        print(f"  Total pattern matches: {total_matches}")

    return _filter_and_format(word_data, min_freq, verbose)


# ---------------------------------------------------------------------------
# Expand seed with discovered words
# ---------------------------------------------------------------------------

def expand_seed_with_discoveries(
    discovered: Dict[str, Dict[str, Any]],
    existing_seed_normalized: Set[str],
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build new-word entries from discovered words.

    Only includes words NOT already in the existing seed.
    """
    new_words = {}

    for word, data in discovered.items():
        if word in existing_seed_normalized:
            continue

        emotions = NOUN_EMOTION_MAP.get(word, ["discovered"])

        new_words[word] = {
            "emotions": emotions,
            "frequency": data["frequency"],
            "patterns": data["patterns"],
            "examples": data["examples"],
        }

    if verbose:
        known_emotion = sum(1 for w in new_words.values() if w["emotions"] != ["discovered"])
        print(f"\n  New words (not in seed): {len(new_words)}")
        print(f"    With known emotion: {known_emotion}")
        print(f"    Unknown emotion: {len(new_words) - known_emotion}")

    return new_words


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_distributional_mining(
    data_dir: Path,
    seed_normalized: Set[str],
    min_freq: int = 2,
    sources: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run distributional mining on JSONL files."""
    discovered = discover_words(data_dir, min_freq=min_freq, sources=sources, verbose=verbose)
    new_words = expand_seed_with_discoveries(discovered, seed_normalized, verbose=verbose)

    return {
        "new_words": new_words,
        "discovered": discovered,
        "stats": {"total_discovered": len(discovered), "new_words": len(new_words), "min_freq": min_freq},
    }


def run_distributional_mining_streaming(
    text_iterator,
    seed_normalized: Set[str],
    min_freq: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run distributional mining on a streaming text source (e.g., FULG)."""
    discovered = discover_words_streaming(text_iterator, min_freq=min_freq, verbose=verbose)
    new_words = expand_seed_with_discoveries(discovered, seed_normalized, verbose=verbose)

    return {
        "new_words": new_words,
        "discovered": discovered,
        "stats": {"total_discovered": len(discovered), "new_words": len(new_words), "min_freq": min_freq},
    }
