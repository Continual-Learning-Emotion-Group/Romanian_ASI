"""
Distributional mining for seed enrichment.

Discovers new emotion words via explicit labeling patterns that don't need a
seed: "un sentiment de X", "o stare de X", "plin de X", etc.

These patterns primarily discover nouns.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    ("stare_de", r"\bo\s+stare\s+de\s+(\w+)"),
    ("starea_de", r"\bstarea\s+de\s+(\w+)"),
    ("emotie_de", r"\bemoti[ae]\s+de\s+(\w+)"),
    ("senzatie_de", r"\bo\s+senzatie\s+de\s+(\w+)"),
    ("plin_de", r"\bplin[aă]?\s+de\s+(\w+)"),
    ("cuprins_de", r"\bcuprins[aă]?\s+de\s+(\w+)"),
    ("coplesit_de", r"\bcoplesit[aă]?\s+de\s+(\w+)"),
]

COMPILED_DISCOVERY = [
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE))
    for name, pattern in DISCOVERY_PATTERNS
]


# ---------------------------------------------------------------------------
# Discovery phase
# ---------------------------------------------------------------------------

def discover_words(
    data_dir: Path,
    min_freq: int = 2,
    sources: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Mine explicit labeling patterns to discover new emotion words.

    Args:
        data_dir: Directory with JSONL corpus files.
        min_freq: Minimum frequency for a word to be kept.
        sources: Which JSONL files to use (None = all).
        verbose: Print progress.

    Returns:
        Dict of word → {"frequency", "patterns", "examples"}.
    """
    if verbose:
        print(f"\nDistributional mining: scanning with {len(COMPILED_DISCOVERY)} patterns")

    word_data = defaultdict(lambda: {
        "frequency": 0,
        "patterns": set(),
        "examples": [],
    })

    total_matches = 0

    for record_id, text, source in iter_corpus(data_dir, sources=sources):
        normalized = normalize_text(text)

        for pattern_name, pattern in COMPILED_DISCOVERY:
            for match in pattern.finditer(normalized):
                word = match.group(1).strip().lower()

                # Basic filters
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

    # Filter by frequency and convert sets to lists
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
        print(f"  Total pattern matches: {total_matches}")
        print(f"  Unique words (freq >= {min_freq}): {len(result)}")
        if result:
            print(f"\n  Top 20 discovered words:")
            for i, (word, data) in enumerate(list(result.items())[:20]):
                pats = ", ".join(data["patterns"])
                print(f"    {i + 1:3d}. {word:<20s}  freq={data['frequency']:3d}  [{pats}]")

    return result


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

    Args:
        discovered: Output of discover_words().
        existing_seed_normalized: Normalized forms already in the seed.
        verbose: Print progress.

    Returns:
        Dict of word → {"emotions", "frequency", "patterns", "examples"}.
        Only includes words NOT already in the existing seed.
    """
    new_words = {}

    for word, data in discovered.items():
        if word in existing_seed_normalized:
            continue

        # Assign emotion from NOUN_EMOTION_MAP if available
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
# Main entry point
# ---------------------------------------------------------------------------

def run_distributional_mining(
    data_dir: Path,
    seed_normalized: Set[str],
    min_freq: int = 2,
    sources: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run distributional mining pipeline (Phase 1 + Phase 2).

    Args:
        data_dir: Directory with JSONL corpus files.
        seed_normalized: Set of normalized words already in seed.
        min_freq: Minimum frequency threshold.
        sources: Which JSONL files to use.
        verbose: Print progress.

    Returns:
        {"new_words": {word: info}, "discovered": {word: data}, "stats": {...}}
    """
    # Phase 1: Discovery
    discovered = discover_words(data_dir, min_freq=min_freq, sources=sources, verbose=verbose)

    # Phase 2: Expand
    new_words = expand_seed_with_discoveries(discovered, seed_normalized, verbose=verbose)

    return {
        "new_words": new_words,
        "discovered": discovered,
        "stats": {
            "total_discovered": len(discovered),
            "new_words": len(new_words),
            "min_freq": min_freq,
        },
    }
