"""
Merge bootstrapping and distributional mining results into an enriched seed.
"""

import json
from pathlib import Path
from typing import Any, Dict, Set

from pipeline.utils.text_utils import normalize_text


def merge_enrichment_results(
    bootstrap_result: Dict[str, Any],
    distributional_result: Dict[str, Any],
    original_seed_normalized: Set[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Merge new words from both methods into a single enriched word list.

    Deduplicates by normalized form. When both methods find the same word,
    merges provenance and takes the higher confidence.

    Returns:
        {"new_words": {word: info}, "stats": {...}}
    """
    merged: Dict[str, Dict[str, Any]] = {}

    # Add bootstrapping results
    bootstrap_words = bootstrap_result.get("new_words", {})
    for word, info in bootstrap_words.items():
        norm = normalize_text(word)
        if norm in original_seed_normalized:
            continue
        merged[norm] = {
            "word": word,
            "emotions": info.get("emotions", []),
            "confidence": info.get("confidence", 0.0) if "confidence" in info else 0.5,
            "sources": ["bootstrapping"],
            "gender": info.get("gender", None),
        }

    # Add distributional results
    distrib_words = distributional_result.get("new_words", {})
    for word, info in distrib_words.items():
        norm = normalize_text(word)
        if norm in original_seed_normalized:
            continue

        if norm in merged:
            # Both methods found this word — merge
            existing = merged[norm]
            existing["sources"].append("distributional")
            # Merge emotions
            for e in info.get("emotions", []):
                if e not in existing["emotions"]:
                    existing["emotions"].append(e)
            # Boost confidence for words found by both methods
            existing["confidence"] = min(existing["confidence"] + 0.2, 1.0)
        else:
            # Distributional confidence based on frequency
            freq = info.get("frequency", 1)
            conf = min(freq / 10.0, 0.8)
            merged[norm] = {
                "word": word,
                "emotions": info.get("emotions", ["discovered"]),
                "confidence": round(conf, 3),
                "sources": ["distributional"],
                "gender": None,
            }

    # Stats
    both_methods = sum(1 for w in merged.values() if len(w["sources"]) > 1)
    bootstrap_only = sum(1 for w in merged.values() if w["sources"] == ["bootstrapping"])
    distrib_only = sum(1 for w in merged.values() if w["sources"] == ["distributional"])

    stats = {
        "total_new_words": len(merged),
        "from_bootstrapping_only": bootstrap_only,
        "from_distributional_only": distrib_only,
        "from_both_methods": both_methods,
    }

    if verbose:
        print(f"\nMerge results:")
        print(f"  Total new words: {len(merged)}")
        print(f"  Bootstrapping only: {bootstrap_only}")
        print(f"  Distributional only: {distrib_only}")
        print(f"  Both methods: {both_methods}")

    return {"new_words": merged, "stats": stats}


def build_enriched_seed(
    original_seed: Dict[str, Any],
    merged_new_words: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a complete enriched seed by combining original + new words.

    Returns dict in same format as pipeline.seed.merged.build_seed().
    """
    adjectives = dict(original_seed.get("adjectives", {}))
    nouns = dict(original_seed.get("nouns", {}))
    adverbs = dict(original_seed.get("adverbs", {}))

    new_adj = 0
    new_noun = 0
    new_other = 0

    for norm, info in merged_new_words.items():
        word = info["word"]
        emotions = info["emotions"]
        categ = emotions[0] if emotions else "discovered"

        # Already in seed?
        if word in adjectives or word in nouns or word in adverbs:
            continue

        # Guess POS from gender or pattern
        gender = info.get("gender")
        if gender in ("m", "f"):
            # Likely adjective
            adjectives[word] = categ
            new_adj += 1
        elif "distributional" in info.get("sources", []):
            # Discovery patterns find nouns
            nouns[word] = categ
            new_noun += 1
        else:
            # Default to adjective for bootstrapped words
            adjectives[word] = categ
            new_other += 1

    # Build output matching merged.py format
    all_words = set(adjectives.keys()) | set(nouns.keys()) | set(adverbs.keys())

    # word_to_affect_categ mapping
    word_to_categ = {}
    for d in (adjectives, nouns, adverbs):
        word_to_categ.update(d)

    enriched = {
        "adjectives": adjectives,
        "nouns": nouns,
        "adverbs": adverbs,
        "all_words": sorted(all_words),
        "word_to_affect_categ": word_to_categ,
        "statistics": {
            "total": len(all_words),
            "adjectives": len(adjectives),
            "nouns": len(nouns),
            "adverbs": len(adverbs),
            "original_seed_size": (
                len(original_seed.get("adjectives", {}))
                + len(original_seed.get("nouns", {}))
                + len(original_seed.get("adverbs", {}))
            ),
            "new_adjectives": new_adj + new_other,
            "new_nouns": new_noun,
        },
    }

    return enriched


def save_enriched_seed(enriched: Dict[str, Any], output_path: Path):
    """Save enriched seed as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"\nEnriched seed saved to {output_path}")
    stats = enriched.get("statistics", {})
    print(f"  Total: {stats.get('total', '?')} words")
    print(f"  Adjectives: {stats.get('adjectives', '?')}")
    print(f"  Nouns: {stats.get('nouns', '?')}")
    print(f"  Adverbs: {stats.get('adverbs', '?')}")
