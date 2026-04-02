"""
Enriched seed loader.

Reads the enriched_seed.json produced by the seed enrichment pipeline and
exports it in the same format as pipeline.seed.merged.build_seed().
"""

import json
from pathlib import Path

ENRICHED_SEED_PATH = Path(__file__).parent.parent / "data" / "enriched_seed.json"


def build_enriched_seed(path: Path = None) -> dict:
    """
    Load the enriched seed from JSON.

    Returns dict with keys: adjectives, nouns, adverbs, all_words,
    word_to_affect_categ, statistics.
    """
    if path is None:
        path = ENRICHED_SEED_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Enriched seed not found at {path}. "
            "Run: python -m pipeline.seed_enrichment.run"
        )

    with open(path, encoding="utf-8") as f:
        return json.load(f)
