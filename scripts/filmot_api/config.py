"""
Configuration for Filmot API extraction pipeline.

Defines API key setup, trigger queries, rate limiting, and file paths.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def setup_api_key() -> str:
    """
    Load RapidAPI key from .env and set FILMOT_RAPIDAPI_KEY env var.

    The filmot package's Config.__init__ reads FILMOT_RAPIDAPI_KEY automatically.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If no API key found in .env or environment.
    """
    # Already set in environment
    existing = os.environ.get("FILMOT_RAPIDAPI_KEY")
    if existing:
        return existing

    # Load from .env file
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass

    # Map RAPIDAPI_KEY → FILMOT_RAPIDAPI_KEY
    rapid_key = os.environ.get("RAPIDAPI_KEY")
    if rapid_key:
        os.environ["FILMOT_RAPIDAPI_KEY"] = rapid_key
        return rapid_key

    # Check if filmot key was set by dotenv
    filmot_key = os.environ.get("FILMOT_RAPIDAPI_KEY")
    if filmot_key:
        return filmot_key

    raise RuntimeError(
        "No RapidAPI key found. Set RAPIDAPI_KEY in .env or "
        "FILMOT_RAPIDAPI_KEY in environment."
    )


@dataclass
class FilmotAPIConfig:
    """Configuration for Filmot API extraction pipeline."""

    # === API Settings ===
    language: str = "ro"
    max_pages_per_query: int = 500  # 50 results/page, 0 = unlimited

    # === Rate Limiting ===
    delay_between_pages: float = 0.5   # seconds between API pages
    delay_between_queries: float = 2.0  # seconds between queries

    # === Filtering ===
    min_duration_seconds: int = 60   # Skip shorts
    max_duration_seconds: int = 7200  # 2 hours max

    # === File Paths ===
    output_raw_path: Path = field(
        default_factory=lambda: Path("data/filmot_api_raw_hits.jsonl")
    )
    output_candidates_path: Path = field(
        default_factory=lambda: Path("data/filmot_api_candidates.jsonl")
    )
    checkpoint_path: Path = field(
        default_factory=lambda: Path("data/filmot_api_checkpoint.json")
    )
    stats_path: Path = field(
        default_factory=lambda: Path("data/filmot_api_stats.json")
    )


# Primary trigger queries — broad verb phrases from pattern_matcher.py
# WITHOUT emotion words, to maximize recall. The API returns subtitle context
# which we filter locally with PatternMatcher.
PRIMARY_QUERIES = [
    '"mă simt"',          # ~130K results, covers ma_simt_present
    '"mi-e"',             # ~221K results, covers mie_short
    '"m-am simțit"',      # ~32K results, covers mam_simtit_perfect
    '"îmi este"',         # ~31K results, covers imi_este_present
    '"mă simțeam"',       # ~19K results, covers ma_simteam_imperfect
    '"ne simțim"',        # covers ne_simtim_present
    '"ne-am simțit"',     # covers neam_simtit_perfect
    '"simt că"',          # covers simt_ca
    '"îmi era"',          # covers imi_era_imperfect
    '"mă voi simți"',     # covers ma_voi_simti_future
]

# Secondary trigger queries — broader patterns that may return more noise
# Skipped: "sunt", "eram", "am fost", "suntem", "am", "aveam", "simt" (bare)
# because they are too common and would return millions of non-affective results.
SECONDARY_QUERIES = [
    '"mă sim"',           # Catches diacritic variations of "mă simt/simțeam"
    '"m-am simtit"',      # No-diacritic variant of m-am simțit
    '"ma simt"',          # No-diacritic variant of mă simt
    '"imi este"',         # No-diacritic variant of îmi este
    '"imi era"',          # No-diacritic variant of îmi era
    '"ma simteam"',       # No-diacritic variant of mă simțeam
    '"ne simtim"',        # No-diacritic variant of ne simțim
    '"ne-am simtit"',     # No-diacritic variant of ne-am simțit
]


def get_trigger_queries(include_secondary: bool = True) -> List[str]:
    """
    Get list of trigger queries for API search.

    Args:
        include_secondary: Include secondary (no-diacritic) queries.

    Returns:
        List of quoted search phrases.
    """
    queries = list(PRIMARY_QUERIES)
    if include_secondary:
        queries.extend(SECONDARY_QUERIES)
    return queries
