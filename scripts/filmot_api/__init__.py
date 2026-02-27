"""
Filmot API Extraction Package.

Uses the filmot Python package (RapidAPI) to search YouTube subtitles
for Romanian affective state expressions, bypassing Cloudflare issues
with the old Playwright-based pipeline.

Two-phase pipeline:
    Phase 1 (collect.py): Paginated API search → raw hits JSONL
    Phase 2 (filter_candidates.py): PatternMatcher on raw hits → ASI candidates JSONL

Usage:
    python -m scripts.filmot_api.collect
    python -m scripts.filmot_api.collect --resume
    python -m scripts.filmot_api.filter_candidates
    python -m scripts.filmot_api.filter_candidates --sample 20
"""

__version__ = "0.1.0"
