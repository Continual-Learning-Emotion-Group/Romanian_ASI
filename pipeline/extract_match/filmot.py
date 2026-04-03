#!/usr/bin/env python3
"""
Filmot extraction: query ALL "I feel" triggers via Filmot API, filter by
PatternMatcher + enriched seed, save only real candidates.

Unlike pipeline/collect/stream_filmot.py (which collects raw hits for seed
enrichment using only "simt"-family triggers), this script queries ALL 20
pattern triggers (including "sunt", "eram", "am fost") and immediately
filters each hit through the PatternMatcher. Only hits where a seed word
follows the trigger pattern are saved.

Requires:
    pip install filmot python-dotenv tqdm
    RAPIDAPI_KEY in .env

Usage:
    python -m pipeline.extract_match.filmot                          # default
    python -m pipeline.extract_match.filmot --max-hits 500000        # more
    python -m pipeline.extract_match.filmot --workers 8              # faster
    python -m pipeline.extract_match.filmot --resume                 # resume
    python -m pipeline.extract_match.filmot --max-pages-per-query 2  # quick test
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_SEED_PATH = DATA_DIR / "enriched_seed_merged.json"
DEFAULT_OUTPUT = DATA_DIR / "pattern_candidates_filmot.jsonl"


# ---------------------------------------------------------------------------
# API key setup (same as collect/stream_filmot.py)
# ---------------------------------------------------------------------------

def setup_api_key() -> str:
    """Load RapidAPI key from .env or environment."""
    existing = os.environ.get("FILMOT_RAPIDAPI_KEY")
    if existing:
        return existing

    try:
        from dotenv import load_dotenv
        for env_path in [
            Path(__file__).parent.parent.parent / ".env",
            Path(__file__).parent.parent / ".env",
        ]:
            if env_path.exists():
                load_dotenv(env_path)
                break
    except ImportError:
        pass

    rapid_key = os.environ.get("RAPIDAPI_KEY")
    if rapid_key:
        os.environ["FILMOT_RAPIDAPI_KEY"] = rapid_key
        return rapid_key

    filmot_key = os.environ.get("FILMOT_RAPIDAPI_KEY")
    if filmot_key:
        return filmot_key

    raise RuntimeError(
        "No RapidAPI key found. Set RAPIDAPI_KEY in .env or "
        "FILMOT_RAPIDAPI_KEY in environment."
    )


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def load_enriched_seed(path: Path = None):
    """Load enriched seed and return (word_to_emotions, noun_words)."""
    if path is None:
        path = ENRICHED_SEED_PATH

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    word_to_emotions = {}
    for word, emotion in raw.get("adjectives", {}).items():
        word_to_emotions[word] = [emotion] if isinstance(emotion, str) else emotion
    for word, emotion in raw.get("nouns", {}).items():
        word_to_emotions[word] = [emotion] if isinstance(emotion, str) else emotion
    for word, emotion in raw.get("adverbs", {}).items():
        word_to_emotions[word] = [emotion] if isinstance(emotion, str) else emotion

    noun_words = list(raw.get("nouns", {}).keys())
    return word_to_emotions, noun_words


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "completed_queries": [],
            "seen_hashes": [],
            "total_candidates": 0,
            "total_hits_scanned": 0,
            "started_at": datetime.now().isoformat(),
        }
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["checkpoint_at"] = datetime.now().isoformat()
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    temp_path.rename(checkpoint_path)


# ---------------------------------------------------------------------------
# Hit parsing (same as collect/stream_filmot.py)
# ---------------------------------------------------------------------------

def parse_hits(video: dict, query: str) -> List[dict]:
    """Parse a single video API result into one record per subtitle hit."""
    video_id = video.get("id", "")
    if not video_id:
        return []

    subtitle_hits = video.get("hits", video.get("subtitles", []))
    if not subtitle_hits:
        return []

    results = []
    for sub in subtitle_hits:
        ctx_before = sub.get("ctx_before", "")
        token = sub.get("token", "")
        ctx_after = sub.get("ctx_after", "")
        full_context = f"{ctx_before} {token} {ctx_after}".strip()
        if not full_context:
            continue

        hit_start = sub.get("start", 0.0)
        start_seconds = int(float(hit_start))

        results.append({
            "video_id": video_id,
            "video_title": video.get("title", ""),
            "channel_name": video.get("channelname", video.get("channelName", "")),
            "query": query.strip('"'),
            "hit_start": hit_start,
            "hit_dur": sub.get("dur", 0.0),
            "full_context": full_context,
            "youtube_url": f"https://youtube.com/watch?v={video_id}&t={start_seconds}s",
            "view_count": video.get("viewcount", video.get("viewCount", 0)),
            "duration": video.get("duration", 0),
            "upload_date": video.get("uploaddate", video.get("publishDate", "")),
        })

    return results


# ---------------------------------------------------------------------------
# Per-query worker: collect + filter
# ---------------------------------------------------------------------------

def _collect_and_filter_one_query(
    query: str,
    query_index: int,
    total_queries: int,
    matcher,
    max_pages: int,
    delay_pages: float,
    verbose: bool,
) -> dict:
    """
    Collect hits for one query and filter through PatternMatcher.

    Returns dict with:
        candidates: list of candidate dicts (matched)
        total_scanned: number of raw hits checked
    """
    from filmot import Filmot
    client = Filmot()

    candidates = []
    total_scanned = 0
    page = 0
    consecutive_empty = 0

    while page < max_pages:
        try:
            results = client.send_api(
                "getsearchsubtitles",
                {"query": query, "lang": "ro", "page": page},
            )
        except Exception as e:
            if verbose:
                print(f"  {query}: API error page {page}: {e}")
            break

        video_list = (
            results.get("result", []) if isinstance(results, dict)
            else results if isinstance(results, list)
            else []
        )

        if not video_list:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            page += 1
            time.sleep(delay_pages)
            continue

        consecutive_empty = 0

        # Parse hits and filter through PatternMatcher
        for video in video_list:
            hits = parse_hits(video, query)
            for hit in hits:
                total_scanned += 1
                text = hit["full_context"]
                matches = matcher.find_matches(text, extract_sentences=True)
                if not matches:
                    continue

                # One candidate per match
                for match in matches:
                    candidates.append({
                        "id": f"filmot_{hit['video_id']}_{hit['hit_start']}",
                        "text": text,
                        "matched_sentence": match.matched_text,
                        "pattern_used": match.pattern_name,
                        "pattern_category": match.pattern_category,
                        "seed_word": match.seed_word,
                        "seed_word_normalized": match.seed_word_normalized,
                        "emotion_category": match.emotions,
                        "source": "filmot",
                        "video_id": hit["video_id"],
                        "video_title": hit["video_title"],
                        "channel_name": hit["channel_name"],
                        "youtube_url": hit["youtube_url"],
                        "view_count": hit["view_count"],
                    })

        if verbose and page % 10 == 0:
            print(f"  {query}: page {page}, {total_scanned} scanned, "
                  f"{len(candidates)} matched")

        page += 1
        time.sleep(delay_pages)

    if verbose:
        print(f"  {query}: done — {total_scanned} scanned, "
              f"{len(candidates)} matched ({page} pages)")

    return {"candidates": candidates, "total_scanned": total_scanned}


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_filmot(
    output_path: Path = None,
    seed_path: Path = None,
    max_hits: int = 200_000,
    max_pages_per_query: int = 500,
    delay_pages: float = 0.1,
    workers: int = 4,
    include_secondary: bool = True,
    resume: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Query Filmot API with all triggers, filter by PatternMatcher, save candidates.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    setup_api_key()

    from pipeline.utils.pattern_matcher import PatternMatcher, get_filmot_queries_all

    # Load seed + init matcher
    word_to_emotions, noun_words = load_enriched_seed(seed_path)
    print(f"Seed: {len(word_to_emotions)} words ({len(noun_words)} nouns)")
    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words, expand_forms=True)

    # Get queries
    queries = get_filmot_queries_all(include_secondary=include_secondary)

    # Checkpoint
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        seen = set(checkpoint["seen_hashes"])
        total_candidates = checkpoint["total_candidates"]
        total_scanned = checkpoint["total_hits_scanned"]
        completed_queries = set(checkpoint.get("completed_queries", []))
        if verbose:
            print(f"Resuming: {total_candidates} candidates, "
                  f"{len(completed_queries)} queries done")
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        seen = set()
        total_candidates = 0
        total_scanned = 0
        completed_queries = set()

    # Filter out completed queries
    work = []
    for qi, query in enumerate(queries):
        if query in completed_queries:
            if verbose:
                print(f"  Skipping {query} (already completed)")
            continue
        work.append((query, qi))

    if verbose:
        print(f"\nFilmot extraction: {len(work)} queries, {workers} workers, "
              f"max {max_pages_per_query} pages/query")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"

    # Stats
    by_pattern = defaultdict(int)
    by_emotion = defaultdict(int)
    by_query = {}
    unique_seed_words = set()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _collect_and_filter_one_query,
                query=query,
                query_index=qi,
                total_queries=len(queries),
                matcher=matcher,
                max_pages=max_pages_per_query,
                delay_pages=delay_pages,
                verbose=verbose,
            ): query
            for query, qi in work
        }

        with open(output_path, write_mode, encoding="utf-8") as f:
            for future in as_completed(futures):
                query = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  {query}: worker failed: {e}")
                    continue

                query_new = 0
                total_scanned += result["total_scanned"]

                for candidate in result["candidates"]:
                    dedup_key = hashlib.md5(
                        f"{candidate['video_id']}_{candidate.get('matched_sentence', '')}"
                        .encode()
                    ).hexdigest()
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    total_candidates += 1
                    query_new += 1

                    by_pattern[candidate["pattern_used"]] += 1
                    for e in candidate["emotion_category"]:
                        by_emotion[e] += 1
                    unique_seed_words.add(candidate["seed_word_normalized"])

                by_query[query] = query_new
                completed_queries.add(query)

                # Checkpoint
                checkpoint["completed_queries"] = list(completed_queries)
                checkpoint["total_candidates"] = total_candidates
                checkpoint["total_hits_scanned"] = total_scanned
                checkpoint["seen_hashes"] = list(seen)
                save_checkpoint(checkpoint_path, checkpoint)

                if verbose:
                    print(f"  → {query}: {query_new} candidates "
                          f"(total: {total_candidates}, scanned: {total_scanned})")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Filmot extraction complete")
        print(f"{'='*60}")
        print(f"Total scanned: {total_scanned:,}")
        print(f"Total candidates: {total_candidates:,}")
        print(f"Hit rate: {total_candidates/max(total_scanned,1)*100:.2f}%")
        print(f"Unique seed words: {len(unique_seed_words)}")
        print(f"Output: {output_path}")

        print(f"\nBy query:")
        for q, cnt in sorted(by_query.items(), key=lambda x: -x[1]):
            print(f"  {q}: {cnt:,}")

        print(f"\nBy pattern (top 10):")
        for pat, cnt in sorted(by_pattern.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pat}: {cnt:,}")

        print(f"\nBy emotion (top 10):")
        for emo, cnt in sorted(by_emotion.items(), key=lambda x: -x[1])[:10]:
            print(f"  {emo}: {cnt:,}")

    # Save stats
    stats = {
        "total_scanned": total_scanned,
        "total_candidates": total_candidates,
        "unique_seed_words": len(unique_seed_words),
        "by_query": by_query,
        "by_pattern": dict(by_pattern),
        "by_emotion": dict(by_emotion),
        "finished_at": datetime.now().isoformat(),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filmot extraction: query all triggers, filter by PatternMatcher"
    )
    parser.add_argument(
        "--max-hits", type=int, default=200_000,
        help="Max raw hits to scan across all queries (default: 200000)",
    )
    parser.add_argument(
        "--max-pages-per-query", type=int, default=500,
        help="Max API pages per query (default: 500)",
    )
    parser.add_argument(
        "--delay-pages", type=float, default=0.1,
        help="Seconds between API pages per worker (default: 0.1)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel query workers (default: 4)",
    )
    parser.add_argument(
        "--no-secondary", action="store_true",
        help="Skip no-diacritic variant queries",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed", type=Path, default=None,
        help=f"Enriched seed path (default: {ENRICHED_SEED_PATH})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    extract_filmot(
        output_path=args.output,
        seed_path=args.seed,
        max_hits=args.max_hits,
        max_pages_per_query=args.max_pages_per_query,
        delay_pages=args.delay_pages,
        workers=args.workers,
        include_secondary=not args.no_secondary,
        resume=args.resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
