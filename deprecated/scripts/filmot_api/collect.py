#!/usr/bin/env python3
"""
Phase 1: Collect raw subtitle hits from Filmot API.

Uses the filmot Python package (RapidAPI) to search for Romanian subtitle
content matching trigger phrases. Saves all hits as raw JSONL for local
filtering in Phase 2.

Features:
- Paginated API search (50 results/page)
- Checkpoint/resume support per (query, page)
- Deduplication by video_id + hit_start
- Rate limiting between pages and queries
- Atomic checkpoint writes

Usage:
    python -m scripts.filmot_api.collect
    python -m scripts.filmot_api.collect --resume
    python -m scripts.filmot_api.collect --max-pages-per-query 20
    python -m scripts.filmot_api.collect --no-secondary
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .config import FilmotAPIConfig, get_trigger_queries, setup_api_key


def make_hit_id(video_id: str, hit_start: float) -> str:
    """Create dedup key from video_id and hit timestamp."""
    raw = f"{video_id}_{hit_start}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint from previous run."""
    if not checkpoint_path.exists():
        return {
            "completed_pages": {},  # {query: last_completed_page}
            "seen_hashes": [],
            "total_hits": 0,
            "total_api_calls": 0,
            "started_at": datetime.now().isoformat(),
        }

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]):
    """Atomic checkpoint write via temp file + rename."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["checkpoint_at"] = datetime.now().isoformat()

    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    temp_path.rename(checkpoint_path)


def parse_video_result(video: Dict, query: str) -> list:
    """
    Parse a single video result from the API into raw hit records.

    Each video can have multiple subtitle hits (in the 'hits' list).
    Returns one record per hit.
    """
    hits = []
    video_id = video.get("id", "")
    if not video_id:
        return hits

    # Video metadata (API uses lowercase field names)
    video_title = video.get("title", "")
    channel_name = video.get("channelname", video.get("channelName", ""))
    channel_id = video.get("channelid", video.get("channelId", ""))
    view_count = video.get("viewcount", video.get("viewCount", 0))
    like_count = video.get("likecount", video.get("likeCount", 0))
    duration = video.get("duration", 0)
    upload_date = video.get("uploaddate", video.get("publishDate", ""))
    category = video.get("category", "")

    # Parse subtitle hits (API uses "hits" key)
    subtitle_hits = video.get("hits", video.get("subtitles", []))
    if not subtitle_hits:
        return hits

    for sub in subtitle_hits:
        hit_start = sub.get("start", 0.0)
        hit_dur = sub.get("dur", 0.0)
        ctx_before = sub.get("ctx_before", "")
        token = sub.get("token", "")
        ctx_after = sub.get("ctx_after", "")

        full_context = f"{ctx_before} {token} {ctx_after}".strip()
        if not full_context:
            continue

        # Build YouTube URL with timestamp
        start_seconds = int(float(hit_start))
        youtube_url = f"https://youtube.com/watch?v={video_id}&t={start_seconds}s"

        hit = {
            "video_id": video_id,
            "video_title": video_title,
            "channel_name": channel_name,
            "channel_id": channel_id,
            "view_count": view_count,
            "like_count": like_count,
            "duration": duration,
            "upload_date": upload_date,
            "category": category,
            "query": query,
            "hit_start": hit_start,
            "hit_dur": hit_dur,
            "ctx_before": ctx_before,
            "token": token,
            "ctx_after": ctx_after,
            "full_context": full_context,
            "youtube_url": youtube_url,
        }
        hits.append(hit)

    return hits


def collect_hits(
    config: FilmotAPIConfig,
    resume: bool = False,
    include_secondary: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main collection loop: paginated API search across all trigger queries.

    Args:
        config: API configuration.
        resume: Resume from checkpoint.
        include_secondary: Include secondary (no-diacritic) queries.
        verbose: Print progress.

    Returns:
        Final statistics dictionary.
    """
    # Setup API key
    api_key = setup_api_key()
    print(f"API key loaded ({api_key[:8]}...)")

    # Import filmot
    from filmot import Filmot
    client = Filmot()

    # Load checkpoint
    if resume:
        checkpoint = load_checkpoint(config.checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint['total_hits']:,} hits, "
              f"{checkpoint['total_api_calls']:,} API calls")
    else:
        checkpoint = load_checkpoint(Path("/nonexistent"))

    completed_pages = checkpoint.get("completed_pages", {})
    seen_hashes: Set[str] = set(checkpoint.get("seen_hashes", []))
    total_hits = checkpoint.get("total_hits", 0)
    total_api_calls = checkpoint.get("total_api_calls", 0)
    stats_by_query: Dict[str, int] = defaultdict(int, checkpoint.get("stats_by_query", {}))

    queries = get_trigger_queries(include_secondary=include_secondary)

    print(f"\nFilmot API Collection")
    print("=" * 70)
    print(f"Queries: {len(queries)}")
    print(f"Max pages/query: {config.max_pages_per_query}")
    print(f"Output: {config.output_raw_path}")
    print()

    # Ensure output directory exists
    config.output_raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output file in append mode
    with open(config.output_raw_path, "a", encoding="utf-8") as out_f:
        for qi, query in enumerate(queries):
            last_page = completed_pages.get(query, -1)

            if verbose:
                print(f"\n[{qi + 1}/{len(queries)}] Query: {query}")
                if last_page >= 0:
                    print(f"  Resuming from page {last_page + 1}")

            page = last_page + 1
            consecutive_empty = 0
            query_hits = 0

            while True:
                # Check page limit
                if config.max_pages_per_query > 0 and page >= config.max_pages_per_query:
                    if verbose:
                        print(f"  Reached max pages ({config.max_pages_per_query})")
                    break

                # API call
                try:
                    results = client.send_api(
                        "getsearchsubtitles",
                        {
                            "query": query,
                            "lang": config.language,
                            "page": page,
                        },
                    )
                    total_api_calls += 1
                except Exception as e:
                    print(f"  API error on page {page}: {e}")
                    # Save checkpoint and skip to next query on persistent errors
                    break

                # Response is {"totalresultcount": N, "result": [...]}
                if isinstance(results, dict):
                    video_list = results.get("result", [])
                elif isinstance(results, list):
                    video_list = results
                else:
                    video_list = []

                # Empty results = query exhausted
                if not video_list:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        if verbose:
                            print(f"  Query exhausted at page {page}")
                        break
                    page += 1
                    time.sleep(config.delay_between_pages)
                    continue

                consecutive_empty = 0

                # Parse results
                page_hits = 0
                page_dupes = 0

                for video in video_list:
                    hits = parse_video_result(video, query.strip('"'))

                    for hit in hits:
                        hit_hash = make_hit_id(hit["video_id"], hit["hit_start"])

                        if hit_hash in seen_hashes:
                            page_dupes += 1
                            continue

                        seen_hashes.add(hit_hash)
                        out_f.write(json.dumps(hit, ensure_ascii=False) + "\n")
                        total_hits += 1
                        page_hits += 1
                        query_hits += 1

                if verbose and page % 10 == 0:
                    print(f"  Page {page}: {page_hits} new hits, "
                          f"{page_dupes} dupes | Total: {total_hits:,}")

                # Update checkpoint
                completed_pages[query] = page

                # Save checkpoint periodically
                if page % 50 == 0:
                    checkpoint.update({
                        "completed_pages": completed_pages,
                        "seen_hashes": list(seen_hashes),
                        "total_hits": total_hits,
                        "total_api_calls": total_api_calls,
                        "stats_by_query": dict(stats_by_query),
                    })
                    save_checkpoint(config.checkpoint_path, checkpoint)
                    out_f.flush()

                page += 1
                time.sleep(config.delay_between_pages)

            stats_by_query[query] = query_hits
            if verbose:
                print(f"  Query total: {query_hits:,} hits")

            # Save checkpoint between queries
            checkpoint.update({
                "completed_pages": completed_pages,
                "seen_hashes": list(seen_hashes),
                "total_hits": total_hits,
                "total_api_calls": total_api_calls,
                "stats_by_query": dict(stats_by_query),
            })
            save_checkpoint(config.checkpoint_path, checkpoint)
            out_f.flush()

            # Delay between queries
            if qi < len(queries) - 1:
                time.sleep(config.delay_between_queries)

    # Final stats
    final_stats = {
        "total_hits": total_hits,
        "total_api_calls": total_api_calls,
        "unique_videos": len({h.split("_")[0] for h in seen_hashes}),
        "queries_completed": len(completed_pages),
        "stats_by_query": dict(stats_by_query),
        "started_at": checkpoint.get("started_at", ""),
        "finished_at": datetime.now().isoformat(),
    }

    # Save final stats
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Collection complete")
    print(f"  Total hits: {total_hits:,}")
    print(f"  API calls: {total_api_calls:,}")
    print(f"  Output: {config.output_raw_path}")
    print(f"  Stats: {config.stats_path}")

    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect raw subtitle hits from Filmot API"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=500,
        help="Maximum pages per query (default: 500, 0 = unlimited)",
    )
    parser.add_argument(
        "--no-secondary",
        action="store_true",
        help="Skip secondary (no-diacritic) queries",
    )
    parser.add_argument(
        "--delay-pages",
        type=float,
        default=0.5,
        help="Delay between API pages in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--delay-queries",
        type=float,
        default=2.0,
        help="Delay between queries in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    config = FilmotAPIConfig(
        max_pages_per_query=args.max_pages_per_query,
        delay_between_pages=args.delay_pages,
        delay_between_queries=args.delay_queries,
    )

    try:
        collect_hits(
            config=config,
            resume=args.resume,
            include_secondary=not args.no_secondary,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Checkpoint saved. Resume with --resume flag.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
