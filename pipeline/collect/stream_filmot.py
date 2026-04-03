#!/usr/bin/env python3
"""
Stream raw subtitle hits from the Filmot API (RapidAPI).

No filtering — just collects N subtitle hits matching Romanian trigger phrases
and saves them as JSONL. Filtering happens in a later pipeline stage.

Queries are derived from pipeline.utils.pattern_matcher (single source of truth).

Requires:
    pip install filmot python-dotenv
    RAPIDAPI_KEY in .env or FILMOT_RAPIDAPI_KEY in environment

Usage:
    python -m pipeline.collect.stream_filmot
    python -m pipeline.collect.stream_filmot --max-hits 100000
    python -m pipeline.collect.stream_filmot --max-pages-per-query 200
    python -m pipeline.collect.stream_filmot --no-secondary
    python -m pipeline.collect.stream_filmot --resume
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).parent.parent / "data"


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


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint from previous run."""
    if not checkpoint_path.exists():
        return {
            "completed_pages": {},
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
            "ctx_before": ctx_before,
            "token": token,
            "ctx_after": ctx_after,
            "full_context": full_context,
            "youtube_url": f"https://youtube.com/watch?v={video_id}&t={start_seconds}s",
            "view_count": video.get("viewcount", video.get("viewCount", 0)),
            "duration": video.get("duration", 0),
            "upload_date": video.get("uploaddate", video.get("publishDate", "")),
        })

    return results


def stream_filmot(
    output_path: Path,
    max_hits: int = 50_000,
    max_pages_per_query: int = 500,
    delay_pages: float = 0.5,
    delay_queries: float = 2.0,
    include_secondary: bool = True,
    resume: bool = False,
    verbose: bool = True,
):
    """
    Stream Filmot API hits and save as JSONL.

    Queries are sourced from pipeline.utils.pattern_matcher.get_filmot_queries().

    Args:
        output_path: Where to write the JSONL output.
        max_hits: Stop after this many total saved hits.
        max_pages_per_query: Max API pages per query (50 results/page).
        delay_pages: Seconds between API pages.
        delay_queries: Seconds between queries.
        include_secondary: Include no-diacritic variant queries.
        resume: Resume from checkpoint.
        verbose: Print progress.
    """
    setup_api_key()

    from filmot import Filmot
    from pipeline.utils.pattern_matcher import get_filmot_queries

    client = Filmot()
    queries = get_filmot_queries(include_secondary=include_secondary)

    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        seen = set(checkpoint["seen_hashes"])
        total = checkpoint["total_hits"]
        if verbose:
            print(f"Resuming from checkpoint: {total:,} hits, {len(seen):,} seen")
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        seen = set()
        total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"

    stats_by_query = {}

    with open(output_path, write_mode, encoding="utf-8") as f:
        for qi, query in enumerate(queries):
            if total >= max_hits:
                break

            # Skip completed queries on resume
            completed_page = checkpoint["completed_pages"].get(query, -1)
            start_page = completed_page + 1 if resume else 0

            if verbose:
                print(f"\n[{qi + 1}/{len(queries)}] Query: {query}"
                      f"{f' (resuming from page {start_page})' if start_page > 0 else ''}")

            page = start_page
            consecutive_empty = 0
            query_hits = 0

            while page < max_pages_per_query and total < max_hits:
                try:
                    results = client.send_api(
                        "getsearchsubtitles",
                        {"query": query, "lang": "ro", "page": page},
                    )
                    checkpoint["total_api_calls"] = checkpoint.get("total_api_calls", 0) + 1
                except Exception as e:
                    print(f"  API error page {page}: {e}")
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

                for video in video_list:
                    for hit in parse_hits(video, query):
                        dedup_key = hashlib.md5(
                            f"{hit['video_id']}_{hit['hit_start']}".encode()
                        ).hexdigest()
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        f.write(json.dumps(hit, ensure_ascii=False) + "\n")
                        total += 1
                        query_hits += 1

                        if total >= max_hits:
                            break
                    if total >= max_hits:
                        break

                if verbose and page % 10 == 0:
                    print(f"  page {page}, {query_hits:,} hits so far (total: {total:,})")

                # Update checkpoint every page
                checkpoint["completed_pages"][query] = page
                checkpoint["total_hits"] = total
                checkpoint["seen_hashes"] = list(seen)
                save_checkpoint(checkpoint_path, checkpoint)

                page += 1
                time.sleep(delay_pages)

            stats_by_query[query] = query_hits
            if verbose:
                print(f"  {query_hits:,} new hits (total: {total:,})")

            if qi < len(queries) - 1 and total < max_hits:
                time.sleep(delay_queries)

    # Save final stats
    stats = {
        "total_hits": total,
        "total_api_calls": checkpoint.get("total_api_calls", 0),
        "unique_videos": total,
        "queries_completed": len([q for q in stats_by_query]),
        "stats_by_query": stats_by_query,
        "started_at": checkpoint.get("started_at", ""),
        "finished_at": datetime.now().isoformat(),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {total:,} hits saved to {output_path}")
    print(f"Stats: {stats_path}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Stream raw Filmot subtitle hits (no filtering)"
    )
    parser.add_argument(
        "--max-hits", type=int, default=50_000,
        help="Total hits to collect (default: 50000)",
    )
    parser.add_argument(
        "--max-pages-per-query", type=int, default=500,
        help="Max API pages per query (default: 500)",
    )
    parser.add_argument(
        "--delay-pages", type=float, default=0.5,
        help="Seconds between API pages (default: 0.5)",
    )
    parser.add_argument(
        "--delay-queries", type=float, default=2.0,
        help="Seconds between queries (default: 2.0)",
    )
    parser.add_argument(
        "--no-secondary", action="store_true",
        help="Skip secondary (no-diacritic) queries",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    output = args.output or DATA_DIR / "filmot_raw.jsonl"

    stream_filmot(
        output_path=output,
        max_hits=args.max_hits,
        max_pages_per_query=args.max_pages_per_query,
        delay_pages=args.delay_pages,
        delay_queries=args.delay_queries,
        include_secondary=not args.no_secondary,
        resume=args.resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
