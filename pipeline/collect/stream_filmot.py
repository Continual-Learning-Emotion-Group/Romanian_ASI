#!/usr/bin/env python3
"""
Stream raw subtitle hits from the Filmot API (RapidAPI).

No filtering — just collects N subtitle hits matching Romanian trigger phrases
and saves them as JSONL. Filtering happens in a later pipeline stage.

Requires:
    pip install filmot python-dotenv
    RAPIDAPI_KEY in .env or FILMOT_RAPIDAPI_KEY in environment

Usage:
    python -m pipeline.collect.stream_filmot
    python -m pipeline.collect.stream_filmot --max-hits 10000
    python -m pipeline.collect.stream_filmot --max-pages-per-query 10
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).parent.parent / "data"

# Trigger phrases for Romanian "I feel" patterns.
# Broad verb phrases WITHOUT emotion words — the API returns subtitle context
# around each match, which we filter in a later stage.
QUERIES = [
    '"mă simt"',
    '"mi-e"',
    '"m-am simțit"',
    '"îmi este"',
    '"mă simțeam"',
    '"ne simțim"',
    '"ne-am simțit"',
    '"simt că"',
    '"îmi era"',
    '"mă voi simți"',
    # No-diacritic variants (common in YouTube subtitles)
    '"ma simt"',
    '"m-am simtit"',
    '"imi este"',
    '"imi era"',
    '"ma simteam"',
    '"ne simtim"',
    '"ne-am simtit"',
]


def setup_api_key() -> str:
    """Load RapidAPI key from .env or environment."""
    existing = os.environ.get("FILMOT_RAPIDAPI_KEY")
    if existing:
        return existing

    try:
        from dotenv import load_dotenv
        # Check both pipeline root and project root
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
    verbose: bool = True,
):
    """
    Stream Filmot API hits and save as JSONL.

    Args:
        output_path: Where to write the JSONL output.
        max_hits: Stop after this many total saved hits.
        max_pages_per_query: Max API pages per query (50 results/page).
        delay_pages: Seconds between API pages.
        delay_queries: Seconds between queries.
        verbose: Print progress.
    """
    setup_api_key()

    from filmot import Filmot
    client = Filmot()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for qi, query in enumerate(QUERIES):
            if total >= max_hits:
                break

            if verbose:
                print(f"[{qi + 1}/{len(QUERIES)}] Query: {query}")

            page = 0
            consecutive_empty = 0

            while page < max_pages_per_query and total < max_hits:
                try:
                    results = client.send_api(
                        "getsearchsubtitles",
                        {"query": query, "lang": "ro", "page": page},
                    )
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

                        if total >= max_hits:
                            break
                    if total >= max_hits:
                        break

                page += 1
                time.sleep(delay_pages)

            if verbose:
                print(f"  Total so far: {total:,}")

            if qi < len(QUERIES) - 1 and total < max_hits:
                time.sleep(delay_queries)

    print(f"\nDone: {total:,} hits saved")
    print(f"Output: {output_path}")
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
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
