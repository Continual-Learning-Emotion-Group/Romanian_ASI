#!/usr/bin/env python3
"""
FULG extraction: stream the FULG dataset (150B tokens) from HuggingFace,
filter by PatternMatcher + enriched seed, save candidates with sentence-level
context and domain categorization.

Supports parallel workers (--workers N) to stream different shards
concurrently, overcoming the single-stream network bottleneck.

Requires:
    pip install datasets tqdm

Usage:
    python -m pipeline.extract_match.fulg                          # default (50K samples)
    python -m pipeline.extract_match.fulg --max-samples 100000     # more
    python -m pipeline.extract_match.fulg --workers 4              # parallel shards
    python -m pipeline.extract_match.fulg --resume                 # resume from checkpoint
    python -m pipeline.extract_match.fulg --max-records 10000 --max-samples 100  # quick test
"""

import argparse
import hashlib
import json
import re
import signal
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from pipeline.extract_match.run import load_enriched_seed
from pipeline.utils.pattern_matcher import PatternMatcher, get_trigger_words
from pipeline.utils.text_utils import extract_context_window

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "pattern_candidates_fulg.jsonl"
ENRICHED_SEED_PATH = DATA_DIR / "enriched_seed_merged.json"

FULG_DATASET_ID = "faur-ai/fulg"

# Sentinel value to signal worker completion
_WORKER_DONE = None

# ---------------------------------------------------------------------------
# Domain categorization (soft tagging for analysis, NOT filtering)
# Ported from scripts/fulg/extract_candidates.py
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS = {
    "forum": [
        r"forum\.", r"\.forum\.", r"/forum",
        r"softpedia\.ro", r"sfatulmedicului\.ro", r"ciao\.ro",
        r"pcgarage\.ro", r"emag\.ro/forum",
        r"reddit\.com", r"quora\.com",
        r"stackexchange\.com", r"stackoverflow\.com",
    ],
    "social": [
        r"facebook\.com", r"fb\.com", r"twitter\.com", r"x\.com",
        r"instagram\.com", r"tiktok\.com", r"youtube\.com",
        r"linkedin\.com", r"pinterest\.com", r"tumblr\.com",
        r"trilulilu\.ro",
    ],
    "blog": [
        r"blogspot\.", r"wordpress\.com", r"wordpress\.org",
        r"medium\.com", r"substack\.com", r"blogger\.com",
        r"livejournal\.com", r"blog\.", r"\.blog",
    ],
    "qa": [
        r"answers\.", r"ask\.", r"raspunsuri\.",
        r"intrebari\.", r"discutii\.", r"comunitate\.",
        r"yahoo\.com/answers", r"askfm\.com",
    ],
    "news": [
        r"hotnews\.ro", r"digi24\.ro", r"mediafax\.ro",
        r"adevarul\.ro", r"libertatea\.ro", r"gandul\.ro",
        r"stirileprotv\.ro", r"observator\.tv", r"antena3\.ro",
        r"realitatea\.net", r"ziare\.com", r"news\.",
        r"bbc\.com", r"cnn\.com", r"reuters\.com",
    ],
    "wiki": [
        r"wikipedia\.org", r"wikimedia\.org", r"wiktionary\.org",
        r"wiki\.", r"fandom\.com",
    ],
    "reviews": [
        r"tripadvisor\.", r"booking\.com", r"yelp\.com",
        r"trustpilot\.com", r"amazon\.", r"emag\.ro",
        r"reviews\.", r"recenzii\.",
    ],
}


def categorize_domain(domain: str, url: str = "") -> str:
    """Categorize a source domain into a content type (soft tag)."""
    check_str = f"{domain} {url}".lower()
    for category, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, check_str):
                return category
    return "other"


# ---------------------------------------------------------------------------
# Checkpoint helpers (same pattern as filmot.py)
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "records_offset": 0,
            "total_processed": 0,
            "total_matches": 0,
            "filtered_out": 0,
            "filter_reasons": {},
            "duplicates_skipped": 0,
            "seen_hashes": [],
            "seen_domain_sentences": [],
            "by_source_category": {},
            "by_pattern": {},
            "by_emotion": {},
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
# Worker: streams one shard, filters, puts candidates on queue
# ---------------------------------------------------------------------------

def _shard_worker(
    shard_index: int,
    num_shards: int,
    matcher: PatternMatcher,
    trigger_words: Set[str],
    context_sentences: int,
    max_context_length: int,
    queue: Queue,
    stop_event: threading.Event,
):
    """
    Stream one shard of FULG, run pattern matching, put candidates on queue.

    Each item on the queue is either:
    - A dict with candidate info + "_worker_stats" key for filtered/scanned counts
    - _WORKER_DONE sentinel when this worker finishes
    """
    from datasets import load_dataset

    ds = load_dataset(FULG_DATASET_ID, split="train", streaming=True)
    shard = ds.shard(num_shards=num_shards, index=shard_index)

    local_filtered = 0
    local_filter_reasons = defaultdict(int)
    local_scanned = 0

    for record in shard:
        if stop_event.is_set():
            break

        local_scanned += 1

        text = record.get("raw_content", "")
        lang_score = record.get("language_score", 0)

        if lang_score < 0.8:
            local_filtered += 1
            local_filter_reasons["low_language_score"] += 1
            continue
        if len(text) < 100:
            local_filtered += 1
            local_filter_reasons["too_short"] += 1
            continue
        if len(text) > 100_000:
            local_filtered += 1
            local_filter_reasons["too_long"] += 1
            continue

        text_lower = text.lower()
        if not any(tw in text_lower for tw in trigger_words):
            local_filtered += 1
            local_filter_reasons["no_trigger_words"] += 1
            continue

        # Pattern matching
        matches = matcher.find_matches(text, extract_sentences=True)
        if not matches:
            continue

        digest = record.get("digest", "")
        record_id = f"fulg_{digest[:12]}" if digest else f"fulg_{hash(text) & 0xFFFFFFFF:08x}"
        source_domain = record.get("source_domain", "")
        url = record.get("url", "")
        source_category = categorize_domain(source_domain, url)

        seen_in_page = set()
        for match in matches:
            if stop_event.is_set():
                break

            # In-page dedup
            match_key = (record_id, match.matched_text)
            if match_key in seen_in_page:
                continue
            seen_in_page.add(match_key)

            ctx_before, matched_sent, ctx_after = extract_context_window(
                text, match.start_pos, match.end_pos,
                num_sentences=context_sentences,
                max_length=max_context_length,
            )

            context = ""
            if ctx_before:
                context += ctx_before + " "
            context += matched_sent
            if ctx_after:
                context += " " + ctx_after

            candidate = {
                "id": record_id,
                "text": context.strip(),
                "context_before": ctx_before,
                "context_after": ctx_after,
                "matched_sentence": match.matched_text,
                "pattern_used": match.pattern_name,
                "pattern_category": match.pattern_category,
                "seed_word": match.seed_word,
                "seed_word_normalized": match.seed_word_normalized,
                "emotion_category": match.emotions,
                "source": "fulg",
                "source_domain": source_domain,
                "source_category": source_category,
                "url": url,
                "title": record.get("title", ""),
                "full_text_length": len(text),
                # For dedup in main thread
                "_text_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                "_domain_sent_hash": hashlib.md5(
                    f"{source_domain}|{match.matched_text}".encode()
                ).hexdigest(),
            }
            queue.put(candidate)

        # Periodically report filter stats (as deltas)
        if local_scanned % 5000 == 0:
            queue.put({
                "_stats": True,
                "filtered": local_filtered,
                "filter_reasons": dict(local_filter_reasons),
                "scanned": local_scanned,
                "shard": shard_index,
            })
            local_filtered = 0
            local_filter_reasons = defaultdict(int)
            local_scanned = 0

    # Final stats flush (remaining delta)
    queue.put({
        "_stats": True,
        "filtered": local_filtered,
        "filter_reasons": dict(local_filter_reasons),
        "scanned": local_scanned,
        "shard": shard_index,
    })
    queue.put(_WORKER_DONE)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_fulg(
    output_path: Path = None,
    seed_path: Path = None,
    max_samples: int = 50_000,
    max_records: int = 0,
    workers: int = 1,
    context_sentences: int = 2,
    max_context_length: int = 1000,
    checkpoint_every: int = 10_000,
    resume: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Stream FULG, filter by PatternMatcher, save candidates with context.

    With workers > 1, streams multiple shards in parallel for higher throughput.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    # Load seed + init matcher
    word_to_emotions, noun_words = load_enriched_seed(seed_path)
    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words, expand_forms=True)
    trigger_words = get_trigger_words()

    # Checkpoint
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        seen = set(checkpoint["seen_hashes"])
        seen_domain_sentences = set(checkpoint.get("seen_domain_sentences", []))
        if verbose:
            print(f"Resuming: {checkpoint['total_matches']:,} candidates so far")
    else:
        # Fresh run — start with empty state
        checkpoint = {
            "total_processed": 0, "total_matches": 0,
            "filtered_out": 0, "filter_reasons": {},
            "duplicates_skipped": 0,
            "by_source_category": {}, "by_pattern": {}, "by_emotion": {},
            "started_at": datetime.now().isoformat(),
        }
        seen = set()
        seen_domain_sentences = set()

    # Stats (restore from checkpoint or fresh)
    total_processed = checkpoint["total_processed"]
    total_matches = checkpoint["total_matches"]
    filtered_out = checkpoint["filtered_out"]
    filter_reasons = defaultdict(int, checkpoint.get("filter_reasons", {}))
    dupes_skipped = checkpoint["duplicates_skipped"]
    by_category = defaultdict(int, checkpoint.get("by_source_category", {}))
    by_pattern = defaultdict(int, checkpoint.get("by_pattern", {}))
    by_emotion = defaultdict(int, checkpoint.get("by_emotion", {}))
    unique_seed_words = set()
    records_scanned = 0

    # Stop event for graceful shutdown
    stop_event = threading.Event()
    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        stop_event.set()
        if verbose:
            print("\nInterrupted — stopping workers and saving checkpoint...")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"

    if verbose:
        print(f"Streaming FULG from {FULG_DATASET_ID}...")
        print(f"  Workers: {workers} (shards: {workers})")
        print(f"  Trigger filter: {len(trigger_words)} words/phrases")
        print(f"  Max samples: {max_samples:,}"
              + (f", max records: {max_records:,}" if max_records else ""))
        print(f"  Context: {context_sentences} sentences, max {max_context_length} chars")

    pbar = tqdm(
        total=max_samples, desc="FULG extract", unit="cand"
    ) if verbose else None
    if pbar and total_matches > 0:
        pbar.update(total_matches)

    # Launch workers
    queue: Queue = Queue(maxsize=workers * 200)
    threads: List[threading.Thread] = []
    for i in range(workers):
        t = threading.Thread(
            target=_shard_worker,
            args=(
                i, workers, matcher, trigger_words,
                context_sentences, max_context_length,
                queue, stop_event,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    workers_done = 0
    candidates_since_checkpoint = 0

    with open(output_path, write_mode, encoding="utf-8") as f:
        while workers_done < workers:
            try:
                item = queue.get(timeout=1.0)
            except Empty:
                continue

            # Worker finished
            if item is _WORKER_DONE:
                workers_done += 1
                continue

            # Stats update from worker
            if item.get("_stats"):
                filtered_out += item["filtered"]
                for reason, cnt in item["filter_reasons"].items():
                    filter_reasons[reason] += cnt
                records_scanned += item["scanned"]
                continue

            # Candidate — apply cross-worker dedup
            text_hash = item.pop("_text_hash")
            domain_sent_hash = item.pop("_domain_sent_hash")

            # Boilerplate dedup: same sentence from same domain → skip
            if domain_sent_hash in seen_domain_sentences:
                continue
            seen_domain_sentences.add(domain_sent_hash)

            # Track processed pages (for stats; a page may yield multiple candidates)
            if text_hash not in seen:
                seen.add(text_hash)
                total_processed += 1

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_matches += 1
            candidates_since_checkpoint += 1

            by_category[item["source_category"]] += 1
            by_pattern[item["pattern_used"]] += 1
            for e in item["emotion_category"]:
                by_emotion[e] += 1
            unique_seed_words.add(item["seed_word_normalized"])

            if pbar:
                pbar.n = total_matches
                pbar.set_postfix(
                    processed=f"{total_processed:,}",
                    scanned=f"{records_scanned:,}",
                )
                pbar.refresh()

            # Max limits
            if total_matches >= max_samples:
                stop_event.set()
                break
            if max_records > 0 and total_processed >= max_records:
                stop_event.set()
                break

            # Checkpoint
            if candidates_since_checkpoint >= checkpoint_every:
                checkpoint.update({
                    "total_processed": total_processed,
                    "total_matches": total_matches,
                    "filtered_out": filtered_out,
                    "filter_reasons": dict(filter_reasons),
                    "duplicates_skipped": dupes_skipped,
                    "seen_hashes": list(seen),
                    "seen_domain_sentences": list(seen_domain_sentences),
                    "by_source_category": dict(by_category),
                    "by_pattern": dict(by_pattern),
                    "by_emotion": dict(by_emotion),
                })
                save_checkpoint(checkpoint_path, checkpoint)
                candidates_since_checkpoint = 0

    if pbar:
        pbar.close()

    # Signal workers to stop and drain queue so they can exit
    # (workers may be blocked on queue.put() if queue is full)
    stop_event.set()
    while any(t.is_alive() for t in threads):
        try:
            item = queue.get(timeout=0.5)
            # Still collect stats from draining
            if item is _WORKER_DONE:
                continue
            if isinstance(item, dict) and item.get("_stats"):
                filtered_out += item["filtered"]
                for reason, cnt in item["filter_reasons"].items():
                    filter_reasons[reason] += cnt
                records_scanned += item["scanned"]
        except Empty:
            continue
    for t in threads:
        t.join(timeout=2.0)

    signal.signal(signal.SIGINT, prev_handler)

    # Final checkpoint
    checkpoint.update({
        "total_processed": total_processed,
        "total_matches": total_matches,
        "filtered_out": filtered_out,
        "filter_reasons": dict(filter_reasons),
        "duplicates_skipped": dupes_skipped,
        "seen_hashes": list(seen),
        "seen_domain_sentences": list(seen_domain_sentences),
        "by_source_category": dict(by_category),
        "by_pattern": dict(by_pattern),
        "by_emotion": dict(by_emotion),
    })
    save_checkpoint(checkpoint_path, checkpoint)

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"FULG extraction {'interrupted' if interrupted else 'complete'}")
        print(f"{'='*60}")
        print(f"Records scanned: {records_scanned:,}")
        print(f"Records processed (after filter): {total_processed:,}")
        print(f"Filtered out: {filtered_out:,}")
        if filter_reasons:
            for reason, cnt in sorted(filter_reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {cnt:,}")
        print(f"Duplicates skipped: {dupes_skipped:,}")
        print(f"Total candidates: {total_matches:,}")
        print(f"Unique seed words: {len(unique_seed_words)}")
        print(f"Output: {output_path}")

        if by_category:
            print(f"\nBy source category:")
            for cat, cnt in sorted(by_category.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {cnt:,}")

        if by_pattern:
            print(f"\nBy pattern (top 10):")
            for pat, cnt in sorted(by_pattern.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pat}: {cnt:,}")

        if by_emotion:
            print(f"\nBy emotion (top 10):")
            for emo, cnt in sorted(by_emotion.items(), key=lambda x: -x[1])[:10]:
                print(f"  {emo}: {cnt:,}")

    # Save stats
    stats = {
        "records_scanned": records_scanned,
        "total_processed": total_processed,
        "filtered_out": filtered_out,
        "filter_reasons": dict(filter_reasons),
        "duplicates_skipped": dupes_skipped,
        "total_candidates": total_matches,
        "unique_seed_words": len(unique_seed_words),
        "by_source_category": dict(by_category),
        "by_pattern": dict(by_pattern),
        "by_emotion": dict(by_emotion),
        "finished_at": datetime.now().isoformat(),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="FULG extraction: stream, filter by PatternMatcher, save candidates"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50_000,
        help="Stop after extracting this many candidates (default: 50000)",
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Max source records to process after filtering (0 = unlimited)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel shard workers (default: 1)",
    )
    parser.add_argument(
        "--context-sentences", type=int, default=2,
        help="Sentences of context before/after match (default: 2)",
    )
    parser.add_argument(
        "--max-context-length", type=int, default=1000,
        help="Max total context length in chars (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10_000,
        help="Save checkpoint every N candidates (default: 10000)",
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

    extract_fulg(
        output_path=args.output,
        seed_path=args.seed,
        max_samples=args.max_samples,
        max_records=args.max_records,
        workers=args.workers,
        context_sentences=args.context_sentences,
        max_context_length=args.max_context_length,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
