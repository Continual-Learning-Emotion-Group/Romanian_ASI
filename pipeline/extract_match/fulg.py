#!/usr/bin/env python3
"""
FULG extraction: stream the FULG dataset (150B tokens) from HuggingFace,
filter by PatternMatcher + enriched seed, save candidates with sentence-level
context and domain categorization.

Requires:
    pip install datasets tqdm

Usage:
    python -m pipeline.extract_match.fulg                          # default (50K samples)
    python -m pipeline.extract_match.fulg --max-samples 100000     # more
    python -m pipeline.extract_match.fulg --max-records 5000000    # limit scanned
    python -m pipeline.extract_match.fulg --resume                 # resume from checkpoint
    python -m pipeline.extract_match.fulg --context-sentences 3    # wider context
    python -m pipeline.extract_match.fulg --max-records 10000 --max-samples 100  # quick test
"""

import argparse
import hashlib
import json
import re
import signal
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Set

from tqdm import tqdm

from pipeline.extract_match.run import load_enriched_seed
from pipeline.utils.pattern_matcher import PatternMatcher, get_trigger_words
from pipeline.utils.text_utils import extract_context_window

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "pattern_candidates_fulg.jsonl"
ENRICHED_SEED_PATH = DATA_DIR / "enriched_seed_merged.json"

FULG_DATASET_ID = "faur-ai/fulg"

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
# FULG streaming
# ---------------------------------------------------------------------------

def stream_fulg_raw(
    min_language_score: float = 0.8,
    min_text_length: int = 100,
    max_text_length: int = 100_000,
    trigger_words: Set[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream FULG records from HuggingFace with pre-filters.

    Yields full records (dict) with all metadata intact.
    """
    from datasets import load_dataset

    ds = load_dataset(FULG_DATASET_ID, split="train", streaming=True)

    for record in ds:
        text = record.get("raw_content", "")
        lang_score = record.get("language_score", 0)

        if lang_score < min_language_score:
            yield {"_skip": "low_language_score"}
            continue
        if len(text) < min_text_length:
            yield {"_skip": "too_short"}
            continue
        if len(text) > max_text_length:
            yield {"_skip": "too_long"}
            continue

        if trigger_words:
            text_lower = text.lower()
            if not any(tw in text_lower for tw in trigger_words):
                yield {"_skip": "no_trigger_words"}
                continue

        digest = record.get("digest", "")
        yield {
            "id": f"fulg_{digest[:12]}" if digest else f"fulg_{hash(text) & 0xFFFFFFFF:08x}",
            "text": text,
            "url": record.get("url", ""),
            "title": record.get("title", ""),
            "source_domain": record.get("source_domain", ""),
            "language_score": lang_score,
            "length": record.get("length", len(text)),
        }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_fulg(
    output_path: Path = None,
    seed_path: Path = None,
    max_samples: int = 50_000,
    max_records: int = 0,
    context_sentences: int = 2,
    max_context_length: int = 1000,
    checkpoint_every: int = 10_000,
    resume: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Stream FULG, filter by PatternMatcher, save candidates with context.
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
        start_offset = checkpoint["records_offset"]
        if verbose:
            print(f"Resuming from record {start_offset:,}, "
                  f"{checkpoint['total_matches']:,} candidates so far")
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        seen = set()
        start_offset = 0

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

    # Graceful Ctrl+C
    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        if verbose:
            print("\nInterrupted — saving checkpoint...")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"

    if verbose:
        print(f"Streaming FULG from {FULG_DATASET_ID}...")
        print(f"  Trigger filter: {len(trigger_words)} words/phrases")
        print(f"  Max samples: {max_samples:,}"
              + (f", max records: {max_records:,}" if max_records else ""))
        print(f"  Context: {context_sentences} sentences, max {max_context_length} chars")

    pbar = tqdm(
        total=max_samples, desc="FULG extract", unit="cand"
    ) if verbose else None
    if pbar and total_matches > 0:
        pbar.update(total_matches)

    records_seen = 0

    with open(output_path, write_mode, encoding="utf-8") as f:
        for record in stream_fulg_raw(
            trigger_words=trigger_words,
        ):
            if interrupted:
                break

            # Skip records until we reach the resume offset
            if records_seen < start_offset:
                records_seen += 1
                continue
            records_seen += 1

            # Handle filtered records
            skip_reason = record.get("_skip")
            if skip_reason:
                filtered_out += 1
                filter_reasons[skip_reason] += 1
                continue

            # Max records limit
            if max_records > 0 and total_processed >= max_records:
                break

            # Max samples limit
            if total_matches >= max_samples:
                break

            text = record["text"]

            # Dedup by text hash
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash in seen:
                dupes_skipped += 1
                continue
            seen.add(text_hash)
            total_processed += 1

            # Pattern matching
            matches = matcher.find_matches(text, extract_sentences=True)
            if not matches:
                continue

            source_domain = record.get("source_domain", "")
            url = record.get("url", "")
            source_category = categorize_domain(source_domain, url)

            for match in matches:
                if total_matches >= max_samples:
                    break

                # Extract sentence-level context
                ctx_before, matched_sent, ctx_after = extract_context_window(
                    text,
                    match.start_pos,
                    match.end_pos,
                    num_sentences=context_sentences,
                    max_length=max_context_length,
                )

                # Build context string
                context = ""
                if ctx_before:
                    context += ctx_before + " "
                context += matched_sent
                if ctx_after:
                    context += " " + ctx_after

                candidate = {
                    "id": record["id"],
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
                }

                f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                total_matches += 1

                by_category[source_category] += 1
                by_pattern[match.pattern_name] += 1
                for e in match.emotions:
                    by_emotion[e] += 1
                unique_seed_words.add(match.seed_word_normalized)

            # Checkpoint
            if total_processed % checkpoint_every == 0:
                checkpoint.update({
                    "records_offset": records_seen,
                    "total_processed": total_processed,
                    "total_matches": total_matches,
                    "filtered_out": filtered_out,
                    "filter_reasons": dict(filter_reasons),
                    "duplicates_skipped": dupes_skipped,
                    "seen_hashes": list(seen),
                    "by_source_category": dict(by_category),
                    "by_pattern": dict(by_pattern),
                    "by_emotion": dict(by_emotion),
                })
                save_checkpoint(checkpoint_path, checkpoint)

            if pbar:
                pbar.n = total_matches
                pbar.set_postfix(
                    processed=f"{total_processed:,}",
                    scanned=f"{records_seen:,}",
                )
                pbar.refresh()

    if pbar:
        pbar.close()

    # Restore original signal handler
    signal.signal(signal.SIGINT, prev_handler)

    # Final checkpoint
    checkpoint.update({
        "records_offset": records_seen,
        "total_processed": total_processed,
        "total_matches": total_matches,
        "filtered_out": filtered_out,
        "filter_reasons": dict(filter_reasons),
        "duplicates_skipped": dupes_skipped,
        "seen_hashes": list(seen),
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
        print(f"Records scanned: {records_seen:,}")
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
        "records_scanned": records_seen,
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
        "--context-sentences", type=int, default=2,
        help="Sentences of context before/after match (default: 2)",
    )
    parser.add_argument(
        "--max-context-length", type=int, default=1000,
        help="Max total context length in chars (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10_000,
        help="Save checkpoint every N processed records (default: 10000)",
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
        context_sentences=args.context_sentences,
        max_context_length=args.max_context_length,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
