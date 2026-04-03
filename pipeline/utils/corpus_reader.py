"""
Unified corpus reader for the ASI pipeline.

Reads JSONL files from pipeline/data/ and/or streams from HuggingFace (FULG).
Yields (record_id, text, source) tuples.
"""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"

FULG_DATASET_ID = "faur-ai/fulg"

# Text field names in order of preference
TEXT_FIELDS = ["text", "full_context", "raw_content"]


def _get_text(record: dict) -> str:
    """Extract text from a record, trying multiple field names."""
    for field in TEXT_FIELDS:
        val = record.get(field, "")
        if val:
            return val
    return ""


def iter_corpus(
    data_dir: Path = None,
    sources: Optional[List[str]] = None,
    trigger_words: Optional[Set[str]] = None,
) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (record_id, text, source_filename) from all JSONL files.

    Args:
        data_dir: Directory to scan for *.jsonl files.
        sources: If given, only read files whose stem matches one of these
                 (e.g., ["merged_corpus", "fulg_raw"]).
        trigger_words: If given, skip records that don't contain any of these
                       words/phrases (case-insensitive check on raw text).
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if not data_dir.exists():
        print(f"Warning: data directory does not exist: {data_dir}")
        return

    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if sources is not None:
        source_set = set(sources)
        jsonl_files = [f for f in jsonl_files if f.stem in source_set]

    if not jsonl_files:
        print(f"Warning: no JSONL files found in {data_dir}")
        return

    for fpath in jsonl_files:
        source_name = fpath.stem
        print(f"  Reading {fpath.name}...")
        count = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                text = _get_text(record)
                if not text:
                    continue

                # Optional trigger word filter
                if trigger_words:
                    text_lower = text.lower()
                    if not any(tw in text_lower for tw in trigger_words):
                        continue

                record_id = record.get("id", f"{source_name}_{count}")
                count += 1
                yield (record_id, text, source_name)

        print(f"    → {count:,} records")


def stream_fulg(
    max_records: int = 0,
    min_language_score: float = 0.8,
    min_text_length: int = 100,
    trigger_words: Optional[Set[str]] = None,
    progress: bool = True,
) -> Iterator[Tuple[str, str, str]]:
    """
    Stream FULG records directly from HuggingFace (no intermediate files).

    Args:
        max_records: Stop after this many yielded records (0 = unlimited).
        min_language_score: Skip records below this Romanian confidence.
        min_text_length: Skip records shorter than this.
        trigger_words: If given, skip records without any trigger word.
        progress: Show tqdm progress bar.

    Yields:
        (record_id, text, "fulg") tuples.
    """
    from datasets import load_dataset

    print(f"  Streaming FULG from {FULG_DATASET_ID}...")
    if trigger_words:
        print(f"    Trigger filter: {len(trigger_words)} words/phrases")
    ds = load_dataset(FULG_DATASET_ID, split="train", streaming=True)

    yielded = 0
    skipped = 0

    pbar = None
    if progress and max_records > 0:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_records, desc="FULG", unit="rec")
        except ImportError:
            pass

    for i, record in enumerate(ds):
        text = record.get("raw_content", "")
        lang_score = record.get("language_score", 0)

        if lang_score < min_language_score:
            skipped += 1
            continue

        if len(text) < min_text_length:
            skipped += 1
            continue

        if trigger_words:
            text_lower = text.lower()
            if not any(tw in text_lower for tw in trigger_words):
                skipped += 1
                continue

        digest = record.get("digest", "")
        record_id = f"fulg_{digest[:12]}" if digest else f"fulg_{i}"
        yielded += 1

        if pbar:
            pbar.update(1)
            pbar.set_postfix(skipped=skipped, scanned=i + 1)

        yield (record_id, text, "fulg")

        if max_records > 0 and yielded >= max_records:
            break

    if pbar:
        pbar.close()

    print(f"    FULG done: {yielded:,} yielded, {skipped:,} skipped (scanned {i + 1:,})")
