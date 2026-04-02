"""
Unified JSONL corpus reader for the ASI pipeline.

Reads all JSONL files from pipeline/data/ and yields (record_id, text, source)
tuples. Handles different text field names across data sources.
"""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"

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
