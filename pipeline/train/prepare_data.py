"""Build the multilingual SFT DatasetDict from presentation_data CSVs.

Reads `presentation_data/presentation_{ro,en,es,fa,hi}/{train,val,test}.csv`,
tags each row with its `language`, then:
- train: concatenate all 5 languages (5000 rows) and shuffle with seed=42
- val:   concatenate all 5 languages (1250 rows) — used during training for
         best-checkpoint selection
- test_<lang>: one 1000-row split per language, kept separate so eval_sft.py
               can report per-language metrics

Each row exposes:
    input:    str, sentence containing "[MASK]"
    label:    str, the target word (first element of the `labels` list in CSV)
    language: str, one of ro|en|es|fa|hi
    id:       str, stable sample id

Usage:
    python -m pipeline.train.prepare_data --output /tmp/asi_multilingual
"""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = ROOT / "presentation_data"
LANGUAGES = ("ro", "en", "es", "fa", "hi")
SEED = 42


def _parse_label(raw: str) -> str:
    """CSV `labels` column holds a stringified Python list, e.g. "['uluita']"."""
    parsed = ast.literal_eval(raw)
    if not parsed:
        raise ValueError(f"empty labels list: {raw!r}")
    return parsed[0]


def _load_csv(path: Path, language: str) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                "id": row["id"],
                "input": row["input"],
                "label": _parse_label(row["labels"]),
                "language": language,
            })
    return rows


def build(input_dir: Path = DEFAULT_INPUT) -> DatasetDict:
    per_lang_train: list[dict] = []
    per_lang_val: list[dict] = []
    test_splits: dict[str, Dataset] = {}

    for lang in LANGUAGES:
        lang_dir = input_dir / f"presentation_{lang}"
        per_lang_train.extend(_load_csv(lang_dir / "train.csv", lang))
        per_lang_val.extend(_load_csv(lang_dir / "val.csv", lang))
        test_splits[f"test_{lang}"] = Dataset.from_list(_load_csv(lang_dir / "test.csv", lang))

    train = Dataset.from_list(per_lang_train).shuffle(seed=SEED)
    val = Dataset.from_list(per_lang_val)

    return DatasetDict({"train": train, "val": val, **test_splits})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dsd = build(args.input)
    dsd.save_to_disk(str(args.output))

    print(f"Saved DatasetDict → {args.output}")
    for name, ds in dsd.items():
        lang_counts: dict[str, int] = {}
        for lang in ds["language"]:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        print(f"  {name}: {len(ds)} rows  {lang_counts}")


if __name__ == "__main__":
    main()
