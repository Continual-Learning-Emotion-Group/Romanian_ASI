"""Build the multilingual SFT DatasetDict from per-language CSVs.

Reads `{input_dir}/{ro,en,es,fa,hi}/{train,val,test}.csv`, tags each row with
its `language`, then:
- train: concatenate all 5 languages (~5000 rows) and shuffle with seed=42
- val:   concatenate all 5 languages (~1200 rows) — used during training for
         best-checkpoint selection
- test_<lang>: one 1000-row split per language, kept separate so eval_sft.py
               can report per-language metrics

Each row exposes:
    input:    str, sentence containing one or more "[MASK]" tokens
    label:    str, supervision target — the `labels` list space-joined verbatim
    labels:   list[str], the full label list (set of distinct affective
              expressions that fill the mask positions)
    n_masks:  int, number of [MASK] tokens in the input (useful for analysis)
    language: str, one of ro|en|es|fa|hi
    id:       str, stable sample id

Supervision semantics — why target = " ".join(labels):
- The `labels` column is the *set of distinct affective expressions* that fill
  the [MASK] positions, in order of first appearance. When len(labels) ==
  n_masks each mask is a distinct expression (EN/ES matched rows). When
  len(labels) < n_masks the same expression covers multiple masks
  (e.g. EN "feel [MASK]. Or rather I am [MASK]" with labels=['unfit']).
  FA multi-word labels like "دلم تنگ شده" are single idiomatic expressions
  that occupy one [MASK] span.
- Joining all labels with a space gives a uniform contract the prompt can
  describe: "output the distinct expressions in order of first appearance,
  space-separated." No row is dropped; no row carries contradictory signal.

Usage:
    python -m pipeline.ft_qwen_mixed.prepare_data --output /tmp/asi_multilingual
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


def _parse_labels(raw: str) -> list[str]:
    """CSV `labels` column holds a stringified Python list, e.g. "['uluita']"."""
    parsed = ast.literal_eval(raw)
    if not parsed:
        raise ValueError(f"empty labels list: {raw!r}")
    return [str(x) for x in parsed]


def _supervision_target(labels: list[str], n_masks: int) -> str:
    """Build the assistant turn text.

    `labels` is the set of distinct affective expressions that fill the mask
    positions, in order of first appearance. We always space-join the full
    list, whether or not len(labels) matches n_masks — the model learns to
    emit the set, not enumerate per-position.
    """
    del n_masks  # kept in dataset rows for analysis, not used for supervision
    return " ".join(labels)


def _load_csv(path: Path, language: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            input_text = row["input"]
            labels = _parse_labels(row["labels"])
            n_masks = input_text.count("[MASK]")
            rows.append({
                "id": row["id"],
                "input": input_text,
                "label": _supervision_target(labels, n_masks),
                "labels": labels,
                "n_masks": n_masks,
                "language": language,
            })
    return rows


def build(input_dir: Path = DEFAULT_INPUT) -> DatasetDict:
    per_lang_train: list[dict] = []
    per_lang_val: list[dict] = []
    test_splits: dict[str, Dataset] = {}

    for lang in LANGUAGES:
        # Accept both layouts: `{input_dir}/{lang}/*.csv` (sequential_data-style)
        # and `{input_dir}/presentation_{lang}/*.csv` (original presentation_data).
        lang_dir = input_dir / lang
        if not lang_dir.is_dir():
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
        multi_mask_counts: dict[str, int] = {}
        for row in ds:
            lang = row["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if row["n_masks"] > 1:
                multi_mask_counts[lang] = multi_mask_counts.get(lang, 0) + 1
        print(f"  {name}: {len(ds)} rows  {lang_counts}")
        if multi_mask_counts:
            print(f"    multi-mask rows: {multi_mask_counts}")


if __name__ == "__main__":
    main()
