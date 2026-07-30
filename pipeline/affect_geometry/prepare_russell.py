"""Manifest builder for the Russell-anchor variant.

Identical to prepare.py except:
- Romanian keeps ALL benchmark patterns (no adjective-frame filter), so noun
  states such as frica/teama/dor enter the pool.
- No anchor forcing at manifest time: lemma merging is purely morphological,
  and Russell labels are assigned later, at analysis time, from
  anchors_russell.json. The manifest's basic_emotion field is left null.

English/Spanish pools are unchanged from v1 (MASIVE collection never had a POS
filter), so their v1 manifests and centroid archives remain valid; this script
only needs to run for Romanian.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from pipeline.affect_geometry.common import is_single_word, load_json, normalize
from pipeline.affect_geometry.prepare import build_manifest, collect_masive


def collect_romanian_all_patterns(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            start, end = row.get("seed_word_start"), row.get("seed_word_end")
            if start is None or end is None or int(end) <= int(start):
                continue
            state = normalize(row.get("seed_word_normalized", row.get("seed_word", "")))
            if not is_single_word(state):
                continue
            rows.append({
                "occurrence_id": f"ro:{row['id']}:{int(start)}:{int(end)}",
                "source_id": str(row["id"]),
                "split": "benchmark",
                "text": row["text"],
                "start": int(start),
                "end": int(end),
                "surface": row["text"][int(start):int(end)],
                "form": state,
                "source": row.get("source", "romanian_asi"),
                "pattern_used": row.get("pattern_used"),
            })
    return rows


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["ro", "en", "es"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--config", default=str(package / "config.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    if args.language == "ro":
        rows = collect_romanian_all_patterns(Path(args.source))
    else:
        rows = collect_masive(Path(args.source), args.language)
    neutral_anchors = {"languages": {args.language: {}}}
    manifest, raw_counts = build_manifest(rows, args.language, config, neutral_anchors)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lemma_rows = collections.Counter(row["lemma"] for row in manifest)
    summary = {
        "language": args.language,
        "experiment": "affect_geometry_russell_v2",
        "source": str(args.source),
        "raw_candidate_occurrences": len(rows),
        "eligible_lemmas": len(lemma_rows),
        "manifest_occurrences": len(manifest),
        "raw_occurrences_by_eligible_lemma": {key: raw_counts[key] for key in sorted(raw_counts)},
        "sampled_occurrences_by_lemma": dict(sorted(lemma_rows.items())),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "language", "raw_candidate_occurrences", "eligible_lemmas",
        "manifest_occurrences")}, indent=2))


if __name__ == "__main__":
    main()
