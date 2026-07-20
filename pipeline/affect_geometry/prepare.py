"""Build nested, lemma-balanced manifests from the collected datasets."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from pipeline.affect_geometry.common import (
    anchor_maps,
    context_window,
    is_single_word,
    load_json,
    morphology_map,
    normalize,
    reconstruct,
    stable_int,
    target_map,
)


def collect_romanian(path: Path, adjective_frames: set[str]):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("pattern_used") not in adjective_frames:
                continue
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
            })
    return rows


def collect_masive(root: Path, language: str):
    rows = []
    for split in ("train", "val", "test"):
        path = root / language / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for source_row in csv.DictReader(handle):
                text, slots = reconstruct(source_row["clip_input"], target_map(source_row["t5_trg"]))
                if text is None:
                    continue
                for slot, surface, start, end in slots:
                    state = normalize(surface)
                    if not is_single_word(state):
                        continue
                    rows.append({
                        "occurrence_id": f"{language}:{source_row['id']}:{slot}",
                        "source_id": str(source_row["id"]),
                        "split": split,
                        "text": text,
                        "start": start,
                        "end": end,
                        "surface": surface,
                        "form": state,
                        "source": f"masive_{language}",
                    })
    return rows


def build_manifest(rows: list[dict], language: str, config: dict, anchors: dict):
    form_to_anchor, lemma_to_emotion = anchor_maps(anchors, language)
    forms = {row["form"] for row in rows}
    lemma_map = morphology_map(forms, language, form_to_anchor)
    by_lemma = collections.defaultdict(list)
    seen = set()
    for row in rows:
        if row["occurrence_id"] in seen:
            continue
        seen.add(row["occurrence_id"])
        row["lemma"] = lemma_map[row["form"]]
        by_lemma[row["lemma"]].append(row)

    sampling = config["sampling"]
    minimum = int(sampling["minimum_occurrences_per_lemma"])
    maximum = int(sampling["maximum_contexts_per_lemma"])
    half_width = int(sampling["context_characters_each_side"])
    selected = []
    lemma_counts = {}
    for lemma, candidates in sorted(by_lemma.items()):
        if len(candidates) < minimum:
            continue
        candidates.sort(key=lambda row: stable_int(config["random_seed"], row["occurrence_id"]))
        lemma_counts[lemma] = len(candidates)
        for row in candidates[:maximum]:
            text, start, end = context_window(row["text"], row["start"], row["end"], half_width)
            if text[start:end] != row["surface"]:
                raise ValueError(f"Broken target offsets for {row['occurrence_id']}")
            selected.append({
                "occurrence_id": row["occurrence_id"],
                "language": language,
                "lemma": lemma,
                "form": row["form"],
                "surface": row["surface"],
                "text": text,
                "target_start": start,
                "target_end": end,
                "basic_emotion": lemma_to_emotion.get(lemma),
                "is_basic": lemma in lemma_to_emotion,
                "source": row["source"],
                "split": row["split"],
            })
    selected.sort(key=lambda row: stable_int(config["random_seed"], row["occurrence_id"]))
    return selected, lemma_counts


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["ro", "en", "es"], required=True)
    parser.add_argument("--source", required=True, help="RO JSONL, or MASIVE directory containing en/ and es/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--config", default=str(package / "config.json"))
    parser.add_argument("--anchors", default=str(package / "anchors.json"))
    args = parser.parse_args()

    config, anchors = load_json(args.config), load_json(args.anchors)
    if args.language == "ro":
        rows = collect_romanian(Path(args.source), set(config["romanian_adjective_frames"]))
    else:
        rows = collect_masive(Path(args.source), args.language)
    manifest, raw_counts = build_manifest(rows, args.language, config, anchors)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lemma_rows = collections.Counter(row["lemma"] for row in manifest)
    emotion_lemmas = collections.defaultdict(set)
    emotion_rows = collections.Counter()
    for row in manifest:
        if row["basic_emotion"]:
            emotion_lemmas[row["basic_emotion"]].add(row["lemma"])
            emotion_rows[row["basic_emotion"]] += 1
    summary = {
        "language": args.language,
        "source": str(args.source),
        "raw_candidate_occurrences": len(rows),
        "eligible_lemmas": len(lemma_rows),
        "manifest_occurrences": len(manifest),
        "basic_lemmas": {key: sorted(value) for key, value in sorted(emotion_lemmas.items())},
        "basic_occurrences": dict(sorted(emotion_rows.items())),
        "raw_occurrences_by_eligible_lemma": {key: raw_counts[key] for key in sorted(raw_counts)},
        "sampled_occurrences_by_lemma": dict(sorted(lemma_rows.items())),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "language", "raw_candidate_occurrences", "eligible_lemmas", "manifest_occurrences",
        "basic_lemmas", "basic_occurrences")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

