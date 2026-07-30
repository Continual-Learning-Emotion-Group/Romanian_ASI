"""Manifest builder for the Mandarin / Persian / Hindi extension.

Sources (dump_new_data/):
- zh: mandarin_llm_kept_ge3_cleaned.csv  (MASIVE masked format, LLM score >= 3)
- fa: persian_llm_kept_ge3.csv           (masked_context/[MASK], LLM score >= 3;
      states are single words or somatic idiom templates like "دلم ... گرفته")
- hi: Hindi_all.csv                      (sentence marked [[...]] inside a wider
      context; never LLM-filtered, so the full dump is used)

Same frozen sampling as v1/v2 (config.json): >= 30 occurrences per lemma,
<= 80 contexts, 350-character windows, seed-stable ordering. Russell labels
are assigned at analysis time (basic_emotion stays null), as in
prepare_russell.py.

Normalization is language-specific: the Latin normalize() in common.py strips
combining marks, which would destroy Devanagari vowel signs and Persian
letters, so it is NOT used here.

Lemma merging: Mandarin and Persian states are used as-is (no inflection /
already-templated idioms). Hindi merges only productive adjective gender
pairs (…ा / …ी) when both forms occur, mirroring the conservative ro/es rule.

Run:
  python -m pipeline.affect_geometry.prepare_new_languages --language zh \
      --source dump_new_data/mandarin_llm_kept_ge3_cleaned.csv \
      --output pipeline/affect_geometry/artifacts/manifests/zh.jsonl \
      --summary pipeline/affect_geometry/artifacts/manifests/zh_summary.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path

from pipeline.affect_geometry.common import context_window, load_json, stable_int

csv.field_size_limit(10**8)

CJK = re.compile(r"^[一-鿿]{1,6}$")
DEVANAGARI_LETTER = re.compile(r"[ऀ-ॿ]")
LATIN_LETTER = re.compile(r"[a-zA-Z]")


def normalize_zh(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text).strip())


def normalize_fa(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text).strip())
    # Arabic-presentation -> Persian letters; drop tatweel
    return text.replace("ي", "ی").replace("ك", "ک").replace("ـ", "")


def normalize_hi(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text).strip())
    return text.lower() if LATIN_LETTER.search(text) else text


def is_word_char(char: str) -> bool:
    return bool(DEVANAGARI_LETTER.match(char) or LATIN_LETTER.match(char))


def collect_mandarin(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for source_row in csv.DictReader(handle):
            state = normalize_zh(source_row["target_affective_state"])
            if not CJK.fullmatch(state):
                continue
            masked = source_row["input"]
            marker = masked.find("<extra_id_0>")
            if marker < 0:
                continue
            text = masked[:marker] + state + masked[marker + len("<extra_id_0>"):]
            rows.append({
                "occurrence_id": f"zh:{source_row['id']}:0",
                "source_id": str(source_row["id"]),
                "split": "all",
                "text": text,
                "start": marker,
                "end": marker + len(state),
                "surface": state,
                "form": state,
                "source": source_row.get("source", "mandarin_asi"),
            })
    return rows


def template_matches(filled: str, template: str) -> bool:
    parts = [part for part in template.split("...") if part.strip()]
    parts = [part.strip() for part in parts]
    if len(parts) == 1:
        return filled == parts[0]
    if not (filled.startswith(parts[0]) and filled.endswith(parts[-1])):
        return False
    position = 0
    for part in parts:
        found = filled.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def collect_persian(path: Path):
    rows = []
    skipped_ambiguous = skipped_unmatched = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for source_row in csv.DictReader(handle):
            state = normalize_fa(source_row["target_affective_state"])
            if not state or "\n" in state:
                continue
            masked = source_row["masked_context"]
            labels = [normalize_fa(part) for part in source_row["mask_labels"].split("|||")]
            if masked.count("[MASK]") != len(labels):
                continue
            text, spans, cursor = "", [], 0
            for label in labels:
                marker = masked.find("[MASK]", cursor)
                text += masked[cursor:marker]
                spans.append((len(text), len(text) + len(label), label))
                text += label
                cursor = marker + len("[MASK]")
            text += masked[cursor:]
            matches = [span for span in spans if template_matches(span[2], state)]
            if not matches:
                skipped_unmatched += 1
                continue
            if len(matches) > 1:
                skipped_ambiguous += 1
                continue
            start, end, surface = matches[0]
            rows.append({
                "occurrence_id": f"fa:{source_row['row_id']}",
                "source_id": str(source_row["row_id"]),
                "split": "all",
                "text": text,
                "start": start,
                "end": end,
                "surface": surface,
                "form": state,
                "source": source_row.get("source", "persian_asi"),
            })
    print(f"fa: skipped {skipped_unmatched} unmatched, {skipped_ambiguous} ambiguous-mask rows")
    return rows


def collect_hindi(path: Path):
    rows = []
    skipped = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_index, source_row in enumerate(csv.DictReader(handle)):
            word = normalize_hi(source_row["emotion_word"])
            if not word or " " in word:
                continue
            context = source_row["context"]
            open_marker = context.find("[[")
            close_marker = context.find("]]", open_marker + 2)
            if open_marker >= 0 and close_marker >= 0 and context.count("[[") == 1:
                text = (context[:open_marker] + context[open_marker + 2:close_marker]
                        + context[close_marker + 2:])
                sentence_start = open_marker
                sentence_end = open_marker + (close_marker - open_marker - 2)
            elif not context.strip():
                # v4_legacy rows ship no context window, only the sentence
                text = source_row["sentence"]
                sentence_start, sentence_end = 0, len(text)
            else:
                skipped += 1
                continue
            haystack = normalize_hi(text) if LATIN_LETTER.search(word) else text
            start = None
            position = sentence_start
            while True:
                found = haystack.find(word, position)
                if found < 0 or found >= sentence_end:
                    break
                before_ok = found == 0 or not is_word_char(haystack[found - 1])
                after_position = found + len(word)
                after_ok = after_position >= len(haystack) or not is_word_char(haystack[after_position])
                if before_ok and after_ok:
                    start = found
                    break
                position = found + 1
            if start is None:
                skipped += 1
                continue
            rows.append({
                "occurrence_id": f"hi:{line_index}",
                "source_id": source_row.get("source_id", str(line_index)),
                "split": "all",
                "text": text,
                "start": start,
                "end": start + len(word),
                "surface": text[start:start + len(word)],
                "form": word,
                "source": source_row.get("source", "hindi_asi"),
            })
    print(f"hi: skipped {skipped} rows (bad markers or unlocatable word)")
    return rows


COLLECTORS = {"zh": collect_mandarin, "fa": collect_persian, "hi": collect_hindi}


def hindi_gender_map(forms: set[str]) -> dict[str, str]:
    """Merge adjective gender pairs …ी -> …ा only when both occur."""
    mapping = {form: form for form in forms}
    for form in sorted(forms):
        if form.endswith("ी") and len(form) > 2:
            candidate = form[:-1] + "ा"
            if candidate in forms:
                mapping[form] = candidate
    return mapping


def build_manifest(rows, language, config):
    forms = {row["form"] for row in rows}
    lemma_map = hindi_gender_map(forms) if language == "hi" else {form: form for form in forms}
    by_lemma = collections.defaultdict(list)
    seen_ids, seen_content = set(), set()
    for row in rows:
        content_key = (row["text"], row["start"], row["end"])
        if row["occurrence_id"] in seen_ids or content_key in seen_content:
            continue
        seen_ids.add(row["occurrence_id"])
        seen_content.add(content_key)
        row["lemma"] = lemma_map[row["form"]]
        by_lemma[row["lemma"]].append(row)

    sampling = config["sampling"]
    minimum = int(sampling["minimum_occurrences_per_lemma"])
    maximum = int(sampling["maximum_contexts_per_lemma"])
    half_width = int(sampling["context_characters_each_side"])
    selected, lemma_counts = [], {}
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
                "basic_emotion": None,
                "is_basic": False,
                "source": row["source"],
                "split": row["split"],
            })
    selected.sort(key=lambda row: stable_int(config["random_seed"], row["occurrence_id"]))
    return selected, lemma_counts


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=sorted(COLLECTORS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--config", default=str(package / "config.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    rows = COLLECTORS[args.language](Path(args.source))
    manifest, raw_counts = build_manifest(rows, args.language, config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lemma_rows = collections.Counter(row["lemma"] for row in manifest)
    summary = {
        "language": args.language,
        "experiment": "affect_geometry_russell_v2_multilingual",
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
