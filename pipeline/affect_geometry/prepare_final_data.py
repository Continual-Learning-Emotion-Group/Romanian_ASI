"""Manifest builder for the Final Data refresh (all 8 languages).

Sources (~/Downloads/Final Data):
- en/es/fa/fr/hi/ro/zh: {lang}_judged.csv.gz — unified judged schema
  (judge_term, original_context, llm_score, match_ok; fa additionally ships
  masked_context/[MASK] + mask_labels so spans are exact).
- id: ind/id_judge_results_v2/scored_csvs/id_judge_10k_v2.csv — MASIVE-style
  masked schema (clip_input + t5_trg) with judge columns appended; spans are
  reconstructed exactly from the masks.

Filters: llm_score == 3 AND match_ok == True everywhere (matches the ro
benchmark precedent and the zh/fa v2 sources; fixes the old hi asymmetry).

Target spans: fa and id are exact (mask reconstruction). For the other
languages the judged term is located in original_context with word-boundary
checks; rows where the term is absent or occurs more than once (ambiguous)
are DROPPED, mirroring the conservative fa rule from prepare_new_languages.

Lemma merging:
- ro/es: Latin normalize() + the frozen ro/es gender transforms (common.py).
- fr: Latin normalize() + French gender transforms (euse->eux, ere->er,
  ve->f, sse->'', ette->et, e->'') applied only when both forms occur.
  The multiword phrase "en colere" is allowed as a lemma (fa template
  precedent).
- en: Latin normalize() only.
- id: lowercase + orthographic variant merges frustrasi->frustasi,
  stress->stres.
- zh: NFKC, CJK-only 1-6 chars, as-is.
- fa: lemma = target_affective_state template (same as the old fa manifest,
  so the frozen fa anchors keep matching).
- hi: NFC + adjective gender pairs ...i -> ...aa when both occur.

Sampling: frozen config.json (>=30 occurrences per lemma, <=80 contexts,
350-character windows, seed 20260720, stable_int ordering).

Run:
  python -m pipeline.affect_geometry.prepare_final_data \
      --data-root "/Users/alexjerpelea/Downloads/Final Data" \
      --id-csv ".../id_judge_10k_v2.csv" \
      --output-dir pipeline/affect_geometry/artifacts/manifests_final
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import io
import json
import re
import unicodedata
from pathlib import Path

from pipeline.affect_geometry.common import (
    context_window, is_single_word, load_json, normalize, reconstruct,
    stable_int, target_map,
)
from pipeline.affect_geometry.prepare_new_languages import (
    normalize_fa, normalize_hi, normalize_zh, template_matches,
)

csv.field_size_limit(10**8)

CJK = re.compile(r"^[一-鿿]{1,6}$")

FR_TRANSFORMS = [("euse", "eux"), ("ere", "er"), ("ve", "f"),
                 ("sse", ""), ("ette", "et"), ("e", "")]
FR_MULTIWORD = {"en colere"}
ID_VARIANTS = {"frustrasi": "frustasi", "stress": "stres"}

JUDGED_LANGS = ("en", "es", "fa", "fr", "hi", "ro", "zh")

# Russell-anchor lemmas admitted below the frozen 30-occurrence floor
# (>=25 judged-clean occurrences) so the deactivation quadrant ("tired",
# 268 deg) is present in the global 8-language anchor-label intersection:
# fr fatigue has 25 clean manifest candidates, id lelah 24 — both
# near-misses of the floor (2 judged rows each drop at mask/dedup).
ANCHOR_FLOOR_EXCEPTIONS = {"fr": {"fatigue"}, "id": {"lelah"}}
ANCHOR_FLOOR_MINIMUM = 24


def read_judged(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def score_ok(row) -> bool:
    return row.get("llm_score", "").split(".")[0] == "3" and row.get("match_ok") == "True"


def locate(term: str, text: str, boundaries: bool):
    """All boundary-valid occurrences of term in text (case-sensitive, then
    case-insensitive fallback when none found)."""
    for haystack, needle in ((text, term), (text.lower(), term.lower())):
        found = []
        position = 0
        while True:
            index = haystack.find(needle, position)
            if index < 0:
                break
            if boundaries:
                before_ok = index == 0 or not haystack[index - 1].isalpha()
                after = index + len(needle)
                after_ok = after >= len(haystack) or not haystack[after].isalpha()
            else:
                before_ok = after_ok = True
            if before_ok and after_ok:
                found.append(index)
            position = index + 1
        if found:
            return found
    return []


def collect_located(path: Path, language: str, form_of, form_ok, boundaries=True):
    """Shared collector for languages located via judge_term in original_context."""
    rows = []
    skips = collections.Counter()
    for source_row in read_judged(path):
        if not score_ok(source_row):
            skips["score"] += 1
            continue
        term = str(source_row.get("judge_term", "")).strip()
        text = str(source_row.get("original_context", ""))
        if not term or not text:
            skips["empty"] += 1
            continue
        form = form_of(term)
        if not form_ok(form):
            skips["form"] += 1
            continue
        found = locate(term, text, boundaries)
        if not found:
            skips["unlocatable"] += 1
            continue
        if len(found) > 1:
            skips["ambiguous"] += 1
            continue
        start = found[0]
        rows.append({
            "occurrence_id": f"{language}:{source_row['uid']}",
            "split": "all",
            "text": text,
            "start": start,
            "end": start + len(term),
            "surface": text[start:start + len(term)],
            "form": form,
            "source": str(source_row.get("source", f"{language}_asi")),
        })
    return rows, skips


def latin_form_ok(form: str) -> bool:
    return is_single_word(form)


def collect_latin(path: Path, language: str):
    def form_ok(form):
        return is_single_word(form) or (language == "fr" and form in FR_MULTIWORD)
    return collect_located(path, language, normalize, form_ok)


def collect_zh(path: Path):
    return collect_located(path, "zh", normalize_zh,
                           lambda f: bool(CJK.fullmatch(f)), boundaries=False)


def collect_hi(path: Path):
    return collect_located(path, "hi", normalize_hi,
                           lambda f: bool(f) and " " not in f)


def collect_fa(path: Path):
    rows = []
    skips = collections.Counter()
    for source_row in read_judged(path):
        if not score_ok(source_row):
            skips["score"] += 1
            continue
        state = normalize_fa(source_row.get("target_affective_state", ""))
        if not state or "\n" in state:
            skips["form"] += 1
            continue
        masked = str(source_row.get("masked_context", ""))
        raw_labels = str(source_row.get("mask_labels", "") or "")
        if "[MASK]" not in masked or not raw_labels.strip():
            skips["empty"] += 1
            continue
        labels = [normalize_fa(part) for part in raw_labels.split("|||")]
        if masked.count("[MASK]") != len(labels):
            skips["mask_mismatch"] += 1
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
            skips["unlocatable"] += 1
            continue
        if len(matches) > 1:
            skips["ambiguous"] += 1
            continue
        start, end, surface = matches[0]
        rows.append({
            "occurrence_id": f"fa:{source_row['uid']}",
            "split": "all",
            "text": text,
            "start": start,
            "end": end,
            "surface": surface,
            "form": state,
            "source": str(source_row.get("source", "fa_asi")),
        })
    return rows, skips


def collect_id(path: Path):
    rows = []
    skips = collections.Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for source_row in csv.DictReader(handle):
            if not score_ok(source_row):
                skips["score"] += 1
                continue
            term = str(source_row.get("judge_term", "")).strip().lower()
            if not term or not is_single_word(term):
                skips["form"] += 1
                continue
            text = str(source_row.get("selftext") or source_row.get("input") or "")
            if not text:
                skips["empty"] += 1
                continue
            found = locate(term, text, boundaries=True)
            if not found:
                skips["unlocatable"] += 1
                continue
            if len(found) > 1:
                skips["ambiguous"] += 1
                continue
            start = found[0]
            rows.append({
                "occurrence_id": f"id:{source_row['id']}",
                "split": "all",
                "text": text,
                "start": start,
                "end": start + len(term),
                "surface": text[start:start + len(term)],
                "form": term,
                "source": str(source_row.get("source", "") or "id_asi"),
            })
    return rows, skips


def id_safe_terms(judged_csv: Path, min_rate: float = 0.7, min_count: int = 5):
    """Per-term score-3 precision from the judged 10k sample. Terms above the
    (rate, count) bar are safe to mine from UNJUDGED corpora: the judged
    sample acts as a calibration set in place of per-row LLM judging."""
    stats = collections.defaultdict(lambda: [0, 0])
    judged_ids = set()
    with judged_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            judged_ids.add(str(row.get("id", "")))
            term = str(row.get("judge_term", "")).strip().lower()
            if not term:
                continue
            stats[term][1] += 1
            if score_ok(row):
                stats[term][0] += 1
    safe = {term for term, (hits, total) in stats.items()
            if total >= min_count and hits / total >= min_rate}
    rates = {term: round(stats[term][0] / stats[term][1], 3) for term in sorted(safe)}
    return safe, rates, judged_ids


def collect_id_extra(extra_csvs, judged_csv: Path):
    """Mine the unjudged id corpora (threads/x/yt/kaggle_reddit) for terms
    whose judged precision clears the calibration bar. Negated/challenge rows
    and rows already present in the judged sample are skipped."""
    safe, rates, judged_ids = id_safe_terms(judged_csv)
    rows = []
    skips = collections.Counter()
    for path in extra_csvs:
        corpus = path.stem
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for source_row in csv.DictReader(handle):
                row_id = str(source_row.get("id", ""))
                if row_id in judged_ids:
                    skips["already_judged"] += 1
                    continue
                if source_row.get("is_negated") == "True":
                    skips["negated"] += 1
                    continue
                if source_row.get("challenge") == "True":
                    skips["challenge"] += 1
                    continue
                raw_labels = str(source_row.get("labels", ""))
                terms = [t.strip().strip("'\"").lower()
                         for t in raw_labels.strip("[]").split(",") if t.strip()]
                text = str(source_row.get("selftext") or source_row.get("input") or "")
                if not text:
                    skips["empty"] += 1
                    continue
                for term in terms:
                    if not term or not is_single_word(term):
                        skips["form"] += 1
                        continue
                    if term not in safe:
                        skips["below_calibration_bar"] += 1
                        continue
                    found = locate(term, text, boundaries=True)
                    if not found:
                        skips["unlocatable"] += 1
                        continue
                    if len(found) > 1:
                        skips["ambiguous"] += 1
                        continue
                    start = found[0]
                    rows.append({
                        "occurrence_id": f"id:{row_id}:{term}",
                        "split": "all",
                        "text": text,
                        "start": start,
                        "end": start + len(term),
                        "surface": text[start:start + len(term)],
                        "form": term,
                        "source": f"{corpus}_unjudged_calibrated",
                    })
    return rows, skips, rates


def gender_merge(forms: set[str], transforms) -> dict[str, str]:
    mapping = {form: form for form in forms}
    for form in sorted(forms):
        for feminine, masculine in transforms:
            if not form.endswith(feminine) or len(form) <= len(feminine) + 1:
                continue
            candidate = form[:-len(feminine)] + masculine
            if candidate in forms:
                mapping[form] = candidate
                break
    return mapping


def hindi_gender_merge(forms: set[str]) -> dict[str, str]:
    mapping = {form: form for form in forms}
    for form in sorted(forms):
        if form.endswith("ी") and len(form) > 2:
            candidate = form[:-1] + "ा"
            if candidate in forms:
                mapping[form] = candidate
    return mapping


RO_ES_TRANSFORMS = [("oasa", "os"), ("oare", "or"), ("ora", "or"), ("ona", "on"),
                    ("ina", "in"), ("esa", "es"), ("a", "o"), ("a", "")]


def lemma_mapping(forms: set[str], language: str) -> dict[str, str]:
    if language in {"ro", "es"}:
        return gender_merge(forms, RO_ES_TRANSFORMS)
    if language == "fr":
        return gender_merge(forms, FR_TRANSFORMS)
    if language == "hi":
        return hindi_gender_merge(forms)
    if language == "id":
        return {form: ID_VARIANTS.get(form, form) for form in forms}
    return {form: form for form in forms}


def build_manifest(rows, language, config):
    lemma_map = lemma_mapping({row["form"] for row in rows}, language)
    by_lemma = collections.defaultdict(list)
    seen_ids, seen_content = set(), set()
    duplicates = 0
    for row in rows:
        content_key = (row["text"], row["start"], row["end"])
        if row["occurrence_id"] in seen_ids or content_key in seen_content:
            duplicates += 1
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
        floor = (ANCHOR_FLOOR_MINIMUM
                 if lemma in ANCHOR_FLOOR_EXCEPTIONS.get(language, ())
                 else minimum)
        if len(candidates) < floor:
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
    return selected, lemma_counts, duplicates


def collect(language: str, data_root: Path, id_csv: Path, id_extra_csvs=()):
    if language == "id":
        rows, skips = collect_id(id_csv)
        if id_extra_csvs:
            extra_rows, extra_skips, rates = collect_id_extra(
                [Path(p) for p in id_extra_csvs], id_csv)
            rows += extra_rows
            skips.update({f"extra_{k}": v for k, v in extra_skips.items()})
            skips["extra_rows_added"] = len(extra_rows)
        return rows, skips
    path = data_root / f"{language}_judged.csv.gz"
    if language == "fa":
        return collect_fa(path)
    if language == "zh":
        return collect_zh(path)
    if language == "hi":
        return collect_hi(path)
    return collect_latin(path, language)


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--id-csv", required=True)
    parser.add_argument("--id-extra-csvs", nargs="*", default=[],
                        help="Unjudged id corpora mined via the calibration bar")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--languages", nargs="*",
                        default=list(JUDGED_LANGS) + ["id"])
    parser.add_argument("--config", default=str(package / "config.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for language in args.languages:
        rows, skips = collect(language, Path(args.data_root), Path(args.id_csv),
                              args.id_extra_csvs)
        manifest, raw_counts, duplicates = build_manifest(rows, language, config)
        with (output_dir / f"{language}.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        lemma_rows = collections.Counter(row["lemma"] for row in manifest)
        summary = {
            "language": language,
            "experiment": "affect_geometry_final_data_8lang",
            "source": str(Path(args.data_root) / f"{language}_judged.csv.gz")
                      if language != "id" else str(args.id_csv),
            "filter": "llm_score == 3 and match_ok",
            "skipped": dict(skips),
            "duplicates_dropped": duplicates,
            "raw_candidate_occurrences": len(rows),
            "eligible_lemmas": len(lemma_rows),
            "manifest_occurrences": len(manifest),
            "raw_occurrences_by_eligible_lemma": {k: raw_counts[k] for k in sorted(raw_counts)},
            "sampled_occurrences_by_lemma": dict(sorted(lemma_rows.items())),
        }
        (output_dir / f"{language}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: summary[key] for key in (
            "language", "raw_candidate_occurrences", "eligible_lemmas",
            "manifest_occurrences", "skipped")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
