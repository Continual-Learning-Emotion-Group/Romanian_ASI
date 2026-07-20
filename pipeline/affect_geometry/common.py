from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

EXTRA_ID = re.compile(r"<extra_id_(\d+)>")
WORD = re.compile(r"^[a-z]+(?:-[a-z]+)?$")
WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.strip(".,;:!?\"'()[]{}<>«»¿¡…")


def is_single_word(text: str) -> bool:
    return bool(WORD.fullmatch(text))


def target_map(target: str) -> dict[int, str]:
    parts = EXTRA_ID.split(str(target))
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def reconstruct(masked: str, targets: dict[int, str]):
    text, slots, last = "", [], 0
    for match in EXTRA_ID.finditer(str(masked)):
        slot = int(match.group(1))
        if slot not in targets:
            return None, None
        text += masked[last:match.start()]
        start = len(text)
        text += targets[slot]
        slots.append((slot, targets[slot], start, len(text)))
        last = match.end()
    return text + masked[last:], slots


def context_window(text: str, start: int, end: int, half_width: int):
    low, high = max(0, start - half_width), min(len(text), end + half_width)
    while low > 0 and not text[low - 1].isspace():
        low -= 1
    while high < len(text) and not text[high].isspace():
        high += 1
    return text[low:high], start - low, end - low


def stable_int(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def anchor_maps(anchor_config: dict, language: str):
    form_to_lemma, lemma_to_emotion = {}, {}
    for emotion, lemmas in anchor_config["languages"][language].items():
        for lemma, forms in lemmas.items():
            lemma = normalize(lemma)
            lemma_to_emotion[lemma] = emotion
            for form in forms:
                key = normalize(form)
                if key in form_to_lemma and form_to_lemma[key] != lemma:
                    raise ValueError(f"Ambiguous anchor form: {key}")
                form_to_lemma[key] = lemma
    return form_to_lemma, lemma_to_emotion


def morphology_map(forms: set[str], language: str, forced: dict[str, str]) -> dict[str, str]:
    """Merge only productive gender pairs that both occur in the dataset."""
    mapping = {form: forced.get(form, form) for form in forms}
    transforms = []
    if language in {"ro", "es"}:
        transforms.extend([
            ("oasa", "os"),
            ("oare", "or"),
            ("ora", "or"),
            ("ona", "on"),
            ("ina", "in"),
            ("esa", "es"),
            ("a", "o"),
            ("a", ""),
        ])
    for form in sorted(forms):
        if form in forced:
            continue
        for feminine, masculine in transforms:
            if not form.endswith(feminine) or len(form) <= len(feminine) + 1:
                continue
            candidate = form[:-len(feminine)] + masculine
            if candidate in forms:
                mapping[form] = forced.get(candidate, candidate)
                break
    return mapping

