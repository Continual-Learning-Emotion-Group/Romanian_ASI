"""Convert sampled JSONL to MTurk-compatible CSV for human evaluation."""

import argparse
import csv
import html
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "human_eval_sample.jsonl"
OUTPUT_FILE = DATA_DIR / "human_eval_mturk.csv"

MAX_CONTEXT_CHARS = 5000
HIGHLIGHT_STYLE = 'background-color:#2dfe5f'


def escape_for_mturk(text: str) -> str:
    """HTML-encode non-ASCII characters so MTurk accepts the CSV.

    MTurk CSV upload rejects raw UTF-8 diacritics (ă, ș, ț, î, â, etc.).
    Convert them to HTML numeric character references (e.g. ă → &#259;)
    while preserving existing HTML tags.
    """
    out = []
    for ch in text:
        if ord(ch) > 127:
            out.append(f"&#{ord(ch)};")
        else:
            out.append(ch)
    return "".join(out)


def highlight_seed(text: str, seed_word: str) -> str:
    """Wrap first occurrence of seed_word in a green highlight span."""
    pattern = re.compile(re.escape(seed_word), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        s, e = match.start(), match.end()
        return (text[:s]
                + f'<span style="{HIGHLIGHT_STYLE}">'
                + text[s:e]
                + '</span>'
                + text[e:])
    return text


def make_short_context(candidate: dict) -> str:
    """Build short context: matched sentence with highlighted seed word."""
    sentence = candidate.get("matched_sentence", "")
    seed = candidate.get("seed_word", "")
    highlighted = highlight_seed(sentence, seed)
    return f"<strong>{highlighted}</strong><br>"


def make_full_context(candidate: dict) -> str:
    """Build full context: title (if available) + text with highlighted match."""
    text = candidate.get("text", "")
    seed = candidate.get("seed_word", "")
    matched = candidate.get("matched_sentence", "")
    source = candidate.get("source", "")

    # Truncate if too long
    if len(text) > MAX_CONTEXT_CHARS:
        idx = text.find(matched)
        if idx >= 0:
            window = MAX_CONTEXT_CHARS // 2
            start = max(0, idx - window)
            end = min(len(text), idx + len(matched) + window)
            text = text[start:end]
        else:
            text = text[:MAX_CONTEXT_CHARS]

    # Highlight seed word in text
    text_highlighted = highlight_seed(text, seed)

    # Prepend title for filmot/other sources
    parts = []
    title = candidate.get("title") or candidate.get("video_title")
    if title:
        parts.append(f"<em>{title}</em><br><br>")
    parts.append(text_highlighted + "<br>")

    return "".join(parts)


def convert_to_csv(input_path: Path, output_path: Path):
    rows = []
    with open(input_path) as f:
        for line in f:
            c = json.loads(line)
            rows.append({
                "id": c["id"],
                "short_context": escape_for_mturk(make_short_context(c)),
                "full_context": escape_for_mturk(make_full_context(c)),
                "emo_term": escape_for_mturk(c.get("seed_word", "")),
                "show_inst": "show",
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "short_context", "full_context", "emo_term", "show_inst"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert sample JSONL to MTurk CSV")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    convert_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
