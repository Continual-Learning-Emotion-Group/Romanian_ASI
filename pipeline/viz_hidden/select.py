"""Select + window the Tier-1 (Plutchik-8) subset for hidden-state extraction.

Reads the clean benchmark, assigns each row a Plutchik-8 emotion (unambiguous
rollup only), balances classes with a per-class cap, and windows `text` around
the seed word so sequences are short and the emotion-word char offsets are known
relative to the window. Output is a compact JSONL the remote extractor consumes.

Usage:
    python -m pipeline.viz_hidden.select \
        --benchmark pipeline/data/benchmark_ro_asi_clean.jsonl \
        --out pipeline/viz_hidden/out/tier1_ro_input.jsonl \
        --cap 1200 --window 350
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from pipeline.viz_hidden.tiers import PlutchikMapper, PLUTCHIK8


def window_text(text: str, start: int, end: int, half: int) -> tuple[str, int, int]:
    """Return (windowed_text, new_start, new_end) keeping `half` chars either side
    of [start,end). Snaps to word boundaries so we don't cut mid-token."""
    lo = max(0, start - half)
    hi = min(len(text), end + half)
    # snap outward to whitespace to avoid clipping a subword
    while lo > 0 and not text[lo - 1].isspace():
        lo -= 1
    while hi < len(text) and not text[hi].isspace():
        hi += 1
    return text[lo:hi], start - lo, end - lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="pipeline/data/benchmark_ro_asi_clean.jsonl")
    ap.add_argument("--out", default="pipeline/viz_hidden/out/tier1_ro_input.jsonl")
    ap.add_argument("--cap", type=int, default=1200, help="max rows per Plutchik class")
    ap.add_argument("--window", type=int, default=350, help="chars kept either side of the seed word")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    mapper = PlutchikMapper()
    rng = random.Random(args.seed)

    # 1. bucket rows by Plutchik emotion
    buckets: dict[str, list[dict]] = defaultdict(list)
    n_total = n_kept = 0
    with open(args.benchmark) as f:
        for line in f:
            d = json.loads(line)
            n_total += 1
            p = mapper.row_plutchik(d.get("emotion_category", []))
            if p is None:
                continue
            s, e = d.get("seed_word_start"), d.get("seed_word_end")
            if s is None or e is None or e <= s:
                continue
            wtext, ws, we = window_text(d["text"], s, e, args.window)
            # sanity: the windowed span must still be the seed word
            if wtext[ws:we] != d["text"][s:e]:
                continue
            buckets[p].append({
                "id": d["id"],
                "text": wtext,
                "word_start": ws,
                "word_end": we,
                "surface": wtext[ws:we],
                "plutchik": p,
                "valence": mapper.row_valence(d["emotion_category"]),
                "seed_word_normalized": d.get("seed_word_normalized"),
                "emotion_category": d.get("emotion_category"),
                "source": d.get("source"),
            })
            n_kept += 1

    # 2. balance with per-class cap
    out_rows: list[dict] = []
    summary = {}
    for p in PLUTCHIK8:
        rows = buckets.get(p, [])
        rng.shuffle(rows)
        take = rows[: args.cap]
        out_rows.extend(take)
        summary[p] = {"available": len(rows), "taken": len(take)}
    rng.shuffle(out_rows)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"benchmark rows: {n_total} | unambiguous Plutchik rows: {n_kept}")
    print(f"selected: {len(out_rows)} -> {outp}")
    for p in PLUTCHIK8:
        s = summary[p]
        print(f"  {p:12s} available={s['available']:6d}  taken={s['taken']:5d}")


if __name__ == "__main__":
    main()
