"""Build the WHOLE-CORPUS extraction input: the top-N most frequent MASIVE
affective states (open vocabulary), `cap` samples each.

Unlike build_input.py (which keeps only the 8 basic emotions), this keeps every
state and just ranks them by frequency. Each emitted row carries the state word
in `seed_word_normalized` (the aggregation key for analyze_full.py) and, ONLY if
the state happens to be one of the canonical basic-8 forms, its Plutchik rollup
in `plutchik` (else null) so the basic emotions can be overlaid as ring anchors.
Schema is otherwise identical to select.py, so extract.py runs unchanged.

Usage:
    python -m pipeline.viz_hidden.masive.build_input_full --lang en \
        --masive-dir <dir> --top-n 200 --cap 50 --window 350 --write
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from pipeline.viz_hidden.masive.build_input import load_masive, reconstruct, trg_map, window
from pipeline.viz_hidden.masive.lexicons import VALENCE, form2emo, norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masive-dir", required=True)
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-n", type=int, default=200, help="most frequent states to keep")
    ap.add_argument("--cap", type=int, default=50, help="samples per state")
    ap.add_argument("--window", type=int, default=350)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    f2e = form2emo(args.lang)
    df = load_masive(Path(args.masive_dir), args.lang, args.splits.split(","))
    rows_iter = list(df.itertuples(index=False))

    # pass 1: count reconstructable slots per state
    count: collections.Counter = collections.Counter()
    for r in rows_iter:
        text, slots = reconstruct(r.clip_input, trg_map(r.t5_trg))
        if text is None:
            continue
        for _, word, _, _ in slots:
            key = norm(word)
            if key:
                count[key] += 1
    keep = {w for w, _ in count.most_common(args.top_n)}
    n_basic = sum(1 for w in keep if w in f2e)
    print(f"{args.lang}: {len(count)} states seen; keeping top {len(keep)} "
          f"(min count {count.most_common(args.top_n)[-1][1]}; {n_basic} are basic-8 anchors)")

    # pass 2: collect slots for kept states
    by_state: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows_iter:
        text, slots = reconstruct(r.clip_input, trg_map(r.t5_trg))
        if text is None:
            continue
        for k, word, ws, we in slots:
            key = norm(word)
            if key not in keep:
                continue
            wtext, nws, nwe = window(text, ws, we, args.window)
            if wtext[nws:nwe] != text[ws:we]:
                continue
            p, lem = f2e.get(key, (None, key))
            by_state[key].append({
                "id": f"{r.id}#{k}", "text": wtext, "word_start": nws, "word_end": nwe,
                "surface": wtext[nws:nwe], "plutchik": p,
                "valence": VALENCE[p] if p else 0, "lemma": lem,
                "seed_word_normalized": key, "source": f"masive_{args.lang}",
            })

    out_rows: list[dict] = []
    for key in keep:
        rows = by_state.get(key, [])
        rng.shuffle(rows)
        out_rows.extend(rows[: args.cap])
    rng.shuffle(out_rows)

    print(f"selected {len(out_rows)} rows across {len(keep)} states "
          f"(cap {args.cap}); {sum(1 for r in out_rows if r['plutchik'])} anchor rows")
    if not args.write:
        print("(dry run — pass --write)")
        return

    out = Path(args.out or f"pipeline/viz_hidden/masive/out/masive_{args.lang}_full_input.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
