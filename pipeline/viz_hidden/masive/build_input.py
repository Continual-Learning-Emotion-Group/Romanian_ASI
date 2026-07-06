"""Build MASIVE basic-emotion (Plutchik-8) extraction input for the ring test.

MASIVE is a T5-style masked-span task: `clip_input` has affective spans replaced
by <extra_id_K> and `t5_trg` gives the gold fills ("<extra_id_0> valued ...").
We reconstruct each post by filling EVERY slot with its gold word (so context is
the natural sentence), then for each slot whose word is a canonical basic-emotion
lexeme we emit ONE windowed extraction row pointing at that slot's char span.

Output row schema matches pipeline/viz_hidden/select.py so the remote extractor
(pipeline/viz_hidden/extract.py) consumes it unchanged:
    id, text, word_start, word_end, surface, plutchik, valence, lemma,
    seed_word_normalized, source

Usage:
    python -m pipeline.viz_hidden.masive.build_input --lang en \
        --masive-dir <dir containing masive/> \
        --out pipeline/viz_hidden/masive/out/masive_en_input.jsonl \
        --cap 600 --window 350 --write
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import pandas as pd

from pipeline.viz_hidden.masive.lexicons import (TABLES, VALENCE, WHEEL, form2emo, norm)

SENT = re.compile(r"<extra_id_(\d+)>")


def trg_map(t5_trg: str) -> dict[int, str]:
    """Parse '<extra_id_0> valued <extra_id_1> seen' -> {0: 'valued', 1: 'seen'}."""
    parts = SENT.split(str(t5_trg))          # ['', '0', ' valued ', '1', ' seen']
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def reconstruct(clip_input: str, tmap: dict[int, str]):
    """Fill every <extra_id_K> with tmap[K]; return (text, slots) where slots is
    a list of (k, word, char_start, char_end) in the reconstructed text. Returns
    (None, None) if clip_input references a slot missing from tmap (~1% of rows)."""
    res, slots, last = "", [], 0
    for m in SENT.finditer(str(clip_input)):
        k = int(m.group(1))
        if k not in tmap:
            return None, None
        res += clip_input[last:m.start()]
        ws = len(res)
        res += tmap[k]
        slots.append((k, tmap[k], ws, len(res)))
        last = m.end()
    res += clip_input[last:]
    return res, slots


def window(text: str, s: int, e: int, half: int):
    """Keep `half` chars either side of [s,e), snapping outward to whitespace."""
    lo, hi = max(0, s - half), min(len(text), e + half)
    while lo > 0 and not text[lo - 1].isspace():
        lo -= 1
    while hi < len(text) and not text[hi].isspace():
        hi += 1
    return text[lo:hi], s - lo, e - lo


def load_masive(masive_dir: Path, lang: str, splits: list[str]) -> pd.DataFrame:
    frames = []
    for sp in splits:
        fp = masive_dir / "masive" / lang / f"{sp}.csv"
        frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masive-dir", required=True,
                    help="directory that CONTAINS the 'masive/' folder")
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=int, default=350, help="chars either side of the word")
    ap.add_argument("--cap", type=int, default=600, help="max rows per emotion")
    ap.add_argument("--samples", type=int, default=2, help="example snippets per lemma in report")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    f2e = form2emo(args.lang)
    table = TABLES[args.lang]
    df = load_masive(Path(args.masive_dir), args.lang, args.splits.split(","))

    # --- scan: reconstruct posts, collect basic-emotion slots ---
    by_form: dict[str, list[dict]] = collections.defaultdict(list)
    n_rows = n_recon = n_skip = 0
    for r in df.itertuples(index=False):
        n_rows += 1
        text, slots = reconstruct(r.clip_input, trg_map(r.t5_trg))
        if text is None:
            n_skip += 1
            continue
        n_recon += 1
        for k, word, ws, we in slots:
            key = norm(word)
            if key not in f2e:
                continue
            wtext, nws, nwe = window(text, ws, we, args.window)
            if wtext[nws:nwe] != text[ws:we]:      # window integrity
                continue
            p, lem = f2e[key]
            by_form[key].append({
                "id": f"{r.id}#{k}", "text": wtext, "word_start": nws, "word_end": nwe,
                "surface": wtext[nws:nwe], "plutchik": p, "valence": VALENCE[p],
                "lemma": lem, "seed_word_normalized": key, "source": f"masive_{args.lang}",
            })

    # --- report ---
    print("=" * 74)
    print(f"MASIVE {args.lang.upper()} basic-emotion selection — review report")
    print(f"rows scanned={n_rows}  reconstructed={n_recon}  skipped(missing slot)={n_skip}")
    print("=" * 74)
    grand = 0
    for p in WHEEL:
        forms = [f for lem in table[p].values() for f in lem]
        emo_total = sum(len(by_form.get(norm(f), [])) for f in forms)
        grand += emo_total
        print(f"\n### {p.upper()}  ({emo_total} rows, valence {VALENCE[p]:+d})")
        for lem, lem_forms in table[p].items():
            per = "  ".join(f"{f}={len(by_form.get(norm(f), []))}" for f in lem_forms)
            print(f"  {lem:12s} [{per}]")
        pool = [row for f in forms for row in by_form.get(norm(f), [])]
        rng.shuffle(pool)
        for row in pool[: args.samples]:
            snip, ws, we = window(row["text"], row["word_start"], row["word_end"], 55)
            snip = snip.replace("\n", " ")
            print(f"    · …{snip[:ws]}⟦{snip[ws:we]}⟧{snip[we:]}…")
    print(f"\nGRAND TOTAL: {grand} basic-emotion rows")
    print("balance:", {p: sum(len(by_form.get(norm(f), []))
                              for lem in table[p].values() for f in lem) for p in WHEEL})

    if not args.write:
        print("\n(dry run — pass --write to emit the extraction input)")
        return

    # --- balance with per-emotion cap (shuffle synonyms so the cap mixes them) ---
    by_emo: dict[str, list[dict]] = collections.defaultdict(list)
    for key, rows in by_form.items():
        by_emo[f2e[key][0]].extend(rows)
    out_rows: list[dict] = []
    for p in WHEEL:
        rows = by_emo[p]
        rng.shuffle(rows)
        out_rows.extend(rows[: args.cap])
    rng.shuffle(out_rows)

    out = Path(args.out or f"pipeline/viz_hidden/masive/out/masive_{args.lang}_input.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out_rows)} rows (cap {args.cap}/emotion) -> {out}")
    print("final balance:", {p: min(len(by_emo[p]), args.cap) for p in WHEEL})


if __name__ == "__main__":
    main()
