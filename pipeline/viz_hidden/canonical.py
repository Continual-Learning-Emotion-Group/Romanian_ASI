"""Canonical basic-emotion lexeme selection (Plutchik-8, safe words only).

Selects benchmark rows purely by the SURFACE WORD being one of a small set of
classical basic-emotion lexemes + their gender/number inflections (matched on
`seed_word_normalized`, which is diacritic-stripped & lowercased). Emotion label
comes from the word, not from emotion_category. This removes the lexical/domain
confound of the category-family approach (e.g. 'dor' dominating sadness).

Prints a review report (counts, balance, category sanity, samples) and, with
--write, emits the windowed extraction input.
"""
from __future__ import annotations

import argparse, json, collections, random
from pathlib import Path

# Adjectives-only (uniform "sunt / ma simt [adj]" frame -> no noun/adjective POS
# confound), several classical synonyms per emotion so the vertex is an emotion,
# not a single lexeme. Plurals omitted (0 hits in this 1st-person-sg benchmark).
# plutchik -> {lemma: [normalized forms (m.sg, f.sg)]}
CANONICAL: dict[str, dict[str, list[str]]] = {
    "joy":          {"fericit": ["fericit", "fericita"], "bucuros": ["bucuros", "bucuroasa"],
                     "incantat": ["incantat", "incantata"]},
    "trust":        {"increzator": ["increzator", "increzatoare"]},
    "fear":         {"speriat": ["speriat", "speriata"], "ingrozit": ["ingrozit", "ingrozita"],
                     "inspaimantat": ["inspaimantat", "inspaimantata"],
                     "infricosat": ["infricosat", "infricosata"]},
    "surprise":     {"surprins": ["surprins", "surprinsa"], "uimit": ["uimit", "uimita"],
                     "socat": ["socat", "socata"], "uluit": ["uluit", "uluita"],
                     "mirat": ["mirat", "mirata"], "stupefiat": ["stupefiat", "stupefiata"]},
    "sadness":      {"trist": ["trist", "trista"], "deprimat": ["deprimat", "deprimata"],
                     "nefericit": ["nefericit", "nefericita"], "mahnit": ["mahnit", "mahnita"],
                     "amarat": ["amarat", "amarata"]},
    "disgust":      {"dezgustat": ["dezgustat", "dezgustata"], "scarbit": ["scarbit", "scarbita"],
                     "oripilat": ["oripilat", "oripilata"]},
    "anger":        {"furios": ["furios", "furioasa"], "revoltat": ["revoltat", "revoltata"],
                     "indignat": ["indignat", "indignata"], "nervos": ["nervos", "nervoasa"],
                     "manios": ["manios", "manioasa"], "enervat": ["enervat", "enervata"],
                     "iritat": ["iritat", "iritata"]},
    "anticipation": {"curios": ["curios", "curioasa"], "nerabdator": ["nerabdator", "nerabdatoare"],
                     "dornic": ["dornic", "dornica"], "intrigat": ["intrigat", "intrigata"]},
}
WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
VALENCE = {"joy": 1, "trust": 1, "anticipation": 1, "surprise": 0,
           "fear": -1, "sadness": -1, "disgust": -1, "anger": -1}

# form -> (plutchik, lemma)
FORM2EMO = {f: (p, lem) for p, lemmas in CANONICAL.items()
            for lem, forms in lemmas.items() for f in forms}


def window(text, s, e, half):
    lo, hi = max(0, s - half), min(len(text), e + half)
    while lo > 0 and not text[lo - 1].isspace(): lo -= 1
    while hi < len(text) and not text[hi].isspace(): hi += 1
    return text[lo:hi], s - lo, e - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="pipeline/data/benchmark_ro_asi_clean.jsonl")
    ap.add_argument("--out", default="pipeline/viz_hidden/out/canon_ro_input.jsonl")
    ap.add_argument("--window", type=int, default=350)
    ap.add_argument("--cap", type=int, default=600, help="max rows per emotion (use all if fewer)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_form = collections.defaultdict(list)   # form -> [row dict]
    cat_by_form = collections.defaultdict(collections.Counter)
    for line in open(args.benchmark):
        d = json.loads(line)
        w = d.get("seed_word_normalized")
        if w in FORM2EMO:
            s, e = d.get("seed_word_start"), d.get("seed_word_end")
            if s is None or e is None or e <= s:
                continue
            by_form[w].append(d)
            for c in d.get("emotion_category", []):
                cat_by_form[w][c] += 1

    # ---- report ----
    print("=" * 72)
    print("CANONICAL BASIC-EMOTION SELECTION — review report")
    print("=" * 72)
    grand = 0
    for p in WHEEL:
        emo_total = sum(len(by_form.get(f, [])) for lem in CANONICAL[p].values() for f in lem)
        grand += emo_total
        print(f"\n### {p.upper()}  ({emo_total} rows)")
        for lem, forms in CANONICAL[p].items():
            lem_total = sum(len(by_form.get(f, [])) for f in forms)
            per_form = "  ".join(f"{f}={len(by_form.get(f, []))}" for f in forms)
            print(f"  {lem:12s} total={lem_total:5d}   [{per_form}]")
            cats = sum((cat_by_form[f] for f in forms), collections.Counter())
            print(f"      categories: {dict(cats.most_common(4))}")
            # samples
            pool = [r for f in forms for r in by_form.get(f, [])]
            rng.shuffle(pool)
            for r in pool[: args.samples]:
                s, e = r["seed_word_start"], r["seed_word_end"]
                snip, ws, we = window(r["text"], s, e, 60)
                snip = snip.replace("\n", " ")
                hi = snip[:ws] + "⟦" + snip[ws:we] + "⟧" + snip[we:]
                print(f"        · [{r['source']}] …{hi}…")
    print(f"\nGRAND TOTAL: {grand} rows across {len(FORM2EMO)} forms")
    print("balance (per emotion):",
          {p: sum(len(by_form.get(f, [])) for lem in CANONICAL[p].values() for f in lem) for p in WHEEL})

    if args.write:
        by_emo = collections.defaultdict(list)
        for f, lst in by_form.items():
            p, lem = FORM2EMO[f]
            for d in lst:
                wtext, ws, we = window(d["text"], d["seed_word_start"], d["seed_word_end"], args.window)
                if wtext[ws:we] != d["text"][d["seed_word_start"]:d["seed_word_end"]]:
                    continue
                by_emo[p].append({"id": d["id"], "text": wtext, "word_start": ws, "word_end": we,
                                  "surface": wtext[ws:we], "plutchik": p, "lemma": lem,
                                  "valence": VALENCE[p], "seed_word_normalized": f,
                                  "source": d.get("source")})
        rows = []
        for p in WHEEL:
            pr = by_emo[p]
            rng.shuffle(pr)                       # cap mixes synonyms evenly
            rows.extend(pr[: args.cap])
        rng.shuffle(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows)} rows (cap {args.cap}/emotion) -> {args.out}")
        print("final balance:", {p: min(len(by_emo[p]), args.cap) for p in WHEEL})


if __name__ == "__main__":
    main()
