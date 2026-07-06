"""Step 2 (fill-in) selection: project the *broader* ASI affective-state vocabulary
onto the Step-1 circumplex, one centroid per lexeme.

Step 1 used only 8 canonical basic-emotion adjectives -> grouping was trivial.
Here the unit becomes the LEXEME (gender-merged seed word): ~85 distinct affective
states, each an independent point, labeled by an EXTERNAL lexicon (RoEmoLex V3) so
"does it fill the circle" is falsifiable rather than circular.

Key discipline (mirrors Step 1):
  * ADJECTIVE FRAMES ONLY. `pattern_used` encodes the syntactic slot; we keep only
    copula/`ma simt` adjective frames so POS is held constant (the noun frames
    `mi-e frica` etc. are a systematic confound centroids do NOT remove).
  * GENDER-MERGE to a lexeme key (fericit/fericita -> fericit) — same emotion,
    more stable centroid, and it fixes the RoEmoLex join (lexicon lists the masc.).
  * ANCHORS vs FILL. The Step-1 canonical lexemes keep their theory-clean
    (plutchik, valence, arousal) so they orient the plane; every other lexeme is
    "fill", labeled by RoEmoLex (family = argmax of the 8 columns, valence = Poz-Neg).

Prints a review report; with --write emits the windowed extraction input (same
schema extract.py consumes, plus tier / roemolex_family / arousal fields).
"""
from __future__ import annotations

import argparse, json, collections, csv, random, unicodedata
from pathlib import Path

from pipeline.viz_hidden.canonical import CANONICAL, VALENCE, WHEEL, FORM2EMO, window

# Step-1 anchor arousal (canonical per-emotion, from valence_arousal.py). Anchors
# carry arousal so they can orient the model's arousal axis; fill words do not
# (RoEmoLex has no arousal dimension) -> their arousal is model-derived only.
ARO = {"joy": .5, "trust": -.1, "anticipation": .45, "surprise": .85,
       "fear": .8, "anger": .7, "disgust": .3, "sadness": -.4}

# pattern_used values whose slot is an ADJECTIVE (uniform "sunt / ma simt [adj]"
# family). Everything else (mie_short, imi_este, am_noun, simt_noun, ...) is a
# NOUN slot -> dropped to keep POS constant.
ADJ_FRAMES = {"sunt_adj_present", "ma_simt_present", "am_fost_adj_perfect",
              "eram_adj_imperfect", "ma_simteam_imperfect", "mam_simtit_perfect",
              "o_sa_fiu_future", "ma_voi_simti_future", "mas_simti_conditional",
              "ma_fac_reflexive"}

RO2EN = {"Furie": "anger", "Anticipare": "anticipation", "Dezgust": "disgust",
         "Frica": "fear", "Bucurie": "joy", "Tristete": "sadness",
         "Surpriza": "surprise", "Incredere": "trust"}


def strip(s: str) -> str:
    s = s.strip().lower()
    for a, b in [("ș", "s"), ("ş", "s"), ("ț", "t"), ("ţ", "t"),
                 ("ă", "a"), ("â", "a"), ("î", "i")]:
        s = s.replace(a, b)
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def load_roemolex(rdir: str) -> dict:
    """word (diacritic-stripped) -> aggregated emotion profile over all synset rows."""
    lex = collections.defaultdict(lambda: {"poz": 0, "neg": 0,
          "pos": collections.Counter(), **{e: 0 for e in RO2EN.values()}})
    for name in ("RoEmoLex_V3_pos.csv", "RoEmoLex_V3_expr.csv"):
        p = Path(rdir) / name
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f, delimiter=";"):
                e = lex[strip(row["word"])]
                e["poz"] += int(row["Pozitivitate"] or 0)
                e["neg"] += int(row["Negativitate"] or 0)
                if row["part_of_speech"]:
                    e["pos"][row["part_of_speech"]] += 1
                for ro, en in RO2EN.items():
                    e[en] += int(row[ro] or 0)
    return lex


def make_deinflect(lex: dict):
    """fem/other inflection -> gender-merged lemma key. Canonical forms map to their
    canonical lemma (so fericit/fericita -> fericit); others reduce to a RoEmoLex form."""
    def deinflect(w: str) -> str:
        if w in FORM2EMO:                 # canonical (incl. fem) -> its masc. lemma
            return FORM2EMO[w][1]
        cands = [w]
        if w.endswith("oasa"): cands.append(w[:-4] + "os")   # bucuroasa->bucuros
        if w.endswith("oare"): cands.append(w[:-4] + "or")   # increzatoare->increzator
        if w.endswith("a"):    cands.append(w[:-1])          # suparata->suparat
        if w.endswith("i"):    cands.append(w[:-1])
        for k in cands:
            if k in FORM2EMO:
                return FORM2EMO[k][1]
            if k in lex:
                return k
        return w
    return deinflect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="pipeline/data/benchmark_ro_asi_clean.jsonl")
    ap.add_argument("--roemolex-dir", default="deprecated/data/roemolex")
    ap.add_argument("--out", default="pipeline/viz_hidden/out/fillin_ro_input.jsonl")
    ap.add_argument("--min-count", type=int, default=30, help="min adj-frame samples per lexeme")
    ap.add_argument("--cap", type=int, default=150, help="max rows per lexeme")
    ap.add_argument("--window", type=int, default=350)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    lex = load_roemolex(args.roemolex_dir)
    deinflect = make_deinflect(lex)
    EN = list(RO2EN.values())

    # canonical anchor forms -> their (plutchik, canonical valence, arousal)
    def anchor_label(key):
        if key in FORM2EMO:
            p = FORM2EMO[key][0]
            return p, VALENCE[p], ARO[p]
        return None

    # group ADJ-frame benchmark rows by lexeme key
    by_lex = collections.defaultdict(list)
    emo_cat = collections.defaultdict(collections.Counter)
    for line in open(args.benchmark):
        d = json.loads(line)
        if d["pattern_used"] not in ADJ_FRAMES:
            continue
        s, e = d.get("seed_word_start"), d.get("seed_word_end")
        if s is None or e is None or int(e) <= int(s):
            continue
        d["seed_word_start"], d["seed_word_end"] = int(s), int(e)
        key = deinflect(d["seed_word_normalized"])
        by_lex[key].append(d)
        for c in json.loads(d["emotion_category"].replace("'", '"')) if isinstance(
                d.get("emotion_category"), str) else (d.get("emotion_category") or []):
            emo_cat[key][c] += 1

    # label each qualifying lexeme
    records = {}   # key -> dict(plutchik, valence, arousal, tier, family_src)
    for key, rows in by_lex.items():
        if len(rows) < args.min_count:
            continue
        anc = anchor_label(key)
        if anc:
            p, val, aro = anc
            records[key] = dict(plutchik=p, valence=float(val), arousal=aro,
                                tier="anchor", family_src="canonical")
        elif key in lex and any(lex[key][x] for x in EN):
            e = lex[key]
            fam = max(EN, key=lambda x: e[x])
            records[key] = dict(plutchik=fam, valence=float(e["poz"] - e["neg"]),
                                arousal=None, tier="fill", family_src="roemolex")
        # else: unlabeled -> skipped

    # ---------------- review report ----------------
    print("=" * 74)
    print("FILL-IN SELECTION (broader ASI lexemes, adjective-frame) — review report")
    print("=" * 74)
    n_anc = sum(1 for r in records.values() if r["tier"] == "anchor")
    n_fill = sum(1 for r in records.values() if r["tier"] == "fill")
    tot_rows = sum(min(len(by_lex[k]), args.cap) for k in records)
    print(f"lexemes labeled: {len(records)}  (anchor={n_anc}, fill={n_fill})   "
          f"~rows after cap {args.cap}: {tot_rows}")
    fam_ct = collections.Counter(r["plutchik"] for r in records.values())
    print("family balance (lexemes/family):", dict(sorted(fam_ct.items(), key=lambda x: -x[1])))

    for fam in WHEEL:
        keys = sorted([k for k, r in records.items() if r["plutchik"] == fam],
                      key=lambda k: -len(by_lex[k]))
        if not keys:
            continue
        print(f"\n### {fam.upper()}  ({len(keys)} lexemes)")
        for k in keys:
            r = records[k]
            tag = "A" if r["tier"] == "anchor" else " "
            ec = dict(emo_cat[k].most_common(2))
            print(f"  [{tag}] {k:16s} n={len(by_lex[k]):5d}  val={r['valence']:+.0f}  "
                  f"src={r['family_src']:9s}  bench_cat={ec}")
        # a couple of samples for the top lexeme of this family
        top = keys[0]
        pool = list(by_lex[top]); rng.shuffle(pool)
        for d in pool[:args.samples]:
            s, e = d["seed_word_start"], d["seed_word_end"]
            snip, ws, we = window(d["text"], s, e, 55)
            snip = snip.replace("\n", " ")
            print(f"        · {top}: …{snip[:ws]}⟦{snip[ws:we]}⟧{snip[we:]}…")

    if args.write:
        out_rows = []
        for key, r in records.items():
            emc = emo_cat[key].most_common(1)[0][0] if emo_cat[key] else None
            pool = list(by_lex[key]); rng.shuffle(pool); pool = pool[:args.cap]
            for d in pool:
                wt, ws, we = window(d["text"], d["seed_word_start"], d["seed_word_end"], args.window)
                if wt[ws:we] != d["text"][d["seed_word_start"]:d["seed_word_end"]]:
                    continue
                out_rows.append({
                    "id": d["id"], "text": wt, "word_start": ws, "word_end": we,
                    "surface": wt[ws:we], "plutchik": r["plutchik"], "lemma": key,
                    "valence": r["valence"], "arousal": r["arousal"], "tier": r["tier"],
                    "family_src": r["family_src"], "emo_cat": emc,
                    "seed_word_normalized": d["seed_word_normalized"],
                    "source": d.get("source")})
        rng.shuffle(out_rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for row in out_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(out_rows)} rows over {len(records)} lexemes "
              f"(min-count {args.min_count}, cap {args.cap}) -> {args.out}")


if __name__ == "__main__":
    main()
