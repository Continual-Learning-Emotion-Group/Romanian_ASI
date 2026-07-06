"""Project the broader ASI states onto the SAME 8 basic-emotion centroids.

Complements fillin_analyze.py: that projected onto a 2-axis valence x arousal plane
*derived* from anchors; here we project onto the 8 centroids themselves (a <=7-dim
emotion basis), and ask two interpretable questions per broader state:

  1. BLEND: cosine loading on each of the 8 basic emotions (is `coplesit` = sadness?
     is `frustrat` = anger? is `disperat` = fear+sadness?).
  2. RESIDUAL: what fraction of the state's displacement-from-mean lives OUTSIDE the
     span of the 8 centroids — i.e. affective content the basic 8 cannot express.

Done in the top-K PCA signal subspace (raw 2560-dim residual is noise-dominated),
with a split-half reliability check so the residual is genuine signal.

Outputs: fig_project8_heatmap_L<L>.png, metrics_project8_L<L>.json
"""
from __future__ import annotations

import argparse, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WHEEL = ["joy", "trust", "anticipation", "surprise", "fear", "anger", "disgust", "sadness"]


def load(npz, inp_p, layer):
    z = np.load(npz); valid = z["valid"].astype(bool)
    inp = [json.loads(l) for l in open(inp_p)]
    X = z[f"layer_{layer}"].astype(np.float32)[valid]
    rows = [r for r, v in zip(inp, valid) if v]
    return X, rows


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="pipeline/viz_hidden/out/fillin_ro_hidden.npz")
    ap.add_argument("--input", default="pipeline/viz_hidden/out/fillin_ro_input.jsonl")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--ncomp", type=int, default=30)
    ap.add_argument("--outdir", default="pipeline/viz_hidden/out/figs")
    args = ap.parse_args()

    X, rows = load(args.npz, args.input, args.layer)
    lemma = np.array([r["lemma"] for r in rows])
    fam = np.array([r["plutchik"] for r in rows])
    tier = np.array([r["tier"] for r in rows])
    is_anc = tier == "anchor"

    Xs = StandardScaler().fit_transform(X)
    P = PCA(args.ncomp, random_state=0).fit_transform(Xs)     # signal subspace

    # --- the 8 basic-emotion centroids from ANCHOR samples (Step-1's 8) ---
    cen8 = {e: P[is_anc & (fam == e)].mean(0) for e in WHEEL}
    mu = np.mean([cen8[e] for e in WHEEL], axis=0)            # center of the 8
    A = np.array([cen8[e] - mu for e in WHEEL])               # 8 centered anchor vecs
    # orthonormal basis Q of the span of the 8 centroids (<=7-dim)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > s[0] * 1e-6).sum())      # =7 after centering 8 pts
    Q = Vt[:rank]                                             # [rank, ncomp]
    print(f"layer {args.layer}: 8-centroid span rank = {rank} (of {args.ncomp} PCA dims)")

    # --- broader ('fill') lexeme centroids, projected onto the 8-centroid span ---
    fills = sorted({l for l, a in zip(lemma, is_anc) if not a})
    prof, resid_frac, near, notable = {}, {}, {}, []
    for l in fills:
        c = P[lemma == l].mean(0) - mu
        proj = c @ Q.T @ Q                                   # component in span of 8
        resid_frac[l] = float(np.linalg.norm(c - proj) ** 2 / (np.linalg.norm(c) ** 2 + 1e-12))
        # cosine loading on each anchor direction
        prof[l] = np.array([float(np.dot(unit(c), unit(cen8[e] - mu))) for e in WHEEL])
        near[l] = WHEEL[int(np.argmax(prof[l]))]

    # RoEmoLex family per fill lexeme + agreement with nearest anchor
    lex_fam = {l: collections.Counter(fam[lemma == l]).most_common(1)[0][0] for l in fills}
    agree = np.mean([near[l] == lex_fam[l] for l in fills])
    # nearest-anchor agreement counting top-2 (blends)
    top2 = {l: [WHEEL[i] for i in np.argsort(prof[l])[::-1][:2]] for l in fills}
    agree2 = np.mean([lex_fam[l] in top2[l] for l in fills])

    # --- span-of-8 vs residual: how much broader-state variance the 8 capture ---
    Cf = np.array([P[lemma == l].mean(0) - mu for l in fills])
    proj_all = Cf @ Q.T @ Q
    var_span = (np.linalg.norm(proj_all, axis=1) ** 2).sum()
    var_tot = (np.linalg.norm(Cf, axis=1) ** 2).sum()
    frac_span = var_span / var_tot

    # --- split-half reliability of the residual (is the leftover real signal?) ---
    rng = np.random.RandomState(0)
    rA, rB = [], []
    for l in fills:
        idx = np.where(lemma == l)[0]; rng.shuffle(idx); h = len(idx) // 2
        for half, store in ((idx[:h], rA), (idx[h:], rB)):
            c = P[half].mean(0) - mu
            store.append(c - c @ Q.T @ Q)                    # residual vector
        # correlation of the two half residual vectors, per lexeme, averaged
    rel_resid = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(rA, rB)]))

    print(f"broader states: span-of-8 captures {frac_span*100:.1f}% of their variance, "
          f"residual {(1-frac_span)*100:.1f}%")
    print(f"residual split-half reliability (mean per-lexeme vec corr) = {rel_resid:.2f}")
    print(f"nearest-anchor == RoEmoLex family: {agree*100:.0f}%   (in top-2: {agree2*100:.0f}%)")

    # notable: highest-residual (off-manifold) and clearest blends
    hi_res = sorted(fills, key=lambda l: -resid_frac[l])[:8]
    print("\nmost off-basic-8 (high residual):",
          [f"{l}({resid_frac[l]*100:.0f}%)" for l in hi_res])
    print("blends (top-2 anchors, examples):")
    for l in sorted(fills, key=lambda l: -abs(prof[l][np.argsort(prof[l])[-2]]))[:12]:
        a, b = top2[l]
        print(f"  {l:14s} -> {a} {prof[l][WHEEL.index(a)]:.2f} + {b} {prof[l][WHEEL.index(b)]:.2f}"
              f"   (RoEmoLex={lex_fam[l]}, resid={resid_frac[l]*100:.0f}%)")

    # ================= heatmap: 8-emotion cosine profile per broader state =========
    order = sorted(fills, key=lambda l: (WHEEL.index(lex_fam[l]) if lex_fam[l] in WHEEL else 9,
                                         -prof[l].max()))
    M = np.array([prof[l] for l in order])
    R = np.array([resid_frac[l] for l in order])
    fig, (axh, axr) = plt.subplots(1, 2, figsize=(11, 15),
                                   gridspec_kw={"width_ratios": [8, 1]}, sharey=True)
    im = axh.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-.8, vmax=.8)
    axh.set_xticks(range(8)); axh.set_xticklabels(WHEEL, rotation=40, ha="right")
    axh.set_yticks(range(len(order)))
    axh.set_yticklabels([f"{l}  ·{lex_fam[l][:4]}" for l in order], fontsize=7)
    axh.set_title(f"L{args.layer}: broader states projected on the 8 basic-emotion "
                  f"centroids\n(cosine loading; rows grouped by RoEmoLex family)", fontsize=10)
    fig.colorbar(im, ax=axh, fraction=0.025, pad=0.01, label="cosine to anchor")
    axr.barh(range(len(order)), R, color="0.5"); axr.set_xlim(0, 1)
    axr.set_title("residual\n(outside 8)", fontsize=9); axr.invert_yaxis()
    axr.set_xlabel("frac")
    plt.tight_layout()
    p = f"{args.outdir}/fig_project8_heatmap_L{args.layer}.png"
    plt.savefig(p, dpi=130); plt.close()

    metrics = {"layer": args.layer, "span8_rank": rank, "n_fill": len(fills),
               "span8_variance_frac": round(float(frac_span), 3),
               "residual_frac": round(float(1 - frac_span), 3),
               "residual_splithalf_reliability": round(rel_resid, 3),
               "nearest_anchor_agree": round(float(agree), 3),
               "nearest_anchor_agree_top2": round(float(agree2), 3),
               "high_residual_states": {l: round(resid_frac[l], 3) for l in hi_res}}
    mp = f"{args.outdir}/metrics_project8_L{args.layer}.json"
    json.dump(metrics, open(mp, "w"), indent=2)
    print("\nwrote", p, mp)


if __name__ == "__main__":
    main()
