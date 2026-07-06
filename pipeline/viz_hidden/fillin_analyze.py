"""Step 2 analysis: do the broader ASI affective states FILL the Step-1 circumplex,
or live on a more complex (higher-dimensional) shape?

Design (train/test framing):
  * The valence x arousal PLANE is defined from the ANCHORS only (the 8 canonical
    basic emotions, which carry theory-clean valence AND arousal). This is the
    Step-1 circumplex frame.
  * The 60 broader "fill" lexemes are HELD OUT: we project their per-lexeme
    centroids into that anchor-defined plane and test whether their INDEPENDENT
    RoEmoLex valence predicts where they land (generalization, not fitting).

Outputs (pipeline/viz_hidden/out/figs/):
  fig_fillin_plane_L<L>.png   anchors + fill lexemes in the valence x arousal plane
  fig_fillin_scree_L<L>.png   intrinsic dimensionality of the lexeme-centroid cloud
  metrics_fillin_L<L>.json    valence generalization r, effective dim, plane fraction
"""
from __future__ import annotations

import argparse, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
COLORS = {e: hsv_to_rgb((i / len(WHEEL), 0.65, 0.9)) for i, e in enumerate(WHEEL)}
# anchor arousal (canonical, from valence_arousal.py) — defines the arousal axis
ARO = {"joy": .5, "trust": -.1, "anticipation": .45, "surprise": .85,
       "fear": .8, "anger": .7, "disgust": .3, "sadness": -.4}


def load(npz, meta_p, inp_p, layer):
    z = np.load(npz); valid = z["valid"].astype(bool)
    inp = [json.loads(l) for l in open(inp_p)]
    X = z[f"layer_{layer}"].astype(np.float32)[valid]
    rows = [r for r, v in zip(inp, valid) if v]
    return X, rows


def eff_dim(ev):
    """participation ratio: (sum ev)^2 / sum ev^2 — a soft count of active dims."""
    ev = np.asarray(ev, float)
    return (ev.sum() ** 2) / (np.square(ev).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="pipeline/viz_hidden/out/fillin_ro_hidden.npz")
    ap.add_argument("--meta", default="pipeline/viz_hidden/out/fillin_ro_hidden.meta.jsonl")
    ap.add_argument("--input", default="pipeline/viz_hidden/out/fillin_ro_input.jsonl")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--ncomp", type=int, default=15)
    ap.add_argument("--outdir", default="pipeline/viz_hidden/out/figs")
    args = ap.parse_args()

    X, rows = load(args.npz, args.meta, args.input, args.layer)
    lemma = np.array([r["lemma"] for r in rows])
    fam = np.array([r["plutchik"] for r in rows])
    val = np.array([float(r["valence"]) for r in rows])
    tier = np.array([r["tier"] for r in rows])
    is_anc = tier == "anchor"
    aro_anc = np.array([ARO[r["plutchik"]] for r in rows])   # only meaningful for anchors

    Xs = StandardScaler().fit_transform(X)
    P = PCA(args.ncomp, random_state=0).fit_transform(Xs)

    # --- define valence & arousal axes from ANCHORS only (Step-1 frame) ---
    vc = [abs(np.corrcoef(P[is_anc, k], val[is_anc])[0, 1]) for k in range(args.ncomp)]
    ac = [abs(np.corrcoef(P[is_anc, k], aro_anc[is_anc])[0, 1]) for k in range(args.ncomp)]
    vpc = int(np.argmax(vc))
    apc = int(np.argmax([c if k != vpc else -1 for k, c in enumerate(ac)]))
    sv = np.sign(np.corrcoef(P[is_anc, vpc], val[is_anc])[0, 1])
    sa = np.sign(np.corrcoef(P[is_anc, apc], aro_anc[is_anc])[0, 1])
    xv, ya = P[:, vpc] * sv, P[:, apc] * sa
    print(f"layer {args.layer}: valence axis = PC{vpc} (anchor |r|={vc[vpc]:.2f}); "
          f"arousal axis = PC{apc} (anchor |r|={ac[apc]:.2f})")

    # --- per-lexeme centroids ---
    lexemes = sorted(set(lemma))
    cen_x = {l: xv[lemma == l].mean() for l in lexemes}
    cen_y = {l: ya[lemma == l].mean() for l in lexemes}
    lex_fam = {l: collections.Counter(fam[lemma == l]).most_common(1)[0][0] for l in lexemes}
    lex_val = {l: val[lemma == l][0] for l in lexemes}
    lex_anc = {l: bool(is_anc[lemma == l][0]) for l in lexemes}

    # --- Test 1: valence generalization on HELD-OUT fill lexemes ---
    fill = [l for l in lexemes if not lex_anc[l]]
    anc = [l for l in lexemes if lex_anc[l]]
    r_fill = np.corrcoef([cen_x[l] for l in fill], [lex_val[l] for l in fill])[0, 1]
    r_anc = np.corrcoef([cen_x[l] for l in anc], [lex_val[l] for l in anc])[0, 1]
    print(f"valence generalization (centroid valence-axis pos vs independent valence):")
    print(f"   anchors (in-sample) r={r_anc:.3f}  |  fill (held-out) r={r_fill:.3f}  "
          f"(n_fill={len(fill)})")

    # --- Intrinsic dimensionality of the lexeme-centroid cloud ---
    # Work in the top-ncomp PCA (signal) subspace; raw 2560-dim centroids are
    # noise-inflated. Then use a SPLIT-HALF reliability test so we only count
    # axes that carry genuine between-lexeme signal (not sampling noise).
    Pc_full = np.array([P[lemma == l].mean(0) for l in lexemes])   # [n_lex, ncomp]
    pcaC = PCA(random_state=0).fit(Pc_full - Pc_full.mean(0))
    ev, evr = pcaC.explained_variance_, pcaC.explained_variance_ratio_
    cum = np.cumsum(evr)
    Zc = pcaC.transform(Pc_full - Pc_full.mean(0))                 # centroid PC scores

    # split each lexeme's rows in two -> independent half-centroids in P-space
    rng = np.random.RandomState(0)
    A, B = [], []
    for l in lexemes:
        idx = np.where(lemma == l)[0]; rng.shuffle(idx)
        h = len(idx) // 2
        A.append(P[idx[:h]].mean(0)); B.append(P[idx[h:]].mean(0))
    A, B = np.array(A), np.array(B)
    Za = pcaC.transform(A - Pc_full.mean(0)); Zb = pcaC.transform(B - Pc_full.mean(0))
    rel = np.array([np.corrcoef(Za[:, k], Zb[:, k])[0, 1] for k in range(Zc.shape[1])])
    reliable = int((rel > 0.5).sum())          # axes reproducible across halves
    d_signal = eff_dim(ev[rel > 0.5]) if reliable else 0.0
    k80 = int(np.argmax(cum >= 0.80) + 1); k90 = int(np.argmax(cum >= 0.90) + 1)
    print(f"lexeme-centroid cloud ({len(lexemes)} pts) in top-{args.ncomp} PCA subspace:")
    print(f"   raw effective dim={eff_dim(ev):.2f} | reliable axes (split-half r>0.5)="
          f"{reliable} | reliable eff dim={d_signal:.2f} | PCs 80%={k80} 90%={k90}")
    print(f"   evr PC1..PC5: {[round(float(x),3) for x in evr[:5]]}")
    print(f"   split-half reliability PC1..PC6: {[round(float(x),2) for x in rel[:6]]}")

    # --- fraction of RELIABLE between-lexeme variance in the valence-arousal plane ---
    # restrict to reliable axes; compare plane (vpc,apc) vs the rest
    Pc2 = np.array([[cen_x[l], cen_y[l]] for l in lexemes])
    var_plane = Pc2[:, 0].var() + Pc2[:, 1].var()
    var_total = Pc_full.var(0).sum()
    print(f"between-lexeme variance in valence-arousal plane: "
          f"{var_plane/var_total*100:.1f}% of total (top-{args.ncomp} PCA var)")

    # --- 3rd centroid axis: what is it? ---
    order = np.argsort(Zc[:, 2])
    lo = [lexemes[i] for i in order[:6]]; hi = [lexemes[i] for i in order[-6:]]
    print(f"3rd centroid PC (evr={evr[2]:.3f}, reliab={rel[2]:.2f})  low: {lo}\n"
          f"                       high: {hi}")

    # ================= figures =================
    # (1) fill-in plane
    fig, ax = plt.subplots(figsize=(11, 9))
    for l in lexemes:
        e = lex_fam[l]; big = lex_anc[l]
        ax.scatter(cen_x[l], cen_y[l], s=260 if big else 90,
                   color=COLORS[e], edgecolor="k",
                   linewidths=1.8 if big else 0.7, alpha=0.95 if big else 0.75,
                   marker="o" if big else "s", zorder=4 if big else 3)
        ax.annotate(l, (cen_x[l], cen_y[l]), fontsize=9 if big else 7,
                    fontweight="bold" if big else "normal",
                    ha="center", va="center", zorder=6)
    ax.axhline(0, color="0.7", lw=.8); ax.axvline(0, color="0.7", lw=.8)
    ax.set_xlabel(f"valence axis (PC{vpc})  →  positive")
    ax.set_ylabel(f"arousal axis (PC{apc})  →  high")
    ax.set_title(f"RO fill-in, layer {args.layer}: 8 anchors (○) + {len(fill)} broader "
                 f"ASI states (□)\nvalence generalization on held-out fill r={r_fill:.2f}  "
                 f"| plane holds {var_plane/var_total*100:.0f}% of centroid variance")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS[e], label=e,
                              markeredgecolor="k") for e in WHEEL],
              loc="best", fontsize=8, ncol=2, title="RoEmoLex family")
    ax.grid(alpha=0.2)
    p1 = f"{args.outdir}/fig_fillin_plane_L{args.layer}.png"
    plt.tight_layout(); plt.savefig(p1, dpi=140); plt.close()

    # (2) scree
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, 11), evr[:10], color="0.6")
    ax.plot(range(1, 11), cum[:10], "o-", color="C3", label="cumulative")
    ax.axhline(0.8, color="0.7", ls="--", lw=.8)
    ax.set_xlabel("principal component of the lexeme-centroid cloud")
    ax.set_ylabel("variance explained")
    ax.set_title(f"RO fill-in, layer {args.layer}: intrinsic dimensionality\n"
                 f"reliable axes (split-half)={reliable}  |  {k80} PCs for 80%, {k90} for 90%")
    ax.legend(); ax.grid(alpha=0.2)
    p2 = f"{args.outdir}/fig_fillin_scree_L{args.layer}.png"
    plt.tight_layout(); plt.savefig(p2, dpi=140); plt.close()

    metrics = {"layer": args.layer, "n_lexemes": len(lexemes), "n_fill": len(fill),
               "n_anchor": len(anc), "valence_axis_pc": vpc, "arousal_axis_pc": apc,
               "valence_r_anchor": round(float(r_anc), 3),
               "valence_r_fill_heldout": round(float(r_fill), 3),
               "raw_effective_dim": round(float(eff_dim(ev)), 2),
               "reliable_axes_splithalf": reliable,
               "reliable_effective_dim": round(float(d_signal), 2),
               "pcs_for_80pct": k80, "pcs_for_90pct": k90,
               "plane_variance_frac": round(float(var_plane / var_total), 3),
               "evr_top5": [round(float(x), 4) for x in evr[:5]],
               "reliability_top6": [round(float(x), 3) for x in rel[:6]]}
    mp = f"{args.outdir}/metrics_fillin_L{args.layer}.json"
    json.dump(metrics, open(mp, "w"), indent=2)
    print("wrote", p1, p2, mp)


if __name__ == "__main__":
    main()
