"""Is there ONE axis that explains valence?

Trains a single linear direction (logistic probe) on the basic-8 anchors only
(positive: joy/trust/anticipation  vs  negative: fear/sadness/disgust/anger;
surprise is neutral -> held out), then:
  * reports 5-fold CV AUC of that one axis (how well a single direction
    separates valence),
  * projects ALL ~200 states onto the SAME axis and lists the extremes, to check
    it generalizes to open-vocabulary states it never saw in training,
  * reports how much variance the axis carries and its alignment with PC1
    (is valence a high-variance direction or a subtle low-variance one?).

Usage:
    python -m pipeline.viz_hidden.masive.valence_axis \
        --npz .../masive_en_full_hidden.npz --meta .../masive_en_full_hidden.meta.jsonl \
        --outdir .../figs --lang en
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pipeline.viz_hidden.masive.analyze_full import load

POS = {"joy", "trust", "anticipation"}
NEG = {"fear", "sadness", "disgust", "anger"}


def valence_probe(Xstd, plut, states):
    """1-axis logistic valence probe on the basic-8 anchors, scored by
    leave-one-WORD-out CV (each anchor word held out entirely, so the axis must
    generalize to words it never saw). Returns (auc, unit_direction, clf)."""
    y = np.full(len(plut), -1)
    y[np.isin(plut, list(POS))] = 1
    y[np.isin(plut, list(NEG))] = 0
    mask = y >= 0
    Xa, ya, ga = Xstd[mask], y[mask], states[mask]
    clf = LogisticRegression(C=0.1, max_iter=2000)
    scores = cross_val_predict(clf, Xa, ya, groups=ga, cv=LeaveOneGroupOut(),
                               method="decision_function")
    auc = roc_auc_score(ya, scores)
    clf.fit(Xa, ya)
    w = clf.coef_[0]
    return float(auc), w / np.linalg.norm(w), clf


def sweep(X, plut, states, layers):
    rows = []
    for li in layers:
        Xs = StandardScaler().fit_transform(X[li])
        auc, _, _ = valence_probe(Xs, plut, states)
        rows.append({"layer": li, "valence_auc": auc})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--outdir", default="pipeline/viz_hidden/masive/out/figs")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    X, states, plut, layers = load(args.npz, args.meta)
    rows = sweep(X, plut, states, layers)
    print("\nlayer  valence_AUC (1 axis, basic-8, leave-one-WORD-out CV)")
    for r in rows:
        print(f"{r['layer']:5d}   {r['valence_auc']:.3f}")
    best = max(rows, key=lambda r: r["valence_auc"])
    layer = args.layer if args.layer is not None else best["layer"]
    print(f"\nbest layer {best['layer']} (AUC {best['valence_auc']:.3f}); using layer {layer}")

    Xs = StandardScaler().fit_transform(X[layer])
    auc, w, _ = valence_probe(Xs, plut, states)
    score = Xs @ w                                   # projection onto the valence axis
    # per-state mean score
    uniq = np.unique(states)
    smean = {s: float(score[states == s].mean()) for s in uniq}
    ranked = sorted(uniq, key=lambda s: smean[s])

    # how special is this direction? variance along it vs PC1, alignment with PC1
    pca = PCA(n_components=1, random_state=0).fit(Xs)
    var_w = float(np.var(score))
    var_pc1 = float(pca.explained_variance_[0])
    align_pc1 = float(abs(np.dot(w, pca.components_[0])))
    total_var = float(np.var(Xs, axis=0).sum())
    print(f"\nvalence-axis variance share: {100*var_w/total_var:.1f}%   "
          f"PC1 share: {100*var_pc1/total_var:.1f}%   |cos(w,PC1)|={align_pc1:.2f}")
    anchors = [s for s in uniq if plut[states == s][0] in (POS | NEG)]
    non_anchor = [str(s) for s in ranked if s not in set(anchors)]
    print("\nmost NEGATIVE open-vocab states:", non_anchor[:12])
    print("most POSITIVE open-vocab states:", non_anchor[-12:][::-1])

    # ---- figure: states along the single valence axis ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})
    rng = np.random.RandomState(0)
    xs = np.array([smean[s] for s in uniq])
    ys = rng.uniform(0, 1, len(uniq))
    is_anc = np.array([plut[states == s][0] in (POS | NEG) for s in uniq])
    ax1.scatter(xs[~is_anc], ys[~is_anc], s=16, color="0.6", alpha=0.6, label="open-vocab states")
    ax1.scatter(xs[is_anc], ys[is_anc], s=40, color="tab:red", alpha=0.9, label="basic-8 anchor")
    # annotate extremes + a few anchors
    show = non_anchor[:8] + non_anchor[-8:] + anchors
    for s in show:
        ax1.annotate(s, (smean[s], dict(zip(uniq, ys))[s]), fontsize=7,
                     color="black" if s in anchors else "0.3", ha="center")
    ax1.set_yticks([]); ax1.set_xlabel("projection onto the valence axis  (neg <-- --> pos)")
    ax1.set_title(f"{args.lang.upper()} layer {layer}: 200 states on ONE valence axis "
                  f"(CV AUC={auc:.2f}, trained on 8 basic emotions only)")
    ax1.legend(loc="upper left", fontsize=8)
    L = [r["layer"] for r in rows]
    ax2.plot(L, [r["valence_auc"] for r in rows], "o-", color="tab:blue")
    ax2.axhline(0.5, ls="--", color="0.7"); ax2.set_ylim(0.4, 1.02)
    ax2.set_xlabel("layer"); ax2.set_ylabel("valence AUC"); ax2.grid(alpha=0.2)
    plt.tight_layout()
    p = Path(args.outdir) / f"fig_{args.lang}_valence_axis.png"
    plt.savefig(p, dpi=140); plt.close()

    json.dump({"layers": rows, "headline_layer": layer, "auc": auc,
               "valence_var_share": var_w / total_var, "pc1_var_share": var_pc1 / total_var,
               "align_pc1": align_pc1,
               "most_negative": non_anchor[:12], "most_positive": non_anchor[-12:][::-1]},
              open(Path(args.outdir) / f"metrics_{args.lang}_valence.json", "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
