"""Confound diagnostics + aggregated (centroid) circumplex test.

Two things the token-cloud PCA can't answer:
  1. Is the detached 'sadness' blob a semantic fact or a source/word confound?
  2. Does the circle appear once we average out per-sample context (per-word and
     per-emotion centroids)?
"""
from __future__ import annotations

import argparse, json, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
COLORS = {e: hsv_to_rgb((i / len(WHEEL), 0.65, 0.9)) for i, e in enumerate(WHEEL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--outdir", default="pipeline/viz_hidden/out/figs")
    ap.add_argument("--lang", default="ro"); ap.add_argument("--min-word", type=int, default=15)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz); valid = z["valid"].astype(bool)
    meta = [json.loads(l) for l in open(args.meta)]
    lab = np.array([m["plutchik"] for m in meta])[valid]
    word = np.array([m.get("seed_word_normalized") for m in meta])[valid]
    src = np.array([m.get("source") for m in meta])[valid]
    X = z[f"layer_{args.layer}"].astype(np.float32)[valid]
    Xs = StandardScaler().fit_transform(X)
    proj = PCA(3, random_state=0).fit(Xs); P = proj.transform(Xs)

    # 1. Confound: what dominates the far-left (low PC1) region?
    thr = np.percentile(P[:, 0], 5)
    far = P[:, 0] < thr
    print(f"=== far-left tail (PC1<{thr:.1f}, n={far.sum()}) ===")
    print("  emotions:", collections.Counter(lab[far]).most_common())
    print("  sources :", collections.Counter(src[far]).most_common())
    print("  top seed words:", collections.Counter(word[far]).most_common(12))
    print(f"\n=== sadness overall: sources ===", collections.Counter(src[lab == 'sadness']).most_common())
    print("=== sadness top words:", collections.Counter(word[lab == 'sadness']).most_common(10))

    # 2. Per-seed-word centroids (averages out per-sample context)
    words, wc_vec, wc_lab, wc_n = [], [], [], []
    for w in set(word):
        m = word == w
        if m.sum() >= args.min_word:
            words.append(w); wc_vec.append(Xs[m].mean(0))
            wc_lab.append(collections.Counter(lab[m]).most_common(1)[0][0]); wc_n.append(int(m.sum()))
    wc_vec = np.array(wc_vec); wc_lab = np.array(wc_lab)
    wp = PCA(2, random_state=0).fit_transform(wc_vec)
    print(f"\nword-centroids: {len(words)} words (>= {args.min_word} occ)")

    # 3. Per-emotion centroids in full standardized space -> PCA(2)
    ec = np.array([Xs[lab == e].mean(0) for e in WHEEL])
    ep = PCA(2, random_state=0).fit(ec)
    ecp = ep.transform(ec)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    ax = axes[0]
    for e in WHEEL:
        m = wc_lab == e
        if m.any():
            ax.scatter(wp[m, 0], wp[m, 1], s=[wc_n[i] for i in np.where(m)[0]],
                       color=COLORS[e], alpha=0.7, edgecolor="k", linewidths=0.4, label=e)
    ax.set_title(f"{args.lang.upper()} layer {args.layer}: per-seed-word centroids (size=freq)\n"
                 "context averaged out per word")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[1]
    ring = np.array([ecp[i] for i in range(len(WHEEL))] + [ecp[0]])
    ax.plot(ring[:, 0], ring[:, 1], "-", color="0.4", lw=1.2)
    for i, e in enumerate(WHEEL):
        ax.scatter(*ecp[i], s=320, color=COLORS[e], edgecolor="k", linewidths=1.5)
        ax.annotate(e, ecp[i], fontsize=12, fontweight="bold", ha="center", va="center")
    ax.set_title(f"8 emotion centroids only  (PCA on centroids)\nevr={ep.explained_variance_ratio_[:2].round(2)}")
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    p = Path(args.outdir) / f"fig_{args.lang}_centroids_layer{args.layer}.png"
    plt.savefig(p, dpi=140); plt.close()
    print("wrote", p)


if __name__ == "__main__":
    main()
