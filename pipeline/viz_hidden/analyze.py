"""PCA / circumplex analysis of extracted hidden states (runs locally).

Tests the core hypothesis: do the Plutchik-8 emotion centroids arrange in a ring
(Russell's circumplex) in the model's mid-layer hidden space, and does one axis
track valence? Produces a layer sweep, a headline mid-layer figure, and a 3D view.

Usage:
    python -m pipeline.viz_hidden.analyze \
        --npz pipeline/viz_hidden/out/tier1_ro_hidden.npz \
        --meta pipeline/viz_hidden/out/tier1_ro_hidden.meta.jsonl \
        --outdir pipeline/viz_hidden/out/figs --lang ro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Plutchik wheel order (adjacent emotions are adjacent on the wheel)
WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
# color each emotion by its wheel angle -> a rainbow that goes around the ring,
# so a recovered circumplex shows smoothly-varying hue around the circle.
COLORS = {e: hsv_to_rgb((i / len(WHEEL), 0.65, 0.9)) for i, e in enumerate(WHEEL)}


def load(npz_path: str, meta_path: str):
    z = np.load(npz_path)
    valid = z["valid"].astype(bool)
    layers = list(z["save_layers"])
    meta = [json.loads(l) for l in open(meta_path)]
    labels = np.array([m["plutchik"] for m in meta])
    valence = np.array([m["valence"] for m in meta], dtype=float)
    X = {int(li): z[f"layer_{li}"].astype(np.float32)[valid] for li in layers}
    return X, labels[valid], valence[valid], [int(x) for x in layers]


def project(x, n=3, drop_pc1=False):
    xs = StandardScaler().fit_transform(x)
    p = PCA(n_components=n + (1 if drop_pc1 else 0), random_state=0).fit(xs)
    z = p.transform(xs)
    if drop_pc1:
        z = z[:, 1:]
        evr = p.explained_variance_ratio_[1:]
    else:
        evr = p.explained_variance_ratio_
    return z, evr


def centroids(z, labels):
    return {e: z[labels == e].mean(0) for e in WHEEL if (labels == e).any()}


def circle_residual(cents2d):
    """Fit a circle to the 2D centroids; return normalized RMS radial residual
    (0 = perfectly on a circle). Kasa algebraic fit."""
    P = np.array([cents2d[e] for e in WHEEL if e in cents2d])
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x**2 + y**2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(c + cx**2 + cy**2)
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(np.sqrt(np.mean((radii - r) ** 2)) / r), (cx, cy, r)


def sweep(X, labels, valence, layers):
    rows = []
    for li in layers:
        z, evr = project(X[li], n=3)
        # silhouette of Plutchik clusters (subsample for speed if large)
        sil = silhouette_score(z, labels) if len(z) <= 6000 else \
            silhouette_score(z[:6000], labels[:6000])
        cents = centroids(z[:, :2], labels)
        resid, _ = circle_residual(cents)
        vcorr = max(abs(np.corrcoef(z[:, k], valence)[0, 1]) for k in range(2))
        rows.append({"layer": li, "evr_pc1": float(evr[0]), "evr_pc2": float(evr[1]),
                     "evr_top3": float(evr[:3].sum()), "silhouette": float(sil),
                     "circle_resid": resid, "best_valence_corr": float(vcorr)})
    return rows


def plot_headline(X, labels, valence, layer, outdir, lang):
    z, evr = project(X[layer], n=3)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # panel 1: colored by emotion
    ax = axes[0]
    for e in WHEEL:
        m = labels == e
        if m.any():
            ax.scatter(z[m, 0], z[m, 1], s=8, alpha=0.18, color=COLORS[e], linewidths=0)
    cents = centroids(z[:, :2], labels)
    ring = [cents[e] for e in WHEEL if e in cents] + [cents[WHEEL[0]]]
    ring = np.array(ring)
    ax.plot(ring[:, 0], ring[:, 1], "-", color="0.4", lw=1.2, zorder=3)
    for e, c in cents.items():
        ax.scatter(*c, s=260, color=COLORS[e], edgecolor="black", linewidths=1.5, zorder=4)
        ax.annotate(e, c, fontsize=11, fontweight="bold", ha="center", va="center",
                    xytext=(0, 0), textcoords="offset points", zorder=5)
    resid, (cx, cy, r) = circle_residual(cents)
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ls="--", color="0.6"))
    ax.set_title(f"{lang.upper()} layer {layer}: Plutchik-8 centroids\n"
                 f"circle residual={resid:.2f}  (0=perfect ring)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    # panel 2: colored by valence
    ax = axes[1]
    sc = ax.scatter(z[:, 0], z[:, 1], c=valence, cmap="coolwarm_r", s=10, alpha=0.5,
                    vmin=-1, vmax=1, linewidths=0)
    for e, c in cents.items():
        ax.annotate(e, c, fontsize=10, fontweight="bold", ha="center", va="center")
    vcorr = [np.corrcoef(z[:, k], valence)[0, 1] for k in range(2)]
    ax.set_title(f"colored by valence   corr(PC1,val)={vcorr[0]:+.2f}  corr(PC2,val)={vcorr[1]:+.2f}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax, label="valence (- neg / + pos)")
    plt.tight_layout()
    p = Path(outdir) / f"fig_{lang}_headline_layer{layer}.png"
    plt.savefig(p, dpi=140); plt.close()
    return p, resid, vcorr


def plot_sweep(rows, outdir, lang):
    L = [r["layer"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(L, [r["silhouette"] for r in rows], "o-", color="tab:blue", label="silhouette (cluster sep.)")
    ax1.set_xlabel("layer"); ax1.set_ylabel("silhouette", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(L, [r["circle_resid"] for r in rows], "s--", color="tab:red", label="circle residual")
    ax2.plot(L, [r["best_valence_corr"] for r in rows], "^:", color="tab:green", label="|valence corr|")
    ax2.set_ylabel("circle residual / valence corr")
    ax1.set_title(f"{lang.upper()} Tier-1 layer sweep")
    ax1.grid(alpha=0.2)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=9)
    plt.tight_layout()
    p = Path(outdir) / f"fig_{lang}_layer_sweep.png"
    plt.savefig(p, dpi=140); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--outdir", default="pipeline/viz_hidden/out/figs")
    ap.add_argument("--lang", default="ro")
    ap.add_argument("--layer", type=int, default=None, help="headline layer (default: middle saved)")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    X, labels, valence, layers = load(args.npz, args.meta)
    print(f"loaded {len(labels)} valid rows | layers {layers} | dim {X[layers[0]].shape[1]}")
    print("class counts:", {e: int((labels == e).sum()) for e in WHEEL})

    rows = sweep(X, labels, valence, layers)
    print("\nlayer  evr1  evr2  top3   silhou  circleRes  valCorr")
    for r in rows:
        print(f"{r['layer']:5d}  {r['evr_pc1']:.2f}  {r['evr_pc2']:.2f}  {r['evr_top3']:.2f}   "
              f"{r['silhouette']:+.3f}   {r['circle_resid']:.3f}     {r['best_valence_corr']:.2f}")

    layer = args.layer if args.layer is not None else layers[len(layers) // 2]
    best = max(rows, key=lambda r: r["silhouette"])
    print(f"\nheadline layer: {layer} | best-silhouette layer: {best['layer']} ({best['silhouette']:+.3f})")

    fp, resid, vcorr = plot_headline(X, labels, valence, layer, args.outdir, args.lang)
    sp = plot_sweep(rows, args.outdir, args.lang)
    json.dump({"layers": rows, "headline_layer": layer, "headline_circle_resid": resid,
               "headline_valence_corr": vcorr, "best_silhouette_layer": best["layer"]},
              open(Path(args.outdir) / f"metrics_{args.lang}.json", "w"), indent=2)
    print(f"\nwrote {fp}\n      {sp}\n      metrics_{args.lang}.json")


if __name__ == "__main__":
    main()
