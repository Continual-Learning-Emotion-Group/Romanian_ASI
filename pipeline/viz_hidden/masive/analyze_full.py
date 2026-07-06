"""Whole-corpus affective-state manifold analysis (unsupervised).

Projects the top-N MASIVE affective states into the model's hidden space, draws
the per-state centroid cloud with the basic-8 emotions overlaid as ring anchors,
and quantifies the shape:
  * intrinsic dim  — participation ratio of PCA eigenvalues (~2 => circumplex-like)
  * angular spread — Rayleigh R of centroid angles (0 => spread all around a ring,
                     1 => bunched on one side => horseshoe/blob)
  * radius profile — ring (radii concentrated at R) vs disk (radii fill 0..R)
  * anchor circle  — do the 8 basic emotions sit on a circle / on the rim?

Usage:
    python -m pipeline.viz_hidden.masive.analyze_full \
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
from sklearn.metrics import silhouette_score

from pipeline.viz_hidden.analyze import COLORS, WHEEL, circle_residual, project


def load(npz_path, meta_path):
    z = np.load(npz_path)
    valid = z["valid"].astype(bool)
    layers = [int(x) for x in z["save_layers"]]
    meta = [json.loads(l) for l in open(meta_path)]
    states = np.array([m["seed_word_normalized"] for m in meta])
    plut = np.array([m["plutchik"] if m["plutchik"] else "" for m in meta])
    X = {li: z[f"layer_{li}"].astype(np.float32)[valid] for li in layers}
    return X, states[valid], plut[valid], layers


def state_centroids(z2, states):
    uniq = np.unique(states)
    return uniq, np.array([z2[states == s].mean(0) for s in uniq])


def participation_ratio(evr):
    e = np.asarray(evr)
    return float((e.sum() ** 2) / (e ** 2).sum())


def shape_stats(cents):
    """Rayleigh R (angular bunching) and radius coefficient of variation."""
    c = cents - cents.mean(0)
    r = np.hypot(c[:, 0], c[:, 1])
    ang = np.arctan2(c[:, 1], c[:, 0])
    rayleigh = float(abs(np.exp(1j * ang).mean()))
    cv = float(r.std() / r.mean()) if r.mean() else 0.0
    return rayleigh, cv, r


def anchor_centroids(z2, plut):
    return {e: z2[plut == e].mean(0) for e in WHEEL if (plut == e).any()}


def sweep(X, states, plut, layers):
    rows = []
    has_anchor = plut != ""
    for li in layers:
        z, evr = project(X[li], n=10)
        _, cents = state_centroids(z[:, :2], states)
        ray, cv, _ = shape_stats(cents)
        anc = anchor_centroids(z[:, :2], plut)
        resid, _ = circle_residual(anc) if len(anc) >= 3 else (float("nan"), None)
        sil = (silhouette_score(z[has_anchor][:, :5], plut[has_anchor])
               if has_anchor.sum() > 10 else float("nan"))
        rows.append({"layer": li, "pr_full": participation_ratio(evr),
                     "evr_top2": float(evr[:2].sum()), "rayleigh": ray,
                     "radius_cv": cv, "anchor_circle_resid": resid,
                     "anchor_silhouette": float(sil)})
    return rows


def plot(X, states, plut, layer, outdir, lang, topk_labels=25):
    z, evr = project(X[layer], n=3)
    uniq, cents = state_centroids(z[:, :2], states)
    ray, cv, radii = shape_stats(cents)
    anc = anchor_centroids(z[:, :2], plut)

    # most frequent states (for text labels)
    order = sorted(uniq, key=lambda s: -(states == s).sum())
    top_states = set(order[:topk_labels])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # ---- panel 1: state-centroid manifold ----
    ax = axes[0]
    ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.04, color="0.5", linewidths=0)
    ax.scatter(cents[:, 0], cents[:, 1], s=14, color="0.15", zorder=3)
    for s, c in zip(uniq, cents):
        if s in top_states:
            ax.annotate(s, c, fontsize=7, color="0.25", ha="center", va="bottom", zorder=4)
    for e, c in anc.items():
        ax.scatter(*c, s=300, color=COLORS[e], edgecolor="black", linewidths=1.6, zorder=6)
        ax.annotate(e, c, fontsize=11, fontweight="bold", ha="center", va="center", zorder=7)
    if len(anc) >= 3:
        resid, (cx, cy, r) = circle_residual(anc)
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ls="-", color="0.45", lw=1.6, zorder=5))
        ax.scatter([cx], [cy], marker="+", s=90, color="0.45", zorder=5)
    else:
        resid = float("nan")
    ax.set_title(f"{lang.upper()} layer {layer}: {len(uniq)} affective-state centroids\n"
                 f"Rayleigh={ray:.2f} (0=ring/disk, 1=horseshoe)  radius CV={cv:.2f}  "
                 f"anchor circle resid={resid:.2f}")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # ---- panel 2: occurrences, basic-8 colored, rest gray ----
    ax = axes[1]
    other = plut == ""
    ax.scatter(z[other, 0], z[other, 1], s=6, alpha=0.10, color="0.6", linewidths=0)
    for e in WHEEL:
        m = plut == e
        if m.any():
            ax.scatter(z[m, 0], z[m, 1], s=10, alpha=0.5, color=COLORS[e], linewidths=0)
    for e, c in anc.items():
        ax.annotate(e, c, fontsize=11, fontweight="bold", ha="center", va="center", zorder=7)
    ax.set_title("occurrences: basic-8 colored, open-vocabulary states gray")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_aspect("equal"); ax.grid(alpha=0.2)

    plt.tight_layout()
    p = Path(outdir) / f"fig_{lang}_full_layer{layer}.png"
    plt.savefig(p, dpi=140); plt.close()
    return p, {"rayleigh": ray, "radius_cv": cv, "anchor_circle_resid": resid}


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
    print(f"loaded {len(states)} occ | {len(np.unique(states))} states | "
          f"{(plut!='').sum()} anchor occ | layers {layers}")

    rows = sweep(X, states, plut, layers)
    print("\nlayer  PR   evr2  rayleigh radCV  ancCircle ancSilhou")
    for r in rows:
        print(f"{r['layer']:5d}  {r['pr_full']:4.1f}  {r['evr_top2']:.2f}   {r['rayleigh']:.2f}"
              f"     {r['radius_cv']:.2f}   {r['anchor_circle_resid']:.2f}     {r['anchor_silhouette']:+.3f}")

    layer = args.layer if args.layer is not None else layers[len(layers) // 2]
    p, stats = plot(X, states, plut, layer, args.outdir, args.lang)
    json.dump({"headline_layer": layer, "layers": rows, **stats},
              open(Path(args.outdir) / f"metrics_{args.lang}_full.json", "w"), indent=2)
    print(f"\nheadline layer {layer} -> {p}")


if __name__ == "__main__":
    main()
