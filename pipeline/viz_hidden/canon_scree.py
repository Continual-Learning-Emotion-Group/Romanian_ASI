"""Variance-explained (scree) plot for the INITIAL 8-centroid run.

Companion to fillin_analyze.py's scree (which measured the intrinsic
dimensionality of the ~89 broader-state centroids and found it high — 16 PCs for
80%). Here we run the SAME analysis on the Step-1 canonical basic-8 emotions, so
the two scree plots are directly comparable: how many principal components does
the variance of the 8 Plutchik centroids spread across?

With only 8 centroids the cloud spans at most 7 dims after centering, so the
plot shows up to 7 PCs. A concentrated spectrum (most variance in PC1-PC2) means
the basic 8 sit near a low-dim (circle-like) shape; a flat spectrum would mean
even the basic 8 are high-dimensional.

Same recipe as fillin: z-score dims -> top-`ncomp` PCA signal subspace -> the 8
per-emotion centroids -> PCA of that 8-point cloud, with a split-half reliability
check so we only trust axes with reproducible between-emotion signal.

Outputs: fig_ro_canon_scree_L<L>.png, metrics_ro_canon_scree_L<L>.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


def load(npz_path, meta_path, layer):
    z = np.load(npz_path)
    valid = z["valid"].astype(bool)
    meta = [json.loads(l) for l in open(meta_path)]
    X = z[f"layer_{layer}"].astype(np.float32)[valid]
    lab = np.array([m["plutchik"] for m in meta])[valid]
    return X, lab


def eff_dim(ev):
    """participation ratio: (sum ev)^2 / sum ev^2 — a soft count of active dims."""
    ev = np.asarray(ev, float)
    return (ev.sum() ** 2) / (np.square(ev).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="pipeline/viz_hidden/out/canon_ro_hidden.npz")
    ap.add_argument("--meta", default="pipeline/viz_hidden/out/canon_ro_hidden.meta.jsonl")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--ncomp", type=int, default=15, help="signal subspace dim (matches fillin)")
    ap.add_argument("--outdir", default="pipeline/viz_hidden/out/figs")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    X, lab = load(args.npz, args.meta, args.layer)
    present = [e for e in WHEEL if (lab == e).any()]
    print(f"layer {args.layer}: {len(X)} rows, {len(present)} emotions "
          f"({dict((e, int((lab == e).sum())) for e in present)})")

    # top-ncomp PCA signal subspace (same as fillin), then the 8 emotion centroids
    Xs = StandardScaler().fit_transform(X)
    P = PCA(args.ncomp, random_state=0).fit_transform(Xs)
    Cen = np.array([P[lab == e].mean(0) for e in present])       # [8, ncomp]

    # PCA of the 8-point centroid cloud (rank <= n_emotions-1 = 7)
    pcaC = PCA(random_state=0).fit(Cen - Cen.mean(0))
    ev, evr = pcaC.explained_variance_, pcaC.explained_variance_ratio_
    cum = np.cumsum(evr)
    rank = int((ev > ev[0] * 1e-9).sum())
    k80 = int(np.argmax(cum >= 0.80) + 1)
    k90 = int(np.argmax(cum >= 0.90) + 1)

    # split-half reliability of each centroid axis (is between-emotion signal real?)
    rng = np.random.RandomState(0)
    A, B = [], []
    for e in present:
        idx = np.where(lab == e)[0]; rng.shuffle(idx); h = len(idx) // 2
        A.append(P[idx[:h]].mean(0)); B.append(P[idx[h:]].mean(0))
    A, B = np.array(A), np.array(B)
    Za = pcaC.transform(A - Cen.mean(0)); Zb = pcaC.transform(B - Cen.mean(0))
    with np.errstate(invalid="ignore"):
        rel = np.array([np.corrcoef(Za[:, k], Zb[:, k])[0, 1] for k in range(rank)])
    reliable = int((rel > 0.5).sum())
    d_signal = eff_dim(ev[:rank][rel > 0.5]) if reliable else 0.0

    print(f"  centroid-cloud rank={rank} | raw eff dim={eff_dim(ev[:rank]):.2f} | "
          f"reliable axes (split-half r>0.5)={reliable} | reliable eff dim={d_signal:.2f}")
    print(f"  {k80} PCs for 80%, {k90} for 90%")
    print(f"  evr: {[round(float(x), 3) for x in evr[:rank]]}")
    print(f"  split-half reliability: {[round(float(x), 2) for x in rel[:rank]]}")

    # ================= scree (same layout as fillin) =================
    n = min(10, rank)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), evr[:n], color="0.6", label="per-PC")
    ax.plot(range(1, n + 1), cum[:n], "o-", color="C3", label="cumulative")
    ax.axhline(0.8, color="0.7", ls="--", lw=.8)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("principal component of the 8 emotion-centroid cloud")
    ax.set_ylabel("variance explained")
    ax.set_title(f"RO canonical basic-8, layer {args.layer}: intrinsic dimensionality\n"
                 f"{len(present)} centroids (rank {rank}) | {k80} PCs for 80%, {k90} for 90% "
                 f"| reliable axes (split-half)={reliable}")
    ax.legend(); ax.grid(alpha=0.2)
    p = f"{args.outdir}/fig_ro_canon_scree_L{args.layer}.png"
    plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()

    metrics = {"layer": args.layer, "n_centroids": len(present), "rank": rank,
               "ncomp_subspace": args.ncomp,
               "raw_effective_dim": round(float(eff_dim(ev[:rank])), 2),
               "reliable_axes_splithalf": reliable,
               "reliable_effective_dim": round(float(d_signal), 2),
               "pcs_for_80pct": k80, "pcs_for_90pct": k90,
               "evr": [round(float(x), 4) for x in evr[:rank]],
               "cum_evr": [round(float(x), 4) for x in cum[:rank]],
               "reliability": [round(float(x), 3) for x in rel[:rank]]}
    mp = f"{args.outdir}/metrics_ro_canon_scree_L{args.layer}.json"
    json.dump(metrics, open(mp, "w"), indent=2)
    print("wrote", p, mp)


if __name__ == "__main__":
    main()
