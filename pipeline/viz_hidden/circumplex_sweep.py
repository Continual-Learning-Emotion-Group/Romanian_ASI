"""Cross-lingual searched-axis circumplex sweep.

For each language and each saved layer, applies the SAME label-informed axis
search as `valence_arousal.py` (pick the top-`ncomp` PC best correlated with
canonical valence and the one best correlated with arousal, then measure how
close the 8 emotion centroids are to Russell's circumplex via Procrustes
disparity). Overlays all languages on one shared y-axis so each can be compared
at its own best layer.

Usage:
    python -m pipeline.viz_hidden.circumplex_sweep \
        --lang ro:pipeline/viz_hidden/out/canon_ro_hidden \
        --lang en:pipeline/viz_hidden/masive/out/masive_en_hidden \
        --lang es:pipeline/viz_hidden/masive/out/masive_es_hidden \
        --out pipeline/viz_hidden/out/figs/fig_circumplex_disparity_sweep.png
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes

from pipeline.viz_hidden.valence_arousal import WHEEL, VAL, ARO

LANG_COLOR = {"ro": "tab:red", "en": "tab:blue", "es": "tab:green"}


def disparity_at_layer(Xlayer, lab, ncomp):
    """Searched-axis Procrustes disparity for one layer (mirrors valence_arousal.py)."""
    Xs = StandardScaler().fit_transform(Xlayer)
    P = PCA(ncomp, random_state=0).fit_transform(Xs)
    val = np.array([VAL[e] for e in lab]); aro = np.array([ARO[e] for e in lab])
    vc = [abs(np.corrcoef(P[:, k], val)[0, 1]) for k in range(ncomp)]
    ac = [abs(np.corrcoef(P[:, k], aro)[0, 1]) for k in range(ncomp)]
    vpc = int(np.argmax(vc))
    apc = int(np.argmax([c if k != vpc else -1 for k, c in enumerate(ac)]))
    sv = np.sign(np.corrcoef(P[:, vpc], val)[0, 1]); sa = np.sign(np.corrcoef(P[:, apc], aro)[0, 1])
    xv, ya = P[:, vpc] * sv, P[:, apc] * sa
    M = np.array([[xv[lab == e].mean(), ya[lab == e].mean()] for e in WHEEL])
    T = np.array([[VAL[e], ARO[e]] for e in WHEEL])
    _, _, disp = procrustes(T, M)
    return float(disp), vpc, apc, float(vc[vpc]), float(ac[apc])


def sweep(stem, ncomp):
    z = np.load(f"{stem}.npz"); valid = z["valid"].astype(bool)
    layers = [int(x) for x in z["save_layers"]]
    lab = np.array([json.loads(l)["plutchik"] for l in open(f"{stem}.meta.jsonl")])[valid]
    rows = []
    for li in layers:
        X = z[f"layer_{li}"].astype(np.float32)[valid]
        disp, vpc, apc, vr, ar = disparity_at_layer(X, lab, ncomp)
        rows.append({"layer": li, "disparity": disp, "vpc": vpc, "apc": apc,
                     "valence_r": vr, "arousal_r": ar})
    return layers, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", action="append", required=True,
                    help="repeatable, form 'code:stem' where stem is path minus .npz")
    ap.add_argument("--ncomp", type=int, default=10)
    ap.add_argument("--out", default="pipeline/viz_hidden/out/figs/fig_circumplex_disparity_sweep.png")
    args = ap.parse_args()

    results = {}
    for spec in args.lang:
        code, stem = spec.split(":", 1)
        layers, rows = sweep(stem, args.ncomp)
        results[code] = rows
        best = min(rows, key=lambda r: r["disparity"])
        print(f"\n=== {code.upper()} ===  best: layer {best['layer']} disparity={best['disparity']:.3f} "
              f"(valence PC{best['vpc']} |r|={best['valence_r']:.2f}, arousal PC{best['apc']} |r|={best['arousal_r']:.2f})")
        print(f"  {'layer':>5} {'disparity':>10} {'valPC':>6} {'aroPC':>6}")
        for r in rows:
            print(f"  {r['layer']:>5} {r['disparity']:>10.3f} {r['vpc']:>6} {r['apc']:>6}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for code, rows in results.items():
        L = [r["layer"] for r in rows]; D = [r["disparity"] for r in rows]
        col = LANG_COLOR.get(code, None)
        ax.plot(L, D, "o-", color=col, lw=2, label=code.upper())
        bi = int(np.argmin(D))
        ax.scatter([L[bi]], [D[bi]], s=140, facecolor="white", edgecolor=col, zorder=5, lw=2)
        ax.annotate(f"{D[bi]:.2f}", (L[bi], D[bi]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color=col, fontweight="bold")
    ax.set_ylim(0, 1)                    # shared, fixed scale (Procrustes disparity in [0,1])
    ax.axhline(0.3, color="0.6", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], 0.3, " good-fit (<0.3)", va="center", fontsize=8, color="0.4")
    ax.set_xlabel("layer"); ax.set_ylabel("circumplex Procrustes disparity  (0 = perfect ring)")
    ax.set_title("Searched-axis circumplex fit by layer (same method & scale, all languages)")
    ax.grid(alpha=0.2); ax.legend(title="language", loc="upper right")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140); plt.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
