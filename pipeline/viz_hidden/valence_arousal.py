"""Is the circumplex present but not in PC1-PC2? Find which principal components
best track valence and arousal, then view the emotions in that (valence, arousal)
plane. Arousal is used only to *select* an existing PC axis (per-emotion canonical
arousal), not to construct one — so this is a fair 'which dims hold the circle' test.
"""
from __future__ import annotations

import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes

WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
COLORS = {e: hsv_to_rgb((i / len(WHEEL), 0.65, 0.9)) for i, e in enumerate(WHEEL)}
VAL = {"joy": .8, "trust": .5, "anticipation": .3, "surprise": 0., "fear": -.65, "anger": -.6, "disgust": -.6, "sadness": -.65}
ARO = {"joy": .5, "trust": -.1, "anticipation": .45, "surprise": .85, "fear": .8, "anger": .7, "disgust": .3, "sadness": -.4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--ncomp", type=int, default=10)
    ap.add_argument("--outdir", default="pipeline/viz_hidden/out/figs"); ap.add_argument("--lang", default="ro_canon")
    args = ap.parse_args()

    z = np.load(args.npz); valid = z["valid"].astype(bool)
    meta = [json.loads(l) for l in open(args.meta)]
    lab = np.array([m["plutchik"] for m in meta])[valid]
    Xs = StandardScaler().fit_transform(z[f"layer_{args.layer}"].astype(np.float32)[valid])
    P = PCA(args.ncomp, random_state=0).fit_transform(Xs)
    val = np.array([VAL[e] for e in lab]); aro = np.array([ARO[e] for e in lab])

    vc = [abs(np.corrcoef(P[:, k], val)[0, 1]) for k in range(args.ncomp)]
    ac = [abs(np.corrcoef(P[:, k], aro)[0, 1]) for k in range(args.ncomp)]
    vpc = int(np.argmax(vc)); apc = int(np.argmax([c if k != vpc else -1 for k, c in enumerate(ac)]))
    print(f"layer {args.layer}: per-PC |corr| with valence: {[round(c,2) for c in vc]}")
    print(f"               per-PC |corr| with arousal: {[round(c,2) for c in ac]}")
    print(f"  -> valence axis = PC{vpc} (|r|={vc[vpc]:.2f}) ; arousal axis = PC{apc} (|r|={ac[apc]:.2f})")

    # orient so valence increases rightward, arousal upward
    sv = np.sign(np.corrcoef(P[:, vpc], val)[0, 1]); sa = np.sign(np.corrcoef(P[:, apc], aro)[0, 1])
    xv, ya = P[:, vpc] * sv, P[:, apc] * sa
    cents = {e: (xv[lab == e].mean(), ya[lab == e].mean()) for e in WHEEL}
    M = np.array([cents[e] for e in WHEEL]); T = np.array([[VAL[e], ARO[e]] for e in WHEEL])
    _, _, disp = procrustes(T, M)

    fig, ax = plt.subplots(figsize=(9, 8))
    for e in WHEEL:
        m = lab == e
        ax.scatter(xv[m], ya[m], s=9, alpha=0.2, color=COLORS[e], linewidths=0)
    for e in WHEEL:
        ax.scatter(*cents[e], s=300, color=COLORS[e], edgecolor="k", linewidths=1.6, zorder=5)
        ax.annotate(e, cents[e], fontsize=12, fontweight="bold", ha="center", va="center", zorder=6)
    ax.axhline(0, color="0.7", lw=.8); ax.axvline(0, color="0.7", lw=.8)
    ax.set_xlabel(f"valence axis (PC{vpc}, |r|={vc[vpc]:.2f})")
    ax.set_ylabel(f"arousal axis (PC{apc}, |r|={ac[apc]:.2f})")
    ax.set_title(f"{args.lang.upper()} layer {args.layer}: emotions in valence×arousal PC plane\n"
                 f"circumplex Procrustes disparity={disp:.3f}")
    ax.grid(alpha=0.2)
    p = f"{args.outdir}/fig_{args.lang}_valence_arousal_L{args.layer}.png"
    plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()
    print("wrote", p, "| disparity", round(disp, 3))


if __name__ == "__main__":
    main()
