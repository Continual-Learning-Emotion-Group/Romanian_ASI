"""Quantitative circumplex test: how well do the model's emotion centroids match
Russell's theoretical valence-arousal layout, per layer?

For each layer we take the 8 emotion centroids (mean hidden state, standardized),
PCA to 2D, then Procrustes-align to the theoretical (valence, arousal) coords.
Procrustes disparity in [0,1]: 0 = identical shape (perfect circumplex match).
Reported both with all 8 and without the single-lexeme 'trust' vertex.
"""
from __future__ import annotations

import argparse, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes

WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
# Approx NRC-VAD / Russell circumplex coords (valence, arousal), each in [-1,1].
THEORY = {
    "joy":          (0.80, 0.50), "trust":       (0.50, -0.10),
    "anticipation": (0.30, 0.45), "surprise":    (0.00, 0.85),
    "fear":         (-0.65, 0.80), "anger":       (-0.60, 0.70),
    "disgust":      (-0.60, 0.30), "sadness":     (-0.65, -0.40),
}


def centroids_2d(Xs, labels, emos):
    C = np.array([Xs[labels == e].mean(0) for e in emos])
    return PCA(2, random_state=0).fit_transform(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    z = np.load(args.npz); valid = z["valid"].astype(bool)
    meta = [json.loads(l) for l in open(args.meta)]
    labels = np.array([m["plutchik"] for m in meta])[valid]
    layers = [int(x) for x in z["save_layers"]]

    print("layer | procrustes(all8)  procrustes(no-trust) | silhouette-note")
    print("------+------------------------------------------")
    best = (1e9, None, None)
    for li in layers:
        Xs = StandardScaler().fit_transform(z[f"layer_{li}"].astype(np.float32)[valid])
        # all 8
        M = centroids_2d(Xs, labels, WHEEL)
        T = np.array([THEORY[e] for e in WHEEL])
        _, _, d8 = procrustes(T, M)
        # no trust
        emos7 = [e for e in WHEEL if e != "trust"]
        M7 = centroids_2d(Xs, labels, emos7)
        T7 = np.array([THEORY[e] for e in emos7])
        _, _, d7 = procrustes(T7, M7)
        print(f"{li:5d} |   {d8:.3f}            {d7:.3f}")
        if d7 < best[0]:
            best = (d7, li, "no-trust")
    print(f"\nbest circumplex match: layer {best[1]} ({best[2]}) disparity={best[0]:.3f}")
    print("(disparity 0=perfect circumplex, ~1=unrelated; <0.3 is a good match)")


if __name__ == "__main__":
    main()
