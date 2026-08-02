"""Figure for the convex-combination test: one row per language.
A) per-lemma convex R2 histogram + null distribution of the mean
B) affine R2 vs convex R2 (distance below diagonal = cost of convexity)
C) weight-implied angle vs observed PC1+PC2 plane angle (outer-radius half)

Run: python -m pipeline.affect_geometry.plot_convexity_russell
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.affect_geometry.common import LANGUAGE_NAMES, discover_archives, model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = discover_archives(HIDDEN_DIR)
NAMES = dict(LANGUAGE_NAMES)


def main():
    results = json.loads((RESULTS_DIR / "convexity_russell.json").read_text())
    langs = [lang for lang in ARCHIVES if lang in results]
    fig, axes = plt.subplots(len(langs), 3, figsize=(16.5, 5 * len(langs)),
                             squeeze=False)
    for row, lang in enumerate(langs):
        s = results[lang]["summary"]
        rows = results[lang]["broader"]
        cvx = np.array([r["convex_r2"] for r in rows])
        aff = np.array([r["affine_r2"] for r in rows])
        radii = np.array([r["plane_radius"] for r in rows])

        ax = axes[row, 0]
        ax.hist(cvx, bins=24, color="#1f6fb4", alpha=0.75)
        ax.axvline(s["mean_convex_r2"], color="#1f6fb4", lw=2,
                   label=f"observed mean {s['mean_convex_r2']:.3f}")
        ax.axvline(s["null_mean_convex_r2"]["q50"], color="0.45", ls="--", lw=1.5,
                   label=f"null mean (median) {s['null_mean_convex_r2']['q50']:.3f}")
        ax.axvline(s["null_mean_convex_r2"]["max"], color="0.45", ls=":", lw=1.2,
                   label=f"null mean (max of 500) {s['null_mean_convex_r2']['max']:.3f}")
        ax.set_xlabel("convex R² per broader lemma")
        ax.set_ylabel("lemmas")
        ax.set_title(f"{NAMES[lang]} — layer {s['layer']}, {s['n_anchor_labels']} "
                     f"anchors, {s['n_broader_lemmas']} broader lemmas\n"
                     f"p = {s['p_mean_convex_r2']:.4f}", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row, 1]
        lim = max(aff.max(), cvx.max()) * 1.08
        ax.plot([0, lim], [0, lim], color="0.8", lw=1)
        ax.scatter(aff, cvx, s=16, color="#d1500a", alpha=0.6)
        ax.set_xlabel("affine R² (anchor subspace, unconstrained)")
        ax.set_ylabel("convex R² (simplex weights)")
        ax.set_title(f"convexity costs little: mean gap {s['mean_convexity_gap']:.3f}\n"
                     f"(hull retains {s['mean_convex_r2']/s['mean_affine_r2']:.0%} "
                     "of subspace fit)", fontsize=11)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

        ax = axes[row, 2]
        sel = radii >= s["angle_radius_cut"]
        implied = np.array([r["implied_angle_deg"] for r in rows])
        observed = np.array([r["plane_angle_deg"] for r in rows])
        sc = ax.scatter(observed[sel], implied[sel], s=22, c=radii[sel],
                        cmap="viridis", alpha=0.8)
        ax.plot([0, 360], [0, 360], color="0.8", lw=1)
        ax.set_xlabel("angle in aligned PC1+PC2 plane (deg)")
        ax.set_ylabel("angle implied by convex weights (deg)")
        ax.set_title(f"weight adjacency (outer-radius half): circ. corr "
                     f"{s['angle_circular_corr_outer_half']:.2f}, "
                     f"median |err| {s['median_abs_angle_error_deg_outer_half']:.0f}°",
                     fontsize=11)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        fig.colorbar(sc, ax=ax, label="plane radius", shrink=0.8)

    for ax in axes.ravel():
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Broader affective states as convex combinations of Russell anchor "
                 "centroids (full 2560-dim standardized space)", fontsize=14, y=0.995)
    fig.tight_layout()
    path = FIGURES_DIR / "russell_convexity.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
