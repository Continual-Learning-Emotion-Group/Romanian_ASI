"""Per-language 3-panel figure for the anchor-only Russell PCA
(metrics_russell_{lang}.json), in the same style as plot_all_states.py:
A) confirmatory PC1+PC2 plane at its best layer
B) best searched pair at its best layer
C) disparity across all 33 layers (PC1+PC2 + searched)
The PCA here is fit on anchor-lemma centroids only (full label scope), exactly
as in analyze_russell.py; broader states are projected into that plane.

Run: python -m pipeline.affect_geometry.plot_russell_geometry [langs...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pipeline.affect_geometry.analyze import align_to_theory  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
from pipeline.affect_geometry.common import LANGUAGE_NAMES, discover_archives, model_paths
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = discover_archives(HIDDEN_DIR)
NAMES = dict(LANGUAGE_NAMES)


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    angle_of = anchors["angles_degrees"]
    config = json.loads((PACKAGE / "config.json").read_text())
    candidate_pcs = int(config["analysis"]["candidate_pcs"])

    for lang in (sys.argv[1:] or ["en", "ro", "es", "zh", "fa", "hi"]):
        metrics = json.loads(
            (RESULTS_DIR / f"metrics_russell_{lang}.json").read_text())
        full = metrics["full"]
        lemma_to_label = {}
        for label, lemma_list in anchors["languages"][lang].items():
            for lemma in lemma_list:
                lemma_to_label[lemma] = label

        archive = np.load(ARCHIVES[lang])
        layers = archive["layers"].astype(int)
        lemmas = archive["lemmas"].astype(str)
        all_centroids = archive["centroids"].astype(np.float64)
        labels_arr = np.asarray([lemma_to_label.get(l, "") for l in lemmas])
        anchor_mask = labels_arr != ""
        label_list = sorted({l for l in labels_arr if l}, key=lambda l: angle_of[l])
        radians = np.radians([angle_of[l] for l in label_list])
        theory = np.column_stack((np.cos(radians), np.sin(radians)))

        def aligned_plane(layer, pair):
            index = int(np.flatnonzero(layers == layer)[0])
            centroids = all_centroids[index]
            scaler = StandardScaler().fit(centroids[anchor_mask])
            standardized = scaler.transform(centroids)
            n_comp = min(candidate_pcs, int(anchor_mask.sum()) - 1)
            pca = PCA(n_components=n_comp, random_state=0).fit(
                standardized[anchor_mask])
            points = pca.transform(standardized)[:, pair]
            scores = np.asarray([
                points[labels_arr == label].mean(0) for label in label_list
            ])
            return align_to_theory(points, scores, theory)

        pc12_layer = full["best_pc1_pc2_layer"]
        searched_layer = full["best_searched_layer"]
        searched_pair = full["best_searched_pc_pair"]
        plane_specs = [
            (aligned_plane(pc12_layer, [0, 1]),
             f"anchor-PCA PC1+PC2 | layer {pc12_layer}\n"
             f"disparity {full['best_pc1_pc2_layer_metrics']['pc1_pc2_disparity']:.3f}, "
             f"corrected p={full['global_pc1_pc2_layer_corrected_p']:.4f}"),
            (aligned_plane(searched_layer, searched_pair),
             f"searched PC{searched_pair[0]+1}+PC{searched_pair[1]+1} "
             f"| layer {searched_layer}\n"
             f"disparity {full['best_searched_layer_metrics']['searched_disparity']:.3f}, "
             f"corrected p={full['global_search_corrected_p']:.4f}"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        theta = np.linspace(0, 2 * np.pi, 200)
        for (aligned, title), ax in zip(plane_specs, axes[:2]):
            ax.plot(np.cos(theta), np.sin(theta), "--", color="0.75", lw=1.0,
                    zorder=0)
            ax.axhline(0, color="0.9", lw=0.8, zorder=0)
            ax.axvline(0, color="0.9", lw=0.8, zorder=0)
            broader_pts = aligned[~anchor_mask]
            ax.scatter(broader_pts[:, 0], broader_pts[:, 1], s=12, color="0.65",
                       alpha=0.5, zorder=1,
                       label=f"broader states (n={int((~anchor_mask).sum())})")
            for label in label_list:
                hue = cm.hsv(angle_of[label] / 360.0)
                mx, my = aligned[labels_arr == label].mean(0)
                ax.scatter(mx, my, s=110, color=[hue], zorder=4)
                ax.annotate(label, (mx, my), textcoords="offset points",
                            xytext=(6, 6), fontsize=9, fontweight="bold")
            ax.set_title(title, fontsize=11)
            ax.set_aspect("equal")
            lim = 2.2
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0].legend(loc="lower left", fontsize=8, frameon=False)

        ax = axes[2]
        rows = full["layers"]
        xs = [r["layer"] for r in rows]
        ax.plot(xs, [r["pc1_pc2_disparity"] for r in rows],
                "o-", ms=3.5, lw=1.4, color="#1f6fb4", label="PC1+PC2")
        ax.plot(xs, [r["searched_disparity"] for r in rows],
                "s-", ms=3.5, lw=1.4, color="#d1500a",
                label="best searched pair (top 10)")
        null_q50 = full["searched_null_min_disparity_quantiles"]["q50"]
        ax.axhline(null_q50, color="0.6", ls="--", lw=1.0)
        ax.annotate("searched null median (min over layers+pairs)",
                    (xs[2], null_q50), textcoords="offset points",
                    xytext=(0, 5), fontsize=8, color="0.4")
        for best_layer, color in ((pc12_layer, "#1f6fb4"),
                                  (searched_layer, "#d1500a")):
            ax.axvline(best_layer, color=color, ls=":", lw=1.0, alpha=0.6)
        ax.set_xlabel("layer")
        ax.set_ylabel("Procrustes disparity vs Russell angles")
        ax.set_ylim(0, 1.0)
        ax.set_title("anchor-shape fit across layers (anchor-only PCA)",
                     fontsize=11)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.suptitle(f"{NAMES[lang]} — anchor-only Russell PCA geometry "
                     f"({full['label_count']} labels; PCA fit on anchor "
                     "centroids only; filled = anchor label centroid, "
                     "hue = Russell angle)", fontsize=13, y=1.0)
        fig.tight_layout()
        path = FIGURES_DIR / f"russell_anchor_geometry_{lang}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
