"""Paper figure for the broader-only PCA (F = S \\ A): same layout and style
as plot_allstates_fit_paper.py, but the scaler + PCA are fit on the broader
states ONLY (anchors fully held out, mirroring analyze_broader_only.py) and
the plotted plane is the best searched pair from broader_only_{lang}.json
(w10 search).

Run from repo root:
  python3 -m pipeline.affect_geometry.plot_broaderonly_fit_paper [langs...]
Output: figures/<variant>/russell_broaderonly_fit_<lang>.{pdf,png}
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
from pipeline.affect_geometry.plot_anchor_fit_paper import (  # noqa: E402
    ARCHIVES,
    HIGHLIGHT_TARGET_ANGLES,
    NAMES,
    resolve_label_overlaps,
)
from pipeline.affect_geometry.paper_style import (  # noqa: E402
    BLUE, RED, apply_style, inplot_legend,
)

apply_style()

from pipeline.affect_geometry.common import model_paths  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
N_COMPONENTS = 20


def draw_language(fig, ax, ax2, lang, anchors, title_prefix=""):
    angle_of = anchors["angles_degrees"]
    summary = json.loads(
        (RESULTS_DIR / f"broader_only_{lang}.json").read_text())
    best = summary["w10"]
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
    angles = np.asarray([angle_of[l] for l in label_list])
    radians = np.radians(angles)
    theory = np.column_stack((np.cos(radians), np.sin(radians)))

    layer = int(best["best_layer"])
    pair = list(best["best_pair"])
    index = int(np.flatnonzero(layers == layer)[0])
    centroids = all_centroids[index]
    scaler = StandardScaler().fit(centroids[~anchor_mask])
    standardized = scaler.transform(centroids)
    n_comp = min(N_COMPONENTS, int((~anchor_mask).sum()) - 1)
    pca = PCA(n_components=n_comp, random_state=0).fit(
        standardized[~anchor_mask])
    points = pca.transform(standardized)[:, pair]
    scores = np.asarray([
        points[labels_arr == label].mean(0) for label in label_list
    ])
    aligned = align_to_theory(points, scores, theory)
    title = (f"broader-only PC{pair[0]+1}+PC{pair[1]+1} | layer {layer} | "
             f"disparity {best['best_disparity']:.2f}")

    highlighted = []
    for wanted in HIGHLIGHT_TARGET_ANGLES:
        deltas = np.abs((angles - wanted + 180.0) % 360.0 - 180.0)
        i = int(np.argmin(deltas))
        if i not in highlighted:
            highlighted.append(i)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.75", lw=1.0, zorder=0)
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.axvline(0, color="0.9", lw=0.8, zorder=0)
    broader_pts = aligned[~anchor_mask]
    ax.scatter(broader_pts[:, 0], broader_pts[:, 1], s=12, color="0.65",
               alpha=0.5, zorder=1,
               label=f"broader states (n={int((~anchor_mask).sum())})")
    drew_target = drew_highlight = False
    for i, label in enumerate(label_list):
        mx, my = aligned[labels_arr == label].mean(0)
        if i in highlighted:
            ax.plot([theory[i, 0], mx], [theory[i, 1], my], color="0.45",
                    lw=0.9, zorder=2)
            ax.scatter(*theory[i], s=34, facecolors="white",
                       edgecolors="0.2", linewidths=1.0, zorder=3,
                       label=None if drew_highlight
                       else "Russell target (line to its anchor)")
            drew_highlight = True
        else:
            ax.scatter(*theory[i], s=16, color="black", zorder=3,
                       label=None if drew_target else "Russell targets")
            drew_target = True
    annotations = []
    dot_positions = []
    for i, label in enumerate(label_list):
        hue = cm.hsv(angle_of[label] / 360.0)
        mx, my = aligned[labels_arr == label].mean(0)
        ax.scatter(mx, my, s=110, color=[hue], zorder=4)
        dot_positions.append((mx, my))
        annotations.append(ax.annotate(
            label, (mx, my), textcoords="offset points", xytext=(6, 6),
            ha="left", fontsize=9, fontweight="bold"))

    lim = max(2.2, 1.04 * float(np.abs(aligned).max()))
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    resolve_label_overlaps(fig, ax, annotations, np.asarray(dot_positions),
                           obstacles=theory)
    ax.set_title(title_prefix + title, fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    inplot_legend(ax, loc="lower left", fontsize=8)

    rows = summary["layers"]
    xs = [r["layer"] for r in rows]
    ax2.plot(xs, [r["pc1_pc2_disparity"] for r in rows],
             "o-", ms=3.5, lw=1.4, color=BLUE, label="PC1+PC2")
    ax2.plot(xs, [r["searched_w10_disparity"] for r in rows],
             "s-", ms=3.5, lw=1.4, color=RED, label="best searched pair")
    null_q50 = best["null_min_disparity_q50"]
    ax2.axhline(null_q50, color="0.6", ls="--", lw=1.0)
    ax2.annotate("searched null median", (xs[2], null_q50),
                 textcoords="offset points", xytext=(0, 5), fontsize=8,
                 color="0.4")
    for best_layer, color in ((summary["pc1_pc2_best_layer"], BLUE),
                              (layer, RED)):
        ax2.axvline(best_layer, color=color, ls=":", lw=1.0, alpha=0.6)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("Procrustes disparity $D$")
    ax2.set_ylim(0, 1.0)
    inplot_legend(ax2, loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    return [label_list[i] for i in highlighted], lim


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    args = sys.argv[1:]
    for lang in (args or ["en", "ro", "es", "zh", "fa", "hi", "fr", "id"]):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(6.0, 8.6),
            gridspec_kw={"height_ratios": [6.0, 2.5], "hspace": 0.25})
        highlighted, lim = draw_language(fig, ax, ax2, lang, anchors)
        for suffix in ("pdf", "png"):
            fig.savefig(
                FIGURES_DIR / f"russell_broaderonly_fit_{lang}.{suffix}",
                dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(FIGURES_DIR / f"russell_broaderonly_fit_{lang}.pdf",
              "| highlighted:", highlighted, "| lim:", round(lim, 2))


if __name__ == "__main__":
    main()


