"""Appendix grids: every language's best searched plane per fit set, drawn
in the exact style of the main-body figure (plot_allstates_fit_main.py):
origin axis lines instead of spines, dashed reference circle, black Russell
targets, valence-arousal colors for the standard emotions, gray broader
states, fixed crop. One 4x2 figure per fit set with a shared legend.

Run from repo root:
  AFFECT_GEOMETRY_MODEL=qwen3-8b-final \
  python3 -m pipeline.affect_geometry.plot_fit_grids_paper
Output: figures/<model>/russell_{anchor,allstates,broaderonly}_fit_grid.{pdf,png}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pipeline.affect_geometry.analyze import align_to_theory  # noqa: E402
from pipeline.affect_geometry.plot_anchor_fit_paper import (  # noqa: E402
    ARCHIVES,
    NAMES,
)
from pipeline.affect_geometry.plot_allstates_fit_main import (  # noqa: E402
    russell_color,
)
from pipeline.affect_geometry.paper_style import apply_style  # noqa: E402
from pipeline.affect_geometry.common import model_paths  # noqa: E402

apply_style()

PACKAGE = Path(__file__).resolve().parent
_, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
N_COMPONENTS = 20

# same order as the paper's fit-set tables
LANGS = ["es", "id", "zh", "en", "fr", "ro", "fa", "hi"]
CODE3 = {"en": "eng", "es": "spa", "zh": "cmn", "id": "ind",
         "ro": "ron", "fa": "pes", "hi": "hin", "fr": "fra"}
X_LIM, Y_LIM = 1.85, 1.50  # same crop as the main-body figure


def best_plane(lang, fit_set):
    """(layer, pair) of the best searched plane for this fit set."""
    if fit_set == "anchor":
        full = json.loads(
            (RESULTS_DIR / f"metrics_russell_{lang}.json").read_text())["full"]
        return int(full["best_searched_layer"]), list(full["best_searched_pc_pair"])
    stem = "all_states" if fit_set == "all" else "broader_only"
    best = json.loads((RESULTS_DIR / f"{stem}_{lang}.json").read_text())["w10"]
    return int(best["best_layer"]), list(best["best_pair"])


def draw_panel(ax, lang, anchors, fit_set):
    angle_of = anchors["angles_degrees"]
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

    layer, pair = best_plane(lang, fit_set)
    index = int(np.flatnonzero(layers == layer)[0])
    centroids = all_centroids[index]
    if fit_set == "anchor":
        fit_mask = anchor_mask
    elif fit_set == "broader":
        fit_mask = ~anchor_mask
    else:
        fit_mask = np.ones(len(centroids), dtype=bool)
    scaler = StandardScaler().fit(centroids[fit_mask])
    standardized = scaler.transform(centroids)
    n_comp = min(N_COMPONENTS, int(fit_mask.sum()) - 1)
    pca = PCA(n_components=n_comp, random_state=0).fit(standardized[fit_mask])
    points = pca.transform(standardized)[:, pair]
    scores = np.asarray([
        points[labels_arr == label].mean(0) for label in label_list
    ])
    aligned = align_to_theory(points, scores, theory)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.35", lw=0.9, zorder=1)
    ax.axhline(0, color="0.15", lw=0.8, zorder=2)
    ax.axvline(0, color="0.15", lw=0.8, zorder=2)

    broader_pts = aligned[~anchor_mask]
    ax.scatter(broader_pts[:, 0], broader_pts[:, 1], s=4, color="0.66",
               alpha=0.45, linewidths=0, zorder=3)
    for i, label in enumerate(label_list):
        mx, my = aligned[labels_arr == label].mean(0)
        ax.scatter(mx, my, s=13, color=[russell_color(angle_of[label])],
                   edgecolors="0.25", linewidths=0.4, zorder=6)
        ax.scatter(*theory[i], s=4, color="0.1", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-X_LIM, X_LIM)
    ax.set_ylim(-Y_LIM, Y_LIM)
    ax.set_title(f"{NAMES[lang]} ({CODE3[lang]})\n"
                 f"PC{pair[0]+1}+PC{pair[1]+1}, layer {layer}", fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", top=False, right=False,
                   left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    for fit_set, stem in (("anchor", "russell_anchor_fit_grid"),
                          ("all", "russell_allstates_fit_grid"),
                          ("broader", "russell_broaderonly_fit_grid")):
        fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.4))
        for ax, lang in zip(axes.flat, LANGS):
            draw_panel(ax, lang, anchors, fit_set)
        for ax in axes[1]:
            ax.set_xlabel("Valence", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("Arousal", fontsize=7)
        handles = [
            Line2D([], [], marker="o", ls="none", ms=4,
                   markerfacecolor="0.66", markeredgewidth=0, alpha=0.6,
                   label="Broader affective states"),
            Line2D([], [], marker="o", ls="none", ms=5,
                   markerfacecolor=russell_color(45.0),
                   markeredgecolor="0.25", markeredgewidth=0.4,
                   label="Original emotions (colors as in the main figure)"),
            Line2D([], [], marker="o", ls="none", ms=2.5, color="0.1",
                   label="Expected coordinates (Russell)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{stem}.{ext}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
        print("wrote", FIGURES_DIR / f"{stem}.pdf")


if __name__ == "__main__":
    main()
