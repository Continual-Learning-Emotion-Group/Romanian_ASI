"""Main-body version of the all-states PCA figure (F = S), redesigned per
co-author feedback on the earlier two-language panel:

  * one language (Spanish: the one non-English language whose best
    all-states plane is the confirmatory PC1+PC2), one panel, no layer sweep;
  * named valence/arousal axes drawn as dark origin lines, no spines/ticks
    (Figure/Table Formatting Guidelines), visible reference circle;
  * the vertical range is clipped below the circle so the circle sits at
    the bottom of the frame (the broader-state tail below is cut);
  * small markers; disparity lines + labels only for `astonished` and
    `happy`, placed by hand so no text crosses the axis lines;
  * every Russell target is drawn identically (small black dot) whether or
    not it gets a line;
  * color encodes the word's Russell coordinates: red-green spectrum for
    valence; saturation AND brightness for arousal, for clearer contrast.

Run from repo root:
  AFFECT_GEOMETRY_MODEL=qwen3-8b-final \
  python3 -m pipeline.affect_geometry.plot_allstates_fit_main [langs...]
Output: figures/<model>/russell_allstates_fit_main_<lang>.{pdf,png}
"""
import colorsys
import json
import sys
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
from pipeline.affect_geometry.paper_style import (  # noqa: E402
    apply_style, inplot_legend,
)

apply_style()

from pipeline.affect_geometry.common import model_paths  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
N_COMPONENTS = 20

# The two circumplex words that get a target-to-centroid line and labels.
HIGHLIGHTED = ["astonished", "happy"]

# Hand-placed label offsets (points): (dx, dy) from the marker, keyed by
# (lang, fit_set). Chosen so no text crosses the origin axis lines.
LABEL_OFFSETS = {
    ("es", "all"): {
        ("astonished", "centroid"): {"xytext": (-3, -14), "ha": "center"},
        ("astonished", "target"): {"xytext": (-2, 9), "ha": "center"},
        ("happy", "centroid"): {"xytext": (7, 5)},
        ("happy", "target"): {"xytext": (4, 10)},
    },
    ("es", "broader"): {
        ("astonished", "centroid"): {"xytext": (-3, -14), "ha": "center"},
        ("astonished", "target"): {"xytext": (-2, 9), "ha": "center"},
        ("happy", "centroid"): {"xytext": (7, -2)},
        ("happy", "target"): {"xytext": (-9, -3), "ha": "right"},
    },
}

# Fixed, origin-centered limits: the circle sits in the middle of the frame
# and everything outside is cropped. Color key pinned to the top-right corner.
X_LIM = 1.85
Y_LIM = 1.50
KEY_BOUNDS = (1.25, 0.90, 0.55, 0.55)


def va_color(valence, arousal):
    """Red-green spectrum for valence; saturation and brightness together
    encode arousal (dark & muted = deactivated, bright & vivid = activated).
    Both inputs in [0, 1]."""
    hue = valence * 0.35                     # 0.0 = red -> 0.35 = green
    saturation = 0.50 + 0.50 * arousal
    value = 0.45 + 0.50 * arousal
    return colorsys.hsv_to_rgb(hue, saturation, value)


def russell_color(angle_deg):
    theta = np.radians(angle_deg)
    return va_color((np.cos(theta) + 1.0) / 2.0, (np.sin(theta) + 1.0) / 2.0)


def draw_color_key(ax, bounds):
    """Small 2D color legend (valence x arousal) as an inset, in data coords."""
    from matplotlib.colors import hsv_to_rgb

    iax = ax.inset_axes(bounds, transform=ax.transData, zorder=9)
    n = 96
    v, a = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    img = hsv_to_rgb(np.dstack([v * 0.35, 0.50 + 0.50 * a, 0.45 + 0.50 * a]))
    iax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    iax.set_xticks([])
    iax.set_yticks([])
    iax.tick_params(axis="both", which="both", top=False, right=False,
                    left=False, bottom=False)
    iax.set_xlabel(r"Valence $\rightarrow$", fontsize=6, labelpad=1.5)
    iax.set_ylabel(r"Arousal $\rightarrow$", fontsize=6, labelpad=1.5)
    for spine in iax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.3")
        spine.set_linewidth(0.6)


def draw_language(fig, ax, lang, anchors, fit_set="all"):
    """fit_set='all' fits scaler+PCA on every centroid (F = S);
    fit_set='broader' fits them on broader states only (F = S \\ A), with
    the anchors projected strictly out-of-sample, mirroring
    analyze_broader_only.py."""
    angle_of = anchors["angles_degrees"]
    stem = "all_states" if fit_set == "all" else "broader_only"
    summary = json.loads((RESULTS_DIR / f"{stem}_{lang}.json").read_text())
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
    if fit_set == "broader":
        scaler = StandardScaler().fit(centroids[~anchor_mask])
        standardized = scaler.transform(centroids)
        n_comp = min(N_COMPONENTS, int((~anchor_mask).sum()) - 1)
        pca = PCA(n_components=n_comp, random_state=0).fit(
            standardized[~anchor_mask])
    else:
        standardized = StandardScaler().fit_transform(centroids)
        n_comp = min(N_COMPONENTS, standardized.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=0).fit(standardized)
    points = pca.transform(standardized)[:, pair]
    scores = np.asarray([
        points[labels_arr == label].mean(0) for label in label_list
    ])
    aligned = align_to_theory(points, scores, theory)

    # Reference circle + valence/arousal axis lines through the origin
    # (no spines or ticks; the origin lines are the axes).
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.35", lw=1.3,
            zorder=1)
    ax.axhline(0, color="0.15", lw=1.1, zorder=2)
    ax.axvline(0, color="0.15", lw=1.1, zorder=2)

    broader_pts = aligned[~anchor_mask]
    ax.scatter(broader_pts[:, 0], broader_pts[:, 1], s=7, color="0.66",
               alpha=0.45, linewidths=0, zorder=3)

    offsets = LABEL_OFFSETS.get((lang, fit_set), {})
    for i, label in enumerate(label_list):
        mx, my = aligned[labels_arr == label].mean(0)
        color = russell_color(angle_of[label])
        if label in HIGHLIGHTED:
            ax.plot([theory[i, 0], mx], [theory[i, 1], my], color="0.35",
                    lw=1.0, zorder=4)
            ax.scatter(mx, my, s=42, color=[color], edgecolors="0.1",
                       linewidths=0.7, zorder=7)
            spec = offsets.get((label, "centroid"), {})
            ax.annotate(label, (mx, my), textcoords="offset points",
                        xytext=spec.get("xytext", (7, 3)),
                        ha=spec.get("ha", "left"),
                        fontsize=9, fontweight="bold", zorder=8)
            spec = offsets.get((label, "target"), {})
            ax.annotate(label, tuple(theory[i]), textcoords="offset points",
                        xytext=spec.get("xytext", (7, 3)),
                        ha=spec.get("ha", "left"),
                        fontsize=8, style="italic", color="0.35", zorder=8)
        else:
            ax.scatter(mx, my, s=26, color=[color], edgecolors="0.25",
                       linewidths=0.5, zorder=6)
        # Every Russell target is drawn identically, highlighted or not.
        ax.scatter(*theory[i], s=9, color="0.1", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-X_LIM, X_LIM)
    ax.set_ylim(-Y_LIM, Y_LIM)
    ax.set_title(NAMES[lang], fontsize=11)
    ax.set_xlabel(r"Valence  (unpleasant $\rightarrow$ pleasant)",
                  fontsize=10)
    ax.set_ylabel(r"Arousal  (deactivated $\rightarrow$ activated)",
                  fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", top=False, right=False,
                   left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.5,
               markerfacecolor="0.66", markeredgewidth=0, alpha=0.6,
               label=f"Broader affective states (n={int((~anchor_mask).sum())})"),
        Line2D([], [], marker="o", ls="none", ms=6,
               markerfacecolor=russell_color(45.0), markeredgecolor="0.25",
               markeredgewidth=0.5, label="Original emotions"),
        Line2D([], [], marker="o", ls="none", ms=3.5, color="0.1",
               label="Expected coordinates (Russell)"),
        Line2D([], [], color="0.35", lw=1.0, label="Target-centroid line"),
    ]
    inplot_legend(ax, handles=handles, loc="upper left", fontsize=7.5,
                  handletextpad=0.5, borderpad=0.5, labelspacing=0.45)

    # 2D color key for the standard-emotion colors, top-right corner.
    draw_color_key(ax, KEY_BOUNDS)

    return best, layer, pair


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    args = sys.argv[1:]
    fit_set = "all"
    if "--broader" in args:
        fit_set = "broader"
        args = [a for a in args if a != "--broader"]
    langs = args or ["es"]
    prefix = "allstates" if fit_set == "all" else "broaderonly"
    for lang in langs:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        best, layer, pair = draw_language(fig, ax, lang, anchors,
                                          fit_set=fit_set)
        stem = f"russell_{prefix}_fit_main_{lang}"
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(FIGURES_DIR / f"{stem}.pdf",
              f"| PC{pair[0]+1}+PC{pair[1]+1} layer {layer} "
              f"D {best['best_disparity']:.2f}")


if __name__ == "__main__":
    main()
