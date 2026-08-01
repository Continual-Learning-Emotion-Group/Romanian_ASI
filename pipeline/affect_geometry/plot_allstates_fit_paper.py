"""Paper figure for the all-states PCA (F = S): same layout and style as
plot_anchor_fit_paper.py, but the scaler + PCA are fit on ALL state centroids
and the plotted plane is the best searched pair from all_states_{lang}.json
(w10 search). Broader states now genuinely live in the plane, so axis limits
adapt to the data instead of the fixed +-2.2 of the anchor version.

Run from repo root:
  python3 -m pipeline.affect_geometry.plot_allstates_fit_paper [langs...]
  python3 -m pipeline.affect_geometry.plot_allstates_fit_paper --combined en fa
Output: figures/russell_allstates_fit_<lang>.{pdf,png} or
        figures/russell_allstates_fit_<lang1>_<lang2>.{pdf,png}
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

from pipeline.affect_geometry.common import model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
N_COMPONENTS = 20

MANUAL_LABEL_OFFSETS = {
    "en": {
        "angry": {"xytext": (0, 12), "ha": "center"},
        "alarmed": {"xytext": (8, 0)},
        "afraid": {"xytext": (-8, 6), "ha": "right"},
        "annoyed": {"xytext": (-8, -2), "ha": "right"},
        "distressed": {"xytext": (8, -8)},
        "miserable": {"xytext": (-8, -2), "ha": "right"},
        "sad": {"xytext": (-8, 0), "ha": "right"},
        "depressed": {"xytext": (-8, -4), "ha": "right"},
        "tired": {"xytext": (6, -10)},
        "glad": {"xytext": (2, -13), "ha": "center"},
        "happy": {"xytext": (-8, -6), "ha": "right"},
        "calm": {"xytext": (8, 0)},
        "astonished": {"xytext": (0, 13), "ha": "center"},
        "delighted": {"xytext": (8, -2)},
        "sleepy": {"xytext": (-8, 2), "ha": "right"},
    },
    "fa": {
        "excited": {"xytext": (0, 14), "ha": "center"},
        "astonished": {"xytext": (4, -14), "ha": "center"},
        "distressed": {"xytext": (0, -14), "ha": "center"},
        "afraid": {"xytext": (-8, 0), "ha": "right"},
        "annoyed": {"xytext": (-8, 0), "ha": "right"},
        "frustrated": {"xytext": (-8, -8), "ha": "right"},
        "depressed": {"xytext": (-8, -2), "ha": "right"},
        "gloomy": {"xytext": (-8, -6), "ha": "right"},
        "pleased": {"xytext": (0, 12), "ha": "center"},
        "glad": {"xytext": (8, -2)},
        "delighted": {"xytext": (-8, -4), "ha": "right"},
        "happy": {"xytext": (8, -4)},
        "at ease": {"xytext": (-8, -2), "ha": "right"},
        "satisfied": {"xytext": (8, -4)},
    },
}


def draw_language(fig, ax, ax2, lang, anchors, title_prefix=""):
    angle_of = anchors["angles_degrees"]
    summary = json.loads(
        (RESULTS_DIR / f"all_states_{lang}.json").read_text())
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
    standardized = StandardScaler().fit_transform(centroids)
    n_comp = min(N_COMPONENTS, standardized.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=0).fit(standardized)
    points = pca.transform(standardized)[:, pair]
    scores = np.asarray([
        points[labels_arr == label].mean(0) for label in label_list
    ])
    aligned = align_to_theory(points, scores, theory)
    title = (f"all-states PC{pair[0]+1}+PC{pair[1]+1} | layer {layer} | "
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
    manual = MANUAL_LABEL_OFFSETS.get(lang, {})
    annotations = []
    dot_positions = []
    frozen = []
    for i, label in enumerate(label_list):
        hue = cm.hsv(angle_of[label] / 360.0)
        mx, my = aligned[labels_arr == label].mean(0)
        ax.scatter(mx, my, s=110, color=[hue], zorder=4)
        dot_positions.append((mx, my))
        spec = manual.get(label, {})
        kwargs = dict(textcoords="offset points",
                      xytext=spec.get("xytext", (6, 6)),
                      ha=spec.get("ha", "left"),
                      fontsize=9, fontweight="bold")
        if spec.get("arrow"):
            kwargs["arrowprops"] = dict(arrowstyle="-", color="0.6", lw=0.7,
                                        shrinkA=2, shrinkB=4)
        annotations.append(ax.annotate(label, (mx, my), **kwargs))
        if spec:
            frozen.append(i)

    lim = max(2.2, 1.04 * float(np.abs(aligned).max()))
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    resolve_label_overlaps(fig, ax, annotations, np.asarray(dot_positions),
                           obstacles=theory, frozen=frozen)
    ax.set_title(title_prefix + title, fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    rows = summary["layers"]
    xs = [r["layer"] for r in rows]
    ax2.plot(xs, [r["pc1_pc2_disparity"] for r in rows],
             "o-", ms=3.5, lw=1.4, color="#1f6fb4", label="PC1+PC2")
    ax2.plot(xs, [r["searched_w10_disparity"] for r in rows],
             "s-", ms=3.5, lw=1.4, color="#d1500a", label="best searched pair")
    null_q50 = best["null_min_disparity_q50"]
    ax2.axhline(null_q50, color="0.6", ls="--", lw=1.0)
    ax2.annotate("searched null median", (xs[2], null_q50),
                 textcoords="offset points", xytext=(0, 5), fontsize=8,
                 color="0.4")
    for best_layer, color in ((summary["pc1_pc2_best_layer"], "#1f6fb4"),
                              (layer, "#d1500a")):
        ax2.axvline(best_layer, color=color, ls=":", lw=1.0, alpha=0.6)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("Procrustes disparity $D$")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    return [label_list[i] for i in highlighted], lim


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    args = sys.argv[1:]

    if args and args[0] == "--combined":
        langs = args[1:] or ["en", "fa"]
        fig, axes = plt.subplots(
            2, len(langs), figsize=(6.0 * len(langs), 8.6),
            gridspec_kw={"height_ratios": [6.0, 2.5], "hspace": 0.25,
                         "wspace": 0.18})
        for column, lang in enumerate(langs):
            highlighted, lim = draw_language(
                fig, axes[0, column], axes[1, column], lang, anchors,
                title_prefix=f"{NAMES[lang]} — ")
            print(lang, "highlighted:", highlighted, "| lim:", round(lim, 2))
        stem = "russell_allstates_fit_" + "_".join(langs)
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(FIGURES_DIR / f"{stem}.pdf")
        return

    for lang in (args or ["en", "ro", "es", "zh", "fa", "hi"]):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(6.0, 8.6),
            gridspec_kw={"height_ratios": [6.0, 2.5], "hspace": 0.25})
        highlighted, lim = draw_language(fig, ax, ax2, lang, anchors)
        for suffix in ("pdf", "png"):
            fig.savefig(
                FIGURES_DIR / f"russell_allstates_fit_{lang}.{suffix}",
                dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(FIGURES_DIR / f"russell_allstates_fit_{lang}.pdf",
              "| highlighted:", highlighted, "| lim:", round(lim, 2))


if __name__ == "__main__":
    main()
