"""Paper version of the anchor-only Russell PCA figure, in the exact style of
plot_russell_geometry.py, with three changes:
  * single scatter panel (the searched plane if it differs from PC1+PC2,
    which coincide for en/es/zh) stacked above the layer sweep;
  * small black marks on the dashed Russell circle at every circumplex word's
    target angle, with 2-3 highlighted words (open circles) connected by a
    line to their model centroid;
  * shorter titles/labels, no suptitle.

Run from repo root:
  python3 -m pipeline.affect_geometry.plot_anchor_fit_paper [langs...]
  python3 -m pipeline.affect_geometry.plot_anchor_fit_paper --combined en fa
Output: figures/russell_anchor_fit_<lang>.{pdf,png} or
        figures/russell_anchor_fit_<lang1>_<lang2>.{pdf,png}
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
from pipeline.affect_geometry.common import (  # noqa: E402
    LANGUAGE_NAMES, discover_archives, model_paths,
)
from pipeline.affect_geometry.paper_style import (  # noqa: E402
    BLUE, RED, apply_style, inplot_legend,
)

apply_style()
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = discover_archives(HIDDEN_DIR)
NAMES = LANGUAGE_NAMES
HIGHLIGHT_TARGET_ANGLES = (45.0, 165.0, 285.0)  # one word picked near each

# Hand-placed offsets (points) for labels the automatic pass cannot settle:
# coincident centroid pairs and dense clusters. "arrow" adds a thin connector.
MANUAL_LABEL_OFFSETS = {
    "en": {
        "astonished": {"xytext": (2, 16), "ha": "right"},
        "distressed": {"xytext": (6, -12)},
        "sad": {"xytext": (8, -2)},
        "miserable": {"xytext": (-8, -2), "ha": "right"},
        "tense": {"xytext": (8, -6)},
        "depressed": {"xytext": (-8, -4), "ha": "right"},
        "afraid": {"xytext": (-8, 0), "ha": "right"},
        "alarmed": {"xytext": (-14, -22), "ha": "right", "arrow": True},
    },
    "ro": {
        "happy": {"xytext": (-2, -16), "ha": "center"},
        "relaxed": {"xytext": (-8, -6), "ha": "right"},
        "satisfied": {"xytext": (8, -4)},
        "content": {"xytext": (8, -2)},
        "glad": {"xytext": (-8, 4), "ha": "right"},
        "astonished": {"xytext": (0, -16), "ha": "center"},
        "sad": {"xytext": (-8, 0), "ha": "right"},
        "depressed": {"xytext": (-8, 0), "ha": "right"},
        "distressed": {"xytext": (8, -2)},
        "annoyed": {"xytext": (8, -12)},
        "angry": {"xytext": (2, 12), "ha": "center"},
    },
    "fa": {
        "excited": {"xytext": (-8, 2), "ha": "right"},
        "distressed": {"xytext": (8, -12)},
        "delighted": {"xytext": (8, 6)},
        "glad": {"xytext": (8, -10)},
        "pleased": {"xytext": (-8, 0), "ha": "right"},
        "miserable": {"xytext": (-8, -6), "ha": "right"},
        "sad": {"xytext": (8, -2)},
        "frustrated": {"xytext": (-8, -2), "ha": "right"},
        "angry": {"xytext": (0, 12), "ha": "center"},
        "annoyed": {"xytext": (8, -6)},
        "tired": {"xytext": (-8, 0), "ha": "right"},
        "depressed": {"xytext": (-8, 0), "ha": "right"},
        "happy": {"xytext": (0, -14), "ha": "center"},
        "calm": {"xytext": (6, 8)},
        "gloomy": {"xytext": (8, -4)},
    },
}


def resolve_label_overlaps(fig, ax, annotations, dot_positions, obstacles=None,
                           frozen=(), iterations=250, dot_radius_px=9.0,
                           obstacle_radius_px=6.0, max_shift_pt=28.0):
    """Nudge annotation offsets until text boxes clear each other and the dots."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    initial = [np.asarray(a.get_position(), dtype=float) for a in annotations]
    dots_display = ax.transData.transform(dot_positions)
    dot_boxes = [Bbox([[x - dot_radius_px, y - dot_radius_px],
                       [x + dot_radius_px, y + dot_radius_px]])
                 for x, y in dots_display]
    if obstacles is not None and len(obstacles):
        for x, y in ax.transData.transform(obstacles):
            dots_display = np.vstack([dots_display, (x, y)])
            dot_boxes.append(Bbox([[x - obstacle_radius_px, y - obstacle_radius_px],
                                   [x + obstacle_radius_px, y + obstacle_radius_px]]))
    for _ in range(iterations):
        boxes = [a.get_window_extent(renderer).expanded(1.14, 1.35)
                 for a in annotations]
        moved = False
        for i in range(len(annotations)):
            if i in frozen:
                continue
            shift = np.zeros(2)
            overlapping = 0
            for j in range(len(annotations)):
                if i == j or not boxes[i].overlaps(boxes[j]):
                    continue
                overlapping += 1
                delta = np.array([(boxes[i].x0 + boxes[i].x1
                                   - boxes[j].x0 - boxes[j].x1) / 2,
                                  (boxes[i].y0 + boxes[i].y1
                                   - boxes[j].y0 - boxes[j].y1) / 2])
                norm = np.linalg.norm(delta)
                shift += (delta / norm if norm > 1e-9 else np.array([0.0, 1.0])) * 2.0
            for k, dot_box in enumerate(dot_boxes):
                if k == i or not boxes[i].overlaps(dot_box):
                    continue
                overlapping += 1
                delta = np.array([(boxes[i].x0 + boxes[i].x1) / 2 - dots_display[k][0],
                                  (boxes[i].y0 + boxes[i].y1) / 2 - dots_display[k][1]])
                norm = np.linalg.norm(delta)
                shift += (delta / norm if norm > 1e-9 else np.array([0.0, 1.0])) * 2.0
            if overlapping:
                # Opposing pushes can cancel; break the stall sideways.
                if np.linalg.norm(shift) < 0.6:
                    shift += np.array([1.0, -1.0]) * (1.5 if i % 2 else -1.5)
                position = np.asarray(annotations[i].get_position()) + shift
                delta = position - initial[i]
                norm = np.linalg.norm(delta)
                if norm > max_shift_pt:
                    position = initial[i] + delta / norm * max_shift_pt
                if not np.allclose(position, annotations[i].get_position()):
                    annotations[i].set_position(tuple(position))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def draw_language(fig, ax, ax2, lang, anchors, candidate_pcs, title_prefix=""):
    """Draw the scatter (ax) and layer sweep (ax2) for one language."""
    angle_of = anchors["angles_degrees"]
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
    angles = np.asarray([angle_of[l] for l in label_list])
    radians = np.radians(angles)
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
    searched_differs = searched_pair != [0, 1] or searched_layer != pc12_layer
    if searched_differs:
        aligned = aligned_plane(searched_layer, searched_pair)
        disparity = full["best_searched_layer_metrics"]["searched_disparity"]
        title = (f"searched PC{searched_pair[0]+1}+PC{searched_pair[1]+1} | "
                 f"layer {searched_layer} | disparity {disparity:.2f}")
    else:
        aligned = aligned_plane(pc12_layer, [0, 1])
        disparity = full["best_pc1_pc2_layer_metrics"]["pc1_pc2_disparity"]
        title = f"PC1+PC2 | layer {pc12_layer} | disparity {disparity:.2f}"

    highlighted = []
    for wanted in HIGHLIGHT_TARGET_ANGLES:
        deltas = np.abs((angles - wanted + 180.0) % 360.0 - 180.0)
        index = int(np.argmin(deltas))
        if index not in highlighted:
            highlighted.append(index)

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
    resolve_label_overlaps(fig, ax, annotations, np.asarray(dot_positions),
                           obstacles=theory, frozen=frozen)

    ax.set_title(title_prefix + title, fontsize=11)
    ax.set_aspect("equal")
    lim = 2.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    inplot_legend(ax, loc="lower left", fontsize=8)

    rows = full["layers"]
    xs = [r["layer"] for r in rows]
    ax2.plot(xs, [r["pc1_pc2_disparity"] for r in rows],
             "o-", ms=3.5, lw=1.4, color=BLUE, label="PC1+PC2")
    ax2.plot(xs, [r["searched_disparity"] for r in rows],
             "s-", ms=3.5, lw=1.4, color=RED, label="best searched pair")
    null_q50 = full["searched_null_min_disparity_quantiles"]["q50"]
    ax2.axhline(null_q50, color="0.6", ls="--", lw=1.0)
    ax2.annotate("searched null median", (xs[2], null_q50),
                 textcoords="offset points", xytext=(0, 5), fontsize=8,
                 color="0.4")
    for best_layer, color in ((pc12_layer, BLUE),
                              (searched_layer, RED)):
        ax2.axvline(best_layer, color=color, ls=":", lw=1.0, alpha=0.6)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("Procrustes disparity $D$")
    ax2.set_ylim(0, 1.0)
    inplot_legend(ax2, loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    return [label_list[i] for i in highlighted]


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    config = json.loads((PACKAGE / "config.json").read_text())
    candidate_pcs = int(config["analysis"]["candidate_pcs"])
    args = sys.argv[1:]

    if args and args[0] == "--combined":
        langs = args[1:] or ["en", "fa"]
        fig, axes = plt.subplots(
            2, len(langs), figsize=(6.0 * len(langs), 8.6),
            gridspec_kw={"height_ratios": [6.0, 2.5], "hspace": 0.25,
                         "wspace": 0.18})
        for column, lang in enumerate(langs):
            highlighted = draw_language(
                fig, axes[0, column], axes[1, column], lang, anchors,
                candidate_pcs, title_prefix=f"{NAMES[lang]}: ")
            print(lang, "highlighted:", highlighted)
        stem = "russell_anchor_fit_" + "_".join(langs)
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(FIGURES_DIR / f"{stem}.pdf")
        return

    for lang in (args or ["en", "ro", "es", "zh", "fa", "hi", "fr", "id"]):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(6.0, 8.6),
            gridspec_kw={"height_ratios": [6.0, 2.5], "hspace": 0.25})
        highlighted = draw_language(fig, ax, ax2, lang, anchors, candidate_pcs)
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"russell_anchor_fit_{lang}.{suffix}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(FIGURES_DIR / f"russell_anchor_fit_{lang}.pdf",
              "| highlighted:", highlighted)


if __name__ == "__main__":
    main()
