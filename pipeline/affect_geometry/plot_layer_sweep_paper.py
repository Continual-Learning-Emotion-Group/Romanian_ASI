"""Paper figure: Procrustes disparity across layers for every language and
fit set. One panel per language; solid lines are the leading plane PC1+PC2,
dashed lines the best searched pair at each layer (top-10 components).

Run from repo root:
  python3 -m pipeline.affect_geometry.plot_layer_sweep_paper
Output: figures/russell_layer_sweep.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from pipeline.affect_geometry.paper_style import (  # noqa: E402
    BLUE, RED, GREEN, apply_style,
)
from pipeline.affect_geometry.common import model_paths  # noqa: E402

apply_style()

PACKAGE = Path(__file__).resolve().parent
_, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)

# same order as the paper's fit-set tables
LANGS = ["es", "id", "zh", "en", "fr", "ro", "fa", "hi"]
NAMES = {"en": "English", "es": "Spanish", "zh": "Mandarin",
         "id": "Indonesian", "ro": "Romanian", "fa": "Persian",
         "hi": "Hindi", "fr": "French"}
CODE3 = {"en": "eng", "es": "spa", "zh": "cmn", "id": "ind",
         "ro": "ron", "fa": "pes", "hi": "hin", "fr": "fra"}

FIT_SETS = [
    ("A", BLUE),    # F = A
    ("B", RED),     # F = S \ A
    ("S", GREEN),   # F = S
]


def sweep(lang):
    """Per-layer disparities for the three fit sets: (leading, searched)."""
    metrics = json.loads(
        (RESULTS_DIR / f"metrics_russell_{lang}.json").read_text())["full"]
    all_states = json.loads(
        (RESULTS_DIR / f"all_states_{lang}.json").read_text())
    broader = json.loads(
        (RESULTS_DIR / f"broader_only_{lang}.json").read_text())
    out = {}
    out["A"] = ([e["pc1_pc2_disparity"] for e in metrics["layers"]],
                [e["searched_disparity"] for e in metrics["layers"]])
    for key, data in (("S", all_states), ("B", broader)):
        out[key] = ([e["pc1_pc2_disparity"] for e in data["layers"]],
                    [e["searched_w10_disparity"] for e in data["layers"]])
    return out


def main():
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4), sharex=True, sharey=True)
    for ax, lang in zip(axes.flat, LANGS):
        curves = sweep(lang)
        layers = range(len(curves["A"][0]))
        for key, color in FIT_SETS:
            leading, searched = curves[key]
            ax.plot(layers, leading, color=color, lw=1.1)
            ax.plot(layers, searched, color=color, lw=0.9, ls="--", alpha=0.8)
        ax.set_title(f"{NAMES[lang]} ({CODE3[lang]})", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 36)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="both", which="both", top=False, right=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Layer", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Disparity $D$", fontsize=8)

    handles = [
        Line2D([], [], color=BLUE, lw=1.4,
               label=r"$\mathcal{F}=\mathcal{A}$"),
        Line2D([], [], color=RED, lw=1.4,
               label=r"$\mathcal{F}=\mathcal{S}\setminus\mathcal{A}$"),
        Line2D([], [], color=GREEN, lw=1.4,
               label=r"$\mathcal{F}=\mathcal{S}$"),
        Line2D([], [], color="0.2", lw=1.1, label="leading plane"),
        Line2D([], [], color="0.2", lw=1.1, ls="--", label="searched plane"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"russell_layer_sweep.{ext}",
                    bbox_inches="tight", dpi=200)
    print("wrote", FIGURES_DIR / "russell_layer_sweep.pdf")


if __name__ == "__main__":
    main()
