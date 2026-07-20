"""Generate paper-ready cross-language figures from frozen analysis outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.affect_geometry.common import WHEEL

LANGUAGES = ["ro", "en", "es"]
LANGUAGE_NAMES = {"ro": "Romanian", "en": "English", "es": "Spanish"}
COLORS = {
    "joy": "#E69F00", "trust": "#009E73", "fear": "#7B61A8", "surprise": "#56B4E9",
    "sadness": "#0072B2", "disgust": "#8C6D31", "anger": "#D55E00", "anticipation": "#CC79A7",
}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    results_dir, output_dir = Path(args.results_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {language: load(results_dir / f"metrics_{language}.json") for language in LANGUAGES}
    projections = {language: load(results_dir / f"projections_{language}.json") for language in LANGUAGES}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 120,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), constrained_layout=True)
    for axis, language in zip(axes, LANGUAGES):
        data = projections[language]
        states = data["states"]
        broader = [state for state in states if not state["is_basic"]]
        axis.scatter([state["searched_x"] for state in broader],
                     [state["searched_y"] for state in broader],
                     s=11, color="#737373", alpha=0.38, linewidths=0, rasterized=True)
        for emotion in WHEEL:
            members = [state for state in states if state["basic_emotion"] == emotion]
            x = np.mean([state["searched_x"] for state in members])
            y = np.mean([state["searched_y"] for state in members])
            axis.scatter(x, y, s=145, color=COLORS[emotion], edgecolor="white", linewidth=1.2, zorder=4)
            axis.annotate(emotion, (x, y), xytext=(0, 8), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8, fontweight="bold")
        axis.axhline(0, color="#D0D0D0", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#D0D0D0", linewidth=0.7, zorder=0)
        axis.set_aspect("equal", adjustable="datalim")
        metric = metrics[language]
        fraction = metric["best_layer_metrics"]["searched_broader_variance_fraction"]
        disparity = metric["best_layer_metrics"]["searched_disparity"]
        first, second = metric["best_pc_pair"]
        axis.set_title(f"{LANGUAGE_NAMES[language]}  |  layer {metric['best_layer']}  |  PC{first + 1}+PC{second + 1}")
        axis.set_xlabel(f"Valence-aligned direction\nfit={disparity:.3f}; broader variance={fraction:.1%}")
        axis.set_ylabel("Arousal-aligned direction" if language == "ro" else "")
    figure_path = output_dir / "figure_circumplex_projection"
    fig.savefig(figure_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 6.8), sharex="col", constrained_layout=True)
    for column, language in enumerate(LANGUAGES):
        rows = metrics[language]["layers"]
        layers = [row["layer"] for row in rows]
        top = axes[0, column]
        top.plot(layers, [row["pc1_pc2_disparity"] for row in rows], color="#777777",
                 marker="o", markersize=3, linewidth=1.4, label="PC1+PC2")
        top.plot(layers, [row["searched_disparity"] for row in rows], color="#0072B2",
                 marker="o", markersize=3, linewidth=1.8, label="best pair")
        top.axvline(metrics[language]["best_layer"], color="#D55E00", linestyle="--", linewidth=1)
        top.set_title(LANGUAGE_NAMES[language])
        top.set_ylabel("Circumplex disparity" if column == 0 else "")
        top.set_ylim(bottom=0)
        top.grid(alpha=0.18)
        if column == 2:
            top.legend(frameon=False)
        bottom = axes[1, column]
        bottom.plot(layers, [row["pc1_pc2_broader_variance_fraction"] for row in rows],
                    color="#777777", marker="o", markersize=3, linewidth=1.4)
        bottom.plot(layers, [row["searched_broader_variance_fraction"] for row in rows],
                    color="#009E73", marker="o", markersize=3, linewidth=1.8)
        bottom.axvline(metrics[language]["best_layer"], color="#D55E00", linestyle="--", linewidth=1)
        bottom.set_ylabel("Broader variance captured" if column == 0 else "")
        bottom.set_xlabel("Model layer")
        bottom.set_ylim(bottom=0)
        bottom.grid(alpha=0.18)
    sweep_path = output_dir / "figure_layer_sweep"
    fig.savefig(sweep_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(sweep_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(figure_path.with_suffix(".png"))
    print(sweep_path.with_suffix(".png"))


if __name__ == "__main__":
    main()

