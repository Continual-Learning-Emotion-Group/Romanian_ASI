"""Generate paper-ready cross-language figures from frozen analysis outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pipeline.affect_geometry.common import WHEEL
from pipeline.affect_geometry.analyze import fit_circle

LANGUAGES = ["ro", "en", "es"]
LANGUAGE_NAMES = {"ro": "Romanian", "en": "English", "es": "Spanish"}
COLORS = {
    "joy": "#E69F00", "trust": "#009E73", "fear": "#7B61A8", "surprise": "#56B4E9",
    "sadness": "#0072B2", "disgust": "#8C6D31", "anger": "#D55E00", "anticipation": "#CC79A7",
}
LABEL_OFFSETS = {
    "joy": (9, 5), "trust": (0, -14), "fear": (-9, 8), "surprise": (0, 10),
    "sadness": (0, -14), "disgust": (-12, -13), "anger": (-12, 10), "anticipation": (10, 8),
}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def projection_figure(metrics, projections, mode, output_path):
    coordinate_prefix = "searched" if mode == "searched" else "pc1_pc2"
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), constrained_layout=True)
    for axis, language in zip(axes, LANGUAGES):
        data = projections[language]
        states = data["states"]
        broader = [state for state in states if not state["is_basic"]]
        x_key, y_key = coordinate_prefix + "_x", coordinate_prefix + "_y"
        axis.scatter([state[x_key] for state in broader], [state[y_key] for state in broader],
                     s=11, color="#737373", alpha=0.38, linewidths=0, rasterized=True)
        anchor_points = []
        for emotion in WHEEL:
            members = [state for state in states if state["basic_emotion"] == emotion]
            x = np.mean([state[x_key] for state in members])
            y = np.mean([state[y_key] for state in members])
            anchor_points.append([x, y])
            axis.scatter(x, y, s=145, color=COLORS[emotion], edgecolor="white", linewidth=1.2, zorder=4)
        circle_center, circle_radius = fit_circle(np.asarray(anchor_points))
        axis.add_patch(plt.Circle(circle_center, circle_radius, fill=False, color="#3D3D3D",
                                  linestyle="--", linewidth=1.0, alpha=0.7, zorder=2))
        for emotion, point in zip(WHEEL, anchor_points):
            dx, dy = LABEL_OFFSETS[emotion]
            axis.annotate(emotion, point, xytext=(dx, dy), textcoords="offset points",
                          ha="center", va="center", fontsize=8, fontweight="bold")
        axis.axhline(0, color="#D0D0D0", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#D0D0D0", linewidth=0.7, zorder=0)
        axis.margins(x=0.12, y=0.12)
        axis.set_aspect("equal", adjustable="datalim")
        metric = metrics[language]
        if mode == "searched":
            selected = metric["best_layer_metrics"]
            layer = metric["best_layer"]
            pair = metric["best_pc_pair"]
            disparity = selected["searched_disparity"]
            fraction = selected["searched_broader_variance_fraction"]
            p_value = metric["global_search_corrected_p"]
        else:
            selected = metric["best_pc1_pc2_layer_metrics"]
            layer = metric["best_pc1_pc2_layer"]
            pair = [0, 1]
            disparity = selected["pc1_pc2_disparity"]
            fraction = selected["pc1_pc2_broader_variance_fraction"]
            p_value = metric["global_pc1_pc2_layer_corrected_p"]
        pair_label = "+".join(f"PC{index + 1}" for index in pair)
        axis.set_title(f"{LANGUAGE_NAMES[language]}  |  layer {layer}  |  {pair_label}")
        axis.set_xlabel(f"Valence-aligned direction\nfit={disparity:.3f}; p={p_value:.3f}; broader var.={fraction:.1%}")
        axis.set_ylabel("Arousal-aligned direction" if language == "ro" else "")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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

    figure_path = output_dir / "figure_searched_circumplex_projection"
    projection_figure(metrics, projections, "searched", figure_path)
    pc12_path = output_dir / "figure_pc1_pc2_projection"
    projection_figure(metrics, projections, "pc1_pc2", pc12_path)

    fig, axes = plt.subplots(3, 3, figsize=(14.4, 9.2), sharex="col")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.075, top=0.835,
                        wspace=0.16, hspace=0.27)
    fig.suptitle("Layer sweep: axes learned and selected on the basic-emotion slice",
                 y=0.975, fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.943,
        "Top: fit to fixed eight-emotion theory. Middle and bottom: evaluation on non-basic "
        "affective-state lemma centroids.",
        ha="center", va="center", fontsize=9,
    )
    fixed_color = "#666666"
    searched_color = "#0072B2"
    selected_color = "#D55E00"
    legend_handles = [
        Line2D([0], [0], color=fixed_color, marker="o", markersize=3, linewidth=1.4,
               label="Fixed PC1+PC2"),
        Line2D([0], [0], color=searched_color, marker="o", markersize=3, linewidth=1.8,
               label="Best PC pair at each layer (selected on basic slice)"),
        Line2D([0], [0], color=fixed_color, linestyle=":", linewidth=1.3,
               label="Selected PC1+PC2 layer"),
        Line2D([0], [0], color=selected_color, linestyle="--", linewidth=1.3,
               label="Selected searched-pair layer"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.912),
               ncol=4, frameon=False, handlelength=2.8, columnspacing=1.8)

    for column, language in enumerate(LANGUAGES):
        rows = metrics[language]["layers"]
        layers = [row["layer"] for row in rows]
        pc12_layer = metrics[language]["best_pc1_pc2_layer"]
        searched_layer = metrics[language]["best_layer"]

        def mark_selected_layers(axis):
            axis.axvline(pc12_layer, color=fixed_color, linestyle=":", linewidth=1.3,
                         zorder=0)
            axis.axvline(searched_layer, color=selected_color, linestyle="--", linewidth=1.3,
                         zorder=0)

        top = axes[0, column]
        top.plot(layers, [row["pc1_pc2_disparity"] for row in rows], color=fixed_color,
                 marker="o", markersize=3, linewidth=1.4)
        top.plot(layers, [row["searched_disparity"] for row in rows], color=searched_color,
                 marker="o", markersize=3, linewidth=1.8)
        mark_selected_layers(top)
        top.set_title(LANGUAGE_NAMES[language])
        top.set_ylabel("Basic slice\nTheory disparity\n(lower is better)" if column == 0 else "")
        top.set_ylim(bottom=0)
        top.grid(alpha=0.18)

        middle = axes[1, column]
        middle.plot(layers, [row["pc1_pc2_broader_variance_fraction"] for row in rows],
                    color=fixed_color, marker="o", markersize=3, linewidth=1.4)
        middle.plot(layers, [row["searched_broader_variance_fraction"] for row in rows],
                    color=searched_color, marker="o", markersize=3, linewidth=1.8)
        mark_selected_layers(middle)
        middle.set_ylabel("Non-basic states\nVariance captured\n(higher is better)"
                          if column == 0 else "")
        middle.set_ylim(bottom=0)
        middle.grid(alpha=0.18)

        bottom = axes[2, column]
        bottom.plot(layers, [row["pc1_pc2_geometry"]["ring_rmse"] for row in rows],
                    color=fixed_color, marker="o", markersize=3, linewidth=1.4)
        bottom.plot(layers, [row["searched_geometry"]["ring_rmse"] for row in rows],
                    color=searched_color, marker="o", markersize=3, linewidth=1.8)
        mark_selected_layers(bottom)
        bottom.set_ylabel("Non-basic states\nRing deviation (RMSE)\n(lower is better)"
                          if column == 0 else "")
        bottom.set_xlabel("Model layer (0 = embedding output)")
        bottom.set_ylim(bottom=0)
        bottom.grid(alpha=0.18)
    sweep_path = output_dir / "figure_layer_sweep"
    fig.savefig(sweep_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(sweep_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    table_path = output_dir / "table_geometry_summary.csv"
    fieldnames = [
        "language", "eligible_lemmas", "basic_lemmas", "broader_lemmas", "best_layer",
        "best_pc_pair", "searched_disparity", "global_search_corrected_p",
        "best_pc1_pc2_layer", "pc1_pc2_disparity", "global_pc1_pc2_layer_corrected_p",
        "pc1_pc2_broader_variance_fraction", "searched_broader_variance_fraction",
        "searched_enrichment_over_isotropic_random", "broader_effective_dimension",
        "searched_ring_rmse", "searched_angular_coverage",
    ]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for language in LANGUAGES:
            metric = metrics[language]
            best = metric["best_layer_metrics"]
            best_pc12 = metric["best_pc1_pc2_layer_metrics"]
            writer.writerow({
                "language": language,
                "eligible_lemmas": best["basic_lemma_count"] + best["broader_lemma_count"],
                "basic_lemmas": best["basic_lemma_count"],
                "broader_lemmas": best["broader_lemma_count"],
                "best_layer": metric["best_layer"],
                "best_pc_pair": "+".join(f"PC{index + 1}" for index in metric["best_pc_pair"]),
                "searched_disparity": best["searched_disparity"],
                "global_search_corrected_p": metric["global_search_corrected_p"],
                "best_pc1_pc2_layer": metric["best_pc1_pc2_layer"],
                "pc1_pc2_disparity": best_pc12["pc1_pc2_disparity"],
                "global_pc1_pc2_layer_corrected_p": metric["global_pc1_pc2_layer_corrected_p"],
                "pc1_pc2_broader_variance_fraction": best_pc12["pc1_pc2_broader_variance_fraction"],
                "searched_broader_variance_fraction": best["searched_broader_variance_fraction"],
                "searched_enrichment_over_isotropic_random": best["searched_enrichment_over_isotropic_random"],
                "broader_effective_dimension": best["broader_effective_dimension"],
                "searched_ring_rmse": best["searched_geometry"]["ring_rmse"],
                "searched_angular_coverage": best["searched_geometry"]["angular_coverage"],
            })
    print(figure_path.with_suffix(".png"))
    print(pc12_path.with_suffix(".png"))
    print(sweep_path.with_suffix(".png"))
    print(table_path)


if __name__ == "__main__":
    main()
