"""Broader-only PCA experiment: fit StandardScaler + PCA on the centroids of
broader states ONLY (anchors fully excluded from the fit), then project the
anchors into that basis and search PC pairs for the best Procrustes fit to the
Russell anchor-label shape. Unlike analyze_all_states.py, the anchors are
strictly out-of-sample here: they influence neither the standardization nor
the axes, so a circumplex found in this basis is expressible purely in the
broader states' variance directions.

Search widths, permutation correction, and outputs mirror
analyze_all_states.py. Plane shares: anchors are held-out, broader in-sample.

Run: python -m pipeline.affect_geometry.analyze_broader_only [langs...]
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.analyze import (
    global_pair_search_permutation_p,
    normalized_shape,
    permutation_p_from_shapes,
    procrustes_disparity,
)

from pipeline.affect_geometry.common import discover_archives, model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = discover_archives(HIDDEN_DIR)
WIDTHS = (10, 20)
N_COMPONENTS = 20


def pair_search_within(scores, theory, width):
    best = None
    for first, second in itertools.combinations(range(min(width, scores.shape[1])), 2):
        disparity = procrustes_disparity(theory, scores[:, [first, second]])
        candidate = (disparity, first, second)
        if best is None or candidate < best:
            best = candidate
    return best


def run_language(lang, config, anchors):
    seed = int(config["random_seed"])
    permutations = int(config["analysis"]["permutations"])
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

    per_layer = []
    category_by_layer = []
    projections_by_layer = []
    norms_by_layer = []
    for index, layer in enumerate(layers):
        centroids = all_centroids[index]
        scaler = StandardScaler().fit(centroids[~anchor_mask])
        standardized = scaler.transform(centroids)
        n_comp = min(N_COMPONENTS, int((~anchor_mask).sum()) - 1)
        pca = PCA(n_components=n_comp, random_state=0).fit(
            standardized[~anchor_mask])
        projected = pca.transform(standardized)
        scores = np.asarray([
            projected[labels_arr == label].mean(0) for label in label_list
        ])
        category_by_layer.append(scores)
        projections_by_layer.append(projected)
        norms_by_layer.append(np.square(standardized).sum(axis=1))
        row = {"layer": int(layer),
               "pc1_pc2_disparity": procrustes_disparity(theory, scores[:, :2])}
        for width in WIDTHS:
            disparity, first, second = pair_search_within(scores, theory, width)
            row[f"searched_w{width}_pair"] = [int(first), int(second)]
            row[f"searched_w{width}_disparity"] = disparity
        per_layer.append(row)

    summary = {"language": lang, "n_labels": len(label_list),
               "n_lemmas": int(len(lemmas)),
               "n_anchor_lemmas": int(anchor_mask.sum()),
               "n_broader_lemmas": int((~anchor_mask).sum()),
               "layers": per_layer}

    for width in WIDTHS:
        best_index = min(range(len(per_layer)),
                         key=lambda i: per_layer[i][f"searched_w{width}_disparity"])
        best = per_layer[best_index]
        p, null = global_pair_search_permutation_p(
            category_by_layer, theory, best[f"searched_w{width}_disparity"],
            width, permutations, seed)

        pair = best[f"searched_w{width}_pair"]
        _, plain_null = permutation_p_from_shapes(
            [normalized_shape(category_by_layer[best_index][:, pair])],
            theory, best[f"searched_w{width}_disparity"], permutations, seed)
        projected = projections_by_layer[best_index]
        norms = norms_by_layer[best_index]
        pair_sq = np.square(projected[:, pair]).sum(axis=1)
        share = pair_sq / np.maximum(norms, 1e-12)
        summary[f"w{width}"] = {
            "best_layer": int(best["layer"]),
            "best_pair": pair,
            "best_disparity": best[f"searched_w{width}_disparity"],
            "corrected_p": p,
            "null_min_disparity_q50": float(np.quantile(null, 0.50)),
            "null_min_disparity_q05": float(np.quantile(null, 0.05)),
            "null_mean": float(np.mean(null)),
            "null_sd": float(np.std(null)),
            "pre": float(
                1 - best[f"searched_w{width}_disparity"] / np.mean(null)),
            "plain_null_mean": float(np.mean(plain_null)),
            "pre_plain": float(
                1 - best[f"searched_w{width}_disparity"] / np.mean(plain_null)),
            "anchor_mean_plane_share": float(share[anchor_mask].mean()),
            "anchor_median_plane_share": float(np.median(share[anchor_mask])),
            "broader_mean_plane_share": float(share[~anchor_mask].mean()),
            "broader_median_plane_share": float(np.median(share[~anchor_mask])),
        }

    best_pc12_index = min(range(len(per_layer)),
                          key=lambda i: per_layer[i]["pc1_pc2_disparity"])
    shapes = [normalized_shape(s[:, :2]) for s in category_by_layer]
    pc12_p, pc12_null = permutation_p_from_shapes(
        shapes, theory, per_layer[best_pc12_index]["pc1_pc2_disparity"],
        permutations, seed)
    summary["pc1_pc2_best_layer"] = int(per_layer[best_pc12_index]["layer"])
    summary["pc1_pc2_best_disparity"] = per_layer[best_pc12_index]["pc1_pc2_disparity"]
    summary["pc1_pc2_corrected_p"] = pc12_p
    summary["pc1_pc2_null_mean"] = float(np.mean(pc12_null))
    summary["pc1_pc2_null_sd"] = float(np.std(pc12_null))
    summary["pc1_pc2_pre"] = float(
        1 - per_layer[best_pc12_index]["pc1_pc2_disparity"] / np.mean(pc12_null))
    _, pc12_plain_null = permutation_p_from_shapes(
        [shapes[best_pc12_index]], theory,
        per_layer[best_pc12_index]["pc1_pc2_disparity"], permutations, seed)
    summary["pc1_pc2_plain_null_mean"] = float(np.mean(pc12_plain_null))
    summary["pc1_pc2_pre_plain"] = float(
        1 - per_layer[best_pc12_index]["pc1_pc2_disparity"]
        / np.mean(pc12_plain_null))

    out = RESULTS_DIR / f"broader_only_{lang}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k != "layers"}
    print(f"=== {lang} ===")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    config = json.loads((PACKAGE / "config.json").read_text())
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    for lang in (sys.argv[1:] or ["en", "ro", "es", "zh", "fa", "hi"]):
        run_language(lang, config, anchors)
