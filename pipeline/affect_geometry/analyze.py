"""All-layer PCA plane search and held-out broader-state geometry analysis."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.common import WHEEL, load_json


def best_pc_pair(category_scores: np.ndarray, theory: np.ndarray):
    best = None
    for first, second in itertools.combinations(range(category_scores.shape[1]), 2):
        model = category_scores[:, [first, second]]
        disparity = float(procrustes(theory, model)[2])
        candidate = (disparity, first, second)
        if best is None or candidate < best:
            best = candidate
    return best


def align_to_theory(points: np.ndarray, anchors: np.ndarray, theory: np.ndarray):
    anchor_center = anchors.mean(0)
    model_scale = np.sqrt(np.mean(np.sum((anchors - anchor_center) ** 2, axis=1)))
    theory_center = theory.mean(0)
    theory_scale = np.sqrt(np.mean(np.sum((theory - theory_center) ** 2, axis=1)))
    model_unit = (anchors - anchor_center) / model_scale
    theory_unit = (theory - theory_center) / theory_scale
    rotation, _ = orthogonal_procrustes(model_unit, theory_unit)
    return ((points - anchor_center) / model_scale) @ rotation


def fit_circle(points: np.ndarray):
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
    cx, cy, constant = np.linalg.lstsq(design, x * x + y * y, rcond=None)[0]
    radius = np.sqrt(max(constant + cx * cx + cy * cy, 0.0))
    return np.array([cx, cy]), float(radius)


def geometry(points: np.ndarray, anchor_points: np.ndarray):
    center, anchor_radius = fit_circle(anchor_points)
    relative = points - center
    radii = np.linalg.norm(relative, axis=1) / max(anchor_radius, 1e-12)
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    angular_resultant = float(np.abs(np.exp(1j * angles).mean()))
    return {
        "plane_radius_mean": float(radii.mean()),
        "plane_radius_cv": float(radii.std() / max(radii.mean(), 1e-12)),
        "ring_rmse": float(np.sqrt(np.mean((radii - 1.0) ** 2))),
        "inside_anchor_circle_fraction": float(np.mean(radii <= 1.0)),
        "angular_resultant": angular_resultant,
        "angular_coverage": 1.0 - angular_resultant,
    }


def participation_ratio(points: np.ndarray):
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    eigenvalues = singular * singular / max(len(points) - 1, 1)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def analyze_layer(centroids, emotions, theory, candidate_pcs):
    basic = emotions != ""
    broader = ~basic
    scaler = StandardScaler().fit(centroids[basic])
    standardized = scaler.transform(centroids)
    n_components = min(candidate_pcs, int(basic.sum()) - 1)
    pca = PCA(n_components=n_components, random_state=0).fit(standardized[basic])
    projected = pca.transform(standardized)
    category_scores = np.asarray([
        projected[emotions == emotion].mean(0) for emotion in WHEEL
    ])
    pc12_disparity = float(procrustes(theory, category_scores[:, :2])[2])
    searched_disparity, first, second = best_pc_pair(category_scores, theory)
    total_variance = float(np.var(standardized[broader], axis=0, ddof=1).sum())

    result = {
        "basic_lemma_count": int(basic.sum()),
        "broader_lemma_count": int(broader.sum()),
        "pc1_pc2_basic_variance_fraction": float(pca.explained_variance_ratio_[:2].sum()),
        "pc1_pc2_disparity": pc12_disparity,
        "pc1_pc2_broader_variance_fraction": float(
            np.var(projected[broader, :2], axis=0, ddof=1).sum() / total_variance),
        "searched_pc_pair": [int(first), int(second)],
        "searched_disparity": searched_disparity,
        "searched_broader_variance_fraction": float(
            np.var(projected[broader][:, [first, second]], axis=0, ddof=1).sum() / total_variance),
        "broader_effective_dimension": participation_ratio(standardized[broader]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    planes = {}
    for name, pair in (("pc1_pc2", (0, 1)), ("searched", (first, second))):
        aligned = align_to_theory(projected[:, pair], category_scores[:, pair], theory)
        anchor_aligned = np.asarray([
            aligned[emotions == emotion].mean(0) for emotion in WHEEL
        ])
        result[name + "_geometry"] = geometry(aligned[broader], anchor_aligned)
        planes[name] = (aligned, anchor_aligned)
    return result, planes, category_scores


def global_permutation_p(category_by_layer, theory, observed, candidate_pcs, permutations, seed):
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for permutation in range(permutations):
        permuted = theory[rng.permutation(len(theory))]
        null[permutation] = min(
            best_pc_pair(scores[:, :candidate_pcs], permuted)[0]
            for scores in category_by_layer
        )
    return float((1 + np.sum(null <= observed)) / (permutations + 1)), null


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--language", choices=["ro", "en", "es"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection-output", required=True)
    parser.add_argument("--config", default=str(package / "config.json"))
    parser.add_argument("--anchors", default=str(package / "anchors.json"))
    args = parser.parse_args()

    config, anchor_config = load_json(args.config), load_json(args.anchors)
    theory = np.asarray([anchor_config["theory_coordinates"][emotion] for emotion in WHEEL], float)
    archive = np.load(args.hidden)
    centroids = archive["centroids"].astype(np.float64)
    layers = archive["layers"].astype(int)
    lemmas = archive["lemmas"].astype(str)
    counts = archive["counts"].astype(int)
    emotions = archive["basic_emotions"].astype(str)
    missing = [emotion for emotion in WHEEL if not np.any(emotions == emotion)]
    if missing:
        raise ValueError(f"Missing basic-emotion categories: {missing}")

    candidate_pcs = int(config["analysis"]["candidate_pcs"])
    rows, planes_by_layer, category_by_layer = [], [], []
    for index, layer in enumerate(layers):
        result, planes, category_scores = analyze_layer(
            centroids[index], emotions, theory, candidate_pcs)
        result["layer"] = int(layer)
        rows.append(result)
        planes_by_layer.append(planes)
        category_by_layer.append(category_scores)
    best_index = min(range(len(rows)), key=lambda index: rows[index]["searched_disparity"])
    best = rows[best_index]
    permutations = int(config["analysis"]["permutations"])
    global_p, null = global_permutation_p(
        category_by_layer,
        theory,
        best["searched_disparity"],
        candidate_pcs,
        permutations,
        int(config["random_seed"]),
    )
    result = {
        "language": args.language,
        "selection_rule": "minimum basic-slice Procrustes disparity across all layers and top-PC pairs",
        "candidate_pcs": candidate_pcs,
        "permutations": permutations,
        "global_search_corrected_p": global_p,
        "best_layer": int(layers[best_index]),
        "best_pc_pair": best["searched_pc_pair"],
        "best_layer_metrics": best,
        "layers": rows,
        "null_min_disparity_quantiles": {
            "q01": float(np.quantile(null, 0.01)),
            "q05": float(np.quantile(null, 0.05)),
            "q50": float(np.quantile(null, 0.50)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    projection_rows = []
    best_planes = planes_by_layer[best_index]
    for index, lemma in enumerate(lemmas):
        projection_rows.append({
            "language": args.language,
            "lemma": lemma,
            "count": int(counts[index]),
            "basic_emotion": emotions[index] or None,
            "is_basic": bool(emotions[index]),
            "pc1_pc2_x": float(best_planes["pc1_pc2"][0][index, 0]),
            "pc1_pc2_y": float(best_planes["pc1_pc2"][0][index, 1]),
            "searched_x": float(best_planes["searched"][0][index, 0]),
            "searched_y": float(best_planes["searched"][0][index, 1]),
        })
    projection_path = Path(args.projection_output)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    projection_path.write_text(json.dumps({
        "language": args.language,
        "best_layer": int(layers[best_index]),
        "best_pc_pair": best["searched_pc_pair"],
        "theory": {emotion: anchor_config["theory_coordinates"][emotion] for emotion in WHEEL},
        "states": projection_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "language", "best_layer", "best_pc_pair", "global_search_corrected_p",
        "best_layer_metrics")}, indent=2))


if __name__ == "__main__":
    main()

