"""All-layer PCA plane search and held-out broader-state geometry analysis."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.common import WHEEL, load_json


def normalized_shape(points: np.ndarray):
    centered = points - points.mean(0)
    norm = np.linalg.norm(centered)
    if norm == 0:
        return None
    return centered / norm


def procrustes_disparity(reference: np.ndarray, model: np.ndarray):
    """SciPy-compatible Procrustes disparity for two-dimensional shapes."""
    reference_shape, model_shape = normalized_shape(reference), normalized_shape(model)
    if reference_shape is None or model_shape is None:
        return float("inf")
    cross = model_shape.T @ reference_shape
    nuclear_norm = np.linalg.svd(cross, compute_uv=False).sum()
    return float(np.clip(1.0 - nuclear_norm * nuclear_norm, 0.0, 1.0))


def best_pc_pair(category_scores: np.ndarray, theory: np.ndarray):
    best = None
    for first, second in itertools.combinations(range(category_scores.shape[1]), 2):
        model = category_scores[:, [first, second]]
        disparity = procrustes_disparity(theory, model)
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
    pc12_disparity = procrustes_disparity(theory, category_scores[:, :2])
    searched_disparity, first, second = best_pc_pair(category_scores, theory)
    total_variance = float(np.var(standardized[broader], axis=0, ddof=1).sum())
    component_variances = np.var(projected[broader], axis=0, ddof=1) / total_variance
    pair_fractions = np.asarray([
        component_variances[left] + component_variances[right]
        for left, right in itertools.combinations(range(n_components), 2)
    ])
    random_expected = 2.0 / standardized.shape[1]

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
        "isotropic_random_2d_expected_fraction": random_expected,
        "pc1_pc2_enrichment_over_isotropic_random": float(
            component_variances[:2].sum() / random_expected),
        "searched_enrichment_over_isotropic_random": float(
            component_variances[[first, second]].sum() / random_expected),
        "candidate_pair_broader_fraction_mean": float(pair_fractions.mean()),
        "candidate_pair_broader_fraction_range": [float(pair_fractions.min()), float(pair_fractions.max())],
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


def permutation_p_from_shapes(model_shapes, theory, observed, permutations, seed):
    rng = np.random.default_rng(seed)
    model_shapes = np.asarray(model_shapes)
    theory_shape = normalized_shape(theory)
    permutations_array = np.asarray([rng.permutation(len(theory)) for _ in range(permutations)])
    null = np.empty(permutations)
    chunk_size = 250
    for start in range(0, permutations, chunk_size):
        stop = min(start + chunk_size, permutations)
        permuted = theory_shape[permutations_array[start:stop]]
        cross = np.einsum("pki,nkj->npij", model_shapes, permuted, optimize=True)
        frobenius_sq = np.square(cross).sum(axis=(2, 3))
        determinant = cross[:, :, 0, 0] * cross[:, :, 1, 1] - cross[:, :, 0, 1] * cross[:, :, 1, 0]
        nuclear_sq = frobenius_sq + 2.0 * np.abs(determinant)
        disparities = np.clip(1.0 - nuclear_sq, 0.0, 1.0)
        null[start:stop] = disparities.min(axis=1)
    return float((1 + np.sum(null <= observed)) / (permutations + 1)), null


def global_pair_search_permutation_p(category_by_layer, theory, observed, candidate_pcs, permutations, seed):
    model_shapes = []
    for scores in category_by_layer:
        for first, second in itertools.combinations(
                range(min(candidate_pcs, scores.shape[1])), 2):
            shape = normalized_shape(scores[:, [first, second]])
            if shape is not None:
                model_shapes.append(shape)
    return permutation_p_from_shapes(model_shapes, theory, observed, permutations, seed)


def global_pc12_permutation_p(category_by_layer, theory, observed, permutations, seed):
    model_shapes = [normalized_shape(scores[:, :2]) for scores in category_by_layer]
    return permutation_p_from_shapes(model_shapes, theory, observed, permutations, seed)


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
    best_pc12_index = min(range(len(rows)), key=lambda index: rows[index]["pc1_pc2_disparity"])
    best_pc12 = rows[best_pc12_index]
    permutations = int(config["analysis"]["permutations"])
    global_p, null = global_pair_search_permutation_p(
        category_by_layer,
        theory,
        best["searched_disparity"],
        candidate_pcs,
        permutations,
        int(config["random_seed"]),
    )
    global_pc12_p, pc12_null = global_pc12_permutation_p(
        category_by_layer,
        theory,
        best_pc12["pc1_pc2_disparity"],
        permutations,
        int(config["random_seed"]),
    )
    result = {
        "language": args.language,
        "selection_rule": "minimum basic-slice Procrustes disparity; PC1+PC2 and searched pairs select layers independently",
        "candidate_pcs": candidate_pcs,
        "permutations": permutations,
        "global_search_corrected_p": global_p,
        "best_pc1_pc2_layer": int(layers[best_pc12_index]),
        "global_pc1_pc2_layer_corrected_p": global_pc12_p,
        "best_pc1_pc2_layer_metrics": best_pc12,
        "best_layer": int(layers[best_index]),
        "best_pc_pair": best["searched_pc_pair"],
        "best_layer_metrics": best,
        "layers": rows,
        "null_min_disparity_quantiles": {
            "q01": float(np.quantile(null, 0.01)),
            "q05": float(np.quantile(null, 0.05)),
            "q50": float(np.quantile(null, 0.50)),
        },
        "pc1_pc2_null_min_disparity_quantiles": {
            "q01": float(np.quantile(pc12_null, 0.01)),
            "q05": float(np.quantile(pc12_null, 0.05)),
            "q50": float(np.quantile(pc12_null, 0.50)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    projection_rows = []
    best_searched_plane = planes_by_layer[best_index]["searched"]
    best_pc12_plane = planes_by_layer[best_pc12_index]["pc1_pc2"]
    for index, lemma in enumerate(lemmas):
        projection_rows.append({
            "language": args.language,
            "lemma": lemma,
            "count": int(counts[index]),
            "basic_emotion": emotions[index] or None,
            "is_basic": bool(emotions[index]),
            "pc1_pc2_x": float(best_pc12_plane[0][index, 0]),
            "pc1_pc2_y": float(best_pc12_plane[0][index, 1]),
            "searched_x": float(best_searched_plane[0][index, 0]),
            "searched_y": float(best_searched_plane[0][index, 1]),
        })
    projection_path = Path(args.projection_output)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    projection_path.write_text(json.dumps({
        "language": args.language,
        "best_pc1_pc2_layer": int(layers[best_pc12_index]),
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
