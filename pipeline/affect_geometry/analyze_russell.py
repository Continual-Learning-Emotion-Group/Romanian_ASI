"""Russell-anchor variant of the geometry analysis.

Differences from analyze.py (v1, frozen):
- Anchor targets are Russell (1980) circumplex angles placed on the unit circle,
  not hand-set Plutchik valence/arousal coordinates.
- The anchor label set is per-language (labels with no attested lemma are absent),
  read from anchors_russell.json at analysis time; the archive's baked-in
  basic_emotions array is ignored, so v1 centroid archives are reusable.
- Every analysis is run at two scopes: "full" (all labels the language covers)
  and "shared" (labels covered by every language in the anchors file), the
  latter making disparities comparable across languages.
- Labels with both noun and adjective lemmas get a construction-consistency
  check (noun-mean vs adjective-mean displacement in the selected plane).

Statistics are unchanged: PCA fit on anchor-lemma centroids only, confirmatory
PC1+PC2 and exploratory searched-pair selection, global permutation tests that
repeat layer (and pair) selection inside the null.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.analyze import (
    align_to_theory,
    best_pc_pair,
    geometry,
    global_pair_search_permutation_p,
    global_pc12_permutation_p,
    participation_ratio,
    procrustes_disparity,
)
from pipeline.affect_geometry.common import load_json


def theory_from_angles(anchor_config: dict, labels: list[str]) -> np.ndarray:
    radians = np.radians([anchor_config["angles_degrees"][label] for label in labels])
    return np.column_stack((np.cos(radians), np.sin(radians)))


def label_array(lemmas: np.ndarray, lemma_to_label: dict[str, str]) -> np.ndarray:
    return np.asarray([lemma_to_label.get(lemma, "") for lemma in lemmas])


def analyze_layer(centroids, labels_arr, label_list, theory, candidate_pcs):
    basic = labels_arr != ""
    broader = ~basic
    scaler = StandardScaler().fit(centroids[basic])
    standardized = scaler.transform(centroids)
    n_components = min(candidate_pcs, int(basic.sum()) - 1)
    pca = PCA(n_components=n_components, random_state=0).fit(standardized[basic])
    projected = pca.transform(standardized)
    category_scores = np.asarray([
        projected[labels_arr == label].mean(0) for label in label_list
    ])
    pc12_disparity = procrustes_disparity(theory, category_scores[:, :2])
    searched_disparity, first, second = best_pc_pair(category_scores, theory)
    total_variance = float(np.var(standardized[broader], axis=0, ddof=1).sum())
    component_variances = np.var(projected[broader], axis=0, ddof=1) / total_variance
    random_expected = 2.0 / standardized.shape[1]

    result = {
        "anchor_lemma_count": int(basic.sum()),
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
        "broader_effective_dimension": participation_ratio(standardized[broader]),
    }
    planes = {}
    for name, pair in (("pc1_pc2", (0, 1)), ("searched", (first, second))):
        aligned = align_to_theory(projected[:, pair], category_scores[:, pair], theory)
        anchor_aligned = np.asarray([
            aligned[labels_arr == label].mean(0) for label in label_list
        ])
        result[name + "_geometry"] = geometry(aligned[broader], anchor_aligned)
        planes[name] = (aligned, anchor_aligned)
    return result, planes, category_scores


def noun_adjective_checks(plane, labels_arr, label_list, lemmas, noun_lemmas, theory):
    """Displacement between noun-lemma and adjective-lemma means per label,
    in units of the anchor configuration's RMS radius."""
    aligned, anchor_aligned = plane
    scale = float(np.sqrt(np.mean(np.sum(
        (anchor_aligned - anchor_aligned.mean(0)) ** 2, axis=1))))
    nouns = set(noun_lemmas)
    checks = {}
    for label in label_list:
        indices = np.flatnonzero(labels_arr == label)
        noun_idx = [i for i in indices if lemmas[i] in nouns]
        adj_idx = [i for i in indices if lemmas[i] not in nouns]
        if not noun_idx or not adj_idx:
            continue
        noun_mean = aligned[noun_idx].mean(0)
        adj_mean = aligned[adj_idx].mean(0)
        checks[label] = {
            "noun_lemmas": [str(lemmas[i]) for i in noun_idx],
            "adjective_lemmas": [str(lemmas[i]) for i in adj_idx],
            "displacement_over_anchor_rms_radius": float(
                np.linalg.norm(noun_mean - adj_mean) / max(scale, 1e-12)),
        }
    return checks


def run_scope(scope_name, label_list, lemma_to_label, anchor_config, archive_data, config):
    centroids, layers, lemmas = archive_data
    labels_arr = label_array(lemmas, lemma_to_label)
    present = sorted({label for label in labels_arr if label},
                     key=lambda l: anchor_config["angles_degrees"][l])
    if present != sorted(label_list, key=lambda l: anchor_config["angles_degrees"][l]):
        missing = set(label_list) - set(present)
        raise ValueError(f"{scope_name}: labels without any archived lemma: {sorted(missing)}")
    theory = theory_from_angles(anchor_config, label_list)
    candidate_pcs = int(config["analysis"]["candidate_pcs"])
    permutations = int(config["analysis"]["permutations"])
    seed = int(config["random_seed"])

    rows, planes_by_layer, category_by_layer = [], [], []
    for index, layer in enumerate(layers):
        result, planes, category_scores = analyze_layer(
            centroids[index], labels_arr, label_list, theory, candidate_pcs)
        result["layer"] = int(layer)
        rows.append(result)
        planes_by_layer.append(planes)
        category_by_layer.append(category_scores)

    best_index = min(range(len(rows)), key=lambda i: rows[i]["searched_disparity"])
    best_pc12_index = min(range(len(rows)), key=lambda i: rows[i]["pc1_pc2_disparity"])
    best, best_pc12 = rows[best_index], rows[best_pc12_index]
    global_p, null = global_pair_search_permutation_p(
        category_by_layer, theory, best["searched_disparity"],
        candidate_pcs, permutations, seed)
    global_pc12_p, pc12_null = global_pc12_permutation_p(
        category_by_layer, theory, best_pc12["pc1_pc2_disparity"], permutations, seed)

    summary = {
        "labels": label_list,
        "label_count": len(label_list),
        "best_pc1_pc2_layer": int(layers[best_pc12_index]),
        "global_pc1_pc2_layer_corrected_p": global_pc12_p,
        "best_pc1_pc2_layer_metrics": best_pc12,
        "best_searched_layer": int(layers[best_index]),
        "best_searched_pc_pair": best["searched_pc_pair"],
        "global_search_corrected_p": global_p,
        "best_searched_layer_metrics": best,
        "pc1_pc2_null_min_disparity_quantiles": {
            "q01": float(np.quantile(pc12_null, 0.01)),
            "q05": float(np.quantile(pc12_null, 0.05)),
            "q50": float(np.quantile(pc12_null, 0.50)),
        },
        "searched_null_min_disparity_quantiles": {
            "q01": float(np.quantile(null, 0.01)),
            "q05": float(np.quantile(null, 0.05)),
            "q50": float(np.quantile(null, 0.50)),
        },
        "layers": rows,
    }
    selected = {
        "pc1_pc2": (best_pc12_index, planes_by_layer[best_pc12_index]["pc1_pc2"]),
        "searched": (best_index, planes_by_layer[best_index]["searched"]),
    }
    return summary, selected, labels_arr


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection-output", required=True)
    parser.add_argument("--config", default=str(package / "config.json"))
    parser.add_argument("--anchors", default=str(package / "anchors_russell.json"))
    args = parser.parse_args()

    config, anchor_config = load_json(args.config), load_json(args.anchors)
    languages = anchor_config["languages"]
    if args.language not in languages:
        raise SystemExit(f"language {args.language} not in anchors file")

    lemma_to_label = {}
    for label, lemma_list in languages[args.language].items():
        for lemma in lemma_list:
            if lemma in lemma_to_label:
                raise ValueError(f"lemma '{lemma}' mapped to two labels")
            lemma_to_label[lemma] = label
    angle_of = anchor_config["angles_degrees"]
    full_labels = sorted(languages[args.language], key=lambda l: angle_of[l])
    shared = set.intersection(*[set(v) for v in languages.values()])
    shared_labels = sorted(shared, key=lambda l: angle_of[l])

    archive = np.load(args.hidden)
    centroids = archive["centroids"].astype(np.float64)
    layers = archive["layers"].astype(int)
    lemmas = archive["lemmas"].astype(str)
    counts = archive["counts"].astype(int)
    archived = set(lemmas)
    unarchived = [lemma for lemma in lemma_to_label if lemma not in archived]
    archive_data = (centroids, layers, lemmas)

    full_map = {k: v for k, v in lemma_to_label.items() if k in archived}
    shared_map = {k: v for k, v in full_map.items() if v in shared}
    full_present = sorted({v for v in full_map.values()}, key=lambda l: angle_of[l])
    shared_present = sorted({v for v in shared_map.values()}, key=lambda l: angle_of[l])

    full_summary, full_selected, labels_arr = run_scope(
        "full", full_present, full_map, anchor_config, archive_data, config)
    shared_summary, _, _ = run_scope(
        "shared", shared_present, shared_map, anchor_config, archive_data, config)

    noun_lemmas = anchor_config.get("noun_lemmas", {}).get(args.language, [])
    checks = {}
    if noun_lemmas:
        for name, (_, plane) in full_selected.items():
            checks[name] = noun_adjective_checks(
                plane, labels_arr, full_present, lemmas, noun_lemmas,
                theory_from_angles(anchor_config, full_present))

    result = {
        "language": args.language,
        "experiment": "affect_geometry_russell_v2",
        "anchors": str(args.anchors),
        "anchor_lemmas_missing_from_archive": sorted(unarchived),
        "selection_rule": ("minimum anchor-slice Procrustes disparity against Russell-angle "
                           "unit-circle targets; PC1+PC2 and searched pairs select layers "
                           "independently; permutation nulls repeat all selection"),
        "candidate_pcs": int(config["analysis"]["candidate_pcs"]),
        "permutations": int(config["analysis"]["permutations"]),
        "full": full_summary,
        "shared": shared_summary,
        "noun_adjective_checks": checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    pc12_index, pc12_plane = full_selected["pc1_pc2"]
    searched_index, searched_plane = full_selected["searched"]
    projection_rows = []
    for index, lemma in enumerate(lemmas):
        projection_rows.append({
            "language": args.language,
            "lemma": str(lemma),
            "count": int(counts[index]),
            "russell_label": full_map.get(lemma),
            "is_anchor": lemma in full_map,
            "in_shared_subset": full_map.get(lemma) in shared,
            "pc1_pc2_x": float(pc12_plane[0][index, 0]),
            "pc1_pc2_y": float(pc12_plane[0][index, 1]),
            "searched_x": float(searched_plane[0][index, 0]),
            "searched_y": float(searched_plane[0][index, 1]),
        })
    projection_path = Path(args.projection_output)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    projection_path.write_text(json.dumps({
        "language": args.language,
        "experiment": "affect_geometry_russell_v2",
        "best_pc1_pc2_layer": int(layers[pc12_index]),
        "best_searched_layer": int(layers[searched_index]),
        "best_searched_pc_pair": full_summary["best_searched_pc_pair"],
        "labels": full_present,
        "angles_degrees": {label: angle_of[label] for label in full_present},
        "states": projection_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "language": args.language,
        "anchor_lemmas_missing_from_archive": sorted(unarchived),
        "full": {k: full_summary[k] for k in (
            "label_count", "best_pc1_pc2_layer", "global_pc1_pc2_layer_corrected_p",
            "best_searched_layer", "best_searched_pc_pair", "global_search_corrected_p")},
        "shared": {k: shared_summary[k] for k in (
            "label_count", "best_pc1_pc2_layer", "global_pc1_pc2_layer_corrected_p",
            "global_search_corrected_p")},
    }, indent=2))


if __name__ == "__main__":
    main()
