import numpy as np

from pipeline.affect_geometry.analyze import (
    analyze_layer,
    align_to_theory,
    best_pc_pair,
    geometry,
    global_permutation_p,
    procrustes_disparity,
)
from pipeline.affect_geometry.common import WHEEL
from pipeline.affect_geometry.common import morphology_map, reconstruct, target_map


def test_reconstructs_all_masive_slots_and_offsets():
    targets = target_map("<extra_id_0> happy <extra_id_1> calm")
    text, slots = reconstruct("I feel <extra_id_0> and <extra_id_1>.", targets)
    assert text == "I feel happy and calm."
    assert [text[start:end] for _, _, start, end in slots] == ["happy", "calm"]


def test_morphology_only_merges_observed_gender_pairs():
    forms = {"contento", "contenta", "triste", "casa"}
    mapping = morphology_map(forms, "es", {})
    assert mapping["contenta"] == "contento"
    assert mapping["triste"] == "triste"
    assert mapping["casa"] == "casa"


def test_pc_search_recovers_theory_bearing_pair():
    theory = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    scores = np.column_stack((np.arange(4), theory[:, 0], np.ones(4), theory[:, 1]))
    disparity, first, second = best_pc_pair(scores, theory)
    assert (first, second) == (1, 3)
    assert disparity < 1e-12


def test_alignment_and_geometry_are_similarity_invariant():
    theory = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    rotation = np.array([[0, -1], [1, 0]], dtype=float)
    transformed = 4 * theory @ rotation + np.array([7, -3])
    aligned = align_to_theory(transformed, transformed, theory)
    assert np.allclose(aligned, theory, atol=1e-7)
    stats = geometry(aligned, aligned)
    assert stats["ring_rmse"] < 1e-7
    assert stats["angular_coverage"] > 0.999


def test_procrustes_disparity_is_similarity_invariant():
    reference = np.array([[1, 2], [-2, 1], [0, -1], [2, 0]], dtype=float)
    rotation = np.array([[0, -1], [1, 0]], dtype=float)
    transformed = 3.5 * reference @ rotation + np.array([4, 9])
    assert procrustes_disparity(reference, transformed) < 1e-12


def test_layer_analysis_and_search_corrected_permutation_are_finite():
    rng = np.random.default_rng(9)
    theory = np.column_stack((np.cos(np.arange(8) * np.pi / 4),
                              np.sin(np.arange(8) * np.pi / 4)))
    basis = rng.normal(size=(2, 24))
    basic = np.vstack([theory[index] @ basis + rng.normal(scale=0.05, size=(2, 24))
                       for index in range(8)])
    broader = rng.normal(size=(20, 24))
    centroids = np.vstack((basic, broader))
    emotions = np.array([emotion for emotion in WHEEL for _ in range(2)] + [""] * 20)
    result, _, category_scores = analyze_layer(centroids, emotions, theory, candidate_pcs=10)
    assert result["searched_disparity"] <= result["pc1_pc2_disparity"]
    assert result["searched_disparity"] < 0.05
    assert 0 <= result["searched_broader_variance_fraction"] <= 1
    p_value, null = global_permutation_p(
        [category_scores], theory, result["searched_disparity"], 10, 50, 3)
    assert 0 < p_value <= 1
    assert np.all(np.isfinite(null))
