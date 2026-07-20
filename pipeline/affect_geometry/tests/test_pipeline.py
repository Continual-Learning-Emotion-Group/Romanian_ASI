import numpy as np

from pipeline.affect_geometry.analyze import (
    align_to_theory,
    best_pc_pair,
    geometry,
    procrustes_disparity,
)
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
