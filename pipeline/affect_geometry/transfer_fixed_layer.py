"""Fixed-layer, fixed-label cross-language transfer — the simple protocol.

Each source language's layer (and hence plane) is frozen ONCE from its own
native fit on its own full anchor set, before any target is seen. Every cell
(A -> B) is then a single Procrustes disparity of B's centroids, projected on
A's frozen plane, against Russell's theory circle — scored on the global
8-language label intersection so all 64 cells share one label set and raw
disparities are directly comparable across the whole table.

Because nothing is searched per target, significance per cell is a plain
PROTEST label-permutation test (Jackson 1995) at that single layer: shuffle
the label -> theory-angle assignment, recompute D, p = fraction of shuffles
at least as good. No multiple-comparison correction is needed.

Diagonal cells are reference values (the frozen layer was chosen on the same
language's data).

Run: python -m pipeline.affect_geometry.transfer_fixed_layer
Output: results/<variant>/transfer_fixed_layer.json
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from pipeline.affect_geometry.analyze import (
    normalized_shape,
    permutation_p_from_shapes,
    procrustes_disparity,
)
from pipeline.affect_geometry.transfer_shared_labels import (
    ARCHIVES, PERMUTATIONS, RESULTS_DIR, SEED, Language, angle_of, theory_of,
)


def project_scores(source, target, index, labels):
    info = source.planes[index]
    raw = target.centroids[index]
    z = (raw - raw.mean(0)) / info["scale"]
    points = z @ info["components"].T
    return np.asarray([points[target.labels_arr == label].mean(0)
                       for label in labels])


def native_best_index(source):
    """Layer index minimizing the source's own disparity on its own full
    anchor set — chosen without ever seeing a target."""
    labels = sorted(source.present_labels, key=lambda l: angle_of[l])
    theory = theory_of(labels)
    best_d, best_i = None, None
    for index in range(len(source.planes)):
        scores = project_scores(source, source, index, labels)
        d = procrustes_disparity(theory, scores)
        if best_d is None or d < best_d:
            best_d, best_i = d, index
    return best_i, best_d


def fixed_cell(source, target, index, shared):
    theory = theory_of(shared)
    scores = project_scores(source, target, index, shared)
    disparity = procrustes_disparity(theory, scores)
    shape = normalized_shape(scores)
    p, null = permutation_p_from_shapes(
        np.asarray([shape]), theory, disparity, PERMUTATIONS, SEED)
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))
    return {
        "source": source.lang, "target": target.lang,
        "native": source.lang == target.lang,
        "n_labels": len(shared),
        "frozen_layer": int(source.layers[index]),
        "source_pair": source.planes[index]["pair"],
        "disparity": disparity,
        "protest_p": p,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_z": (null_mean - disparity) / max(null_sd, 1e-12),
    }


def main():
    languages = {lang: Language(lang) for lang in ARCHIVES}
    shared = sorted(set.intersection(
        *(l.present_labels for l in languages.values())),
        key=lambda l: angle_of[l])
    print(f"global intersection: {len(shared)} labels: {shared}")

    frozen = {}
    for lang, language in languages.items():
        index, native_d = native_best_index(language)
        frozen[lang] = index
        print(f"{lang}: frozen layer L{int(language.layers[index])} "
              f"(native full-anchor D={native_d:.3f})")

    results = {"languages": list(languages), "shared_labels": shared,
               "frozen_layers": {lang: int(languages[lang].layers[i])
                                 for lang, i in frozen.items()},
               "cells": {}}
    for a, b in itertools.product(languages, repeat=2):
        cell = fixed_cell(languages[a], languages[b], frozen[a], shared)
        results["cells"][f"{a}->{b}"] = cell
        print(f"{a}->{b}: D={cell['disparity']:.3f} "
              f"p={cell['protest_p']:.4f} @L{cell['frozen_layer']}"
              + ("  [native]" if cell["native"] else ""), flush=True)

    results["matrix"] = {metric: {
        a: {b: results["cells"][f"{a}->{b}"][metric] for b in languages}
        for a in languages} for metric in ("disparity", "protest_p")}
    out = RESULTS_DIR / "transfer_fixed_layer.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
