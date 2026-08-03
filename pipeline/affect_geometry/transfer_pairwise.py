"""Frozen-plane cross-language transfer scored on PER-PAIR shared labels,
standardized against each cell's own permutation null.

Motivation: the global shared-label tiers (transfer_shared_labels.py) cannot
include Indonesian without collapsing the common label set to 4. Here every
ordered cell (A -> B) is scored on the labels shared by just A and B
(6-25 across the 8 languages), which keeps each cell fair to the pair.
Raw Procrustes disparities are not directly comparable across cells with
different label counts/configurations (fewer points fit more easily), so
each cell also reports its disparity standardized against its own
layer-search-corrected label-permutation null:

    null_z = (mean(null) - observed) / sd(null)

where null is the distribution of min-over-layers disparities under label
permutation (the same null used for the corrected p). null_z answers "how
far below chance, given this cell's exact label geometry and layer search"
and is comparable across all 64 cells; n_labels is reported so thin cells
(Indonesian pairs) stay visible.

Protocol per cell, per layer, unchanged from the frozen tier: A's all-states
PCA, A's circumplex pair chosen on A's own FULL anchor set (selection never
sees B), B recentered by its own mean, scaled with A's per-dimension scales.
Diagonal cells score A on its own plane and full label set (reference values;
selection saw the scored labels).

Run: python -m pipeline.affect_geometry.transfer_pairwise
Output: results/transfer_pairwise.json
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


def transfer_cell(source, target, fixed_labels=None):
    shared = sorted(fixed_labels if fixed_labels is not None
                    else source.present_labels & target.present_labels,
                    key=lambda l: angle_of[l])
    theory = theory_of(shared)
    rows, shapes = [], []
    for index in range(len(source.planes)):
        info = source.planes[index]
        raw = target.centroids[index]
        z = (raw - raw.mean(0)) / info["scale"]
        points = z @ info["components"].T
        scores = np.asarray([
            points[target.labels_arr == label].mean(0) for label in shared
        ])
        rows.append({
            "layer": int(source.layers[index]),
            "source_pair": info["pair"],
            "disparity": procrustes_disparity(theory, scores),
        })
        shapes.append(normalized_shape(scores))
    best_index = min(range(len(rows)), key=lambda i: rows[i]["disparity"])
    best = rows[best_index]
    p, null = permutation_p_from_shapes(
        shapes, theory, best["disparity"], PERMUTATIONS, SEED)
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))
    return {
        "source": source.lang, "target": target.lang,
        "native": source.lang == target.lang,
        "n_labels": len(shared),
        "shared_labels": shared,
        "best_layer": best["layer"],
        "source_pair_at_best": best["source_pair"],
        "disparity": best["disparity"],
        "layer_corrected_p": p,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_min_disparity_q50": float(np.quantile(null, 0.50)),
        "null_z": (null_mean - best["disparity"]) / max(null_sd, 1e-12),
        "layers": rows,
    }


def main():
    languages = {lang: Language(lang) for lang in ARCHIVES}
    results = {"languages": list(languages), "cells": {}}
    for a, b in itertools.product(languages, repeat=2):
        cell = transfer_cell(languages[a], languages[b])
        results["cells"][f"{a}->{b}"] = cell
        print(f"{a}->{b}: n={cell['n_labels']} D={cell['disparity']:.3f} "
              f"z={cell['null_z']:.2f} p={cell['layer_corrected_p']:.4f} "
              f"@L{cell['best_layer']}"
              + ("  [native]" if cell["native"] else ""), flush=True)

    matrix = {metric: {
        a: {b: results["cells"][f"{a}->{b}"][metric] for b in languages}
        for a in languages} for metric in ("disparity", "null_z", "n_labels")}
    results["matrix"] = matrix
    out = RESULTS_DIR / "transfer_pairwise.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
