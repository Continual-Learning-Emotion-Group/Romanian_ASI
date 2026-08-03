"""Fixed-layer cross-language transfer on PER-PAIR shared labels.

The headline-protocol combination: each source language's layer (and plane)
is frozen ONCE at its native best — the layer where its own full anchor set
best fits Russell's circle, chosen before any target is seen — and each cell
(A -> B) scores B's centroids on the labels shared by just A and B, keeping
every pair's circle coverage as rich as the pair allows.

Because nothing is searched per target, significance per cell is a plain
PROTEST label-permutation test (Jackson 1995) at that single frozen layer.
Raw disparities are not comparable across cells with different label sets,
so the table reports each cell's null_z = (mean(null) - D) / sd(null)
(higher = better fit vs chance), with n_labels shown per cell.

Diagonal cells are reference values (the frozen layer was chosen on the
same language's data).

Run: python -m pipeline.affect_geometry.transfer_fixed_layer_pairwise
Output: results/<variant>/transfer_fixed_layer_pairwise.json
"""
from __future__ import annotations

import itertools
import json

from pipeline.affect_geometry.transfer_fixed_layer import (
    fixed_cell, native_best_index,
)
from pipeline.affect_geometry.transfer_shared_labels import (
    ARCHIVES, RESULTS_DIR, Language, angle_of,
)


def main():
    languages = {lang: Language(lang) for lang in ARCHIVES}
    frozen = {}
    for lang, language in languages.items():
        index, native_d = native_best_index(language)
        frozen[lang] = index
        print(f"{lang}: frozen layer L{int(language.layers[index])} "
              f"(native full-anchor D={native_d:.3f})")

    results = {"languages": list(languages),
               "frozen_layers": {lang: int(languages[lang].layers[i])
                                 for lang, i in frozen.items()},
               "cells": {}}
    for a, b in itertools.product(languages, repeat=2):
        shared = sorted(
            languages[a].present_labels & languages[b].present_labels,
            key=lambda l: angle_of[l])
        cell = fixed_cell(languages[a], languages[b], frozen[a], shared)
        results["cells"][f"{a}->{b}"] = cell
        print(f"{a}->{b}: n={cell['n_labels']} D={cell['disparity']:.3f} "
              f"z={cell['null_z']:.2f} p={cell['protest_p']:.4f} "
              f"@L{cell['frozen_layer']}"
              + ("  [native]" if cell["native"] else ""), flush=True)

    results["matrix"] = {metric: {
        a: {b: results["cells"][f"{a}->{b}"][metric] for b in languages}
        for a in languages} for metric in ("disparity", "null_z",
                                           "protest_p", "n_labels")}
    out = RESULTS_DIR / "transfer_fixed_layer_pairwise.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
