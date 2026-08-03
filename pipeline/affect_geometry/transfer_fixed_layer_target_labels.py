"""Fixed-layer cross-language transfer scored on the TARGET'S FULL labels.

Column-fair variant: cell (A -> B) projects B's centroids onto A's plane
(frozen once at A's native-best layer, target never seen) and scores B's
ENTIRE anchor label set — including labels A itself lacks. Every cell in a
column therefore answers the identical exam (B's full circumplex), so
within-column comparisons of sources are exactly fair; rows/cross-column
comparisons use different label sets and should not be compared directly.

Significance per cell: plain PROTEST label permutation at the frozen layer.
Reported effect size: skill = 1 - D / null_mean (fraction of chance-level
disparity eliminated). Diagonal = native fit on own full set (reference:
frozen layer chosen on the same data).

Run: python -m pipeline.affect_geometry.transfer_fixed_layer_target_labels
Output: results/<variant>/transfer_fixed_layer_target_labels.json
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
               "target_label_counts": {lang: len(l.present_labels)
                                       for lang, l in languages.items()},
               "cells": {}}
    for a, b in itertools.product(languages, repeat=2):
        labels = sorted(languages[b].present_labels,
                        key=lambda l: angle_of[l])
        cell = fixed_cell(languages[a], languages[b], frozen[a], labels)
        cell["skill"] = 1 - cell["disparity"] / cell["null_mean"]
        results["cells"][f"{a}->{b}"] = cell
        print(f"{a}->{b}: n={cell['n_labels']} D={cell['disparity']:.3f} "
              f"skill={cell['skill']:.2f} p={cell['protest_p']:.4f} "
              f"@L{cell['frozen_layer']}"
              + ("  [native]" if cell["native"] else ""), flush=True)

    results["matrix"] = {metric: {
        a: {b: results["cells"][f"{a}->{b}"][metric] for b in languages}
        for a in languages} for metric in ("disparity", "skill",
                                           "protest_p", "n_labels")}
    out = RESULTS_DIR / "transfer_fixed_layer_target_labels.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
