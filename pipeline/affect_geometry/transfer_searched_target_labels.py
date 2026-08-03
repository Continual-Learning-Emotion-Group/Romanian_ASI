"""Layer-SEARCHED cross-language transfer scored on the TARGET'S FULL labels.

Companion to transfer_fixed_layer_target_labels: cell (A -> B) again scores
B's ENTIRE anchor label set (columns = identical exams, within-column
comparisons fair), but A's layer is chosen per target — all layers are
swept and the one where B fits best is kept (A's plane per layer is still
built only from A's own anchors). Because the reported disparity benefits
from that search, p and the null used for skill come from the
layer-search-corrected permutation null (each label shuffle also takes its
min over layers), as in transfer_pairwise.

skill = 1 - D / null_mean, with null_mean from the corrected
(min-over-layers) null, so skill answers "fraction of chance-level
disparity eliminated, where chance also enjoyed the layer search".

Run: python -m pipeline.affect_geometry.transfer_searched_target_labels
Output: results/<variant>/transfer_searched_target_labels.json
"""
from __future__ import annotations

import itertools
import json

from pipeline.affect_geometry.transfer_pairwise import transfer_cell
from pipeline.affect_geometry.transfer_shared_labels import (
    ARCHIVES, RESULTS_DIR, Language, angle_of,
)


def main():
    languages = {lang: Language(lang) for lang in ARCHIVES}
    results = {"languages": list(languages),
               "target_label_counts": {lang: len(l.present_labels)
                                       for lang, l in languages.items()},
               "cells": {}}
    for a, b in itertools.product(languages, repeat=2):
        labels = sorted(languages[b].present_labels,
                        key=lambda l: angle_of[l])
        cell = transfer_cell(languages[a], languages[b], fixed_labels=labels)
        cell.pop("layers", None)
        cell["skill"] = 1 - cell["disparity"] / cell["null_mean"]
        results["cells"][f"{a}->{b}"] = cell
        print(f"{a}->{b}: n={cell['n_labels']} D={cell['disparity']:.3f} "
              f"skill={cell['skill']:.2f} p={cell['layer_corrected_p']:.4f} "
              f"@L{cell['best_layer']}"
              + ("  [native]" if cell["native"] else ""), flush=True)

    results["matrix"] = {metric: {
        a: {b: results["cells"][f"{a}->{b}"][metric] for b in languages}
        for a in languages} for metric in ("disparity", "skill",
                                           "layer_corrected_p", "n_labels",
                                           "best_layer")}
    out = RESULTS_DIR / "transfer_searched_target_labels.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
