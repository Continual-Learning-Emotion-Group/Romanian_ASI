"""Frozen-plane cross-language transfer scored on the GLOBAL 8-language
label intersection — every one of the 64 cells uses the identical label set,
so raw disparities are directly comparable across the whole table (null_z is
still reported for continuity with transfer_pairwise).

Under the strict anchors this set had only 4 labels (unfalsifiable: 4! = 24
permutations); the 2026-08-02 loose anchor pass raised it to 10, which makes
this table meaningful for the first time.

Protocol per cell identical to transfer_pairwise (plane selection on the
source's FULL anchor set; targets recentered by own mean, source scales).

Run: python -m pipeline.affect_geometry.transfer_global_intersection
Output: results/transfer_global_intersection.json
"""
from __future__ import annotations

import itertools
import json

from pipeline.affect_geometry.transfer_pairwise import transfer_cell
from pipeline.affect_geometry.transfer_shared_labels import (
    ARCHIVES, RESULTS_DIR, Language,
)


def main():
    languages = {lang: Language(lang) for lang in ARCHIVES}
    shared = set.intersection(*(l.present_labels for l in languages.values()))
    print(f"global intersection: {len(shared)} labels: {sorted(shared)}")
    results = {"languages": list(languages),
               "shared_labels": sorted(shared), "cells": {}}
    for a, b in itertools.product(languages, repeat=2):
        cell = transfer_cell(languages[a], languages[b], fixed_labels=shared)
        results["cells"][f"{a}->{b}"] = cell
        print(f"{a}->{b}: D={cell['disparity']:.3f} z={cell['null_z']:.2f} "
              f"p={cell['layer_corrected_p']:.4f} @L{cell['best_layer']}"
              + ("  [native]" if cell["native"] else ""), flush=True)

    results["matrix"] = {metric: {
        a: {b: results["cells"][f"{a}->{b}"][metric] for b in languages}
        for a in languages} for metric in ("disparity", "null_z")}
    out = RESULTS_DIR / "transfer_global_intersection.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
