"""Frozen-plane cross-language transfer scored on SHARED label sets (tiered).

Same frozen protocol as transfer_cross_language.py (source A's all-states
PCA per layer, A's circumplex pair chosen on A's own full anchor set, target
B recentered by its own mean and scaled with A's per-dimension scales), but
every cell (including the native diagonal) is scored on one fixed label set
per TIER, so each tier's table fits the same shape and is comparable within
columns, across columns, and against the diagonal.

Tiers (a tier runs only when all its archives exist):
- six_lang: en/es/zh/ro/fa/hi on their shared labels (11 on the legacy
  archives; intersections are computed from labels actually present).
- seven_lang_fr: the six plus French (8 shared labels). Indonesian is
  excluded from tiers: its 9-label anchor set would collapse the global
  intersection to 4 labels, which a 2D Procrustes cannot falsify (see the
  pairwise-intersection analysis in transfer_pairwise.py for id).

Layer selection on the target side is corrected with a label-permutation
null that replays the layer search (permutation_p_from_shapes). Diagonal
cells are the native plane scored on shared labels; their pair selection saw
(a superset of) the scored labels, so they are reported as reference values
with the same corrected p for completeness.

Run: python -m pipeline.affect_geometry.transfer_shared_labels
Output: results/transfer_shared_labels.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.analyze import (
    normalized_shape,
    permutation_p_from_shapes,
    procrustes_disparity,
)

from pipeline.affect_geometry.common import discover_archives, model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = discover_archives(HIDDEN_DIR)
N_COMPONENTS = 20
SEARCH_WIDTH = 10
TIERS = {
    "six_lang": ["en", "es", "zh", "ro", "fa", "hi"],
    "seven_lang_fr": ["en", "es", "zh", "ro", "fa", "hi", "fr"],
}

config = json.loads((PACKAGE / "config.json").read_text())
SEED = int(config["random_seed"])
PERMUTATIONS = int(config["analysis"]["permutations"])
anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
angle_of = anchors["angles_degrees"]


def theory_of(labels):
    radians = np.radians([angle_of[l] for l in labels])
    return np.column_stack((np.cos(radians), np.sin(radians)))


def search_pair(scores, theory, width):
    best = None
    for first, second in itertools.combinations(range(min(width, scores.shape[1])), 2):
        disparity = procrustes_disparity(theory, scores[:, [first, second]])
        candidate = (disparity, first, second)
        if best is None or candidate < best:
            best = candidate
    return best


class Language:
    def __init__(self, lang):
        self.lang = lang
        lemma_to_label = {}
        for label, lemma_list in anchors["languages"][lang].items():
            for lemma in lemma_list:
                lemma_to_label[lemma] = label
        archive = np.load(ARCHIVES[lang])
        self.layers = archive["layers"].astype(int)
        self.centroids = archive["centroids"].astype(np.float64)
        self.labels_arr = np.asarray([
            lemma_to_label.get(l, "") for l in archive["lemmas"].astype(str)])
        self.present_labels = {l for l in self.labels_arr if l}
        self.full_label_list = sorted(self.present_labels,
                                      key=lambda l: angle_of[l])
        radians = np.radians([angle_of[l] for l in self.full_label_list])
        self.full_theory = np.column_stack((np.cos(radians), np.sin(radians)))
        self.planes = []
        for index in range(len(self.layers)):
            scaler = StandardScaler().fit(self.centroids[index])
            standardized = scaler.transform(self.centroids[index])
            n_comp = min(N_COMPONENTS, standardized.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=0).fit(standardized)
            projected = pca.transform(standardized)
            scores = np.asarray([
                projected[self.labels_arr == label].mean(0)
                for label in self.full_label_list
            ])
            _, first, second = search_pair(scores, self.full_theory, SEARCH_WIDTH)
            self.planes.append({
                "scale": scaler.scale_,
                "components": pca.components_[[first, second]],
                "pair": [int(first), int(second)],
            })


def transfer_shared(source, target, shared, theory):
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
    return {
        "source": source.lang, "target": target.lang,
        "native": source.lang == target.lang,
        "n_labels": len(shared),
        "best_layer": best["layer"],
        "source_pair_at_best": best["source_pair"],
        "disparity": best["disparity"],
        "layer_corrected_p": p,
        "null_min_disparity_q50": float(np.quantile(null, 0.50)),
        "layers": rows,
    }


def main():
    cache = {}
    results = {"tiers": {}}
    for tier, tier_langs in TIERS.items():
        if not all(lang in ARCHIVES for lang in tier_langs):
            print(f"tier {tier}: skipped (missing archives)")
            continue
        for lang in tier_langs:
            cache.setdefault(lang, Language(lang))
        shared = sorted(
            set.intersection(*[cache[lang].present_labels for lang in tier_langs]),
            key=lambda l: angle_of[l])
        theory = theory_of(shared)
        tier_result = {"languages": tier_langs, "shared_labels": shared, "cells": {}}
        print(f"tier {tier}: {len(shared)} shared labels: {shared}")
        for a, b in itertools.product(tier_langs, repeat=2):
            summary = transfer_shared(cache[a], cache[b], shared, theory)
            tier_result["cells"][f"{a}->{b}"] = summary
            print(f"  {a}->{b}: D={summary['disparity']:.3f} @L{summary['best_layer']} "
                  f"p={summary['layer_corrected_p']:.4f}"
                  + ("  [native]" if summary["native"] else ""), flush=True)
        results["tiers"][tier] = tier_result
    out = RESULTS_DIR / "transfer_shared_labels.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
