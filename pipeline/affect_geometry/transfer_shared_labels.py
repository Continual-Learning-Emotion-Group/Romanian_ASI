"""Frozen-plane cross-language transfer scored on the SHARED label set.

Same frozen protocol as transfer_cross_language.py (source A's all-states
PCA per layer, A's circumplex pair chosen on A's own full anchor set, target
B recentered by its own mean and scaled with A's per-dimension scales), but
every cell (including the native diagonal) is scored on the 11 circumplex
labels shared by all six languages, so the whole 6x6 table fits the same
11-point shape and is comparable within columns, across columns, and against
the diagonal.

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

from pipeline.affect_geometry.common import model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = {
    "ro": HIDDEN_DIR / "ro_russell.npz",
    "en": HIDDEN_DIR / "en.npz",
    "es": HIDDEN_DIR / "es.npz",
    "zh": HIDDEN_DIR / "zh.npz",
    "fa": HIDDEN_DIR / "fa.npz",
    "hi": HIDDEN_DIR / "hi.npz",
}
N_COMPONENTS = 20
SEARCH_WIDTH = 10

config = json.loads((PACKAGE / "config.json").read_text())
SEED = int(config["random_seed"])
PERMUTATIONS = int(config["analysis"]["permutations"])
anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
angle_of = anchors["angles_degrees"]
SHARED = sorted(set.intersection(*[set(v) for v in anchors["languages"].values()]),
                key=lambda l: angle_of[l])
SHARED_RADIANS = np.radians([angle_of[l] for l in SHARED])
SHARED_THEORY = np.column_stack((np.cos(SHARED_RADIANS), np.sin(SHARED_RADIANS)))


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
        self.full_label_list = sorted({l for l in self.labels_arr if l},
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


def transfer_shared(source, target):
    rows, shapes = [], []
    for index in range(len(source.planes)):
        info = source.planes[index]
        raw = target.centroids[index]
        z = (raw - raw.mean(0)) / info["scale"]
        points = z @ info["components"].T
        scores = np.asarray([
            points[target.labels_arr == label].mean(0) for label in SHARED
        ])
        rows.append({
            "layer": int(source.layers[index]),
            "source_pair": info["pair"],
            "disparity": procrustes_disparity(SHARED_THEORY, scores),
        })
        shapes.append(normalized_shape(scores))
    best_index = min(range(len(rows)), key=lambda i: rows[i]["disparity"])
    best = rows[best_index]
    p, null = permutation_p_from_shapes(
        shapes, SHARED_THEORY, best["disparity"], PERMUTATIONS, SEED)
    return {
        "source": source.lang, "target": target.lang,
        "native": source.lang == target.lang,
        "n_labels": len(SHARED),
        "best_layer": best["layer"],
        "source_pair_at_best": best["source_pair"],
        "disparity": best["disparity"],
        "layer_corrected_p": p,
        "null_min_disparity_q50": float(np.quantile(null, 0.50)),
        "layers": rows,
    }


def main():
    languages = {lang: Language(lang) for lang in ("en", "es", "zh", "ro", "fa", "hi")}
    results = {"shared_labels": SHARED}
    for a, b in itertools.product(languages, repeat=2):
        summary = transfer_shared(languages[a], languages[b])
        results[f"{a}->{b}"] = summary
        print(f"{a}->{b}: D={summary['disparity']:.3f} @L{summary['best_layer']} "
              f"pair PC{summary['source_pair_at_best'][0]+1}+"
              f"PC{summary['source_pair_at_best'][1]+1} "
              f"p={summary['layer_corrected_p']:.4f} "
              f"null q50={summary['null_min_disparity_q50']:.3f}"
              + ("  [native]" if summary["native"] else ""), flush=True)
    out = RESULTS_DIR / "transfer_shared_labels.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
