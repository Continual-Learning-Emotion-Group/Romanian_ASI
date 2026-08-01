"""Cross-language basis search (free-search tier).

For every ordered pair (source basis B, target A) INCLUDING the diagonal
(native basis), search ALL of B's top-10 PC pairs at ALL layers for the plane
that best fits A's anchors. Selection (pair + layer) happens on the target
side, exactly mirroring the native searched analysis, and the permutation
null repeats the full search.

Two scopes:
- "full": each target evaluated on its complete label set (21/21/18 labels);
  rows are comparable within a target, not across targets
- "shared": every cell evaluated on the 15-label intersection, making the
  whole 3x3 table comparable (used for the basis-quality comparison)

Run: python -m pipeline.affect_geometry.basis_search_cross_language
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.analyze import (
    global_pair_search_permutation_p,
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
WIDTH = 10

config = json.loads((PACKAGE / "config.json").read_text())
SEED = int(config["random_seed"])
PERMUTATIONS = int(config["analysis"]["permutations"])
anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
angle_of = anchors["angles_degrees"]
SHARED = set.intersection(*[set(v) for v in anchors["languages"].values()])


def load_languages(scope):
    data = {}
    for lang, path in ARCHIVES.items():
        lemma_to_label = {}
        for label, lemma_list in anchors["languages"][lang].items():
            for lemma in lemma_list:
                lemma_to_label[lemma] = label
        archive = np.load(path)
        labels_arr = np.asarray([lemma_to_label.get(l, "")
                                 for l in archive["lemmas"].astype(str)])
        present = {l for l in labels_arr if l}
        if scope == "shared":
            present &= SHARED
        label_list = sorted(present, key=lambda l: angle_of[l])
        radians = np.radians([angle_of[l] for l in label_list])
        data[lang] = {
            "layers": archive["layers"].astype(int),
            "centroids": archive["centroids"].astype(np.float64),
            "labels_arr": labels_arr,
            "label_list": label_list,
            "theory": np.column_stack((np.cos(radians), np.sin(radians))),
            "scalers": [], "pcas": [],
        }
        for index in range(len(data[lang]["layers"])):
            scaler = StandardScaler().fit(data[lang]["centroids"][index])
            pca = PCA(n_components=WIDTH, random_state=0).fit(
                scaler.transform(data[lang]["centroids"][index]))
            data[lang]["scalers"].append(scaler)
            data[lang]["pcas"].append(pca)
    return data


def run_scope(scope):
    data = load_languages(scope)
    results = {}
    for src, tgt in itertools.product(data, repeat=2):
        S, T = data[src], data[tgt]
        category_by_layer = []
        best = None
        for index in range(len(S["layers"])):
            raw = T["centroids"][index]
            z = (raw - raw.mean(0)) / S["scalers"][index].scale_
            projected = z @ S["pcas"][index].components_.T
            scores = np.asarray([
                projected[T["labels_arr"] == label].mean(0)
                for label in T["label_list"]
            ])
            category_by_layer.append(scores)
            for first, second in itertools.combinations(range(WIDTH), 2):
                disparity = procrustes_disparity(
                    T["theory"], scores[:, [first, second]])
                candidate = (disparity, int(S["layers"][index]), first, second)
                if best is None or candidate < best:
                    best = candidate
        disparity, layer, first, second = best
        p, null = global_pair_search_permutation_p(
            category_by_layer, T["theory"], disparity, WIDTH, PERMUTATIONS, SEED)
        results[f"{src}->{tgt}"] = {
            "n_labels": len(T["label_list"]),
            "best_disparity": disparity, "best_layer": layer,
            "best_pair": [first, second], "corrected_p": p,
            "null_min_disparity_q50": float(np.quantile(null, 0.50)),
        }
        print(f"[{scope}] {src}->{tgt}: disparity {disparity:.3f} @L{layer} "
              f"PC{first+1}+PC{second+1}, corrected p={p:.4f}, "
              f"null q50={np.quantile(null, 0.50):.3f}")
    return results


def main():
    results = {scope: run_scope(scope) for scope in ("full", "shared")}
    results["shared_labels"] = sorted(SHARED, key=lambda l: angle_of[l])
    out = RESULTS_DIR / "cross_language_basis_search.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
