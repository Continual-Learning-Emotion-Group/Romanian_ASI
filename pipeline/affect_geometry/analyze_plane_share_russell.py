"""Plane-share experiment: does the circumplex plane (PC1+PC2 of anchor-lemma
PCA) explain broader-lemma centroid variance as well as it explains held-out
anchor (classic emotion) lemmas?

Per lemma: plane share = ||in-plane component||^2 / ||total displacement||^2
in standardized space (displacement is from the anchor mean, which is the
origin after StandardScaler).

- anchor lemmas: leave-one-out (refit scaler+PCA on the other anchors,
  project the held-out lemma) -> honest out-of-sample plane share
- broader lemmas: ordinary plane (they never entered the PCA)
- comparison: Mann-Whitney U on the two plane-share distributions

Run: python -m pipeline.affect_geometry.analyze_plane_share_russell
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.common import model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = {
    "ro": HIDDEN_DIR / "ro_russell.npz",
    "en": HIDDEN_DIR / "en.npz",
    "es": HIDDEN_DIR / "es.npz",
}


def plane_share(vector, components):
    """components: (2, dim) orthonormal rows."""
    total = float(vector @ vector)
    if total == 0:
        return 1.0
    coords = components @ vector
    return float(coords @ coords / total)


def run_language(lang):
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    metrics = json.loads(
        (RESULTS_DIR / f"metrics_russell_{lang}.json").read_text())
    best_layer = metrics["full"]["best_pc1_pc2_layer"]

    lemma_to_label = {}
    for label, lemma_list in anchors["languages"][lang].items():
        for lemma in lemma_list:
            lemma_to_label[lemma] = label

    archive = np.load(ARCHIVES[lang])
    layers = archive["layers"].astype(int)
    lemmas = archive["lemmas"].astype(str)
    counts = archive["counts"].astype(int)
    layer_index = int(np.flatnonzero(layers == best_layer)[0])
    centroids = archive["centroids"][layer_index].astype(np.float64)

    anchor_mask = np.asarray([l in lemma_to_label for l in lemmas])
    anchor_idx = np.flatnonzero(anchor_mask)
    broader_idx = np.flatnonzero(~anchor_mask)
    anchor_centroids = centroids[anchor_mask]

    # full fit (used for broader lemmas + the inflated in-sample reference)
    scaler = StandardScaler().fit(anchor_centroids)
    standardized = scaler.transform(centroids)
    pca = PCA(n_components=2, random_state=0).fit(standardized[anchor_mask])
    plane = pca.components_

    broader_rows = [{
        "lemma": str(lemmas[i]),
        "count": int(counts[i]),
        "plane_share": plane_share(standardized[i], plane),
    } for i in broader_idx]

    in_sample_anchor = [plane_share(standardized[i], plane) for i in anchor_idx]

    # leave-one-out anchors
    anchor_rows = []
    for pos, i in enumerate(anchor_idx):
        rest = np.delete(anchor_centroids, pos, axis=0)
        scaler_loo = StandardScaler().fit(rest)
        pca_loo = PCA(n_components=2, random_state=0).fit(scaler_loo.transform(rest))
        vec = scaler_loo.transform(centroids[i][None, :])[0]
        anchor_rows.append({
            "lemma": str(lemmas[i]),
            "label": lemma_to_label[lemmas[i]],
            "count": int(counts[i]),
            "plane_share": plane_share(vec, pca_loo.components_),
        })

    a = np.array([r["plane_share"] for r in anchor_rows])
    b = np.array([r["plane_share"] for r in broader_rows])
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial effect size: P(anchor > broader) - P(broader > anchor)
    rank_biserial = float(2.0 * stat / (len(a) * len(b)) - 1.0)

    summary = {
        "language": lang,
        "layer": int(best_layer),
        "n_anchor_lemmas": int(len(a)),
        "n_broader_lemmas": int(len(b)),
        "anchor_in_sample_median_plane_share": float(np.median(in_sample_anchor)),
        "anchor_in_sample_mean_plane_share": float(np.mean(in_sample_anchor)),
        "anchor_loo_median_plane_share": float(np.median(a)),
        "anchor_loo_mean_plane_share": float(np.mean(a)),
        "broader_median_plane_share": float(np.median(b)),
        "broader_mean_plane_share": float(np.mean(b)),
        "mannwhitney_u": float(stat),
        "mannwhitney_p_two_sided": float(p),
        "rank_biserial_effect": rank_biserial,
        "prob_random_anchor_exceeds_random_broader": float(stat / (len(a) * len(b))),
    }
    result = {"summary": summary, "anchors_loo": anchor_rows, "broader": broader_rows}
    out = RESULTS_DIR / f"plane_share_russell_{lang}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    print(f"=== {lang} (layer {best_layer}) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["en", "ro", "es"]):
        run_language(lang)
