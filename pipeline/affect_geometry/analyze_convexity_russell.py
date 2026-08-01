"""Convex-combination test: are broader-state centroids convex combinations of
the Russell anchor label centroids in the FULL 2560-dim standardized space?

Per language, at the frozen best PC1+PC2 layer (from metrics_russell_*.json):
- affine R2: projection of (b - anchor_mean) onto the anchors' affine subspace
- convex R2: simplex-constrained (w >= 0, sum w = 1) least-squares fit
- null: convex R2 against randomly rotated anchor simplexes (same shape,
  random orientation inside the span of all centroids), 500 rotations
- control: leave-one-out convex fit of each anchor label from the others
- adjacency: weight-implied circular angle vs the lemma's angle in the
  aligned PC1+PC2 plane (circular correlation, radius-filtered)

All distances are computed in reduced orthonormal coordinates of the span of
all centroids (exact for ratios of Euclidean norms).

Run: python -m pipeline.affect_geometry.analyze_convexity_russell
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler

from pipeline.affect_geometry.common import model_paths

PACKAGE = Path(__file__).resolve().parent
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
SEED = 20260730
N_ROTATIONS = 500
LAMBDA_FACTOR = 100.0  # simplex-sum penalty scale relative to data norm
ARCHIVES = {
    "ro": HIDDEN_DIR / "ro_russell.npz",
    "en": HIDDEN_DIR / "en.npz",
    "es": HIDDEN_DIR / "es.npz",
}


def simplex_fit(anchor_matrix, target, lam):
    """min ||A^T w - b|| s.t. w >= 0, sum w ~= 1 (augmented NNLS), then
    renormalize to exact simplex weights."""
    n_anchors = anchor_matrix.shape[0]
    design = np.vstack([anchor_matrix.T, lam * np.ones((1, n_anchors))])
    rhs = np.concatenate([target, [lam]])
    weights, _ = nnls(design, rhs)
    total = weights.sum()
    if total <= 0:
        weights = np.full(n_anchors, 1.0 / n_anchors)
    else:
        weights = weights / total
    fit = anchor_matrix.T @ weights
    return weights, fit


def circular_corr(a, b):
    """Jammalamadaka-SenGupta circular correlation of two angle arrays (rad)."""
    abar = np.angle(np.exp(1j * a).mean())
    bbar = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - abar), np.sin(b - bbar)
    denom = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / denom) if denom > 0 else float("nan")


def main():
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    angle_of = anchors["angles_degrees"]
    rng = np.random.default_rng(SEED)
    all_results = {}

    for lang, npz_path in ARCHIVES.items():
        metrics = json.loads(
            (RESULTS_DIR / f"metrics_russell_{lang}.json").read_text())
        best_layer = metrics["full"]["best_pc1_pc2_layer"]
        proj = json.loads(
            (RESULTS_DIR / f"projections_russell_{lang}.json").read_text())
        plane_xy = {s["lemma"]: (s["pc1_pc2_x"], s["pc1_pc2_y"])
                    for s in proj["states"]}

        lemma_to_label = {}
        for label, lemma_list in anchors["languages"][lang].items():
            for lemma in lemma_list:
                lemma_to_label[lemma] = label

        archive = np.load(npz_path)
        layers = archive["layers"].astype(int)
        lemmas = archive["lemmas"].astype(str)
        counts = archive["counts"].astype(int)
        layer_index = int(np.flatnonzero(layers == best_layer)[0])
        centroids = archive["centroids"][layer_index].astype(np.float64)

        labels_arr = np.asarray([lemma_to_label.get(l, "") for l in lemmas])
        anchor_mask = labels_arr != ""
        label_list = sorted({l for l in labels_arr if l}, key=lambda l: angle_of[l])
        angles_rad = np.radians([angle_of[l] for l in label_list])

        scaler = StandardScaler().fit(centroids[anchor_mask])
        standardized = scaler.transform(centroids)

        # reduced orthonormal coordinates of the span of all centered centroids
        grand_mean = standardized.mean(0)
        centered = standardized - grand_mean
        _, svals, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int((svals > svals[0] * 1e-10).sum())
        coords = centered @ vt[:rank].T                      # (n_lemmas, rank)

        label_centroids = np.asarray([
            coords[labels_arr == label].mean(0) for label in label_list
        ])                                                    # (L, rank)
        anchor_mean = label_centroids.mean(0)
        broader_idx = np.flatnonzero(~anchor_mask)
        lam = LAMBDA_FACTOR * float(np.linalg.norm(coords, axis=1).mean())

        # orthonormal basis of the anchors' affine subspace
        span = label_centroids - anchor_mean
        q, r = np.linalg.qr(span.T)
        keep = np.abs(np.diag(r)) > np.abs(r).max() * 1e-10
        basis = q[:, keep]                                    # (rank, L-1)

        def affine_r2(target):
            diff = target - anchor_mean
            base = float(diff @ diff)
            if base == 0:
                return 1.0
            proj_vec = basis @ (basis.T @ diff)
            return float(proj_vec @ proj_vec / base)

        def convex_r2(anchor_matrix, target):
            weights, fit = simplex_fit(anchor_matrix, target, lam)
            diff = target - anchor_matrix.mean(0)
            base = float(diff @ diff)
            resid = target - fit
            r2 = 1.0 - float(resid @ resid) / base if base > 0 else 1.0
            return weights, max(r2, 0.0)

        rows = []
        for i in broader_idx:
            target = coords[i]
            weights, cvx = convex_r2(label_centroids, target)
            aff = affine_r2(target)
            order = np.argsort(weights)[::-1]
            resultant = complex(np.sum(weights * np.exp(1j * angles_rad)))
            implied_angle = float(np.degrees(np.angle(resultant)) % 360.0)
            wpos = weights[weights > 1e-12]
            eff_anchors = float(np.exp(-(wpos * np.log(wpos)).sum()))
            x, y = plane_xy[lemmas[i]]
            rows.append({
                "lemma": str(lemmas[i]),
                "count": int(counts[i]),
                "affine_r2": aff,
                "convex_r2": cvx,
                "convexity_gap": aff - cvx,
                "effective_anchors": eff_anchors,
                "weight_resultant_length": float(abs(resultant)),
                "implied_angle_deg": implied_angle,
                "plane_angle_deg": float(np.degrees(np.arctan2(y, x)) % 360.0),
                "plane_radius": float(np.hypot(x, y)),
                "top_weights": [
                    {"label": label_list[j], "weight": float(weights[j])}
                    for j in order[:3] if weights[j] > 0.01
                ],
            })

        # anchor leave-one-out control (are anchors extreme points?)
        loo = []
        for k, label in enumerate(label_list):
            others = np.delete(label_centroids, k, axis=0)
            _, cvx = convex_r2(others, label_centroids[k])
            loo.append({"label": label, "convex_r2": cvx})

        # null: same simplex shape, random orientation within the centroid span
        broader_targets = coords[broader_idx]
        null_means = np.empty(N_ROTATIONS)
        for t in range(N_ROTATIONS):
            gauss = rng.standard_normal((rank, rank))
            q_rot, r_rot = np.linalg.qr(gauss)
            q_rot *= np.sign(np.diag(r_rot))
            rotated = anchor_mean + (label_centroids - anchor_mean) @ q_rot.T
            vals = [convex_r2(rotated, bt)[1] for bt in broader_targets]
            null_means[t] = float(np.mean(vals))

        obs_mean = float(np.mean([r["convex_r2"] for r in rows]))
        p_value = float((1 + np.sum(null_means >= obs_mean)) / (N_ROTATIONS + 1))

        # angle agreement, filtered to lemmas away from the plane origin
        radii = np.asarray([r["plane_radius"] for r in rows])
        radius_cut = float(np.median(radii))
        sel = radii >= radius_cut
        implied = np.radians([r["implied_angle_deg"] for r in rows])
        observed = np.radians([r["plane_angle_deg"] for r in rows])
        ang_corr = circular_corr(implied[sel], observed[sel])
        ang_err = np.degrees(np.abs(np.angle(np.exp(1j * (implied - observed)))))

        summary = {
            "layer": int(best_layer),
            "n_anchor_labels": len(label_list),
            "n_broader_lemmas": len(rows),
            "span_rank": rank,
            "mean_convex_r2": obs_mean,
            "median_convex_r2": float(np.median([r["convex_r2"] for r in rows])),
            "mean_affine_r2": float(np.mean([r["affine_r2"] for r in rows])),
            "mean_convexity_gap": float(np.mean([r["convexity_gap"] for r in rows])),
            "null_mean_convex_r2": {
                "q50": float(np.quantile(null_means, 0.50)),
                "q95": float(np.quantile(null_means, 0.95)),
                "max": float(null_means.max()),
            },
            "p_mean_convex_r2": p_value,
            "anchor_loo_mean_convex_r2": float(np.mean([r["convex_r2"] for r in loo])),
            "anchor_loo_median_convex_r2": float(np.median([r["convex_r2"] for r in loo])),
            "median_effective_anchors": float(np.median(
                [r["effective_anchors"] for r in rows])),
            "angle_circular_corr_outer_half": ang_corr,
            "angle_radius_cut": radius_cut,
            "median_abs_angle_error_deg_outer_half": float(np.median(ang_err[sel])),
            "median_abs_angle_error_deg_all": float(np.median(ang_err)),
        }
        all_results[lang] = {"summary": summary, "anchor_loo": loo, "broader": rows}
        print(f"=== {lang} (layer {best_layer}) ===")
        print(json.dumps(summary, indent=2))

    output = RESULTS_DIR / "convexity_russell.json"
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("wrote", output)


if __name__ == "__main__":
    main()
