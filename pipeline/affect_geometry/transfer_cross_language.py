"""Cross-language circumplex transfer (frozen-plane tier).

For each ordered language pair (A -> B), per layer:
- fit scaler + PCA on ALL of A's state centroids (all-states PCA),
  choose A's circumplex pair by searching A's top-10 PCs against A's anchors
  (selection happens entirely on A's side);
- project B's centroids onto A's frozen pair: B is recentered by its own mean
  (language offset removed) but scaled with A's per-dimension scaling;
- score B's anchor labels against B's Russell targets (Procrustes disparity).

Reported per pair: best transfer layer + disparity, layer-selection-corrected
label-permutation p (pair choice on A is independent of B's labels), B's
native all-states disparity at the same layer, broader-variance fraction on
the transferred plane vs B's own plane (transfer ratio), and principal angles
between A's and B's planes in raw functional space.

Figures: for each target B, its centroids projected on its own axes and on
each other language's axes (aligned to B's Russell targets).

Run: python -m pipeline.affect_geometry.transfer_cross_language
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pipeline.affect_geometry.analyze import (  # noqa: E402
    align_to_theory,
    normalized_shape,
    permutation_p_from_shapes,
    procrustes_disparity,
)

PACKAGE = Path(__file__).resolve().parent
ARCHIVES = {
    "ro": PACKAGE / "artifacts/hidden/ro_russell.npz",
    "en": PACKAGE / "artifacts/hidden/en.npz",
    "es": PACKAGE / "artifacts/hidden/es.npz",
    "zh": PACKAGE / "artifacts/hidden/zh.npz",
    "fa": PACKAGE / "artifacts/hidden/fa.npz",
    "hi": PACKAGE / "artifacts/hidden/hi.npz",
}
NAMES = {"ro": "Romanian", "en": "English", "es": "Spanish",
         "zh": "Mandarin", "fa": "Persian", "hi": "Hindi"}
N_COMPONENTS = 20
SEARCH_WIDTH = 10

config = json.loads((PACKAGE / "config.json").read_text())
SEED = int(config["random_seed"])
PERMUTATIONS = int(config["analysis"]["permutations"])
anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
angle_of = anchors["angles_degrees"]


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
        self.lemmas = archive["lemmas"].astype(str)
        self.centroids = archive["centroids"].astype(np.float64)
        self.labels_arr = np.asarray([lemma_to_label.get(l, "") for l in self.lemmas])
        self.anchor_mask = self.labels_arr != ""
        self.label_list = sorted({l for l in self.labels_arr if l},
                                 key=lambda l: angle_of[l])
        radians = np.radians([angle_of[l] for l in self.label_list])
        self.theory = np.column_stack((np.cos(radians), np.sin(radians)))
        self.per_layer = []
        for index in range(len(self.layers)):
            raw = self.centroids[index]
            scaler = StandardScaler().fit(raw)
            standardized = scaler.transform(raw)
            n_comp = min(N_COMPONENTS, standardized.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=0).fit(standardized)
            projected = pca.transform(standardized)
            scores = np.asarray([
                projected[self.labels_arr == label].mean(0)
                for label in self.label_list
            ])
            disparity, first, second = search_pair(scores, self.theory, SEARCH_WIDTH)
            pair = [first, second]
            broader = ~self.anchor_mask
            own_frac = float(
                np.var(projected[broader][:, pair], axis=0, ddof=1).sum()
                / np.var(standardized[broader], axis=0, ddof=1).sum())
            # raw-space functional directions of the plane (z = (x-m)/s)
            raw_dirs = pca.components_[pair] / scaler.scale_
            raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
            self.per_layer.append({
                "scale": scaler.scale_, "components": pca.components_,
                "pair": pair, "own_disparity": disparity,
                "own_broader_frac": own_frac, "raw_plane": raw_dirs,
            })
        self.own_best_index = min(
            range(len(self.per_layer)),
            key=lambda i: self.per_layer[i]["own_disparity"])

    def project_own(self, index):
        raw = self.centroids[index]
        info = self.per_layer[index]
        standardized = StandardScaler().fit_transform(raw)
        return (standardized @ info["components"].T)[:, info["pair"]]


def transfer(source, target):
    rows, shapes, points_by_layer = [], [], []
    for index in range(len(source.per_layer)):
        info = source.per_layer[index]
        raw = target.centroids[index]
        z = (raw - raw.mean(0)) / info["scale"]
        points = z @ info["components"][info["pair"]].T
        scores = np.asarray([
            points[target.labels_arr == label].mean(0)
            for label in target.label_list
        ])
        disparity = procrustes_disparity(target.theory, scores)
        broader = ~target.anchor_mask
        frac = float(np.var(points[broader], axis=0, ddof=1).sum()
                     / np.var(z[broader], axis=0, ddof=1).sum())
        overlap = np.linalg.svd(
            info["raw_plane"] @ target.per_layer[index]["raw_plane"].T,
            compute_uv=False)
        rows.append({
            "layer": int(source.layers[index]),
            "source_pair": info["pair"],
            "transfer_disparity": disparity,
            "target_native_disparity": target.per_layer[index]["own_disparity"],
            "transfer_broader_frac": frac,
            "target_own_broader_frac": target.per_layer[index]["own_broader_frac"],
            "plane_principal_angles_deg": [
                float(np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))) for s in overlap],
        })
        shapes.append(normalized_shape(scores))
        points_by_layer.append(points)

    best_index = min(range(len(rows)), key=lambda i: rows[i]["transfer_disparity"])
    best = rows[best_index]
    p, null = permutation_p_from_shapes(
        [s for s in shapes if s is not None], target.theory,
        best["transfer_disparity"], PERMUTATIONS, SEED)
    summary = {
        "source": source.lang, "target": target.lang,
        "best_transfer_layer": best["layer"],
        "source_pair_at_best": [int(v) for v in best["source_pair"]],
        "transfer_disparity": best["transfer_disparity"],
        "layer_corrected_p": p,
        "null_min_disparity_q50": float(np.quantile(null, 0.50)),
        "target_native_disparity_same_layer": best["target_native_disparity"],
        "transfer_broader_frac": best["transfer_broader_frac"],
        "target_own_broader_frac_same_layer": best["target_own_broader_frac"],
        "transfer_ratio": best["transfer_broader_frac"]
        / max(best["target_own_broader_frac"], 1e-12),
        "plane_principal_angles_deg_at_best": best["plane_principal_angles_deg"],
        "layers": rows,
    }
    return summary, best_index, points_by_layer[best_index]


def panel(ax, points, language, title):
    theta = np.linspace(0, 2 * np.pi, 200)
    scores = np.asarray([
        points[language.labels_arr == label].mean(0)
        for label in language.label_list
    ])
    aligned = align_to_theory(points, scores, language.theory)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.75", lw=1.0, zorder=0)
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.axvline(0, color="0.9", lw=0.8, zorder=0)
    broader = aligned[~language.anchor_mask]
    ax.scatter(broader[:, 0], broader[:, 1], s=12, color="0.65", alpha=0.5,
               zorder=1, label=f"broader states (n={len(broader)})")
    for label in language.label_list:
        hue = cm.hsv(angle_of[label] / 360.0)
        mx, my = aligned[language.labels_arr == label].mean(0)
        ax.scatter(mx, my, s=110, color=[hue], zorder=4)
        ax.annotate(label, (mx, my), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    languages = {lang: Language(lang) for lang in ("en", "ro", "es", "zh", "fa", "hi")}
    results = {}
    best_points = {}
    for a, b in itertools.permutations(languages, 2):
        summary, best_index, points = transfer(languages[a], languages[b])
        results[f"{a}->{b}"] = summary
        best_points[(a, b)] = (best_index, points)
        printable = {k: v for k, v in summary.items() if k != "layers"}
        print(json.dumps(printable, indent=2))

    out = PACKAGE / "results/cross_language_transfer.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    for b, target in languages.items():
        sources = [a for a in languages if a != b]
        fig, axes = plt.subplots(2, 3, figsize=(19, 12.8))
        axes = axes.ravel()
        own_index = target.own_best_index
        own = target.per_layer[own_index]
        panel(axes[0], target.project_own(own_index), target,
              f"own axes: PC{own['pair'][0]+1}+PC{own['pair'][1]+1} "
              f"| layer {int(target.layers[own_index])}\n"
              f"disparity {own['own_disparity']:.3f}")
        for ax, a in zip(axes[1:], sources):
            summary = results[f"{a}->{b}"]
            _, points = best_points[(a, b)]
            pair = summary["source_pair_at_best"]
            panel(ax, points, target,
                  f"{NAMES[a]} axes: PC{pair[0]+1}+PC{pair[1]+1} "
                  f"| layer {summary['best_transfer_layer']}\n"
                  f"disparity {summary['transfer_disparity']:.3f}, "
                  f"corrected p={summary['layer_corrected_p']:.4f}")
        axes[0].legend(loc="lower left", fontsize=8, frameon=False)
        fig.suptitle(f"{NAMES[b]} affective states on native vs foreign circumplex "
                     "axes (all-states PCA; hue = Russell angle)", fontsize=13, y=1.0)
        fig.tight_layout()
        path = PACKAGE / f"figures/russell_transfer_{b}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
