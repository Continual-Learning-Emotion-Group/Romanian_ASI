"""Count-matched control: does the en/zh basis degrade to (or below) fa's
transfer quality when its broader vocabulary is subsampled to fa's size (204
broader lemmas, anchors kept)? If en@204 stays at en-full level, vocabulary
size does not explain Persian's edge; if en@204 ~ fa, it does."""
import sys, json, itertools
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/alexjerpelea/Romanian_ASI')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pipeline.affect_geometry.analyze import procrustes_disparity

PACKAGE = Path('/Users/alexjerpelea/Romanian_ASI/pipeline/affect_geometry')
from pipeline.affect_geometry.common import model_paths
HIDDEN_DIR, RESULTS_DIR, FIGURES_DIR = model_paths(PACKAGE)
ARCHIVES = {
    "ro": HIDDEN_DIR / "ro_russell.npz",
    "en": HIDDEN_DIR / "en.npz",
    "es": HIDDEN_DIR / "es.npz",
    "zh": HIDDEN_DIR / "zh.npz",
    "fa": HIDDEN_DIR / "fa.npz",
    "hi": HIDDEN_DIR / "hi.npz",
}
LANGS = ["en", "es", "zh", "ro", "fa", "hi"]
anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
angle_of = anchors["angles_degrees"]
TARGET_BROADER = 204  # fa's broader count
SEEDS = (0, 1, 2)


def search_pair(scores, theory, width=10):
    best = None
    for i, j in itertools.combinations(range(min(width, scores.shape[1])), 2):
        d = procrustes_disparity(theory, scores[:, [i, j]])
        if best is None or (d, i, j) < best:
            best = (d, i, j)
    return best


def load(lang):
    lemma_to_label = {}
    for label, lemma_list in anchors["languages"][lang].items():
        for lemma in lemma_list:
            lemma_to_label[lemma] = label
    ar = np.load(ARCHIVES[lang])
    labels_arr = np.asarray([lemma_to_label.get(l, "")
                             for l in ar["lemmas"].astype(str)])
    label_list = sorted({l for l in labels_arr if l}, key=lambda l: angle_of[l])
    rad = np.radians([angle_of[l] for l in label_list])
    return {"layers": ar["layers"].astype(int),
            "cent": ar["centroids"].astype(np.float64),
            "labels_arr": labels_arr, "amask": labels_arr != "",
            "label_list": label_list,
            "theory": np.column_stack((np.cos(rad), np.sin(rad)))}


data = {lang: load(lang) for lang in LANGS}
n_layers = len(data["en"]["layers"])


def source_planes(src, keep):
    """Per-layer (scale, components, pair) for source restricted to `keep`."""
    D = data[src]
    planes = []
    for idx in range(n_layers):
        sub = D["cent"][idx][keep]
        scaler = StandardScaler().fit(sub)
        Z = scaler.transform(sub)
        pca = PCA(n_components=min(20, Z.shape[0] - 1), random_state=0).fit(Z)
        P = pca.transform(Z)
        labs = D["labels_arr"][keep]
        scores = np.asarray([P[labs == lab].mean(0) for lab in D["label_list"]])
        _, i, j = search_pair(scores, D["theory"])
        planes.append((scaler.scale_, pca.components_[[i, j]]))
    return planes


def best_transfer(planes, tgt):
    T = data[tgt]
    best = None
    for idx in range(n_layers):
        scale, comp = planes[idx]
        raw = T["cent"][idx]
        z = (raw - raw.mean(0)) / scale
        pts = z @ comp.T
        scores = np.asarray([pts[T["labels_arr"] == lab].mean(0)
                             for lab in T["label_list"]])
        d = procrustes_disparity(T["theory"], scores)
        if best is None or d < best:
            best = d
    return best


for src in ("en", "zh"):
    D = data[src]
    n_all = len(D["labels_arr"])
    anchor_idx = np.flatnonzero(D["amask"])
    broader_idx = np.flatnonzero(~D["amask"])
    # full basis reference
    full = source_planes(src, np.arange(n_all))
    ref = {t: best_transfer(full, t) for t in LANGS if t != src}
    print(f"{src} full ({len(broader_idx)} broader): " +
          " ".join(f"->{t} {ref[t]:.3f}" for t in ref), flush=True)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub = rng.choice(broader_idx, TARGET_BROADER, replace=False)
        keep = np.sort(np.concatenate([anchor_idx, sub]))
        planes = source_planes(src, keep)
        res = {t: best_transfer(planes, t) for t in LANGS if t != src}
        print(f"{src}@{TARGET_BROADER} seed{seed}: " +
              " ".join(f"->{t} {res[t]:.3f}" for t in res), flush=True)

print("\nfa reference (from stored run): "
      "->en 0.295 ->es 0.244 ->zh 0.295 ->ro 0.261 ->hi 0.350")
print("done")
