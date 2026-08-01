"""Why does the Persian basis transfer so well? Diagnostics.

A: additive row/column decomposition of the 6x6 transfer matrix
C: consensus-plane analysis: principal-angle affinity between languages'
   circumplex planes at matched layers (is fa the medoid?)
D: spectrum diagnostics: variance concentration, chosen-pair depth,
   between-anchor-label variance captured by top-10 PCs
E: anchor coherence: within-label vs cross-label cosine of anchor centroids
"""
import sys, json, itertools, collections
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
transfer = json.loads((RESULTS_DIR / "cross_language_transfer.json").read_text())

# ---------- A: additive decomposition ----------
print("=== A: row/column decomposition of transfer matrix ===")
M = {}
for a, b in itertools.permutations(LANGS, 2):
    M[(a, b)] = transfer[f"{a}->{b}"]["transfer_disparity"]
grand = np.mean(list(M.values()))
row = {a: np.mean([M[(a, b)] for b in LANGS if b != a]) - grand for a in LANGS}
col = {b: np.mean([M[(a, b)] for a in LANGS if a != b]) - grand for b in LANGS}
resid = {k: M[k] - grand - row[k[0]] - col[k[1]] for k in M}
print("grand mean D = %.3f" % grand)
print("source (row) effects:", {a: round(row[a], 3) for a in LANGS})
print("target (col) effects:", {b: round(col[b], 3) for b in LANGS})
ss_tot = sum((v - grand) ** 2 for v in M.values())
ss_row = sum(5 * row[a] ** 2 for a in LANGS)
ss_col = sum(5 * col[b] ** 2 for b in LANGS)
ss_res = sum(v ** 2 for v in resid.values())
print("variance explained: rows %.0f%%, cols %.0f%%, resid %.0f%%"
      % (100 * ss_row / ss_tot, 100 * ss_col / ss_tot, 100 * ss_res / ss_tot))
big = sorted(resid.items(), key=lambda kv: -abs(kv[1]))[:5]
print("largest residuals:", [(f"{a}->{b}", round(r, 3)) for (a, b), r in big])

# ---------- B: per-language per-layer PCA ----------
def search_pair(scores, theory, width=10):
    best = None
    for i, j in itertools.combinations(range(min(width, scores.shape[1])), 2):
        d = procrustes_disparity(theory, scores[:, [i, j]])
        if best is None or (d, i, j) < best:
            best = (d, i, j)
    return best

data = {}
for lang in LANGS:
    lemma_to_label = {}
    for label, lemma_list in anchors["languages"][lang].items():
        for lemma in lemma_list:
            lemma_to_label[lemma] = label
    ar = np.load(ARCHIVES[lang])
    layers = ar["layers"].astype(int)
    lemmas = ar["lemmas"].astype(str)
    cent = ar["centroids"].astype(np.float64)
    labels_arr = np.asarray([lemma_to_label.get(l, "") for l in lemmas])
    amask = labels_arr != ""
    label_list = sorted({l for l in labels_arr if l}, key=lambda l: angle_of[l])
    rad = np.radians([angle_of[l] for l in label_list])
    theory = np.column_stack((np.cos(rad), np.sin(rad)))
    per_layer = []
    for idx in range(len(layers)):
        scaler = StandardScaler().fit(cent[idx])
        Z = scaler.transform(cent[idx])
        pca = PCA(n_components=min(20, Z.shape[0] - 1), random_state=0).fit(Z)
        P = pca.transform(Z)
        scores = np.asarray([P[labels_arr == lab].mean(0) for lab in label_list])
        d, i, j = search_pair(scores, theory)
        raw = pca.components_[[i, j]] / scaler.scale_
        Q, _ = np.linalg.qr(raw.T)  # 2560 x 2 orthonormal
        per_layer.append({"pair": (i, j), "own_d": d, "Q": Q,
                          "evr": pca.explained_variance_ratio_,
                          "Z": Z if idx in (12, 16, 24) else None,
                          "pca": pca if idx in (12, 16, 24) else None})
    data[lang] = {"layers": layers, "labels_arr": labels_arr, "amask": amask,
                  "label_list": label_list, "theory": theory,
                  "per_layer": per_layer, "cent": cent}
    print(f"[loaded {lang}]", flush=True)

# ---------- C: consensus-plane affinity ----------
print("\n=== C: plane affinity across languages (mean sq cos of principal angles) ===")
BAND = range(8, 29)
aff = {(a, b): [] for a, b in itertools.combinations(LANGS, 2)}
for idx in BAND:
    for a, b in itertools.combinations(LANGS, 2):
        s = np.linalg.svd(data[a]["per_layer"][idx]["Q"].T
                          @ data[b]["per_layer"][idx]["Q"], compute_uv=False)
        aff[(a, b)].append(float(np.mean(s ** 2)))
mean_aff = {}
for a in LANGS:
    vals = []
    for b in LANGS:
        if a == b:
            continue
        key = (a, b) if (a, b) in aff else (b, a)
        vals.append(np.mean(aff[key]))
    mean_aff[a] = np.mean(vals)
print("random-plane baseline ~ 2/2560 = %.4f" % (2 / 2560))
print("mean affinity to other languages' planes, layers 8-28:")
for a in sorted(mean_aff, key=lambda x: -mean_aff[x]):
    print("  %s: %.3f" % (a, mean_aff[a]))
print("pairwise (band mean):")
for a, b in itertools.combinations(LANGS, 2):
    print("  %s-%s: %.3f" % (a, b, np.mean(aff[(a, b)])))

# ---------- D: spectrum diagnostics ----------
print("\n=== D: spectrum + anchor-signal concentration (layers 12/16/24) ===")
for lang in LANGS:
    D = data[lang]
    for idx in (12, 16, 24):
        pl = D["per_layer"][idx]
        Z, pca = pl["Z"], pl["pca"]
        sv = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        lam = sv ** 2
        pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        top10 = float(lam[:10].sum() / lam.sum())
        # between-anchor-label variance captured by top-10 PC subspace
        amask, labels_arr = D["amask"], D["labels_arr"]
        means = np.asarray([Z[labels_arr == lab].mean(0) for lab in D["label_list"]])
        means = means - means.mean(0)
        V = pca.components_[:10]
        cap10 = float(np.square(means @ V.T).sum() / np.square(means).sum())
        Vp = pca.components_[list(pl["pair"])]
        cap_pair = float(np.square(means @ Vp.T).sum() / np.square(means).sum())
        print("%s L%d: PR=%5.1f top10evr=%.2f | anchor-label var in top10=%.2f "
              "in chosen pair=%.2f | pair=PC%d+PC%d ownD=%.3f"
              % (lang, idx, pr, top10, cap10, cap_pair,
                 pl["pair"][0] + 1, pl["pair"][1] + 1, pl["own_d"]))
# chosen-pair depth over the band
print("chosen pair indices, layers 8-28 (mode):")
for lang in LANGS:
    pairs = [data[lang]["per_layer"][i]["pair"] for i in BAND]
    c = collections.Counter(pairs)
    print("  %s: %s" % (lang, c.most_common(4)))

# ---------- E: anchor coherence at layer 16 ----------
print("\n=== E: anchor coherence at layer 16 (cosine of standardized centroids) ===")
for lang in LANGS:
    D = data[lang]
    idx = 16
    scaler = StandardScaler().fit(D["cent"][idx])
    Z = scaler.transform(D["cent"][idx])
    Z = Z - Z.mean(0)
    A = Z[D["amask"]]
    labs = D["labels_arr"][D["amask"]]
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    cos = An @ An.T
    same, diff = [], []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            (same if labs[i] == labs[j] else diff).append(cos[i, j])
    print("  %s: within-label %.3f (n=%d)  cross-label %.3f  gap %.3f"
          % (lang, np.mean(same) if same else float("nan"), len(same),
             np.mean(diff), (np.mean(same) - np.mean(diff)) if same else float("nan")))
print("done")
