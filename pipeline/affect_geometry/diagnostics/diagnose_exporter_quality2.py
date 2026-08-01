"""Round 2: (F) frozen transfer on the shared 11 labels (does fa still
dominate without the free search?); (G) containment of each target's native
circumplex plane in each source's top-10 PC subspace (basis genericity);
(H) ingredient swap: source scales vs source axes; (I) scale-vector stats."""
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
SHARED = sorted(set.intersection(*[set(v) for v in anchors["languages"].values()]),
                key=lambda l: angle_of[l])
BAND = range(8, 29)


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
    lemmas = ar["lemmas"].astype(str)
    cent = ar["centroids"].astype(np.float64)
    labels_arr = np.asarray([lemma_to_label.get(l, "") for l in lemmas])
    label_list = sorted({l for l in labels_arr if l}, key=lambda l: angle_of[l])
    rad = np.radians([angle_of[l] for l in label_list])
    theory = np.column_stack((np.cos(rad), np.sin(rad)))
    srad = np.radians([angle_of[l] for l in SHARED])
    stheory = np.column_stack((np.cos(srad), np.sin(srad)))
    per_layer = []
    for idx in range(cent.shape[0]):
        scaler = StandardScaler().fit(cent[idx])
        Z = scaler.transform(cent[idx])
        pca = PCA(n_components=min(20, Z.shape[0] - 1), random_state=0).fit(Z)
        P = pca.transform(Z)
        scores = np.asarray([P[labels_arr == lab].mean(0) for lab in label_list])
        _, i, j = search_pair(scores, theory)
        raw_plane = pca.components_[[i, j]] / scaler.scale_
        Qp, _ = np.linalg.qr(raw_plane.T)
        raw10 = pca.components_[:10] / scaler.scale_
        Q10, _ = np.linalg.qr(raw10.T)
        per_layer.append({"scale": scaler.scale_, "comp": pca.components_,
                          "pair": (i, j), "Qp": Qp, "Q10": Q10})
    data[lang] = {"cent": cent, "labels_arr": labels_arr, "theory": theory,
                  "stheory": stheory, "label_list": label_list,
                  "per_layer": per_layer}
    print(f"[loaded {lang}]", flush=True)
n_layers = data["en"]["cent"].shape[0]

# ---------- F: frozen transfer, shared 11 labels ----------
print("\n=== F: FROZEN transfer scored on shared 11 labels (best layer) ===")
print("rows=source basis, cols=target; diag = native plane on shared labels")
print("      " + "  ".join(f"{t:>6}" for t in LANGS))
for src in LANGS:
    row = []
    for tgt in LANGS:
        T = data[tgt]
        best = None
        for idx in range(n_layers):
            info = data[src]["per_layer"][idx]
            raw = T["cent"][idx]
            z = (raw - raw.mean(0)) / info["scale"]
            pts = z @ info["comp"][list(info["pair"])].T
            scores = np.asarray([pts[T["labels_arr"] == lab].mean(0)
                                 for lab in SHARED])
            d = procrustes_disparity(T["stheory"], scores)
            if best is None or d < best:
                best = d
        row.append(best)
    print(f"{src:>5} " + "  ".join(f"{v:.3f}" for v in row))

# ---------- G: containment of target plane in source top-10 ----------
print("\n=== G: containment of target's native plane in source's top-10 PCs ===")
print("mean sq cos over layers 8-28 (1.0 = fully contained; random ~ 10/2560)")
print("      " + "  ".join(f"{t:>6}" for t in LANGS))
for src in LANGS:
    row = []
    for tgt in LANGS:
        vals = []
        for idx in BAND:
            s = np.linalg.svd(data[src]["per_layer"][idx]["Q10"].T
                              @ data[tgt]["per_layer"][idx]["Qp"],
                              compute_uv=False)
            vals.append(float(np.mean(s ** 2)))
        row.append(np.mean(vals))
    print(f"{src:>5} " + "  ".join(f"{v:.3f}" for v in row))

# ---------- H: ingredient swap ----------
print("\n=== H: which ingredient carries transfer? (best-layer D, full labels) ===")
print("variants: std = src scales + src axes | tgtscale = tgt scales + src axes"
      " | noscale = no standardization + src axes")
for src in ("fa", "zh", "en"):
    for tgt in LANGS:
        if tgt == src:
            continue
        T = data[tgt]
        best = {"std": None, "tgtscale": None, "noscale": None}
        for idx in range(n_layers):
            info = data[src]["per_layer"][idx]
            raw = T["cent"][idx]
            centered = raw - raw.mean(0)
            tscale = data[tgt]["per_layer"][idx]["scale"]
            variants = {
                "std": (centered / info["scale"]) @ info["comp"][list(info["pair"])].T,
                "tgtscale": (centered / tscale) @ info["comp"][list(info["pair"])].T,
                "noscale": centered @ info["comp"][list(info["pair"])].T,
            }
            for name, pts in variants.items():
                scores = np.asarray([pts[T["labels_arr"] == lab].mean(0)
                                     for lab in T["label_list"]])
                d = procrustes_disparity(T["theory"], scores)
                if best[name] is None or d < best[name]:
                    best[name] = d
        print(f"{src}->{tgt}: std {best['std']:.3f} | tgtscale "
              f"{best['tgtscale']:.3f} | noscale {best['noscale']:.3f}")

# ---------- I: scale-vector stats at L16 ----------
print("\n=== I: per-dimension scale vectors, layer 16 ===")
idx = 16
scales = {lang: data[lang]["per_layer"][idx]["scale"] for lang in LANGS}
print("pairwise corr of log-scales:")
for a, b in itertools.combinations(LANGS, 2):
    c = np.corrcoef(np.log(scales[a]), np.log(scales[b]))[0, 1]
    print(f"  {a}-{b}: {c:.3f}")
for lang in LANGS:
    s = scales[lang]
    print(f"  {lang}: max/median scale ratio {np.max(s)/np.median(s):.1f}, "
          f"kurtosis {float(((s-s.mean())**4).mean()/(s.var()**2)):.1f}")
print("done")
