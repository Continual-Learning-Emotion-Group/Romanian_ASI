# Affective-State Geometry

This package is an independent replacement for the exploratory `viz_hidden`
experiments. It uses only the collected Romanian ASI benchmark and the original
English/Spanish MASIVE corpora.

## Frozen design

- The basic-emotion rows are a nested subset of each language's broader manifest.
- Occurrences are capped per lemma; PCA is fit to lemma centroids, so frequent words
  and gender variants do not receive extra weight.
- All 33 Qwen3.5-4B hidden states are extracted at the target word span.
- At each layer, the confirmatory plane is PC1+PC2.
- A second analysis exhaustively searches pairs among the first 10 PCs for the pair
  with the lowest Procrustes disparity against the fixed eight-emotion theoretical
  coordinates in `anchors.json`.
- Axis-pair and layer selection use only the basic slice. All broader-state metrics
  are computed after selection is frozen.
- A global permutation test repeats both PC-pair and layer selection, preventing the
  searched basic-slice fit from being interpreted as an uncorrected test.

The broader states have no assigned valence/arousal labels. Their analysis therefore
measures geometry: variance captured by the frozen plane, radial distribution,
angular coverage, and position relative to the basic-emotion circle.

## Outputs

Large reproducible manifests and centroid archives live under `artifacts/` and are
ignored by Git. Selection summaries, numerical results, and paper figures are kept
under `results/` and `figures/`.

Run individual stages with:

```bash
python -m pipeline.affect_geometry.prepare --help
python -m pipeline.affect_geometry.extract --help
python -m pipeline.affect_geometry.analyze --help
python -m pipeline.affect_geometry.plot --help
```
