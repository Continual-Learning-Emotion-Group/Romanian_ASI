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

## Exact rerun

From the repository root, build the Romanian manifest locally:

```bash
python -m pipeline.affect_geometry.prepare \
  --language ro \
  --source pipeline/data/benchmark_ro_asi_clean.jsonl \
  --output pipeline/affect_geometry/artifacts/manifests/ro.jsonl \
  --summary pipeline/affect_geometry/results/selection_ro.json
```

On `tigerfish`, build each MASIVE manifest directly from the original corpus:

```bash
python -m pipeline.affect_geometry.prepare \
  --language en \
  --source /mnt/swordfish-pool2/eturcan-ndeas/MASIVE/masive \
  --output pipeline/affect_geometry/artifacts/manifests/en.jsonl \
  --summary pipeline/affect_geometry/results/selection_en.json
```

Replace `en` with `es` for Spanish. Extract one language on an unoccupied GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipeline.affect_geometry.extract \
  --manifest pipeline/affect_geometry/artifacts/manifests/en.jsonl \
  --output pipeline/affect_geometry/artifacts/hidden/en.npz \
  --run-metadata pipeline/affect_geometry/artifacts/hidden/en_run.json \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --batch-size 32 \
  --maximum-tokens 512
```

After the three centroid archives are available locally, regenerate all numerical
results and figures:

```bash
for language in ro en es; do
  python -m pipeline.affect_geometry.analyze \
    --hidden pipeline/affect_geometry/artifacts/hidden/$language.npz \
    --language $language \
    --output pipeline/affect_geometry/results/metrics_$language.json \
    --projection-output pipeline/affect_geometry/results/projections_$language.json
done

python -m pipeline.affect_geometry.plot \
  --results-dir pipeline/affect_geometry/results \
  --output-dir pipeline/affect_geometry/figures
```
