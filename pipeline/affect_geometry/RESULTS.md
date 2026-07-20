# Affective-State Geometry v1 Results

## Data and method

The experiment uses only the collected Romanian ASI benchmark and the original
English/Spanish MASIVE splits. The primary comparison is restricted to adjective
states. The full manifests contain 98 Romanian, 503 English, and 251 Spanish lemmas;
the nested basic-emotion slices contain 27, 33, and 23 lemmas, respectively.

Qwen3.5-4B target-word centroids were extracted at all 33 hidden states. At each
layer, PCA was fit only to the basic lemma centroids. Two analyses were frozen:

1. **PC1+PC2:** use the two dominant unsupervised basic-slice components and select
   the layer with the lowest basic-slice circumplex disparity.
2. **Searched pair:** search all 45 pairs among the first 10 PCs at every layer,
   select the lowest basic-slice disparity, then freeze that plane.

Both p-values are global permutation tests. The first repeats layer selection; the
second repeats both layer and PC-pair selection. Broader states are never used to
select a layer or plane.

## Main results

| Language | PC1+PC2 best layer | Disparity | Corrected p | Broader variance | Best searched pair | Disparity | Corrected p | Broader variance |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Romanian | 18 | 0.272 | 0.0116 | 7.26% | L4, PC2+PC4 | 0.225 | 0.119 | 1.92% |
| English | 18 | 0.452 | 0.1020 | 3.95% | L32, PC1+PC3 | 0.236 | 0.123 | 3.15% |
| Spanish | 31 | 0.211 | 0.0024 | 5.02% | L31, PC1+PC2 | 0.211 | 0.098 | 5.02% |

### Interpretation

- The dominant two basic-emotion PCs match the fixed circumplex significantly in
  Romanian and Spanish, but not English, after correcting for layer selection.
- Searching additional PC pairs reduces the observed disparities, but none of the
  three searched fits survives correction for searching 45 pairs across 33 layers.
  The searched result therefore should not be presented as stronger circumplex
  evidence than the PC1+PC2 result.
- The successful PC1+PC2 planes explain a meaningful but minority share of the
  broader geometry: 7.26% in Romanian and 5.02% in Spanish. English PC1+PC2 explains
  3.95%, but its anchor fit is not significant.
- These fractions are substantially above an isotropic random two-dimensional plane:
  approximately 93x for Romanian, 51x for English, and 64x for Spanish. Thus the
  planes are unusually information-dense while remaining far from a complete account
  of the broader affective-state space.
- In the selected PC1+PC2 plane, most English and Spanish broader-state centroids lie
  inside the basic-emotion circle (96.6% and 96.1%). Romanian broader states are more
  radially dispersed, with 53.5% inside the basic circle.

## Figures

- `figures/figure_pc1_pc2_projection.{png,pdf}`: primary unsupervised result.
- `figures/figure_searched_circumplex_projection.{png,pdf}`: exploratory searched-pair result.
- `figures/figure_layer_sweep.{png,pdf}`: basic-slice theory fit, non-basic variance
  captured, and non-basic radial deviation from the basic-emotion ring at every layer.
  Procrustes disparity is only defined for the basic slice because the non-basic
  states have no fixed theoretical valence/arousal targets.
- `figures/table_geometry_summary.csv`: machine-readable headline values.

## Limitations

Romanian trust, English disgust, and Spanish surprise have thin lexical coverage.
MASIVE contains noisier state labels than the LLM-filtered Romanian benchmark. The
broader states have no assigned valence/arousal coordinates, so their projected
distribution supports geometric claims only; it does not validate the semantic
location of individual broader-state words.
