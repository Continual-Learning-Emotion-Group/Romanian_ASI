#!/bin/bash
# Run the full Russell-geometry analysis suite for one extraction variant
# over the Final Data 8-language manifests.
# Usage: ./run_final_suite.sh qwen3-8b-final [languages...]
set -e
VARIANT=$1
shift
LANGS=${@:-"en ro es zh fa hi fr id"}
cd "$(dirname "$0")/../.."
source venv/bin/activate
export AFFECT_GEOMETRY_MODEL=$VARIANT
H=pipeline/affect_geometry/artifacts/hidden/$VARIANT
R=pipeline/affect_geometry/results/$VARIANT
mkdir -p "$R"

echo "=== native circumplex per language ==="
for L in $LANGS; do
  echo "--- analyze_russell $L"
  python -m pipeline.affect_geometry.analyze_russell \
    --hidden $H/$L.npz --language $L \
    --output $R/metrics_russell_$L.json \
    --projection-output $R/projections_russell_$L.json
done

echo "=== all-states + broader-only PCA ==="
python -m pipeline.affect_geometry.analyze_all_states $LANGS
python -m pipeline.affect_geometry.analyze_broader_only $LANGS

echo "=== convexity + plane share ==="
python -m pipeline.affect_geometry.analyze_convexity_russell
python -m pipeline.affect_geometry.analyze_plane_share_russell $LANGS

echo "=== cross-language ==="
python -m pipeline.affect_geometry.basis_search_cross_language
python -m pipeline.affect_geometry.transfer_cross_language
python -m pipeline.affect_geometry.transfer_shared_labels
python -m pipeline.affect_geometry.transfer_pairwise

echo "=== figures ==="
python -m pipeline.affect_geometry.plot_russell_geometry $LANGS
python -m pipeline.affect_geometry.plot_all_states
python -m pipeline.affect_geometry.plot_convexity_russell
python -m pipeline.affect_geometry.plot_plane_share_table

echo "SUITE_DONE $VARIANT"
