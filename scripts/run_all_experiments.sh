#!/usr/bin/env bash
set -euo pipefail

DATASET="ud_gum_dataset"
MODELS="gpt2"  # you can add more here

for MODEL in $MODELS; do
    PROBE_TYPE="$1"
    PCA_DIM="$2"
    PCA_SUFFIX=""
    if [ "$PCA_DIM" -gt 0 ]; then
        PCA_SUFFIX="_pca$PCA_DIM"
    fi

    OUT_LEX="output/probes/${DATASET}_${MODEL}_lexeme_${PROBE_TYPE}${PCA_SUFFIX}"
    OUT_INF="output/probes/${DATASET}_${MODEL}_inflection_${PROBE_TYPE}${PCA_SUFFIX}"
    if [ -d "$OUT_LEX" ] || [ -d "$OUT_INF" ]; then
        echo "Skipping $MODEL/$PROBE_TYPE: already done."
        continue
    fi

    echo "Running pipeline for $MODEL ($PROBE_TYPE, pca=$PCA_DIM)"
    python3 -u -m src.experiment \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --probe_type "$PROBE_TYPE" \
        --pca_dim "$PCA_DIM" \
        --no_analysis
done

echo "All done."
