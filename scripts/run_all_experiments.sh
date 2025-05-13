#!/usr/bin/env bash
set -euo pipefail

DATASET="ud_gum_dataset"
# MODELS="gpt2 gpt2-large gpt2-xl qwen2-instruct qwen2 pythia1.4b gemma2b gemma2b-it bert-base-uncased bert-large-uncased deberta-v3-large"
# MODELS="gpt2-large gpt2-xl"
MODELS="gpt2-large gpt2-xl gemma2b-it"
PROBE_TYPES="nn reg rf"

for MODEL in $MODELS; do
    PCA_DIM="$1"
    PCA_SUFFIX=""
    if [ "$PCA_DIM" -gt 0 ]; then
        PCA_SUFFIX="_pca$PCA_DIM"
    fi

    echo "=== model=${MODEL}, pca_dim=${PCA_DIM} ==="

    for PROBE_TYPE in $PROBE_TYPES; do
        OUT_LEX="output/probes/${DATASET}_${MODEL}_lexeme_${PROBE_TYPE}${PCA_SUFFIX}"
        OUT_INF="output/probes/${DATASET}_${MODEL}_inflection_${PROBE_TYPE}${PCA_SUFFIX}"
        if [ -d "$OUT_LEX" ] || [ -d "$OUT_INF" ]; then
            echo "Skipping $MODEL/$PROBE_TYPE: already done."
            continue
        fi

        echo "Running pipeline for $MODEL ($PROBE_TYPE, pca=$PCA_DIM)"
        CMD="python3 -u -m src.experiment \
            --model \"$MODEL\" \
            --dataset \"$DATASET\" \
            --probe_type \"$PROBE_TYPE\""
        if [ "$PCA_DIM" -gt 0 ]; then
            CMD="$CMD --pca_dim \"$PCA_DIM\""
        fi
        CMD="$CMD --no_analysis"
        eval "$CMD"
    done
done

echo "All done."
