#!/usr/bin/env bash
set -euo pipefail

DATASET="ud_gum_dataset"
MODELS="gpt2 qwen2-instruct qwen2 gemma2b bert-base-uncased bert-large-uncased deberta-v3-large"
PROBE_TYPES=("reg" "nn")

for MODEL in $MODELS; do
    PCA_DIM="$1"
    PCA_SUFFIX=""
    if [ "$PCA_DIM" -gt 0 ]; then
        PCA_SUFFIX="_pca$PCA_DIM"
    fi

    for PROBE_TYPE in "${PROBE_TYPES[@]}"; do
        for TASK in "lexeme" "inflection"; do
            OUT_DIR="output/probes/${DATASET}_${MODEL}_${TASK}_${PROBE_TYPE}${PCA_SUFFIX}"
            # if [ -d "$OUT_DIR" ]; then
            #     echo "Skipping $MODEL/$PROBE_TYPE/$TASK: already done."
            #     continue
            # fi

            echo "Running pipeline for $MODEL ($PROBE_TYPE, $TASK, pca=$PCA_DIM)"
            CMD="python3 -u -m src.experiment \
                --model \"$MODEL\" \
                --dataset \"$DATASET\" \
                --probe_type \"$PROBE_TYPE\" \
                --experiment \"$TASK\""
            if [ "$PCA_DIM" -gt 0 ]; then
                CMD="$CMD --pca_dim \"$PCA_DIM\""
            fi
            CMD="$CMD --no_analysis"
            eval "$CMD"

            PRED_CSV=\"$OUT_DIR/predictions.csv\"
            if [ ! -f $PRED_CSV ]; then
                echo "Warning: predictions.csv not found in $OUT_DIR"
            fi
        done
    done
done

echo "All done."
