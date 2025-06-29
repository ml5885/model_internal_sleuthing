#!/usr/bin/env bash
set -euo pipefail

PCA_DIM="${1:-0}"
PCA_SUFFIX=""
if [ "$PCA_DIM" -gt 0 ]; then
  PCA_SUFFIX="_pca${PCA_DIM}"
fi

DATASETS=(
    "ud_gum_dataset"
    "ud_zh_gsd_dataset"
    "ud_de_gsd_dataset"
    "ud_fr_gsd_dataset"
    "ud_ru_syntagrus_dataset"
    "ud_tr_imst_dataset"
)
MODELS=(
    "goldfish_eng_latn_1000mb"
    "goldfish_zho_hans_1000mb"
    "goldfish_deu_latn_1000mb"
    "goldfish_fra_latn_1000mb"
    "goldfish_rus_cyrl_1000mb"
    "goldfish_tur_latn_1000mb"
)
PROBE_TYPES=(
    "reg"
    "nn"
)
TASKS=(
    "lexeme"
    # "inflection"
)

for idx in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx]}"
  MODEL="${MODELS[$idx]}"
  for PROBE_TYPE in "${PROBE_TYPES[@]}"; do
    for TASK in "${TASKS[@]}"; do
      OUT_DIR="output/probes/${DATASET}_${MODEL##*/}_${TASK}_${PROBE_TYPE}${PCA_SUFFIX}"
      CMD="python3 -u -m src.experiment --model \"$MODEL\" --dataset \"$DATASET\" --probe_type \"$PROBE_TYPE\" --experiment \"$TASK\""
      if [ "$PCA_DIM" -gt 0 ]; then
        CMD="$CMD --pca_dim $PCA_DIM"
      fi
      CMD="$CMD --no_analysis"
      eval "$CMD"
      if [ ! -f "$OUT_DIR/predictions.csv" ]; then
        echo "Warning: predictions.csv not found in $OUT_DIR"
      fi
    done
  done
done

echo "All done."
