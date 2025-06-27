#!/usr/bin/env bash
set -euo pipefail

MODELS="bert-base-uncased \
bert-large-uncased \
deberta-v3-large \
gpt2 \
gpt2-large \
gpt2-xl \
pythia-6.9b \
pythia-6.9b-tulu \
gemma2b \
gemma2b-it \
qwen2 \
qwen2-instruct \
llama3-8b \
llama3-8b-instruct \
olmo2-7b \
olmo2-7b-instruct"
DATASETS=("de_gsd" "zh_gsd" "fr_gsd" "ru_syntagrus" "tr_imst")
OUT_DIR="src/figures4"

for DATASET in "${DATASETS[@]}"; do
    python3 src/pca_experiment.py \
      --models $MODELS \
      --dataset $DATASET \
      --output-dir $OUT_DIR/$DATASET
done

echo "All intrinsic-dimension analyses complete."
