#!/usr/bin/env bash
set -euo pipefail

DATASET="ud_gum_dataset"

# hardcoded model keys
# MODELS="gpt2 qwen2-instruct qwen2 gemma2b pythia1.4b bert-base-uncased bert-large-uncased distilbert-base-uncased deberta-v3-large"
MODELS="gpt2"
for MODEL in $MODELS; do
    # if any probe directory matching the pattern exists, skip
    PATTERN1="output/probes/${DATASET}_${MODEL}_lexeme_$1_$2"
    PATTERN2="output/probes/${DATASET}_${MODEL}_inflection_$1_$2"

    if [ -d "$PATTERN1" ] || [ -d "$PATTERN2" ]; then
        echo "Skipping ${MODEL}: probe output already exists."
        continue
    fi

    echo "Running pipeline for ${MODEL}..."
    python3 -u -m src.experiment --model "$MODEL" --dataset "$DATASET" --no_analysis --probe_type "$1" --pca_dim "$2"
    echo "Completed ${MODEL}"
done

echo "Done with all models."
