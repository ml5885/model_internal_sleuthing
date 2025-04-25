#!/usr/bin/env bash
set -euo pipefail

DATASET="ud_gum_dataset"

# get all model keys from src/config.py
MODELS=$(python3 - << 'PYCODE'
import src.config as c
print(" ".join(c.MODEL_CONFIGS.keys()))
PYCODE
)

for MODEL in $MODELS; do
    # if any probe directory matching the pattern exists, skip
    PATTERN1="output/probes/${DATASET}_${MODEL}_lexeme_$0"
    PATTERN2="output/probes/${DATASET}_${MODEL}_inflection_$0"

    if [ -d "$PATTERN1" ] || [ -d "$PATTERN2" ]; then
        echo "Skipping ${MODEL}: probe output already exists."
        continue
    fi

    echo "Running pipeline for ${MODEL}..."
    python3 -u -m src.experiment --model "$MODEL" --dataset "$DATASET" --no_analysis --probe_type "$0"
    echo "Completed ${MODEL}"
done

echo "Done with all models."
