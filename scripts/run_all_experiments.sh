#!/usr/bin/env bash
set -euo pipefail

DATASET="ud_gum_dataset"
MODELS=(gpt2 qwen2 gemma2b)

for MODEL in "${MODELS[@]}"; do
    REPS="output/${MODEL}_${DATASET}_reps.npz"
    ANALYSIS_DIR="output/${MODEL}_${DATASET}_analysis"

    if [[ -f "$REPS" && -d "$ANALYSIS_DIR" ]]; then
        echo "Skipping ${MODEL}: outputs already exist."
        continue
    fi

    echo "Running pipeline for ${MODEL}..."

    python3 -u -m src.experiment --model "$MODEL" --dataset "$DATASET"

    echo "Completed ${MODEL}"
done

echo "Done with all models."
