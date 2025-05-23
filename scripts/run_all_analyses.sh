#!/usr/bin/env bash
set -euo pipefail

# List of models to run
models=(
    gpt2
    qwen2
    gemma2b
    bert-base-uncased
    pythia1.4b
)

dataset="ud_gum_dataset"
labels="data/ud_gum_dataset.csv"

# Ensure output directory exists
mkdir -p output

for model in "${models[@]}"; do
    analysis_dir="output/${model}_${dataset}_analysis"
    activations_dir="output/${model}_${dataset}_reps"
    if [ -d "${analysis_dir}" ]; then
        echo "Skipping '${model}' as analysis folder '${analysis_dir}' already exists."
        continue
    fi
    echo "Running analysis for model '${model}'..."
    python -m src.analysis \
        --model "${model}" \
        --dataset "${dataset}" \
        --activations-dir "${activations_dir}" \
        --labels "${labels}"
done

echo "All models processed."
