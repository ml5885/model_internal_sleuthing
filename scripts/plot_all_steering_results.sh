#!/bin/bash

STEERING_DIR="../output/steering"
OUTPUT_DIR="../output/plots"
DATASET="ud_gum_dataset"
PLOT_SCRIPT="../plots/plot_steering_results.py"

mkdir -p "$OUTPUT_DIR"

MODELS=(
    "bert-base-uncased"
    "bert-large-uncased"
    "deberta-v3-large"
    "gemma2b"
    "gemma2b-it"
    "gpt2"
    "gpt2-large"
    "gpt2-xl"
    "llama3-8b"
    "llama3-8b-instruct"
    "olmo2-7b"
    "olmo2-7b-instruct"
    "pythia-6.9b"
    "pythia-6.9b-tulu"
    "qwen2"
    "qwen2-instruct"
    "qwen2.5-7B"
    "qwen2.5-7B-instruct"
)

echo "Plotting steering results for models: ${MODELS[*]}"
python "$PLOT_SCRIPT" \
    --steering_dir "$STEERING_DIR" \
    --models "${MODELS[@]}" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR"
