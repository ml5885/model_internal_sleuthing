#!/bin/bash

STEERING_DIR="../output/steering"
OUTPUT_DIR="../output/plots"
DATASET="ud_gum_dataset"
PLOT_SCRIPT="../plots/plot_steering_results.py"

mkdir -p "$OUTPUT_DIR"

# Extract unique model names from steering output directories
MODELS=$(ls "$STEERING_DIR" | grep "^${DATASET}_" | sed -E "s/^${DATASET}_([^_]+).*$/\1/" | sort | uniq)

echo "Plotting steering results for models: $MODELS"
python "$PLOT_SCRIPT" \
    --steering_dir "$STEERING_DIR" \
    --models $MODELS \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR"
        --output_dir "$OUTPUT_DIR"
done
