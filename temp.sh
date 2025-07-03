#!/bin/bash
#
# This script automatically finds all steering experiment results and generates
# plots for each unique experiment configuration (model, dataset, probe_type).

set -euo pipefail

STEERING_OUTPUT_DIR="output/steering"

if [ ! -d "$STEERING_OUTPUT_DIR" ]; then
  echo "Error: Steering output directory '$STEERING_OUTPUT_DIR' not found."
  echo "Please ensure your steering results are located there."
  exit 1
fi

echo "Searching for steering results in: $STEERING_OUTPUT_DIR"

# Find unique experiment prefixes (e.g., ud_gum_dataset_qwen2_mlp)
# by listing all directories, removing the lambda suffix, and finding unique names.
find "$STEERING_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_lambda*' | \
  sed -E 's/^(.*)_lambda.*$/\1/' | \
  sort -u | \
  while read -r prefix; do
    # For each unique prefix, find the first corresponding directory to pass to the plot script.
    # The plot script will then find all other sibling lambda directories.
    first_dir=$(find "$STEERING_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "$(basename "$prefix")_lambda*" | head -n 1)
    
    if [ -n "$first_dir" ]; then
      echo "--------------------------------------------------"
      echo "Plotting results for: $(basename "$prefix")"
      echo "Using directory for plotting: $first_dir"
      echo "--------------------------------------------------"
      
      echo "python plots/plot_steering_results.py --results_dir ${first_dir}"
      echo "Done plotting for $(basename "$prefix")."
      echo
    fi
  done

echo "All plotting tasks are complete."