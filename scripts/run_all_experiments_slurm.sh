#!/bin/bash
#SBATCH --job-name=nlp_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --array=0-3  # 4 jobs: one per model

# Set up Hugging Face environment variables
export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

# Create output directories
mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache
mkdir -p /data/user_data/ml6/probing_outputs

# Define output directory
USER_DATA_OUTPUT="/data/user_data/ml6/probing_outputs"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_probing

# Change to project directory
cd /home/ml6/lexeme-inflection-probing

# Define experiment configurations with exact model keys from config.py
MODELS=("llama3-8b" "llama3-8b-instruct" "pythia-6.9b" "pythia-6.9b-tulu")
PROBE_TYPES=("reg" "nn")
DATASET="ud_gum_dataset"

# Get model based on array task ID
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Starting experiments for model: $MODEL"

# Run experiments for both probe types
for PROBE in "${PROBE_TYPES[@]}"; do
    echo "Running experiment with model=${MODEL}, probe_type=${PROBE}"
    
    python -u -m src.experiment \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --probe_type "$PROBE" \
        --output_dir "$USER_DATA_OUTPUT" \
        --no_analysis
done

echo "All experiments completed for model ${MODEL}"