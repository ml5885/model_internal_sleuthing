#!/bin/bash
#SBATCH --job-name=test_probe
#SBATCH --output=/home/ml6/logs/sbatch/test_probe_%j.out
#SBATCH --error=/home/ml6/logs/sbatch/test_probe_%j.err
#SBATCH --partition=debug
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G

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

echo "Test script started at $(date)"
echo "Setting up environment..."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_probing

# Change to project directory
cd /home/ml6/lexeme-inflection-probing

echo "Testing Python imports..."
python -c "import torch; import numpy; import pandas; import transformers; print('All imports successful')"

echo "Contents of data directory:"
ls -la data/ud_gum_dataset.csv

MODEL="gpt2"
PROBE="reg"
DATASET="ud_gum_dataset"

echo "Running minimal test experiment with model=${MODEL}, probe_type=${PROBE}"
    
python -u -m src.experiment \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --probe_type "$PROBE" \
    --output_dir "$USER_DATA_OUTPUT" \
    --no_analysis

echo "Test completed at $(date)"
echo "Checking output location:"
find "$USER_DATA_OUTPUT" -name "*.npz" | head -5