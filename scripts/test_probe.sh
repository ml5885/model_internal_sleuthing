#!/bin/bash
#SBATCH --job-name=test_gpt2
#SBATCH --output=/home/ml6/logs/sbatch/test_gpt2_%j.out
#SBATCH --error=/home/ml6/logs/sbatch/test_gpt2_%j.err
#SBATCH --partition=debug
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G

# Setup directories and variables
mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache
mkdir -p /data/user_data/ml6/probing_outputs

# HuggingFace environment
export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

# Activate conda
eval "$(conda shell.bash hook)"
conda activate llm_probing

# Run experiment
cd /home/ml6/lexeme-inflection-probing
python -u -m src.experiment \
    --model "gpt2" \
    --dataset "ud_gum_dataset" \
    --probe_type "reg" \
    --output_dir "/data/user_data/ml6/probing_outputs" \
    --no_analysis
