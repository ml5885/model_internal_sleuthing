#!/bin/bash
#SBATCH --job-name=build_ml_data
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A.err
#SBATCH --partition=general
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

# Build all multilingual edge datasets (zh/tr/fr/ru/de) with the fixed,
# token-aligned builder. Compute nodes have internet for the source downloads.
export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_OFFLINE=0
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
eval "$(conda shell.bash hook)"; conda activate llm_probing
cd /home/ml6/model_internal_sleuthing
python -u data_utils/create_multilingual_edge_datasets.py --splits train
echo "=== BUILD EXIT $? ==="
