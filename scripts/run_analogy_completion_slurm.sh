#!/bin/bash
#SBATCH --job-name=llm_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%j.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%j.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=64G

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

# Do not restrict CUDA_VISIBLE_DEVICES so both GPUs are available
# export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

set -e

OUTDIR="notebooks/figures5"
MODELS="EleutherAI/pythia-6.9b allenai/open-instruct-pythia-6.9b-tulu allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct"

python notebooks/sanitycheck.py --models $MODELS --outdir $OUTDIR
