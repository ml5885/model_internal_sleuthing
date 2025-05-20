#!/bin/bash
#SBATCH --job-name=analogy_large_models
#SBATCH --output=logs/analogy_large_models_%j.out
#SBATCH --error=logs/analogy_large_models_%j.err
#SBATCH --partition=compute
#SBATCH --gres=L40S:2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p logs
mkdir -p /data/user_data/ml6/.hf_cache

export CUDA_VISIBLE_DEVICES=0,1

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /Users/michaelli/Desktop/Research/lexeme-inflection-probing

set -e

OUTDIR="notebooks/figures5"
MODELS="bert-base-uncased bert-large-uncased deberta-v3-large gpt2 gpt2-large gpt2-xl pythia-6.9b pythia-6.9b-tulu olmo2-7b olmo2-7b-instruct gemma2b gemma2b-it qwen2 qwen2-instruct llama3-8b llama3-8b-instruct"

python notebooks/sanitycheck.py --models $MODELS --outdir $OUTDIR
