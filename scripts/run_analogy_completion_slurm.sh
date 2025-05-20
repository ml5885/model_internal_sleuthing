#!/bin/bash
#SBATCH --job-name=llm_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%j.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%j.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p logs
mkdir -p /data/user_data/ml6/.hf_cache

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /Users/michaelli/Desktop/Research/lexeme-inflection-probing

set -e

OUTDIR="notebooks/figures5"
MODELS="bert-base-uncased bert-large-uncased deberta-v3-large gpt2 gpt2-large gpt2-xl pythia-6.9b pythia-6.9b-tulu olmo2-7b olmo2-7b-instruct gemma2b gemma2b-it qwen2 qwen2-instruct llama3-8b llama3-8b-instruct"

python notebooks/sanitycheck.py --models $MODELS --outdir $OUTDIR
