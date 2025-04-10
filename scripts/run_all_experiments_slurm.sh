#!/bin/bash
#SBATCH --job-name=all_experiments
#SBATCH --output=logs/all_experiments_%j.out
#SBATCH --error=logs/all_experiments_%j.err
#SBATCH --partition=swl_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Setup Hugging Face cache directories (adjust if needed)
export HF_HOME=/data/user_data/$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

# Activate the virtual environment
source ~/venv/bin/activate

echo "Starting activation extraction for GPT-2..."
python -m src.activation_extraction --data data/controlled_sentences.csv --output output/gpt2_reps.npz --model gpt2

echo "Running inflection probe experiments (real inflection labels and control experiment) using k-sparse probe (k=10)..."
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task inflection --sparse_k 10 --control_inflection

echo "Running lexeme probe experiments (real lexeme labels and control experiment) using k-sparse probe (k=10)..."
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task lexeme --sparse_k 10 --control_lexeme

echo "Running unsupervised analysis on GPT-2 activations..."
python -m src.unsupervised_analysis --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv

echo "All experiments completed."
