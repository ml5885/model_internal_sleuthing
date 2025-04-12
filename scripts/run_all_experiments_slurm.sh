#!/bin/bash
#SBATCH --job-name=all_gpt2_experiments
#SBATCH --output=logs/all_gpt2_experiments_%j.out
#SBATCH --error=logs/all_gpt2_experiments_%j.err
#SBATCH --partition=swl_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Set Hugging Face environment variables as needed.
export HF_HOME=/data/user_data/$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

# Activate the virtual environment.
source ~/venv/bin/activate

echo "=================================================="
echo "Starting activation extraction using forward hooks for GPT-2..."
echo "=================================================="
python -m src.activation_extraction --data data/controlled_sentences.csv --output output/gpt2_reps.npz --model gpt2

echo "=================================================="
echo "Running dense inflection probe (real labels) with control experiment..."
echo "=================================================="
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task inflection --sparse_k 0 --control_inflection

echo "=================================================="
echo "Running sparse (k=10) inflection probe with control experiment..."
echo "=================================================="
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task inflection --sparse_k 10 --control_inflection

echo "=================================================="
echo "Running dense lexeme probe (real labels) with control experiment..."
echo "=================================================="
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task lexeme --sparse_k 0 --control_lexeme

echo "=================================================="
echo "Running sparse (k=10) lexeme probe with control experiment..."
echo "=================================================="
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task lexeme --sparse_k 10 --control_lexeme

echo "=================================================="
echo "Running combined dense probe (both tasks) with control experiments..."
echo "=================================================="
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task both --sparse_k 0 --control_inflection --control_lexeme

echo "=================================================="
echo "Running combined sparse (k=10) probe (both tasks) with control experiments..."
echo "=================================================="
python -m src.probe_training --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task both --sparse_k 10 --control_inflection --control_lexeme

echo "=================================================="
echo "Running unsupervised analysis on GPT-2 activations..."
echo "=================================================="
python -m src.unsupervised_analysis --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv

echo "=================================================="
echo "All GPT-2 experiments completed."
echo "=================================================="
