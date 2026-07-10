#!/bin/bash
#SBATCH --job-name=coref_denoise
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A.err
#SBATCH --partition=general
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G

# Re-run coref cumulative with heavier Monte-Carlo-CV averaging (split + init).
# Coref is small (~1255 examples) so many seeds are cheap.
export HF_HOME=/data/user_data/ml6/.hf_cache
export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"; conda activate llm_probing
cd /home/ml6/model_internal_sleuthing
REMOTE_FIXED="/data/user_data/ml6/probing_outputs_fixed"
for MODEL in bert-base-uncased bert-large-uncased; do
  ACT="${REMOTE_FIXED}/${MODEL}_ud_gum_coref_pairs_reps"
  ls "$ACT"/activations_part*.npz >/dev/null 2>&1 || { echo "SKIP $ACT"; continue; }
  python -m src.train --activations "$ACT" --labels data/ud_gum_coref_pairs_train.csv \
    --task coref --dataset ud_gum_coref_pairs --exp_label "$MODEL" \
    --probe_type cumulative --n_seeds 12 \
    --output_dir "output/probes_fixed/ud_gum_coref_pairs_${MODEL}_coref_cumulative" \
    || echo "ERROR $MODEL coref"
done
echo done.
