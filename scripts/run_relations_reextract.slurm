#!/bin/bash
#SBATCH --job-name=rel_reextract
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A.err
#SBATCH --partition=general
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G

# Re-extract SemEval relations with FIXED (word-index) spans, then re-run probes.
export HF_HOME=/data/user_data/ml6/.hf_cache
export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"; conda activate llm_probing
cd /home/ml6/model_internal_sleuthing
REMOTE_FIXED="/data/user_data/ml6/probing_outputs_fixed"
for MODEL in bert-base-uncased bert-large-uncased; do
  ACT="${REMOTE_FIXED}/${MODEL}_semeval2010_relations_reps"
  rm -rf "$ACT"                      # discard broken-span activations
  python -m src.activation_extraction --data data/semeval2010_relations_train.csv \
    --output-dir "$ACT" --model "$MODEL" || { echo "extract fail $MODEL"; continue; }
  for PROBE in scalarmix cumulative; do
    python -m src.train --activations "$ACT" --labels data/semeval2010_relations_train.csv \
      --task relation --dataset semeval2010_relations --exp_label "$MODEL" \
      --probe_type "$PROBE" --max_examples 50000 \
      --output_dir "output/probes_fixed/semeval2010_relations_${MODEL}_relation_${PROBE}" \
      || echo "ERROR $MODEL $PROBE"
  done
done
echo done.
