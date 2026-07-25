#!/bin/bash
#SBATCH --job-name=relner_multiseed
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=128G
#SBATCH --array=0-24%8

# Re-run ONLY the cumulative cells whose expected-layer is a high-variance ratio
# estimator -- relations (small, imbalanced) and NER (near-flat curve). At the
# grid's n_seeds=4 a single bad top-layer probe fit swings the expected layer by
# several layers, so independent runs disagree (this is what made the multiling
# English panel diverge from the faithful figure1_fixed BERT numbers). Here we let
# the per-task seed map in config (ner/relation -> 12) denoise them, mirroring the
# coref fix. POS/deps/SRL are well-conditioned at 4 and are left untouched.
# Overwrites the n_seeds=4 results in place; resumable via the saved n_seeds field.

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_OFFLINE=0
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/ml6/logs/sbatch
eval "$(conda shell.bash hook)"; conda activate llm_probing
cd /home/ml6/model_internal_sleuthing

REMOTE_FIXED="/data/user_data/ml6/probing_outputs_fixed"

# task|dataset_id|csv lines -- NER (all six languages) and relations (en/zh/fr/de).
emit_jobs() {
  case "$1" in
    en) echo "ner|ud_gum_ner|data/ud_gum_ner_train.csv"
        echo "relation|semeval2010_relations|data/semeval2010_relations_train.csv" ;;
    zh) echo "ner|msra_ner_ner|data/msra_ner_ner_train.csv"
        echo "relation|duie_relation|data/duie_relation_train.csv" ;;
    tr) echo "ner|turkish_wikiner_ner|data/turkish_wikiner_ner_train.csv" ;;
    fr) echo "ner|wikiner_fr_gold_ner|data/wikiner_fr_gold_ner_train.csv"
        echo "relation|redfm_fr_relation|data/redfm_fr_relation_train.csv" ;;
    ru) echo "ner|nerel_ner|data/nerel_ner_train.csv" ;;
    de) echo "ner|germeval_14_ner|data/germeval_14_ner_train.csv"
        echo "relation|redfm_de_relation|data/redfm_de_relation_train.csv" ;;
  esac
}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" scripts/scalarmix_manifest.tsv)
MODEL=$(echo "$LINE" | cut -f1)
LANGS=$(echo "$LINE" | cut -f2)
echo "=== model $SLURM_ARRAY_TASK_ID : $MODEL langs=$LANGS ==="

for LANG in ${LANGS//,/ }; do
  emit_jobs "$LANG" | while IFS='|' read -r TASK DSID CSV; do
    [ -f "$CSV" ] || { echo "SKIP (no csv): $CSV"; continue; }
    ACT="${REMOTE_FIXED}/${MODEL}_${DSID}_reps"
    ls "$ACT"/activations_part*.npz >/dev/null 2>&1 || { echo "SKIP (no activations yet): $ACT"; continue; }
    OUT="output/probes_multiling/${DSID}_${MODEL}_${TASK}_cumulative"
    JS="$OUT/cumulative_scores.json"
    if [ -f "$JS" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$JS')).get('n_seeds',0)>=12 else 1)" 2>/dev/null; then
      echo "skip done (>=12 seeds) $OUT"; continue
    fi
    echo "--- multiseed cumulative $MODEL / $LANG / $TASK"
    python -m src.train --activations "$ACT" --labels "$CSV" --task "$TASK" \
      --dataset "$DSID" --exp_label "$MODEL" --probe_type cumulative --max_examples 50000 \
      --output_dir "$OUT" || echo "ERROR probe $MODEL/$LANG/$TASK"
  done
done
echo "=== model $SLURM_ARRAY_TASK_ID done ==="
