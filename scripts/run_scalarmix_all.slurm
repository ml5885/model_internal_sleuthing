#!/bin/bash
#SBATCH --job-name=scalarmix_all
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=128G
#SBATCH --array=0-119%24

# Extract token-aligned activations and run the scalar-mixing probe for one
# (model, language) cell per array index (manifest: scripts/scalarmix_manifest.tsv).
# Datasets that did not build are skipped. Existing activations are reused.

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_OFFLINE=0
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/ml6/logs/sbatch
eval "$(conda shell.bash hook)"; conda activate llm_probing
cd /home/ml6/model_internal_sleuthing

REMOTE_FIXED="/data/user_data/ml6/probing_outputs_fixed"

# task|dataset_id|csv lines for a language (dataset_id keeps activations per task)
emit_jobs() {
  case "$1" in
    en)
      echo "pos|ud_gum_pos|data/ud_gum_pos_train.csv"
      echo "dep|ud_gum_dep|data/ud_gum_dep_train.csv"
      echo "ner|ud_gum_ner|data/ud_gum_ner_train.csv"
      echo "constituents|ud_gum_constituents|data/ud_gum_constituents_train.csv"
      echo "coref|ud_gum_coref_pairs|data/ud_gum_coref_pairs_train.csv"
      echo "srl|up_ewt_srl|data/up_ewt_srl_train.csv"
      echo "relation|semeval2010_relations|data/semeval2010_relations_train.csv" ;;
    zh)
      for t in pos dep constituents; do echo "$t|ud_chinese_gsdsimp_$t|data/ud_chinese_gsdsimp_${t}_train.csv"; done
      echo "ner|msra_ner_ner|data/msra_ner_ner_train.csv"
      echo "srl|up_chinese_srl|data/up_chinese_srl_train.csv"
      echo "relation|duie_relation|data/duie_relation_train.csv" ;;
    tr)
      for t in pos dep constituents; do echo "$t|ud_turkish_boun_$t|data/ud_turkish_boun_${t}_train.csv"; done
      echo "ner|turkish_wikiner_ner|data/turkish_wikiner_ner_train.csv"
      echo "coref|corefud_turkish_itcc_coref|data/corefud_turkish_itcc_coref_train.csv" ;;
    fr)
      for t in pos dep constituents; do echo "$t|ud_french_gsd_$t|data/ud_french_gsd_${t}_train.csv"; done
      echo "ner|wikiner_fr_gold_ner|data/wikiner_fr_gold_ner_train.csv"
      echo "srl|up_french_srl|data/up_french_srl_train.csv"
      echo "coref|corefud_french_democrat_coref|data/corefud_french_democrat_coref_train.csv"
      echo "relation|redfm_fr_relation|data/redfm_fr_relation_train.csv" ;;
    ru)
      for t in pos dep constituents; do echo "$t|ud_russian_syntagrus_$t|data/ud_russian_syntagrus_${t}_train.csv"; done
      echo "ner|nerel_ner|data/nerel_ner_train.csv"
      echo "coref|rucoco_coref|data/rucoco_coref_train.csv" ;;
    de)
      for t in pos dep constituents; do echo "$t|ud_german_hdt_$t|data/ud_german_hdt_${t}_train.csv"; done
      echo "ner|germeval_14_ner|data/germeval_14_ner_train.csv"
      echo "srl|up_german_srl|data/up_german_srl_train.csv"
      echo "coref|corefud_german_potsdamcc_coref|data/corefud_german_potsdamcc_coref_train.csv"
      echo "relation|redfm_de_relation|data/redfm_de_relation_train.csv" ;;
  esac
}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" scripts/scalarmix_manifest.tsv)
MODEL=$(echo "$LINE" | cut -f1)
LANG=$(echo "$LINE" | cut -f2)
echo "=== cell $SLURM_ARRAY_TASK_ID : model=$MODEL lang=$LANG ==="

emit_jobs "$LANG" | while IFS='|' read -r TASK DSID CSV; do
  [ -f "$CSV" ] || { echo "SKIP (no csv): $CSV"; continue; }
  ACT="${REMOTE_FIXED}/${MODEL}_${DSID}_reps"
  if ! ls "$ACT"/activations_part*.npz >/dev/null 2>&1; then
    echo "--- extract $MODEL / $DSID"
    python -m src.activation_extraction --data "$CSV" --output-dir "$ACT" --model "$MODEL" \
      || { echo "ERROR extract $MODEL/$DSID"; continue; }
  fi
  echo "--- scalarmix $MODEL / $LANG / $TASK"
  python -m src.train --activations "$ACT" --labels "$CSV" --task "$TASK" \
    --dataset "$DSID" --exp_label "$MODEL" --probe_type scalarmix --max_examples 50000 \
    --output_dir "output/probes_multiling/${DSID}_${MODEL}_${TASK}_scalarmix" \
    || echo "ERROR probe $MODEL/$LANG/$TASK"
done
echo "=== cell $SLURM_ARRAY_TASK_ID done ==="
