#!/bin/bash
#SBATCH --job-name=bert_fixed_fig1
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --array=0-7

# Re-extract edge activations with the FIXED (token-aligned) CSVs and run
# scalar-mix + cumulative probes for BERT-base/large. Array index = task.
# Writes fresh activations to probing_outputs_fixed so the broken ones are kept.

MODELS=("bert-base-uncased" "bert-large-uncased")
PROBES=("scalarmix" "cumulative")
TASKS=("pos" "dep" "ner" "coref" "constituents" "srl" "spr" "relation")

declare -A DATASET_MAP=(
  [pos]="ud_gum_pos"           [dep]="ud_gum_dep"
  [ner]="ud_gum_ner"           [coref]="ud_gum_coref_pairs"
  [constituents]="ud_gum_constituents" [srl]="up_ewt_srl"
  [spr]="spr"                  [relation]="semeval2010_relations"
)
declare -A LABELS_MAP=(
  [pos]="data/ud_gum_pos_train.csv"       [dep]="data/ud_gum_dep_train.csv"
  [ner]="data/ud_gum_ner_train.csv"       [coref]="data/ud_gum_coref_pairs_train.csv"
  [constituents]="data/ud_gum_constituents_train.csv" [srl]="data/up_ewt_srl_train.csv"
  [spr]="data/spr_all_properties.csv"     [relation]="data/semeval2010_relations_train.csv"
)
# Source CSV used for activation extraction (same as labels except naming)
datafile_for_task() {
  case "$1" in
    spr) echo "data/spr_all_properties.csv" ;;
    relation) echo "data/semeval2010_relations_train.csv" ;;
    coref) echo "data/ud_gum_coref_pairs_train.csv" ;;
    *) echo "data/${DATASET_MAP[$1]}_train.csv" ;;
  esac
}

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/ml6/logs/sbatch

REMOTE_FIXED="/data/user_data/ml6/probing_outputs_fixed"
eval "$(conda shell.bash hook)"
conda activate llm_probing
cd /home/ml6/model_internal_sleuthing

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASET_MAP[$TASK]}
LABELS=${LABELS_MAP[$TASK]}
DATAFILE=$(datafile_for_task "$TASK")

for MODEL in "${MODELS[@]}"; do
  ACT="${REMOTE_FIXED}/${MODEL}_${DATASET}_reps"
  if ! ls "$ACT"/activations_part*.npz >/dev/null 2>&1; then
    echo "=== Extracting $MODEL / $TASK -> $ACT"
    python -m src.activation_extraction --data "$DATAFILE" --output-dir "$ACT" --model "$MODEL" \
      || { echo "ERROR: extraction failed $MODEL/$TASK"; continue; }
  else
    echo "=== Using existing $ACT"
  fi
  for PROBE in "${PROBES[@]}"; do
    echo "=== Probe $MODEL / $TASK / $PROBE"
    python -m src.train --activations "$ACT" --labels "$LABELS" \
      --task "$TASK" --dataset "$DATASET" --exp_label "$MODEL" \
      --probe_type "$PROBE" --max_examples 50000 \
      --output_dir "output/probes_fixed/${DATASET}_${MODEL}_${TASK}_${PROBE}" \
      || echo "ERROR: probe failed $MODEL/$TASK/$PROBE"
  done
done
echo "task $TASK done."
