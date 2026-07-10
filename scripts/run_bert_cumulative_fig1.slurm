#!/bin/bash
#SBATCH --job-name=cumulative_fig1
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A_%a.err
#SBATCH --partition=debug
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-7

# One-off: scalar-mix probes for the two BERT models across all 8 edge tasks,
# cumulative scoring (Eq.4) + F1 columns for Figure 1. Array index = task.

MODELS=("bert-base-uncased" "bert-large-uncased")
PROBES=("cumulative")
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

REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs"
export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"
conda activate llm_probing
cd /home/ml6/model_internal_sleuthing

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASET_MAP[$TASK]}
LABELS_FILE=${LABELS_MAP[$TASK]}

for MODEL in "${MODELS[@]}"; do
  ACT="${REMOTE_ACTIVATIONS}/${MODEL}_${DATASET}_reps"
  if ! ls "$ACT"/activations_part*.npz >/dev/null 2>&1; then
    echo "SKIP (no activations): $ACT"; continue
  fi
  for PROBE in "${PROBES[@]}"; do
    echo "==== model=$MODEL task=$TASK probe=$PROBE"
    python -m src.train --activations "$ACT" --labels "$LABELS_FILE" \
      --task "$TASK" --dataset "$DATASET" --exp_label "$MODEL" \
      --probe_type "$PROBE" --max_examples 50000 \
      || echo "ERROR: model=$MODEL task=$TASK probe=$PROBE"
  done
done
echo "task $TASK done."
