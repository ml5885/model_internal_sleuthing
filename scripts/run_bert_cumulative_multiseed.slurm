#!/bin/bash
#SBATCH --job-name=cumul_multiseed
#SBATCH --output=/home/ml6/logs/sbatch/%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --array=0-7

# Re-run cumulative scoring with multi-seed averaging (config CUMULATIVE_PARAMS.n_seeds)
# on the already-extracted FIXED activations, to denoise the expected-layer metric.
MODELS=("bert-base-uncased" "bert-large-uncased")
TASKS=("pos" "dep" "ner" "coref" "constituents" "srl" "spr" "relation")
declare -A DATASET_MAP=([pos]="ud_gum_pos" [dep]="ud_gum_dep" [ner]="ud_gum_ner" [coref]="ud_gum_coref_pairs" [constituents]="ud_gum_constituents" [srl]="up_ewt_srl" [spr]="spr" [relation]="semeval2010_relations")
declare -A LABELS_MAP=([pos]="data/ud_gum_pos_train.csv" [dep]="data/ud_gum_dep_train.csv" [ner]="data/ud_gum_ner_train.csv" [coref]="data/ud_gum_coref_pairs_train.csv" [constituents]="data/ud_gum_constituents_train.csv" [srl]="data/up_ewt_srl_train.csv" [spr]="data/spr_all_properties.csv" [relation]="data/semeval2010_relations_train.csv")

export HF_HOME=/data/user_data/ml6/.hf_cache
export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"; conda activate llm_probing
cd /home/ml6/model_internal_sleuthing
REMOTE_FIXED="/data/user_data/ml6/probing_outputs_fixed"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}; DS=${DATASET_MAP[$TASK]}; LB=${LABELS_MAP[$TASK]}
for MODEL in "${MODELS[@]}"; do
  ACT="${REMOTE_FIXED}/${MODEL}_${DS}_reps"
  ls "$ACT"/activations_part*.npz >/dev/null 2>&1 || { echo "SKIP no acts $ACT"; continue; }
  echo "=== cumulative(multiseed) $MODEL / $TASK"
  python -m src.train --activations "$ACT" --labels "$LB" --task "$TASK" --dataset "$DS" \
    --exp_label "$MODEL" --probe_type cumulative --max_examples 50000 \
    --output_dir "output/probes_fixed/${DS}_${MODEL}_${TASK}_cumulative" \
    || echo "ERROR $MODEL/$TASK"
done
echo "task $TASK done."
