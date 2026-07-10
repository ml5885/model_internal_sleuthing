#!/bin/bash
#SBATCH --job-name=llm_scalarmix_mdl
#SBATCH --output=/home/ml6/logs/sbatch/probing_%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%x_%A_%a.err
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=128G
#SBATCH --array=0-7

# Scalar-mix (Tenney et al. 2019 reproduction) and MDL (Voita & Titov 2020) probes.
# Reuses the same extracted activations as run_edge_probes.slurm; only trains new heads.

MODELS=(
  "bert-base-uncased" "bert-large-uncased" "deberta-v3-large"
  "gpt2" "gpt2-large" "gpt2-xl"
  "pythia-6.9b" "pythia-6.9b-tulu"
  "olmo2-7b" "olmo2-7b-instruct"
  "gemma2b" "gemma2b-it"
  "qwen2" "qwen2-instruct" "qwen2.5-7B" "qwen2.5-7B-instruct"
  "llama3-8b" "llama3-8b-instruct"
)

# Linear variants reproduce the paper most directly; add scalarmix_mlp / mdl_mlp for the MLP head.
PROBES=("scalarmix" "mdl")
TASKS=("pos" "dep" "ner" "coref" "constituents" "srl" "spr" "relation")

# Memory cap for scalar-mix (loads [N, n_layers, H] into RAM); 0 = all.
SCALARMIX_MAX_EXAMPLES=50000

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

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
export HF_HUB_OFFLINE=0
mkdir -p /home/ml6/logs/sbatch /data/user_data/ml6/.hf_cache

REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs"
export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing
cd /home/ml6/model_internal_sleuthing

N_GROUPS=8
GROUP_ID=${SLURM_ARRAY_TASK_ID}
GROUP_MODELS=()
for i in "${!MODELS[@]}"; do
  if (( i % N_GROUPS == GROUP_ID )); then
    GROUP_MODELS+=("${MODELS[$i]}")
  fi
done

echo "=== Array $SLURM_ARRAY_TASK_ID: ${#GROUP_MODELS[@]} models ==="
[ ${#GROUP_MODELS[@]} -eq 0 ] && { echo "No models for this index."; exit 0; }

for MODEL in "${GROUP_MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    DATASET=${DATASET_MAP[$TASK]}
    LABELS_FILE=${LABELS_MAP[$TASK]}
    ACTIVATIONS_DIR="${REMOTE_ACTIVATIONS}/${MODEL}_${DATASET}_reps"

    if [ ! -d "$ACTIVATIONS_DIR" ] || ! ls "$ACTIVATIONS_DIR"/activations_part*.npz >/dev/null 2>&1; then
      echo "SKIP (no activations): model=$MODEL task=$TASK -> $ACTIVATIONS_DIR"
      echo "     run run_edge_probes.slurm first to extract them."
      continue
    fi

    for PROBE in "${PROBES[@]}"; do
      echo "---- model=$MODEL task=$TASK probe=$PROBE"
      EXTRA=""
      if [[ "$PROBE" == scalarmix* ]]; then
        EXTRA="--max_examples ${SCALARMIX_MAX_EXAMPLES}"
      fi
      python -m src.train \
        --activations "$ACTIVATIONS_DIR" \
        --labels "$LABELS_FILE" \
        --task "$TASK" \
        --dataset "$DATASET" \
        --exp_label "$MODEL" \
        --probe_type "$PROBE" $EXTRA \
        || echo "ERROR: model=$MODEL task=$TASK probe=$PROBE"
    done
  done
  echo ">>> [Model: $MODEL] done <<<"
done

echo "Array $SLURM_ARRAY_TASK_ID finished."
