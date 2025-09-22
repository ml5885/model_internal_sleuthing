#!/bin/bash
#SBATCH --job-name=llm_edge_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%x_%A_%a.err
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --array=0-7

MODELS=(
    "bert-base-uncased" "bert-large-uncased" "deberta-v3-large" 
    "gpt2" "gpt2-large" "gpt2-xl"
    "pythia-6.9b" "pythia-6.9b-tulu" 
    "olmo2-7b" "olmo2-7b-instruct"
    "gemma2b" "gemma2b-it" "qwen2" "qwen2-instruct" "qwen2.5-7B" "qwen2.5-7B-instruct"
    "llama3-8b" "llama3-8b-instruct"
)

PROBES=("reg" "nn")

# Eight probing tasks
TASKS=("pos" "dep" "ner" "coref" "constituents" "srl" "spr" "relation")

# Dataset and label-file mapping
declare -A DATASET_MAP
declare -A LABELS_MAP
DATASET_MAP=(
  [pos]="ud_gum_pos"
  [dep]="ud_gum_dep"
  [ner]="ud_gum_ner"
  [coref]="ud_gum_coref"
  [constituents]="ud_gum_constituents"
  [srl]="up_ewt_srl"
  [spr]="spr"
  [relation]="semeval2010_relations"
)
LABELS_MAP=(
  [pos]="data/ud_gum_pos_train.csv"
  [dep]="data/ud_gum_dep_train.csv"
  [ner]="data/ud_gum_ner_train.csv"
  [coref]="data/ud_gum_coref_pairs_train.csv"
  [constituents]="data/ud_gum_constituents_train.csv"
  [srl]="data/up_ewt_srl_train.csv"
  [spr]="data/spr_all_properties.csv"
  [relation]="data/semeval2010_relations_train.csv"
)

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token


mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs"
LOCAL_PROBES="/home/ml6/model_internal_sleuthing/output/probes"
mkdir -p "${LOCAL_PROBES}"

export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/model_internal_sleuthing

# Split models into 8 groups by index mod 8, one group per array task
N_GROUPS=8
SELECTED_MODELS=()
for idx in "${!MODELS[@]}"; do
  if (( idx % N_GROUPS == SLURM_ARRAY_TASK_ID )); then
    SELECTED_MODELS+=("${MODELS[$idx]}")
  fi
done

echo "=== Starting array task $SLURM_ARRAY_TASK_ID: ${#SELECTED_MODELS[@]} models in this slice ==="
if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
  echo "No models assigned to this array index. Exiting."
  exit 0
fi

# Loop over selected models, all tasks, and probe types
for MODEL in "${SELECTED_MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    DATASET=${DATASET_MAP[$TASK]}
    LABELS_FILE=${LABELS_MAP[$TASK]}
    for PROBE in "${PROBES[@]}"; do
      echo "---- Model=${MODEL}, task=${TASK}, probe=${PROBE} ----"

        # Determine activation directory for this model/dataset
        ACTIVATIONS_DIR="${REMOTE_ACTIVATIONS}/${MODEL}_${DATASET}_reps"

        # If activations for this model/dataset do not exist, run extraction on the training split
        if [ ! -d "$ACTIVATIONS_DIR" ] || ! ls "$ACTIVATIONS_DIR"/activations_part*.npz 1>/dev/null 2>&1; then
        echo "No activations found for ${MODEL} on ${DATASET}. Extracting..."
        python -m src.activation_extraction \
            --data "data/${DATASET}_train.csv" \
            --output-dir "$ACTIVATIONS_DIR" \
            --model "$MODEL"
        if [ $? -ne 0 ]; then
          echo "Error: activation extraction failed for model=${MODEL}, dataset=${DATASET}"
          exit 1
        fi
      else
        echo "Using existing activations in $ACTIVATIONS_DIR"
      fi

      # Train the probe on the extracted activations
      python -m src.train \
        --activations "$ACTIVATIONS_DIR" \
        --labels "$LABELS_FILE" \
        --task "$TASK" \
        --dataset "$DATASET" \
        --exp_label "$MODEL" \
        --probe_type "$PROBE"
      if [ $? -ne 0 ]; then
        echo "Error: training failed for model=${MODEL}, task=${TASK}, probe=${PROBE}"
        exit 1
      fi
    done
  done
done

echo "Finished array task $SLURM_ARRAY_TASK_ID"