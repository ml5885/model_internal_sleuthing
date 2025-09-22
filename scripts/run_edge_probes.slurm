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
  "gemma2b" "gemma2b-it"
  "qwen2" "qwen2-instruct" "qwen2.5-7B" "qwen2.5-7B-instruct"
  "llama3-8b" "llama3-8b-instruct"
)

PROBES=("reg" "nn")
TASKS=("pos" "dep" "ner" "coref" "constituents" "srl" "spr" "relation")

declare -A DATASET_MAP=(
  [pos]="ud_gum_pos"
  [dep]="ud_gum_dep"
  [ner]="ud_gum_ner"
  [coref]="ud_gum_coref"
  [constituents]="ud_gum_constituents"
  [srl]="up_ewt_srl"
  [spr]="spr"
  [relation]="semeval2010_relations"
)
declare -A LABELS_MAP=(
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
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs"
LOCAL_PROBES="/home/ml6/model_internal_sleuthing/output/probes"
mkdir -p "${LOCAL_PROBES}"

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

echo "=== Array $SLURM_ARRAY_TASK_ID: ${#GROUP_MODELS[@]} models (sequential) ==="
if [ ${#GROUP_MODELS[@]} -eq 0 ]; then
  echo "No models assigned to this array index. Exiting."
  exit 0
fi

ensure_labels() {
  local task="$1"
  local labels_path="$2"
  if [ -f "$labels_path" ]; then
    return 0
  fi
  echo "Labels missing ($labels_path). Building for task=$task ..."
  if [ "$task" = "spr" ] || [ "$task" = "relation" ]; then
    python data_utils/create_edge_datasets.py --tasks "$task"
  else
    python data_utils/create_edge_datasets.py --tasks "$task" --splits train
  fi
}

datafile_for_task() {
  local task="$1"
  local dataset="${DATASET_MAP[$task]}"
  if [ "$task" = "spr" ]; then
    echo "data/spr_all_properties.csv"
  elif [ "$task" = "relation" ]; then
    echo "data/semeval2010_relations_train.csv"
  else
    echo "data/${dataset}_train.csv"
  fi
}

for MODEL in "${GROUP_MODELS[@]}"; do
  echo ">>> [Model: $MODEL] starting (this array will run it alone, sequentially) <<<"

  # Loop ALL tasks for this model inside this array (prevents same-model clashes across arrays)
  for TASK in "${TASKS[@]}"; do
    DATASET=${DATASET_MAP[$TASK]}
    LABELS_FILE=${LABELS_MAP[$TASK]}
    ACTIVATIONS_DIR="${REMOTE_ACTIVATIONS}/${MODEL}_${DATASET}_reps"

    # Ensure labels exist
    ensure_labels "$TASK" "$LABELS_FILE" || {
      echo "ERROR: Could not build labels for task=$TASK"
      continue
    }

    # Pick the source CSV for activation extraction
    DATAFILE_TRAIN="$(datafile_for_task "$TASK")"
    if [ ! -f "$DATAFILE_TRAIN" ]; then
      echo "ERROR: Data file not found for extraction: $DATAFILE_TRAIN"
      continue
    fi

    # Extract activations once per (model, dataset) if missing
    if [ ! -d "$ACTIVATIONS_DIR" ] || ! ls "$ACTIVATIONS_DIR"/activations_part*.npz >/dev/null 2>&1; then
      echo "--- Extracting activations: model=$MODEL dataset=$DATASET -> $ACTIVATIONS_DIR"
      python -m src.activation_extraction \
        --data "$DATAFILE_TRAIN" \
        --output-dir "$ACTIVATIONS_DIR" \
        --model "$MODEL"
      if [ $? -ne 0 ]; then
        echo "ERROR: activation extraction failed (model=$MODEL, dataset=$DATASET)"
        # move on to next TASK; keep this model’s other tasks going
        continue
      fi
    else
      echo "--- Using existing activations at $ACTIVATIONS_DIR"
    fi

    # Train both probes for this task
    for PROBE in "${PROBES[@]}"; do
      echo "---- Training probe: model=$MODEL task=$TASK probe=$PROBE"
      python -m src.train \
        --activations "$ACTIVATIONS_DIR" \
        --labels "$LABELS_FILE" \
        --task "$TASK" \
        --dataset "$DATASET" \
        --exp_label "$MODEL" \
        --probe_type "$PROBE"
      if [ $? -ne 0 ]; then
        echo "ERROR: training failed (model=$MODEL, task=$TASK, probe=$PROBE)"
        # continue to next probe/task instead of aborting whole array
        continue
      fi
    done
  done

  echo ">>> [Model: $MODEL] done <<<"
done

echo "Array $SLURM_ARRAY_TASK_ID finished (models processed sequentially)."
