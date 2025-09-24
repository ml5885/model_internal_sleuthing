#!/bin/bash
#SBATCH --job-name=spr_probing
#SBATCH --output=/home/ml6/logs/sbatch/spr_probing_%x_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/spr_probing_%x_%A_%a.err
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --array=0-8

# models to run
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
# only run SPR
TASKS=("spr")

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs2"
# save probe outputs to output_spr
LOCAL_PROBES="/home/ml6/model_internal_sleuthing/output_spr"
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

# Hardcode SPR dataset/labels
SPR_DATASET="spr"
SPR_LABELS="data/spr_all_properties.csv"
SPR_DATAFILE="data/spr_all_properties.csv"

ensure_spr_labels() {
  if [ -f "$SPR_LABELS" ]; then
    return 0
  fi
  echo "Labels missing ($SPR_LABELS). Building SPR labels ..."
  python data_utils/create_edge_datasets.py --tasks spr
}

for MODEL in "${GROUP_MODELS[@]}"; do
  echo ">>> [Model: $MODEL] starting (this array will run it alone, sequentially) <<<"

  TASK="spr"
  DATASET="${SPR_DATASET}"
  LABELS_FILE="${SPR_LABELS}"
  ACTIVATIONS_DIR="${REMOTE_ACTIVATIONS}/${MODEL}_${DATASET}_reps"

  ensure_spr_labels || {
    echo "ERROR: Could not build SPR labels. Skipping model=$MODEL"
    continue
  }

  if [ ! -f "$SPR_DATAFILE" ]; then
    echo "ERROR: SPR data file not found: $SPR_DATAFILE"
    continue
  fi

  # Extract activations once per (model, dataset) if missing
  if [ ! -d "$ACTIVATIONS_DIR" ] || ! ls "$ACTIVATIONS_DIR"/activations_part*.npz >/dev/null 2>&1; then
    echo "--- Extracting activations: model=$MODEL dataset=$DATASET -> $ACTIVATIONS_DIR"
    python -m src.activation_extraction \
      --data "$SPR_DATAFILE" \
      --output-dir "$ACTIVATIONS_DIR" \
      --model "$MODEL"
    if [ $? -ne 0 ]; then
      echo "ERROR: activation extraction failed (model=$MODEL, dataset=$DATASET)"
      continue
    fi
  else
    echo "--- Using existing activations at $ACTIVATIONS_DIR"
  fi

  # Train both probes for SPR and save to LOCAL_PROBES
  for PROBE in "${PROBES[@]}"; do
    echo "---- Training probe: model=$MODEL task=$TASK probe=$PROBE"
    python -m src.train \
      --activations "$ACTIVATIONS_DIR" \
      --labels "$LABELS_FILE" \
      --task "$TASK" \
      --dataset "$DATASET" \
      --exp_label "$MODEL" \
      --probe_type "$PROBE" \
      --output-dir "${LOCAL_PROBES}"
    if [ $? -ne 0 ]; then
      echo "ERROR: training failed (model=$MODEL, task=$TASK, probe=$PROBE)"
      continue
    fi
  done

  echo ">>> [Model: $MODEL] done <<<"
done

echo "Array $SLURM_ARRAY_TASK_ID finished (SPR models processed sequentially)."