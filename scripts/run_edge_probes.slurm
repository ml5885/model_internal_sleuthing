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
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --mem=64G
#SBATCH --array=0-7

# List of models to probe
MODELS=(
    "bert-base-uncased" "bert-large-uncased" "deberta-v3-large" 
    "gpt2" "gpt2-large" "gpt2-xl"
    "pythia-6.9b" "pythia-6.9b-tulu" 
    # "olmo2-7b" "olmo2-7b-instruct"
    # "gemma2b" "gemma2b-it" "qwen2" "qwen2-instruct" "qwen2.5-7B" "qwen2.5-7B-instruct"
    # "llama3-8b" "llama3-8b-instruct"
)

# Probe architectures
PROBES=("reg" "nn")

# Eight probing tasks (map to array indices 0–7)
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

# Determine which task to run based on the array index
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASET_MAP[$TASK]}
LABELS_FILE=${LABELS_MAP[$TASK]}

# Hugging Face caches and offline settings
export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

# Directories for activations and probe outputs
REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs"
LOCAL_PROBES="/home/ml6/model_internal_sleuthing/output/probes"
mkdir -p "${LOCAL_PROBES}"

export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/model_internal_sleuthing

echo "=== Starting array task $SLURM_ARRAY_TASK_ID: task=$TASK ==="

# Loop over all models and both probe types
for MODEL in "${MODELS[@]}"; do
  for PROBE in "${PROBES[@]}"; do
    echo "---- Running model=${MODEL}, probe=${PROBE}, task=${TASK} ----"
    python -m src.train \
      --activations "${REMOTE_ACTIVATIONS}/${MODEL}_reps" \
      --labels "${LABELS_FILE}" \
      --task "${TASK}" \
      --dataset "${DATASET}" \
      --exp_label "${MODEL}" \
      --probe_type "${PROBE}"
    if [ $? -ne 0 ]; then
      echo "Error: task ${TASK} on model ${MODEL} with probe ${PROBE} failed."
      exit 1
    fi
  done
done

echo "Finished array task $SLURM_ARRAY_TASK_ID: task=$TASK"
