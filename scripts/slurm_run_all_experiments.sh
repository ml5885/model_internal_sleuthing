#!/bin/bash
#SBATCH --job-name=llm_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --array=0-3

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
# export HF_HUB_OFFLINE=1

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

REMOTE_ACTIVATIONS="/data/user_data/ml6/probing_outputs"
LOCAL_PROBES="/home/ml6/lexeme-inflection-probing/output/probes"

mkdir -p "${LOCAL_PROBES}"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

# MODELS=("llama3-8b" "llama3-8b-instruct" "pythia-6.9b" "pythia-6.9b-tulu" "qwen2" "qwen2-instruct")
MODELS=("olmo2-7b-instruct" "gemma2b-it")
PROBES=("reg" "nn")

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / ${#PROBES[@]}))
PROBE_IDX=$((SLURM_ARRAY_TASK_ID % ${#PROBES[@]}))

MODEL=${MODELS[$MODEL_IDX]}
PROBE=${PROBES[$PROBE_IDX]}

DATASET="ud_gum_dataset"
# PCA_DIM=50
# PCA_SUFFIX="_pca_${PCA_DIM}"

echo "=== model=${MODEL}, probe_type=${PROBE} ==="

python -m src.experiment \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --probe_type "${PROBE}" \
    --activations_dir "${REMOTE_ACTIVATIONS}" \
    --output_dir "${LOCAL_PROBES}" \
    --no_analysis

if [ $? -ne 0 ]; then
    echo "Experiment failed with exit code $?"
    exit 1
fi

echo "Done. Probe CSVs are in ${LOCAL_PROBES}."
