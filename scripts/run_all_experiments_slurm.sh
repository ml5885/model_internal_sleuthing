#!/bin/bash
#SBATCH --job-name=llm_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --array=0-11

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache
mkdir -p /data/user_data/ml6/probing_outputs/probes
mkdir -p /home/ml6/lexeme-inflection-probing/output/probes

USER_DATA_OUTPUT="/data/user_data/ml6/probing_outputs"
LOCAL_OUTPUT="/home/ml6/lexeme-inflection-probing/output"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

# MODELS=("llama3-8b" "llama3-8b-instruct" "pythia-6.9b" "pythia-6.9b-tulu" "qwen2" "qwen2-instruct")
MODELS=("gpt2")
PROBES=("reg" "nn")

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 2))
PROBE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

MODEL=${MODELS[$MODEL_IDX]}
PROBE=${PROBES[$PROBE_IDX]}

DATASET="ud_gum_dataset"
PCA_DIM=50
PCA_SUFFIX="_pca_${PCA_DIM}"

EXP_FOLDER="${DATASET}-${MODEL}_${PROBE}${PCA_SUFFIX}"
REMOTE_PROBE_PATH="${USER_DATA_OUTPUT}/probes/${EXP_FOLDER}"
LOCAL_PROBE_PATH="${LOCAL_OUTPUT}/probes/${EXP_FOLDER}"

mkdir -p "$REMOTE_PROBE_PATH"
mkdir -p "$LOCAL_PROBE_PATH"

echo "Running experiment with model=$MODEL, probe_type=$PROBE"

python -m src.experiment \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --probe_type "$PROBE" \
    --pca_dim "$PCA_DIM" \
    --output_dir "$USER_DATA_OUTPUT" \
    --no_analysis

if [ $? -eq 0 ]; then
    echo "Experiment completed successfully, copying .csv files to local output..."

    for TASK in "lexeme" "inflection"; do
        SRC_TASK_PATH="${USER_DATA_OUTPUT}/probes/${DATASET}_${MODEL}_${TASK}_${PROBE}${PCA_SUFFIX}"
        DST_TASK_PATH="${LOCAL_OUTPUT}/probes/${DATASET}_${MODEL}_${TASK}_${PROBE}${PCA_SUFFIX}"
        mkdir -p "$DST_TASK_PATH"

        if [ -d "$SRC_TASK_PATH" ]; then
            find "$SRC_TASK_PATH" -type f -name "*.csv" -exec cp {} "$DST_TASK_PATH/" \;
            echo "Copied CSVs for $TASK to $DST_TASK_PATH"
        else
            echo "Warning: Source directory $SRC_TASK_PATH not found."
        fi
    done
else
    echo "Experiment failed with error code $?"
fi
