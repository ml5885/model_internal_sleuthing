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
#SBATCH --array=0-3

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache
mkdir -p /data/user_data/ml6/probing_outputs
mkdir -p /data/user_data/ml6/probing_outputs/probes

USER_DATA_OUTPUT="/data/user_data/ml6/probing_outputs"
LOCAL_OUTPUT="/home/ml6/lexeme-inflection-probing/output"

mkdir -p "$LOCAL_OUTPUT/probes"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

MODELS=("llama3-8b" "llama3-8b-instruct")
PROBE_TYPES=("reg" "nn")

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 2))
PROBE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

MODEL=${MODELS[$MODEL_IDX]}
PROBE=${PROBE_TYPES[$PROBE_IDX]}
DATASET="ud_gum_dataset"

PCA_DIM=0  # Set this to your desired PCA dimension or pass as an argument
PCA_SUFFIX=""
if [ "$PCA_DIM" -gt 0 ]; then
    PCA_SUFFIX="_pca$PCA_DIM"
fi

OUTDIR_LEX="$USER_DATA_OUTPUT/probes/${DATASET}_${MODEL}_lexeme_${PROBE}${PCA_SUFFIX}"
OUTDIR_INF="$USER_DATA_OUTPUT/probes/${DATASET}_${MODEL}_inflection_${PROBE}${PCA_SUFFIX}"

if [ -d "$OUTDIR_LEX" ] || [ -d "$OUTDIR_INF" ]; then
    echo "Skipping ${MODEL}: probe output already exists."
    exit 0
fi

OUTDIR="$USER_DATA_OUTPUT/probes/${DATASET}_${MODEL}_${PROBE}${PCA_SUFFIX}"
mkdir -p "$OUTDIR"

echo "Running experiment with model=$MODEL, probe_type=$PROBE"

python -m src.experiment \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --probe_type "$PROBE" \
    --pca_dim "$PCA_DIM" \
    --output_dir "$USER_DATA_OUTPUT" \
    --no_analysis

if [ $? -eq 0 ]; then
    echo "Experiment completed successfully, copying results..."
    TASKS=("lexeme" "inflection")
    for TASK in "${TASKS[@]}"; do
        DEST_DIR="$LOCAL_OUTPUT/probes/${DATASET}_${MODEL}_${TASK}_${PROBE}${PCA_SUFFIX}"
        echo "Copying $TASK results to $DEST_DIR"
        mkdir -p "$DEST_DIR"
        RESULT_FILES=$(find "$OUTDIR" -type f \( -name "*.csv" -o -name "*.png" -o -name "*results.npz" \) -path "*${DATASET}_${MODEL}_${TASK}_${PROBE}${PCA_SUFFIX}*" 2>/dev/null || echo "")
        if [ -n "$RESULT_FILES" ]; then
            for FILE in $RESULT_FILES; do
                cp "$FILE" "$DEST_DIR/"
            done
            echo "Copied $(echo "$RESULT_FILES" | wc -l) files to $DEST_DIR"
        else
            echo "No result files found for $TASK"
        fi
    done
    echo "Results have been copied to $LOCAL_OUTPUT/probes/"
else
    echo "Experiment failed with error code $?"
fi