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
#SBATCH --array=0-1

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1

# ensure logs and caches exist
mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

# where experiment.py will write its per-task probe folders
USER_DATA_OUTPUT="/data/user_data/ml6/probing_outputs"
PROBES_SUBDIR="${USER_DATA_OUTPUT}/probes"

# where we want to copy everything locally
LOCAL_ROOT="/home/ml6/lexeme-inflection-probing/output"
LOCAL_PROBES="${LOCAL_ROOT}/probes"

mkdir -p "${PROBES_SUBDIR}"
mkdir -p "${LOCAL_PROBES}"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

# MODELS=("llama3-8b" "llama3-8b-instruct" "pythia-6.9b" "pythia-6.9b-tulu" "qwen2" "qwen2-instruct")
MODELS=("llama3-8b")
PROBES=("reg" "nn")

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / ${#PROBES[@]}))
PROBE_IDX=$((SLURM_ARRAY_TASK_ID % ${#PROBES[@]}))

MODEL=${MODELS[$MODEL_IDX]}
PROBE=${PROBES[$PROBE_IDX]}

DATASET="ud_gum_dataset"
PCA_DIM=50
PCA_SUFFIX="_pca_${PCA_DIM}"

echo "Running experiment with model=${MODEL}, probe_type=${PROBE}"

python -m src.experiment \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --probe_type "${PROBE}" \
    --pca_dim "${PCA_DIM}" \
    --output_dir "${PROBES_SUBDIR}" \
    --no_analysis

if [ $? -ne 0 ]; then
    echo "Experiment failed with error code $?"
    exit 1
fi

echo "Experiment completed successfully; copying CSV files back to local..."

for TASK in lexeme inflection; do
    EXP_FOLDER="${DATASET}_${MODEL}_${TASK}_${PROBE}${PCA_SUFFIX}"
    SRC_TASK_DIR="${PROBES_SUBDIR}/${EXP_FOLDER}"
    DST_TASK_DIR="${LOCAL_PROBES}/${EXP_FOLDER}"

    if [ -d "${SRC_TASK_DIR}" ]; then
        mkdir -p "${DST_TASK_DIR}"
        cp "${SRC_TASK_DIR}/predictions.csv" "${DST_TASK_DIR}/" 2>/dev/null || echo "  no predictions.csv for ${TASK}"
        cp "${SRC_TASK_DIR}/${TASK}_results.csv" "${DST_TASK_DIR}/" 2>/dev/null || echo "  no ${TASK}_results.csv"
        echo "Copied CSVs for ${TASK} into ${DST_TASK_DIR}"
    else
        echo "Warning: remote directory ${SRC_TASK_DIR} not found"
    fi
done
