#!/bin/bash
#SBATCH --job-name=llm_probing_checkpoints
#SBATCH --output=/home/ml6/logs/sbatch/probing_checkpoints_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_checkpoints_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --array=0-7

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache

# Model and revision arrays
MODEL_KEYS=(
    "pythia-6.9b"
    "pythia-6.9b"
    "pythia-6.9b"
    "pythia-6.9b"
    "olmo2-7b"
    "olmo2-7b"
    "olmo2-7b"
    "olmo2-7b"
)
REVISIONS=(
    "step0"
    "step512"
    "step1000"
    "step143000"
    "stage1-step1000-tokens5B"
    "stage1-step200000-tokens839B"
    "stage1-step400000-tokens1678B"
    "stage1-step600000-tokens2517B"
)
PROBES=("reg" "mlp" "rf")

MODEL_REV_IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODEL_KEYS[$MODEL_REV_IDX]}
REVISION=${REVISIONS[$MODEL_REV_IDX]}

DATASET_NAME="ud_gum_dataset"
REMOTE_ACTIVATIONS="/data/user_data/ml6/output/activations"
LOCAL_PROBES="/data/user_data/ml6/output/probes"

mkdir -p "${LOCAL_PROBES}"
mkdir -p "${REMOTE_ACTIVATIONS}"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

for PROBE in "${PROBES[@]}"; do
    echo "=== model=${MODEL}, revision=${REVISION}, probe_type=${PROBE} ==="
    if [ "$PROBE" = "rf" ]; then
        python -m src.experiment \
            --model "$MODEL" \
            --revision "$REVISION" \
            --dataset "$DATASET_NAME" \
            --probe_type "$PROBE" \
            --activations_dir "$REMOTE_ACTIVATIONS" \
            --output_dir "$LOCAL_PROBES" \
            --experiment "inflection"
    else
        python -m src.experiment \
            --model "$MODEL" \
            --revision "$REVISION" \
            --dataset "$DATASET_NAME" \
            --probe_type "$PROBE" \
            --activations_dir "$REMOTE_ACTIVATIONS" \
            --output_dir "$LOCAL_PROBES"
    fi
    if [ $? -ne 0 ]; then
        echo "Experiment failed for probe $PROBE with exit code $?"
        exit 1
    fi
done

echo "Done. Probe CSVs are in ${LOCAL_PROBES}."