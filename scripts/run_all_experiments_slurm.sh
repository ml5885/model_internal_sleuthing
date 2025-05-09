#!/bin/bash
#SBATCH --job-name=nlp_probing
#SBATCH --output=/home/ml6/logs/sbatch/probing_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/probing_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --array=0-3  # 4 jobs: one per model

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p /data/user_data/ml6/.hf_cache
mkdir -p /data/user_data/ml6/probing_outputs

USER_DATA_OUTPUT="/data/user_data/ml6/probing_outputs"
LOCAL_OUTPUT="/home/ml6/lexeme-inflection-probing/output"
mkdir -p $LOCAL_OUTPUT
mkdir -p $LOCAL_OUTPUT/probes

eval "$(conda shell.bash hook)"
conda activate llm_probing

cd /home/ml6/lexeme-inflection-probing

MODELS=("llama3-8b" "llama3-8b-instruct" "pythia-6.9b" "pythia-6.9b-tulu")
PROBE_TYPES=("reg" "nn")
DATASET="ud_gum_dataset"

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Starting experiments for model: $MODEL"

# Run experiments for both probe types
for PROBE in "${PROBE_TYPES[@]}"; do
    echo "Running experiment with model=${MODEL}, probe_type=${PROBE}"
    
    python -m src.experiment \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --probe_type "$PROBE" \
        --output_dir "$USER_DATA_OUTPUT" \
        --no_analysis
    
    RESULT_DIR_NAME="${DATASET}_${MODEL}_lexeme_${PROBE}_"
    RESULT_DIR_NAME2="${DATASET}_${MODEL}_inflection_${PROBE}_"
    
    echo "Copying result files back to local directory..."
    
    mkdir -p $LOCAL_OUTPUT/probes/
    
    find $USER_DATA_OUTPUT/probes -name "*.csv" -path "*${RESULT_DIR_NAME}*" -exec cp {} $LOCAL_OUTPUT/probes/ \;
    find $USER_DATA_OUTPUT/probes -name "*.csv" -path "*${RESULT_DIR_NAME2}*" -exec cp {} $LOCAL_OUTPUT/probes/ \;
    
    find $USER_DATA_OUTPUT/probes -name "*.png" -path "*${RESULT_DIR_NAME}*" -exec cp {} $LOCAL_OUTPUT/probes/ \;
    find $USER_DATA_OUTPUT/probes -name "*.png" -path "*${RESULT_DIR_NAME2}*" -exec cp {} $LOCAL_OUTPUT/probes/ \;
done

echo "All experiments completed for model ${MODEL}"
echo "Result files have been copied to $LOCAL_OUTPUT/probes/"
