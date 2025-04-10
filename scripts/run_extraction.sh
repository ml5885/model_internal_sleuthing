#!/bin/bash
# scripts/run_extraction.sh
# Runs the activation extraction script with specified parameters

DATA_FILE="data/controlled_sentences.csv"
OUTPUT_FILE="output/gpt2_reps.npz"
MODEL_KEY="gpt2"

python -m src.activation_extraction --data ${DATA_FILE} --output ${OUTPUT_FILE} --model ${MODEL_KEY}
