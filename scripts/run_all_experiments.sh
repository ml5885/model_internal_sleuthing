#!/bin/bash

models=("gpt2" "pythia1.4b" "gemma2b" "qwen2")
dataset="ud_gum_dataset"

for model in "${models[@]}"
do
    echo "Running experiment for model: $model"
    python -m src.experiment --model "$model" --dataset "$dataset" --one_vs_rest
done