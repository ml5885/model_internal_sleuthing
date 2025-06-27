#!/bin/bash

OUTDIR="notebooks/figures5"
MODELS="google-bert/bert-base-uncased \
bert-large-uncased \
microsoft/deberta-v3-large \
gpt2 \
openai-community/gpt2-large \
openai-community/gpt2-xl \
EleutherAI/pythia-6.9b \
allenai/open-instruct-pythia-6.9b-tulu \
allenai/OLMo-2-1124-7B \
allenai/OLMo-2-1124-7B-Instruct \
google/gemma-2-2b \
google/gemma-2-2b-it \
Qwen/Qwen2.5-1.5B \
Qwen/Qwen2.5-1.5B-Instruct \
meta-llama/Llama-3.1-8B \
meta-llama/Llama-3.1-8B-Instruct"
DATASETS=("de_gsd" "zh_gsd" "fr_gsd" "ru_syntagrus" "tr_imst")

for DATASET in "${DATASETS[@]}"; do
    python notebooks/sanitycheck.py --models $MODELS --outdir $OUTDIR/$DATASET
done