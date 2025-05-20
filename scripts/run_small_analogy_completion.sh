#!/bin/bash

OUTDIR="notebooks/figures5"
MODELS="google-bert/bert-base-uncased bert-large-uncased microsoft/deberta-v3-large gpt2 openai-community/gpt2-large openai-community/gpt2-xl google/gemma-2-2b google/gemma-2-2b-it Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct"

python notebooks/sanitycheck.py --models $MODELS --outdir $OUTDIR