#!/bin/bash

OUTDIR="notebooks/figures5"
MODELS="bert-base-uncased bert-large-uncased deberta-v3-large gpt2 gpt2-large gpt2-xl gemma2b gemma2b-it qwen2 qwen2-instruct"

python notebooks/sanitycheck.py --models $MODELS --outdir $OUTDIR