#!/bin/bash
# scripts/run_probes.sh
# Runs the probe training script

ACTIVATIONS_FILE="output/gpt2_reps.npz"
LABELS_FILE="data/controlled_sentences.csv"
TASK="both"  # Options: 'tense', 'lexeme', or 'both'

python -m src.probe_training --activations ${ACTIVATIONS_FILE} --labels ${LABELS_FILE} --task ${TASK}
