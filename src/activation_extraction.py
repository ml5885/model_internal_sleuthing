import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model_wrapper import ModelWrapper
from src import config, utils

def extract_and_save(data_path, output_path, model_key):
    df = pd.read_csv(data_path)
    sentences = df["Sentence"].tolist()
    target_indices = df["Target Index"].tolist()

    model_wrapper = ModelWrapper(model_key)
    batch_size = config.MODEL_CONFIGS[model_key]["batch_size"]
    total = len(sentences)
    activations_list = []

    for start in tqdm(range(0, total, batch_size), desc="Extracting Batches"):
        end = min(start + batch_size, total)
        batch_sentences = sentences[start:end]
        batch_target_indices = target_indices[start:end]

        if start == 0:
            inputs = model_wrapper.tokenize(batch_sentences)
            utils.log_debug("First batch token ids", inputs["input_ids"][0].tolist())

        batch_acts = model_wrapper.extract_activations(batch_sentences, batch_target_indices)
        activations_list.append(batch_acts.numpy())

    all_activations = np.concatenate(activations_list, axis=0)
    utils.log_info(f"Extracted activations shape: {all_activations.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, activations=all_activations)
    utils.log_info(f"Saved activations to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract model activations from sentences.")
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Output NPZ file path.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model key from config.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_and_save(args.data, args.output, args.model)
