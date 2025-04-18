import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model_wrapper import ModelWrapper
from src import config, utils

def extract_and_save(data_path, output_dir, model_key):
    df = pd.read_csv(data_path)
    sentences = df["Sentence"].tolist()
    target_indices = df["Target Index"].tolist()

    model_wrapper = ModelWrapper(model_key)
    batch_size = config.MODEL_CONFIGS[model_key]["batch_size"]
    total = len(sentences)
    if total == 0:
        utils.log_error("Input data is empty, no activations to extract.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Process in shards and save each
    for part_idx, start in enumerate(tqdm(range(0, total, batch_size), desc="Extracting Batches")):
        end = min(start + batch_size, total)
        batch_sents = sentences[start:end]
        batch_idxs = target_indices[start:end]

        # Extract activations: (batch, n_layers, d_model)
        batch_acts = model_wrapper.extract_activations(batch_sents, batch_idxs).numpy()

        # Save this shard
        shard_path = os.path.join(output_dir, f"activations_part{part_idx}.npz")
        np.savez_compressed(shard_path, activations=batch_acts)

    utils.log_info(f"Saved {part_idx+1} activation shards to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract model activations in shards.")
    parser.add_argument("--data",       type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save activation shards.")
    parser.add_argument("--model",      type=str, default="gpt2", help="Model key from config.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_and_save(args.data, args.output_dir, args.model)