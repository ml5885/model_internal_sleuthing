import argparse
import os
import numpy as np
import pandas as pd
import torch
import sys
import math
from tqdm import tqdm
from src.model_wrapper import ModelWrapper
from src import config, utils

def extract_and_save(data_path, output_dir, model_key, revision=None): 
    """
    Extract hidden-state activations for each target word and save them in
    compressed .npz shards.
    """
    batch_size = config.MODEL_CONFIGS[model_key]["batch_size"]

    num_rows = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        next(f)
        for _ in f:
            num_rows += 1
            
    total = math.ceil(num_rows / batch_size)
    reader = pd.read_csv(
        data_path,
        usecols=["Sentence", "Target Index"],
        chunksize=batch_size
    )

    os.makedirs(output_dir, exist_ok=True)
    model_wrapper = ModelWrapper(model_key, revision=revision) 
    shard_paths = []

    for part_idx, chunk in enumerate(tqdm(reader,desc="Extracting Batches",
                                          total=total,dynamic_ncols=True,
                                          leave=True, file=sys.stdout)):
        sentences = chunk["Sentence"].tolist()
        target_indices = chunk["Target Index"].tolist()

        with torch.no_grad():
            activations = model_wrapper.extract_activations(sentences, target_indices)

        batch_array = activations.cpu().numpy()

        fname = f"activations_part_{part_idx:05d}.npz"
        path = os.path.join(output_dir, fname)
        np.savez_compressed(path, activations=batch_array)
        shard_paths.append(path)

    utils.log_info(f"Saved {len(shard_paths)} activation shards to {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model activations in shards (and optional combined file).")
    parser.add_argument("--data", "-d", type=str, required=True, help="CSV with columns 'Sentence' and 'Target Index'.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Directory in which to write activation_part_xxxxx.npz files.")
    parser.add_argument("--model", "-m", type=str, default="gpt2", help="Key into MODEL_CONFIGS (e.g. 'gpt2' or 'gemma2b').")
    parser.add_argument("--revision", type=str, default=None, help="Model revision or checkpoint (e.g., 'step1000', 'main').") 
    args = parser.parse_args()

    extract_and_save(
        data_path=args.data,
        output_dir=args.output_dir,
        model_key=args.model,
        revision=args.revision
    )