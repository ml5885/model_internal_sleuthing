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

def extract_and_save(data_path, output_dir, model_key, save_combined=False):
    """
    Extract hidden-state activations for each target word and save them in
    compressed .npz shards.  If save_combined=True, also concatenate all
    shards into a single output_dir.npz at the end.
    """
    # batch size from your config
    batch_size = config.MODEL_CONFIGS[model_key]["batch_size"]

    # stream through the CSV in chunks of `batch_size`
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
    model_wrapper = ModelWrapper(model_key)
    shard_paths = []

    for part_idx, chunk in enumerate(tqdm(reader,
                                          desc="Extracting Batches",
                                          total=total,
                                          dynamic_ncols=True,
                                          leave=True,
                                          file=sys.stdout)):
        sentences = chunk["Sentence"].tolist()
        target_indices = chunk["Target Index"].tolist()

        # sanity check lengths
        if len(sentences) != len(target_indices):
            raise ValueError(f"Chunk #{part_idx}: sentence/count mismatch")

        # clamp any out‐of‐range indices
        for i, (sent, idx) in enumerate(zip(sentences, target_indices)):
            n_words = len(sent.split())
            if idx < 0 or idx >= n_words:
                utils.log_error(
                    f"Row {part_idx*batch_size + i}: "
                    f"index {idx} out of [0, {n_words}) - clamping to valid range."
                )
                target_indices[i] = max(0, min(idx, n_words - 1))

        # extract on GPU if available (with no_grad)
        with torch.no_grad():
            activations = model_wrapper.extract_activations(sentences, target_indices)

        # bring to CPU numpy
        batch_array = activations.cpu().numpy()

        # zero‑pad the part index so shard_00001.npz sorts after shard_00002, etc.
        fname = f"activations_part_{part_idx:05d}.npz"
        path = os.path.join(output_dir, fname)
        np.savez_compressed(path, activations=batch_array)
        shard_paths.append(path)

    utils.log_info(f"Saved {len(shard_paths)} activation shards to {output_dir}")

    # optional: spit out one combined .npz for downstream code that expects it
    if save_combined and shard_paths:
        utils.log_info("Combining all shards into one big .npz (this will use memory)...")
        arrays = [np.load(p)["activations"] for p in shard_paths]
        combined = np.concatenate(arrays, axis=0)
        combined_path = output_dir.rstrip(os.sep) + ".npz"
        np.savez_compressed(combined_path, activations=combined)
        utils.log_info(f"Saved combined activations to {combined_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model activations in shards (and optional combined file).")
    parser.add_argument("--data", "-d", type=str, required=True, help="CSV with columns 'Sentence' and 'Target Index'.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Directory in which to write activation_part_xxxxx.npz files.")
    parser.add_argument("--model", "-m", type=str, default="gpt2", help="Key into MODEL_CONFIGS (e.g. 'gpt2' or 'gemma2b').")
    parser.add_argument("--combined", action="store_true", help="Also create one combined .npz at output-dir.npz")
    args = parser.parse_args()

    extract_and_save(
        data_path=args.data,
        output_dir=args.output_dir,
        model_key=args.model,
        save_combined=args.combined
    )
