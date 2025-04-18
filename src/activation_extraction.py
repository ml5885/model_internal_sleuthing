import argparse
import os
import numpy as np
import pandas as pd
import torch
import zipfile
from tqdm import tqdm
from numpy.lib.format import write_array_header_1_0
from src.model_wrapper import ModelWrapper
from src import config, utils


def extract_and_save(data_path, output_path, model_key):
    df = pd.read_csv(data_path)
    sentences = df["Sentence"].tolist()
    target_indices = df["Target Index"].tolist()

    model_wrapper = ModelWrapper(model_key)
    batch_size = config.MODEL_CONFIGS[model_key]["batch_size"]
    total = len(sentences)
    if total == 0:
        utils.log_error("Input data is empty, no activations to extract.")
        return

    # Open a compressed .npz (zip) and write the .npy entry incrementally
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # We'll write a single entry 'activations.npy' inside the zip
        with zf.open('activations.npy', mode='w') as arr_f:
            # Process batches and stream raw data
            for start in tqdm(range(0, total, batch_size), desc="Extracting Batches"):
                end = min(start + batch_size, total)
                batch_sents = sentences[start:end]
                batch_idxs = target_indices[start:end]

                # Extract activations (batch_size, n_layers, d_model)
                batch_acts = model_wrapper.extract_activations(batch_sents, batch_idxs).numpy()

                if start == 0:
                    # On first batch, write the .npy header for the full array
                    n_layers = batch_acts.shape[1]
                    d_model = batch_acts.shape[2]
                    header = {
                        'descr': np.lib.format.dtype_to_descr(batch_acts.dtype),
                        'fortran_order': False,
                        'shape': (total, n_layers, d_model)
                    }
                    write_array_header_1_0(arr_f, header)

                # Stream the raw bytes of this batch
                arr_f.write(batch_acts.tobytes(order='C'))

    utils.log_info(f"Saved compressed activations to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract model activations from sentences.")
    parser.add_argument("--data",    type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output",  type=str, required=True, help="Output NPZ file path.")
    parser.add_argument("--model",   type=str, default="gpt2", help="Model key from config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_and_save(args.data, args.output, args.model)
