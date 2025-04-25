import argparse
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config, utils
from src.probe import process_layer, plot_probe_results

def load_shards(path):
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".npz") and "activations_part" in f
        ]
        files.sort(
            key=lambda fn: int(re.search(r"part_?(\d+)", os.path.basename(fn)).group(1))
        )
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise ValueError(f"{path} is not a .npz file or directory of shards")

def load_layer(shards, layer_idx):
    parts = []
    for shard in shards:
        try:
            arr = np.load(shard, mmap_mode="r")["activations"]
        except Exception as e:
            utils.log_info(
                f"Warning: failed mmap load of {shard}: {e}, retrying without mmap"
            )
            arr = np.load(shard)["activations"]
        parts.append(arr[:, layer_idx, :])
    return np.concatenate(parts, axis=0)

def run_probes(activations, labels, task, lambda_reg, exp_label, dataset, probe_type):
    shards = load_shards(activations)
    sample = np.load(shards[0], mmap_mode="r")["activations"]
    n_layers = sample.shape[1]

    df = pd.read_csv(labels)
    lemmas = df["Lemma"].values
    uniq = sorted(set(lemmas))
    lex_labels = np.array([uniq.index(w) for w in lemmas], dtype=int)

    if "Inflection Label" in df.columns:
        infl = df["Inflection Label"].values
        uniq_infl = sorted(set(infl))
        inf_labels = np.array([uniq_infl.index(x) for x in infl], dtype=int)
    else:
        inf_labels = lex_labels

    y_true = lex_labels if task == "lexeme" else inf_labels
    y_ctrl = lex_labels if task == "lexeme" else inf_labels
    results = {}

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        X_flat = load_layer(shards, layer_idx)
        seed = config.SEED + layer_idx
        _, res = process_layer(seed, X_flat, y_true, y_ctrl, lambda_reg, 
                               task, probe_type, layer_idx)
        del X_flat
        results[f"layer_{layer_idx}"] = res

    outdir = os.path.join(
        config.OUTPUT_DIR, "probes", f"{dataset}_{exp_label}_{probe_type}"
    )
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "probe_results.npz"), results=results)
    utils.log_info(f"Saved probe results to {outdir}")
    plot_probe_results(results, outdir, task)

def parse_args():
    parser = argparse.ArgumentParser(description="Train probes on activations")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task", required=True, choices=["inflection", "lexeme"])
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--exp_label", default="exp")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--probe_type", choices=["reg", "mlp"], default="reg")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_probes(args.activations, args.labels, args.task, args.lambda_reg, 
               args.exp_label, args.dataset, args.probe_type)
