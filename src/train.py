import argparse
import os
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Lock
from src import config, utils
from src.probe import process_inflection_layer, process_lexeme_layer, plot_probe_results

tqdm.set_lock(Lock())

def closed_form_ridge_binary_predict(X_train, y_train, X_test, lambda_reg=1e-3):
    d = X_train.shape[1]
    I = torch.eye(d, device=X_train.device, dtype=X_train.dtype)
    A = X_train.t() @ X_train + lambda_reg * I
    w = torch.linalg.solve(A, X_train.t() @ y_train)
    return X_test @ w

def extract_target_representation(layer_data, target_indices):
    reps = []
    for i in range(layer_data.shape[0]):
        idx = int(target_indices[i])
        idx = min(idx, layer_data.shape[1] - 1)
        reps.append(layer_data[i, idx])
    return torch.stack(reps)

def run_probes(activations_input, labels_file, task, lambda_reg, exp_label, dataset):
    if os.path.isdir(activations_input):
        shard_files = sorted(
            [os.path.join(activations_input, f) for f in os.listdir(activations_input)
             if f.endswith('.npz') and 'activations_part' in f]
        )
    else:
        shard_files = [activations_input]

    # Inspect first shard to get number of layers
    sample = np.load(shard_files[0])['activations']
    n_layers = sample.shape[1]

    # Read labels
    df = pd.read_csv(labels_file)
    lexemes = df["Lemma"].values
    uniq_lex = sorted(set(lexemes))
    l2i = {lx: idx for idx, lx in enumerate(uniq_lex)}
    lex_labels = np.array([l2i[lx] for lx in lexemes], dtype=int)

    if "Inflection Label" in df.columns:
        vals = df["Inflection Label"].values
        uniq_inf = sorted(set(vals))
        inf_labels = np.array([uniq_inf.index(v) for v in vals], dtype=int)
    else:
        inf_labels = None

    # Target indices if present
    tgt_idx = df.get("Target Index", pd.Series(0)).astype(int).values

    utils.log_info(f"Streaming activations for {n_layers} layers from {len(shard_files)} shards")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS_CLASS) as executor:
        futures = {}
        for L in range(n_layers):
            # Build X_layer by concatenating only this layer from each shard
            layer_slices = []
            for f in shard_files:
                arr = np.load(f)['activations']
                layer_slices.append(arr[:, L, :])
            X_layer = np.concatenate(layer_slices, axis=0)

            if task == "lexeme":
                fut = executor.submit(
                    process_lexeme_layer, L, X_layer, lex_labels, lambda_reg, l2i, tgt_idx
                )
            else:
                fut = executor.submit(
                    process_inflection_layer, L, X_layer, inf_labels, lambda_reg, tgt_idx
                )
            futures[fut] = L

        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=n_layers, desc="Processing Layers"):
            L, res = fut.result()
            results[L] = res

    # Save and plot
    outdir = os.path.join(config.OUTPUT_DIR, "probes", f"{dataset}_{exp_label}")
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "probe_results.npz"), results=results)
    utils.log_info(f"Saved raw probe results to {outdir}")
    plot_probe_results(results, outdir, task)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--task", required=True, choices=["binary_inflection",
                                                     "multiclass_inflection",
                                                     "lexeme"])
    p.add_argument("--lambda_reg", type=float, default=1e-3)
    p.add_argument("--exp_label", default="default_exp")
    p.add_argument("--dataset", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    run_probes(args.activations, args.labels, args.task, 
               args.lambda_reg, args.exp_label, args.dataset)

if __name__=="__main__":
    main()
