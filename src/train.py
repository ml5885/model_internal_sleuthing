import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config, utils
from src.probe import process_layer, plot_probe_results

def load_shards(path):
    if os.path.isdir(path):
        files = sorted(
            [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith(".npz") and "activations_part" in f],
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
            with np.load(shard, mmap_mode="r") as data:
                arr = data["activations"]
        except EOFError:
            raise RuntimeError(f"Corrupt or empty shard: {shard}")
        except Exception:
            with np.load(shard) as data:
                arr = data["activations"]
        parts.append(arr[:, layer_idx, :])
    return np.concatenate(parts, axis=0)

def run_probes(activations, labels, task, lambda_reg, exp_label,
               dataset, probe_type, pca_dim, output_dir=None):
    pca_suffix = f"_pca_{pca_dim}" if pca_dim > 0 else ""
    if output_dir:
        outdir = output_dir
    else:
        outdir = os.path.join(config.OUTPUT_DIR, "probes",
                              f"{dataset}_{exp_label}_{probe_type}{pca_suffix}")
    os.makedirs(outdir, exist_ok=True)
    utils.log_info(f"Probe outputs will be saved to {outdir}")

    shards = load_shards(activations)
    sample = np.load(shards[0], mmap_mode="r")["activations"]
    n_layers = sample.shape[1]

    df = pd.read_csv(labels)
    
    # If activations were sampled, filter labels to match
    activations_dir = activations if os.path.isdir(activations) else os.path.dirname(activations)
    sampled_indices_path = os.path.join(activations_dir, "sampled_indices.csv")
    if os.path.exists(sampled_indices_path):
        sampled_df = pd.read_csv(sampled_indices_path)
        original_indices = sampled_df['index'].values
        df = df.iloc[original_indices].reset_index(drop=True)
        utils.log_info(f"Loaded {len(df)} labels corresponding to sampled activations.")

    valid_label_mask = df["Lemma"].notna()
    if "Inflection Label" in df.columns:
        valid_label_mask &= df["Inflection Label"].notna()
    
    df = df[valid_label_mask].reset_index(drop=True)
    
    lemmas = df["Lemma"].values
    uniq = sorted(list(set(lemmas)))
    lex_labels = np.array([uniq.index(w) for w in lemmas], dtype=int)

    if "Inflection Label" in df.columns:
        infl = df["Inflection Label"].values
        uniq_infl = sorted(list(set(infl)))
        inf_labels = np.array([uniq_infl.index(x) for x in infl], dtype=int)
    else:
        inf_labels = lex_labels

    y_true = lex_labels if task == "lexeme" else inf_labels

    word_forms = df["Word Form"].values
    uniq_words = sorted(set(word_forms))
    y_control = np.array([uniq_words.index(w) for w in word_forms], dtype=int)

    true_counts    = np.bincount(y_true)
    ctrl_counts    = np.bincount(y_control)
    keep_true_mask = true_counts[y_true] >= 2
    keep_ctrl_mask = ctrl_counts[y_control] >= 2
    keep_mask      = keep_true_mask & keep_ctrl_mask

    y_true_filtered    = y_true[keep_mask]
    y_control_filtered = y_control[keep_mask]

    results = {}
    all_preds = []

    use_llama3_norm = (
        exp_label in ["llama3-8b", "llama3-8b-instruct"] and probe_type in ["mlp", "nn"]
    )
    model_wrapper = None
    if use_llama3_norm:
        from src.model_wrapper import ModelWrapper
        model_wrapper = ModelWrapper(exp_label)

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        X_flat = load_layer(shards, layer_idx)
        X_flat = X_flat[valid_label_mask] # Apply the same mask to activations

        if len(X_flat) != len(keep_mask):
            utils.log_info(f"Warning: X_flat size ({len(X_flat)}) does not match label size ({len(keep_mask)})")
            if len(X_flat) < len(keep_mask):
                adjusted_keep_mask = keep_mask[:len(X_flat)]
                X_filtered = X_flat[adjusted_keep_mask]
                y_true_layer = y_true[:len(X_flat)][adjusted_keep_mask]
                y_control_layer = y_control[:len(X_flat)][adjusted_keep_mask]
            else:
                X_filtered = X_flat[:len(y_true_filtered)]
                y_true_layer = y_true_filtered
                y_control_layer = y_control_filtered
        else:
            X_filtered = X_flat[keep_mask]
            y_true_layer = y_true_filtered
            y_control_layer = y_control_filtered

        norm_weight, norm_bias = None, None
        if use_llama3_norm:
            try:
                norm_weight, norm_bias = model_wrapper.get_layernorm_params(layer_idx)
            except Exception as e:
                utils.log_info(f"Could not extract LayerNorm params for layer {layer_idx+1}: {e}")

        seed = config.SEED + layer_idx
        try:
            if use_llama3_norm:
                norm_weight, norm_bias = None, None
                try:
                    norm_weight, norm_bias = model_wrapper.get_layernorm_params(layer_idx)
                except Exception as e:
                    utils.log_info(f"Could not extract LayerNorm params for layer {layer_idx+1}: {e}")
                _, res, pred_df = process_layer(
                    seed, X_filtered, y_true_layer, y_control_layer,
                    lambda_reg, task, probe_type, layer_idx,
                    pca_dim, outdir=outdir,
                    label_map=uniq_infl if task == "inflection" else uniq,
                    control_label_map=uniq_words,
                    norm_weight=norm_weight,
                    norm_bias=norm_bias
                )
            else:
                _, res, pred_df = process_layer(
                    seed, X_filtered, y_true_layer, y_control_layer,
                    lambda_reg, task, probe_type, layer_idx,
                    pca_dim, outdir=outdir,
                    label_map=uniq_infl if task == "inflection" else uniq,
                    control_label_map=uniq_words
                )
            results[f"layer_{layer_idx}"] = res
            if pred_df is not None and len(pred_df) > 0:
                all_preds.append(pred_df)
        except Exception as e:
            utils.log_info(f"Skipping layer {layer_idx} due to error: {e}")
            continue
        del X_flat, X_filtered

    # Always save predictions, even if empty
    predictions_path = os.path.join(outdir, "predictions.csv")
    if all_preds:
        try:
            combined_preds = pd.concat(all_preds, ignore_index=True)
            utils.log_info(f"Writing {len(combined_preds)} predictions to {predictions_path}")
            combined_preds.to_csv(predictions_path, index=False)
            if not os.path.isfile(predictions_path):
                raise RuntimeError(f"Failed to write predictions.csv to {predictions_path}")
            utils.log_info(f"Saved predictions to {predictions_path}")
        except Exception as e:
            utils.log_info(f"Error saving predictions: {e}")
            # Do not raise an error, just log it.
    else:
        utils.log_info(f"WARNING: No predictions to save for any layer. This may indicate an issue.")
        # Create an empty predictions file
        pd.DataFrame().to_csv(predictions_path, index=False)

    np.savez_compressed(os.path.join(outdir, "probe_results.npz"),
                        results=results)
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
    parser.add_argument("--probe_type", choices=["reg", "mlp", "nn", "rf"], default="reg")
    parser.add_argument("--pca_dim", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_probes(
        args.activations,
        args.labels,
        args.task,
        args.lambda_reg,
        args.exp_label,
        args.dataset,
        args.probe_type,
        args.pca_dim,
        output_dir=args.output_dir
    )
