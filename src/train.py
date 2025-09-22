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
    """Return a sorted list of activation shard files in a directory or the single file itself."""
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
        # mmap_mode keeps the shard on disk; no in‑RAM copy
        parts.append(np.load(shard, mmap_mode="r")["activations"][:, layer_idx, :])
    return np.concatenate(parts, axis=0)


def run_probes(activations, labels_path, task, lambda_reg, exp_label,
               dataset, probe_type, pca_dim, output_dir=None):
    """
    Train probes on the provided activations and labels for a specified task. The
    activations argument should point to a directory of .npz shards produced by
    activation_extraction.py, and labels_path should point to the CSV file with
    the corresponding probing dataset. The task determines how labels and control
    labels are computed from the CSV.
    """
    pca_suffix = f"_pca_{pca_dim}" if pca_dim > 0 else ""
    outdir = output_dir or os.path.join(config.OUTPUT_DIR, "probes",
                               f"{dataset}_{exp_label}_{probe_type}{pca_suffix}")
    os.makedirs(outdir, exist_ok=True)
    utils.log_info(f"Probe outputs will be saved to {outdir}")

    shards = load_shards(activations)
    sample = np.load(shards[0], mmap_mode="r")["activations"]
    n_layers = sample.shape[1]

    # Load labels dataframe
    df = pd.read_csv(labels_path)

    # If activations were sampled, filter labels to match
    activations_dir = activations if os.path.isdir(activations) else os.path.dirname(activations)
    sampled_indices_path = os.path.join(activations_dir, "sampled_indices.csv")
    if os.path.exists(sampled_indices_path):
        sampled_df = pd.read_csv(sampled_indices_path)
        original_indices = sampled_df['index'].values
        df = df.iloc[original_indices].reset_index(drop=True)
        utils.log_info(f"Loaded {len(df)} labels corresponding to sampled activations.")

    # Prepare y_true and y_control based on task
    y_true = None
    y_control = None
    label_map = None
    control_label_map = None

    if task in ["lexeme", "inflection"]:
        # Original lexical/inflection tasks
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
        label_map = uniq_infl if task == "inflection" else uniq
        control_label_map = uniq_words

    elif task in ["pos", "dep", "ner", "constituents"]:
        # Single-span classification tasks. Use the 'Label' column for y_true.
        if "Label" not in df.columns:
            raise ValueError(f"Dataset for task {task} must contain a 'Label' column.")
        labels = df["Label"].astype(str).values
        uniq_labels = sorted(list(set(labels)))
        y_true = np.array([uniq_labels.index(x) for x in labels], dtype=int)
        # Control: lexical identity of the word form, if available; else use token at Target Index
        control_tokens = []
        if "Word Form" in df.columns:
            control_tokens = df["Word Form"].astype(str).tolist()
        else:
            # Fallback: derive the word form from sentence and target index
            if "Target Index" not in df.columns:
                raise ValueError(f"Dataset for task {task} must contain 'Word Form' or 'Target Index'.")
            for _, row in df.iterrows():
                tokens = str(row["Sentence"]).split()
                idx = int(row["Target Index"])
                control_tokens.append(tokens[idx] if idx < len(tokens) else "")
        uniq_controls = sorted(list(set(control_tokens)))
        y_control = np.array([uniq_controls.index(w) for w in control_tokens], dtype=int)
        label_map = uniq_labels
        control_label_map = uniq_controls

    elif task == "coref":
        # Coreference: binary label, two spans
        if "Label" not in df.columns:
            raise ValueError("Coref dataset must contain a 'Label' column.")
        # y_true is 0/1
        y_true = df["Label"].astype(int).values
        label_map = [0, 1]
        # Control: lexical identity of first word in Span1
        control_tokens = []
        for _, row in df.iterrows():
            sent_tokens = str(row["Sentence"]).split()
            s1_start = int(row.get("Span1 Start", 0))
            token = sent_tokens[s1_start] if s1_start < len(sent_tokens) else ""
            control_tokens.append(token)
        uniq_controls = sorted(list(set(control_tokens)))
        y_control = np.array([uniq_controls.index(w) for w in control_tokens], dtype=int)
        control_label_map = uniq_controls

    elif task == "relation":
        # Relation classification: multi-class labels based on relation string
        if "Label" not in df.columns:
            raise ValueError("Relation classification dataset must contain a 'Label' column.")
        labels = df["Label"].astype(str).values
        uniq_labels = sorted(list(set(labels)))
        y_true = np.array([uniq_labels.index(x) for x in labels], dtype=int)
        label_map = uniq_labels
        # Control: lexical identity of first word in Span1
        control_tokens = []
        for _, row in df.iterrows():
            sent_tokens = str(row["Sentence"]).split()
            s1_start = int(row.get("Span1 Start", 0))
            token = sent_tokens[s1_start] if s1_start < len(sent_tokens) else ""
            control_tokens.append(token)
        uniq_controls = sorted(list(set(control_tokens)))
        y_control = np.array([uniq_controls.index(w) for w in control_tokens], dtype=int)
        control_label_map = uniq_controls

    elif task == "srl":
        # Semantic role labeling: label is argument role (e.g., ARG0). Use lexical identity of argument as control.
        if "Label" not in df.columns:
            raise ValueError("SRL dataset must contain a 'Label' column.")
        labels = df["Label"].astype(str).values
        uniq_labels = sorted(list(set(labels)))
        y_true = np.array([uniq_labels.index(x) for x in labels], dtype=int)
        label_map = uniq_labels
        # Control: lexical identity of first word in argument span
        control_tokens = []
        for _, row in df.iterrows():
            sent_tokens = str(row["Sentence"]).split()
            arg_start = int(row.get("Arg Start", 0))
            token = sent_tokens[arg_start] if arg_start < len(sent_tokens) else ""
            control_tokens.append(token)
        uniq_controls = sorted(list(set(control_tokens)))
        y_control = np.array([uniq_controls.index(w) for w in control_tokens], dtype=int)
        control_label_map = uniq_controls

    elif task == "spr":
        # Semantic proto‑role classification. Each row represents a specific
        # proto‑role property with an associated binary label (1 if the property
        # applies to the argument span, 0 otherwise). We use the binary
        # indicator (Label column) as the true task label. The Property name
        # itself is not the class; instead, it is metadata describing which
        # proto‑role is being probed. This mirrors the edge‑probing setup from
        # the original paper, where each proto‑role property is treated as a
        # separate binary classification problem.
        if "Label" not in df.columns:
            raise ValueError("SPR dataset must contain a 'Label' column (binary indicator).")
        # Convert labels to integer 0/1 values
        labels = df["Label"].astype(int).values
        uniq_labels = sorted(list(set(labels)))
        y_true = np.array([uniq_labels.index(x) for x in labels], dtype=int)
        label_map = uniq_labels
        # Control: lexical identity of the first token in the argument span.
        # We do not differentiate by property here; the control task is
        # intended to capture lexical memorization of the argument.
        control_tokens = []
        for _, row in df.iterrows():
            sent_tokens = str(row["Sentence"]).split()
            arg_start = int(row.get("Arg Start", 0))
            token = sent_tokens[arg_start] if arg_start < len(sent_tokens) else ""
            control_tokens.append(token)
        uniq_controls = sorted(list(set(control_tokens)))
        y_control = np.array([uniq_controls.index(w) for w in control_tokens], dtype=int)
        control_label_map = uniq_controls

    else:
        raise ValueError(f"Unknown task: {task}")

    # Filter out classes with too few examples in either the true task or the control task
    true_counts = np.bincount(y_true)
    ctrl_counts = np.bincount(y_control)
    # For classes with count < 4, drop those examples
    keep_true_mask = true_counts[y_true] >= 4
    keep_ctrl_mask = ctrl_counts[y_control] >= 4
    keep_mask = keep_true_mask & keep_ctrl_mask

    y_true_filtered    = y_true[keep_mask]
    y_control_filtered = y_control[keep_mask]

    results = {}
    all_preds = []

    # Optionally load LayerNorm weights for LLaMA3 models
    use_llama3_norm = (
        exp_label in ["llama3-8b", "llama3-8b-instruct"] and probe_type in ["mlp", "nn"]
    )
    model_wrapper = None
    if use_llama3_norm:
        from src.model_wrapper import ModelWrapper
        model_wrapper = ModelWrapper(exp_label)

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        # Skip if probe already exists
        probe_exists = False
        if probe_type in ["mlp", "nn"]:
            probe_model_path = os.path.join(outdir, f"probe_layer_{layer_idx}.pt")
            if os.path.exists(probe_model_path):
                probe_exists = True
        elif probe_type == "rf":
            probe_model_path = os.path.join(outdir, f"probe_layer_{layer_idx}.joblib")
            if os.path.exists(probe_model_path):
                probe_exists = True

        if probe_exists:
            utils.log_info(f"Probe for layer {layer_idx} already exists, skipping training.")
            continue

        # Load activations for this layer
        X_flat = load_layer(shards, layer_idx)
        # Apply the same keep mask to activations
        X_filtered = X_flat[keep_mask]
        y_true_layer = y_true_filtered
        y_control_layer = y_control_filtered

        # Optionally extract LayerNorm weight for normalization
        norm_weight = None
        if use_llama3_norm:
            try:
                norm_weight = model_wrapper.get_layernorm_params(layer_idx)
            except Exception as e:
                utils.log_info(f"Could not extract LayerNorm params for layer {layer_idx+1}: {e}")

        seed = config.SEED + layer_idx
        try:
            process_layer_kwargs = dict(
                seed=seed,
                X_flat=X_filtered,
                y_true=y_true_layer,
                y_control=y_control_layer,
                lambda_reg=lambda_reg,
                task=task,
                probe_type=probe_type,
                layer=layer_idx,
                pca_dim=pca_dim,
                outdir=outdir,
                label_map=label_map,
                control_label_map=control_label_map,
            )
            if norm_weight is not None:
                process_layer_kwargs["norm_weight"] = norm_weight
            _, res, pred_df = process_layer(**process_layer_kwargs)
            results[f"layer_{layer_idx}"] = res
            if pred_df is not None and len(pred_df) > 0:
                all_preds.append(pred_df)
        except Exception as e:
            utils.log_info(f"Skipping layer {layer_idx} due to error: {e}")

    # Save predictions
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
    else:
        utils.log_info(f"WARNING: No predictions to save for any layer. This may indicate an issue.")
        # Create an empty predictions file
        pd.DataFrame().to_csv(predictions_path, index=False)

    if not results:
        utils.log_info("No new results were generated. Skipping results saving and plotting.")
        return

    np.savez_compressed(os.path.join(outdir, "probe_results.npz"),
                        results=results)
    utils.log_info(f"Saved probe results to {outdir}")

    plot_probe_results(results, outdir, task)


def parse_args():
    parser = argparse.ArgumentParser(description="Train probes on activations")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task", required=True,
                        choices=["lexeme", "inflection", "pos", "dep", "ner", "coref", "constituents", "srl", "spr", "relation"])
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