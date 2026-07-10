import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config, utils
from src.probe import (process_layer, process_layer_mdl, process_scalarmix,
                       process_cumulative, plot_probe_results)


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


def load_all_layers(shards):
    """Return the full [N, n_layers, H] activation tensor (needed for scalar mix)."""
    parts = [np.load(shard, mmap_mode="r")["activations"] for shard in shards]
    return np.concatenate(parts, axis=0)


def run_probes(activations, labels_path, task, lambda_reg, exp_label,
               dataset, probe_type, pca_dim, output_dir=None, use_llama3_norm_flag=False):
    """
    Train probes on the provided activations and labels for a specified task. The
    activations argument should point to a directory of .npz shards produced by
    activation_extraction.py, and labels_path should point to the CSV file with
    the corresponding probing dataset. The task determines how labels and control
    labels are computed from the CSV.
    """
    pca_suffix = f"_pca_{pca_dim}" if pca_dim > 0 else ""
    outdir = output_dir or os.path.join(config.OUTPUT_DIR, "probes",
                                      f"{dataset}_{exp_label}_{task}_{probe_type}{pca_suffix}")
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
    indices = np.arange(len(df))

    if task in ["lexeme", "inflection"]:
        valid_label_mask = df["Lemma"].notna()
        if "Inflection Label" in df.columns:
            valid_label_mask &= df["Inflection Label"].notna()
        df = df[valid_label_mask].reset_index(drop=True)
        indices = indices[valid_label_mask]

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

    elif task in ["pos", "dep", "ner", "constituents", "coref", "relation", "srl", "spr"]:
        df = df[df["Label"].notna()].reset_index(drop=True)
        indices = indices[df["Label"].notna()]

        if task == "spr":
            # Filter for rows where the property applies (Label=1), then predict the property.
            df = df[df["Label"] == 1].reset_index(drop=True)
            indices = indices[df["Label"] == 1]
            if df.empty:
                raise ValueError("No positive examples found for SPR task after filtering.")
            labels = df["Property"].astype(str).values
        else:
            labels = df["Label"].astype(str).values

        uniq_labels = sorted(list(set(labels)))
        y_true = np.array([uniq_labels.index(x) for x in labels], dtype=int)
        label_map = uniq_labels

        # Define control task based on span type
        control_tokens = []
        if task in ["pos", "ner", "constituents"]:
            # Single-span tasks: lexical identity of the word at Target Index
            if "Word Form" in df.columns:
                control_tokens = df["Word Form"].astype(str).tolist()
            else:
                for _, row in df.iterrows():
                    tokens = str(row["Sentence"]).split()
                    idx = int(row["Target Index"])
                    control_tokens.append(tokens[idx] if idx < len(tokens) else "")
        elif task in ["dep", "coref", "relation"]:
            # Two-span tasks: lexical identity of the first word in Span1
            for _, row in df.iterrows():
                sent_tokens = str(row["Sentence"]).split()
                s1_start = int(row.get("Span1 Start", 0))
                token = sent_tokens[s1_start] if s1_start < len(sent_tokens) else ""
                control_tokens.append(token)
        elif task in ["srl", "spr"]:
             # SRL/SPR: lexical identity of the first word in the argument span
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

    # Filter out classes with too few examples
    true_counts = np.bincount(y_true)
    ctrl_counts = np.bincount(y_control)
    keep_true_mask = true_counts[y_true] >= 4
    keep_ctrl_mask = ctrl_counts[y_control] >= 4
    keep_mask = keep_true_mask & keep_ctrl_mask

    y_true_filtered = y_true[keep_mask]
    y_control_filtered = y_control[keep_mask]
    indices_filtered = indices[keep_mask]

    results = {}
    all_preds = []

    # ---- Whole-model probes that need ALL layers at once (Tenney et al. 2019) ----
    if probe_type in ("scalarmix", "scalarmix_mlp", "cumulative", "cumulative_mlp"):
        head = "mlp" if probe_type.endswith("_mlp") else "linear"
        X_all = load_all_layers(shards)[indices_filtered]
        y_sm, yc_sm = y_true_filtered, y_control_filtered

        max_ex = config.SCALARMIX_PARAMS["max_examples"]
        if max_ex and len(X_all) > max_ex:
            rng = np.random.RandomState(config.SEED)
            sel = rng.choice(len(X_all), max_ex, replace=False)
            X_all, y_sm, yc_sm = X_all[sel], y_sm[sel], yc_sm[sel]
            utils.log_info(f"Subsampled to {max_ex} examples.")

        if probe_type.startswith("cumulative"):
            res = process_cumulative(config.SEED, X_all, y_sm, task, head,
                                     outdir=outdir, label_map=label_map)
            key = "cumulative"
        else:
            res = process_scalarmix(config.SEED, X_all, y_sm, yc_sm, task, head,
                                    layer_count=X_all.shape[1], outdir=outdir,
                                    label_map=label_map, control_label_map=control_label_map)
            key = "scalarmix"
        np.savez_compressed(os.path.join(outdir, "probe_results.npz"), results={key: res})
        utils.log_info(f"Saved {key} results to {outdir}")
        return

    is_mdl = probe_type in ("mdl", "mdl_mlp")
    mdl_head = "mlp" if probe_type == "mdl_mlp" else "linear"

    use_llama3_norm = (
        use_llama3_norm_flag and
        exp_label in ["llama3-8b", "llama3-8b-instruct"] and
        probe_type in ["mlp", "nn"]
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
        X_filtered = X_flat[indices_filtered]

        # ---- MDL probing: per-layer online codelength (Voita & Titov 2020) ----
        if is_mdl:
            try:
                _, res, _ = process_layer_mdl(config.SEED + layer_idx, X_filtered,
                                              y_true_filtered, y_control_filtered,
                                              task, mdl_head, layer_idx)
                results[f"layer_{layer_idx}"] = res
            except Exception as e:
                utils.log_info(f"Skipping layer {layer_idx} (mdl) due to error: {e}")
            continue

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
                y_true=y_true_filtered,
                y_control=y_control_filtered,
                lambda_reg=lambda_reg,
                task=task,
                probe_type=probe_type,
                layer=layer_idx,
                pca_dim=pca_dim,
                outdir=outdir,
                indices=indices_filtered,
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
        pd.DataFrame().to_csv(predictions_path, index=False)

    if not results:
        utils.log_info("No new results were generated. Skipping results saving and plotting.")
        return

    np.savez_compressed(os.path.join(outdir, "probe_results.npz"),
                        results=results)
    utils.log_info(f"Saved probe results to {outdir}")

    if not is_mdl:
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
    parser.add_argument("--probe_type",
                        choices=["reg", "mlp", "nn", "rf",
                                 "scalarmix", "scalarmix_mlp", "mdl", "mdl_mlp",
                                 "cumulative", "cumulative_mlp"],
                        default="reg")
    parser.add_argument("--pca_dim", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for results")
    parser.add_argument("--use_llama3_norm", action="store_true",
                        help="If set, load LLaMA-3 to fetch LayerNorm weights (memory heavy).")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Cap examples for scalar-mix training (memory); 0 = use all.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.max_examples:
        config.SCALARMIX_PARAMS["max_examples"] = args.max_examples
    run_probes(
        args.activations,
        args.labels,
        args.task,
        args.lambda_reg,
        args.exp_label,
        args.dataset,
        args.probe_type,
        args.pca_dim,
        output_dir=args.output_dir,
        use_llama3_norm_flag=args.use_llama3_norm
    )