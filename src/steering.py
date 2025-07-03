import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import json

from src import config, utils
from src.probe import MLPProbe, get_device
from src.train import load_shards, load_layer

def load_probes(probe_dir, n_layers, probe_type, input_dim):
    probes = {}
    device = get_device()
    label_map_path = os.path.join(probe_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"label_map.json not found in {probe_dir}")

    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    n_classes = len(label_map)

    for layer in range(n_layers):
        if probe_type in ["mlp", "nn"]:
            probe_path = os.path.join(probe_dir, f"probe_layer_{layer}.pt")
            if not os.path.exists(probe_path):
                utils.log_info(
                    f"Warning: Probe for layer {layer} not found at {probe_path}, skipping."
                )
                continue
            probe = MLPProbe(input_dim=input_dim, output_dim=n_classes).to(device)
            probe.load_state_dict(torch.load(probe_path, map_location=device))
            probe.eval()
            probes[layer] = probe
        elif probe_type == "rf":
            probe_path = os.path.join(probe_dir, f"probe_layer_{layer}.joblib")
            if not os.path.exists(probe_path):
                utils.log_info(
                    f"Warning: Probe for layer {layer} not found at {probe_path}, skipping."
                )
                continue
            probes[layer] = joblib.load(probe_path)
        elif probe_type == "reg":
            # For regression probes, we don't load a model but we need to acknowledge
            # the layer exists. The steering logic for 'reg' uses AIVs, not the probe model.
            # We can just add a placeholder.
            probes[layer] = None
    return probes, label_map

def compute_aivs(activations, labels, min_samples=10):
    df = pd.DataFrame({'label': labels})
    aivs = {}
    for label, group in df.groupby('label'):
        if len(group) >= min_samples:
            indices = group.index.values
            aivs[label] = activations[indices].mean(axis=0)
    return aivs

def run_steering(args):
    utils.log_info("Starting steering experiment...")
    os.makedirs(args.output_dir, exist_ok=True)

    shards = load_shards(args.activations)
    with np.load(shards[0]) as data:
        sample_activations = data['activations']
    n_layers, hidden_dim = sample_activations.shape[1], sample_activations.shape[2]

    df_labels = pd.read_csv(args.labels)
    probes, label_map = load_probes(
        args.probe_dir, n_layers, args.probe_type, hidden_dim
    )
    if not probes:
        utils.log_info("No probes found in the specified directory. Exiting.")
        return

    df_labels = df_labels[df_labels["Inflection Label"].notna()].reset_index()
    inflection_labels_str = df_labels["Inflection Label"].values
    valid_mask = np.array([s in label_map for s in inflection_labels_str])
    df_labels = df_labels[valid_mask]
    inflection_labels_str = inflection_labels_str[valid_mask]
    inflection_labels_int = np.array([label_map.index(s) for s in inflection_labels_str])

    if len(df_labels) > args.num_samples:
        target_df = df_labels.sample(n=args.num_samples, random_state=config.SEED)
    else:
        target_df = df_labels

    target_indices_in_df = target_df.index.values
    target_indices_in_activations = df_labels.index[target_indices_in_df].values

    utils.log_info(f"Selected {len(target_df)} target words for steering.")
    all_results = []

    for layer in tqdm(sorted(probes.keys()), desc="Steering Layers"):
        probe = probes[layer]
        activations = load_layer(shards, layer)
        activations = activations[valid_mask]
        aivs = compute_aivs(activations, inflection_labels_int)
        if not aivs:
            utils.log_info(
                f"Not enough samples to compute AIVs for layer {layer}. Skipping."
            )
            continue

        layer_results = []
        target_activations = activations[target_indices_in_activations]
        target_labels_int = inflection_labels_int[target_indices_in_activations]

        for i in range(len(target_df)):
            h_orig = target_activations[i]
            i_orig = target_labels_int[i]
            if i_orig not in aivs:
                continue

            if args.probe_type in ["mlp", "nn"]:
                with torch.no_grad():
                    h_orig_tensor = torch.from_numpy(h_orig).float().unsqueeze(0).to(get_device())
                    logits_orig = probe(h_orig_tensor)
                    probs_orig = torch.softmax(logits_orig, dim=1).squeeze().cpu().numpy()
            elif args.probe_type == "rf":
                probs_orig = probe.predict_proba(h_orig.reshape(1, -1))[0]
            else: # for 'reg' and any other case, we can't get initial probabilities from a probe
                probs_orig = np.zeros(n_classes)


            pred_orig = np.argmax(probs_orig)

            for i_steer in aivs.keys():
                if i_steer == i_orig:
                    continue

                steering_vec = aivs[i_steer] - aivs[i_orig]
                h_steered = h_orig + args.lambda_steer * steering_vec

                if args.probe_type in ["mlp", "nn"]:
                    with torch.no_grad():
                        h_steered_tensor = torch.from_numpy(h_steered).float().unsqueeze(0).to(get_device())
                        logits_steered = probe(h_steered_tensor)
                        probs_steered = torch.softmax(logits_steered, dim=1).squeeze().cpu().numpy()
                elif args.probe_type == "rf":
                    probs_steered = probe.predict_proba(h_steered.reshape(1, -1))[0]
                else:
                    # For 'reg', we can't get probabilities. We can set prob_change to a placeholder or skip.
                    # Let's calculate a pseudo-probability change based on dot products with AIVs.
                    # This is a simplification. A better approach might need probe weights.
                    # For now, let's just record a placeholder.
                    probs_steered = np.zeros(n_classes)


                pred_steered = np.argmax(probs_steered)
                prob_change = probs_steered[i_steer] - probs_orig[i_steer]
                
                # Original flip logic: did the prediction change from something else to the target?
                prediction_flip = (pred_orig != i_steer) and (pred_steered == i_steer)
                # New success logic: does the new prediction match the target?
                steer_success = (pred_steered == i_steer)

                layer_results.append({
                    "layer": layer,
                    "target_word_idx": target_indices_in_df[i],
                    "original_inflection": label_map[i_orig],
                    "steer_inflection": label_map[i_steer],
                    "prob_change": prob_change,
                    "prediction_flip": prediction_flip, # Kept for backward compatibility
                    "steer_success": steer_success,
                })

        if layer_results:
            all_results.extend(layer_results)

    if not all_results:
        utils.log_info("Steering experiment produced no results.")
        return

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, "steering_results.csv")
    results_df.to_csv(results_path, index=False)
    utils.log_info(f"Saved detailed steering results to {results_path}")

    summary = results_df.groupby('layer').agg(
        mean_prob_change=('prob_change', 'mean'),
        flip_rate=('steer_success', 'mean') # Use new metric for flip_rate
    ).reset_index()
    summary_path = os.path.join(args.output_dir, "steering_summary.csv")
    summary.to_csv(summary_path, index=False)
    utils.log_info(f"Saved steering summary to {summary_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run steering experiments on hidden states.")
    parser.add_argument("--activations", required=True, help="Path to activation shards directory or .npz file.")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file.")
    parser.add_argument("--probe_dir", required=True, help="Directory containing trained probes for each layer.")
    parser.add_argument("--output_dir", required=True, help="Directory to save steering results.")
    parser.add_argument("--probe_type", choices=["mlp", "nn", "rf", "reg"], default="mlp", help="Type of probe used for the experiment.")
    parser.add_argument("--lambda_steer", type=float, default=1.0, help="Lambda coefficient for steering vector strength.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of target words to use for steering.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_steering(args)
