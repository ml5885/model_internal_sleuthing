import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import concurrent.futures
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from src import config, utils, probe

from multiprocessing import Lock
tqdm.set_lock(Lock())

def plot_probe_results_separate(results, exp_folder):
    os.makedirs(exp_folder, exist_ok=True)
    layers = sorted(results.keys())
    
    # Separate bar plots: one for task, one for control, one combined (side-by-side)
    task_inflection = [results[layer].get("inflection_acc", np.nan) for layer in layers]
    control_inflection = [results[layer].get("inflection_control_acc", np.nan) for layer in layers]
    task_lexeme = [results[layer].get("lexeme_acc", np.nan) for layer in layers]
    control_lexeme = [results[layer].get("lexeme_control_acc", np.nan) for layer in layers]
    
    ind = np.arange(len(layers))
    width = 0.35
    
    # Inflection plot.
    plt.figure(figsize=(10, 6))
    plt.bar(ind - width/2, task_inflection, width, label="Inflection Task")
    plt.bar(ind + width/2, control_inflection, width, label="Inflection Control")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Inflection Probe Accuracies")
    plt.xticks(ind, layers)
    plt.legend()
    inflection_plot_path = os.path.join(exp_folder, "inflection_probe_accuracy.png")
    plt.savefig(inflection_plot_path, bbox_inches="tight")
    plt.close()
    utils.log_info(f"Saved inflection probe accuracy plot to {inflection_plot_path}")
    
    # Lexeme plot.
    plt.figure(figsize=(10, 6))
    plt.bar(ind - width/2, task_lexeme, width, label="Lexeme Task")
    plt.bar(ind + width/2, control_lexeme, width, label="Lexeme Control")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Lexeme Probe Accuracies")
    plt.xticks(ind, layers)
    plt.legend()
    lexeme_plot_path = os.path.join(exp_folder, "lexeme_probe_accuracy.png")
    plt.savefig(lexeme_plot_path, bbox_inches="tight")
    plt.close()
    utils.log_info(f"Saved lexeme probe accuracy plot to {lexeme_plot_path}")
    
    # Combined plot.
    plt.figure(figsize=(10, 6))
    plt.bar(ind - width/2, task_inflection, width, label="Inflection Task")
    plt.bar(ind - width/2, control_inflection, width, bottom=task_inflection, label="Inflection Control")
    plt.bar(ind + width/2, task_lexeme, width, label="Lexeme Task")
    plt.bar(ind + width/2, control_lexeme, width, bottom=task_lexeme, label="Lexeme Control")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Combined Probe Accuracies")
    plt.xticks(ind, layers)
    plt.legend()
    combined_plot_path = os.path.join(exp_folder, "combined_probe_accuracy.png")
    plt.savefig(combined_plot_path, bbox_inches="tight")
    plt.close()
    utils.log_info(f"Saved combined probe accuracy plot to {combined_plot_path}")
    
    # Save CSV summary.
    csv_rows = []
    for layer in layers:
        row = {
            "Layer": layer,
            "Inflection_Task": results[layer].get("inflection_acc", ""),
            "Inflection_Control": results[layer].get("inflection_control_acc", ""),
            "Lexeme_Task": results[layer].get("lexeme_acc", ""),
            "Lexeme_Control": results[layer].get("lexeme_control_acc", "")
        }
        csv_rows.append(row)
    csv_path = os.path.join(exp_folder, "probe_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["Layer", "Inflection_Task", "Inflection_Control", "Lexeme_Task", "Lexeme_Control"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    utils.log_info(f"Saved probe results CSV to {csv_path}")

def closed_form_ridge_binary_predict(X_train, y_train, X_test, lambda_reg=1e-3):
    d = X_train.shape[1]
    I = torch.eye(d, device=X_train.device, dtype=X_train.dtype)
    A = X_train.t() @ X_train + lambda_reg * I
    w = torch.linalg.solve(A, X_train.t() @ y_train)
    scores = X_test @ w
    return scores

def process_inflection_layer(layer, X_layer, inflection_labels, d_model, lambda_reg):
    n = X_layer.shape[0]
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])
    # True task run.
    X_train = torch.tensor(X_layer[:train_end], dtype=torch.float32)
    X_test = torch.tensor(X_layer[val_end:], dtype=torch.float32)
    y_train_true = inflection_labels[:train_end]
    y_test_true = inflection_labels[val_end:]
    classes = np.unique(inflection_labels)
    true_scores = []
    for c in tqdm(classes, desc=f"Layer {layer} Inflection Task", leave=False, dynamic_ncols=True):
        binary_train = (y_train_true == c).astype(np.float32)
        y_train_tensor = torch.tensor(binary_train, dtype=torch.float32)
        scores = closed_form_ridge_binary_predict(X_train, y_train_tensor, X_test, lambda_reg=lambda_reg)
        true_scores.append(scores.unsqueeze(1))
    all_true_scores = torch.cat(true_scores, dim=1)
    y_pred_true = torch.argmax(all_true_scores, dim=1)
    y_test_true_tensor = torch.tensor(y_test_true, dtype=torch.long)
    true_acc = accuracy_score(y_test_true_tensor.cpu().numpy(), y_pred_true.cpu().numpy())
    
    # Control run with random labels.
    num_classes = len(classes)
    y_train_control = np.random.randint(0, num_classes, size=y_train_true.shape)
    y_test_control = np.random.randint(0, num_classes, size=y_test_true.shape)
    control_scores = []
    for c in tqdm(classes, desc=f"Layer {layer} Inflection Control", leave=False, dynamic_ncols=True):
        binary_train = (y_train_control == c).astype(np.float32)
        y_train_tensor = torch.tensor(binary_train, dtype=torch.float32)
        scores = closed_form_ridge_binary_predict(X_train, y_train_tensor, X_test, lambda_reg=lambda_reg)
        control_scores.append(scores.unsqueeze(1))
    all_control_scores = torch.cat(control_scores, dim=1)
    y_pred_control = torch.argmax(all_control_scores, dim=1)
    y_test_control_tensor = torch.tensor(y_test_control, dtype=torch.long)
    control_acc = accuracy_score(y_test_control_tensor.cpu().numpy(), y_pred_control.cpu().numpy())
    
    utils.log_info(f"[Layer {layer}] Inflection Task Acc: {true_acc:.4f} | Control Acc: {control_acc:.4f}")
    return layer, {"inflection_acc": true_acc, "inflection_control_acc": control_acc}

def process_lexeme_layer(layer, X_layer, lexeme_labels, d_model, lambda_reg, lexeme_to_idx):
    n = X_layer.shape[0]
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])
    # True task run.
    X_train = torch.tensor(X_layer[:train_end], dtype=torch.float32)
    X_test = torch.tensor(X_layer[val_end:], dtype=torch.float32)
    y_train_true = lexeme_labels[:train_end]
    y_test_true = lexeme_labels[val_end:]
    classes = sorted(lexeme_to_idx.keys(), key=lambda k: lexeme_to_idx[k])
    true_scores = []
    for idx in tqdm(range(len(classes)), desc=f"Layer {layer} Lexeme Task", leave=False, dynamic_ncols=True):
        binary_train = (y_train_true == idx).astype(np.float32)
        y_train_tensor = torch.tensor(binary_train, dtype=torch.float32)
        scores = closed_form_ridge_binary_predict(X_train, y_train_tensor, X_test, lambda_reg=lambda_reg)
        true_scores.append(scores.unsqueeze(1))
    all_true_scores = torch.cat(true_scores, dim=1)
    y_pred_true = torch.argmax(all_true_scores, dim=1)
    y_test_true_tensor = torch.tensor(y_test_true, dtype=torch.long)
    true_acc = accuracy_score(y_test_true_tensor.cpu().numpy(), y_pred_true.cpu().numpy())
    
    # Control run with random labels.
    num_classes = len(lexeme_to_idx)
    y_train_control = np.random.randint(0, num_classes, size=y_train_true.shape)
    y_test_control = np.random.randint(0, num_classes, size=y_test_true.shape)
    control_scores = []
    for idx in tqdm(range(len(classes)), desc=f"Layer {layer} Lexeme Control", leave=False, dynamic_ncols=True):
        binary_train = (y_train_control == idx).astype(np.float32)
        y_train_tensor = torch.tensor(binary_train, dtype=torch.float32)
        scores = closed_form_ridge_binary_predict(X_train, y_train_tensor, X_test, lambda_reg=lambda_reg)
        control_scores.append(scores.unsqueeze(1))
    all_control_scores = torch.cat(control_scores, dim=1)
    y_pred_control = torch.argmax(all_control_scores, dim=1)
    y_test_control_tensor = torch.tensor(y_test_control, dtype=torch.long)
    control_acc = accuracy_score(y_test_control_tensor.cpu().numpy(), y_pred_control.cpu().numpy())
    
    utils.log_info(f"[Layer {layer}] Lexeme Task Acc: {true_acc:.4f} | Control Acc: {control_acc:.4f}")
    return layer, {"lexeme_acc": true_acc, "lexeme_control_acc": control_acc}

def run_probes(activations_file, labels_file, task, lambda_reg=1e-3, exp_label="default_exp", dataset="default"):
    data = np.load(activations_file)["activations"]
    n_examples, n_layers, d_model = data.shape
    utils.log_info(f"Loaded activations of shape {data.shape}")
    
    df = pd.read_csv(labels_file)
    if task in ["binary_inflection", "multiclass_inflection"]:
        if task == "binary_inflection":
            inflection_labels = np.array([1 if str(x).lower() == "past" else 0 for x in df["Inflection Label"].values])
        else:
            inflection_labels = df["Inflection Label"].values
            unique_inflections = sorted(set(inflection_labels))
            inflection_to_idx = {inf: idx for idx, inf in enumerate(unique_inflections)}
            inflection_labels = np.array([inflection_to_idx[inf] for inf in inflection_labels])
    else:
        inflection_labels = None

    lexemes = df["Lemma"].values
    lexeme_to_idx = {lex: idx for idx, lex in enumerate(sorted(set(lexemes)))}
    lexeme_labels = np.array([lexeme_to_idx[lex] for lex in lexemes])
    
    utils.log_info(f"Inflection classes: {len(np.unique(inflection_labels)) if inflection_labels is not None else 'N/A'}, Lexeme classes: {len(lexeme_to_idx)}")
    
    results = {}
    max_workers_layer = 4
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_layer) as executor:
            if task == "lexeme":
                layer_futures = {
                    executor.submit(process_lexeme_layer, layer, data[:, layer, :],
                                    lexeme_labels, d_model, lambda_reg, lexeme_to_idx): layer
                    for layer in range(n_layers)
                }
            else:
                layer_futures = {
                    executor.submit(process_inflection_layer, layer, data[:, layer, :],
                                    inflection_labels, d_model, lambda_reg): layer
                    for layer in range(n_layers)
                }
            outer_pbar = tqdm(total=len(layer_futures), desc="Processing Layers", position=0, dynamic_ncols=True)
            for future in concurrent.futures.as_completed(layer_futures):
                layer, layer_result = future.result()
                results[layer] = layer_result
                outer_pbar.update(1)
            outer_pbar.close()
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    exp_folder = os.path.join(config.OUTPUT_DIR, "probes", f"{dataset}_{exp_label}")
    os.makedirs(exp_folder, exist_ok=True)
    output_path = os.path.join(exp_folder, "probe_results.npz")
    np.savez_compressed(output_path, results=results)
    utils.log_info(f"Probe results saved to {output_path}")
    plot_probe_results_separate(results, exp_folder)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train probes on activations using one-vs-rest binary closed-form ridge regression for inflection and lexeme classification."
    )
    parser.add_argument("--activations", type=str, required=True, help="Path to NPZ file with activations.")
    parser.add_argument("--labels", type=str, required=True, help="Path to CSV file with labels.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["binary_inflection", "multiclass_inflection", "lexeme"],
                        help="Probe task: 'binary_inflection', 'multiclass_inflection', or 'lexeme'.")
    parser.add_argument("--lambda_reg", type=float, default=1e-3,
                        help="Regularization strength for closed-form ridge regression.")
    parser.add_argument("--exp_label", type=str, default="default_exp",
                        help="Experiment label (for output folder naming).")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset label (e.g. controlled, wikitext, combined).")
    parser.add_argument("--control_inflection", action="store_true", help="Run control experiment for inflection probe.")
    parser.add_argument("--control_lexeme", action="store_true", help="Run control experiment for lexeme probe.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_probes(
        args.activations, args.labels, args.task, args.lambda_reg,
        args.exp_label, args.dataset
    )
