import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import csv
import concurrent.futures
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from multiprocessing import Lock

from src import config, utils

tqdm.set_lock(Lock())

def closed_form_ridge_binary_predict(X_train, y_train, X_test, lambda_reg=1e-3):
    d = X_train.shape[1]
    I = torch.eye(d, device=X_train.device, dtype=X_train.dtype)
    A = X_train.t() @ X_train + lambda_reg * I
    w = torch.linalg.solve(A, X_train.t() @ y_train)
    return X_test @ w

def extract_target_representation(layer_data, target_indices):
    """
    layer_data: Tensor [n_examples, seq_len, d_model]
    target_indices: array of length n_examples
    returns Tensor [n_examples, d_model]
    """
    reps = []
    for i in range(layer_data.shape[0]):
        idx = int(target_indices[i])
        # clamp if out of bounds
        idx = min(idx, layer_data.shape[1] - 1)
        reps.append(layer_data[i, idx])
    return torch.stack(reps)

def process_inflection_layer(layer, X_layer, inflection_labels, lambda_reg, target_indices):
    n = X_layer.shape[0]
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end   = train_end + int(n * config.SPLIT_RATIOS["val"])

    if X_layer.ndim == 3:
        feats = extract_target_representation(
            torch.tensor(X_layer, dtype=torch.float32),
            target_indices
        )
    else:
        feats = torch.tensor(X_layer, dtype=torch.float32)

    X_train = feats[:train_end]
    X_test  = feats[val_end:]
    y_train = inflection_labels[:train_end]
    y_test  = inflection_labels[val_end:]

    classes = np.unique(inflection_labels)

    true_scores = []
    for c in tqdm(classes, desc=f"Layer {layer} Inflection", leave=False):
        bin_train = (y_train == c).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        true_scores.append(
            closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg)
            .unsqueeze(1)
        )
    all_true = torch.cat(true_scores, dim=1)
    pred_true = torch.argmax(all_true, dim=1).cpu().numpy()
    acc_true = (pred_true == y_test).sum() / y_test.shape[0]

    np.random.seed(config.SEED + layer + 100)
    num_cls = len(classes)
    m = {c: np.random.randint(0, num_cls) for c in classes}
    y_train_c = np.array([m[c] for c in y_train])
    y_test_c  = np.array([m[c] for c in y_test])

    control_scores = []
    control_classes = np.unique(y_train_c)
    for c in tqdm(control_classes, desc=f"Layer {layer} Control", leave=False):
        bin_train = (y_train_c == c).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        control_scores.append(
            closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg)
            .unsqueeze(1)
        )
    all_control = torch.cat(control_scores, dim=1)
    pred_control = torch.argmax(all_control, dim=1).cpu().numpy()
    acc_control = (pred_control == y_test_c).sum() / y_test_c.shape[0]

    utils.log_info(f"[Layer {layer}] Inflection Acc: {acc_true:.4f} | Control Acc: {acc_control:.4f}")
    return layer, {
        "inflection_acc": acc_true,
        "inflection_control_acc": acc_control
    }

def process_lexeme_layer(layer, X_layer, lexeme_labels, lambda_reg, lexeme_to_idx, target_indices):
    n = X_layer.shape[0]
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end   = train_end + int(n * config.SPLIT_RATIOS["val"])

    if X_layer.ndim == 3:
        feats = extract_target_representation(
            torch.tensor(X_layer, dtype=torch.float32),
            target_indices
        )
    else:
        feats = torch.tensor(X_layer, dtype=torch.float32)

    X_train = feats[:train_end]
    X_test  = feats[val_end:]
    y_train = lexeme_labels[:train_end]
    y_test  = lexeme_labels[val_end:]

    num_cls = len(lexeme_to_idx)

    true_scores = []
    for idx in tqdm(range(num_cls), desc=f"Layer {layer} Lexeme", leave=False):
        bin_train = (y_train == idx).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        true_scores.append(
            closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg)
            .unsqueeze(1)
        )
    all_true = torch.cat(true_scores, dim=1)
    pred_true = torch.argmax(all_true, dim=1).cpu().numpy()
    acc_true = (pred_true == y_test).sum() / y_test.shape[0]

    np.random.seed(config.SEED + layer + 1000)
    m = {i: np.random.randint(0, num_cls) for i in range(num_cls)}
    y_train_c = np.array([m[c] for c in y_train])
    y_test_c  = np.array([m[c] for c in y_test])

    control_scores = []
    control_classes = np.unique(y_train_c)
    for c in tqdm(control_classes, desc=f"Layer {layer} Control", leave=False):
        bin_train = (y_train_c == c).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        control_scores.append(
            closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg)
            .unsqueeze(1)
        )
    all_control = torch.cat(control_scores, dim=1)
    pred_control = torch.argmax(all_control, dim=1).cpu().numpy()
    acc_control = (pred_control == y_test_c).sum() / y_test_c.shape[0]

    utils.log_info(f"[Layer {layer}] Lexeme Acc: {acc_true:.4f} | Control Acc: {acc_control:.4f}")
    return layer, {
        "lexeme_acc": acc_true,
        "lexeme_control_acc": acc_control
    }

def plot_probe_results(results, exp_folder, task):
    os.makedirs(exp_folder, exist_ok=True)
    layers = sorted(results.keys())
    ind = np.arange(len(layers))

    if "inflection" in task:
        task_accs = [results[l]["inflection_acc"] for l in layers]
        control_accs = [results[l]["inflection_control_acc"] for l in layers]
        ylabel = "Inflection"
    else:
        task_accs = [results[l]["lexeme_acc"] for l in layers]
        control_accs = [results[l]["lexeme_control_acc"] for l in layers]
        ylabel = "Lexeme"

    # Combined overlayed bars
    plt.figure(figsize=(10, 4))
    plt.bar(ind, task_accs, label=f"{ylabel} Task")
    plt.bar(ind, control_accs, label="Control")
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{ylabel} vs Control")
    plt.xticks(ind, layers)
    plt.legend()
    plt.savefig(os.path.join(exp_folder, f"{task}_combined.png"), bbox_inches="tight")
    plt.close()

    # Linguistic only
    plt.figure(figsize=(10, 4))
    plt.bar(ind, task_accs)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{ylabel} Accuracy by Layer")
    plt.xticks(ind, layers)
    plt.savefig(os.path.join(exp_folder, f"{task}_linguistic.png"), bbox_inches="tight")
    plt.close()

    # Control only
    plt.figure(figsize=(10, 4))
    plt.bar(ind, control_accs)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Control Accuracy by Layer")
    plt.xticks(ind, layers)
    plt.savefig(os.path.join(exp_folder, f"{task}_control.png"), bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(exp_folder, f"{task}_results.csv")
    with open(csv_path, "w", newline="") as cf:
        fieldnames = ["Layer", f"{ylabel}_Task", f"{ylabel}_Control"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for i, l in enumerate(layers):
            writer.writerow({
                "Layer": l,
                f"{ylabel}_Task": task_accs[i],
                f"{ylabel}_Control": control_accs[i]
            })

def run_probes(activations_file, labels_file, task, lambda_reg, exp_label, dataset):
    data = np.load(activations_file)["activations"]
    if data.ndim not in (3, 4):
        raise ValueError(f"Unexpected activations shape: {data.shape}")

    has_seq = data.ndim == 4
    if not has_seq:
        data = data[:, None, :, :]

    n_ex, seq_len, n_layers, d_model = data.shape
    utils.log_info(f"Loaded activations (post‑fix) of shape {data.shape}")

    df = pd.read_csv(labels_file)

    if "Inflection Label" in df.columns:
        if task == "binary_inflection":
            inf_labels = np.array([(str(x).lower() == "past") for x in df["Inflection Label"]], dtype=int)
        else:
            vals = df["Inflection Label"].values
            uniq = sorted(set(vals))
            m = {v: i for i, v in enumerate(uniq)}
            inf_labels = np.array([m[v] for v in vals], dtype=int)
    else:
        inf_labels = None

    lexemes = df["Lemma"].values
    uniq_lex = sorted(set(lexemes))
    l2i = {lx: idx for idx, lx in enumerate(uniq_lex)}
    lex_labels = np.array([l2i[lx] for lx in lexemes], dtype=int)

    tgt_idx = df["Target Index"].values if has_seq and "Target Index" in df.columns else np.zeros(n_ex, dtype=int)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for L in range(n_layers):
            X_layer = data[:, :, L, :]
            if not has_seq:
                X_layer = X_layer[:, 0, :]
            if task == "lexeme":
                fut = ex.submit(process_lexeme_layer, L, X_layer, lex_labels, lambda_reg, l2i, tgt_idx)
            else:
                fut = ex.submit(process_inflection_layer, L, X_layer, inf_labels, lambda_reg, tgt_idx)
            futures[fut] = L

        for fut in tqdm(concurrent.futures.as_completed(futures), total=n_layers, desc="Processing Layers"):
            L, res = fut.result()
            results[L] = res

    outdir = os.path.join(config.OUTPUT_DIR, "probes", f"{dataset}_{exp_label}")
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "probe_results.npz"), results=results)
    utils.log_info(f"Saved raw results to {outdir}")

    plot_probe_results(results, outdir, task)
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--labels",      required=True)
    p.add_argument("--task",        required=True,
                   choices=["binary_inflection","multiclass_inflection","lexeme"])
    p.add_argument("--lambda_reg",  type=float, default=1e-3)
    p.add_argument("--exp_label",   default="default_exp")
    p.add_argument("--dataset",     required=True)
    return p.parse_args()

def main():
    args = parse_args()
    run_probes(
        args.activations,
        args.labels,
        args.task,
        args.lambda_reg,
        args.exp_label,
        args.dataset,
    )

if __name__=="__main__":
    main()
