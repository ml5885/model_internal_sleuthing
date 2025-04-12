import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import concurrent.futures
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from src import config, utils

class LinearProbe(nn.Module):
    """A standard linear probe."""
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class KSparseLinearProbe(nn.Module):
    """
    A k-sparse linear probe that retains only the top-k coefficients (by absolute value)
    in each row of the weight matrix after each optimizer step.
    
    Reference:
        Gurnee et al. (2023). Finding Neurons in a Haystack: Case Studies with Sparse Probing.
    """
    def __init__(self, input_dim: int, output_dim: int, k: int):
        super(KSparseLinearProbe, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.k = k
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)
        
    def project_weights(self):
        """Project the weight matrix so that each row retains only the top-k coefficients."""
        with torch.no_grad():
            for i in range(self.weight.size(0)):
                row = self.weight[i]
                if self.k < row.numel():
                    _, idx = torch.topk(torch.abs(row), self.k)
                    mask = torch.zeros_like(row)
                    mask[idx] = 1.0
                    self.weight[i] *= mask

def train_probe(X_train, y_train, X_val, y_val, input_dim, output_dim, sparse_k=0, quiet=True):
    torch.manual_seed(config.SEED)
    device = torch.device(config.TRAIN_PARAMS["device"] if torch.cuda.is_available() else "cpu")
    
    if sparse_k > 0:
        model = KSparseLinearProbe(input_dim, output_dim, sparse_k).to(device)
    else:
        model = LinearProbe(input_dim, output_dim).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN_PARAMS["learning_rate"],
                             weight_decay=config.TRAIN_PARAMS["weight_decay"])
    epochs = config.TRAIN_PARAMS["epochs"]
    best_val_acc = 0.0
    best_model_state = None

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    epoch_range = range(epochs) if quiet else tqdm(range(epochs), desc="Training Probe Epochs")
    for epoch in epoch_range:
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if sparse_k > 0 and hasattr(model, "project_weights"):
            model.project_weights()            
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            acc = accuracy_score(y_val.cpu().numpy(), preds)
            utils.log_info(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={acc:.4f}")
            if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = model.state_dict().copy()
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    model.load_state_dict(best_model_state)
    return model

def evaluate_probe(model, X_test, y_test):
    device = torch.device(config.TRAIN_PARAMS["device"] if torch.cuda.is_available() else "cpu")
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    # If binary classification, force a 2x2 confusion matrix.
    if set(y_true).issubset({0,1}):
        cm = confusion_matrix(y_true, preds, labels=[0,1])
    else:
        cm = confusion_matrix(y_true, preds)
    acc = accuracy_score(y_true, preds)
    return acc, cm

def split_data(X, y):
    n = X.shape[0]
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]

def plot_probe_results_separate(results, exp_folder):
    """
    Generate and save separate bar plots (one for each experiment) along with a CSV summary.
    Compute selectivity as (linguistic accuracy - control accuracy).
    """
    os.makedirs(exp_folder, exist_ok=True)
    layers = sorted(results.keys())
    
    experiment_keys = {
        "Inflection": "inflection_acc",
        "Lexeme": "lexeme_acc",
        "Inflection_Control": "inflection_control_acc",
        "Lexeme_Control": "lexeme_control_acc"
    }
    
    # Create a separate bar plot for each experiment key.
    for exp_name, key in experiment_keys.items():
        values = [results[layer].get(key, None) for layer in layers]
        if any(v is not None for v in values):
            plt.figure(figsize=(10,6))
            plt.bar(layers, values, color='skyblue')
            plt.xlabel("Layer")
            plt.ylabel("Accuracy")
            plt.title(f"{exp_name} Probe Accuracy vs. Layer")
            plt.tight_layout()
            plot_path = os.path.join(exp_folder, f"{exp_name.lower().replace(' ', '_')}_accuracy.png")
            plt.savefig(plot_path)
            plt.close()
            utils.log_info(f"Saved {exp_name} probe accuracy bar plot to {plot_path}")
    
    # Compute selectivity metrics.
    csv_rows = []
    for layer in layers:
        row = {"Layer": layer}
        for exp_name, key in experiment_keys.items():
            row[exp_name] = results[layer].get(key, "")
        if results[layer].get("inflection_acc", None) is not None and results[layer].get("inflection_control_acc", None) is not None:
            row["Inflection_Selectivity"] = results[layer]["inflection_acc"] - results[layer]["inflection_control_acc"]
        else:
            row["Inflection_Selectivity"] = ""
        if results[layer].get("lexeme_acc", None) is not None and results[layer].get("lexeme_control_acc", None) is not None:
            row["Lexeme_Selectivity"] = results[layer]["lexeme_acc"] - results[layer]["lexeme_control_acc"]
        else:
            row["Lexeme_Selectivity"] = ""
        csv_rows.append(row)
    
    csv_path = os.path.join(exp_folder, "probe_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["Layer"] + list(experiment_keys.keys()) + ["Inflection_Selectivity", "Lexeme_Selectivity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    utils.log_info(f"Saved probe results CSV to {csv_path}")

def train_single_lexeme(class_idx, X, lexeme_labels, d_model, sparse_k, control=False):
    """
    Trains a one-vs-rest binary classifier for a single lexeme class.
    If control is True, binary labels are randomly assigned for this word type.
    Returns the test accuracy, or None if the label set is degenerate.
    """
    binary_labels = (lexeme_labels == class_idx).astype(int)
    if control:
        binary_labels = np.random.randint(0, 2, size=binary_labels.shape)
    if len(np.unique(binary_labels)) < 2:
        return None
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, binary_labels)
    model = train_probe(X_train, y_train, X_val, y_val, d_model, 2, sparse_k, quiet=True)
    acc, _ = evaluate_probe(model, X_test, y_test)
    return acc

def run_probes(activations_file, labels_file, task="multiclass_inflection", sparse_k=0,
               control_inflection=False, control_lexeme=False, exp_label="default_exp", dataset="default",
               one_vs_rest=False):
    data = np.load(activations_file)["activations"]
    n_examples, n_layers, d_model = data.shape
    utils.log_info(f"Loaded activations of shape {data.shape}")
    
    df = pd.read_csv(labels_file)
    if task in ["binary_inflection", "multiclass_inflection"]:
        if task == "binary_inflection":
            # (Not used in the current experiments)
            inflection_labels = np.array([1 if str(x).lower() == "past" else 0 for x in df["Inflection Label"].values])
            num_inflection_classes = 2
        else:
            inflection_labels = df["Inflection Label"].values
            unique_inflections = sorted(set(inflection_labels))
            inflection_to_idx = {inf: idx for idx, inf in enumerate(unique_inflections)}
            inflection_labels = np.array([inflection_to_idx[inf] for inf in inflection_labels])
            num_inflection_classes = len(unique_inflections)
    else:
        inflection_labels = None
        num_inflection_classes = None

    lexemes = df["Lemma"].values
    lexeme_to_idx = {lex: idx for idx, lex in enumerate(sorted(set(lexemes)))}
    lexeme_labels = np.array([lexeme_to_idx[lex] for lex in lexemes])
    utils.log_info(f"Inflection classes: {num_inflection_classes}, Lexeme classes: {len(lexeme_to_idx)}")
    
    results = {}
    for layer in tqdm(range(n_layers), desc="Processing Layers"):
        X = data[:, layer, :]
        results[layer] = {}
        if task in ["binary_inflection", "multiclass_inflection"]:
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, inflection_labels)
            utils.log_info(f"Training inflection probe for layer {layer}")
            model_inflection = train_probe(X_train, y_train, X_val, y_val, d_model, num_inflection_classes, sparse_k, quiet=True)
            acc_inflection, cm_inflection = evaluate_probe(model_inflection, X_test, y_test)
            results[layer]["inflection_acc"] = acc_inflection
            results[layer]["inflection_cm"] = cm_inflection.tolist()
            utils.log_info(f"Layer {layer} inflection probe test accuracy: {acc_inflection:.4f}")
            if control_inflection:
                control_labels = np.random.randint(0, num_inflection_classes, size=inflection_labels.shape)
                X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, control_labels)
                utils.log_info(f"Training inflection control probe for layer {layer}")
                model_inflection_ctrl = train_probe(X_train, y_train, X_val, y_val, d_model, num_inflection_classes, sparse_k, quiet=True)
                acc_ctrl, cm_ctrl = evaluate_probe(model_inflection_ctrl, X_test, y_test)
                results[layer]["inflection_control_acc"] = acc_ctrl
                results[layer]["inflection_control_cm"] = cm_ctrl.tolist()
                utils.log_info(f"Layer {layer} inflection control probe test accuracy: {acc_ctrl:.4f}")
                
        if task == "lexeme":
            if one_vs_rest:
                num_classes = len(lexeme_to_idx)
                class_accs = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(train_single_lexeme, class_idx, X, lexeme_labels, d_model, sparse_k, False): class_idx
                               for class_idx in range(num_classes)}
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                       desc=f"Layer {layer} one-vs-rest lexeme classes", leave=False):
                        acc_class = future.result()
                        if acc_class is not None:
                            class_accs.append(acc_class)
                avg_acc = np.mean(class_accs) if class_accs else float('nan')
                results[layer]["lexeme_acc"] = avg_acc
                utils.log_info(f"Layer {layer} one-vs-rest lexeme probe average test accuracy: {avg_acc:.4f}")
                if control_lexeme:
                    ctrl_class_accs = []
                    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(train_single_lexeme, class_idx, X, lexeme_labels, d_model, sparse_k, True): class_idx
                                   for class_idx in range(num_classes)}
                        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                           desc=f"Layer {layer} one-vs-rest lexeme control classes", leave=False):
                            acc_ctrl = future.result()
                            if acc_ctrl is not None:
                                ctrl_class_accs.append(acc_ctrl)
                    avg_ctrl_acc = np.mean(ctrl_class_accs) if ctrl_class_accs else float('nan')
                    results[layer]["lexeme_control_acc"] = avg_ctrl_acc
                    utils.log_info(f"Layer {layer} one-vs-rest lexeme control probe average test accuracy: {avg_ctrl_acc:.4f}")
            else:
                # Standard multiclass lexeme probe.
                X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, lexeme_labels)
                utils.log_info(f"Training lexeme probe for layer {layer}")
                model_lexeme = train_probe(X_train, y_train, X_val, y_val, d_model, len(lexeme_to_idx), sparse_k, quiet=True)
                acc_lexeme, cm_lexeme = evaluate_probe(model_lexeme, X_test, y_test)
                results[layer]["lexeme_acc"] = acc_lexeme
                results[layer]["lexeme_cm"] = cm_lexeme.tolist()
                utils.log_info(f"Layer {layer} lexeme probe test accuracy: {acc_lexeme:.4f}")
                if control_lexeme:
                    control_labels = np.random.randint(0, len(lexeme_to_idx), size=lexeme_labels.shape)
                    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, control_labels)
                    utils.log_info(f"Training lexeme control probe for layer {layer}")
                    model_lexeme_ctrl = train_probe(X_train, y_train, X_val, y_val, d_model, len(lexeme_to_idx), sparse_k, quiet=True)
                    acc_ctrl, cm_ctrl = evaluate_probe(model_lexeme_ctrl, X_test, y_test)
                    results[layer]["lexeme_control_acc"] = acc_ctrl
                    results[layer]["lexeme_control_cm"] = cm_ctrl.tolist()
                    utils.log_info(f"Layer {layer} lexeme control probe test accuracy: {acc_ctrl:.4f}")
    
    exp_folder = os.path.join(config.OUTPUT_DIR, "probes", f"{dataset}_{exp_label}")
    os.makedirs(exp_folder, exist_ok=True)
    output_path = os.path.join(exp_folder, "probe_results.npz")
    np.savez_compressed(output_path, results=results)
    utils.log_info(f"Probe results saved to {output_path}")
    plot_probe_results_separate(results, exp_folder)

def parse_args():
    parser = argparse.ArgumentParser(description="Train probes on activations with optional k-sparsity and control experiments.")
    parser.add_argument("--activations", type=str, required=True, help="Path to NPZ file with activations.")
    parser.add_argument("--labels", type=str, required=True, help="Path to CSV file with labels.")
    parser.add_argument("--task", type=str, required=True, choices=["binary_inflection", "multiclass_inflection", "lexeme"],
                        help="Probe task: 'binary_inflection', 'multiclass_inflection', or 'lexeme'.")
    parser.add_argument("--sparse_k", type=int, default=0,
                        help="If > 0, use a k-sparse probe retaining only the top k weights per row.")
    parser.add_argument("--control_inflection", action="store_true",
                        help="Run control experiment for inflection probe.")
    parser.add_argument("--control_lexeme", action="store_true",
                        help="Run control experiment for lexeme probe.")
    parser.add_argument("--one_vs_rest", action="store_true",
                        help="If provided with --task lexeme, train one-vs-rest binary classifiers per lexeme class.")
    parser.add_argument("--exp_label", type=str, default="default_exp",
                        help="Experiment label (for output folder naming).")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset label (e.g. controlled, wikitext, combined).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_probes(args.activations, args.labels, args.task, args.sparse_k,
               args.control_inflection, args.control_lexeme, args.exp_label, args.dataset,
               one_vs_rest=args.one_vs_rest)
