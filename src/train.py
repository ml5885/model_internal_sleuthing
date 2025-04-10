import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
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
    A k-sparse linear probe that retains only the top-k (by absolute value)
    coefficients in each row of the weight matrix after each optimizer step.
    
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
        """Project the weight matrix to keep only the top-k coefficients per row."""
        with torch.no_grad():
            for i in range(self.weight.size(0)):
                row = self.weight[i]
                if self.k < row.numel():
                    _, idx = torch.topk(torch.abs(row), self.k)
                    mask = torch.zeros_like(row)
                    mask[idx] = 1.0
                    self.weight[i] *= mask

def train_probe(X_train, y_train, X_val, y_val, input_dim, output_dim, sparse_k=0):
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

    for epoch in range(epochs):
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
            utils.log_info(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} Val Acc: {acc:.4f}")
            if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = model.state_dict()

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
    acc = accuracy_score(y_test.cpu().numpy(), preds)
    cm = confusion_matrix(y_test.cpu().numpy(), preds)
    return acc, cm

def split_data(X, y):
    n = X.shape[0]
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]

def run_probes(activations_file, labels_file, task="both", sparse_k=0,
               control_inflection=False, control_lexeme=False):
    data = np.load(activations_file)["activations"]
    n_examples, n_layers, d_model = data.shape
    utils.log_info(f"Loaded activations of shape {data.shape}")

    df = pd.read_csv(labels_file)
    # Prepare inflection labels: use all unique inflection values.
    inflection_labels = df["Inflection Label"].values
    unique_inflections = sorted(set(inflection_labels))
    inflection_to_idx = {inf: idx for idx, inf in enumerate(unique_inflections)}
    inflection_labels = np.array([inflection_to_idx[inf] for inf in inflection_labels])
    
    # Prepare lexeme labels.
    lexemes = df["Lemma"].values
    lexeme_to_idx = {lex: idx for idx, lex in enumerate(sorted(set(lexemes)))}
    lexeme_labels = np.array([lexeme_to_idx[lex] for lex in lexemes])
    utils.log_info(f"Inflection classes: {len(inflection_to_idx)}, Lexeme classes: {len(lexeme_to_idx)}")

    results = {}
    for layer in range(n_layers):
        X = data[:, layer, :]
        results[layer] = {}
        
        if task in ["inflection", "both"]:
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, inflection_labels)
            utils.log_info(f"Training inflection probe for layer {layer}")
            model_inflection = train_probe(X_train, y_train, X_val, y_val, d_model, len(inflection_to_idx), sparse_k)
            acc_inflection, cm_inflection = evaluate_probe(model_inflection, X_test, y_test)
            results[layer]["inflection_acc"] = acc_inflection
            results[layer]["inflection_cm"] = cm_inflection.tolist()
            utils.log_info(f"Layer {layer} inflection probe test accuracy: {acc_inflection:.4f}")

            if control_inflection:
                control_labels = np.random.randint(0, len(inflection_to_idx), size=inflection_labels.shape)
                X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, control_labels)
                utils.log_info(f"Training inflection control probe for layer {layer}")
                model_inflection_ctrl = train_probe(X_train, y_train, X_val, y_val, d_model, len(inflection_to_idx), sparse_k)
                acc_ctrl, cm_ctrl = evaluate_probe(model_inflection_ctrl, X_test, y_test)
                results[layer]["inflection_control_acc"] = acc_ctrl
                results[layer]["inflection_control_cm"] = cm_ctrl.tolist()
                utils.log_info(f"Layer {layer} inflection control probe test accuracy: {acc_ctrl:.4f}")

        if task in ["lexeme", "both"]:
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, lexeme_labels)
            utils.log_info(f"Training lexeme probe for layer {layer}")
            model_lexeme = train_probe(X_train, y_train, X_val, y_val, d_model, len(lexeme_to_idx), sparse_k)
            acc_lexeme, cm_lexeme = evaluate_probe(model_lexeme, X_test, y_test)
            results[layer]["lexeme_acc"] = acc_lexeme
            results[layer]["lexeme_cm"] = cm_lexeme.tolist()
            utils.log_info(f"Layer {layer} lexeme probe test accuracy: {acc_lexeme:.4f}")

            if control_lexeme:
                control_labels = np.random.randint(0, len(lexeme_to_idx), size=lexeme_labels.shape)
                X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, control_labels)
                utils.log_info(f"Training lexeme control probe for layer {layer}")
                model_lexeme_ctrl = train_probe(X_train, y_train, X_val, y_val, d_model, len(lexeme_to_idx), sparse_k)
                acc_ctrl, cm_ctrl = evaluate_probe(model_lexeme_ctrl, X_test, y_test)
                results[layer]["lexeme_control_acc"] = acc_ctrl
                results[layer]["lexeme_control_cm"] = cm_ctrl.tolist()
                utils.log_info(f"Layer {layer} lexeme control probe test accuracy: {acc_ctrl:.4f}")

    output_path = os.path.join(config.OUTPUT_DIR, "probe_results.npz")
    np.savez_compressed(output_path, results=results)
    utils.log_info(f"Probe results saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probes on activations with optional k-sparsity and control tasks.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the NPZ file with activations.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the CSV file with labels.")
    parser.add_argument("--task", type=str, default="both", choices=["inflection", "lexeme", "both"],
                        help="Which probe task to run: 'inflection' (multiclass morphological prediction), 'lexeme' (verb identity), or 'both'.")
    parser.add_argument("--sparse_k", type=int, default=0,
                        help="If > 0, use a k-sparse probe retaining only the top k weights per row.")
    parser.add_argument("--control_inflection", action="store_true",
                        help="If set, run a control experiment for inflection prediction with random labels.")
    parser.add_argument("--control_lexeme", action="store_true",
                        help="If set, run a control experiment for lexeme prediction with random labels.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_probes(args.activations, args.labels, args.task, args.sparse_k,
               args.control_inflection, args.control_lexeme)
