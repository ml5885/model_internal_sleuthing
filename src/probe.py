import torch
import torch.nn as nn
import math
from src import config, utils
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cpu")

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
        with torch.no_grad():
            for i in range(self.weight.size(0)):
                row = self.weight[i]
                if self.k < row.numel():
                    _, idx = torch.topk(torch.abs(row), self.k)
                    mask = torch.zeros_like(row)
                    mask[idx] = 1.0
                    self.weight[i] *= mask

def train_probe(X_train, y_train, X_val, y_val, input_dim, output_dim, lambda_reg=1e-3, quiet=True):
    torch.manual_seed(config.SEED)
    device = get_device()
    
    model = LinearProbe(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_PARAMS["learning_rate"],
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
    device = get_device()
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    y_true = y_test.cpu().numpy()
    if set(y_true).issubset({0, 1}):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
    else:
        cm = confusion_matrix(y_true, preds)
    
    acc = accuracy_score(y_true, preds)
    return acc, cm

def process_inflection_layer(layer, X_layer, inflection_labels, lambda_reg, target_indices):
    if X_layer.ndim == 3:
        feats = extract_target_representation(torch.tensor(X_layer, dtype=torch.float32), target_indices)
    else:
        feats = torch.tensor(X_layer, dtype=torch.float32)

    n = feats.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.SEED + layer))
    feats = feats[perm]
    inflection_labels = inflection_labels[perm.numpy()]

    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])

    X_train = feats[:train_end]
    X_test = feats[val_end:]
    y_train = inflection_labels[:train_end]
    y_test = inflection_labels[val_end:]
    
    classes = np.unique(inflection_labels)

    true_scores = []
    for c in tqdm(classes, desc=f"Layer {layer} Inflection", leave=False):
        bin_train = (y_train == c).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        true_scores.append(closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg).unsqueeze(1))
        
    all_true = torch.cat(true_scores, dim=1)
    pred_true = torch.argmax(all_true, dim=1).cpu().numpy()
    acc_true = (pred_true == y_test).sum() / y_test.shape[0]

    np.random.seed(config.SEED + layer + 100)
    num_cls = len(classes)
    m = {c: np.random.randint(0, num_cls) for c in classes}
    y_train_c = np.array([m[c] for c in y_train])
    y_test_c = np.array([m[c] for c in y_test])

    control_scores = []
    control_classes = np.unique(y_train_c)
    for c in tqdm(control_classes, desc=f"Layer {layer} Control", leave=False):
        bin_train = (y_train_c == c).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        control_scores.append(closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg).unsqueeze(1))
        
    all_control = torch.cat(control_scores, dim=1)
    pred_control = torch.argmax(all_control, dim=1).cpu().numpy()
    acc_control = (pred_control == y_test_c).sum() / y_test_c.shape[0]

    utils.log_info(f"[Layer {layer}] Inflection Acc: {acc_true:.4f} | Control Acc: {acc_control:.4f}")
    return layer, {"inflection_acc": acc_true, "inflection_control_acc": acc_control}

def process_lexeme_layer(layer, X_layer, lexeme_labels, lambda_reg, lexeme_to_idx, target_indices):
    if X_layer.ndim == 3:
        feats = extract_target_representation(torch.tensor(X_layer, dtype=torch.float32), target_indices)
    else:
        feats = torch.tensor(X_layer, dtype=torch.float32)

    n = feats.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.SEED + layer))
    feats = feats[perm]
    lexeme_labels = lexeme_labels[perm.numpy()]

    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])

    X_train = feats[:train_end]
    X_test = feats[val_end:]
    y_train = lexeme_labels[:train_end]
    y_test = lexeme_labels[val_end:]

    num_cls = len(lexeme_to_idx)

    true_scores = []
    for idx in tqdm(range(num_cls), desc=f"Layer {layer} Lexeme", leave=False):
        bin_train = (y_train == idx).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        true_scores.append(closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg).unsqueeze(1))
        
    all_true = torch.cat(true_scores, dim=1)
    pred_true = torch.argmax(all_true, dim=1).cpu().numpy()
    acc_true = (pred_true == y_test).sum() / y_test.shape[0]

    np.random.seed(config.SEED + layer + 1000)
    m = {i: np.random.randint(0, num_cls) for i in range(num_cls)}
    y_train_c = np.array([m[c] for c in y_train])
    y_test_c = np.array([m[c] for c in y_test])

    control_scores = []
    control_classes = np.unique(y_train_c)
    for c in tqdm(control_classes, desc=f"Layer {layer} Control", leave=False):
        bin_train = (y_train_c == c).astype(np.float32)
        w = torch.tensor(bin_train, dtype=torch.float32)
        control_scores.append(closed_form_ridge_binary_predict(X_train, w, X_test, lambda_reg).unsqueeze(1))
        
    all_control = torch.cat(control_scores, dim=1)
    pred_control = torch.argmax(all_control, dim=1).cpu().numpy()
    acc_control = (pred_control == y_test_c).sum() / y_test_c.shape[0]

    utils.log_info(f"[Layer {layer}] Lexeme Acc: {acc_true:.4f} | Control Acc: {acc_control:.4f}")
    return layer, {"lexeme_acc": acc_true, "lexeme_control_acc": acc_control}

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
    plt.bar(ind, task_accs, label=f"{ylabel} Task", alpha=0.7)
    plt.bar(ind, control_accs, label="Control", alpha=0.7)
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
