import os
import sys
import csv
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, top_k_accuracy_score

from src import config, utils

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_probe(
    X_train, Y_train, X_val, Y_val,
    input_dim, output_dim, lambda_reg, quiet=False
):
    torch.manual_seed(config.SEED)
    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = LinearProbe(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.TRAIN_PARAMS["learning_rate"],
        weight_decay=config.TRAIN_PARAMS["weight_decay"]
    )

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    Y_train_tensor = torch.from_numpy(Y_train).long().to(device)

    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    Y_val_tensor = torch.from_numpy(Y_val).long().to(device)

    best_acc = 0.0
    best_state = None

    for epoch in range(config.TRAIN_PARAMS["epochs"]):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_tensor)
        loss = criterion(logits, Y_train_tensor)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_acc = (val_logits.argmax(dim=1) == Y_val_tensor).float().mean().item()

        if not quiet:
            utils.log_info(
                f"Epoch {epoch+1} loss {loss.item():.4f} val_acc {val_acc:.4f}"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def process_layer(
    seed, X_flat, y_true, y_control,
    lambda_reg, task, probe_type, layer
):
    rng = np.random.RandomState(seed)
    N = X_flat.shape[0]
    permutation = rng.permutation(N)

    X = X_flat[permutation]
    y = y_true[permutation]
    c = y_control[permutation]

    n_train = int(N * config.SPLIT_RATIOS["train"])
    n_val = int(N * config.SPLIT_RATIOS["val"])
    n_test = N - n_train - n_val

    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]

    Y_train = y[:n_train]
    Y_val = y[n_train:n_train + n_val]
    Y_test = y[n_train + n_val:]

    control_train = c[:n_train]
    control_val = c[n_train:n_train + n_val]
    control_test = c[n_train + n_val:]

    # number of inflection classes
    num_classes = len(np.unique(y_true))

    # ---- FIXED CONTROL MAPPING ----
    # map each unique control (lemma) to a random inflection class
    unique_lemmas = np.unique(c)
    num_control = unique_lemmas.shape[0]
    # sample with replacement from [0 .. num_classes-1]
    random_inf_labels = rng.randint(low=0, high=num_classes, size=num_control)
    control_map = {
        lemma: random_inf_labels[i]
        for i, lemma in enumerate(unique_lemmas)
    }

    control_train_mapped = np.array([control_map[x] for x in control_train])
    control_val_mapped   = np.array([control_map[x] for x in control_val])
    control_test_mapped  = np.array([control_map[x] for x in control_test])
    # ---------------------------------

    if probe_type == "nn":
        device = get_device()
        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        workers = config.TRAIN_PARAMS.get("workers", max(1, os.cpu_count() // 2))

        scores = np.zeros((n_test, num_classes), dtype=np.float32)
        control_scores = np.zeros((n_test, num_classes), dtype=np.float32)

        def run_true_class(cls):
            Y_train_bin = (Y_train == cls).astype(int)
            Y_val_bin = (Y_val == cls).astype(int)

            model = train_probe(
                X_train, Y_train_bin, X_val, Y_val_bin,
                input_dim=X_train.shape[1],
                output_dim=2,
                lambda_reg=lambda_reg,
                quiet=True
            )

            with torch.no_grad():
                return cls, model(X_test_tensor)[:, 1].cpu().numpy()

        def run_control_class(cls):
            ctl_train_bin = (control_train_mapped == cls).astype(int)
            ctl_val_bin = (control_val_mapped == cls).astype(int)

            model = train_probe(
                X_train, ctl_train_bin, X_val, ctl_val_bin,
                input_dim=X_train.shape[1],
                output_dim=2,
                lambda_reg=lambda_reg,
                quiet=True
            )

            with torch.no_grad():
                return cls, model(X_test_tensor)[:, 1].cpu().numpy()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures_true = {executor.submit(run_true_class, cls): cls for cls in range(num_classes)}
            for f in tqdm(as_completed(futures_true), total=num_classes,
                          desc=f"Layer {layer} True Probes", leave=True, file=sys.stdout):
                cls, vals = f.result()
                scores[:, cls] = vals

        print(f"\nProcessing Layer {layer} (Control Probes):")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures_control = {executor.submit(run_control_class, cls): cls for cls in range(num_classes)}
            for f in tqdm(as_completed(futures_control), total=num_classes,
                          desc=f"Layer {layer} Control Probes", leave=True, file=sys.stdout):
                cls, vals = f.result()
                control_scores[:, cls] = vals

        preds = np.argmax(scores, axis=1)
        accuracy = (preds == Y_test).mean()

        preds_control = np.argmax(control_scores, axis=1)
        control_accuracy = (preds_control == control_test_mapped).mean()

    else:
        # closed-form ridge regression
        d = X_train.shape[1]
        cov = X_train.T.dot(X_train) + lambda_reg * np.eye(d)

        # true labels one-hot
        Y_train_one = np.eye(num_classes)[Y_train]
        W = np.linalg.solve(cov, X_train.T.dot(Y_train_one))
        scores = X_test.dot(W)

        # control labels one-hot
        Y_control_one = np.eye(num_classes)[control_train_mapped]
        Wc = np.linalg.solve(cov, X_train.T.dot(Y_control_one))
        control_scores = X_test.dot(Wc)

        preds = np.argmax(scores, axis=1)
        accuracy = (preds == Y_test).mean()

        preds_control = np.argmax(control_scores, axis=1)
        control_accuracy = (preds_control == control_test_mapped).mean()

    average_type = 'binary' if num_classes == 2 else 'macro'
    f1 = f1_score(Y_test, preds, average=average_type)
    control_f1 = f1_score(control_test_mapped, preds_control, average=average_type)

    top5 = top_k_accuracy_score(Y_test, scores, k=5, labels=np.arange(num_classes)) if num_classes > 2 else None
    control_top5 = top_k_accuracy_score(control_test_mapped, control_scores, k=5, labels=np.arange(num_classes)) if num_classes > 2 else None

    result = {
        f"{task}_acc": accuracy,
        f"{task}_control_acc": control_accuracy,
        f"{task}_acc_ci_low": -1,
        f"{task}_acc_ci_high": -1,
        f"{task}_control_acc_ci_low": -1,
        f"{task}_control_acc_ci_high": -1,
        f"{task}_f1": f1,
        f"{task}_control_f1": control_f1,
        f"{task}_top5": top5,
        f"{task}_control_top5": control_top5
    }

    utils.log_info(
        f"[Layer {layer} seed {seed}] {task} {probe_type} "
        f"acc {accuracy:.3f} f1 {f1:.3f} "
        f"control_acc {control_accuracy:.3f} control_f1 {control_f1:.3f}"
    )

    return seed, result

def plot_probe_results(results, outdir, task):
    os.makedirs(outdir, exist_ok=True)

    layers = sorted(results.keys(), key=lambda k: int(k.split("_")[1]))
    indices = np.arange(len(layers))

    acc_key = f"{task}_acc"
    ctl_acc_key = f"{task}_control_acc"
    f1_key = f"{task}_f1"
    ctl_f1_key = f"{task}_control_f1"
    top5_key = f"{task}_top5"
    ctl_top5_key = f"{task}_control_top5"

    task_accs = np.array([results[l][acc_key] for l in layers])
    control_accs = np.array([results[l][ctl_acc_key] for l in layers])
    f1_scores = np.array([results[l][f1_key] for l in layers])
    control_f1s = np.array([results[l][ctl_f1_key] for l in layers])
    top5_scores = np.array([results[l][top5_key] or 0 for l in layers])
    control_top5s = np.array([results[l][ctl_top5_key] or 0 for l in layers])

    # Combined accuracy plot
    plt.figure(figsize=(10, 6))
    plt.bar(indices, task_accs, 0.6, label="Task Acc", alpha=0.7)
    plt.bar(indices, control_accs, 0.6, label="Control Acc", alpha=0.7)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{task}: Task vs Control Accuracy")
    plt.xticks(indices, [l.split("_")[1] for l in layers])
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{task}_combined.png"), bbox_inches="tight")
    plt.close()

    # Task-only accuracy plot
    plt.figure(figsize=(10, 6))
    plt.bar(indices, task_accs, 0.6)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{task}: Task Accuracy")
    plt.xticks(indices, [l.split("_")[1] for l in layers])
    plt.savefig(os.path.join(outdir, f"{task}_task.png"), bbox_inches="tight")
    plt.close()

    # Control-only accuracy plot
    plt.figure(figsize=(10, 6))
    plt.bar(indices, control_accs, 0.6)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{task}: Control Accuracy")
    plt.xticks(indices, [l.split("_")[1] for l in layers])
    plt.savefig(os.path.join(outdir, f"{task}_control.png"), bbox_inches="tight")
    plt.close()

    # Write CSV summary
    csv_path = os.path.join(outdir, f"{task}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Layer",
            f"{task}_Accuracy",
            f"{task}_F1",
            f"{task}_Top5",
            f"{task}_ControlAccuracy",
            f"{task}_Control_F1",
            f"{task}_Control_Top5"
        ])
        for layer in layers:
            writer.writerow([
                layer.split("_")[1],
                results[layer][acc_key],
                results[layer][f1_key],
                results[layer][top5_key] or "",
                results[layer][ctl_acc_key],
                results[layer][ctl_f1_key],
                results[layer][ctl_top5_key] or ""
            ])
