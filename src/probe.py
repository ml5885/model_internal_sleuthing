import os
import csv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib
import json

from src import config, utils


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPProbe(nn.Module):
    """MLP probe: one hidden ReLU layer, then softmax output."""
    def __init__(self, input_dim, output_dim, hidden_dim=None, norm_weight=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 64
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.norm.weight = nn.Parameter(torch.ones(input_dim))
        self.norm.elementwise_affine = True # allow weight to be updated
        if norm_weight is not None:
            with torch.no_grad():
                self.norm.weight.copy_(norm_weight)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.norm(x)
        h = self.relu(self.linear1(x))
        d = self.dropout(h)
        return self.linear2(d)
    
    def predict(self, arr, batch_size):
        device = next(self.parameters()).device
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), batch_size):
                chunk = torch.from_numpy(arr[i:i + batch_size]).float().to(device)
                out.append(self(chunk).cpu())
        return torch.cat(out, dim=0).numpy()


def train_probe(X_train, y_train, X_val, y_val, input_dim, n_classes, norm_weight=None):
    torch.manual_seed(config.SEED)
    device = get_device()

    model = MLPProbe(input_dim, n_classes, norm_weight=norm_weight).to(device)
    if norm_weight is not None:
        utils.log_info("Training MLP probe with pre-loaded LayerNorm weight.")
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAIN_PARAMS["learning_rate"],
        weight_decay=config.TRAIN_PARAMS["weight_decay"],
    )
    crit = nn.CrossEntropyLoss()

    bs = config.TRAIN_PARAMS["batch_size"]

    def make_loader(X, y, shuffle):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).long(),
            ),
            batch_size=bs,
            shuffle=shuffle,
            pin_memory=True,
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)

    best_acc, best_state, wait = float("-inf"), None, 0
    early_stop = config.TRAIN_PARAMS["early_stop"]

    for epoch in range(config.TRAIN_PARAMS["epochs"]):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device)).cpu()
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        utils.log_info(f"epoch {epoch+1:02d}  loss {loss.item():.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state)
    return model


def solve_ridge(X_train, y_train, X_test, lambda_reg, n_classes):
    d = X_train.shape[1]
    cov = X_train.T.dot(X_train) + lambda_reg * np.eye(d)
    W = np.linalg.solve(cov, X_train.T.dot(np.eye(n_classes)[y_train]))
    return X_test.dot(W)


def predict(arr, model):
    bs = config.TRAIN_PARAMS["batch_size"]
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(arr), bs):
            chunk = torch.from_numpy(arr[i : i + bs]).float().to(
                next(model.parameters()).device
            )
            out.append(model(chunk).cpu())
    return torch.cat(out, dim=0).numpy()


def process_layer(seed, X_flat, y_true, y_control, lambda_reg, task, probe_type, layer, pca_dim, outdir=None, indices=None, label_map=None, control_label_map=None):
    uniq, counts = np.unique(y_true, return_counts=True)
    keep_classes = uniq[counts >= 1]
    keep_mask = np.isin(y_true, keep_classes)

    X_flat = X_flat[keep_mask]
    y_true = y_true[keep_mask]
    y_control = y_control[keep_mask]
    if indices is not None:
        indices = indices[keep_mask]

    try:
        # first attempt: stratified
        X_train, X_temp, y_train, y_temp, yc_train, yc_temp, idx_train, idx_temp = (
            train_test_split(
                X_flat,
                y_true,
                y_control,
                np.arange(len(X_flat)) if indices is None else indices,
                train_size=config.SPLIT_RATIOS["train"],
                random_state=seed,
                stratify=y_true,
            )
        )
    except ValueError as e:
        utils.log_info(
            f"Stratified split failed ({e}); retrying without stratification."
        )
        # fallback: no stratification
        X_train, X_temp, y_train, y_temp, yc_train, yc_temp, idx_train, idx_temp = (
            train_test_split(
                X_flat,
                y_true,
                y_control,
                np.arange(len(X_flat)) if indices is None else indices,
                train_size=config.SPLIT_RATIOS["train"],
                random_state=seed,
                stratify=None,
            )
        )

    val_frac = config.SPLIT_RATIOS["val"] / (
        config.SPLIT_RATIOS["val"] + config.SPLIT_RATIOS["test"]
    )
    temp_counts = np.bincount(y_temp)
    stratify_val = y_temp if temp_counts.min() > 1 else None

    X_val, X_test, y_val, y_test, yc_val, yc_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, yc_temp, idx_temp,
        train_size=val_frac,
        random_state=seed,
        stratify=stratify_val
    )

    if len(y_test) == 0:
        utils.log_info(f"Layer {layer}: No test samples after split, skipping this layer.")
        raise ValueError(f"Layer {layer}: No test samples after split.")

    pca_explained_variance = -1
    if pca_dim and pca_dim < X_train.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=config.SEED)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        pca_explained_variance = sum(pca.explained_variance_ratio_)

    n_classes = int(np.max(y_true) + 1)
    rng = np.random.RandomState(seed)
    unique_controls = sorted(set(yc_train.tolist() + yc_val.tolist() + yc_test.tolist()))
    perm = rng.permutation(len(unique_controls))
    control_map = {unique_controls[i]: perm[i] % n_classes for i in range(len(unique_controls))}
    yc_train_m = np.array([control_map[v] for v in yc_train])
    yc_val_m = np.array([control_map[v] for v in yc_val])
    yc_test_m = np.array([control_map[v] for v in yc_test])

    bs = config.TRAIN_PARAMS["batch_size"]

    if probe_type in ["mlp", "nn"]:
        model = train_probe(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1], n_classes=n_classes)
        scores = model.predict(X_test, batch_size=bs)
        control_model = train_probe(X_train, yc_train_m, X_val, yc_val_m, input_dim=X_train.shape[1], n_classes=n_classes)
        control_scores = control_model.predict(X_test, batch_size=bs)
        preds = scores.argmax(1)
        preds_control = control_scores.argmax(1)
        
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            model_path = os.path.join(outdir, f"probe_layer_{layer}.pt")
            torch.save(model.state_dict(), model_path)
            if label_map and isinstance(label_map, list):
                label_map_path = os.path.join(outdir, "label_map.json")
                if not os.path.exists(label_map_path):
                    with open(label_map_path, 'w') as f:
                        json.dump(label_map, f)

    elif probe_type == "rf":
        rf = OneVsRestClassifier(RandomForestClassifier(
            n_estimators=config.TRAIN_PARAMS["rf_n_estimators"],
            max_depth=config.TRAIN_PARAMS["rf_max_depth"],
            min_samples_leaf=config.TRAIN_PARAMS["rf_min_samples_leaf"],
            n_jobs=config.TRAIN_PARAMS["workers"],
            random_state=seed
        ))
        
        rf.fit(X_train, y_train)
        scores = rf.predict_proba(X_test)
        preds = rf.predict(X_test)

        rf_ctrl = OneVsRestClassifier(RandomForestClassifier(
            n_estimators=config.TRAIN_PARAMS["rf_n_estimators"],
            max_depth=config.TRAIN_PARAMS["rf_max_depth"],
            min_samples_leaf=config.TRAIN_PARAMS["rf_min_samples_leaf"],
            n_jobs=config.TRAIN_PARAMS["workers"],
            random_state=seed
        ))
        
        rf_ctrl.fit(X_train, yc_train_m)
        control_scores = rf_ctrl.predict_proba(X_test)
        preds_control = rf_ctrl.predict(X_test)

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            model_path = os.path.join(outdir, f"probe_layer_{layer}.joblib")
            joblib.dump(rf, model_path)
            if label_map and isinstance(label_map, list):
                label_map_path = os.path.join(outdir, "label_map.json")
                if not os.path.exists(label_map_path):
                    with open(label_map_path, 'w') as f:
                        json.dump(label_map, f)
    else:
        scores = solve_ridge(X_train, y_train, X_test, lambda_reg, n_classes)
        control_scores = solve_ridge(X_train, yc_train_m, X_test, lambda_reg, n_classes)
        preds = scores.argmax(1)
        preds_control = control_scores.argmax(1)

    y_true_str = [label_map[y] if label_map else y for y in y_test]
    y_pred_str = [label_map[y] if label_map else y for y in preds]
    y_ctrl_str = [control_label_map[y] if control_label_map else y for y in yc_test_m]
    y_ctrl_pred_str = [control_label_map[y] if control_label_map else y for y in preds_control]

    pred_df = pd.DataFrame({
        "Index": idx_test,
        "y_true": y_test,
        "y_true_str": y_true_str,
        "y_pred": preds,
        "y_pred_str": y_pred_str,
        "y_control_true": yc_test_m,
        "y_control_true_str": y_ctrl_str,
        "y_control_pred": preds_control,
        "y_control_pred_str": y_ctrl_pred_str,
        "layer": layer
    })

    accuracy = (preds == y_test).mean()
    control_acc = (preds_control == yc_test_m).mean()
    f1 = f1_score(y_test, preds, average="macro")
    cf1 = f1_score(yc_test_m, preds_control, average="macro")

    utils.log_info(f"[layer {layer}] {task} {probe_type} acc {accuracy:.3f} f1 {f1:.3f} "
                   f"ctrl_acc {control_acc:.3f} ctrl_f1 {cf1:.3f}")

    result = {
        f"{task}_acc": accuracy,
        f"{task}_control_acc": control_acc,
        f"{task}_f1": f1,
        f"{task}_control_f1": cf1,
        f"{task}_acc_ci_low": -1,
        f"{task}_acc_ci_high": -1,
        f"{task}_control_acc_ci_low": -1,
        f"{task}_control_acc_ci_high": -1,
        "pca_explained_variance": pca_explained_variance
    }

    return seed, result, pred_df


def plot_probe_results(results: dict, outdir: str, task: str):
    os.makedirs(outdir, exist_ok=True)
    layers = sorted(results.keys(), key=lambda k: int(k.split("_")[1]))
    idx = np.arange(len(layers))
    col = lambda k: np.array([results[l][k] or 0 for l in layers])

    plt.figure(figsize=(10, 6))
    plt.bar(idx, col(f"{task}_acc"), 0.6, label="Task acc", alpha=0.7)
    plt.bar(idx, col(f"{task}_control_acc"), 0.6, label="Control acc", alpha=0.7)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{task}: Task vs Control accuracy")
    plt.xticks(idx, [l.split("_")[1] for l in layers])
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{task}_combined.png"), bbox_inches="tight")
    plt.close()

    for key, title in [(f"{task}_acc", "Task accuracy"), (f"{task}_control_acc", "Control accuracy")]:
        plt.figure(figsize=(10, 6))
        plt.bar(idx, col(key), 0.6)
        plt.ylim(0, 1)
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title(f"{task}: {title}")
        plt.xticks(idx, [l.split("_")[1] for l in layers])
        plt.savefig(os.path.join(outdir, f"{task}_{key}.png"), bbox_inches="tight")
        plt.close()

    csv_path = os.path.join(outdir, f"{task}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Layer", "Acc", "F1", "controlAcc", "controlF1", "PCA_ExplainedVar"])
        for l in layers:
            r = results[l]
            w.writerow([
                l.split("_")[1],
                r[f"{task}_acc"],
                r[f"{task}_f1"],
                r[f"{task}_control_acc"],
                r[f"{task}_control_f1"],
                r["pca_explained_variance"],
            ])