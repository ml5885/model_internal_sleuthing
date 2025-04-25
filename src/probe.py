import os
import csv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, top_k_accuracy_score
from src import config, utils

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class LinearProbe(nn.Module):
    """One multi-class soft-max layer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
    def predict(self, arr, batch_size):
        device = next(self.parameters()).device
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), batch_size):
                chunk = torch.from_numpy(arr[i:i + batch_size]).float().to(device)
                out.append(self(chunk).cpu())
        return torch.cat(out, dim=0).numpy()

def train_probe(X_train, y_train, X_val, y_val, input_dim, n_classes):
    torch.manual_seed(config.SEED)
    device = get_device()

    model = LinearProbe(input_dim, n_classes).to(device)
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

    best_acc, best_state, wait = 0.0, None, 0
    early_stop = config.TRAIN_PARAMS["early_stop"]

    for epoch in range(config.TRAIN_PARAMS["epochs"]):
        model.train()
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
    
    # X^T * X + lambda * I
    cov = X_train.T.dot(X_train) + lambda_reg * np.eye(d)
    
    # cov * W = X^T * y, solve for W
    W = np.linalg.solve(cov, X_train.T.dot(np.eye(n_classes)[y_train]))
    
    # X_test * W to get predictions
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

def process_layer(seed, X_flat, y_true, y_control, lambda_reg, task, probe_type, layer):
    rng = np.random.RandomState(seed)
    N = X_flat.shape[0]
    order = rng.permutation(N)

    X = X_flat[order]
    y = y_true[order]
    yc = y_control[order]

    n_train = int(N * config.SPLIT_RATIOS["train"])
    n_val = int(N * config.SPLIT_RATIOS["val"])

    X_train, X_val, X_test = (
        X[:n_train],
        X[n_train:n_train + n_val],
        X[n_train + n_val:],
    )
    y_train, y_val, y_test = (
        y[:n_train],
        y[n_train:n_train + n_val],
        y[n_train + n_val:],
    )
    yc_train, yc_val, yc_test = (
        yc[:n_train],
        yc[n_train:n_train + n_val],
        yc[n_train + n_val:],
    )

    n_classes = int(np.max(y_true) + 1)

    # random control-label mapping
    uniq_control = np.unique(yc)
    control_map = {int(u): int(rng.randint(0, n_classes)) for u in uniq_control}
    yc_train_m = np.array([control_map[v] for v in yc_train])
    yc_val_m = np.array([control_map[v] for v in yc_val])
    yc_test_m = np.array([control_map[v] for v in yc_test])

    bs = config.TRAIN_PARAMS["batch_size"]

    if probe_type == "mlp":
        model = train_probe(
            X_train, y_train,
            X_val, y_val,
            input_dim=X.shape[1],
            n_classes=n_classes,
        )
        scores = model.predict(X_test, batch_size=bs)

        control_model = train_probe(
            X_train, yc_train_m,
            X_val, yc_val_m,
            input_dim=X.shape[1],
            n_classes=n_classes,
        )
        control_scores = control_model.predict(X_test, batch_size=bs)
    else:
        scores = solve_ridge(X_train, y_train, X_test, lambda_reg, n_classes)
        control_scores = solve_ridge(X_train, yc_train_m, X_test, lambda_reg, n_classes)

    preds = scores.argmax(1)
    preds_control = control_scores.argmax(1)

    accuracy = (preds == y_test).mean()
    control_acc = (preds_control == yc_test_m).mean()

    f1 = f1_score(y_test, preds, average="macro")
    cf1 = f1_score(yc_test_m, preds_control, average="macro")

    top5 = top_k_accuracy_score(y_test, scores, k=5)
    ctop5 = top_k_accuracy_score(yc_test_m, control_scores, k=5)

    utils.log_info(f"[layer {layer}] {task} {probe_type}  acc {accuracy:.3f}  f1 {f1:.3f}  "
                   f"control_acc {control_acc:.3f}  control_f1 {cf1:.3f}")
    return seed, {
        f"{task}_acc": accuracy,
        f"{task}_control_acc": control_acc,
        f"{task}_f1": f1,
        f"{task}_control_f1": cf1,
        f"{task}_top5": top5,
        f"{task}_control_top5": ctop5,
        f"{task}_acc_ci_low": -1,
        f"{task}_acc_ci_high": -1,
        f"{task}_control_acc_ci_low": -1,
        f"{task}_control_acc_ci_high": -1,
    }

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

    for key, title in [(f"{task}_acc", "Task accuracy"), 
                       (f"{task}_control_acc", "Control accuracy"),]:
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
        w.writerow(["Layer", "Acc", "F1", "Top5", "controlAcc", "controlF1", "controlTop5"])
        for l in layers:
            r = results[l]
            w.writerow(
                [
                    l.split("_")[1],
                    r[f"{task}_acc"],
                    r[f"{task}_f1"],
                    r[f"{task}_top5"] or "",
                    r[f"{task}_control_acc"],
                    r[f"{task}_control_f1"],
                    r[f"{task}_control_top5"] or "",
                ]
            )
