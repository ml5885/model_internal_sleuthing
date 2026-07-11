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


class ScalarMixProbe(nn.Module):
    """
    Tenney et al. (2019) style probe: an ELMo scalar mix over all layers feeding
    a linear or MLP head.

        mixed = gamma * sum_l softmax(s)_l * LN(h_l)

    LN is a per-layer, non-affine LayerNorm over the hidden dimension (ELMo
    convention). It is essential for decoder-only models, whose late-layer norms
    are orders of magnitude larger and would otherwise dominate the mixture.
    """
    def __init__(self, n_layers, input_dim, output_dim, head="linear",
                 hidden_dim=256, dropout=0.3, do_layer_norm=True):
        super().__init__()
        self.scalar_weights = nn.Parameter(torch.zeros(n_layers))  # softmax -> uniform
        self.gamma = nn.Parameter(torch.ones(()))
        self.do_layer_norm = do_layer_norm
        if do_layer_norm:
            self.ln = nn.LayerNorm(input_dim, elementwise_affine=False)
        if head == "linear":
            self.head = nn.Linear(input_dim, output_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def mix(self, x):  # x: [B, n_layers, H]
        if self.do_layer_norm:
            x = self.ln(x)
        w = torch.softmax(self.scalar_weights, dim=0)
        return self.gamma * torch.einsum("l,blh->bh", w, x)

    def forward(self, x):
        return self.head(self.mix(x))

    def weights(self):
        return torch.softmax(self.scalar_weights.detach(), dim=0).cpu().numpy()


def _make_optimizer(model, lr=None, weight_decay=None):
    p = config.TRAIN_PARAMS
    lr = lr if lr is not None else p["learning_rate"]
    wd = weight_decay if weight_decay is not None else p["weight_decay"]
    # Scalar-mix logits need a higher LR and no weight decay to actually
    # concentrate; otherwise the mix stays ~uniform and dilutes the signal.
    if hasattr(model, "scalar_weights"):
        mix_ids = {id(model.scalar_weights), id(model.gamma)}
        head_params = [q for q in model.parameters() if id(q) not in mix_ids]
        return torch.optim.AdamW([
            {"params": head_params, "lr": lr, "weight_decay": wd},
            {"params": [model.scalar_weights, model.gamma],
             "lr": lr * config.SCALARMIX_PARAMS["mix_lr_mult"], "weight_decay": 0.0},
        ])
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _train_torch_model(model, X_train, y_train, X_val, y_val,
                       batch_size=None, epochs=None, lr=None,
                       weight_decay=None, early_stop=None):
    """Generic AdamW / early-stopping trainer for any nn.Module classifier.

    Mirrors ``train_probe`` but accepts inputs of arbitrary trailing shape so it
    can drive both the 2D per-layer probes and the 3D scalar-mix probe.
    """
    p = config.TRAIN_PARAMS
    device = get_device()
    model = model.to(device)
    optim = _make_optimizer(model, lr, weight_decay)
    crit = nn.CrossEntropyLoss()
    bs = batch_size or p["batch_size"]
    epochs = epochs or p["epochs"]
    early_stop = early_stop or p["early_stop"]

    def make_loader(X, y, shuffle):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X)).float(),
                torch.from_numpy(y).long(),
            ),
            batch_size=bs, shuffle=shuffle, pin_memory=True,
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)

    best_acc, best_state, wait = float("-inf"), None, 0
    for _ in range(epochs):
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
        val_acc = correct / max(total, 1)

        if val_acc > best_acc:
            best_acc, best_state, wait = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_logits(model, X, batch_size=None):
    bs = batch_size or config.TRAIN_PARAMS["batch_size"]
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            chunk = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(device)
            out.append(model(chunk).cpu())
    return torch.cat(out, dim=0).numpy()


def _map_control_to_classes(yc_train, yc_val, yc_test, n_classes, seed):
    """Map control-token ids into the label space (same scheme as process_layer)."""
    rng = np.random.RandomState(seed)
    uniq = sorted(set(yc_train.tolist() + yc_val.tolist() + yc_test.tolist()))
    perm = rng.permutation(len(uniq))
    cmap = {uniq[i]: perm[i] % n_classes for i in range(len(uniq))}
    return (np.array([cmap[v] for v in yc_train]),
            np.array([cmap[v] for v in yc_val]),
            np.array([cmap[v] for v in yc_test]))


def process_scalarmix(seed, X_all, y_true, y_control, task, head, layer_count,
                      outdir=None, label_map=None, control_label_map=None):
    """Train a single scalar-mix probe over all layers and report its center of
    gravity, accuracy, F1 and selectivity. ``X_all`` is [N, n_layers, H]."""
    sm = config.SCALARMIX_PARAMS
    uniq, counts = np.unique(y_true, return_counts=True)
    keep = np.isin(y_true, uniq[counts >= 1])
    X_all, y_true, y_control = X_all[keep], y_true[keep], y_control[keep]

    def split3(*arrays, stratify):
        try:
            return train_test_split(*arrays, train_size=config.SPLIT_RATIOS["train"],
                                    random_state=seed, stratify=stratify)
        except ValueError:
            return train_test_split(*arrays, train_size=config.SPLIT_RATIOS["train"],
                                    random_state=seed, stratify=None)

    X_train, X_temp, y_train, y_temp, yc_train, yc_temp = split3(
        X_all, y_true, y_control, stratify=y_true)
    val_frac = config.SPLIT_RATIOS["val"] / (config.SPLIT_RATIOS["val"] + config.SPLIT_RATIOS["test"])
    temp_counts = np.bincount(y_temp)
    X_val, X_test, y_val, y_test, yc_val, yc_test = train_test_split(
        X_temp, y_temp, yc_temp, train_size=val_frac, random_state=seed,
        stratify=y_temp if temp_counts.min() > 1 else None)

    if len(y_test) == 0:
        raise ValueError("scalarmix: no test samples after split.")

    n_classes = int(np.max(y_true) + 1)
    input_dim = X_all.shape[2]
    bs = sm["batch_size"]

    def make_probe():
        return ScalarMixProbe(layer_count, input_dim, n_classes, head=head,
                              hidden_dim=sm["hidden_dim"], dropout=sm["dropout"],
                              do_layer_norm=sm["do_layer_norm"])

    model = _train_torch_model(make_probe(), X_train, y_train, X_val, y_val, batch_size=bs)
    preds = _predict_logits(model, X_test, batch_size=bs).argmax(1)

    yc_train_m, yc_val_m, yc_test_m = _map_control_to_classes(yc_train, yc_val, yc_test, n_classes, seed)
    ctrl = _train_torch_model(make_probe(), X_train, yc_train_m, X_val, yc_val_m, batch_size=bs)
    preds_ctrl = _predict_logits(ctrl, X_test, batch_size=bs).argmax(1)

    weights = model.weights()
    layers = np.arange(layer_count)
    cog = float(np.sum(layers * weights))

    accuracy = float((preds == y_test).mean())
    control_acc = float((preds_ctrl == yc_test_m).mean())
    f1 = float(f1_score(y_test, preds, average="macro"))
    cf1 = float(f1_score(yc_test_m, preds_ctrl, average="macro"))

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "scalarmix_weights.json"), "w") as f:
            json.dump({"weights": weights.tolist(), "gamma": float(model.gamma.detach()),
                       "cog": cog, "n_layers": int(layer_count), "head": head}, f, indent=2)
        torch.save(model.state_dict(), os.path.join(outdir, "scalarmix_probe.pt"))
        if isinstance(label_map, list):
            with open(os.path.join(outdir, "label_map.json"), "w") as f:
                json.dump(label_map, f)

    utils.log_info(f"[scalarmix/{head}] {task} acc {accuracy:.3f} f1 {f1:.3f} "
                   f"ctrl_acc {control_acc:.3f} cog {cog:.2f} gamma {float(model.gamma.detach()):.3f}")

    return {
        f"{task}_acc": accuracy,
        f"{task}_control_acc": control_acc,
        f"{task}_f1": f1,
        f"{task}_control_f1": cf1,
        f"{task}_selectivity": accuracy - control_acc,
        f"{task}_cog": cog,
        "scalarmix_weights": weights.tolist(),
    }


def process_cumulative(seed, X_all, y_true, task, head, outdir=None, label_map=None):
    """Cumulative scoring (Tenney et al. 2019, Eq. 3-4).

    Trains a series of scalar-mix probes P^(l) over layers [0..l] for every l,
    records F1, and returns the expected layer from the differential scores
    Delta^(l) = F1(P^(l)) - F1(P^(l-1)):  E_Delta = sum_l l*Delta / sum_l Delta.
    The endpoints F1(P^(0)) and F1(P^(L)) are the paper's baseline / full-model
    columns. ``X_all`` is [N, n_layers, H]."""
    sm = config.SCALARMIX_PARAMS
    uniq, counts = np.unique(y_true, return_counts=True)
    keep = np.isin(y_true, uniq[counts >= 1])
    X_all, y_true = X_all[keep], y_true[keep]

    n_classes = int(np.max(y_true) + 1)
    input_dim = X_all.shape[2]
    n_layers = X_all.shape[1]
    bs = sm["batch_size"]
    n_seeds = config.CUMULATIVE_PARAMS["n_seeds"]

    def make_split(sd):
        idx = np.arange(len(X_all))
        try:
            tr, tmp = train_test_split(idx, train_size=config.SPLIT_RATIOS["train"],
                                       random_state=sd, stratify=y_true)
        except ValueError:
            tr, tmp = train_test_split(idx, train_size=config.SPLIT_RATIOS["train"],
                                       random_state=sd, stratify=None)
        vf = config.SPLIT_RATIOS["val"] / (config.SPLIT_RATIOS["val"] + config.SPLIT_RATIOS["test"])
        tc = np.bincount(y_true[tmp])
        va, te = train_test_split(tmp, train_size=vf, random_state=sd,
                                  stratify=y_true[tmp] if tc.min() > 1 else None)
        return tr, va, te

    # ---- one-time device residency ----
    # The probes are tiny; naive training is dominated by data movement (a fresh
    # CPU copy of the full [N, L, H] tensor per layer prefix, plus a host->GPU
    # transfer of every ~100-300 MB batch on every step). Instead: apply the
    # (parameter-free) LayerNorm once, park the tensor on the device in fp16, and
    # train every prefix x seed from the resident tensor with on-device index
    # shuffling. The math is identical; only the data path changes.
    p = config.TRAIN_PARAMS
    device = get_device()
    Xg = torch.empty((len(X_all), n_layers, input_dim), dtype=torch.float16, device=device)
    src = torch.from_numpy(X_all)
    for i in range(0, len(X_all), 1024):
        chunk = src[i:i + 1024].to(device, dtype=torch.float32)
        Xg[i:i + 1024] = nn.functional.layer_norm(chunk, (input_dim,)).half()
    yg = torch.from_numpy(y_true.astype(np.int64)).to(device)

    def train_resident(n_prefix, tr, va, seed_offset):
        torch.manual_seed(seed_offset)
        probe = ScalarMixProbe(n_prefix, input_dim, n_classes, head=head,
                               hidden_dim=sm["hidden_dim"], dropout=sm["dropout"],
                               do_layer_norm=False).to(device)   # LN precomputed
        optim = _make_optimizer(probe)
        crit = nn.CrossEntropyLoss()
        tr_t = torch.as_tensor(tr, device=device)
        va_t = torch.as_tensor(va, device=device)
        best_acc, best_state, wait = -1.0, None, 0
        for _ in range(p["epochs"]):
            probe.train()
            perm = tr_t[torch.randperm(len(tr_t), device=device)]
            for i in range(0, len(perm), bs):
                bidx = perm[i:i + bs]
                optim.zero_grad(set_to_none=True)
                loss = crit(probe(Xg[bidx, :n_prefix].float()), yg[bidx])
                loss.backward()
                optim.step()
            probe.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(va_t), 4096):
                    bidx = va_t[i:i + 4096]
                    correct += (probe(Xg[bidx, :n_prefix].float()).argmax(1) == yg[bidx]).sum().item()
            acc = correct / max(len(va_t), 1)
            if acc > best_acc:
                best_acc, wait = acc, 0
                best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            else:
                wait += 1
                if wait >= p["early_stop"]:
                    break
        if best_state is not None:
            probe.load_state_dict(best_state)
        return probe

    def eval_test(probe, n_prefix, te):
        te_t = torch.as_tensor(te, device=device)
        preds = []
        probe.eval()
        with torch.no_grad():
            for i in range(0, len(te_t), 4096):
                bidx = te_t[i:i + 4096]
                preds.append(probe(Xg[bidx, :n_prefix].float()).argmax(1).cpu())
        return torch.cat(preds).numpy()

    # Average each P^(l) over n_seeds runs that vary BOTH the data split and the
    # init (Monte-Carlo CV). Varying the split is essential for small datasets
    # (e.g. coref): there the per-layer noise is dominated by which examples land
    # in the tiny test set, not by training randomness, so a fixed split can't be
    # denoised by re-training alone.
    splits = [make_split(seed + 101 * s) for s in range(n_seeds)]
    acc_runs, f1_runs = np.zeros((n_seeds, n_layers)), np.zeros((n_seeds, n_layers))
    for L in range(n_layers):
        for s, (tr, va, te) in enumerate(splits):
            probe = train_resident(L + 1, tr, va, seed + 101 * s + L)
            preds = eval_test(probe, L + 1, te)
            acc_runs[s, L] = (preds == y_true[te]).mean()
            f1_runs[s, L] = f1_score(y_true[te], preds, average="macro")
        utils.log_info(f"[cumulative/{head}] {task} P^({L}) acc {acc_runs[:, L].mean():.3f} "
                       f"(±{acc_runs[:, L].std():.3f}, {n_seeds} seeds)")
    del Xg

    accs = acc_runs.mean(0).tolist()
    f1s = f1_runs.mean(0).tolist()

    # Expected layer, Tenney et al. Eq. 4, on the RAW differential (no clamping):
    #   E = sum_l l * (acc_l - acc_{l-1}) / (acc_L - acc_0).
    # Leaving the differential unclamped lets plateau noise cancel (up- and
    # down-wiggles offset), so the expectation stays on the layers where accuracy
    # genuinely rises. Clamping to positive-only keeps jitter and drags saturated
    # curves deep. Guard against a near-zero / negative total gain (task not
    # meaningfully learned) -- then the expected layer is undefined.
    acc = np.asarray(accs)
    layers = np.arange(n_layers)
    total = float(acc[-1] - acc[0])
    expected_layer = (float(np.sum(layers[1:] * np.diff(acc)) / total)
                      if total > 1e-3 else float("nan"))

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "cumulative_scores.json"), "w") as f:
            json.dump({"layers": layers.tolist(), "acc": acc.tolist(), "f1": f1s,
                       "baseline_acc": accs[0], "full_acc": accs[-1],
                       "baseline_f1": f1s[0], "full_f1": f1s[-1],
                       "expected_layer": expected_layer, "head": head}, f, indent=2)

    utils.log_info(f"[cumulative/{head}] {task} baseline_acc {accs[0]:.3f} full_acc {accs[-1]:.3f} "
                   f"expected_layer {expected_layer:.2f}")

    return {f"{task}_baseline_acc": float(accs[0]), f"{task}_full_acc": float(accs[-1]),
            f"{task}_baseline_f1": float(f1s[0]), f"{task}_full_f1": float(f1s[-1]),
            f"{task}_expected_layer": expected_layer,
            "cumulative_acc": acc.tolist(), "cumulative_f1": f1s}


def online_code_mdl(X, y, n_classes, head, input_dim, hidden_dim, seed):
    """Prequential (online) codelength in bits (Voita & Titov 2020).

    Data is shuffled once, then transmitted in growing blocks: block 1 is coded
    with a uniform code; each later block is coded by a probe trained on all data
    seen so far. Returns (codelength_bits, compression), where compression is the
    uniform codelength divided by the online codelength.
    """
    rng = np.random.RandomState(seed)
    N = len(X)
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]

    bounds = sorted({int(round(f * N)) for f in config.MDL_PARAMS["fractions"]})
    bounds = [b for b in bounds if b > 0]
    if not bounds or bounds[-1] != N:
        bounds.append(N)

    uniform_bits = np.log2(n_classes)
    codelength = bounds[0] * uniform_bits  # first block: uniform code

    for i in range(len(bounds) - 1):
        tr_end, ev_end = bounds[i], bounds[i + 1]
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_ev, y_ev = X[tr_end:ev_end], y[tr_end:ev_end]

        n_val = max(1, int(0.1 * tr_end))
        if tr_end - n_val >= n_classes:
            Xt, yt, Xv, yv = X_tr[:-n_val], y_tr[:-n_val], X_tr[-n_val:], y_tr[-n_val:]
        else:
            Xt, yt, Xv, yv = X_tr, y_tr, X_tr, y_tr

        if head == "linear":
            model = nn.Linear(input_dim, n_classes)
        else:
            model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(config.MDL_PARAMS["dropout"]),
                                  nn.Linear(hidden_dim, n_classes))
        model = _train_torch_model(model, Xt, yt, Xv, yv)

        logits = _predict_logits(model, X_ev)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))  # log softmax
        codelength += float(-logp[np.arange(len(y_ev)), y_ev].sum() / np.log(2))

    compression = (N * uniform_bits) / codelength if codelength > 0 else float("nan")
    return float(codelength), float(compression)


def process_layer_mdl(seed, X_flat, y_true, y_control, task, head, layer):
    """Per-layer MDL for the task and its control; returns codelength (kbits),
    compression and a compression-based selectivity."""
    n_classes = int(np.max(y_true) + 1)
    input_dim = X_flat.shape[1]
    hidden_dim = config.MDL_PARAMS["hidden_dim"]

    mdl, comp = online_code_mdl(X_flat, y_true, n_classes, head, input_dim, hidden_dim, seed)

    # control: random-but-fixed label per control token, same label space
    rng = np.random.RandomState(seed)
    uniq = sorted(set(y_control.tolist()))
    cmap = {v: rng.permutation(len(uniq))[i] % n_classes for i, v in enumerate(uniq)}
    yc = np.array([cmap[v] for v in y_control])
    mdl_c, comp_c = online_code_mdl(X_flat, yc, n_classes, head, input_dim, hidden_dim, seed)

    utils.log_info(f"[mdl/{head}] layer {layer} {task} mdl {mdl/1e3:.1f}kb comp {comp:.2f} "
                   f"ctrl_comp {comp_c:.2f}")

    return seed, {
        f"{task}_mdl": mdl,
        f"{task}_compression": comp,
        f"{task}_control_mdl": mdl_c,
        f"{task}_control_compression": comp_c,
        f"{task}_compression_selectivity": comp - comp_c,
    }, None


def process_layer(seed, X_flat, y_true, y_control, lambda_reg, task, probe_type, layer, pca_dim, outdir=None, indices=None, label_map=None, control_label_map=None, norm_weight=None):
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
        model = train_probe(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1], n_classes=n_classes, norm_weight=norm_weight)
        scores = model.predict(X_test, batch_size=bs)
        control_model = train_probe(X_train, yc_train_m, X_val, yc_val_m, input_dim=X_train.shape[1], n_classes=n_classes, norm_weight=norm_weight)
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
        d = X_train.shape[1]
        cov = X_train.T.dot(X_train) + lambda_reg * np.eye(d)
        W = np.linalg.solve(cov, X_train.T.dot(np.eye(n_classes)[y_train]))
        
        scores = X_test.dot(W)

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            model_path = os.path.join(outdir, f"probe_layer_{layer}.npy")
            np.save(model_path, W)
            if label_map and isinstance(label_map, list):
                label_map_path = os.path.join(outdir, "label_map.json")
                if not os.path.exists(label_map_path):
                    with open(label_map_path, 'w') as f:
                        json.dump(label_map, f)
        
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

    selectivity = accuracy - control_acc

    utils.log_info(f"[layer {layer}] {task} {probe_type} acc {accuracy:.3f} f1 {f1:.3f} "
                   f"ctrl_acc {control_acc:.3f} ctrl_f1 {cf1:.3f} "
                   f"sel {selectivity:.3f}")

    result = {
        f"{task}_acc": accuracy,
        f"{task}_control_acc": control_acc,
        f"{task}_f1": f1,
        f"{task}_control_f1": cf1,
        f"{task}_selectivity": selectivity,
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
        w.writerow(["Layer", "Acc", "F1", "controlAcc", "controlF1",
                     "Selectivity", "PCA_ExplainedVar"])
        for l in layers:
            r = results[l]
            w.writerow([
                l.split("_")[1],
                r[f"{task}_acc"],
                r[f"{task}_f1"],
                r[f"{task}_control_acc"],
                r[f"{task}_control_f1"],
                r.get(f"{task}_selectivity", -1),
                r["pca_explained_variance"],
            ])