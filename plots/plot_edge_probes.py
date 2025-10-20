import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse

EXP_MIN_REL_GAIN = 0

SMALL_BAR_FRAC = 0.3

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 110
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 30,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.2,
})

model_names = {
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
    "pythia1.4b": "Pythia-1.4B",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "bert-base-uncased": "BERT-Base",
    "bert-large-uncased": "BERT-Large",
    "deberta-v3-large": "DeBERTa-v3-Large",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B",
    "pythia-6.9b_step1": "Pythia-6.9B (step1)",
    "pythia-6.9b_step64": "Pythia-6.9B (step64)",
    "pythia-6.9b_step6000": "Pythia-6.9B (step6k)",
    "pythia-6.9b_step19000": "Pythia-6.9B (step19k)",
    "pythia-6.9b_step37000": "Pythia-6.9B (step37k)",
    "pythia-6.9b_step57000": "Pythia-6.9B (step57k)",
    "pythia-6.9b_step82000": "Pythia-6.9B (step82k)",
    "pythia-6.9b_step111000": "Pythia-6.9B (step111k)",
    "olmo2-7b_stage1-step5000-tokens21B": "OLMo2-7B (5k, 21B tokens)",
    "olmo2-7b_stage1-step40000-tokens168B": "OLMo2-7B (40k, 168B tokens)",
    "olmo2-7b_stage1-step97000-tokens407B": "OLMo2-7B (97k, 407B tokens)",
    "olmo2-7b_stage1-step179000-tokens751B": "OLMo2-7B (179k, 751B tokens)",
    "olmo2-7b_stage1-step282000-tokens1183B": "OLMo2-7B (282k, 1183B tokens)",
    "olmo2-7b_stage1-step409000-tokens1716B": "OLMo2-7B (409k, 1716B tokens)",
    "olmo2-7b_stage1-step559000-tokens2345B": "OLMo2-7B (559k, 2345B tokens)",
    "olmo2-7b_stage1-step734000-tokens3079B": "OLMo2-7B (734k, 3079B tokens)"
}

# ---------------------------------------------------------------------
# Your model list and task setup
# ---------------------------------------------------------------------
all_models = [
    "bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
    "gpt2", "gpt2-large", "gpt2-xl",
    "gemma2b", "gemma2b-it",
    "qwen2", "qwen2-instruct",
    "qwen2.5-7B", "qwen2.5-7B-instruct",
    "pythia-6.9b",
    "pythia-6.9b-tulu",
    "olmo2-7b", "olmo2-7b-instruct",
    "llama3-8b", "llama3-8b-instruct",
]

# Requested order (top to bottom): POS, Consts., Deps., Entities, SRL, Coref., SPR, Relations
task_list = ["pos", "constituents", "dep", "ner", "srl", "coref", "spr", "relation"]

# Display names for y-axis
task_display = {
    "pos": "POS",
    "constituents": "Consts.",
    "dep": "Deps.",
    "ner": "Entities",
    "srl": "SRL",
    "coref": "Coref.",
    "spr": "SPR",
    "relation": "Relations",
}

task_to_dataset = {
    "pos": "ud_gum_pos",
    "dep": "ud_gum_dep",
    "ner": "ud_gum_ner",
    "coref": "ud_gum_coref_pairs",
    "constituents": "ud_gum_constituents",
    "srl": "up_ewt_srl",
    "spr": "spr",
    "relation": "semeval2010_relations",
}

def get_acc_columns(df, prefix):
    if f"{prefix}_Accuracy" in df.columns and f"{prefix}_ControlAccuracy" in df.columns:
        return f"{prefix}_Accuracy", f"{prefix}_ControlAccuracy"
    if "Acc" in df.columns and "controlAcc" in df.columns:
        return "Acc", "controlAcc"
    for acc_col in df.columns:
        if acc_col.lower() == f"{prefix}_accuracy":
            for ctrl_col in df.columns:
                if ctrl_col.lower() == f"{prefix}_controlaccuracy":
                    return acc_col, ctrl_col
    if f"{prefix}_Accuracy" in df.columns:
        return f"{prefix}_Accuracy", None
    if "Acc" in df.columns:
        return "Acc", None
    raise ValueError("Could not find accuracy columns in DataFrame.")

def _to_unit_interval(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0 or not np.isfinite(np.nanmax(a)):
        return a
    mx = float(np.nanmax(a))
    mn = float(np.nanmin(a))
    # Already in [0, 1]
    if 0.0 - 1e-9 <= mn and mx <= 1.0 + 1e-9:
        return a
    # Common case: [0, 100] percentages
    if 0.0 - 1e-9 <= mn and mx <= 100.0 + 1e-9:
        return a / 100.0
    # Fallback: clip to [0, 1] and warn
    print(f"[WARN] Accuracy values outside [0,1] or [0,100]: min={mn:.4g}, max={mx:.4g}. Clipping to [0,1].")
    return np.clip(a, 0.0, 1.0)

def _read_task_curve(dataset, model, probe, task, pca=False, pca_dim=50):
    base_probe_dir = os.path.join("..", "output", "edge_probes")

    # Try a few common directory name formats:
    #  - {dataset}_{model}_{probe}
    #  - {dataset}_{model}_{task}_{probe}  (some runs include the task again)
    #  - {dataset}_{task}_{model}_{probe}  (less common)
    candidate_names = [
        f"{dataset}_{model}_{probe}",
        f"{dataset}_{model}_{task}_{probe}",
        f"{dataset}_{task}_{model}_{probe}",
    ]

    pca_suffixes = [""]
    if pca:
        pca_suffixes = ["", f"_pca_{pca_dim}"]

    searched = []
    found_dir = None
    # Check explicit candidate names first
    for cand in candidate_names:
        for suf in pca_suffixes:
            probe_dir = os.path.join(base_probe_dir, cand + suf)
            csv_path = os.path.join(probe_dir, f"{task}_results.csv")
            searched.append(csv_path)
            if os.path.exists(csv_path):
                found_dir = probe_dir
                break
        if found_dir:
            break

    # If nothing found, try a looser match: any directory under edge_probes that contains
    # the dataset, model and probe substrings (useful when naming varies a bit).
    matches = []
    if found_dir is None:
        try:
            for name in os.listdir(base_probe_dir):
                if dataset in name and model in name and probe in name:
                    probe_dir = os.path.join(base_probe_dir, name)
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    searched.append(csv_path)
                    if os.path.exists(csv_path):
                        matches.append(probe_dir)
        except FileNotFoundError:
            # base directory doesn't exist
            pass

        if matches:
            if len(matches) > 1:
                print(f"[WARN] Multiple candidate probe dirs found for {dataset} {model} {task}: using first: {matches[0]}")
            found_dir = matches[0]

    if found_dir is None:
        print(f"[WARN] No results found for {dataset} {model} {task} {probe}")
        return None

    df = pd.read_csv(os.path.join(found_dir, f"{task}_results.csv"))
    try:
        acc_col, ctrl_col = get_acc_columns(df, task)
    except Exception:
        print(f"[WARN] Columns not found for {dataset} {model} {task}")
        return None
    df = df.sort_values("Layer")

    acc = _to_unit_interval(df[acc_col].to_numpy())
    ctrl = None
    if ctrl_col is not None and ctrl_col in df.columns:
        ctrl = _to_unit_interval(df[ctrl_col].to_numpy())

    out = {
        "layers": df["Layer"].to_numpy(),
        "acc": acc,
        "ctrl": ctrl,
    }
    return out

def _build_matrix_for_model(
    model, task_list, task_to_dataset, probe="nn",
    metric="accuracy", pca=False, pca_dim=50
):
    rows = []
    max_L = 0
    for task in task_list:
        dataset = task_to_dataset[task]
        info = _read_task_curve(dataset, model, probe, task, pca=pca, pca_dim=pca_dim)
        rows.append((task, info))
        if info is not None:
            max_L = max(max_L, int(info["layers"].max()))
    W = max_L + 1 if max_L > 0 else 1
    H = len(task_list)
    M = np.full((H, W), np.nan)
    
    for i, (task, info) in enumerate(rows):
        if info is None:
            continue
        if metric == "accuracy":
            y = info["acc"]
        else:
            base = info["ctrl"] if info["ctrl"] is not None else 0.0
            y = info["acc"] - base

        if metric == "accuracy" and (np.any(y < 0.0) or np.any(y > 1.0)):
            raise ValueError(f"Values out of bounds [0,1]: {y}")

        L = info["layers"].astype(int)
        L = L - L.min()
        for l_idx, val in zip(L, y):
            if 0 <= l_idx < W:
                M[i, l_idx] = val
    return M, W

def _tick_step(W, max_ticks=12):
    return 1 if W <= max_ticks else int(math.ceil(W / max_ticks))

def _percent_xticks(W):
    # centers of heatmap cells are at 0.5, 1.5, ..., W-0.5
    perc = np.array([0, 25, 50, 75, 100], dtype=float)
    if W <= 1:
        # degenerate single-layer case
        return np.array([0.5]), ["0"]
    pos = (perc / 100.0) * max(W - 1, 1) + 0.5
    labels = [str(int(p)) for p in perc]
    return pos, labels

def plot_task_layer_heatmaps(
    task_list, task_to_dataset, model_list,
    probe="nn", metric="accuracy", pca=False, pca_dim=50,
    cmap="Blues", output_dir="figures3",
    fname="heatmaps.png", share_scale=True, cols_per_row=3,
    grid=None,  # None => auto: True if >1 model, False otherwise
):
    if grid is None:
        grid = len(model_list) > 1

    n = len(model_list)
    cols = min(cols_per_row, n) if n > 0 else 1
    rows = int(math.ceil(n / max(cols, 1)))

    # Always use [0, 1] for colorbar limits
    def limits():
        return 0.0, 1.0
    
    if not grid:
        model = model_list[0]
        M_plot, _ = _build_matrix_for_model(
            model, task_list, task_to_dataset, probe, metric, pca, pca_dim
        )
        Wp = M_plot.shape[1]

        fig_w = 11.5
        fig_h = max(6.0, 2.2 + 0.62 * len(task_list))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

        vmin, vmax = limits()
        hm = sns.heatmap(
            M_plot,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar=True,
            linewidths=0.0,
            square=False,
        )

        # y labels
        ax.set_yticks(np.arange(len(task_list)) + 0.5)
        ax.set_yticklabels([task_display.get(t, t) for t in task_list], rotation=0, ha="right")

        # x labels (percentage-based)
        xticks, labels = _percent_xticks(Wp)
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("Layer Depth (%)")

        # title and spines
        ax.set_title(model_names.get(model, model), pad=6, fontweight="bold")
        ax.set_aspect("auto")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)

        probe_type = "Linear" if probe == "reg" else "MLP"

        if metric == "accuracy":
            hm.collections[0].colorbar.set_label(f"{probe_type} Probe - Accuracy", rotation=270, labelpad=30, fontsize=20)
        elif metric == "selectivity":
            hm.collections[0].colorbar.set_label(f"{probe_type} Probe - Selectivity", rotation=270, labelpad=30, fontsize=20)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved heatmaps to {out_path}")
        plt.close(fig)
        return

    # Grid mode: many models per figure, one shared colorbar, sparse ticks
    # Make subplots a bit wider than square by adjusting width and height per subplot
    fig_w_per_col = 5.2
    fig_h_per_row = 4.5
    fig_w = fig_w_per_col * cols
    fig_h = fig_h_per_row * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    first_mappable = None
    for idx, model in enumerate(model_list):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        M_plot, _ = _build_matrix_for_model(model, task_list, task_to_dataset, probe, metric, pca, pca_dim)
        Wp = M_plot.shape[1]
        vmin, vmax = limits()

        hm = sns.heatmap(
            M_plot,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar=False,  # one shared cbar later
            linewidths=0.0,
            square=False,
        )
        if first_mappable is None and len(hm.collections) > 0:
            first_mappable = hm.collections[0]

        # y ticks only on first column
        if c == 0:
            ax.set_yticks(np.arange(len(task_list)) + 0.5)
            ax.set_yticklabels([task_display.get(t, t) for t in task_list], rotation=0, ha="right")
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        # x ticks only on last row (percentage-based)
        if r == rows - 1:
            xticks, labels = _percent_xticks(Wp)
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels, rotation=0)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.set_title(model_names.get(model, model), pad=6, fontsize=24)
        ax.set_aspect("auto")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)

    # Hide any unused axes
    for k in range(n, rows * cols):
        r = k // cols
        c = k % cols
        axes[r][c].axis("off")

    # Shared x label
    fig.text(0.5, 0.02, "Layer Depth (%)", ha="center", va="center", fontsize=24)

    # Reserve space on the right and add a figure-level colorbar in a fixed spot
    right_margin = 0.92
    plt.tight_layout(rect=[0.03, 0.04, right_margin, 0.98])

    if first_mappable is not None:
        cax = fig.add_axes([right_margin + 0.01, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(first_mappable, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        probe_type = "Linear" if probe == "reg" else "MLP"

        if metric == "accuracy":
            cbar.set_label(f"{probe_type} Probe - Accuracy", rotation=270, labelpad=35, fontsize=20)
        elif metric == "selectivity":
            cbar.set_label(f"{probe_type} Probe - Selectivity", rotation=270, labelpad=35, fontsize=20)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved heatmaps to {out_path}")
    plt.close(fig)

def _expected_layer_from_curve(layers: np.ndarray,
                               acc: np.ndarray,
                               min_rel_gain: float = EXP_MIN_REL_GAIN) -> float:
    """
    Expected layer per Tenney et al.:
      use a cumulative (nondecreasing) score curve and take an
      expectation over its differential gains.

    We approximate the cumulative curve with the running maximum
    of the per-layer accuracy (no external deps, robust to noise).

      b[l] = max_{j <= l} acc[j]
      delta[l] = max(b[l] - b[l-1], 0), for l >= 1

    We then drop very small increments (< min_rel_gain of total)
    so late-time noise does not push the expectation too deep.
    """
    if layers.size < 2 or acc.size < 2:
        return np.nan

    L = layers.astype(int) - int(layers.min())
    b = np.maximum.accumulate(np.asarray(acc, dtype=float))
    deltas = np.maximum(b[1:] - b[:-1], 0.0)
    total = float(np.sum(deltas))
    if not np.isfinite(total) or total <= 0:
        return np.nan

    # zero-out tiny increments that are just jitter
    thr = float(min_rel_gain) * total
    deltas = np.where(deltas >= thr, deltas, 0.0)
    denom = float(np.sum(deltas))
    if denom <= 0:
        return np.nan

    layers_ge1 = L[1:]
    return float(np.sum(layers_ge1 * deltas) / denom)

def _compute_summary_stats_for_plot(
    model: str,
    task_list: list,
    task_to_dataset: dict,
    pca: bool = False,
    pca_dim: int = 50,
):
    summary_data = {task: {} for task in task_list}
    max_depth_global = 0

    for task in task_list:
        dataset = task_to_dataset[task]

        # Linear probe (reg)
        info_reg = _read_task_curve(dataset, model, "reg", task, pca=pca, pca_dim=pca_dim)
        if info_reg is None or len(info_reg["acc"]) == 0:
            summary_data[task]["linear_acc"] = np.nan
            summary_data[task]["selectivity_reg"] = np.nan  # <-- add this line
        else:
            layers_r = info_reg["layers"].astype(int)
            layers_r = layers_r - layers_r.min()
            summary_data[task]["linear_acc"] = info_reg["acc"][-1]
            # Compute selectivity for reg
            if info_reg["ctrl"] is not None and len(info_reg["ctrl"]) == len(info_reg["acc"]):
                summary_data[task]["selectivity_reg"] = float(info_reg["acc"][-1] - info_reg["ctrl"][-1])
            else:
                summary_data[task]["selectivity_reg"] = np.nan
            max_depth_global = max(max_depth_global, int(layers_r.max()))


        # MLP probe (nn)
        info_nn = _read_task_curve(dataset, model, "nn", task, pca=pca, pca_dim=pca_dim)
        if info_nn is None or len(info_nn["acc"]) == 0:
            summary_data[task].update(
                {"mlp_acc": np.nan, "selectivity": np.nan, "exp_layer": np.nan, "cog": np.nan}
            )
            continue

        layers = info_nn["layers"].astype(int)
        layers = layers - layers.min()
        max_depth_global = max(max_depth_global, int(layers.max()))

        y = np.asarray(info_nn["acc"], dtype=float)
        summary_data[task]["mlp_acc"] = float(y[-1])

        if info_nn["ctrl"] is not None and len(info_nn["ctrl"]) == len(y):
            summary_data[task]["selectivity"] = float(y[-1] - info_nn["ctrl"][-1])
        else:
            summary_data[task]["selectivity"] = np.nan

        summary_data[task]["exp_layer"] = _expected_layer_from_curve(layers, y)

        # Center of gravity (Eq. 2) using proxy: nonnegative scores as soft weights.
        s_proxy = y - np.nanmin(y)
        if np.sum(s_proxy) > 1e-8:
            s_proxy = s_proxy / np.sum(s_proxy)
            summary_data[task]["cog"] = float(np.sum(layers * s_proxy))
        else:
            summary_data[task]["cog"] = np.nan

    return summary_data, max_depth_global

def plot_bertology_summary(
    model: str,
    task_list: list,
    task_to_dataset: dict,
    pca: bool = False,
    pca_dim: int = 50,
    output_dir: str = "figures_bertology",
    fname: str = None,
    color_exp: str = "#C7B9FF",
    color_cog: str = "#0072b2",
):
    def fmt_pct(x):
        return f"{x * 100:.1f}" if pd.notna(x) else "-"

    summary_data, max_depth = _compute_summary_stats_for_plot(
        model, task_list, task_to_dataset, pca=pca, pca_dim=pca_dim
    )

    exp_arr = np.array([summary_data[t].get("exp_layer", np.nan) for t in task_list], dtype=float)
    cog_arr = np.array([summary_data[t].get("cog", np.nan) for t in task_list], dtype=float)

    left_vals = [
        (
            fmt_pct(summary_data[t].get("linear_acc")),
            fmt_pct(summary_data[t].get("mlp_acc")),
            fmt_pct(summary_data[t].get("selectivity")),
            fmt_pct(summary_data[t].get("selectivity_reg")),
        )
        for t in task_list
    ]

    row_labels = [task_display.get(t, t) for t in task_list]
    n_tasks = len(task_list)
    y_pos = np.arange(n_tasks)

    fig_w = 11.2
    fig_h = max(4.8, 1.1 + 0.55 * n_tasks)
    fig, (ax_names, ax_left, ax_bar) = plt.subplots(
        1, 3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.00, 1.10, 5.60]},
        sharey=True,
    )

    bar_h = 0.85
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([])
    ax_bar.set_ylim(-0.5, n_tasks - 0.5)
    ax_bar.set_title(model_names.get(model, model), pad=12, fontsize=20)

    finite = np.concatenate([exp_arr[np.isfinite(exp_arr)], cog_arr[np.isfinite(cog_arr)]])
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    ax_bar.set_xlim(0, max(xmax + 1.0, max(1, max_depth) * 1.02))

    major_step = max(2, int(np.ceil(ax_bar.get_xlim()[1] / 8.0)))
    ax_bar.set_xticks(np.arange(0, ax_bar.get_xlim()[1] + 0.1, major_step))
    ax_bar.set_xticks(np.arange(0, ax_bar.get_xlim()[1] + 0.1, 1), minor=True)
    ax_bar.set_xlabel("Expected layer and center-of-gravity", fontsize=16, labelpad=8)
    ax_bar.tick_params(axis="x", which="major", length=12, width=2.5, top=False, bottom=True, direction="out", labelsize=12)
    ax_bar.tick_params(axis="x", which="minor", length=5, width=1.5, top=False, bottom=True, direction="out", labelbottom=False)

    # Increase border line thickness
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(True)
    ax_bar.spines["left"].set_color("black")
    ax_bar.spines["left"].set_linewidth(2.5)
    ax_bar.spines["bottom"].set_linewidth(2.5)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax_bar.set_axisbelow(True)

    def pos_for_label(val, prefer_inside_color, outside_color,
                      force_outside=False, outside_at=None):
        if not np.isfinite(val):
            return None
        xr = ax_bar.get_xlim()[1]
        inside = (val >= 1.2) and not force_outside
        if inside:
            x = min(max(0.35, val - 0.35), xr - 0.05)
            ha = "right"
            color = prefer_inside_color
        else:
            anchor = outside_at if (outside_at is not None) else val
            x = min(anchor + 0.25, xr - 0.05)
            ha = "left"
            color = outside_color
        return x, ha, color

    bar_left_margin = 0.1
    for yi, (e, c) in enumerate(zip(exp_arr, cog_arr)):
        if not (np.isfinite(e) or np.isfinite(c)):
            continue

        items = [("exp", e, color_exp), ("cog", c, color_cog)]
        items.sort(key=lambda t: t[1], reverse=True)
        (k1, v1, col1), (k2, v2, col2) = items

        ax_bar.barh(yi, v1 - bar_left_margin, left=bar_left_margin, color=col1, height=bar_h, zorder=2)
        ax_bar.barh(yi, v2 - bar_left_margin, left=bar_left_margin, color=col2, height=bar_h, zorder=3)

        # Decide where to place labels
        force_out_e = False
        force_out_c = False
        outside_at_e = None
        outside_at_c = None

        m = max(e, c) if (np.isfinite(e) and np.isfinite(c)) else (e if np.isfinite(e) else c)
        m = m if np.isfinite(m) else 0.0

        # If bars are close, push the larger one outside so both are legible
        if m > 0 and np.isfinite(e) and np.isfinite(c) and abs(e - c) < 0.1 * m:
            if e >= c:
                force_out_e = True
            else:
                force_out_c = True

        if m > 0 and np.isfinite(e) and np.isfinite(c):
            if e <= c and e / m < SMALL_BAR_FRAC:
                force_out_e = True
                outside_at_e = e
            elif c < e and c / m < SMALL_BAR_FRAC:
                force_out_c = True
                outside_at_c = c

        # Place labels
        pc = pos_for_label(c, "white", "black", force_out_c, outside_at_c) if np.isfinite(c) else None
        pe = pos_for_label(e, "black", "black", force_out_e, outside_at_e) if np.isfinite(e) else None

        # If exp is tiny and label is outside, use white text
        if force_out_e and outside_at_e is not None and e <= c and e / m < SMALL_BAR_FRAC:
            pe = pos_for_label(e, "white", "white", force_out_e, outside_at_e)
        if force_out_c and outside_at_c is not None and c < e and c / m < SMALL_BAR_FRAC:
            pc = pos_for_label(c, "white", "black", force_out_c, outside_at_c)

        if pc:
            x, ha, clr = pc
            ax_bar.text(x, yi, f"{c:.2f}", va="center", ha=ha, fontsize=16, color=clr, clip_on=True, zorder=6)
        if pe:
            x, ha, clr = pe
            ax_bar.text(x, yi, f"{e:.2f}", va="center", ha=ha, fontsize=16, color=clr, clip_on=True, zorder=6)

    ax_names.set_xlim(0, 1)
    ax_names.set_ylim(-0.5, n_tasks - 0.5)
    ax_names.invert_yaxis()
    ax_names.axis("off")
    ax_names.invert_yaxis()
    for yi, name in enumerate(row_labels):
        ax_names.text(0.5, yi, name, va="center", ha="right", fontsize=16)

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(-0.5, n_tasks - 0.5)
    ax_left.invert_yaxis()
    ax_left.axis("off")
    headers = ["Linear", r"$\tau_{Lin}$", "MLP", r"$\tau_{MLP}$"]
    col_x = [-0.38, 0.06, 0.48, 0.9]
    for i, h in enumerate(headers):
        ax_left.text(col_x[i], 1.02, h, transform=ax_left.transAxes, ha="center", va="bottom", fontsize=16, color="black")
        
    x0 = min(col_x) - 0.25
    x1 = 0.95 + 0.2
    y0 = -0.45
    y1 = (n_tasks - 0.55)
    ax_left.fill(
        [x0, x1, x1, x0],
        [y0, y0, y1, y1],
        facecolor="#e9e9e9",
        edgecolor="#bdbdbd",
        linewidth=1.0,
        zorder=1,
        clip_on=False,
    )

    for yi, (lin_v, mlp_v, sel_mlp_v, sel_lin_v) in enumerate(left_vals):
        ax_left.text(col_x[0], yi, lin_v, va="center", ha="center", fontsize=14, color="black", zorder=5)
        ax_left.text(col_x[1], yi, sel_lin_v, va="center", ha="center", fontsize=14, color="black", zorder=5)
        ax_left.text(col_x[2], yi, mlp_v, va="center", ha="center", fontsize=14, color="black", zorder=5)
        ax_left.text(col_x[3], yi, sel_mlp_v, va="center", ha="center", fontsize=14, color="black", zorder=5)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.10, wspace=0.12)
    os.makedirs(output_dir, exist_ok=True)
    if fname is None:
        fname = f"bertology_summary_{model}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot to {out_path}")

def plot_bertology_summary_grid(
    model_list: list,
    task_list: list,
    task_to_dataset: dict,
    pca: bool = False,
    pca_dim: int = 50,
    output_dir: str = "figures_bertology_grid",
    fname: str = "bertology_summary_grid.png",
    cols_per_row: int = 3,
    color_exp: str = "#C7B9FF",
    color_cog: str = "#0072b2",
):
    n = len(model_list)
    cols = min(cols_per_row, n) if n > 0 else 1
    rows = int(np.ceil(n / max(cols, 1)))
    n_tasks = len(task_list)
    fig_w = 11.2 * cols
    fig_h = max(6.5, 1.6 + 0.85 * n_tasks) * rows
    # Add an extra column for row labels
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer_gs = gridspec.GridSpec(rows, cols, wspace=0.12, hspace=0.5)
    for idx, model in enumerate(model_list):
        r = idx // cols
        c = idx % cols
        # Add a 3-column layout: [row labels | number columns | bar plot]
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[r, c], width_ratios=[1.8, 1.2, 5.6], wspace=0.04)
        # Row labels axis
        ax_labels = fig.add_subplot(inner_gs[0, 0])
        # Number columns
        ax_left = fig.add_subplot(inner_gs[0, 1])
        # Bar plot
        ax_bar = fig.add_subplot(inner_gs[0, 2])
        summary_data, max_depth = _compute_summary_stats_for_plot(
            model, task_list, task_to_dataset, pca=pca, pca_dim=pca_dim
        )
        exp_arr = np.array([summary_data[t].get("exp_layer", np.nan) for t in task_list], dtype=float)
        cog_arr = np.array([summary_data[t].get("cog", np.nan) for t in task_list], dtype=float)
        def fmt_pct(x):
            return f"{x * 100:.1f}" if pd.notna(x) else "-"
        left_vals = [
            (
                fmt_pct(summary_data[t].get("linear_acc")),
                fmt_pct(summary_data[t].get("mlp_acc")),
                fmt_pct(summary_data[t].get("selectivity")),
                fmt_pct(summary_data[t].get("selectivity_reg")),
            )
            for t in task_list
        ]
        row_labels = [task_display.get(t, t) for t in task_list]
        y_pos = np.arange(n_tasks)
        # --- Row labels ---
        ax_labels.set_xlim(0, 1)
        ax_labels.set_ylim(-0.5, n_tasks - 0.5)
        ax_labels.axis("off")
        ax_labels.invert_yaxis()
        for yi, name in enumerate(row_labels):
            ax_labels.text(0.0, yi, name, va="center", ha="right", fontsize=18)
        # --- Number columns ---
        ax_left.set_xlim(0, 1)
        ax_left.set_ylim(-0.5, n_tasks - 0.5)
        ax_left.axis("off")
        ax_left.invert_yaxis()
        headers = ["Linear", r"$\tau_{Lin}$", "MLP", r"$\tau_{MLP}$"]
        # col_x = [-1.18, -0.58, 0.02, 0.62]
        col_x = [-1.08, -0.48, 0.12, 0.72]
        for i, h in enumerate(headers):
            ax_left.text(col_x[i], 1.02, h, transform=ax_left.transAxes, ha="center", va="bottom", fontsize=14, color="black")
        x0 = min(col_x) - 0.35
        x1 = 1.0
        y0 = -0.45
        y1 = (n_tasks - 0.65)
        ax_left.fill(
            [x0, x1, x1, x0],
            [y0, y0, y1, y1],
            facecolor="#e9e9e9",
            edgecolor="#bdbdbd",
            linewidth=1.0,
            zorder=1,
            clip_on=False,
        )
        for yi, (lin_v, mlp_v, sel_mlp_v, sel_lin_v) in enumerate(left_vals):
            ax_left.text(col_x[0], yi, lin_v, va="center", ha="center", fontsize=18, color="black", zorder=5, fontfamily="sans-serif", fontweight="bold")
            ax_left.text(col_x[1], yi, sel_lin_v, va="center", ha="center", fontsize=18, color="black", zorder=5, fontfamily="sans-serif", fontweight="bold")
            ax_left.text(col_x[2], yi, mlp_v, va="center", ha="center", fontsize=18, color="black", zorder=5, fontfamily="sans-serif", fontweight="bold")
            ax_left.text(col_x[3], yi, sel_mlp_v, va="center", ha="center", fontsize=18, color="black", zorder=5, fontfamily="sans-serif", fontweight="bold")

        # --- Bar plot ---
        bar_h = 0.85
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([])
        ax_bar.set_ylim(-0.5, n_tasks - 0.5)
        ax_bar.invert_yaxis()
        ax_bar.set_title(model_names.get(model, model), pad=12, fontsize=20)
        
        finite = np.concatenate([exp_arr[np.isfinite(exp_arr)], cog_arr[np.isfinite(cog_arr)]])
        xmax = float(np.nanmax(finite)) if finite.size else 1.0
        ax_bar.set_xlim(0, max(xmax + 1.0, max(1, max_depth) * 1.02))
        major_step = max(2, int(np.ceil(ax_bar.get_xlim()[1] / 8.0)))
        
        ax_bar.set_xticks(np.arange(0, ax_bar.get_xlim()[1] + 0.1, major_step))
        ax_bar.set_xticks(np.arange(0, ax_bar.get_xlim()[1] + 0.1, 1), minor=True)
        ax_bar.set_xlabel("Layer", fontsize=14, labelpad=10)
        # Custom tick marks: large for major, short for minor
        ax_bar.tick_params(axis="x", which="major", length=12, width=2.5, top=False, bottom=True, direction="out", labelsize=12)
        ax_bar.tick_params(axis="x", which="minor", length=5, width=1.5, top=False, bottom=True, direction="out", labelbottom=False)
        
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_visible(True)
        ax_bar.spines["left"].set_color("black")
        ax_bar.spines["left"].set_linewidth(2.5)
        ax_bar.spines["bottom"].set_linewidth(2.5)
        
        ax_bar.tick_params(axis="y", length=0)
        ax_bar.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax_bar.set_axisbelow(True)
        bar_left_margin = 0.1
        xr = ax_bar.get_xlim()[1]
        def right_of(x, pad=0.2):
            return min(x + pad, xr - 0.05)

        for yi, (e, c) in enumerate(zip(exp_arr, cog_arr)):
            if not (np.isfinite(e) or np.isfinite(c)):
                continue

            items = [("exp", e, color_exp), ("cog", c, color_cog)]
            items.sort(key=lambda t: t[1], reverse=True)
            (k1, v1, col1), (k2, v2, col2) = items

            ax_bar.barh(yi, v1 - bar_left_margin, left=bar_left_margin, color=col1, height=bar_h, zorder=2)
            ax_bar.barh(yi, v2 - bar_left_margin, left=bar_left_margin, color=col2, height=bar_h, zorder=3)

            m = max(e, c) if (np.isfinite(e) and np.isfinite(c)) else (e if np.isfinite(e) else c)
            m = m if np.isfinite(m) else 0.0

            # Default placements: cog to the right of its own bar, exp just inside its bar
            x_c = right_of(c if np.isfinite(c) else 0.0)
            ha_c = "left"
            x_e = max(0.35, (e if np.isfinite(e) else 0.0) - 0.15)
            ha_e = "right"
            
            if m > 0 and np.isfinite(e) and np.isfinite(c) and abs(e - c) < 0.3 * m:
                if e >= c:
                    x_e = right_of(e)
                    ha_e = "left"
                else:
                    x_c = right_of(c)
                    ha_c = "left"

            color_e = "black"
            color_c = "black"
            if m > 0 and np.isfinite(e) and np.isfinite(c):
                if e <= c and e / m < SMALL_BAR_FRAC:
                    x_e = right_of(e)
                    ha_e = "left"
                    color_e = "white"  # exp is always light purple
                elif c < e and c / m < SMALL_BAR_FRAC:
                    x_c = right_of(c)
                    ha_c = "left"
                    color_c = "white" if k2 == "exp" else "black"

            if np.isfinite(c):
                ax_bar.text(x_c, yi, f"{c:.2f}", va="center", ha=ha_c, fontsize=18, color=color_c, clip_on=True, zorder=6, fontfamily="sans-serif", fontweight="bold")
            if np.isfinite(e):
                ax_bar.text(x_e, yi, f"{e:.2f}", va="center", ha=ha_e, fontsize=18, color=color_e, clip_on=True, zorder=6, fontfamily="sans-serif", fontweight="bold")

    plt.tight_layout(rect=[0.03, 0.04, 0.98, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bertology summary grid to {out_path}")

def _percent_grid_for_model(
    model,
    task_list,
    task_to_dataset,
    probe="nn",
    metric="accuracy",
    bins=40,
    pca=False,
    pca_dim=50
):
    H = len(task_list)
    G = np.full((H, bins), np.nan, dtype=float)
    xs = np.linspace(0.0, 1.0, bins)

    for i, task in enumerate(task_list):
        dataset = task_to_dataset[task]
        info = _read_task_curve(dataset, model, probe, task, pca=pca, pca_dim=pca_dim)
        if info is None or info["acc"] is None or len(info["acc"]) == 0:
            continue
        layers = np.asarray(info["layers"], dtype=float)
        # Align to start at 0 and normalize to [0, 1]
        L = layers - np.nanmin(layers)
        Lmax = np.nanmax(L)
        if not np.isfinite(Lmax) or Lmax <= 0:
            Lnorm = np.zeros_like(L)
        else:
            Lnorm = L / Lmax

        acc = np.asarray(info["acc"], dtype=float)
        if metric == "selectivity":
            if info["ctrl"] is not None and len(info["ctrl"]) == len(acc):
                vals = acc - np.asarray(info["ctrl"], dtype=float)
            else:
                vals = acc - 0.0
        else:
            vals = acc

        # Assign each value to the closest bin (no interpolation)
        bin_idx = np.round(Lnorm * (bins - 1)).astype(int)
        for j, v in zip(bin_idx, vals):
            if 0 <= j < bins:
                G[i, j] = v
    return G

def _vectorize_grid(G):
    return G.reshape(-1)

def _pearsonr_nonnan(x, y):
    """Pearson correlation without SciPy. Returns np.nan if undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n < 2:
        return np.nan
    x = x[:n]
    y = y[:n]
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.sqrt(np.sum(x * x))
    sy = np.sqrt(np.sum(y * y))
    if sx == 0.0 or sy == 0.0:
        return np.nan
    return float(np.sum(x * y) / (sx * sy))

def _spearman_from_vectors(a, b):
    """
    Spearman correlation = Pearson correlation of ranks.
    Uses pandas for stable tie-handling; no SciPy dependency.
    """
    if a.size == 0 or b.size == 0:
        return np.nan
    # Use ranks with average tie handling. Convert to numpy arrays.
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return _pearsonr_nonnan(ra, rb)

def _model_similarity_vector(
    model,
    task_list,
    task_to_dataset,
    probe="nn",
    metric="accuracy",
    bins=40,
    pca=False,
    pca_dim=50):
    """
    Build the resampled percent grid for a model and vectorize it.
    """
    G = _percent_grid_for_model(
        model, task_list, task_to_dataset,
        probe=probe, metric=metric, bins=bins, pca=pca, pca_dim=pca_dim
    )
    return _vectorize_grid(G)

def _pairwise_similarity_matrix(
    model_list,
    task_list,
    task_to_dataset,
    probe="nn",
    metric="accuracy",
    bins=40,
    pca=False,
    pca_dim=50,
    method="spearman"):
    vecs = {}
    for m in model_list:
        vecs[m] = _model_similarity_vector(
            m, task_list, task_to_dataset,
            probe=probe, metric=metric, bins=bins, pca=pca, pca_dim=pca_dim
        )

    N = len(model_list)
    S = np.full((N, N), np.nan, dtype=float)
    for i, mi in enumerate(model_list):
        vi = vecs[mi]
        for j, mj in enumerate(model_list):
            if j <= i:
                continue
            vj = vecs[mj]
            if vi.size == 0 or vj.size == 0:
                S[i, j] = np.nan
                continue

            # Align on bins where BOTH are finite
            mask = np.isfinite(vi) & np.isfinite(vj)
            ai = vi[mask]
            bj = vj[mask]
            if ai.size < 2:
                S[i, j] = np.nan
                continue

            if method == "pearson":
                S[i, j] = _pearsonr_nonnan(ai, bj)
            else:
                S[i, j] = _spearman_from_vectors(ai, bj)
    return S

def plot_model_similarity_triangular(
    model_list,
    task_list,
    task_to_dataset,
    metric="accuracy",
    bins=40,
    method="spearman",
    output_dir="heatmap_figures",
    fname="model_similarity_triangular.png",
    title=None,
    pca=False,
    pca_dim=50):
    # Upper triangle (reg)
    S_reg = _pairwise_similarity_matrix(
        model_list, task_list, task_to_dataset,
        probe="reg", metric=metric, bins=bins, pca=pca, pca_dim=pca_dim, method=method
    )
    # Lower triangle (nn)
    S_nn = _pairwise_similarity_matrix(
        model_list, task_list, task_to_dataset,
        probe="nn", metric=metric, bins=bins, pca=pca, pca_dim=pca_dim, method=method
    )

    N = len(model_list)
    S = np.full((N, N), np.nan, dtype=float)
    for i in range(N):
        for j in range(N):
            if j > i:
                S[i, j] = S_reg[i, j]
            elif i > j:
                S[i, j] = S_nn[j, i]  # mirror to fill lower triangle
            else:
                S[i, j] = np.nan

    fig_w = max(10.0, 0.6 * N + 6.0)
    fig_h = max(8.0, 0.6 * N + 5.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Nicer diverging palette for correlations
    cmap = sns.color_palette("viridis", as_cmap=True)

    hm = sns.heatmap(
        S,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        mask=np.isnan(S),
        square=True,
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "pad": 0.02},
    )

    labels = [model_names.get(m, m) for m in model_list]
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=18)
    ax.set_yticklabels(labels, rotation=0, fontsize=18)

    # Cleaner, compact title carrying the upper/lower info
    if title is None:
        title = f"Model similarity (upper=linear, lower=MLP)\nmetric={metric}, method={method}"
    ax.set_title(title, pad=10, fontsize=24)

    # Smaller colorbar label and ticks
    if len(hm.collections) > 0:
        cbar = hm.collections[0].colorbar
        lbl = "Correlation (Spearman r)" if method == "spearman" else "Correlation (Pearson r)"
        cbar.set_label(lbl, rotation=270, labelpad=40, fontsize=30)
        cbar.ax.tick_params(labelsize=20)

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model similarity matrix to {out_path}")

if __name__ == "__main__":
    # add argparser
    parser = argparse.ArgumentParser(description="Plot edge probe heatmaps and summaries.")
    parser.add_argument("--bertology", action="store_true", help="Generate bertology summary plots.")
    parser.add_argument("--similarity", action="store_true", help="Generate model similarity plots.")
    parser.add_argument("--examples", action="store_true", help="Generate example task curves.")
    args = parser.parse_args()

    if args.examples:
        output_dir = "heatmap_figures2"
        all_models = [
            "bert-base-uncased", 
            # "bert-large-uncased", 
            # "deberta-v3-large",
            # "gpt2", "gpt2-large", "gpt2-xl",
            # "gemma2b", "gemma2b-it",
            "qwen2", 
            # "qwen2-instruct",
            # "qwen2.5-7B", "qwen2.5-7B-instruct",
            # "pythia-6.9b",
            # "pythia-6.9b-tulu",
            # "olmo2-7b",
            "olmo2-7b-instruct",
            # "llama3-8b", "llama3-8b-instruct",
        ]
    else:
        output_dir = "heatmap_figures"

    if not args.bertology and not args.similarity:
        for probe_type in ["nn", "reg"]:
            for model in all_models:
                M_acc, _ = _build_matrix_for_model(model, task_list, task_to_dataset, probe=probe_type, metric="accuracy")
                if np.isfinite(M_acc).any():
                    plot_task_layer_heatmaps(
                        task_list, task_to_dataset, [model],
                        probe=probe_type, metric="accuracy",
                        share_scale=True,
                        output_dir=output_dir, fname=f"heatmap_accuracy_{model}_{probe_type}.png",
                        cols_per_row=1
                    )

                M_sel, _ = _build_matrix_for_model(model, task_list, task_to_dataset, probe=probe_type, metric="selectivity")
                if np.isfinite(M_sel).any():
                    plot_task_layer_heatmaps(
                        task_list, task_to_dataset, [model],
                        probe=probe_type, metric="selectivity",
                        share_scale=True,
                        output_dir=output_dir, fname=f"heatmap_selectivity_{model}_{probe_type}.png",
                        cols_per_row=1
                    )

            for metric in ["selectivity", "accuracy"]:
                plot_task_layer_heatmaps(
                    task_list, task_to_dataset, all_models,
                    probe=probe_type, metric=metric,
                    share_scale=True,
                    output_dir=output_dir, fname=f"heatmaps_{metric}_raw_{probe_type}.png",
                    cols_per_row=3, grid=True
                )
    
    if not args.similarity:
        for model in all_models:
            plot_bertology_summary(
                model=model,
                task_list=task_list,
                task_to_dataset=task_to_dataset,
                output_dir=output_dir,
                fname=f"bertology_summary_{model}.png",
            )
        plot_bertology_summary_grid(
            model_list=all_models,
            task_list=task_list,
            task_to_dataset=task_to_dataset,
            output_dir=output_dir,
            fname="grid_bertology_summary.png",
            cols_per_row=3,
        )

    for metric in ["accuracy", "selectivity"]:
        for method in ["spearman", "pearson"]:
            plot_model_similarity_triangular(
                model_list=all_models,
                task_list=task_list,
                task_to_dataset=task_to_dataset,
                metric=metric,
                bins=12,
                method=method,
                output_dir=output_dir,
                fname=f"model_similarity_triangular_{metric}_{method}png",
            )