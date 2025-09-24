import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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
    "olmo2-7b_stage1-step734000-tokens3079B": "OLMo2-7B (734k, 3079B tokens)",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
}

# ---------------------------------------------------------------------
# Your model list and task setup
# ---------------------------------------------------------------------
all_models = [
    "bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
    "gpt2", "gpt2-large", "gpt2-xl",
    "pythia-6.9b",
    "pythia-6.9b-tulu",
    "olmo2-7b", "olmo2-7b-instruct",
    "gemma2b", "gemma2b-it",
    "qwen2", "qwen2-instruct",
    "qwen2.5-7B", "qwen2.5-7B-instruct",
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

def _read_task_curve(dataset, model, probe, task, pca=False, pca_dim=50):
    probe_dir = os.path.join("..", "output", "edge_probes", f"{dataset}_{model}_{probe}")
    if pca:
        probe_dir += f"_pca_{pca_dim}"
    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    try:
        acc_col, ctrl_col = get_acc_columns(df, task)
    except Exception:
        print(f"[WARN] Columns not found for {dataset} {model} {task}")
        return None
    df = df.sort_values("Layer")
    out = {
        "layers": df["Layer"].to_numpy(),
        "acc": df[acc_col].to_numpy(),
        "ctrl": df[ctrl_col].to_numpy() if (ctrl_col is not None and ctrl_col in df.columns) else None,
    }
    return out

def _build_matrix_for_model(
    model, task_list, task_to_dataset, probe="nn",
    metric="accuracy", pca=False, pca_dim=50, normalize="per_task"
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
        y = info["acc"] if metric == "accuracy" else (info["acc"] - (info["ctrl"] if info["ctrl"] is not None else 0.0))
        L = info["layers"].astype(int)
        L = L - L.min()  # align to 0
        for l_idx, val in zip(L, y):
            if 0 <= l_idx < W:
                M[i, l_idx] = val

    # Normalization
    if normalize == "per_task":
        for i in range(H):
            v = M[i, :]
            mask = ~np.isnan(v)
            if mask.sum() >= 2:
                a, b = np.nanmin(v[mask]), np.nanmax(v[mask])
                if b > a:
                    M[i, mask] = (v[mask] - a) / (b - a)
    elif normalize == "global":
        gmask = ~np.isnan(M)
        if gmask.sum() >= 2:
            a, b = np.nanmin(M[gmask]), np.nanmax(M[gmask])
            if b > a:
                M[gmask] = (M[gmask] - a) / (b - a)
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
    normalize="per_task", cmap="Blues", output_dir="figures3",
    fname="heatmaps.png", share_scale=True, cols_per_row=3,
    grid=None,  # None => auto: True if >1 model, False otherwise
):
    if grid is None:
        grid = len(model_list) > 1

    n = len(model_list)
    cols = min(cols_per_row, n) if n > 0 else 1
    rows = int(math.ceil(n / max(cols, 1)))

    # Global limits if needed for raw accuracy with shared scale
    vmin_all, vmax_all = None, None
    if normalize == "none" and share_scale:
        vals = []
        for m in model_list:
            M, _ = _build_matrix_for_model(m, task_list, task_to_dataset, probe, metric, pca, pca_dim, normalize)
            if np.isfinite(M).any():
                vals.append(np.nanmin(M))
                vals.append(np.nanmax(M))
        if vals:
            vmin_all, vmax_all = float(np.nanmin(vals)), float(np.nanmax(vals))

    # Decide vmin/vmax for panels
    def limits():
        # Always lock the color scale to [0, 1] when sharing scale,
        # so single-panel and grid figures are consistent.
        if share_scale:
            return 0.0, 1.0
        return None, None
    
    if not grid:
        model = model_list[0]
        M_plot, _ = _build_matrix_for_model(
            model, task_list, task_to_dataset, probe, metric, pca, pca_dim, normalize
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

        # x labels (density independent of figure width)
        step = _tick_step(Wp, max_ticks=14)
        xticks = np.arange(0.5, Wp + 0.5, step)
        labels = [str(int(x - 0.5)) for x in xticks if int(x - 0.5) < Wp]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("Layer")

        # title and spines
        ax.set_title(model_names.get(model, model), pad=6, fontweight="bold")
        ax.set_aspect("auto")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
            
        if metric == "accuracy":
            hm.collections[0].colorbar.set_label("Accuracy", rotation=270, labelpad=30, fontsize=20)
        elif metric == "selectivity":
            hm.collections[0].colorbar.set_label("Selectivity", rotation=270, labelpad=30, fontsize=20)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved heatmaps to {out_path}")
        return

    # Grid mode: many models per figure, one shared colorbar, sparse ticks
    fig_w_per_col = 6.2
    fig_h_per_row = 0.9 + 0.45 * len(task_list)
    fig_w = fig_w_per_col * cols
    fig_h = fig_h_per_row * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    first_mappable = None
    for idx, model in enumerate(model_list):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        M_plot, _ = _build_matrix_for_model(model, task_list, task_to_dataset, probe, metric, pca, pca_dim, normalize)
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

        # x ticks only on last row
        if r == rows - 1:
            step = _tick_step(Wp, max_ticks=12)
            xticks = np.arange(0.5, Wp + 0.5, step)
            labels = [str(int(x - 0.5)) for x in xticks if int(x - 0.5) < Wp]
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
    fig.text(0.5, 0.02, "Layer", ha="center", va="center", fontsize=24)

    # Reserve space on the right and add a figure-level colorbar in a fixed spot
    right_margin = 0.92
    plt.tight_layout(rect=[0.03, 0.04, right_margin, 0.98])

    if first_mappable is not None:
        cax = fig.add_axes([right_margin + 0.01, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(first_mappable, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        if metric == "accuracy":
            cbar.set_label("Accuracy", rotation=270, labelpad=25, fontsize=20)
        elif metric == "selectivity":
            cbar.set_label("Selectivity", rotation=270, labelpad=25, fontsize=20)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved heatmaps to {out_path}")

if __name__ == "__main__":
    output_dir = "figures3"

    for probe_type in ["nn", "reg"]:
        for model in all_models:
            M_acc, _ = _build_matrix_for_model(model, task_list, task_to_dataset, probe=probe_type,
                                               metric="accuracy", normalize="per_task")
            if np.isfinite(M_acc).any():
                plot_task_layer_heatmaps(
                    task_list, task_to_dataset, [model],
                    probe=probe_type, metric="accuracy",
                    normalize="per_task", share_scale=True,
                    output_dir=output_dir, fname=f"heatmap_accuracy_{model}_{probe_type}.png",
                    cols_per_row=1
                )

            M_sel, _ = _build_matrix_for_model(model, task_list, task_to_dataset, probe=probe_type,
                                               metric="selectivity", normalize="per_task")
            if np.isfinite(M_sel).any():
                plot_task_layer_heatmaps(
                    task_list, task_to_dataset, [model],
                    probe=probe_type, metric="selectivity",
                    normalize="per_task", share_scale=True,
                    output_dir=output_dir, fname=f"heatmap_selectivity_{model}_{probe_type}.png",
                    cols_per_row=1
                )

        plot_task_layer_heatmaps(
            task_list, task_to_dataset, all_models,
            probe=probe_type, metric="accuracy",
            normalize="none", share_scale=True,
            output_dir=output_dir, fname=f"heatmaps_accuracy_raw_{probe_type}.png",
            cols_per_row=3, grid=True
        )
