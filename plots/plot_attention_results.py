import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors

from plot_multilingual_results import (
    model_names,
    get_acc_columns, MODEL_COLORS
)

plt.rcParams.update({'font.family': 'serif'})

if not isinstance(MODEL_COLORS, dict):
    MODEL_COLORS = {}


def _normalize_layer(df: pd.DataFrame) -> pd.Series:
    """
    Normalize Layer indices into [0, 1] for cross-model averaging.
    """
    if "Layer_Normalized" in df.columns:
        return df["Layer_Normalized"]
    if "Layer" not in df.columns:
        raise ValueError("Missing 'Layer' column for normalization.")
    layer = df["Layer"].astype(float)
    denom = (layer.max() - layer.min())
    if denom == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (layer - layer.min()) / denom


def _load_attention_curve(csv_path: str, task: str, plot_selectivity: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_normalized, y_values) for one result csv.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("Layer")
    acc_col, ctrl_col = get_acc_columns(df, task)
    x = _normalize_layer(df).to_numpy()
    if plot_selectivity:
        y = (df[acc_col] - df[ctrl_col]).to_numpy()
    else:
        y = df[acc_col].to_numpy()
    return x, y


def plot_condensed_attention_summary(
    model_to_dataset,
    model_list,
    output_dir="figures3",
    filename="attention_condensed_singlecol.png",
    probe_types=("reg", "mlp"),
    metric="selectivity",
    aggregate_over_probes=True,
    shading=False,
):
    """
    Condensed, single-column attention summary for the paper.

    Produces a 2x1 vertical figure:
    - Top: Attention Output
    - Bottom: Residual Stream
    Each axis plots averaged curves for Lemma vs Inflection.

    Averaging recipe (mirrors grouped classifier plots):
    - For each (model, task, source), load per-layer curve(s) for the requested probe_types
    - Interpolate onto a common normalized layer grid common_x
    - Optionally average across probe_types within each model (aggregate_over_probes=True)
    - Average across models (and optionally shade min–max envelope)
    """
    if metric not in {"selectivity", "accuracy"}:
        raise ValueError(f"metric must be 'selectivity' or 'accuracy', got: {metric}")

    tasks = ["lexeme", "inflection"]
    plot_selectivity = (metric == "selectivity")
    sources = [
        (True, "Attention Output"),
        (False, "Residual Stream"),
    ]

    common_x = np.linspace(0, 1, 101)

    # Precompute file availability once.
    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe in probe_types:
                for attn in [True, False]:
                    key = (task, probe, attn)
                    if probe == "mlp":
                        csv_path = find_csv_file_probe(model_key, dataset, task, "mlp", attn=attn)
                        if not csv_path:
                            csv_path = find_csv_file_probe(model_key, dataset, task, "nn", attn=attn)
                    else:
                        csv_path = find_csv_file_probe(model_key, dataset, task, probe, attn=attn)
                    file_availability[model_key][key] = csv_path

    # Single-column friendly sizes.
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 1, figsize=(4.2, 6.2), sharex=True, constrained_layout=True)

    task_color = {
        "lexeme": "tab:blue",
        "inflection": "tab:orange",
    }
    task_label = {
        "lexeme": "Lemma",
        "inflection": "Inflection",
    }

    for ax_idx, (attn, source_title) in enumerate(sources):
        ax = axes[ax_idx]

        for task in tasks:
            ys = []
            for model_key in model_list:
                per_probe = []
                for probe in probe_types:
                    csv_path = file_availability[model_key].get((task, probe, attn))
                    if not csv_path:
                        continue
                    try:
                        x, y = _load_attention_curve(csv_path, task=task, plot_selectivity=plot_selectivity)
                        # Guard: np.interp needs sorted x.
                        order = np.argsort(x)
                        y_interp = np.interp(common_x, x[order], y[order])
                        per_probe.append(y_interp)
                    except Exception:
                        continue

                if not per_probe:
                    continue

                if aggregate_over_probes and len(per_probe) > 1:
                    ys.append(np.mean(per_probe, axis=0))
                else:
                    ys.extend(per_probe)

            if ys:
                ys = np.asarray(ys)
                avg_y = np.mean(ys, axis=0)
                min_y = np.min(ys, axis=0)
                max_y = np.max(ys, axis=0)
                ax.plot(
                    common_x,
                    avg_y,
                    linewidth=3.0,
                    color=task_color[task],
                    label=task_label[task],
                )
                if shading:
                    ax.fill_between(common_x, min_y, max_y, color=task_color[task], alpha=0.18)

        ax.set_title(source_title, pad=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.grid(True, linestyle="--", alpha=0.35)

        if plot_selectivity:
            ax.set_ylim(0, 0.8)
            ax.set_yticks(np.arange(0, 0.81, 0.2))
            ax.set_ylabel("Selectivity")
        else:
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_ylabel("Accuracy")

    axes[-1].set_xlabel("Normalized layer number (%)")
    # Single legend for both panels
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved condensed attention summary to {out_path}")


def plot_attention_results_averaged(
    model_to_dataset,
    model_list,
    output_dir="figures3",
    filename="attention_averaged_combined.png",
    shading=False,
):
    """
    Averaged version of `plot_attention_results` (same 2x4 layout), but with curves
    averaged across all models (instead of plotting each model).

    Each panel shows:
    - Solid: mean over models for Attention Output
    - Solid: mean over models for Residual Stream (distinguished by color)
    Optional min–max shading over models can be enabled with `shading=True`.
    """
    probe_types = ["reg", "mlp"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types) * 2

    aspect_ratio, base_height = 3.0, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = n_rows * base_height
    fig_size = (fig_width, fig_height)

    common_x = np.linspace(0, 1, 101)

    # Precompute file availability once.
    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe in probe_types:
                for attn in [True, False]:
                    key = (task, probe, attn)
                    if probe == "mlp":
                        csv_path = find_csv_file_probe(model_key, dataset, task, "mlp", attn=attn)
                        if not csv_path:
                            csv_path = find_csv_file_probe(model_key, dataset, task, "nn", attn=attn)
                    else:
                        csv_path = find_csv_file_probe(model_key, dataset, task, probe, attn=attn)
                    file_availability[model_key][key] = csv_path

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
    axes = np.atleast_2d(axes)

    def _collect_ys(task: str, probe: str, attn: bool, plot_selectivity: bool) -> np.ndarray:
        ys = []
        for model_key in model_list:
            csv_path = file_availability[model_key].get((task, probe, attn))
            if not csv_path:
                continue
            try:
                x, y = _load_attention_curve(csv_path, task=task, plot_selectivity=plot_selectivity)
                order = np.argsort(x)
                y_interp = np.interp(common_x, x[order], y[order])
                ys.append(y_interp)
            except Exception:
                continue
        return np.asarray(ys) if ys else np.asarray([])

    for row, task in enumerate(tasks):
        for col in range(n_cols):
            if col < 2:
                probe = probe_types[col]
                plot_selectivity = False
            else:
                probe = probe_types[col - 2]
                plot_selectivity = True

            ax = axes[row, col]

            if row == 0:
                title_idx = col % 2
                ax.set_title(titles[title_idx], pad=15, fontsize=24)

            for attn, alpha, zorder in [
                (True, 1.0, 2),
                (False, 0.7, 1),
            ]:
                ys = _collect_ys(task=task, probe=probe, attn=attn, plot_selectivity=plot_selectivity)
                if ys.size == 0:
                    continue
                avg_y = np.mean(ys, axis=0)
                min_y = np.min(ys, axis=0)
                max_y = np.max(ys, axis=0)

                ax.plot(
                    common_x,
                    avg_y,
                    linewidth=3.0 if attn else 2.0,
                    color="black",
                    linestyle="-",
                    alpha=alpha,
                    zorder=zorder,
                )
                if shading:
                    ax.fill_between(common_x, min_y, max_y, color="black", alpha=0.10 if attn else 0.06)

            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "25", "50", "75", "100"])
            ax.tick_params(axis='both', which='major', length=10, width=2)

            if plot_selectivity:
                ax.set_ylim(0, 0.8)
                ax.set_yticks(np.arange(0, 0.81, 0.2))
            else:
                ax.set_ylim(0, 1.0)
                ax.set_yticks(np.arange(0, 1.01, 0.2))

            if col == 0 or col == 2:
                task_label = "Lemma" if task == "lexeme" else task.title()
                ylabel = f"{task_label} {'Selectivity' if plot_selectivity else 'Accuracy'}"
                ax.set_ylabel(ylabel, labelpad=15, fontsize=24)
                ax.yaxis.set_tick_params(labelleft=True)
            else:
                ax.set_ylabel("")
                ax.yaxis.set_tick_params(labelleft=False)

            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

            if row == 1:
                pass
            else:
                ax.set_xticklabels([])

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 2])

    fig.text(0.25, -0.03, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])
    fig.text(0.75, -0.03, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])

    # Legend: just source (solid vs dashed), no model legend.
    from matplotlib.lines import Line2D
    source_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=3.0, label="Attention Output"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, alpha=0.7, label="Residual Stream"),
    ]
    fig.legend(
        source_handles, [h.get_label() for h in source_handles],
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        ncol=2, frameon=True, fontsize=24, title="", title_fontsize=24
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved averaged attention combined figure to {out_path}")


def plot_attention_accuracy_two_panel_averaged(
    model_to_dataset,
    model_list,
    output_dir="figures3",
    filename="attention_accuracy_two_panel_averaged.png",
    probe_types=("reg", "mlp"),
    shading=False,
):
    tasks = [("lexeme", "Lemma"), ("inflection", "Inflection")]
    common_x = np.linspace(0, 1, 101)

    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task, _ in tasks:
            for probe in probe_types:
                for attn in [True, False]:
                    key = (task, probe, attn)
                    if probe == "mlp":
                        csv_path = find_csv_file_probe(model_key, dataset, task, "mlp", attn=attn)
                        if not csv_path:
                            csv_path = find_csv_file_probe(model_key, dataset, task, "nn", attn=attn)
                    else:
                        csv_path = find_csv_file_probe(model_key, dataset, task, probe, attn=attn)
                    file_availability[model_key][key] = csv_path

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.0,
    })

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    line_specs = [
        (True, "Attention Output", "tab:blue", "-", 3.0, 1.0),
        (False, "Residual Stream", "tab:orange", "-", 2.8, 0.95),
    ]

    def _avg_over_models(task: str, attn: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Returns (avg, min, max) over models, after averaging probe types within each model.
        """
        ys = []
        for model_key in model_list:
            per_probe = []
            for probe in probe_types:
                csv_path = file_availability[model_key].get((task, probe, attn))
                if not csv_path:
                    continue
                try:
                    x, y = _load_attention_curve(csv_path, task=task, plot_selectivity=False)
                    order = np.argsort(x)
                    y_interp = np.interp(common_x, x[order], y[order])
                    per_probe.append(y_interp)
                except Exception:
                    continue
            if not per_probe:
                continue
            ys.append(np.mean(per_probe, axis=0))  # average across probe types within-model

        if not ys:
            return None

        ys = np.asarray(ys)
        return np.mean(ys, axis=0), np.min(ys, axis=0), np.max(ys, axis=0)

    for ax, (task, title) in zip(axes, tasks):
        for attn, label, color, linestyle, lw, alpha in line_specs:
            stats = _avg_over_models(task=task, attn=attn)
            if stats is None:
                continue
            avg_y, min_y, max_y = stats
            ax.plot(
                common_x,
                avg_y,
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=alpha,
                label=label,
            )
            if shading:
                ax.fill_between(common_x, min_y, max_y, color=color, alpha=0.15)

        ax.set_title(title, pad=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis='both', which='major', length=8, width=1.6)
        ax.set_xticklabels(["0", "25", "50", "75", "100"])

    for ax in axes:
        ax.set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    axes[-1].set_xlabel("Normalized layer number (%)")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved 2-panel averaged attention accuracy figure to {out_path}")

def find_csv_file_probe(model_key, dataset, task, probe_type, attn=True):
    base_model = model_key.split('_')[0]
    probe_name = probe_type
    suffix = "_attn" if attn else ""
    probe_dirs = [
        os.path.join("..", "output", "probes", f"{dataset}_{base_model}_{task}_{probe_name}{suffix}"),
        os.path.join("..", "output", "probes2", f"{dataset}_{base_model}_{task}_{probe_name}{suffix}")
    ]
    for probe_dir in probe_dirs:
        csv_path = os.path.join(probe_dir, f"{task}_results.csv")
        if os.path.exists(csv_path):
            return csv_path
    return None

def add_legends(fig, model_list, title_model="Model", title_source="Source"):
    from matplotlib.lines import Line2D

    source_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=3.0, label="Attention Output"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, alpha=0.7, label="Residual Stream"),
    ]

    model_handles = []
    model_labels_and_base_models = [
        ("GPT-2-Small", "gpt2"),
        ("GPT-2-Large", "gpt2-large"),
        ("GPT-2-XL", "gpt2-xl"),
        ("Qwen2.5-1.5B", "qwen2"),
        ("Qwen2.5-1.5B-Instruct", "qwen2-instruct"),
        ("Qwen2.5-7B", "qwen2.5-7B"),
        ("Qwen2.5-7B-Instruct", "qwen2.5-7B-instruct"),
        ("Pythia-1.4B", "pythia1.4b"),
        ("Gemma-2-2B", "gemma2b"),
        ("Gemma-2-2B-Instruct", "gemma2b-it"),
        ("BERT-Base-Uncased", "bert-base-uncased"),
        ("BERT-Large-Uncased", "bert-large-uncased"),
        ("DeBERTa-v3-Large", "deberta-v3-large"),
        ("Llama-3-8B", "llama3-8b"),
        ("Llama-3-8B-Instruct", "llama3-8b-instruct"),
        ("Pythia-6.9B", "pythia-6.9b"),
        ("Pythia-6.9B-Tulu", "pythia-6.9b-tulu"),
        ("OLMo-2-1124-7B-Instruct", "olmo2-7b-instruct"),
        ("OLMo-2-1124-7B", "olmo2-7b"),
    ]

    filtered_model_labels_and_base_models = [
        (label, base_model) for label, base_model in model_labels_and_base_models if base_model in model_list or any(model.startswith(base_model) for model in model_list)
    ]
    for label, base_model in filtered_model_labels_and_base_models:
        model_handles.append(Line2D([0], [0], color=MODEL_COLORS.get(base_model, "gray"), linestyle="-", linewidth=4.0, label=label))
        
    model_bbox_anchor = (0.5, -0.325)
    ncol = 5

    if any(model.startswith("bert") for model in model_list):
        model_bbox_anchor = (0.5, -0.275)
        ncol = 3

    fig.legend(
        source_handles, [h.get_label() for h in source_handles],
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        ncol=2, frameon=True, fontsize=24, title="", title_fontsize=24
    )

    fig.legend(
        model_handles, [h.get_label() for h in model_handles],
        loc="lower center", bbox_to_anchor=model_bbox_anchor,
        ncol=ncol, frameon=True, fontsize=24, title="", title_fontsize=24
    )


def plot_attention_results(model_to_dataset, model_list, output_dir="figures3", filename_prefix="attention_"):
    probe_types = ["reg", "mlp"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types) * 2
    
    aspect_ratio, base_height = 3.0, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = n_rows * base_height
    fig_size = (fig_width, fig_height)

    file_availability = {}
    missing_files = []
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe in probe_types:
                for attn in [True, False]:
                    key = (task, probe, attn)
                    if probe == "mlp":
                        csv_path = find_csv_file_probe(model_key, dataset, task, "mlp", attn=attn)
                        if not csv_path:
                            csv_path = find_csv_file_probe(model_key, dataset, task, "nn", attn=attn)
                    else:
                        csv_path = find_csv_file_probe(model_key, dataset, task, probe, attn=attn)
                    file_availability[model_key][key] = csv_path
                    if csv_path is None:
                        missing_files.append((model_key, task, probe, attn))
    if missing_files:
        print(f"[INFO] Missing {len(missing_files)} probe result files (will skip in plots)")

    def plot_panel(axes, model_list_subset):
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                if col < 2:
                    probe = probe_types[col]
                    plot_selectivity = False
                else:
                    probe = probe_types[col - 2]
                    plot_selectivity = True
                
                ax = axes[row, col]
                
                if row == 0:
                    title_idx = col % 2
                    ax.set_title(titles[title_idx], pad=15, fontsize=24)

                for model_key in model_list_subset:
                    base_model_name = model_key.split('_')[0]

                    for attn, label_suffix, alpha, zorder in [
                        (True, " (Attention Output)", 1.0, 2),
                        (False, " (Residual Stream)", 0.7, 1)
                    ]:
                        csv_path = file_availability[model_key].get((task, probe, attn))
                        if not csv_path:
                            continue
                        try:
                            df = pd.read_csv(csv_path)
                            acc_col, ctrl_col = get_acc_columns(df, task)
                            df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                            y = df[acc_col] - df[ctrl_col] if plot_selectivity else df[acc_col]

                            line_color = MODEL_COLORS.get(base_model_name, "black")
                            
                            ax.plot(
                                df["Layer_Normalized"], y,
                                label=None, 
                                linewidth=3.0 if attn else 2.0,
                                color=line_color,
                                linestyle="-" if attn else "--",
                                alpha=alpha,
                                zorder=zorder
                            )
                        except Exception as e:
                            # print(f"[WARN] Error processing {model_key} {task} {probe} ({'attn' if attn else 'residual'}): {e}")
                            continue
                
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0", "25", "50", "75", "100"])
                ax.tick_params(axis='both', which='major', length=10, width=2)
                
                if plot_selectivity:
                    ax.set_ylim(0, 0.8)
                    ax.set_yticks(np.arange(0, 0.81, 0.2))
                else:
                    ax.set_ylim(0, 1.0)
                    ax.set_yticks(np.arange(0, 1.01, 0.2))
                if col == 0 or col == 2:
                    task_label = "Lemma" if task == "lexeme" else task.title()
                    ylabel = f"{task_label} {'Selectivity' if plot_selectivity else 'Accuracy'}"
                    ax.set_ylabel(ylabel, labelpad=15, fontsize=24)
                    ax.yaxis.set_tick_params(labelleft=True)
                else:
                    ax.set_ylabel("")
                    ax.yaxis.set_tick_params(labelleft=False)
                
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                
                if row == 1:
                    pass # Labels added globally
                else:
                    ax.set_xticklabels([])

    bert_models = [m for m in model_list if "bert" in m or "gpt" in m]
    other_models = [m for m in model_list if "bert" not in m and "gpt" not in m]

    model_groups = {
        "BERT": bert_models,
        "Other": other_models
    }

    for model_type, models in model_groups.items():
        if not models:
            print(f"Skipping {model_type} models (no models found)")
            continue

        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
        axes = np.atleast_2d(axes)
        plot_panel(axes, models)
        
        fig.align_ylabels(axes[:, 0])
        fig.align_ylabels(axes[:, 2])
        
        # Add shared x-axis labels
        fig.text(0.25, -0.03, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])
        fig.text(0.75, -0.03, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])

        add_legends(fig, models)
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{filename_prefix}{model_type.lower()}_combined.png"
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        print(f"Saved {model_type} attention combined figure to {os.path.join(output_dir, filename)}")


def generate_attention_markdown_tables(model_to_dataset, model_list, output_dir="figures3"):
    percentages = [0, 25, 50, 75, 100]
    probe_types = ["reg", "mlp"]
    tasks = ["lexeme", "inflection"]
    probe_names = {
        "reg": "Linear Regression",
        "mlp": "MLP"
    }
    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe_type in probe_types:
                for attn in [True, False]:
                    key = (task, probe_type, attn)
                    if probe_type == "mlp":
                        csv_path = find_csv_file_probe(model_key, dataset, task, "mlp", attn=attn)
                        if not csv_path:
                            csv_path = find_csv_file_probe(model_key, dataset, task, "nn", attn=attn)
                    else:
                        csv_path = find_csv_file_probe(model_key, dataset, task, probe_type, attn=attn)
                    file_availability[model_key][key] = csv_path
    
    model_families = {"Attention Models": model_list}  # Updated title

    for task in tasks:
        for probe_type in probe_types:
            probe_name = probe_names[probe_type]
            print(f"\n## {task.title()} Accuracy - {probe_name} (Attention & Residual)\n")  # Updated title
            family_models = model_families["Attention Models"]  # Updated key

            valid_models = []
            layer_counts = []
            for model_key in family_models:
                for attn in [True, False]:
                    csv_path = file_availability[model_key].get((task, probe_type, attn))
                    if csv_path:
                        try:
                            df = pd.read_csv(csv_path)
                            valid_models.append((model_key, csv_path, attn))
                            layer_counts.append(len(df))
                        except:
                            continue
            if not valid_models:
                print("No valid results found for this probe type.\n")
                continue

            from collections import Counter
            most_common_layers = Counter(layer_counts).most_common(1)[0][0]
            layer_indices = [int(most_common_layers * p / 100) for p in percentages]
            layer_indices[-1] = most_common_layers - 1

            headers = []
            for i, (layer_idx, pct) in enumerate(zip(layer_indices, percentages)):
                if pct == 0:
                    headers.append(f"Layer {layer_idx} (first)")
                elif pct == 100:
                    headers.append(f"Layer {layer_idx} (last)")
                else:
                    headers.append(f"Layer {layer_idx}")
            
            print(f"### Attention Models\n")
            header_row = "| Model | Dataset | Source | " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join([" --- "] * (len(headers) + 3)) + "|"
            print(header_row)
            print(separator_row)

            def attention_sort_key(model_key):
                base_model = model_key.split('_')[0]
                size_priority = 0 if base_model in ['qwen2', 'qwen2-instruct'] else 1
                type_priority = 0 if 'instruct' not in base_model else 1
                return (size_priority, type_priority, base_model)
            
            sorted_models = sorted(family_models, key=attention_sort_key)

            for model_key in sorted_models:
                dataset = model_to_dataset[model_key]
                language = model_names.get(model_key, model_key)
                for attn, src_label in [(True, "Attention Output"), (False, "Residual Stream")]:
                    csv_path = file_availability[model_key].get((task, probe_type, attn))
                    if csv_path is None:
                        row_data = ["N/A"] * len(percentages)
                    else:
                        try:
                            df = pd.read_csv(csv_path)
                            acc_col, _ = get_acc_columns(df, task)
                            actual_n_layers = len(df)
                            actual_layer_indices = [int(actual_n_layers * p / 100) for p in percentages]
                            actual_layer_indices[-1] = actual_n_layers - 1
                            row_data = []
                            for layer_idx in actual_layer_indices:
                                if 0 <= layer_idx < len(df):
                                    accuracy = df.iloc[layer_idx][acc_col]
                                    row_data.append(f"{accuracy:.3f}")
                                else:
                                    row_data.append("N/A")
                        except Exception as e:
                            row_data = ["N/A"] * len(percentages)
                    row = f"| {language} | {dataset} | {src_label} | " + " | ".join(row_data) + " |"
                    print(row)
            print()

if __name__ == "__main__":
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
        "bert-base-uncased": "BERT-Base-Uncased",
        "bert-large-uncased": "BERT-Large-Uncased",
        "deberta-v3-large": "DeBERTa-v3-Large",
        "llama3-8b": "Llama-3-8B",
        "llama3-8b-instruct": "Llama-3-8B-Instruct",
        "pythia-6.9b": "Pythia-6.9B",
        "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
        "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
        "olmo2-7b": "OLMo-2-1124-7B",
    }

    models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "deberta-v3-large",
        "gpt2",
        "gpt2-large",
        "gpt2-xl",
        "qwen2",
        "qwen2-instruct",
        "qwen2.5-7B",
        "qwen2.5-7B-instruct",
        "gemma2b",
        "gemma2b-it",
        "llama3-8b",
        "llama3-8b-instruct",
        "pythia-6.9b",
        "pythia-6.9b-tulu",
        "olmo2-7b-instruct",
        "olmo2-7b"
    ]
    
    attention_models = models
    
    attention_datasets = [
        ("ud_gum_dataset", "English"),
    ]
    
    attention_all_models = []
    attention_model_to_dataset = {}
    
    temp_model_colors = {}
    temp_model_colors["gpt2"] = "tab:brown"
    temp_model_colors["gpt2-large"] = "tab:orange"
    temp_model_colors["gpt2-xl"] = "tab:red"
    temp_model_colors["qwen2"] = "tab:blue"  
    temp_model_colors["qwen2-instruct"] = "tab:cyan"
    temp_model_colors["qwen2.5-7B"] = "mediumseagreen"
    temp_model_colors["qwen2.5-7B-instruct"] = "springgreen"
    temp_model_colors["gemma2b"] = "darkviolet"
    temp_model_colors["gemma2b-it"] = "deeppink"
    temp_model_colors["bert-base-uncased"] = "steelblue"
    temp_model_colors["bert-large-uncased"] = "navy"
    temp_model_colors["deberta-v3-large"] = "darkkhaki"
    temp_model_colors["llama3-8b"] = "lightcoral"
    temp_model_colors["llama3-8b-instruct"] = "rosybrown"
    temp_model_colors["pythia-6.9b"] = "darkgoldenrod"
    temp_model_colors["pythia-6.9b-tulu"] = "lightsalmon"
    temp_model_colors["olmo2-7b-instruct"] = "palegreen"
    temp_model_colors["olmo2-7b"] = "forestgreen"
    MODEL_COLORS.update(temp_model_colors)

    for model in attention_models:
        for dataset, lang in attention_datasets:
            model_key = f"{model}_{lang.lower()}"
            attention_all_models.append(model_key)
            attention_model_to_dataset[model_key] = dataset
            model_names[model_key] = f"{model_names.get(model, model).replace('qwen2', 'Qwen2.5-1.5B').replace('instruct', 'Instruct')} ({lang})"

    print("Generating plots for attention experiments...")
    plot_attention_results(attention_model_to_dataset, attention_all_models)

    print("Generating 2-panel averaged attention accuracy plot (averaged across probe types)...")
    plot_attention_accuracy_two_panel_averaged(
        attention_model_to_dataset,
        attention_all_models,
        output_dir="figures3",
        filename="attention_accuracy_two_panel_averaged.png",
        shading=True,
    )
    
    # print("\nGenerating markdown tables for attention experiments...")
    # generate_attention_markdown_tables(attention_model_to_dataset, attention_all_models)
