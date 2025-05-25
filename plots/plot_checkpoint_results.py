import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_accuracy_data(model_name, task, probe, dataset, pca=False, pca_dim=50):
    probe_dir = os.path.join("..", "output", "probes", f"{dataset}_{model_name}_{task}_{probe}")
    if pca:
        probe_dir += f"_pca_{pca_dim}"
    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Add normalized layer number as in plot_classifier_results.py
        df["Layer_Normalized"] = (
            df["Layer"] - df["Layer"].min()
        ) / (df["Layer"].max() - df["Layer"].min())
        return df
    else:
        print(f"[WARN] Missing results for {model_name} {task} {probe}")
        return None

pythia_models = [
    "pythia-6.9b_step1",
    "pythia-6.9b_step64",
    "pythia-6.9b_step6000",
    "pythia-6.9b_step19000",
    "pythia-6.9b_step37000",
    "pythia-6.9b_step57000",
    "pythia-6.9b_step82000",
    "pythia-6.9b_step111000",
    "pythia-6.9b",
]
olmo_models = [
    "olmo2-7b_stage1-step5000-tokens21B",
    "olmo2-7b_stage1-step40000-tokens168B",
    "olmo2-7b_stage1-step97000-tokens407B",
    "olmo2-7b_stage1-step179000-tokens751B",
    "olmo2-7b_stage1-step282000-tokens1183B",
    "olmo2-7b_stage1-step409000-tokens1716B",
    "olmo2-7b_stage1-step559000-tokens2345B",
    "olmo2-7b_stage1-step734000-tokens3079B",
    "olmo2-7b",
]
pythia_labels = ["step1", "step64", "step6k", "step19k", "step37k", "step57k", "step82k", "step111k", "final"]
olmo_labels = ["5k (21B)", "40k (168B)", "97k (407B)", "179k (751B)", "282k (1183B)", "409k (1716B)", "559k (2345B)", "734k (3079B)", "final"]

# Instead of heatmaps, plot a line for each checkpoint: x=normalized layer, y=accuracy, color=training progress.
# Use a color gradient to show training progress for each model.
from matplotlib.cm import get_cmap

dataset = "ud_gum_dataset"
probe_types = ["nn", "reg"]
tasks = ["lexeme", "inflection"]
colors = {"pythia": "tab:blue", "olmo": "tab:orange"}

# Use matplotlib.colormaps and viridis for both models
import matplotlib

fig, axes = plt.subplots(len(tasks), len(probe_types), figsize=(18, 10), sharex=True, sharey=True)
if len(tasks) == 1 and len(probe_types) == 1:
    axes = np.array([[axes]])
elif len(tasks) == 1 or len(probe_types) == 1:
    axes = axes.reshape(len(tasks), len(probe_types))

for col, probe_type in enumerate(probe_types):
    for row, task in enumerate(tasks):
        ax = axes[row, col]
        # Use viridis colormap for both models
        cmap = matplotlib.colormaps["viridis"]
        n_pythia = len(pythia_models)
        n_olmo = len(olmo_models)

        # Pythia: plot each checkpoint as a line, color by training progress
        for i, model in enumerate(pythia_models):
            df = load_accuracy_data(model, task, probe_type, dataset)
            if df is not None:
                color = cmap(0.1 + 0.8 * i / (n_pythia - 1))
                label = pythia_labels[i] if row == 0 and col == 0 and i in [0, n_pythia-1] else None
                ax.plot(
                    df["Layer_Normalized"] * 100,
                    df["Acc"],
                    color=color,
                    linewidth=2,
                    label=label,
                    alpha=0.9 if i in [0, n_pythia-1] else 0.5,
                    zorder=2
                )
        # OLMo: plot each checkpoint as a line, color by training progress (dashed)
        for i, model in enumerate(olmo_models):
            df = load_accuracy_data(model, task, probe_type, dataset)
            if df is not None:
                color = cmap(0.1 + 0.8 * i / (n_olmo - 1))
                label = olmo_labels[i] if row == 1 and col == 1 and i in [0, n_olmo-1] else None
                ax.plot(
                    df["Layer_Normalized"] * 100,
                    df["Acc"],
                    color=color,
                    linewidth=2,
                    linestyle="--",
                    label=label,
                    alpha=0.9 if i in [0, n_olmo-1] else 0.5,
                    zorder=2
                )

        # Annotate start and end
        if row == 0 and col == 0:
            ax.text(2, 0.98, "Pythia: Early", color=cmap(0.1), fontsize=14, va="top")
            ax.text(98, 0.98, "Pythia: Final", color=cmap(0.9), fontsize=14, va="top", ha="right")
        if row == 1 and col == 1:
            ax.text(2, 0.98, "OLMo: Early", color=cmap(0.1), fontsize=14, va="top")
            ax.text(98, 0.98, "OLMo: Final", color=cmap(0.9), fontsize=14, va="top", ha="right")

        ax.set_xlim(0, 100)
        ax.set_xlabel("Normalized layer number (%)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task.capitalize()} ({probe_type})")
        ax.grid(True, linestyle='--', alpha=0.5)
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=12, title="Pythia checkpoints")
        if row == 1 and col == 1:
            ax.legend(loc="upper right", fontsize=12, title="OLMo checkpoints")

plt.tight_layout()
save_dir = os.path.join("figures3")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"checkpoint_lineplot_layers_vs_checkpoints_gradient.png")
plt.savefig(save_path, bbox_inches="tight", dpi=200)
print(f"Saved lineplot (gradient) plot as {save_path}")
plt.close()

def get_median_acc_table(models, labels, task, probe_type, dataset):
    """
    Returns a DataFrame: rows=normalized layer bins, columns=checkpoints, values=median accuracy.
    """
    # Bin normalized layers into e.g. 10 bins (0-10%, 10-20%, ..., 90-100%)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{int(left*100)}-{int(right*100)}%" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    table = pd.DataFrame(index=bin_labels, columns=labels)
    for i, model in enumerate(models):
        df = load_accuracy_data(model, task, probe_type, dataset)
        if df is not None and len(df) > 0:
            df["LayerBin"] = pd.cut(df["Layer_Normalized"], bins=bin_edges, labels=bin_labels, include_lowest=True)
            medians = df.groupby("LayerBin")["Acc"].median()
            for bin_label in bin_labels:
                table.at[bin_label, labels[i]] = medians.get(bin_label, np.nan)
        else:
            table[labels[i]] = np.nan
    return table

# Example usage for both models, both tasks, both probe types:
dataset = "ud_gum_dataset"
probe_types = ["nn", "reg"]
tasks = ["lexeme", "inflection"]

for probe_type in probe_types:
    for task in tasks:
        print(f"\n=== Median Accuracy Table: {task.capitalize()} ({probe_type}) - Pythia ===")
        table = get_median_acc_table(pythia_models, pythia_labels, task, probe_type, dataset)
        print(table.round(3).to_string())
        print(f"\n=== Median Accuracy Table: {task.capitalize()} ({probe_type}) - OLMo ===")
        table = get_median_acc_table(olmo_models, olmo_labels, task, probe_type, dataset)
        print(table.round(3).to_string())

def get_median_acc_table_all(models_dict, labels_dict, tasks, probe_types, dataset):
    """
    Returns a MultiIndex DataFrame: 
    rows = normalized layer bins,
    columns = (Model, Checkpoint, Task, ProbeType)
    """
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{int(left*100)}-{int(right*100)}%" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    columns = []
    data = {}

    for model_key, models in models_dict.items():
        labels = labels_dict[model_key]
        for probe_type in probe_types:
            for task in tasks:
                for i, model in enumerate(models):
                    col = (model_key, labels[i], task, probe_type)
                    columns.append(col)
                    # Compute median accuracy per bin
                    df = load_accuracy_data(model, task, probe_type, dataset)
                    if df is not None and len(df) > 0:
                        df["LayerBin"] = pd.cut(df["Layer_Normalized"], bins=bin_edges, labels=bin_labels, include_lowest=True)
                        medians = df.groupby("LayerBin")["Acc"].median()
                        data[col] = [medians.get(bin_label, np.nan) for bin_label in bin_labels]
                    else:
                        data[col] = [np.nan] * len(bin_labels)
    # Build DataFrame
    columns = pd.MultiIndex.from_tuples(columns, names=["Model", "Checkpoint", "Task", "ProbeType"])
    table = pd.DataFrame(data, index=bin_labels, columns=columns)
    return table

# --- Only one figure, one table ---
dataset = "ud_gum_dataset"
probe_types = ["nn", "reg"]
tasks = ["lexeme", "inflection"]

models_dict = {
    "Pythia": pythia_models,
    "OLMo": olmo_models,
}
labels_dict = {
    "Pythia": pythia_labels,
    "OLMo": olmo_labels,
}

# Plot: (keep the multiplot for all tasks/probes in one figure)
fig, axes = plt.subplots(len(tasks), len(probe_types), figsize=(18, 10), sharex=True, sharey=True)
if len(tasks) == 1 and len(probe_types) == 1:
    axes = np.array([[axes]])
elif len(tasks) == 1 or len(probe_types) == 1:
    axes = axes.reshape(len(tasks), len(probe_types))

import matplotlib

for col, probe_type in enumerate(probe_types):
    for row, task in enumerate(tasks):
        ax = axes[row, col]
        cmap = matplotlib.colormaps["viridis"]
        for model_key, models in models_dict.items():
            labels = labels_dict[model_key]
            n_models = len(models)
            for i, model in enumerate(models):
                df = load_accuracy_data(model, task, probe_type, dataset)
                if df is not None:
                    color = cmap(0.1 + 0.8 * i / (n_models - 1))
                    linestyle = "-" if model_key == "Pythia" else "--"
                    label = f"{model_key} {labels[i]}" if (i == 0 or i == n_models-1) else None
                    ax.plot(
                        df["Layer_Normalized"] * 100,
                        df["Acc"],
                        color=color,
                        linewidth=2,
                        linestyle=linestyle,
                        label=label,
                        alpha=0.9 if (i == 0 or i == n_models-1) else 0.5,
                        zorder=2
                    )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Normalized layer number (%)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task.capitalize()} ({probe_type})")
        ax.grid(True, linestyle='--', alpha=0.5)
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=10, title="Checkpoints")

plt.tight_layout()
save_dir = os.path.join("figures3")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"checkpoint_lineplot_layers_vs_checkpoints_gradient.png")
plt.savefig(save_path, bbox_inches="tight", dpi=200)
print(f"Saved lineplot (gradient) plot as {save_path}")
plt.close()

def print_latex_table(table, caption="Median accuracy by layer bin, checkpoint, task, probe type", label="tab:median_acc"):
    """
    Print a LaTeX table from a pandas DataFrame with MultiIndex columns.
    """
    # Reorder columns for readability: group by (Task, ProbeType, Model), then Checkpoint
    if isinstance(table.columns, pd.MultiIndex):
        table = table.reorder_levels(['Task', 'ProbeType', 'Model', 'Checkpoint'], axis=1)
        table = table.sort_index(axis=1)
    latex = table.round(3).to_latex(
        multicolumn=True,
        multicolumn_format='c',
        escape=False,
        na_rep="--",
        caption=caption,
        label=label,
        longtable=True
    )
    print("\n" + latex)

# --- One big table for all ---
table = get_median_acc_table_all(models_dict, labels_dict, tasks, probe_types, dataset)
print_latex_table(
    table,
    caption="Median classifier accuracy by normalized layer bin, checkpoint, task, probe type, and model.",
    label="tab:median_acc_all"
)
