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

from matplotlib.cm import get_cmap

dataset = "ud_gum_dataset"
probe_types = ["nn", "reg"]
tasks = ["lexeme", "inflection"]
colors = {"pythia": "tab:blue", "olmo": "tab:orange"}

import matplotlib

fig, axes = plt.subplots(len(tasks), len(probe_types), figsize=(18, 10), sharex=True, sharey=True)
if len(tasks) == 1 and len(probe_types) == 1:
    axes = np.array([[axes]])
elif len(tasks) == 1 or len(probe_types) == 1:
    axes = axes.reshape(len(tasks), len(probe_types))

for col, probe_type in enumerate(probe_types):
    for row, task in enumerate(tasks):
        ax = axes[row, col]
        cmap = matplotlib.colormaps["viridis"]
        n_pythia = len(pythia_models)
        n_olmo = len(olmo_models)
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
                    df = load_accuracy_data(model, task, probe_type, dataset)
                    if df is not None and len(df) > 0:
                        df["LayerBin"] = pd.cut(df["Layer_Normalized"], bins=bin_edges, labels=bin_labels, include_lowest=True)
                        medians = df.groupby("LayerBin")["Acc"].median()
                        data[col] = [medians.get(bin_label, np.nan) for bin_label in bin_labels]
                    else:
                        data[col] = [np.nan] * len(bin_labels)
    columns = pd.MultiIndex.from_tuples(columns, names=["Model", "Checkpoint", "Task", "ProbeType"])
    table = pd.DataFrame(data, index=bin_labels, columns=columns)
    return table

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

def get_layer_accuracies_for_checkpoint(model, task, probe_type, dataset, target_layers=None):
    df = load_accuracy_data(model, task, probe_type, dataset)
    if df is None or len(df) == 0:
        return {}
    if target_layers is None:
        min_layer = df["Layer"].min()
        max_layer = df["Layer"].max()
        target_layers = [min_layer, min_layer + 3, min_layer + 6, min_layer + 9, max_layer]
        target_layers = sorted(list(set(target_layers)))
    layer_accs = {}
    for layer in target_layers:
        layer_data = df[df["Layer"] == layer]
        if len(layer_data) > 0:
            layer_accs[layer] = layer_data["Acc"].iloc[0]
    return layer_accs

def generate_markdown_table(models, labels, model_name, task, probe_type, dataset):
    final_model = models[-1]
    df_final = load_accuracy_data(final_model, task, probe_type, dataset)
    if df_final is None:
        print(f"No data for final model {final_model}")
        return
    min_layer = df_final["Layer"].min()
    max_layer = df_final["Layer"].max()
    layer_range = max_layer - min_layer
    target_layers = [
        min_layer,
        min_layer + int(0.25 * layer_range),
        min_layer + int(0.5 * layer_range),
        min_layer + int(0.75 * layer_range),
        max_layer
    ]
    target_layers = sorted(list(set(target_layers)))
    layer_headers = []
    for layer in target_layers:
        if layer == min_layer:
            layer_headers.append(f"Layer {layer} (first layer)")
        elif layer == max_layer:
            layer_headers.append(f"Layer {layer} (last layer)")
        else:
            layer_headers.append(f"Layer {layer}")
    table_title = f"# {task.capitalize()} Accuracy - {'MLP' if probe_type == 'nn' else 'Linear Regression'} ({model_name})"
    header = f"| Checkpoint | {' | '.join(layer_headers)} |"
    separator = f"| {'---' + ' | ---' * len(layer_headers)} |"
    rows = []
    for i, (model, label) in enumerate(zip(models, labels)):
        layer_accs = get_layer_accuracies_for_checkpoint(model, task, probe_type, dataset, target_layers)
        acc_values = []
        for layer in target_layers:
            if layer in layer_accs:
                acc_values.append(f"{layer_accs[layer]:.3f}")
            else:
                acc_values.append("--")
        if model_name == "Pythia":
            if "step" in model:
                step_str = model.split("_step")[-1]
                if step_str.isdigit():
                    step_num = int(step_str)
                    tokens = step_num * 2_097_152
                    tokens_b = tokens / 1_000_000_000
                    if tokens_b < 1:
                        token_info = f" ({tokens / 1_000_000:.0f}M tokens)"
                    else:
                        token_info = f" ({tokens_b:.1f}B tokens)"
                else:
                    token_info = ""
            else:
                token_info = " (300B tokens)"
        else:
            token_info = ""
        checkpoint_label = label + token_info
        row = f"| {checkpoint_label} | {' | '.join(acc_values)} |"
        rows.append(row)
    table = "\n".join([table_title, header, separator] + rows)
    return table

def print_all_markdown_tables():
    dataset = "ud_gum_dataset"
    probe_types = [("nn", "MLP"), ("reg", "Linear Regression")]
    tasks = ["lexeme", "inflection"]
    models_info = [
        (pythia_models, pythia_labels, "Pythia"),
        (olmo_models, olmo_labels, "OLMo")
    ]
    for models, labels, model_name in models_info:
        for task in tasks:
            for probe_type, probe_name in probe_types:
                table = generate_markdown_table(models, labels, model_name, task, probe_type, dataset)
                if table:
                    print(table)
                    print()

print("\n" + "="*80)
print("MARKDOWN TABLES FOR CHECKPOINT RESULTS")
print("="*80)
print_all_markdown_tables()
