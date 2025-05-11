import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import math

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update({"font.size": 12})

# models = ["bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
#           "gpt2", "qwen2", "qwen2-instruct", "gemma2b", "pythia1.4b",
#           "llama3-8b", "llama3-8b-instruct", "pythia-6.9b", "pythia-6.9b-tulu"]
models = ["llama3-8b", "llama3-8b-instruct"]

model_names = {
    "gpt2": "GPT 2",
    "qwen2": "Qwen 2.5 1.5B",
    "qwen2-instruct": "Qwen 2.5 1.5B-Instruct",
    "pythia1.4b": "Pythia 1.4B",
    "gemma2b": "Gemma 2 2B",
    "bert-base-uncased": "BERT Base Uncased",
    "bert-large-uncased": "BERT Large Uncased",
    "deberta-v3-large": "DeBERTa v3 Large",
    "llama3-8b": "Llama 3 8B",
    "llama3-8b-instruct": "Llama 3 8B Instruct",
    "pythia-6.9b": "Pythia 6.9B",
    "pythia-6.9b-tulu": "Pythia 6.9B Tulu",
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
    raise ValueError("Could not find accuracy columns in DataFrame.")

colors = {
    "task": sns.color_palette("Set2")[0],
    "control": sns.color_palette("Set2")[1],
}

def plot_all_probes(
    dataset: str,
    model_list: list[str],
    probe_type: str = "nn",
    output_dir="figs",
    save_name="all_probes",
    pca: bool = False,
    pca_dim: int = 50,
):
    combos = [
        ("inflection", "reg"),
        ("lexeme", "reg"),
        ("inflection", probe_type),
        ("lexeme", probe_type),
    ]

    print("Models:")
    for model in model_list:
        print(f"- {model}")

    n_models = len(model_list)
    n_combos = len(combos)

    n_cols = n_models
    n_rows = n_combos

    fig_height = 3 * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, fig_height),
        sharey=True
    )
    plt.subplots_adjust(
        top=0.92, bottom=0.08,
        left=0.05, right=0.98,
        hspace=0.4,
        wspace=0.25
    )

    # Create a new figure for PCA Explained Variance
    fig_pca, axes_pca = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, fig_height),
        sharey=True
    )
    plt.subplots_adjust(
        top=0.92, bottom=0.08,
        left=0.05, right=0.98,
        hspace=0.4,
        wspace=0.25
    )

    for col, model in enumerate(model_list):
        for row, (task, probe_type) in enumerate(combos):
            ax = axes[row, col]
            ax_pca = axes_pca[row, col]
            
            ax.tick_params(axis="both", which="both", length=8, width=2)
            ax_pca.tick_params(axis="both", which="both", length=8, width=2)

            probe_dir = os.path.join("..", "output", "probes",
                        f"{dataset}_{model}_{task}_{probe_type}")
            if pca:
                probe_dir = probe_dir + f"_pca_{pca_dim}"
            csv_path = os.path.join(probe_dir, f"{task}_results.csv")

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                try:
                    acc_col, ctrl_col = get_acc_columns(df, task)
                    ax.plot(
                        df["Layer"], df[acc_col],
                        label=f"{task.capitalize()} task",
                        color=colors["task"], linestyle="-",
                        marker="o", markersize=3
                    )
                    ax.plot(
                        df["Layer"], df[ctrl_col],
                        label=f"{task.capitalize()} control",
                        color=colors["control"], linestyle="-",
                        marker="x", markersize=4
                    )
                    n_layers = int(df["Layer"].max() - df["Layer"].min() + 1)
                    max_bins = 6 if n_layers > 24 else (8 if n_layers > 12 else n_layers)
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=max_bins))

                    # Plot PCA Explained Variance
                    ax_pca.plot(
                        df["Layer"], df["PCA_ExplainedVar"],
                        label="PCA Explained Variance",
                        color="purple", linestyle="-",
                        marker="o", markersize=3
                    )
                    ax_pca.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=max_bins))

                except Exception as e:
                    print(e)
                    ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax.transAxes)
                    ax_pca.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax_pca.transAxes)
            else:
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax.transAxes)
                ax_pca.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax_pca.transAxes)

            if row == 0:
                ax.set_title(model_names.get(model, model), fontsize=26, pad=15)
                ax_pca.set_title(model_names.get(model, model), fontsize=26, pad=15)
            if col == 0:
                task_label = f"{task.capitalize()} Accuracy"
                ax.set_ylabel(task_label, fontsize=18, labelpad=15)
                ax_pca.set_ylabel("PCA Explained Var", fontsize=18, labelpad=15)
            if row == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=22)
                ax_pca.set_xlabel("Layer", fontsize=22)
            ax.set_ylim(0, 1)
            ax_pca.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45, labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax_pca.tick_params(axis="x", rotation=45, labelsize=14)
            ax_pca.tick_params(axis="y", labelsize=14)

    left_pad = -0.15
    fig.text(left_pad, 0.75, "Linear probe", va="center", rotation="vertical", fontsize=22)
    fig.text(left_pad, 0.29, "MLP probe", va="center", rotation="vertical", fontsize=22)

    handles = [
        mpl.lines.Line2D([], [], color=colors["task"], marker="o", linestyle="-", markersize=10, linewidth=4, label="Linguistic accuracy"),
        mpl.lines.Line2D([], [], color=colors["control"], marker="x", linestyle="-", markersize=14, linewidth=4, label="Control accuracy"),
    ]
    labels = ["Linguistic accuracy", "Control accuracy"]
    legend = fig.legend(
        handles, labels,
        loc="lower center",
        fontsize=28,
        frameon=True,
        ncol=2,
        bbox_to_anchor=(0.5, -0.16),
        borderaxespad=2.0,
        fancybox=True,
        title="Legend",
        title_fontsize=28
    )
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(0.95)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{save_name}_wide_{probe_type}"
    if pca:
        filename += f"_pca_{pca_dim}"
    filename += ".png"
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    print(f"Saved figure to {os.path.join(output_dir, filename)}")
    if pca:
        pca_filename = f"{save_name}_wide_{probe_type}_pca_{pca_dim}_2.png"
        fig_pca.savefig(os.path.join(output_dir, pca_filename), dpi=300, bbox_inches="tight")
        print(f"Saved PCA figure to {os.path.join(output_dir, pca_filename)}")

dataset = "ud_gum_dataset"
plot_all_probes(dataset, models, probe_type="nn", pca=False)
