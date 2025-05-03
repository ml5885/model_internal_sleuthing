import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update({"font.size": 12})

models = [
    "gpt2", "qwen2", "qwen2-instruct",
    "pythia1.4b", "gemma2b",
    "bert-base-uncased", "bert-large-uncased",
    "deberta-v3-large"
]

model_names = {
    "gpt2": "GPT 2",
    "qwen2": "Qwen 2.5 1.5B",
    "qwen2-instruct": "Qwen 2.5 1.5B-Instruct",
    "pythia1.4b": "Pythia 1.4B",
    "gemma2b": "Gemma 2 2B",
    "bert-base-uncased": "BERT Base Uncased",
    "bert-large-uncased": "BERT Large Uncased",
    "deberta-v3-large": "DeBERTa v3 Large",
}

def get_acc_columns(df, prefix):
    # (same as before)
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
    output_dir="figs",
    save_name="all_probes",
    layout="long",  # "long" (default) or "wide"
):
    # define the four (task, probe) combos in the desired row order
    combos = [
        ("inflection", "reg"),
        ("lexeme", "reg"),
        ("inflection", "mlp"),
        ("lexeme", "mlp"),
    ]

    n_models = len(model_list)
    if layout == "wide":
        n_cols = n_models
        n_rows = 4  # 4 probe-task combos
    else:  # "long"
        n_cols = 4
        n_rows = 8  # 2 tasks x 2 probe types x 2 model groups (4+4)

    # Add extra vertical space between the two 4-row blocks for "long"
    fig_height = 3 * n_rows + (2 if layout == "long" else 0)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, fig_height),
        sharey=True
    )
    plt.subplots_adjust(
        top=0.92, bottom=0.08,
        left=0.05, right=0.98,
        hspace=0.8 if layout == "long" else 0.4,
        wspace=0.25
    )

    if layout == "wide":
        # axes shape: (4, n_models)
        for col, model in enumerate(model_list):
            for row, (task, probe_type) in enumerate(combos):
                ax = axes[row, col]
                probe_dir = os.path.join("..", "output", "probes",
                            f"{dataset}_{model}_{task}_{probe_type}")
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
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax.transAxes)

                if row == 0:
                    ax.set_title(model_names.get(model, model), fontsize=26, pad=15)
                if col == 0:
                    # Label rows with task and probe type
                    probe_name = "MLP" if probe_type == "mlp" else "Linear"
                    task_label = f"{task.capitalize()} Accuracy\n({probe_name} probe)"
                    ax.set_ylabel(task_label, fontsize=22, labelpad=15)
                ax.set_xlabel("Layer", fontsize=22)
                ax.set_ylim(0, 1)
                ax.tick_params(axis="x", rotation=45, labelsize=14)
                ax.tick_params(axis="y", labelsize=14)
    else:
        # axes shape: (8, 4)
        for idx, model in enumerate(model_list):
            block = idx // 4  # 0 or 1
            col = idx % 4
            for row_offset, (task, probe_type) in enumerate(combos):
                row = block * 4 + row_offset
                ax = axes[row, col]
                probe_dir = os.path.join("..", "output", "probes",
                                         f"{dataset}_{model}_{task}_{probe_type}")
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
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=ax.transAxes)

                if row_offset == 0:
                    ax.set_title(model_names.get(model, model), fontsize=26)
                if col == 0:
                    ax.set_ylabel("Accuracy", fontsize=16)
                ax.set_xlabel("Layer", fontsize=16)
                ax.set_ylim(0, 1)
                ax.tick_params(axis="x", rotation=45, labelsize=14)
                ax.tick_params(axis="y", labelsize=14)

    # fig.suptitle(
    #     "Probing accuracies on lexeme and inflection tasks across models",
    #     fontsize=34, weight="bold"
    # )

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
        bbox_to_anchor=(0.5, -0.04) if layout == "long" else (0.5, -0.16),
        borderaxespad=2.0,
        fancybox=True,
        title="Legend",
        title_fontsize=28
    )
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(0.95)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{save_name}_{layout}.png"), dpi=300, bbox_inches="tight")

dataset = "ud_gum_dataset"
plot_all_probes(dataset, models, layout="wide")
