import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update({"font.size": 12})

models = ["bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
          "gpt2", "gpt2-large", "gpt2-xl", "qwen2", "qwen2-instruct", "gemma2b",
          "llama3-8b", "llama3-8b-instruct", "pythia-6.9b", "pythia-6.9b-tulu",
          "olmo2-7b-instruct", # "olmo2-7b", "gemma2b-it",
        ]

model_names = {
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
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

def fit_and_store_regression(df, model_name, task, probe, all_results):
    """Fits a linear regression and saves the results to a list."""
    for col_prefix in ["", "control_"]:
        task_col = f"{task}_{'Accuracy' if col_prefix == '' else 'ControlAccuracy'}"
        if task_col in df.columns:
            y = df[task_col].values
        else:
            y = df["Acc"].values if col_prefix == "" else df["controlAcc"].values
        X = df["Layer_Normalized"].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        all_results.append({
            "model": model_name,
            "task": task,
            "probe": probe,
            "type": "linguistic" if col_prefix == "" else "control",
            "slope": model.coef_[0],
            "intercept": model.intercept_,
            "r2": r2,
        })

def plot_combined_probes(
    dataset: str,
    model_list: list[str],
    probe_type: str = "nn",
    output_dir="figures3",
    save_name="combined_probes",
    pca: bool = False,
    pca_dim: int = 50,
):
    # Define the 4 tasks: (task, probe_type)
    combos = [
        ("lexeme", "reg"),
        ("lexeme", probe_type),
        ("inflection", "reg"),
        ("inflection", probe_type),
    ]
    n_rows = len(combos)
    n_cols = 3  # linguistic, control, selectivity

    fig_height = 3 * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, fig_height),
    )
    plt.subplots_adjust(
        top=0.92, bottom=0.08,
        left=0.07, right=0.98,
        hspace=0.35,
        wspace=0.35
    )

    pastel_colors = sns.color_palette("Dark2", n_colors=len(model_list))

    all_regression_results = []  # Accumulate results here

    for row, (task, probe) in enumerate(combos):
        for i, model in enumerate(model_list):
            probe_dir = os.path.join("..", "output", "probes",
                        f"{dataset}_{model}_{task}_{probe}")
            if pca:
                probe_dir = probe_dir + f"_pca_{pca_dim}"
            csv_path = os.path.join(probe_dir, f"{task}_results.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            try:
                acc_col, ctrl_col = get_acc_columns(df, task)
                # Normalize layer number
                df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())

                # Plot linguistic accuracy
                axes[row, 0].plot(
                    df["Layer_Normalized"], df[acc_col],
                    label=model_names.get(model, model),
                    linewidth=1.5,
                    color=pastel_colors[i],
                    # alpha=0.5
                )
                # Plot control accuracy
                axes[row, 1].plot(
                    df["Layer_Normalized"], df[ctrl_col],
                    label=model_names.get(model, model),
                    linewidth=1.5,
                    color=pastel_colors[i],
                    # alpha=0.5
                )
                # Plot selectivity (difference)
                selectivity = df[acc_col] - df[ctrl_col]
                axes[row, 2].plot(
                    df["Layer_Normalized"], selectivity,
                    label=model_names.get(model, model),
                    linewidth=1.5,
                    color=pastel_colors[i],
                    # alpha=0.5
                )

                for col in range(n_cols):
                    axes[row, col].set_xlim(0, 1)
                    axes[row, col].set_xticks(np.arange(0, 1.1, 0.2))
                    axes[row, col].set_xticklabels([f"{x:.1f}[:-1]" for x in np.arange(0, 1.1, 0.2)])
                    axes[row, col].set_xticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.2)])

                fit_and_store_regression(df, model, task, probe, all_regression_results)

            except Exception as e:
                print(f"{model} {task} error: {e}")
                for col in range(n_cols):
                    axes[row, col].text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=axes[row, col].transAxes)

        # Set titles and labels
        if row == 0:
            axes[row, 0].set_title("Linguistic Accuracy", fontsize=22, pad=15)
            axes[row, 1].set_title("Control Accuracy", fontsize=22, pad=15)
            axes[row, 2].set_title("Selectivity", fontsize=22, pad=15)
        # Row labels
        if row == 0:
            axes[row, 0].set_ylabel("Linear", fontsize=20, labelpad=15)
        elif row == 1:
            axes[row, 0].set_ylabel("MLP", fontsize=20, labelpad=15)
        elif row == 2:
            axes[row, 0].set_ylabel("Linear", fontsize=20, labelpad=15)
        elif row == 3:
            axes[row, 0].set_ylabel("MLP", fontsize=20, labelpad=15)
            
        for col in range(n_cols):
            if row == n_rows - 1:
                axes[row, col].set_xlabel("Layer (relative % of model)", fontsize=20, labelpad=15)
                axes[row, col].set_xticks(np.arange(0.2, 1.1, 0.2))
                axes[row, col].tick_params(axis="x", labelsize=18, width=2, length=10)
            else:
                axes[row, col].set_xticklabels([])
                axes[row, col].set_xlabel("")
                axes[row, col].tick_params(axis="x", length=0)
            if col == 2:
                axes[row, col].set_ylim(-1, 1)
                axes[row, col].set_ylabel("Selectivity", fontsize=20, labelpad=5)
                axes[row, col].set_yticks([-1, -0.5, 0, 0.5, 1])
            else:
                axes[row, col].set_ylim(0, 1)
                if col == 0:
                    axes[row, col].set_ylabel(f"{'Linear' if row % 2 == 0 else 'MLP'} Accuracy", fontsize=20, labelpad=15)
                else:
                    axes[row, col].set_yticks([], [])
                    axes[row, col].set_ylabel("")
            axes[row, col].tick_params(axis="y", labelsize=18, width=2, length=10)
            axes[row, col].grid(True, linestyle="--", alpha=0.3)
            
    for row in range(n_rows):
        axes[row, 1].sharey(axes[row, 0])

    fig.text(-0.03, 0.75, "Lexeme", va="center", rotation="vertical", fontsize=22)
    fig.text(-0.03, 0.29, "Inflection", va="center", rotation="vertical", fontsize=22)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        fontsize=20,
        frameon=True,
        ncol=6,  # Increased number of columns for a wider legend
        bbox_to_anchor=(0.5, 0.02),
        borderaxespad=2.0,
        fancybox=True,
        title="Models",
        title_fontsize=22  # Increased title font size
    )

    regression_df = pd.DataFrame(all_regression_results)
    os.makedirs(output_dir, exist_ok=True)
    regression_filename = "all_regression_results.csv"
    regression_filepath = os.path.join(output_dir, regression_filename)
    regression_df.to_csv(regression_filepath, index=False)
    print(f"Saved all regression results to {regression_filepath}")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{save_name}_combined_{probe_type}"
    if pca:
        filename += f"_pca_{pca_dim}"
    filename += ".png"
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    print(f"Saved combined figure to {os.path.join(output_dir, filename)}")

dataset = "ud_gum_dataset"
plot_combined_probes(dataset, models, probe_type="nn", pca=False)
