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
import ast

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update({"font.size": 12})

models = ["bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
          "gpt2", "qwen2", "qwen2-instruct", "gemma2b", "pythia1.4b"]

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

def fit_and_store_regression(df, model_name, task, probe, all_results):
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

def plot_pca_results(
    dataset: str,
    model_list: list[str],
    probe_types: list[str] = ["nn", "reg"],
    output_dir="figures3",
    pca_dim: int = 50,
):
    combos = [
        ("lexeme", "reg"),
        ("lexeme", "nn"),
        ("inflection", "reg"),
        ("inflection", "nn"),
    ]
    n_rows = len(combos)
    n_cols = 3

    fig1_height = 3 * n_rows
    fig1, axes1 = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, fig1_height),
    )
    plt.subplots_adjust(
        top=0.92, bottom=0.08,
        left=0.07, right=0.98,
        hspace=0.35,
        wspace=0.35
    )

    pastel_colors = sns.color_palette("Dark2", n_colors=len(model_list))

    for row, (task, probe) in enumerate(combos):
        for i, model in enumerate(model_list):
            pca_suffix = f"pca{pca_dim}" if probe == "reg" else f"pca_{pca_dim}"
            probe_dir = os.path.join("..", "output", "probes",
                        f"{dataset}_{model}_{task}_{probe}_{pca_suffix}")
            csv_path = os.path.join(probe_dir, f"{task}_results.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            try:
                acc_col, ctrl_col = get_acc_columns(df, task)
                df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())

                axes1[row, 0].plot(
                    df["Layer_Normalized"], df[acc_col],
                    label=model_names.get(model, model),
                    linewidth=1.5,
                    color=pastel_colors[i],
                )
                axes1[row, 1].plot(
                    df["Layer_Normalized"], df[ctrl_col],
                    label=model_names.get(model, model),
                    linewidth=1.5,
                    color=pastel_colors[i],
                )
                selectivity = df[acc_col] - df[ctrl_col]
                axes1[row, 2].plot(
                    df["Layer_Normalized"], selectivity,
                    label=model_names.get(model, model),
                    linewidth=1.5,
                    color=pastel_colors[i],
                )

                for col in range(n_cols):
                    axes1[row, col].set_xlim(0, 1)
                    axes1[row, col].set_xticks(np.arange(0, 1.1, 0.2))
                    axes1[row, col].set_xticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.2)])

            except Exception as e:
                print(f"{model} {task} error: {e}")
                for col in range(n_cols):
                    axes1[row, col].text(0.5, 0.5, f"No {task} data", ha="center", va="center", transform=axes1[row, col].transAxes)

        if row == 0:
            axes1[row, 0].set_title("Linguistic Accuracy", fontsize=22, pad=15)
            axes1[row, 1].set_title("Control Accuracy", fontsize=22, pad=15)
            axes1[row, 2].set_title("Selectivity", fontsize=22, pad=15)
        if row == 0:
            axes1[row, 0].set_ylabel("Linear", fontsize=20, labelpad=15)
        elif row == 1:
            axes1[row, 0].set_ylabel("MLP", fontsize=20, labelpad=15)
        elif row == 2:
            axes1[row, 0].set_ylabel("Linear", fontsize=20, labelpad=15)
        elif row == 3:
            axes1[row, 0].set_ylabel("MLP", fontsize=20, labelpad=15)
        for col in range(n_cols):
            if row == n_rows - 1:
                axes1[row, col].set_xlabel("Layer (relative % of model)", fontsize=20, labelpad=15)
                axes1[row, col].set_xticks(np.arange(0.2, 1.1, 0.2))
                axes1[row, col].tick_params(axis="x", labelsize=18, width=2, length=10)
            else:
                axes1[row, col].set_xticklabels([])
                axes1[row, col].set_xlabel("")
                axes1[row, col].tick_params(axis="x", length=0)
            if col == 2:
                axes1[row, col].set_ylim(-1, 1)
                axes1[row, col].set_ylabel("Selectivity", fontsize=20, labelpad=5)
                axes1[row, col].set_yticks([-1, -0.5, 0, 0.5, 1])
            else:
                axes1[row, col].set_ylim(0, 1)
                if col == 0:
                    axes1[row, col].set_ylabel(f"{'Linear' if row % 2 == 0 else 'MLP'} Accuracy", fontsize=20, labelpad=15)
                else:
                    axes1[row, col].set_yticks([], [])
                    axes1[row, col].set_ylabel("")
            axes1[row, col].tick_params(axis="y", labelsize=18, width=2, length=10)
            axes1[row, col].grid(True, linestyle="--", alpha=0.3)

    for row in range(n_rows):
        axes1[row, 1].sharey(axes1[row, 0])

    fig1.text(-0.03, 0.75, "Lexeme", va="center", rotation="vertical", fontsize=22)
    fig1.text(-0.03, 0.29, "Inflection", va="center", rotation="vertical", fontsize=22)

    handles, labels = axes1[0,0].get_legend_handles_labels()
    fig1.legend(
        handles, labels,
        loc="upper center",
        fontsize=20,
        frameon=True,
        ncol=6,
        bbox_to_anchor=(0.5, 0.02),
        borderaxespad=2.0,
        fancybox=True,
        title="Models",
        title_fontsize=22
    )

    os.makedirs(output_dir, exist_ok=True)
    filename1 = f"combined_probes_pca_{pca_dim}.png"
    fig1.savefig(os.path.join(output_dir, filename1), dpi=300, bbox_inches="tight")
    print(f"Saved PCA accuracy figure to {os.path.join(output_dir, filename1)}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    probe_type = "nn"
    for i, model in enumerate(model_list):
        pca_suffix = f"pca{pca_dim}" if probe_type == "reg" else f"pca_{pca_dim}"
        probe_dir = os.path.join("..", "output", "probes",
                                 f"{dataset}_{model}_lexeme_{probe_type}_{pca_suffix}")
        csv_path = os.path.join(probe_dir, "lexeme_results.csv")
        if not os.path.exists(csv_path):
            print(f"Missing explained variance for {model}")
            continue
        df = pd.read_csv(csv_path)
        if "PCA_ExplainedVar" not in df.columns:
            print(f"PCA_ExplainedVar column missing for {model}")
            continue
        try:
            explained_var = [float(x) for x in df["PCA_ExplainedVar"].tolist()]

            if not isinstance(explained_var, list):
                explained_var = [explained_var]
            components = np.arange(1, len(explained_var) + 1)
            ax2.plot(
                components, explained_var[:pca_dim],
                label=model_names.get(model, model),
                linewidth=2,
                color=pastel_colors[i]
            )
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing PCA_ExplainedVar for {model}: {e}")
            continue
        except Exception as e:
            print(f"Error plotting PCA_ExplainedVar for {model}: {e}")
            continue
    ax2.set_xlabel("PCA Component", fontsize=18)
    ax2.set_ylabel("Explained Variance Ratio", fontsize=18)
    ax2.set_title(f"PCA Explained Variance (Top {pca_dim} Components)", fontsize=20)
    ax2.legend(fontsize=14, ncol=len(model_list) // 2, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax2.grid(True, linestyle="--", alpha=0.3)
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    filename2 = f"explained_variance_pca_{pca_dim}.png"
    fig2.savefig(os.path.join(output_dir, filename2), dpi=300, bbox_inches="tight")
    print(f"Saved explained variance figure to {os.path.join(output_dir, filename2)}")

dataset = "ud_gum_dataset"
plot_pca_results(dataset, models, pca_dim=50)
