import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 300
plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 22,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0,
})

models = [
    "bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
    "gpt2", "gpt2-large", "gpt2-xl",
    "pythia-6.9b", "pythia-6.9b-tulu",
    "olmo2-7b-instruct", "olmo2-7b",
    "gemma2b", "gemma2b-it",
    "qwen2", "qwen2-instruct",
    "llama3-8b", "llama3-8b-instruct",
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
    "bert-base-uncased": "BERT-Base",
    "bert-large-uncased": "BERT-Large",
    "deberta-v3-large": "DeBERTa-v3-Large",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B",
}

MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#a55194", "#393b79",
    "#637939", "#e6550d", "#9c9ede", "#f7b6d2"
]

def get_model_color(model, model_list):
    idx = model_list.index(model)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

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

def plot_linguistic_and_selectivity(
    dataset: str,
    model_list: list[str],
    probe_type: str = "nn",
    output_dir="figures3",
    save_name="linguistic_selectivity",
    pca: bool = False,
    pca_dim: int = 50,
):
    probe_types = ["reg", probe_type, "rf"]
    titles = ["Linear Regression", "MLP", "Random Forest"]
    tasks = ["lexeme", "inflection"]
    n_rows = len(tasks)
    n_cols = len(probe_types)

    all_regression_results = []

    # Linguistic Accuracy Plot
    aspect_ratio = 8 / 3
    base_height = 5
    fig_width = n_cols * base_height * aspect_ratio / n_rows
    fig_height = n_rows * base_height
    fig1, axes1 = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        sharey=True
    )
    if n_rows == 1:
        axes1 = np.expand_dims(axes1, axis=0)
    if n_cols == 1:
        axes1 = np.expand_dims(axes1, axis=1)

    for row, task in enumerate(tasks):
        for col, probe in enumerate(probe_types):
            ax = axes1[row, col]
            if task == "lexeme" and probe == "rf":
                # Only show placeholder explanation, no lines or legend entry
                ax.text(
                    0.5, 0.5,
                    "(computationally infeasible: too many classes,\nprone to overfitting)",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=18, color="gray"
                )
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.1, 0.2))
                ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.2)])
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                if row == 0:
                    ax.set_title(titles[col], pad=15)
                if col == 0:
                    if row == 0:
                        ax.set_ylabel("Lexeme Accuracy", labelpad=15)
                    if row == 1:
                        ax.set_ylabel("Inflection Accuracy", labelpad=15)
                ax.set_xlabel("Layer (% of model)", labelpad=15)
                continue
            for i, model in enumerate(model_list):
                # Special handling: skip lexeme RF, plot dashes and add note
                if task == "lexeme" and probe == "rf":
                    # Only plot dashes once for all models
                    if i == 0:
                        ax.plot(
                            [0, 1], [np.nan, np.nan],  # invisible line for legend
                            label="No results",
                            color="gray", linestyle="--", linewidth=3.0
                        )
                        ax.text(
                            0.5, 0.5,
                            "— (infeasible: too many classes,\nprone to overfitting)",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=18, color="gray"
                        )
                    continue

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
                    df["Layer_Normalized"] = (
                        df["Layer"] - df["Layer"].min()
                    ) / (df["Layer"].max() - df["Layer"].min())
                    ax.plot(
                        df["Layer_Normalized"], df[acc_col],
                        label=model_names.get(model, model),
                        linewidth=3.0,
                        color=get_model_color(model, model_list),
                    )
                    fit_and_store_regression(df, model, task, probe, all_regression_results)
                except Exception:
                    ax.text(
                        0.5, 0.5, f"No {task} data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=22
                    )

            # Set bigger tick marks
            ax.tick_params(axis='both', which='major', length=10, width=2)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
            ax.set_ylim(0.4, 1)
            ax.set_yticks(np.arange(0.4, 1.05, 0.1))
            ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.4, 1.05, 0.1)])
            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

            # only top row gets titles
            if row == 0:
                ax.set_title(titles[col], pad=15)
            if col == 0:
                if row == 0:
                    ax.set_ylabel("Lexeme Accuracy", labelpad=15)
                if row == 1:
                    ax.set_ylabel("Inflection Accuracy", labelpad=15)
            ax.set_xlabel("Normalized layer number (%)", labelpad=15)

    handles, labels = axes1[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig1.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0, -0.15, 1, 0.1),
            ncol=min(4, len(labels)),
            mode="expand",
            frameon=True,
            title="Models"
        )
    fig1.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    filename1 = "linguistic_accuracy"
    if pca:
        filename1 += f"_pca_{pca_dim}"
    filename1 += ".png"
    fig1.savefig(os.path.join(output_dir, filename1), dpi=200, bbox_inches="tight")
    print(f"Saved linguistic accuracy figure to {os.path.join(output_dir, filename1)}")

    # Selectivity Plot
    fig2, axes2 = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        sharey=True
    )
    if n_rows == 1:
        axes2 = np.expand_dims(axes2, axis=0)
    if n_cols == 1:
        axes2 = np.expand_dims(axes2, axis=1)

    for row, task in enumerate(tasks):
        for col, probe in enumerate(probe_types):
            ax = axes2[row, col]
            if task == "lexeme" and probe == "rf":
                # Only show placeholder explanation, no lines or legend entry
                ax.text(
                    0.5, 0.5,
                    "(computationally infeasible: too many classes,\nprone to overfitting)",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=18, color="gray"
                )
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                ax.set_ylim(-1, 1)
                ax.set_yticks(np.arange(-1, 1.1, 0.5))
                ax.set_yticklabels([f"{y:.1f}" for y in np.arange(-1, 1.1, 0.5)])
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                if row == 0:
                    ax.set_title(titles[col], pad=15)
                    if col == 0:
                        ax.set_ylabel("Lexeme Selectivity", labelpad=15, fontsize=24)
                if row == 1 and col == 0:
                    ax.set_ylabel("Inflection Selectivity", labelpad=15, fontsize=24)
                ax.set_xlabel("Layer (% of model)", labelpad=15)
                continue
            for i, model in enumerate(model_list):
                if task == "lexeme" and probe == "rf":
                    if i == 0:
                        ax.plot(
                            [0, 1], [np.nan, np.nan],
                            label="No results",
                            color="gray", linestyle="--", linewidth=3.0
                        )
                        ax.text(
                            0.5, 0.5,
                            "(computationally infeasible: too many classes,\nprone to overfitting)",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=18, color="gray"
                        )
                    continue

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
                    df["Layer_Normalized"] = (
                        df["Layer"] - df["Layer"].min()
                    ) / (df["Layer"].max() - df["Layer"].min())
                    selectivity = df[acc_col] - df[ctrl_col]
                    ax.plot(
                        df["Layer_Normalized"], selectivity,
                        label=model_names.get(model, model),
                        linewidth=3.0,
                        color=get_model_color(model, model_list),
                    )
                except Exception:
                    ax.text(
                        0.5, 0.5, f"No {task} data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=22
                    )

            # Set bigger tick marks
            ax.tick_params(axis='both', which='major', length=10, width=2)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
            ax.set_ylim(-0.2, 0.8)
            ax.set_yticks(np.arange(-0.2, 0.81, 0.2))
            ax.set_yticklabels([f"{y:.1f}" for y in np.arange(-0.2, 0.81, 0.2)])
            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

            # only top row gets titles
            if row == 0:
                ax.set_title(titles[col], pad=15)
                if col == 0:
                    ax.set_ylabel("Lexeme Selectivity", labelpad=15, fontsize=24)
            if row == 1 and col == 0:
                ax.set_ylabel("Inflection Selectivity", labelpad=15, fontsize=24)
            ax.set_xlabel("Normalized layer number (%)", labelpad=15)

    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    if handles2 and labels2:
        fig2.legend(
            handles2, labels2,
            loc="lower center",
            bbox_to_anchor=(0, -0.15, 1, 0.1),
            ncol=min(4, len(labels2)),
            mode="expand",
            frameon=True,
            title="Models"
        )
    fig2.tight_layout(rect=[0, 0.05, 1, 0.97])
    filename2 = "classifier_selectivity"
    if pca:
        filename2 += f"_pca_{pca_dim}"
    filename2 += ".png"
    fig2.savefig(os.path.join(output_dir, filename2), dpi=200, bbox_inches="tight")
    print(f"Saved selectivity figure to {os.path.join(output_dir, filename2)}")

    # Save regression results
    regression_df = pd.DataFrame(all_regression_results)
    regression_filename = "all_regression_results.csv"
    regression_filepath = os.path.join(output_dir, regression_filename)
    regression_df.to_csv(regression_filepath, index=False)
    print(f"Saved all regression results to {regression_filepath}")

dataset = "ud_gum_dataset"
plot_linguistic_and_selectivity(dataset, models, probe_type="nn", pca=False)
