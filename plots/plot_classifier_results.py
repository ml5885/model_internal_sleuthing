import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 100
plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 22,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0
})

bbox_to_anchor = (0, -0.17, 1, 0.1)

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
}

MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#a55194", "#393b79",
    "#637939", "#e6550d", "#9c9ede", "#f7b6d2",
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
    "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
    "#c7e9c0", "#fdae6b", "#9ecae1", "#fd8d3c"
]

def get_model_color(model, model_list):
    if ("pythia-6.9b" in model and ("step" in model or model == "pythia-6.9b")) or \
       ("olmo2-7b" in model and ("stage1-step" in model or model == "olmo2-7b")):
        if "pythia-6.9b" in model:
            return get_checkpoint_gradient_color(model, model_list, "#2ca02c")
        elif "olmo2-7b" in model:
            return get_checkpoint_gradient_color(model, model_list, "#1f77b4")
    
    idx = model_list.index(model)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

def get_checkpoint_gradient_color(model, model_list, base_color="#1f77b4"):
    if "pythia-6.9b" in model:
        checkpoint_models = [m for m in model_list if "pythia-6.9b" in m]
        base_name = "pythia-6.9b"
    elif "olmo2-7b" in model:
        checkpoint_models = [m for m in model_list if "olmo2-7b" in m and "stage1-step" in m]
        if "olmo2-7b" in model_list:
            checkpoint_models = ["olmo2-7b"] + checkpoint_models
        base_name = "olmo2-7b"
    else:
        idx = model_list.index(model)
        return MODEL_COLORS[idx % len(MODEL_COLORS)]
    
    if len(checkpoint_models) <= 1:
        return base_color
    
    def extract_step_number(model_name):
        if model_name == base_name:
            return float('inf')
        if "step" in model_name:
            try:
                if "tokens" in model_name:
                    tokens_part = model_name.split("tokens")[1].split("B")[0]
                    return float(tokens_part)
                else:
                    step_part = model_name.split("step")[1].split("-")[0].split("_")[0]
                    if "k" in step_part:
                        return float(step_part.replace("k", "")) * 1000
                    return float(step_part)
            except:
                return 0
        return 0
    
    sorted_checkpoints = sorted(checkpoint_models, key=extract_step_number)
    
    try:
        model_idx = sorted_checkpoints.index(model)
    except ValueError:
        return base_color
    
    n_models = len(sorted_checkpoints)
    if n_models == 1:
        return base_color
    
    viridis = plt.cm.viridis
    color_position = 0.2 + (0.7 * model_idx / (n_models - 1))
    rgba_color = viridis(color_position)
    
    return mcolors.to_hex(rgba_color)

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

def get_tick_values(ymin, ymax, min_ticks=6):
    """
    Compute tick values and labels for a given y-axis range, ensuring at least min_ticks (default 6) ticks,
    including start and end. Returns (ticks, labels).
    """
    span = ymax - ymin
    if span == 0:
        return np.array([ymin]), [f"{ymin:.1f}"]
    # Try to find a "nice" step size
    raw_step = span / (min_ticks - 1)
    # Use a set of "nice" steps
    nice_steps = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0])
    step = nice_steps[np.searchsorted(nice_steps, raw_step, side="left")]
    # Compute ticks
    first_tick = np.ceil(ymin / step) * step
    last_tick = np.floor(ymax / step) * step
    ticks = np.arange(first_tick, last_tick + step/2, step)
    # Ensure start and end are included
    if abs(ticks[0] - ymin) > 1e-8:
        ticks = np.insert(ticks, 0, ymin)
    if abs(ticks[-1] - ymax) > 1e-8:
        ticks = np.append(ticks, ymax)
    # Remove duplicates and sort
    ticks = np.unique(np.round(ticks, 8))
    # Format labels
    if step < 0.1:
        labels = [f"{y:.2f}" for y in ticks]
    else:
        labels = [f"{y:.1f}" for y in ticks]
    return ticks, labels

def plot_linguistic_and_selectivity(
    dataset: str,
    model_list: list[str],
    probe_type: str = "nn",
    output_dir="figures3",
    save_name="linguistic_selectivity",
    pca: bool = False,
    pca_dim: int = 50,
    linguistic_filename: str = None,
    selectivity_filename: str = None,
    ylim: tuple = ((0, 1.0), (0, 1.0)),
):
    probe_types = ["reg", probe_type, "rf"]
    titles = ["Linear Regression", "MLP", "Random Forest"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)
    all_regression_results = []
    aspect_ratio, base_height = 8 / 3, 5
    orig_width  = n_cols * base_height * aspect_ratio / n_rows
    orig_height = n_rows * base_height

    height_scale = 0.85
    fig_size = (orig_width, orig_height * height_scale)

    def plot_panel(fig, axes, plot_selectivity=False):
        for row, task in enumerate(tasks):
            for col, probe in enumerate(probe_types):
                ax = axes[row, col]
                if task == "lexeme" and probe == "rf":
                    ax.text(0.5, 0.5, "(computationally infeasible: too many classes,\nprone to overfitting)",
                            ha="center", va="center", transform=ax.transAxes, fontsize=18, color="gray")
                    ax.set_xlim(0, 1)
                    ax.set_xticks(np.arange(0, 1.1, 0.2))
                    ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                    
                    if plot_selectivity:
                        ax.set_ylim(0, 1.0)
                        ax.set_yticks(np.arange(0, 1.01, 0.2))
                        if col == 0:
                            ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.01, 0.2)])
                            if row == 0:
                                ax.set_ylabel("Lexeme Selectivity", labelpad=15)
                            elif row == 1:
                                ax.set_ylabel("Inflection Selectivity", labelpad=15)
                        else:
                            ax.set_yticklabels([])
                    else:
                        ax.set_ylim(0, 1)
                        ax.set_yticks(np.arange(0, 1.1, 0.2))
                        if col == 0:
                            ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.2)])
                            if row == 0:
                                ax.set_ylabel("Lexeme Accuracy", labelpad=15)
                            elif row == 1:
                                ax.set_ylabel("Inflection Accuracy", labelpad=15)
                        else:
                            ax.set_yticklabels([])
                    
                    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                    if row == 1:
                        ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                    else:
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
                    if row == 0:
                        ax.set_title(titles[col], pad=15)
                    ax.tick_params(axis='y', which='major', length=10, width=2)
                    continue
                
                # Main plotting loop
                for i, model in enumerate(model_list):
                    # Use language-specific subfolder for non-English datasets
                    if dataset in ("ud_gum_dataset", "en_gum", "english", "en_gum_dataset"):
                        probe_dir = os.path.join("..", "output", "probes",
                                    f"{dataset}_{model}_{task}_{probe}")
                    else:
                        probe_dir = os.path.join("..", "output", dataset, "probes",
                                    f"{dataset}_{model}_{task}_{probe}")
                    if pca:
                        probe_dir += f"_pca_{pca_dim}"
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    if not os.path.exists(csv_path):
                        print(f"[WARN] Missing results for model: {model} at {csv_path}")
                        continue
                    df = pd.read_csv(csv_path)
                    try:
                        acc_col, ctrl_col = get_acc_columns(df, task)
                        df["Layer_Normalized"] = (
                            df["Layer"] - df["Layer"].min()
                        ) / (df["Layer"].max() - df["Layer"].min())
                        y = df[acc_col] if not plot_selectivity else df[acc_col] - df[ctrl_col]
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=3.0,
                            color=get_model_color(model, model_list),
                        )
                        if not plot_selectivity:
                            fit_and_store_regression(df, model, task, probe, all_regression_results)
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=22)
                
                # Set axis properties
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                
                if plot_selectivity:
                    if row == 0:
                        row_ylim = (0, 0.2)
                        step = 0.1
                    else:
                        row_ylim = (0, 0.8)
                        step = 0.2
                    yticks = np.arange(row_ylim[0], row_ylim[1] + 1e-8, step)
                    ylabels = [f"{y:.1f}" for y in yticks]
                    ax.set_ylim(*row_ylim)
                    ax.set_yticks(yticks)
                    if col == 0:
                        ax.yaxis.set_tick_params(labelleft=True)
                        ax.set_yticklabels(ylabels)
                        if row == 0:
                            ax.set_ylabel("Lexeme Selectivity", labelpad=15)
                        elif row == 1:
                            ax.set_ylabel("Inflection Selectivity", labelpad=15)
                    else:
                        ax.yaxis.set_tick_params(labelleft=False)
                else:
                    # Use per-row ylim if available
                    row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                    yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                    ax.set_ylim(*row_ylim)
                    ax.set_yticks(yticks)
                    if col == 0:
                        ax.yaxis.set_tick_params(labelleft=True)
                        ax.set_yticklabels(ylabels)
                        if row == 0:
                            ax.set_ylabel("Lexeme Accuracy", labelpad=15)
                        elif row == 1:
                            ax.set_ylabel("Inflection Accuracy", labelpad=15)
                    else:
                        ax.yaxis.set_tick_params(labelleft=False)
                
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                
                if row == 0:
                    ax.set_title(titles[col], pad=15)

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=fig_size)
    fig1.subplots_adjust(hspace=0.6)
    axes1 = np.atleast_2d(axes1)
    plot_panel(fig1, axes1, plot_selectivity=False)
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig1.legend(handles, labels, loc="lower center", bbox_to_anchor=bbox_to_anchor,
                    ncol=min(4, len(labels)), mode="expand", frameon=True)
    fig1.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    filename1 = linguistic_filename or f"linguistic_accuracy{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig1.savefig(os.path.join(output_dir, filename1), bbox_inches="tight")
    print(f"Saved linguistic accuracy figure to {os.path.join(output_dir, filename1)}")

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=fig_size)
    # same extra vertical spacing for the selectivity plot
    fig2.subplots_adjust(hspace=0.4)
    axes2 = np.atleast_2d(axes2)
    plot_panel(fig2, axes2, plot_selectivity=True)
    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    if handles2 and labels2:
        fig2.legend(handles2, labels2, loc="lower center", bbox_to_anchor=bbox_to_anchor,
                    ncol=min(4, len(labels2)), mode="expand", frameon=True)
    fig2.tight_layout(rect=[0, 0.05, 1, 0.97])
    filename2 = selectivity_filename or f"classifier_selectivity{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig2.savefig(os.path.join(output_dir, filename2), bbox_inches="tight")
    print(f"Saved selectivity figure to {os.path.join(output_dir, filename2)}")

    regression_df = pd.DataFrame(all_regression_results)
    regression_filepath = os.path.join(output_dir, "all_regression_results.csv")
    regression_df.to_csv(regression_filepath, index=False)
    print(f"Saved all regression results to {regression_filepath}")

# pythia_models = [
#     "pythia-6.9b_step1",
#     "pythia-6.9b_step64",
#     "pythia-6.9b_step6000",
#     "pythia-6.9b_step19000",
#     "pythia-6.9b_step37000",
#     "pythia-6.9b_step57000",
#     "pythia-6.9b_step82000",
#     "pythia-6.9b_step111000",
#     "pythia-6.9b",
# ]
# dataset = "ud_gum_dataset"
# linguistic_filename = "pythia_linguistic_accuracy.png"
# select_filename = "pythia_classifier_selectivity.png"
# plot_linguistic_and_selectivity(
#     dataset,
#     pythia_models,
#     probe_type="nn",
#     pca=False,
#     linguistic_filename=linguistic_filename,
#     selectivity_filename=select_filename,
#     ylim=[(0, 1.0), (0.6, 1.0)],
# )

# olmo_models = [
#     "olmo2-7b_stage1-step5000-tokens21B",
#     "olmo2-7b_stage1-step40000-tokens168B",
#     "olmo2-7b_stage1-step97000-tokens407B",
#     "olmo2-7b_stage1-step179000-tokens751B",
#     "olmo2-7b_stage1-step282000-tokens1183B",
#     "olmo2-7b_stage1-step409000-tokens1716B",
#     "olmo2-7b_stage1-step559000-tokens2345B",
#     "olmo2-7b_stage1-step734000-tokens3079B",
#     "olmo2-7b",
# ]
# linguistic_filename = "olmo_linguistic_accuracy.png"
# select_filename = "olmo_classifier_selectivity.png"
# plot_linguistic_and_selectivity(
#     dataset,
#     olmo_models,
#     probe_type="nn",
#     pca=False,
#     linguistic_filename=linguistic_filename,
#     selectivity_filename=select_filename,
#     ylim=[(0.6, 1.0), (0.6, 1.0)],
# )

# Plot with all models
all_models = [
    "bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
    "gpt2", "gpt2-large", "gpt2-xl",
    "pythia-6.9b",
    "pythia-6.9b-tulu",
    "olmo2-7b", "olmo2-7b-instruct",
    "gemma2b", "gemma2b-it",
    "qwen2", "qwen2-instruct",
    "llama3-8b", "llama3-8b-instruct",
]
dataset = "ud_gum_dataset"
linguistic_filename = "linguistic_accuracy.png"
select_filename = "classifier_selectivity.png"
plot_linguistic_and_selectivity(
    dataset,
    all_models,
    probe_type="nn",
    pca=False,
    linguistic_filename=linguistic_filename,
    selectivity_filename=select_filename,
    ylim=[(0, 1.0), (0.6, 1.0)],
)