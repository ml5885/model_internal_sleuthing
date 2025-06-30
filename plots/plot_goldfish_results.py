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
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 22,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0
})

bbox_to_anchor = (0, -0.11, 1, 0.1)

model_names = {
    "goldfish_eng_latn_1000mb": "Goldfish English (eng_latn_1000mb)",
    "goldfish_zho_hans_1000mb": "Goldfish Chinese (zho_hans_1000mb)",
    "goldfish_deu_latn_1000mb": "Goldfish German (deu_latn_1000mb)",
    "goldfish_fra_latn_1000mb": "Goldfish French (fra_latn_1000mb)",
    "goldfish_rus_cyrl_1000mb": "Goldfish Russian (rus_cyrl_1000mb)",
    "goldfish_tur_latn_1000mb": "Goldfish Turkish (tur_latn_1000mb)",
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
    dataset: str | dict,
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
    fig_size = (n_cols * base_height * aspect_ratio / n_rows, n_rows * base_height)

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
                                ax.set_ylabel("Lexeme Selectivity", labelpad=15, fontsize=24)
                            elif row == 1:
                                ax.set_ylabel("Inflection Selectivity", labelpad=15, fontsize=24)
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
                    current_dataset = dataset.get(model, dataset) if isinstance(dataset, dict) else dataset
                    probe_dir = os.path.join("..", "output", "probes",
                                f"{current_dataset}_{model}_{task}_{probe}")
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
                    row_ylim = (-0.3, 0.3) if row == 0 else (0, 0.8)
                    yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                    ax.set_ylim(*row_ylim)
                    ax.set_yticks(yticks)
                    if col == 0:
                        ax.yaxis.set_tick_params(labelleft=True)
                        ax.set_yticklabels(ylabels)
                        if row == 0:
                            ax.set_ylabel("Lexeme Selectivity", labelpad=15, fontsize=24)
                        elif row == 1:
                            ax.set_ylabel("Inflection Selectivity", labelpad=15, fontsize=24)
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

def generate_markdown_tables(
    dataset_dict: dict,
    model_list: list[str],
    output_dir: str = "figures3",
    n_layers: int = 12
):
    """
    Generate markdown tables showing accuracy at 0%, 25%, 50%, 75%, 100% layer depths.
    Creates 4 tables: lexical accuracy (MLP & reg) and inflectional accuracy (MLP & reg).
    """
    # Define the layer percentages and corresponding layer indices
    percentages = [0, 25, 50, 75, 100]
    layer_indices = [int(n_layers * p / 100) for p in percentages]
    layer_indices[-1] = n_layers - 1  # Ensure last layer is exactly the final layer
    
    # Define probe types and tasks
    # probe_types = ["nn", "reg"]
    probe_types = [ "reg" ]
    tasks = ["lexeme", "inflection"]
    
    # Column headers - simplified layer names
    headers = []
    for i, (layer_idx, pct) in enumerate(zip(layer_indices, percentages)):
        if pct == 0:
            headers.append(f"Layer {layer_idx} (first layer)")
        elif pct == 100:
            headers.append(f"Layer {layer_idx} (last layer)")
        else:
            headers.append(f"Layer {layer_idx}")
    
    for task in tasks:
        for probe_type in probe_types:
            probe_name = "MLP" if probe_type == "nn" else "Linear Regression"
            
            print(f"\n## {task.title()} Accuracy - {probe_name}\n")
            
            # Table header
            header_row = "| Goldfish Model | Dataset | " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join([" --- "] * (len(headers) + 2)) + "|"
            
            print(header_row)
            print(separator_row)
            
            # Data rows
            for model in model_list:
                current_dataset = dataset_dict.get(model, dataset_dict) if isinstance(dataset_dict, dict) else dataset_dict
                
                # Get language name from model_names or extract from model key
                if model in model_names:
                    language = model_names[model]  # Full goldfish model name
                else:
                    language = f"Goldfish {model}"
                
                # Get dataset name
                dataset_name = current_dataset if isinstance(current_dataset, str) else current_dataset
                
                probe_dir = os.path.join("..", "output", "probes",
                            f"{current_dataset}_{model}_{task}_{probe_type}")
                csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                
                if not os.path.exists(csv_path):
                    # Fill with N/A if no data
                    row_data = ["N/A"] * len(layer_indices)
                else:
                    try:
                        df = pd.read_csv(csv_path)
                        acc_col, _ = get_acc_columns(df, task)
                        
                        row_data = []
                        for layer_idx in layer_indices:
                            if layer_idx < len(df):
                                accuracy = df.iloc[layer_idx][acc_col]
                                row_data.append(f"{accuracy:.3f}")
                            else:
                                row_data.append("N/A")
                                
                    except Exception as e:
                        print(f"[WARN] Error processing {model} {task} {probe_type}: {e}")
                        row_data = ["N/A"] * len(layer_indices)
                
                # Create table row
                row = f"| {language} | {dataset_name} | " + " | ".join(row_data) + " |"
                print(row)
            
            print()  # Add spacing between tables

# Plot for goldfish models
goldfish_datasets = {
    "goldfish_eng_latn_1000mb": ("ud_gum_dataset", "English"),
    "goldfish_zho_hans_1000mb": ("ud_zh_gsd_dataset", "Chinese"),
    "goldfish_deu_latn_1000mb": ("ud_de_gsd_dataset", "German"),
    "goldfish_fra_latn_1000mb": ("ud_fr_gsd_dataset", "French"),
    "goldfish_rus_cyrl_1000mb": ("ud_ru_syntagrus_dataset", "Russian"),
    "goldfish_tur_latn_1000mb": ("ud_tr_imst_dataset", "Turkish"),
}

all_goldfish_models = list(goldfish_datasets.keys())
model_to_dataset = {model: data[0] for model, data in goldfish_datasets.items()}

# print("Plotting for all Goldfish models together")
# linguistic_filename = "goldfish_all_linguistic_accuracy.png"
# select_filename = "goldfish_all_classifier_selectivity.png"
# plot_linguistic_and_selectivity(
#     dataset=model_to_dataset,
#     model_list=all_goldfish_models,
#     probe_type="nn",
#     pca=False,
#     linguistic_filename=linguistic_filename,
#     selectivity_filename=select_filename,
#     ylim=[(0, 1.0), (0, 1.0)],
# )

# Generate markdown tables for goldfish models
print("\nGenerating markdown tables for Goldfish models...")
generate_markdown_tables(
    dataset_dict=model_to_dataset,
    model_list=all_goldfish_models,
    output_dir="figures3",
    n_layers=12
)