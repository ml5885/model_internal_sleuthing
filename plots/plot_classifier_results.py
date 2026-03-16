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

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 100
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 28,
    "axes.titlesize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
    "legend.title_fontsize": 24,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0,
    "figure.constrained_layout.h_pad": 0.1
})

bbox_to_anchor = (0, -0.11, 1, 0.1)
PLOT_SHADING = False

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

def get_values_at_normalized_layers(df, y_values, percentiles=None):
    """
    Given a DataFrame with a Layer_Normalized column and a corresponding
    series/array of y_values (e.g., accuracies), interpolate the values at
    the requested percentiles in [0, 1].
    """
    if percentiles is None:
        percentiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    if "Layer_Normalized" not in df.columns or len(df) == 0:
        return [np.nan] * len(percentiles)
    df_sorted = df.sort_values("Layer_Normalized")
    x = df_sorted["Layer_Normalized"].values
    y = np.asarray(y_values)[df_sorted.index]
    # Guard against degenerate cases
    if len(np.unique(x)) == 1:
        return [float(y.mean())] * len(percentiles)
    return list(np.interp(percentiles, x, y))

def print_accuracy_and_selectivity_tables(
    dataset: str,
    model_list: list[str],
    pca: bool = False,
    pca_dim: int = 50,
    percentiles=None,
):
    """
    Print tables of accuracies and selectivities at selected normalized layer
    positions for all models / tasks / probe types used in the main plots.
    Each row corresponds to a (model, task, probe) combination and columns
    correspond to 0%, 25%, 50%, 75%, 100% normalized layer numbers.
    """
    if percentiles is None:
        percentiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    percent_labels = [f"{int(p * 100)}%" for p in percentiles]
    probe_types = ["reg", "nn"]
    tasks = ["lexeme", "inflection"]

    accuracy_rows = []
    selectivity_rows = []

    for task in tasks:
        for probe in probe_types:
            for model in model_list:
                probe_dir = os.path.join(
                    "..", "output", "probes", f"{dataset}_{model}_{task}_{probe}"
                )
                if pca:
                    probe_dir += f"_pca_{pca_dim}"
                csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                if not os.path.exists(csv_path):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    acc_col, ctrl_col = get_acc_columns(df, task)
                    df["Layer_Normalized"] = (
                        df["Layer"] - df["Layer"].min()
                    ) / (df["Layer"].max() - df["Layer"].min())
                    # Accuracy table
                    acc_vals = get_values_at_normalized_layers(df, df[acc_col], percentiles)
                    acc_row = {
                        "Model": model_names.get(model, model),
                        "Task": task.capitalize(),
                        "Probe": "Linear" if probe == "reg" else "MLP",
                    }
                    acc_row.update({label: val for label, val in zip(percent_labels, acc_vals)})
                    accuracy_rows.append(acc_row)
                    # Selectivity table (if control column is present)
                    if ctrl_col in df.columns:
                        sel_series = df[acc_col] - df[ctrl_col]
                        sel_vals = get_values_at_normalized_layers(
                            df, sel_series, percentiles
                        )
                        sel_row = {
                            "Model": model_names.get(model, model),
                            "Task": task.capitalize(),
                            "Probe": "Linear" if probe == "reg" else "MLP",
                        }
                        sel_row.update(
                            {label: val for label, val in zip(percent_labels, sel_vals)}
                        )
                        selectivity_rows.append(sel_row)
                except Exception:
                    continue

    if accuracy_rows:
        acc_df = pd.DataFrame(accuracy_rows)
        acc_df = acc_df.sort_values(["Model", "Task", "Probe"])
        print("\n=== Accuracy table (values at selected normalized layer percentages) ===")
        print(acc_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\n[INFO] No accuracy data available to print tables.")

    if selectivity_rows:
        sel_df = pd.DataFrame(selectivity_rows)
        sel_df = sel_df.sort_values(["Model", "Task", "Probe"])
        print("\n=== Selectivity table (values at selected normalized layer percentages) ===")
        print(sel_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\n[INFO] No selectivity data available to print tables.")

def print_rf_accuracy_and_selectivity_tables(
    dataset: str,
    model_list: list[str],
    pca: bool = False,
    pca_dim: int = 50,
    percentiles=None,
):
    """
    Print tables of RF accuracies and selectivities (inflection task only)
    at selected normalized layer positions for all models.
    """
    if percentiles is None:
        percentiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    percent_labels = [f"{int(p * 100)}%" for p in percentiles]
    task = "inflection"
    probe = "rf"

    accuracy_rows = []
    selectivity_rows = []

    for model in model_list:
        probe_dir = os.path.join(
            "..", "output", "probes", f"{dataset}_{model}_{task}_{probe}"
        )
        if pca:
            probe_dir += f"_pca_{pca_dim}"
        csv_path = os.path.join(probe_dir, f"{task}_results.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            acc_col, ctrl_col = get_acc_columns(df, task)
            df["Layer_Normalized"] = (
                df["Layer"] - df["Layer"].min()
            ) / (df["Layer"].max() - df["Layer"].min())
            # Accuracy table
            acc_vals = get_values_at_normalized_layers(df, df[acc_col], percentiles)
            acc_row = {
                "Model": model_names.get(model, model),
                "Task": task.capitalize(),
                "Probe": "RF",
            }
            acc_row.update({label: val for label, val in zip(percent_labels, acc_vals)})
            accuracy_rows.append(acc_row)
            # Selectivity table (if control column is present)
            if ctrl_col in df.columns:
                sel_series = df[acc_col] - df[ctrl_col]
                sel_vals = get_values_at_normalized_layers(
                    df, sel_series, percentiles
                )
                sel_row = {
                    "Model": model_names.get(model, model),
                    "Task": task.capitalize(),
                    "Probe": "RF",
                }
                sel_row.update(
                    {label: val for label, val in zip(percent_labels, sel_vals)}
                )
                selectivity_rows.append(sel_row)
        except Exception:
            continue

    if accuracy_rows:
        acc_df = pd.DataFrame(accuracy_rows)
        acc_df = acc_df.sort_values(["Model", "Task", "Probe"])
        print("\n=== RF Accuracy table (inflection, selected normalized layer percentages) ===")
        print(acc_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\n[INFO] No RF accuracy data available to print tables.")

    if selectivity_rows:
        sel_df = pd.DataFrame(selectivity_rows)
        sel_df = sel_df.sort_values(["Model", "Task", "Probe"])
        print("\n=== RF Selectivity table (inflection, selected normalized layer percentages) ===")
        print(sel_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\n[INFO] No RF selectivity data available to print tables.")

def plot_linguistic_and_selectivity(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    save_name="linguistic_selectivity",
    pca: bool = False,
    pca_dim: int = 50,
    linguistic_filename: str = None,
    selectivity_filename: str = None,
    ylim: tuple = ((0, 1.0), (0, 1.0)),
):
    probe_types = ["reg", "nn"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows = len(tasks)
    # Use 5 columns: 2 for accuracy, 1 spacer, 2 for selectivity
    n_cols = 5
    all_regression_results = []

    aspect_ratio, base_height = 2.4, 5
    # Adjust fig_width for the spacer column (approx 0.2 width of a plot)
    fig_width = (4.2) * base_height * aspect_ratio / 2.0
    fig_height = n_rows * base_height
    fig_size = (fig_width, fig_height)

    # Adjust gridspec_kw to reduce spacing between columns, spacer column handles the group gap
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, 
                             gridspec_kw={'wspace': 0.1, 'width_ratios': [1, 1, 0.15, 1, 1]})
    axes = np.atleast_2d(axes)

    def plot_panel(fig, axes):
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                # Spacer column
                if col == 2:
                    axes[row, col].axis('off')
                    continue

                # Map 5 columns back to logical probe/selectivity config
                if col < 2:
                    probe_idx = col
                    sel = 0
                    probe = probe_types[probe_idx]
                else:
                    probe_idx = col - 3 # cols 3,4 -> 0,1
                    sel = 1
                    probe = probe_types[probe_idx]

                ax = axes[row, col]
                for i, model in enumerate(model_list):
                    probe_dir = os.path.join("..", "output", "probes",
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
                        if sel == 0:
                            y = df[acc_col]
                            fit_and_store_regression(df, model, task, probe, all_regression_results)
                        else:
                            y = df[acc_col] - df[ctrl_col]
                        model_color = get_model_color(model, model_list)
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=4.0,
                            color=model_color,
                        )
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes)
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0", "25", "50", "75", "100"])
                if sel == 1:
                    row_ylim = (0, 0.8)
                    ylabel = "Lemma Selectivity" if row == 0 else "Inflection Selectivity"
                else:
                    row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                    ylabel = "Lemma Accuracy" if row == 0 else "Inflection Accuracy"
                yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                ax.set_ylim(*row_ylim)
                ax.set_yticks(yticks)
                if col == 0 or col == 3: # Leftmost of each group (0 and 3)
                    ax.yaxis.set_tick_params(labelleft=True)
                    ax.set_yticklabels(ylabels)
                    ax.set_ylabel(ylabel, labelpad=15, fontsize=24)
                else:
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.set_yticklabels([])
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
                if row == 1:
                    pass
                    # ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                if row == 0:
                    # Title mapping: col 0->0, 1->1, 3->0, 4->1
                    title_idx = probe_idx
                    ax.set_title(f"{titles[title_idx]}", pad=10, loc='center')
                
                if row == 1:
                    ax.tick_params(labelbottom=True)
                else:
                    ax.tick_params(labelbottom=False)

    plot_panel(fig, axes)
    
    # Align y-labels
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 3])
    
    # Add shared x-axis labels
    fig.text(0.32, 0.0, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])
    fig.text(0.72, 0.0, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])

    # Filter out the handles/labels from the spacer axes if any (though axis off handles it usually)
    handles_labels = []
    for ax in axes.flatten():
        if ax.lines: # Only look at axes with plots
             handles_labels.append(ax.get_legend_handles_labels())
             
    handles = sum([hl[0] for hl in handles_labels], [])
    labels = sum([hl[1] for hl in handles_labels], [])
    seen = set()
    legend_items = []
    for h, l in zip(handles, labels):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    if legend_items:
        handles, labels = zip(*legend_items)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(5, len(labels)),
            frameon=True,
            fontsize=18,
            columnspacing=1.0,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5, h_pad=0.5)
    os.makedirs(output_dir, exist_ok=True)
    filename = linguistic_filename or f"linguistic_and_selectivity{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"Saved combined linguistic/selectivity figure to {os.path.join(output_dir, filename)}")

    regression_df = pd.DataFrame(all_regression_results)
    regression_filepath = os.path.join(output_dir, "all_regression_results.csv")
    regression_df.to_csv(regression_filepath, index=False)
    print(f"Saved all regression results to {regression_filepath}")

def plot_rf_only(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    linguistic_filename="rf_linguistic_accuracy.png",
    selectivity_filename="rf_classifier_selectivity.png",
    pca: bool = False,
    pca_dim: int = 50,
    ylim: tuple = ((0, 1.0), (0, 1.0)),
):
    probe = "rf"
    tasks = ["inflection"]
    n_rows, n_cols = len(tasks), 2
    all_regression_results = []
    aspect_ratio, base_height = 3.5, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = 2 * base_height
    fig_size = (fig_width, fig_height)
    titles = ["Random Forest Accuracy", "Random Forest Selectivity"]

    def plot_panel(fig, axes):
        axes = np.atleast_2d(axes)
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                ax = axes[row, col]
                plot_selectivity = (col == 1)
                # Main plotting loop for inflection only
                for i, model in enumerate(model_list):
                    probe_dir = os.path.join("..", "output", "probes",
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
                        if plot_selectivity:
                            # If selectivity columns are missing, skip plotting
                            if ctrl_col not in df.columns:
                                ax.text(0.5, 0.5, "No selectivity data", ha="center", va="center",
                                        transform=ax.transAxes)
                                continue
                            y = df[acc_col] - df[ctrl_col]
                        else:
                            y = df[acc_col]
                        model_color = get_model_color(model, model_list)
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=4.0,
                            color=model_color,
                        )
                        if not plot_selectivity:
                            fit_and_store_regression(df, model, task, probe, all_regression_results)
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes)

                # Set axis properties
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0", "25", "50", "75", "100"])
                # Set y-limits: inflection accuracy (col 0) and selectivity (col 1)
                if col == 0:
                    row_ylim = (0, 1.0)
                    ylabel = "Inflection Accuracy"
                    yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                    ax.set_ylim(*row_ylim)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(ylabels)
                    ax.set_ylabel(ylabel, labelpad=30)
                    ax.yaxis.set_tick_params(labelleft=True)
                else:
                    row_ylim = (0, 0.8)
                    ylabel = "Inflection Selectivity"
                    yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                    ax.set_ylim(*row_ylim)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(ylabels)
                    ax.set_ylabel(ylabel, labelpad=30)
                    ax.yaxis.set_tick_params(labelleft=True)
                yticks, _ = get_tick_values(row_ylim[0], row_ylim[1])
                ax.set_ylim(*row_ylim)
                ax.set_yticks(yticks)
                ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
                ax.set_title(titles[col], pad=15)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = np.atleast_2d(axes)
    plot_panel(fig, axes)
    
    # Align y-labels (only one row, but good for consistency if we had more)
    if n_rows > 1:
        fig.align_ylabels(axes[:, 0])
        fig.align_ylabels(axes[:, 1])
    # Reduce horizontal spacing between plots
    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5, h_pad=1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(5, len(labels)),
            frameon=True,
            fontsize=18,
            columnspacing=1.0,
        )
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, linguistic_filename), bbox_inches="tight")
    print(f"Saved RF combined accuracy/selectivity figure to {os.path.join(output_dir, linguistic_filename)}")

    regression_df = pd.DataFrame(all_regression_results)
    regression_filepath = os.path.join(output_dir, "rf_regression_results.csv")
    regression_df.to_csv(regression_filepath, index=False)
    print(f"Saved RF regression results to {regression_filepath}")

def plot_linguistic_accuracy(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    save_name="linguistic_accuracy",
    pca: bool = False,
    pca_dim: int = 50,
    linguistic_filename: str = None,
    ylim: tuple = ((0, 1.0), (0, 1.0)),
):
    probe_types = ["reg", "nn"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)
    all_regression_results = []

    aspect_ratio, base_height = 2.0, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = n_rows * base_height
    fig_size = (fig_width, fig_height)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = np.atleast_2d(axes)

    def plot_panel(fig, axes):
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                probe = probe_types[col]
                ax = axes[row, col]
                for i, model in enumerate(model_list):
                    probe_dir = os.path.join("..", "output", "probes",
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
                        y = df[acc_col]
                        fit_and_store_regression(df, model, task, probe, all_regression_results)
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=4.0,
                            color=get_model_color(model, model_list),
                        )
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes)
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0", "25", "50", "75", "100"])
                row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                ylabel = "Lemma Accuracy" if row == 0 else "Inflection Accuracy"
                yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                ax.set_ylim(*row_ylim)
                ax.set_yticks(yticks)
                if col == 0:
                    ax.yaxis.set_tick_params(labelleft=True)
                    ax.set_yticklabels(ylabels)
                    ax.set_ylabel(ylabel, labelpad=30)
                else:
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.set_yticklabels([])
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                if row == 0:
                    ax.set_title(f"{titles[col]}", pad=15, loc='center')
    plot_panel(fig, axes)
    
    # Align y-labels
    fig.align_ylabels(axes[:, 0])
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flatten()]
    handles = sum([hl[0] for hl in handles_labels], [])
    labels = sum([hl[1] for hl in handles_labels], [])
    seen = set()
    legend_items = []
    for h, l in zip(handles, labels):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    if legend_items:
        handles, labels = zip(*legend_items)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(5, len(labels)),
            frameon=True,
            fontsize=18,
            columnspacing=1.0,
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5)
    os.makedirs(output_dir, exist_ok=True)
    filename = linguistic_filename or f"linguistic_accuracy{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"Saved linguistic accuracy figure to {os.path.join(output_dir, filename)}")

    regression_df = pd.DataFrame(all_regression_results)
    regression_filepath = os.path.join(output_dir, "linguistic_accuracy_regression_results.csv")
    regression_df.to_csv(regression_filepath, index=False)
    print(f"Saved linguistic accuracy regression results to {regression_filepath}")

def plot_selectivity(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    save_name="classifier_selectivity",
    pca: bool = False,
    pca_dim: int = 50,
    selectivity_filename: str = None,
    ylim: tuple = ((0, 0.2), (0, 0.8)),
):
    probe_types = ["reg", "nn"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)

    aspect_ratio, base_height = 2.0, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = n_rows * base_height
    fig_size = (fig_width, fig_height)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = np.atleast_2d(axes)

    def plot_panel(fig, axes):
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                probe = probe_types[col]
                ax = axes[row, col]
                for i, model in enumerate(model_list):
                    probe_dir = os.path.join("..", "output", "probes",
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
                        y = df[acc_col] - df[ctrl_col]
                        model_color = get_model_color(model, model_list)
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=4.0,
                            color=model_color,
                        )
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes)
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0", "25", "50", "75", "100"])
                row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                ylabel = "Lemma Selectivity" if row == 0 else "Inflection Selectivity"
                yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                ax.set_ylim(*row_ylim)
                ax.set_yticks(yticks)
                if col == 0:
                    ax.yaxis.set_tick_params(labelleft=True)
                    ax.set_yticklabels(ylabels)
                    ax.set_ylabel(ylabel, labelpad=30)
                else:
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.set_yticklabels([])
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                if row == 0:
                    ax.set_title(f"{titles[col]}", pad=15, loc='center')

    plot_panel(fig, axes)

    # Align y-labels
    fig.align_ylabels(axes[:, 0])
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flatten()]
    handles = sum([hl[0] for hl in handles_labels], [])
    labels = sum([hl[1] for hl in handles_labels], [])
    seen = set()
    legend_items = []
    for h, l in zip(handles, labels):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    if legend_items:
        handles, labels = zip(*legend_items)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(5, len(labels)),
            frameon=True,
            fontsize=18,
            columnspacing=1.0,
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5)
    os.makedirs(output_dir, exist_ok=True)
    filename = selectivity_filename or f"classifier_selectivity{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"Saved classifier selectivity figure to {os.path.join(output_dir, filename)}")


def plot_grouped_classifier_results(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    filename_prefix="grouped_",
    rf_only=False,
    pca: bool = False,
    pca_dim: int = 50,
    ylim: tuple = ((0, 1.0), (0, 1.0)),
):
    """
    Plot averaged accuracy and selectivity for models grouped by type/size.
    Groups:
    1. Encoder-only (BERT, DeBERTa)
    2. Small Decoder-only (< 5B)
    3. Large Decoder-only (> 5B)
    """
    if rf_only:
        probe_types = ["rf"]
        titles = ["Random Forest Accuracy", "Random Forest Selectivity"]
        tasks = ["inflection"]
        n_rows, n_cols = 1, 2
        aspect_ratio, base_height = 3.5, 5
        fig_width = n_cols * base_height * aspect_ratio / 2.0
        fig_height = 2 * base_height
        fig_size = (fig_width, fig_height)
    else:
        probe_types = ["reg", "nn"]
        titles = ["Linear Regression", "MLP"]
        tasks = ["lexeme", "inflection"]
        n_rows, n_cols = len(tasks), len(probe_types) * 2
        aspect_ratio, base_height = 2.3, 5.0
        fig_width = n_cols * base_height * aspect_ratio / 2.0
        fig_height = n_rows * base_height
        fig_size = (fig_width, fig_height)

    # Define groups
    groups = {
        "Encoder-only": ["bert-base-uncased", "bert-large-uncased", "deberta-v3-large"],
        "Small Decoder (< 5B)": ["gpt2", "gpt2-large", "gpt2-xl", "qwen2", "qwen2-instruct", "gemma2b", "gemma2b-it", "pythia1.4b"],
        "Large Decoder (> 5B)": ["pythia-6.9b", "pythia-6.9b-tulu", "olmo2-7b", "olmo2-7b-instruct", "llama3-8b", "llama3-8b-instruct"]
    }
    
    group_colors = {
        "Encoder-only": "#1f77b4", # Blue
        "Small Decoder (< 5B)": "#ff7f0e", # Orange
        "Large Decoder (> 5B)": "#2ca02c"  # Green
    }
    
    # Assign models to groups
    model_to_group = {}
    for group_name, members in groups.items():
        for m in members:
            model_to_group[m] = group_name

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
    axes = np.atleast_2d(axes)

    common_x = np.linspace(0, 1, 101)
    
    # Plot panel
    for row, task in enumerate(tasks):
        for col in range(n_cols):
            if rf_only:
                probe = "rf"
                plot_selectivity = (col == 1)
                ax = axes[0, col]
            else:
                if col < 2:
                    probe = probe_types[col]
                    plot_selectivity = False
                else:
                    probe = probe_types[col - 2]
                    plot_selectivity = True
                ax = axes[row, col]

            # Iterate over groups
            for group_name, members in groups.items():
                ys = []
                for model in members:
                    if model not in model_list and model not in model_to_group: 
                        # Only skip if not in model_list AND not meant to be searched? 
                        # Actually we should search if it's in the input list.
                        if model not in model_list:
                            continue

                    probe_dir = os.path.join("..", "output", "probes",
                                f"{dataset}_{model}_{task}_{probe}")
                    if pca:
                        probe_dir += f"_pca_{pca_dim}"
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    
                    if not os.path.exists(csv_path):
                        continue
                        
                    try:
                        df = pd.read_csv(csv_path)
                        df.sort_values("Layer", inplace=True)
                        acc_col, ctrl_col = get_acc_columns(df, task)
                        df["Layer_Normalized"] = (
                            df["Layer"] - df["Layer"].min()
                        ) / (df["Layer"].max() - df["Layer"].min())
                        
                        if plot_selectivity:
                             if ctrl_col not in df.columns:
                                continue
                             y_vals = df[acc_col] - df[ctrl_col]
                        else:
                            y_vals = df[acc_col]
                            
                        # Interpolate
                        y_interp = np.interp(common_x, df["Layer_Normalized"], y_vals)
                        ys.append(y_interp)
                    except Exception:
                        continue
                
                if ys:
                    avg_y = np.mean(ys, axis=0)
                    min_y = np.min(ys, axis=0)
                    max_y = np.max(ys, axis=0)
                    ax.plot(
                        common_x, avg_y,
                        label=group_name,
                        linewidth=4.0,
                        color=group_colors[group_name],
                    )
                    if PLOT_SHADING:
                        ax.fill_between(common_x, min_y, max_y, color=group_colors[group_name], alpha=0.2)

            # Axis formatting (copied from plot_linguistic_and_selectivity / plot_rf_only)
            ax.tick_params(axis='both', which='major', length=10, width=2)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=28)
            
            if rf_only:
                 if col == 0:
                    row_ylim = (0, 1.0)
                    ylabel = "Inflection Accuracy"
                 else:
                    row_ylim = (0, 0.8)
                    ylabel = "Inflection Selectivity"
            else:
                if plot_selectivity:
                    row_ylim = (0, 0.8)
                    ylabel = "Lemma Selectivity" if row == 0 else "Inflection Selectivity"
                else:
                    row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                    ylabel = "Lemma Accuracy" if row == 0 else "Inflection Accuracy"

            yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
            ax.set_ylim(*row_ylim)
            ax.set_yticks(yticks)
            
            if (not rf_only and (col == 0 or col == 2)) or (rf_only and col == 0):
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_yticklabels(ylabels, fontsize=28)
                ax.set_ylabel(ylabel, labelpad=30)
            elif rf_only and col == 1:
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_yticklabels(ylabels, fontsize=28)
                ax.set_ylabel(ylabel, labelpad=30)
            else:
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_yticklabels([])
            
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
            
            if (rf_only and True) or (not rf_only and row == 1):
                pass
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if not rf_only and row == 0:
                title_idx = col % 2
                ax.set_title(f"{titles[title_idx]}", pad=10, loc='center')
            elif rf_only:
                ax.set_title(titles[col], pad=15)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(by_label)),
            frameon=True,
        )
    
    # Add shared x-axis labels
    fig.text(0.25, -0.03, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])
    fig.text(0.75, -0.03, "Normalized layer number (%)", ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])
    
    # Align y-labels
    if not rf_only:
        fig.align_ylabels(axes[:, 0])
        fig.align_ylabels(axes[:, 2])
    elif rf_only and n_rows > 1:
         fig.align_ylabels(axes[:, 0])
         fig.align_ylabels(axes[:, 1])
    
    if rf_only:
        fig.tight_layout(rect=[0, 0.18, 1, 1], w_pad=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    if rf_only:
        filename = f"{filename_prefix}rf_accuracy_selectivity.png"
    else:
        filename = f"{filename_prefix}linguistic_and_selectivity{'_pca_' + str(pca_dim) if pca else ''}.png"
        
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"Saved grouped figure to {os.path.join(output_dir, filename)}")



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
linguistic_filename = "linguistic_and_selectivity.png"
plot_linguistic_and_selectivity(
    dataset,
    all_models,
    pca=False,
    linguistic_filename=linguistic_filename,
    selectivity_filename=None,
    ylim=[(0.4, 1.0), (0.4, 1.0)],
)

linguistic_accuracy_filename = "linguistic_accuracy.png"
plot_linguistic_accuracy(
    dataset,
    all_models,
    pca=False,
    linguistic_filename=linguistic_accuracy_filename,
    ylim=[(0.4, 1.0), (0.4, 1.0)],
)

selectivity_filename = "classifier_selectivity.png"
plot_selectivity(
    dataset,
    all_models,
    pca=False,
    selectivity_filename=selectivity_filename,
    ylim=[(0, 0.8), (0, 0.8)],
)

# Plot with all models for random forest only (now combined plot)
rf_combined_filename = "rf_combined_accuracy_selectivity.png"
plot_rf_only(
    dataset,
    all_models,
    pca=False,
    linguistic_filename=rf_combined_filename,
    selectivity_filename=None,  # Not used anymore
    ylim=[(0, 1.0), (0, 1.0)],
)

# Print tabular summaries of accuracies and selectivities at key normalized layers
print_accuracy_and_selectivity_tables(
    dataset,
    all_models,
    pca=False,
)
print_rf_accuracy_and_selectivity_tables(
    dataset,
    all_models,
    pca=False,
)

# Grouped plots
print("Plotting grouped classifier results...")
plot_grouped_classifier_results(
    dataset,
    all_models,
    filename_prefix="grouped_",
    ylim=[(0.4, 1.0), (0.4, 1.0)],
)
plot_grouped_classifier_results(
    dataset,
    all_models,
    filename_prefix="grouped_",
    rf_only=True,
    ylim=[(0, 1.0), (0, 1.0)],
)

