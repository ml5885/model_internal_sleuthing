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
    "byt5": "ByT5-Base",
    "mt5": "mT5-Base",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
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
    idx = model_list.index(model)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

def get_model_linestyle(model_key):
    """Return linestyle based on model type."""
    base_model = model_key.split('_')[0] if '_' in model_key else model_key
    if base_model == "byt5":
        return "--"  # dashed line for byt5
    return "-"  # solid line for all other models

def get_acc_columns(df, prefix):
    if f"{prefix}_Accuracy" in df.columns and f"{prefix}_ControlAccuracy" in df.columns:
        return f"{prefix}_Accuracy", f"{prefix}_ControlAccuracy"
    if "Acc" in df.columns and "controlAcc" in df.columns:
        return "Acc", "controlAcc"
    raise ValueError("Could not find accuracy columns in DataFrame.")

def plot_t5_results(model_to_dataset, model_list, output_dir="figures3", filename_prefix=""):
    probe_types = ["reg", "mlp", "rf"]
    titles = ["Linear Regression", "MLP", "Random Forest"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)
    
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
                    ax.set_ylim(0, 1)
                    ax.set_yticks(np.arange(0, 1.1, 0.2))
                    if col == 0:
                        ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.2)])
                        if row == 0:
                            ylabel = "Lexeme Selectivity" if plot_selectivity else "Lexeme Accuracy"
                            ax.set_ylabel(ylabel, labelpad=15)
                        elif row == 1:
                            ylabel = "Inflection Selectivity" if plot_selectivity else "Inflection Accuracy"
                            ax.set_ylabel(ylabel, labelpad=15)
                    else:
                        ax.set_yticklabels([])
                    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                    if row == 1:
                        ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                    if row == 0:
                        ax.set_title(titles[col], pad=15)
                    continue
                
                # Main plotting loop
                for i, model_key in enumerate(model_list):
                    current_dataset = model_to_dataset[model_key]
                    base_model = model_key.split('_')[0]
                    
                    # Check both probes and probes2 directories
                    probe_dirs = [
                        os.path.join("..", "output", "probes", f"{current_dataset}_{base_model}_{task}_{probe}"),
                        os.path.join("..", "output", "probes2", f"{current_dataset}_{base_model}_{task}_{probe}")
                    ]
                    
                    csv_path = None
                    for probe_dir in probe_dirs:
                        potential_path = os.path.join(probe_dir, f"{task}_results.csv")
                        if os.path.exists(potential_path):
                            csv_path = potential_path
                            break
                    
                    if csv_path is None:
                        print(f"[WARN] Missing results for model: {model_key} in both probes and probes2")
                        continue
                        
                    df = pd.read_csv(csv_path)
                    try:
                        acc_col, ctrl_col = get_acc_columns(df, task)
                        df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                        
                        if plot_selectivity:
                            y = df[acc_col] - df[ctrl_col]
                        else:
                            y = df[acc_col]
                            
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model_key, model_key),
                            linewidth=3.0,
                            color=get_model_color(model_key, model_list),
                            linestyle=get_model_linestyle(model_key),
                        )
                    except Exception as e:
                        print(f"Error processing {model_key}: {e}")
                        continue
                
                # Set axis properties
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                
                if plot_selectivity:
                    row_ylim = (-0.3, 0.3) if row == 0 else (0, 0.8)
                else:
                    row_ylim = (0, 1.0)
                    
                ax.set_ylim(*row_ylim)
                
                if col == 0:
                    if row == 0:
                        ylabel = "Lexeme Selectivity" if plot_selectivity else "Lexeme Accuracy"
                        ax.set_ylabel(ylabel, labelpad=15)
                    elif row == 1:
                        ylabel = "Inflection Selectivity" if plot_selectivity else "Inflection Accuracy"
                        ax.set_ylabel(ylabel, labelpad=15)
                
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                if row == 0:
                    ax.set_title(titles[col], pad=15)

    # Plot linguistic accuracy
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes1 = np.atleast_2d(axes1)
    plot_panel(fig1, axes1, plot_selectivity=False)
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig1.legend(handles, labels, loc="lower center", bbox_to_anchor=bbox_to_anchor,
                    ncol=min(4, len(labels)), mode="expand", frameon=True)
    fig1.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    filename1 = f"{filename_prefix}all_languages_linguistic_accuracy.png"
    fig1.savefig(os.path.join(output_dir, filename1), bbox_inches="tight")
    print(f"Saved linguistic accuracy figure to {os.path.join(output_dir, filename1)}")

    # Plot selectivity
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes2 = np.atleast_2d(axes2)
    plot_panel(fig2, axes2, plot_selectivity=True)
    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    if handles2 and labels2:
        fig2.legend(handles2, labels2, loc="lower center", bbox_to_anchor=bbox_to_anchor,
                    ncol=min(4, len(labels2)), mode="expand", frameon=True)
    fig2.tight_layout(rect=[0, 0.05, 1, 0.97])
    filename2 = f"{filename_prefix}all_languages_classifier_selectivity.png"
    fig2.savefig(os.path.join(output_dir, filename2), bbox_inches="tight")
    print(f"Saved selectivity figure to {os.path.join(output_dir, filename2)}")

def generate_t5_markdown_tables(model_to_dataset, model_list, output_dir="figures3"):
    """Generate markdown tables for T5 models across all languages."""
    percentages = [0, 25, 50, 75, 100]
    
    probe_types = ["reg", "mlp"]
    tasks = ["lexeme", "inflection"]
    
    probe_names = {
        "reg": "Linear Regression",
        "mlp": "MLP"
    }
    
    for task in tasks:
        for probe_type in probe_types:
            probe_name = probe_names[probe_type]
            
            print(f"\n## {task.title()} Accuracy - {probe_name}\n")
            
            # Collect all valid CSVs and their layer counts
            valid_models = []
            layer_counts = []
            
            for model_key in model_list:
                current_dataset = model_to_dataset[model_key]
                base_model = model_key.split('_')[0]
                
                # Check both probes and probes2 directories
                probe_dirs = [
                    os.path.join("..", "output", "probes", f"{current_dataset}_{base_model}_{task}_{probe_type}"),
                    os.path.join("..", "output", "probes2", f"{current_dataset}_{base_model}_{task}_{probe_type}")
                ]
                
                csv_found = False
                for probe_dir in probe_dirs:
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            valid_models.append((model_key, csv_path))
                            layer_counts.append(len(df))
                            csv_found = True
                            break
                        except:
                            continue
                
                if not csv_found:
                    print(f"[DEBUG] No CSV found for {model_key} in either probes or probes2")
            
            if not valid_models:
                print("No valid CSV files found for this task")
                continue
            
            # Use the most common layer count for header generation
            from collections import Counter
            most_common_layers = Counter(layer_counts).most_common(1)[0][0]
            
            layer_indices = [int(most_common_layers * p / 100) for p in percentages]
            layer_indices[-1] = most_common_layers - 1
            
            headers = []
            for i, (layer_idx, pct) in enumerate(zip(layer_indices, percentages)):
                if pct == 0:
                    headers.append(f"Layer {layer_idx} (first)")
                elif pct == 100:
                    headers.append(f"Layer {layer_idx} (last)")
                else:
                    headers.append(f"Layer {layer_idx}")
            
            header_row = "| Model | Dataset | " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join([" --- "] * (len(headers) + 2)) + "|"
            
            print(header_row)
            print(separator_row)
            
            for model_key in model_list:
                current_dataset = model_to_dataset[model_key]
                base_model = model_key.split('_')[0]
                
                language = model_names.get(model_key, model_key)
                
                # Check both probes and probes2 directories
                probe_dirs = [
                    os.path.join("..", "output", "probes", f"{current_dataset}_{base_model}_{task}_{probe_type}"),
                    os.path.join("..", "output", "probes2", f"{current_dataset}_{base_model}_{task}_{probe_type}")
                ]
                
                csv_path = None
                for probe_dir in probe_dirs:
                    potential_path = os.path.join(probe_dir, f"{task}_results.csv")
                    if os.path.exists(potential_path):
                        csv_path = potential_path
                        break
                
                if csv_path is None:
                    row_data = ["N/A"] * len(percentages)
                else:
                    try:
                        df = pd.read_csv(csv_path)
                        acc_col, _ = get_acc_columns(df, task)
                        
                        actual_n_layers = len(df)
                        actual_layer_indices = [int(actual_n_layers * p / 100) for p in percentages]
                        actual_layer_indices[-1] = actual_n_layers - 1
                        
                        row_data = []
                        for layer_idx in actual_layer_indices:
                            if 0 <= layer_idx < len(df):
                                accuracy = df.iloc[layer_idx][acc_col]
                                row_data.append(f"{accuracy:.3f}")
                            else:
                                row_data.append("N/A")
                    except Exception as e:
                        print(f"[WARN] Error processing {model_key} {task} {probe_type}: {e}")
                        row_data = ["N/A"] * len(percentages)
                
                row = f"| {language} | {current_dataset} | " + " | ".join(row_data) + " |"
                print(row)
            
            print()

if __name__ == "__main__":
    all_models = []
    model_to_dataset = {}
    
    language_datasets = [
        ("ud_gum_dataset", "English"),
        ("ud_zh_gsd_dataset", "Chinese"), 
        ("ud_de_gsd_dataset", "German"),
        ("ud_fr_gsd_dataset", "French"),
        ("ud_ru_syntagrus_dataset", "Russian"),
        ("ud_tr_imst_dataset", "Turkish"),
    ]
    
    models_to_plot = [
        "byt5",
        "mt5",
        # "qwen2",
        # "qwen2-instruct",
        # "qwen2.5-7B",
        # "qwen2.5-7B-instruct",
    ]
    
    if "qwen2" in model_names:
        bbox_to_anchor = (0, -0.2, 1, 0.1)
    
    for model in models_to_plot:
        for dataset, lang in language_datasets:
            model_key = f"{model}_{lang.lower()}"
            all_models.append(model_key)
            model_to_dataset[model_key] = dataset
            model_names[model_key] = f"{model_names[model]} ({lang})"
    
    print("Plotting multilingual models across all languages...")
    plot_t5_results(model_to_dataset, all_models, filename_prefix="t5_")
    
    print("\nGenerating markdown tables for multilingual models...")
    generate_t5_markdown_tables(model_to_dataset, all_models)