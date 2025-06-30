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
    "legend.fontsize": 18,
    "legend.title_fontsize": 22,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0
})

bbox_to_anchor = (0, -0.28, 1, 0.1)

model_names = {
    "byt5": "ByT5-Base",
    "mt5": "mT5-Base",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
    "goldfish_eng_latn_1000mb": "Goldfish English (1GB)",
    "goldfish_zho_hans_1000mb": "Goldfish Chinese (1GB)",
    "goldfish_deu_latn_1000mb": "Goldfish German (1GB)",
    "goldfish_fra_latn_1000mb": "Goldfish French (1GB)",
    "goldfish_rus_cyrl_1000mb": "Goldfish Russian (1GB)",
    "goldfish_tur_latn_1000mb": "Goldfish Turkish (1GB)",
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

# Language-based color palettes - each language gets a gradient
LANGUAGE_COLORS = {
    "English": ["#1f77b4", "#4a90c2", "#7fabd0", "#b3c6de"],  # Blues
    "Chinese": ["#d62728", "#e04a4f", "#ea6d77", "#f490a0"],  # Reds
    "German": ["#2ca02c", "#4fb84f", "#72cf72", "#95e695"],   # Greens
    "French": ["#ff7f0e", "#ff9533", "#ffaa58", "#ffbf7d"],  # Oranges
    "Russian": ["#9467bd", "#a67dca", "#b893d7", "#caa9e4"], # Purples
    "Turkish": ["#8c564b", "#a06b5f", "#b48073", "#c89587"], # Browns
}

def get_model_color(model, model_list):
    idx = model_list.index(model)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

def get_model_color_by_language(model_key, model_list):
    """Assign colors based on language with gradients for same language models."""
    # Extract language from model key
    if model_key.startswith('goldfish_'):
        # For goldfish models, extract from the model_names mapping
        full_name = model_names.get(model_key, model_key)
        if "(" in full_name and ")" in full_name:
            language = full_name.split("(")[-1].split(")")[0]
        else:
            language = "Unknown"
    else:
        # For multilingual models, extract from the suffix
        parts = model_key.split('_')
        if len(parts) > 1:
            language = parts[-1].capitalize()
        else:
            language = "Unknown"
    
    # Get models of the same language
    same_lang_models = [m for m in model_list if 
                       (m.startswith('goldfish_') and language in model_names.get(m, '')) or
                       (not m.startswith('goldfish_') and m.endswith(f'_{language.lower()}'))]
    
    # Get color palette for this language
    if language in LANGUAGE_COLORS:
        colors = LANGUAGE_COLORS[language]
    else:
        # Fallback to original color scheme
        idx = list(LANGUAGE_COLORS.keys()).index(language) if language in LANGUAGE_COLORS else 0
        base_color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        colors = [base_color] * 4
    
    # Find position of this model within same-language models
    try:
        pos_in_lang = same_lang_models.index(model_key)
        color_idx = pos_in_lang % len(colors)
        return colors[color_idx]
    except ValueError:
        # Fallback to original color scheme
        idx = model_list.index(model_key)
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

def find_csv_file(model_key, dataset, task, probe_type):
    """Find CSV file for a given model, handling different probe naming conventions."""
    # For goldfish models, use the full model name, otherwise use first part
    if model_key.startswith('goldfish_'):
        base_model = model_key  # Use full goldfish model name
        # Goldfish models use "nn" instead of "mlp"
        probe_name = "nn" if probe_type == "mlp" else probe_type
    else:
        base_model = model_key.split('_')[0]
        probe_name = probe_type
    
    # Check both probes and probes2 directories
    probe_dirs = [
        os.path.join("..", "output", "probes", f"{dataset}_{base_model}_{task}_{probe_name}"),
        os.path.join("..", "output", "probes2", f"{dataset}_{base_model}_{task}_{probe_name}")
    ]
    
    for probe_dir in probe_dirs:
        csv_path = os.path.join(probe_dir, f"{task}_results.csv")
        if os.path.exists(csv_path):
            return csv_path
    
    return None

def plot_t5_results(model_to_dataset, model_list, output_dir="figures3", filename_prefix=""):
    probe_types = ["reg", "mlp", "rf"]
    titles = ["Linear Regression", "MLP", "Random Forest"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)
    
    aspect_ratio, base_height = 8 / 3, 5
    fig_size = (n_cols * base_height * aspect_ratio / n_rows, n_rows * base_height)

    # Pre-check which files exist to reduce redundant warnings
    file_availability = {}
    missing_files = []
    
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in ["lexeme", "inflection"]:
            for probe in ["reg", "mlp"]:
                csv_path = find_csv_file(model_key, dataset, task, probe)
                file_availability[model_key][(task, probe)] = csv_path
                if csv_path is None and (model_key, task, probe) not in missing_files:
                    missing_files.append((model_key, task, probe))
    
    # Print summary of missing files once
    if missing_files:
        print(f"[INFO] Missing {len(missing_files)} probe result files (will skip in plots)")
        # Only show a few examples to avoid spam
        for i, (model, task, probe) in enumerate(missing_files[:5]):
            print(f"  - {model} {task} {probe}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")

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
                    csv_path = file_availability[model_key].get((task, probe))
                    
                    if csv_path is None:
                        continue  # Skip silently since we already reported missing files
                        
                    try:
                        df = pd.read_csv(csv_path)
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
                            color=get_model_color_by_language(model_key, model_list),
                            linestyle=get_model_linestyle(model_key),
                        )
                    except Exception as e:
                        print(f"[WARN] Error processing {model_key} {task} {probe}: {e}")
                        continue
                
                # Set axis properties
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                
                if plot_selectivity:
                    row_ylim = (-0.3, 0.3) if row == 0 else (0, 0.8)
                else:
                    if row == 0:  # lexeme accuracy
                        row_ylim = (0, 1.0)
                    else:  # inflection accuracy
                        row_ylim = (0.6, 1.0)
                    
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
    
    # Create language-grouped legend
    grouped_handles, grouped_labels = create_language_grouped_legend(axes1[0, 0], model_list)
    if grouped_handles and grouped_labels:
        fig1.legend(grouped_handles, grouped_labels, loc="lower center", bbox_to_anchor=bbox_to_anchor,
                    ncol=min(4, len(grouped_labels)), mode="expand", frameon=True)
    
    fig1.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    filename1 = f"{filename_prefix}all_languages_linguistic_accuracy.png"
    fig1.savefig(os.path.join(output_dir, filename1), bbox_inches="tight")
    print(f"Saved linguistic accuracy figure to {os.path.join(output_dir, filename1)}")

    # Plot selectivity
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes2 = np.atleast_2d(axes2)
    plot_panel(fig2, axes2, plot_selectivity=True)
    
    # Create language-grouped legend
    grouped_handles2, grouped_labels2 = create_language_grouped_legend(axes2[0, 0], model_list)
    if grouped_handles2 and grouped_labels2:
        fig2.legend(grouped_handles2, grouped_labels2, loc="lower center", bbox_to_anchor=bbox_to_anchor,
                    ncol=min(4, len(grouped_labels2)), mode="expand", frameon=True)
    
    fig2.tight_layout(rect=[0, 0.05, 1, 0.97])
    filename2 = f"{filename_prefix}all_languages_classifier_selectivity.png"
    fig2.savefig(os.path.join(output_dir, filename2), bbox_inches="tight")
    print(f"Saved selectivity figure to {os.path.join(output_dir, filename2)}")

def create_language_grouped_legend(axes, model_list):
    """Create a legend grouped by language with proper ordering."""
    # Get all handles and labels from the plot
    handles, labels = axes.get_legend_handles_labels()
    
    # Group by language
    language_groups = {}
    for handle, label in zip(handles, labels):
        # Extract language from label (assumes format "Model (Language)")
        if "(" in label and ")" in label:
            language = label.split("(")[-1].split(")")[0]
        else:
            language = "Unknown"
        
        if language not in language_groups:
            language_groups[language] = []
        language_groups[language].append((handle, label))
    
    # Sort languages and create grouped handles/labels
    grouped_handles = []
    grouped_labels = []
    
    for language in sorted(language_groups.keys()):
        for handle, label in language_groups[language]:
            grouped_handles.append(handle)
            grouped_labels.append(label)
    
    return grouped_handles, grouped_labels

def generate_t5_markdown_tables(model_to_dataset, model_list, output_dir="figures3"):
    """Generate markdown tables for T5 models across all languages."""
    percentages = [0, 25, 50, 75, 100]
    
    probe_types = ["reg", "mlp"]
    tasks = ["lexeme", "inflection"]
    
    probe_names = {
        "reg": "Linear Regression",
        "mlp": "MLP"
    }
    
    # Pre-check file availability
    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe_type in probe_types:
                csv_path = find_csv_file(model_key, dataset, task, probe_type)
                file_availability[model_key][(task, probe_type)] = csv_path
    
    # Group models by family
    model_families = {}
    for model_key in model_list:
        if model_key.startswith('goldfish_'):
            family = "Goldfish"
        elif model_key.startswith('mt5_'):
            family = "mT5"
        elif model_key.startswith('qwen2.5-7B-instruct_'):
            family = "Qwen2.5-7B-Instruct"
        elif model_key.startswith('qwen2.5-7B_'):
            family = "Qwen2.5-7B"
        elif model_key.startswith('qwen2-instruct_'):
            family = "Qwen2.5-1.5B-Instruct"
        elif model_key.startswith('qwen2_'):
            family = "Qwen2.5-1.5B"
        else:
            family = "Other"
        
        if family not in model_families:
            model_families[family] = []
        model_families[family].append(model_key)
    
    for task in tasks:
        for probe_type in probe_types:
            probe_name = probe_names[probe_type]
            
            print(f"\n## {task.title()} Accuracy - {probe_name}\n")
            
            # Process each model family separately
            for family_name in sorted(model_families.keys()):
                family_models = model_families[family_name]
                
                # Collect all valid CSVs and their layer counts for this family
                valid_models = []
                layer_counts = []
                
                for model_key in family_models:
                    csv_path = file_availability[model_key].get((task, probe_type))
                    if csv_path:
                        try:
                            df = pd.read_csv(csv_path)
                            valid_models.append((model_key, csv_path))
                            layer_counts.append(len(df))
                        except:
                            continue
                
                if not valid_models:
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
                
                print(f"### {family_name}\n")
                header_row = "| Model | Dataset | " + " | ".join(headers) + " |"
                separator_row = "|" + "|".join([" --- "] * (len(headers) + 2)) + "|"
                
                print(header_row)
                print(separator_row)
                
                # Group models in this family by language
                language_groups = {}
                for model_key in family_models:
                    dataset = model_to_dataset[model_key]
                    language = model_names.get(model_key, model_key)
                    
                    # Extract language from model name (assumes format "Model (Language)")
                    if "(" in language and ")" in language:
                        lang_part = language.split("(")[-1].split(")")[0]
                    else:
                        lang_part = "Unknown"
                    
                    if lang_part not in language_groups:
                        language_groups[lang_part] = []
                    language_groups[lang_part].append((model_key, dataset, language))
                
                # Sort languages alphabetically
                for lang in sorted(language_groups.keys()):
                    models_in_lang = language_groups[lang]
                    
                    for model_key, dataset, language in models_in_lang:
                        csv_path = file_availability[model_key].get((task, probe_type))
                        
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
                                row_data = ["N/A"] * len(percentages)
                        
                        row = f"| {language} | {dataset} | " + " | ".join(row_data) + " |"
                        print(row)
                
                print()  # Add blank line between families

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
    
    multilingual_models = [
        # "byt5",
        "mt5",
        "qwen2",
        "qwen2-instruct",
        "qwen2.5-7B",
        "qwen2.5-7B-instruct",
    ]
    
    # Goldfish models are language-specific
    goldfish_models = [
        ("goldfish_eng_latn_1000mb", "ud_gum_dataset", "English"),
        ("goldfish_zho_hans_1000mb", "ud_zh_gsd_dataset", "Chinese"),
        ("goldfish_deu_latn_1000mb", "ud_de_gsd_dataset", "German"),
        ("goldfish_fra_latn_1000mb", "ud_fr_gsd_dataset", "French"),
        ("goldfish_rus_cyrl_1000mb", "ud_ru_syntagrus_dataset", "Russian"),
        ("goldfish_tur_latn_1000mb", "ud_tr_imst_dataset", "Turkish"),
    ]
    
    # Add multilingual models (tested on all languages)
    for model in multilingual_models:
        for dataset, lang in language_datasets:
            model_key = f"{model}_{lang.lower()}"
            all_models.append(model_key)
            model_to_dataset[model_key] = dataset
            model_names[model_key] = f"{model_names[model]} ({lang})"
    
    # Add goldfish models (each only on their specific language)
    for goldfish_model, dataset, lang in goldfish_models:
        model_key = goldfish_model  # Don't add language suffix for goldfish models
        all_models.append(model_key)
        model_to_dataset[model_key] = dataset
        model_names[model_key] = f"{model_names[goldfish_model]} ({lang})"
    
    print("Plotting multilingual models across all languages...")
    plot_t5_results(model_to_dataset, all_models)
    
    print("\nGenerating markdown tables for multilingual models...")
    generate_t5_markdown_tables(model_to_dataset, all_models)