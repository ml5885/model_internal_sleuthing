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

bbox_to_anchor = (0, -0.3, 1, 0.1)

model_names = {
    "byt5": "ByT5-Base",
    "mt5": "mT5-Base",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
    "goldfish_eng_latn_1000mb": "Goldfish English",
    "goldfish_zho_hans_1000mb": "Goldfish Chinese",
    "goldfish_deu_latn_1000mb": "Goldfish German",
    "goldfish_fra_latn_1000mb": "Goldfish French",
    "goldfish_rus_cyrl_1000mb": "Goldfish Russian",
    "goldfish_tur_latn_1000mb": "Goldfish Turkish",
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

LANGUAGE_COLORS = {
    "English": ["#1f77b4", "#4a90c2", "#7fabd0", "#b3c6de"],
    "Chinese": ["#d62728", "#e04a4f", "#ea6d77", "#f490a0"],
    "German": ["#2ca02c", "#4fb84f", "#72cf72", "#95e695"],
    "French": ["#ff7f0e", "#ff9533", "#ffaa58", "#ffbf7d"],
    "Russian": ["#9467bd", "#a67dca", "#b893d7", "#caa9e4"],
    "Turkish": ["#8c564b", "#a06b5f", "#b48073", "#c89587"],
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

def plot_combined_accuracy_selectivity(
    model_to_dataset, model_list, output_dir="figures3", filename_prefix="", rf_only=False
):
    """
    Plot combined accuracy and selectivity for reg/mlp probes (2x4 grid) or RF probe (1x2 grid).
    If rf_only=True, only plot RF probe for inflection (1x2 grid).
    """
    if rf_only:
        probes = ["rf"]
        tasks = ["inflection"]
        n_rows, n_cols = 1, 2
        titles = ["Random Forest Accuracy", "Random Forest Selectivity"]
        aspect_ratio, base_height = 6.5, 5
        fig_width = n_cols * base_height * aspect_ratio / 2.0
        fig_height = 2 * base_height
        fig_size = (fig_width, fig_height)
    else:
        probes = ["reg", "mlp"]
        tasks = ["lexeme", "inflection"]
        n_rows, n_cols = len(tasks), len(probes) * 2
        titles = ["Linear Regression", "MLP"]
        aspect_ratio, base_height = 3.5, 5
        fig_width = n_cols * base_height * aspect_ratio / 2.0
        fig_height = n_rows * base_height
        fig_size = (fig_width, fig_height)

    # Pre-check file availability
    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe_type in probes:
                csv_path = find_csv_file(model_key, dataset, task, probe_type)
                file_availability[model_key][(task, probe_type)] = csv_path

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, task in enumerate(tasks):
        for col in range(n_cols):
            if rf_only:
                probe_type = "rf"
                plot_selectivity = (col == 1)
                ax = axes[0, col]
            else:
                probe_idx = col % 2
                probe_type = probes[probe_idx]
                plot_selectivity = (col >= 2)
                ax = axes[row, col]
            for i, model_key in enumerate(model_list):
                csv_path = file_availability[model_key].get((task, probe_type))
                if csv_path is None:
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    acc_col, ctrl_col = get_acc_columns(df, task)
                    df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                    if plot_selectivity:
                        if ctrl_col not in df.columns:
                            ax.text(0.5, 0.5, "No selectivity data", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=22)
                            continue
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
                    ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=22)
            ax.tick_params(axis='both', which='major', length=10, width=2, labelsize=20)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)], fontsize=24)
            # Set y-limits and labels
            if rf_only:
                if col == 0:
                    row_ylim = (0.6, 1.0)
                    ylabel = "Inflection Accuracy"
                else:
                    row_ylim = (0, 0.8)
                    ylabel = "Inflection Selectivity"
            else:
                if plot_selectivity:
                    row_ylim = (-0.3, 0.3) if row == 0 else (0, 0.8)
                    ylabel = "Lexeme Selectivity" if row == 0 else "Inflection Selectivity"
                else:
                    row_ylim = (0, 1.0) if row == 0 else (0.6, 1.0)
                    ylabel = "Lexeme Accuracy" if row == 0 else "Inflection Accuracy"
            yticks = np.arange(row_ylim[0], row_ylim[1] + 0.01, 0.2)
            ax.set_ylim(*row_ylim)
            ax.set_yticks(yticks)
            if (not rf_only and (col == 0 or col == 2)) or (rf_only):
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=24)
                if (not rf_only and row == 0 and col == 0) or (rf_only and col == 0):
                    ax.set_ylabel(ylabel, labelpad=20, fontsize=34)
                elif (not rf_only and row == 0 and col == 2) or (rf_only and col == 1):
                    ax.set_ylabel(ylabel, labelpad=20, fontsize=34)
                else:
                    ax.set_ylabel(ylabel, labelpad=20, fontsize=34)
            else:
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_yticklabels([])
            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
            if (rf_only and True) or (not rf_only and row == 1):
                ax.set_xlabel("Normalized layer number (%)", labelpad=15, fontsize=34)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            if not rf_only and row == 0:
                title_idx = col % 2
                ax.set_title(f"{titles[title_idx]}", pad=10, loc='center', fontsize=34)
            elif rf_only:
                ax.set_title(titles[col], pad=15, fontsize=34)

    def get_language_from_label(label):
        if "(" in label and ")" in label:
            return label.split("(")[-1].split(")")[0]
        return "Unknown"

    # Legend grouping by language - use all models from model_list
    language_groups = {}
    for model_key in model_list:
        label = model_names.get(model_key, model_key)
        lang = get_language_from_label(label)
        # Remove language in parentheses from label for legend
        if "(" in label and ")" in label:
            model_label = label[:label.rfind("(")].strip()
        else:
            model_label = label
        if lang not in language_groups:
            language_groups[lang] = []
        # Create a dummy handle with the correct color and linestyle
        color = get_model_color_by_language(model_key, model_list)
        linestyle = get_model_linestyle(model_key)
        handle = plt.Line2D([], [], color=color, linestyle=linestyle, linewidth=3.0)
        language_groups[lang].append((handle, model_label))
    
    grouped_handles = []
    grouped_labels = []
    for lang in sorted(language_groups.keys()):
        # Add language as a dummy handle (invisible)
        grouped_handles.append(plt.Line2D([], [], color='none', label=lang))
        grouped_labels.append(lang)
        for handle, model_label in language_groups[lang]:
            grouped_handles.append(handle)
            grouped_labels.append(model_label)
    if grouped_handles and grouped_labels:
        fig.legend(
            grouped_handles, grouped_labels,
            loc="lower center",
            bbox_to_anchor=(0, -0.45, 1, 0.16),
            ncol=min(6, len(grouped_labels)),
            mode="expand",
            frameon=True,
            fontsize=28,
        )

    os.makedirs(output_dir, exist_ok=True)
    if rf_only:
        filename = f"{filename_prefix}rf_combined_accuracy_selectivity.png"
        printmsg = "Saved RF combined accuracy/selectivity figure to"
    else:
        filename = f"{filename_prefix}combined_linguistic_and_selectivity.png"
        printmsg = "Saved combined linguistic/selectivity (reg/mlp) figure to"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"{printmsg} {os.path.join(output_dir, filename)}")

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
    plot_combined_accuracy_selectivity(model_to_dataset, all_models, filename_prefix="all_languages_")
    plot_combined_accuracy_selectivity(
        model_to_dataset, all_models, output_dir="figures3", filename_prefix="all_languages_", rf_only=True
    )