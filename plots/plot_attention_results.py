import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors

from plot_multilingual_results import (
    plot_t5_results, generate_t5_markdown_tables, model_names,
    get_acc_columns, MODEL_COLORS, LANGUAGE_COLORS,
    get_model_color_by_language, get_model_linestyle, create_language_grouped_legend
)

if not isinstance(MODEL_COLORS, dict):
    MODEL_COLORS = {}

def find_csv_file_probe(model_key, dataset, task, probe_type, attn=True):
    # Use only the base model name (no language suffix) to match your directory/file names
    # e.g., qwen2 or qwen2-instruct
    base_model = model_key.split('_')[0]
    probe_name = probe_type
    suffix = "_attn" if attn else ""
    probe_dirs = [
        os.path.join("..", "output", "probes", f"{dataset}_{base_model}_{task}_{probe_name}{suffix}"),
        os.path.join("..", "output", "probes2", f"{dataset}_{base_model}_{task}_{probe_name}{suffix}")
    ]
    for probe_dir in probe_dirs:
        csv_path = os.path.join(probe_dir, f"{task}_results.csv")
        if os.path.exists(csv_path):
            return csv_path
    return None

def add_legends(fig, model_list, title_model="Model", title_source="Source"):
    from matplotlib.lines import Line2D

    source_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=3.0, label="Attention Output"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, alpha=0.7, label="Residual Stream"),
    ]

    model_handles = []
    model_labels_and_base_models = [
        ("GPT-2-Small", "gpt2"),
        ("GPT-2-Large", "gpt2-large"),
        ("GPT-2-XL", "gpt2-xl"),
        ("Qwen2.5-1.5B", "qwen2"),
        ("Qwen2.5-1.5B-Instruct", "qwen2-instruct"),
        ("Qwen2.5-7B", "qwen2.5-7B"),
        ("Qwen2.5-7B-Instruct", "qwen2.5-7B-instruct"),
        ("Pythia-1.4B", "pythia1.4b"),
        ("Gemma-2-2B", "gemma2b"),
        ("Gemma-2-2B-Instruct", "gemma2b-it"),
        ("BERT-Base-Uncased", "bert-base-uncased"),
        ("BERT-Large-Uncased", "bert-large-uncased"),
        ("DeBERTa-v3-Large", "deberta-v3-large"),
        ("Llama-3-8B", "llama3-8b"),
        ("Llama-3-8B-Instruct", "llama3-8b-instruct"),
        ("Pythia-6.9B", "pythia-6.9b"),
        ("Pythia-6.9B-Tulu", "pythia-6.9b-tulu"),
        ("OLMo-2-1124-7B-Instruct", "olmo2-7b-instruct"),
        ("OLMo-2-1124-7B", "olmo2-7b"),
    ]

    filtered_model_labels_and_base_models = [
        (label, base_model) for label, base_model in model_labels_and_base_models if base_model in model_list or any(model.startswith(base_model) for model in model_list)
    ]
    for label, base_model in filtered_model_labels_and_base_models:
        model_handles.append(Line2D([0], [0], color=MODEL_COLORS.get(base_model, "gray"), linestyle="-", linewidth=4.0, label=label))
        
    model_bbox_anchor = (0.5, 0.03)
    ncol = 5

    if any(model.startswith("bert") for model in model_list):
        model_bbox_anchor = (0.5, 0.07)
        ncol = 3

    fig.legend(
        source_handles, [h.get_label() for h in source_handles],
        loc="lower center", bbox_to_anchor=(0.5, 0.16),
        ncol=2, frameon=True, fontsize=18, title="", title_fontsize=18
    )

    fig.legend(
        model_handles, [h.get_label() for h in model_handles],
        loc="lower center", bbox_to_anchor=model_bbox_anchor,
        ncol=ncol, frameon=True, fontsize=18, title="", title_fontsize=18
    )


def plot_attention_results(model_to_dataset, model_list, output_dir="figures3", filename_prefix="attention_"):
    probe_types = ["reg", "mlp", "rf"]
    titles = ["Linear Regression", "MLP", "Random Forest"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)
    aspect_ratio, base_height = 8 / 3, 5
    fig_size = (n_cols * base_height * aspect_ratio / n_rows, n_rows * base_height)

    file_availability = {}
    missing_files = []
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            # Only include "rf" for inflection, not lexeme
            probes = ["reg", "mlp"]
            if task == "inflection":
                probes.append("rf")
            for probe in probes:
                for attn in [True, False]:
                    key = (task, probe, attn)
                    if probe == "mlp":
                        csv_path = find_csv_file_probe(model_key, dataset, task, "mlp", attn=attn)
                        if not csv_path:
                            csv_path = find_csv_file_probe(model_key, dataset, task, "nn", attn=attn)
                    else:
                        csv_path = find_csv_file_probe(model_key, dataset, task, probe, attn=attn)
                    file_availability[model_key][key] = csv_path
                    if csv_path is None:
                        missing_files.append((model_key, task, probe, attn))
    if missing_files:
        print(f"[INFO] Missing {len(missing_files)} probe result files (will skip in plots)")
        for (model, task, probe, attn) in missing_files:
            if probe != "rf":
                print(f"  - {model} {task} {probe} {'attn' if attn else 'residual'}")

    def plot_panel(axes, model_list_subset, plot_selectivity=False):
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
                        ax.set_ylim(-0.3, 0.3)
                    else:
                        ax.set_ylim(0, 1.0)
                    ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + 0.001, 0.2))
                    if col == 0:
                        ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()])
                    else:
                        ax.set_yticklabels([])
                    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                    if row == 1:
                        ax.set_xlabel("Normalized layer number (%)", labelpad=15)
                    if row == 0:
                        ax.set_title(titles[col], pad=15)
                    continue

                for model_key in model_list_subset:
                    base_model_name = model_key.split('_')[0]

                    for attn, label_suffix, alpha, zorder in [
                        (True, " (Attention Output)", 1.0, 2),
                        (False, " (Residual Stream)", 0.7, 1)
                    ]:
                        if task == "lexeme" and probe == "rf":
                            continue
                        csv_path = file_availability[model_key].get((task, probe, attn))
                        if not csv_path:
                            continue
                        try:
                            df = pd.read_csv(csv_path)
                            acc_col, ctrl_col = get_acc_columns(df, task)
                            df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                            y = df[acc_col] - df[ctrl_col] if plot_selectivity else df[acc_col]

                            line_color = MODEL_COLORS.get(base_model_name, "black")
                            
                            ax.plot(
                                df["Layer_Normalized"], y,
                                label=None, 
                                linewidth=3.0 if attn else 2.0,
                                color=line_color,
                                linestyle="-" if attn else "--",
                                alpha=alpha,
                                zorder=zorder
                            )
                        except Exception as e:
                            print(f"[WARN] Error processing {model_key} {task} {probe} ({'attn' if attn else 'residual'}): {e}")
                            continue
                
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                if plot_selectivity:
                    if row == 0:
                        ax.set_ylim(-0.25, 0.25)
                        ax.set_yticks(np.arange(-0.2, 0.21, 0.2))
                    else:
                        ax.set_ylim(0, 0.8)
                        ax.set_yticks(np.arange(0, 0.81, 0.2))
                else:
                    if row == 0:
                        ax.set_ylim(0, 1.0)
                        ax.set_yticks(np.arange(0, 1.01, 0.2))
                    else:
                        ax.set_ylim(0.6, 1.0)
                        ax.set_yticks(np.arange(0.6, 1.01, 0.1))
                if col == 0:
                    ylabel = f"{task.title()} {'Selectivity' if plot_selectivity else 'Accuracy'}"
                    ax.set_ylabel(ylabel, labelpad=15)
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15)

    bert_models = [m for m in model_list if "bert" in m]
    gpt_models = [m for m in model_list if "gpt2" in m]
    bert_gpt_models = bert_models + gpt_models
    other_models = [m for m in model_list if m not in bert_gpt_models]

    model_groups = {
        "BERT_GPT": bert_gpt_models,
        "Other": other_models
    }

    for model_type, models in model_groups.items():
        if not models:
            print(f"Skipping {model_type} models (no models found)")
            continue

        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=fig_size)
        axes1 = np.atleast_2d(axes1)
        plot_panel(axes1, models, plot_selectivity=False)
        fig1.align_ylabels(axes1[:, 0])
        fig1.tight_layout(rect=[0, 0.2, 1, 0.97]) 
        add_legends(fig1, models)
        os.makedirs(output_dir, exist_ok=True)
        filename1 = f"{filename_prefix}{model_type.lower()}_linguistic_accuracy.png"
        fig1.savefig(os.path.join(output_dir, filename1), bbox_inches="tight")
        print(f"Saved {model_type} attention linguistic accuracy figure to {os.path.join(output_dir, filename1)}")

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=fig_size)
        axes2 = np.atleast_2d(axes2)
        plot_panel(axes2, models, plot_selectivity=True)
        fig2.align_ylabels(axes2[:, 0])
        fig2.tight_layout(rect=[0, 0.2, 1, 0.97])
        add_legends(fig2, models)
        filename2 = f"{filename_prefix}{model_type.lower()}_classifier_selectivity.png"
        fig2.savefig(os.path.join(output_dir, filename2), bbox_inches="tight")
        print(f"Saved {model_type} attention selectivity figure to {os.path.join(output_dir, filename2)}")


def generate_attention_markdown_tables(model_to_dataset, model_list, output_dir="figures3"):
    percentages = [0, 25, 50, 75, 100]
    probe_types = ["reg", "mlp"]
    tasks = ["lexeme", "inflection"]
    probe_names = {
        "reg": "Linear Regression",
        "mlp": "MLP"
    }
    file_availability = {}
    for model_key in model_list:
        dataset = model_to_dataset[model_key]
        file_availability[model_key] = {}
        for task in tasks:
            for probe_type in probe_types:
                for attn in [True, False]:
                    key = (task, probe_type, attn)
                    csv_path = find_csv_file_probe(model_key, dataset, task, probe_type, attn=attn)
                    file_availability[model_key][key] = csv_path
    
    model_families = {"Attention Models": model_list}  # Updated title

    for task in tasks:
        for probe_type in probe_types:
            probe_name = probe_names[probe_type]
            print(f"\n## {task.title()} Accuracy - {probe_name} (Attention & Residual)\n")  # Updated title
            family_models = model_families["Attention Models"]  # Updated key

            valid_models = []
            layer_counts = []
            for model_key in family_models:
                for attn in [True, False]:
                    csv_path = file_availability[model_key].get((task, probe_type, attn))
                    if csv_path:
                        try:
                            df = pd.read_csv(csv_path)
                            valid_models.append((model_key, csv_path, attn))
                            layer_counts.append(len(df))
                        except:
                            continue
            if not valid_models:
                print("No valid results found for this probe type.\n")
                continue

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
            
            print(f"### Attention Models\n")
            header_row = "| Model | Dataset | Source | " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join([" --- "] * (len(headers) + 3)) + "|"
            print(header_row)
            print(separator_row)

            def attention_sort_key(model_key):
                base_model = model_key.split('_')[0]
                size_priority = 0 if base_model in ['qwen2', 'qwen2-instruct'] else 1
                type_priority = 0 if 'instruct' not in base_model else 1
                return (size_priority, type_priority, base_model)
            
            sorted_models = sorted(family_models, key=attention_sort_key)

            for model_key in sorted_models:
                dataset = model_to_dataset[model_key]
                language = model_names.get(model_key, model_key)
                for attn, src_label in [(True, "Attention Output"), (False, "Residual Stream")]:
                    csv_path = file_availability[model_key].get((task, probe_type, attn))
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
                    row = f"| {language} | {dataset} | {src_label} | " + " | ".join(row_data) + " |"
                    print(row)
            print()

if __name__ == "__main__":
    model_names = {
        "gpt2": "GPT-2-Small",
        "gpt2-large": "GPT-2-Large",
        "gpt2-xl": "GPT-2-XL",
        "qwen2": "Qwen2.5-1.5B",
        "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
        "qwen2.5-7B": "Qwen2.5-7B",
        "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
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

    models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "deberta-v3-large",
        # "gpt2",
        "gpt2-large",
        "gpt2-xl",
        "qwen2",
        "qwen2-instruct",
        "qwen2.5-7B",
        "qwen2.5-7B-instruct",
        "gemma2b",
        "gemma2b-it",
        "llama3-8b",
        "llama3-8b-instruct",
        "pythia-6.9b",
        "pythia-6.9b-tulu",
        "olmo2-7b-instruct",
        "olmo2-7b"
    ]
    
    attention_models = models
    
    attention_datasets = [
        ("ud_gum_dataset", "English"),
    ]
    
    attention_all_models = []
    attention_model_to_dataset = {}
    
    temp_model_colors = {}
    temp_model_colors["gpt2"] = "tab:brown"
    temp_model_colors["gpt2-large"] = "tab:orange"
    temp_model_colors["gpt2-xl"] = "tab:red"
    temp_model_colors["qwen2"] = "tab:blue"  
    temp_model_colors["qwen2-instruct"] = "tab:cyan"
    temp_model_colors["qwen2.5-7B"] = "mediumseagreen"
    temp_model_colors["qwen2.5-7B-instruct"] = "springgreen"
    temp_model_colors["gemma2b"] = "darkviolet"
    temp_model_colors["gemma2b-it"] = "deeppink"
    temp_model_colors["bert-base-uncased"] = "steelblue"
    temp_model_colors["bert-large-uncased"] = "navy"
    temp_model_colors["deberta-v3-large"] = "darkkhaki"
    temp_model_colors["llama3-8b"] = "lightcoral"
    temp_model_colors["llama3-8b-instruct"] = "rosybrown"
    temp_model_colors["pythia-6.9b"] = "darkgoldenrod"
    temp_model_colors["pythia-6.9b-tulu"] = "lightsalmon"
    temp_model_colors["olmo2-7b-instruct"] = "palegreen"
    temp_model_colors["olmo2-7b"] = "forestgreen"
    MODEL_COLORS.update(temp_model_colors)

    for model in attention_models:
        for dataset, lang in attention_datasets:
            model_key = f"{model}_{lang.lower()}"
            attention_all_models.append(model_key)
            attention_model_to_dataset[model_key] = dataset
            model_names[model_key] = f"{model_names.get(model, model).replace('qwen2', 'Qwen2.5-1.5B').replace('instruct', 'Instruct')} ({lang})"

    print("Generating plots for attention experiments...")
    plot_attention_results(attention_model_to_dataset, attention_all_models)
    
    print("\nGenerating markdown tables for attention experiments...")
    generate_attention_markdown_tables(attention_model_to_dataset, attention_all_models)