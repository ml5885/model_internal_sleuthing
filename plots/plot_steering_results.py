import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import numpy as np
from matplotlib import cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODEL_DISPLAY_NAMES

# Use the same color mapping as plot_attention_results.py
MODEL_COLORS = {
    "gpt2": "tab:brown",
    "gpt2-large": "tab:orange",
    "gpt2-xl": "tab:red",
    "qwen2": "tab:blue",
    "qwen2-instruct": "tab:cyan",
    "qwen2.5-7B": "mediumseagreen",
    "qwen2.5-7B-instruct": "springgreen",
    "gemma2b": "darkviolet",
    "gemma2b-it": "deeppink",
    "bert-base-uncased": "steelblue",
    "bert-large-uncased": "navy",
    "deberta-v3-large": "darkkhaki",
    "llama3-8b": "lightcoral",
    "llama3-8b-instruct": "rosybrown",
    "pythia-6.9b": "darkgoldenrod",
    "pythia-6.9b-tulu": "lightsalmon",
    "olmo2-7b-instruct": "palegreen",
    "olmo2-7b": "forestgreen",
}

def collect_all_results(steering_dir, models, dataset):
    """
    Collect all steering results into a single DataFrame.
    Returns: DataFrame with columns: model, probe_type, lambda, layer, mean_prob_change, flip_rate, ...
    """
    records = []
    for model in models:
        pattern = re.compile(rf"^{re.escape(dataset)}_{re.escape(model)}_(?P<probe_type>\w+)_lambda(?P<lambda_val>\d+\.?\d*)$")
        for dirname in os.listdir(steering_dir):
            match = pattern.match(dirname)
            if match:
                probe_type = match.group("probe_type")
                lambda_val = float(match.group("lambda_val"))
                results_dir = os.path.join(steering_dir, dirname)
                summary_file = os.path.join(results_dir, "steering_summary.csv")
                if not os.path.exists(summary_file):
                    continue
                df = pd.read_csv(summary_file)
                df["model"] = model
                df["probe_type"] = probe_type
                df["lambda"] = lambda_val
                records.append(df)
    if not records:
        return pd.DataFrame()
    df_all = pd.concat(records, ignore_index=True)
    df_all["layer"] = df_all["layer"].astype(int)
    return df_all

def get_model_cmap(model, n):
    """
    Return a matplotlib colormap instance for a given model and number of lambdas.
    """
    # Map model to a base colormap
    model_cmaps = {
        "gpt2": cm.Oranges,
        "gpt2-large": cm.YlOrBr,
        "gpt2-xl": cm.Reds,
        "qwen2": cm.Blues,
        "qwen2-instruct": cm.cividis,
        "qwen2.5-7B": cm.Greens,
        "qwen2.5-7B-instruct": cm.Greens,
        "gemma2b": cm.Purples,
        "gemma2b-it": cm.Purples,
        "bert-base-uncased": cm.Blues,
        "bert-large-uncased": cm.Blues,
        "deberta-v3-large": cm.YlGn,
        "llama3-8b": cm.Reds,
        "llama3-8b-instruct": cm.Reds,
        "pythia-6.9b": cm.YlOrBr,
        "pythia-6.9b-tulu": cm.YlOrBr,
        "olmo2-7b-instruct": cm.Greens,
        "olmo2-7b": cm.Greens,
    }
    return model_cmaps.get(model, cm.Greys)(np.linspace(0.35, 0.95, n))

def plot_all_models(df_all, dataset, output_dir):
    """
    Plot each model separately for each probe_type, showing all lambdas for that model.
    """
    if df_all.empty:
        print("No results to plot.")
        return

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
    legend_params = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=18,
        ncol=5,
        frameon=True,
        title_fontsize=22
    )
    tight_layout_rect = [0, 0.05, 1, 1]
    figsize = (18, 10)
    grid_params = dict(linestyle="--", alpha=0.4, linewidth=1.0)

    sns.set_theme(style="whitegrid")
    probe_types = df_all["probe_type"].unique()
    for probe_type in probe_types:
        df_probe = df_all[df_all["probe_type"] == probe_type].copy()
        df_probe["model_display"] = df_probe["model"].map(MODEL_DISPLAY_NAMES).fillna(df_probe["model"])
        df_probe["color"] = df_probe["model"].map(MODEL_COLORS).fillna("gray")
        for model in df_probe["model"].unique():
            df_model = df_probe[df_probe["model"] == model]
            model_display = df_model["model_display"].iloc[0]
            base_color = df_model["color"].iloc[0]
            lambdas = sorted(df_model["lambda"].unique())
            colors = get_model_cmap(model, len(lambdas))

            # Mean Probability Change
            plt.figure(figsize=figsize)
            handles = []
            labels = []
            for i, (lambda_val, lambda_group) in enumerate(df_model.groupby("lambda")):
                color = colors[i]
                label = f"λ={lambda_val:g}"
                h, = plt.plot(
                    lambda_group["layer"],
                    lambda_group["mean_prob_change"],
                    marker="o",
                    label=label,
                    color=color,
                )
                handles.append(h)
                labels.append(label)
            plt.title(f"Mean Probability Change ({model_display} on {dataset} - {probe_type.upper()})", fontsize=26)
            plt.xlabel("Layer", fontsize=24)
            plt.ylabel("Mean Probability Change", fontsize=24)
            plt.grid(**grid_params)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.tight_layout(rect=tight_layout_rect)
            plt.legend(handles, labels, **legend_params)
            plot_path = os.path.join(output_dir, f"{dataset}_{model}_{probe_type}_prob_change_multi_lambda.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved probability change plot to {plot_path}")

            # Flip Rate
            plt.figure(figsize=figsize)
            handles = []
            labels = []
            for i, (lambda_val, lambda_group) in enumerate(df_model.groupby("lambda")):
                color = colors[i]
                label = f"λ={lambda_val:g}"
                h, = plt.plot(
                    lambda_group["layer"],
                    lambda_group["flip_rate"],
                    marker="o",
                    label=label,
                    color=color,
                )
                handles.append(h)
                labels.append(label)
            plt.title(f"Prediction Flip Rate ({model_display} on {dataset} - {probe_type.upper()})", fontsize=26)
            plt.xlabel("Layer", fontsize=24)
            plt.ylabel("Flip Rate", fontsize=24)
            plt.grid(**grid_params)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.tight_layout(rect=tight_layout_rect)
            plt.legend(handles, labels, **legend_params)
            plot_path = os.path.join(output_dir, f"{dataset}_{model}_{probe_type}_flip_rate_multi_lambda.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved flip rate plot to {plot_path}")

def print_markdown_tables(df_all):
    """
    Print wide-format markdown tables grouped by model and probe_type.
    """
    if df_all.empty:
        print("No results for markdown tables.")
        return
    for probe_type in df_all["probe_type"].unique():
        df_probe = df_all[df_all["probe_type"] == probe_type]
        for model in df_probe["model"].unique():
            df_model = df_probe[df_probe["model"] == model]
            mean_pivot = df_model.pivot_table(index="layer", columns="lambda", values="mean_prob_change")
            flip_pivot = df_model.pivot_table(index="layer", columns="lambda", values="flip_rate")
            print(f"\n## {model} | {probe_type.upper()} | Mean Probability Change")
            print(mean_pivot.to_markdown(floatfmt=".4f"))
            print(f"\n## {model} | {probe_type.upper()} | Flip Rate")
            print(flip_pivot.to_markdown(floatfmt=".4f"))

def main():
    parser = argparse.ArgumentParser(description="Plot steering experiment results for one or more models and a dataset.")
    parser.add_argument("--steering_dir", required=True, help="Base directory containing all steering experiment subdirectories.")
    parser.add_argument("--models", required=True, nargs="+", help="Model names to filter experiments by (e.g., 'qwen2 qwen1').")
    parser.add_argument("--dataset", required=True, help="Dataset name to filter experiments by (e.g., 'ud_gum_dataset').")
    parser.add_argument("--output_dir", required=True, help="Directory to save the combined plots and markdown files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_all = collect_all_results(args.steering_dir, args.models, args.dataset)
    if df_all.empty:
        print(f"No steering results found for models '{args.models}' and dataset '{args.dataset}' in '{args.steering_dir}'.")
        return

    plot_all_models(df_all, args.dataset, args.output_dir)
    # print_markdown_tables(df_all)

if __name__ == "__main__":
    main()