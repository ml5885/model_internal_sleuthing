import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

def plot_all_models(df_all, dataset, output_dir):
    """
    Plot all models together for each probe_type.
    """
    if df_all.empty:
        print("No results to plot.")
        return

    sns.set_theme(style="whitegrid")
    probe_types = df_all["probe_type"].unique()
    for probe_type in probe_types:
        df_probe = df_all[df_all["probe_type"] == probe_type]
        # Mean Probability Change
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            data=df_probe,
            x="layer",
            y="mean_prob_change",
            hue="model",
            style="lambda",
            markers=True,
            dashes=False,
            palette="Dark2"
        )
        plt.title(f"Mean Probability Change (All Models on {dataset} - {probe_type.upper()})", fontsize=16)
        plt.xlabel("Model Layer", fontsize=12)
        plt.ylabel("Mean Probability Change", fontsize=12)
        plt.legend(title="Model / Lambda", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{dataset}_ALLMODELS_{probe_type}_prob_change_multi_lambda.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved consolidated probability change plot to {plot_path}")

        # Flip Rate
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            data=df_probe,
            x="layer",
            y="flip_rate",
            hue="model",
            style="lambda",
            markers=True,
            dashes=False,
            palette="Dark2"
        )
        plt.title(f"Prediction Flip Rate (All Models on {dataset} - {probe_type.upper()})", fontsize=16)
        plt.xlabel("Model Layer", fontsize=12)
        plt.ylabel("Flip Rate", fontsize=12)
        plt.legend(title="Model / Lambda", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{dataset}_ALLMODELS_{probe_type}_flip_rate_multi_lambda.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved consolidated flip rate plot to {plot_path}")

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
    print_markdown_tables(df_all)

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser(description="Plot steering experiment results for one or more models and a dataset.")
    parser.add_argument("--steering_dir", required=True, help="Base directory containing all steering experiment subdirectories.")
    parser.add_argument("--models", required=True, nargs="+", help="Model names to filter experiments by (e.g., 'qwen2 qwen1').")
    parser.add_argument("--dataset", required=True, help="Dataset name to filter experiments by (e.g., 'ud_gum_dataset').")
    parser.add_argument("--output_dir", required=True, help="Directory to save the combined plots and markdown files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect experiments for all models
    all_experiments = {}  # probe_type -> model -> lambda -> dir
    for model in args.models:
        pattern = re.compile(rf"^{re.escape(args.dataset)}_{re.escape(model)}_(?P<probe_type>\w+)_lambda(?P<lambda_val>\d+\.?\d*)$")
        for dirname in os.listdir(args.steering_dir):
            match = pattern.match(dirname)
            if match:
                probe_type = match.group("probe_type")
                lambda_val = float(match.group("lambda_val"))
                all_experiments.setdefault(probe_type, {}).setdefault(model, {})[lambda_val] = os.path.join(args.steering_dir, dirname)

    if not all_experiments:
        print(f"No steering results found for models '{args.models}' and dataset '{args.dataset}' in '{args.steering_dir}'.")
        return

    # Per-model plots and summaries
    for probe_type, model_dirs in all_experiments.items():
        for model, results_dirs in model_dirs.items():
            print(f"\n--- Processing probe type: {probe_type} | model: {model} ---")
            plot_and_summarize_probe_type(probe_type, results_dirs, model, args.dataset, args.output_dir)
            print("-------------------------------------------\n")

    # Consolidated plots
    for probe_type, model_dirs in all_experiments.items():
        all_model_summaries = {}
        for model, results_dirs in model_dirs.items():
            all_summaries = []
            for lambda_val, results_dir in results_dirs.items():
                summary_file = os.path.join(results_dir, "steering_summary.csv")
                if not os.path.exists(summary_file):
                    continue
                df = pd.read_csv(summary_file)
                df["lambda"] = lambda_val
                all_summaries.append(df)
            if all_summaries:
                df_all = pd.concat(all_summaries, ignore_index=True)
                all_model_summaries[model] = df_all
        print(f"\n--- Processing consolidated plot for probe type: {probe_type} ---")
        plot_consolidated_probe_type(probe_type, all_model_summaries, args.dataset, args.output_dir)
        print("-------------------------------------------\n")

if __name__ == "__main__":
    main()
