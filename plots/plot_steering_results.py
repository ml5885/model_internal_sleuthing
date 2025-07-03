import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_and_summarize_probe_type(probe_type, results_dirs, model, dataset, output_dir):
    """
    Loads all summary files for a given probe type, generates multi-lambda plots,
    and creates a markdown summary table.
    """
    all_summaries = []
    for lambda_val, results_dir in results_dirs.items():
        summary_file = os.path.join(results_dir, "steering_summary.csv")
        if not os.path.exists(summary_file):
            print(f"Warning: steering_summary.csv not found in {results_dir}, skipping.")
            continue
        df = pd.read_csv(summary_file)
        df['lambda'] = lambda_val
        all_summaries.append(df)

    if not all_summaries:
        print(f"No summary files found for {probe_type}. Skipping plots.")
        return

    df_all = pd.concat(all_summaries, ignore_index=True)
    df_all['layer'] = df_all['layer'].astype(int)
    df_all = df_all.sort_values(['lambda', 'layer'])

    sns.set_theme(style="whitegrid")
    
    # Plot 1: Mean Probability Change
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_all, x='layer', y='mean_prob_change', hue='lambda', marker='o', palette='viridis')
    plt.title(f'Mean Probability Change ({model} on {dataset} - {probe_type.upper()})', fontsize=16)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Mean Probability Change', fontsize=12)
    plt.legend(title="Lambda (λ)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{dataset}_{model}_{probe_type}_prob_change_multi_lambda.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved multi-lambda probability change plot to {plot_path}")

    # Plot 2: Prediction Flip Rate
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_all, x='layer', y='flip_rate', hue='lambda', marker='o', palette='viridis')
    plt.title(f'Prediction Flip Rate ({model} on {dataset} - {probe_type.upper()})', fontsize=16)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Flip Rate', fontsize=12)
    plt.legend(title="Lambda (λ)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{dataset}_{model}_{probe_type}_flip_rate_multi_lambda.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved multi-lambda flip rate plot to {plot_path}")

    # Generate and save a Markdown Table
    markdown_table = df_all.to_markdown(index=False, floatfmt=".4f")
    table_path = os.path.join(output_dir, f'{dataset}_{model}_{probe_type}_steering_summary.md')
    with open(table_path, 'w') as f:
        f.write(f"# Steering Summary: {model} on {dataset} ({probe_type.upper()})\n\n")
        f.write(markdown_table)
    print(f"Saved markdown summary to {table_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot steering experiment results for a given model and dataset.")
    parser.add_argument("--steering_dir", required=True, help="Base directory containing all steering experiment subdirectories.")
    parser.add_argument("--model", required=True, help="Model name to filter experiments by (e.g., 'qwen2').")
    parser.add_argument("--dataset", required=True, help="Dataset name to filter experiments by (e.g., 'ud_gum_dataset').")
    parser.add_argument("--output_dir", required=True, help="Directory to save the combined plots and markdown files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Regex to parse directory names like: {dataset}_{model}_{probe_type}_lambda{lambda_val}
    # Example: ud_gum_dataset_qwen2_reg_lambda1.0
    pattern = re.compile(rf"^{re.escape(args.dataset)}_{re.escape(args.model)}_(?P<probe_type>\w+)_lambda(?P<lambda_val>\d+\.?\d*)$")

    experiments = {}
    for dirname in os.listdir(args.steering_dir):
        match = pattern.match(dirname)
        if match:
            probe_type = match.group('probe_type')
            lambda_val = float(match.group('lambda_val'))
            
            if probe_type not in experiments:
                experiments[probe_type] = {}
            
            experiments[probe_type][lambda_val] = os.path.join(args.steering_dir, dirname)

    if not experiments:
        print(f"No steering results found for model '{args.model}' and dataset '{args.dataset}' in '{args.steering_dir}'.")
        return

    for probe_type, results_dirs in experiments.items():
        print(f"\n--- Processing probe type: {probe_type} ---")
        plot_and_summarize_probe_type(probe_type, results_dirs, args.model, args.dataset, args.output_dir)
        print("-------------------------------------------\n")

if __name__ == "__main__":
    main()
