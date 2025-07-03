import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src import utils

def plot_results(results_df, output_dir):
    """Plots steering experiment results and generates a markdown table."""
    if results_df.empty:
        utils.log_info("Results dataframe is empty, skipping plotting.")
        return

    # If loading the summary, the columns are already aggregated.
    # If loading detailed results, we need to aggregate first.
    if 'mean_prob_change' not in results_df.columns:
        summary_df = results_df.groupby('layer').agg(
            mean_prob_change=('prob_change', 'mean'),
            flip_rate=('prediction_flip', 'mean')
        ).reset_index()
    else:
        summary_df = results_df
    
    summary_df = summary_df.sort_values('layer')
    
    # Ensure layers are treated as integers for plotting
    summary_df['layer'] = summary_df['layer'].astype(int)
    layers = summary_df['layer']
    
    sns.set_theme(style="whitegrid")

    # Plot 1: Mean Probability Change
    plt.figure(figsize=(12, 6))
    ax1 = sns.lineplot(data=summary_df, x='layer', y='mean_prob_change', marker='o', legend=False)
    plt.title('Mean Probability Change of Steered Inflection', fontsize=16)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Mean Probability Change', fontsize=12)
    ax1.set_xticks(layers)
    ax1.set_xticklabels(layers, rotation=45, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'steering_prob_change.png')
    plt.savefig(plot_path)
    plt.close()
    utils.log_info(f"Saved probability change plot to {plot_path}")

    # Plot 2: Prediction Flip Rate
    plt.figure(figsize=(12, 6))
    ax2 = sns.lineplot(data=summary_df, x='layer', y='flip_rate', marker='o', color='green', legend=False)
    plt.title('Prediction Flip Rate to Steered Inflection', fontsize=16)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Flip Rate', fontsize=12)
    plt.ylim(0, max(0.1, summary_df['flip_rate'].max() * 1.15)) # Adjust y-axis for better visibility
    ax2.set_xticks(layers)
    ax2.set_xticklabels(layers, rotation=45, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'steering_flip_rate.png')
    plt.savefig(plot_path)
    plt.close()
    utils.log_info(f"Saved flip rate plot to {plot_path}")

    # Generate and save a Markdown Table
    markdown_table = summary_df.to_markdown(index=False, floatfmt=".4f")
    table_path = os.path.join(output_dir, 'steering_summary.md')
    with open(table_path, 'w') as f:
        f.write("# Steering Experiment Summary\n\n")
        f.write(markdown_table)
    utils.log_info(f"Saved markdown summary to {table_path}")
    
    # Print the table to console for immediate feedback
    print("\n--- Steering Results Summary ---")
    print(markdown_table)
    print("------------------------------\n")


def plot_multi_lambda(results_dirs, output_dir):
    """Plot results for multiple lambdas on the same plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_summaries = []
    for lambda_val, results_dir in results_dirs.items():
        summary_file = os.path.join(results_dir, "steering_summary.csv")
        if not os.path.exists(summary_file):
            continue
        df = pd.read_csv(summary_file)
        df['lambda'] = lambda_val
        all_summaries.append(df)
    if not all_summaries:
        print("No summary files found for any lambda.")
        return
    df_all = pd.concat(all_summaries, ignore_index=True)
    df_all['layer'] = df_all['layer'].astype(int)
    df_all = df_all.sort_values(['lambda', 'layer'])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    for lambda_val in sorted(results_dirs.keys(), key=float):
        sub = df_all[df_all['lambda'] == lambda_val]
        plt.plot(sub['layer'], sub['mean_prob_change'], marker='o', label=f"λ={lambda_val}")
    plt.title('Mean Probability Change of Steered Inflection (by λ)', fontsize=16)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Mean Probability Change', fontsize=12)
    plt.legend(title="Lambda")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'steering_prob_change_multi_lambda.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved multi-lambda probability change plot to {plot_path}")

    plt.figure(figsize=(12, 6))
    for lambda_val in sorted(results_dirs.keys(), key=float):
        sub = df_all[df_all['lambda'] == lambda_val]
        plt.plot(sub['layer'], sub['flip_rate'], marker='o', label=f"λ={lambda_val}")
    plt.title('Prediction Flip Rate to Steered Inflection (by λ)', fontsize=16)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Flip Rate', fontsize=12)
    plt.legend(title="Lambda")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'steering_flip_rate_multi_lambda.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved multi-lambda flip rate plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot results from a steering experiment.")
    parser.add_argument("--results_dir", required=True, help="Directory containing steering_results.csv or steering_summary.csv.")
    args = parser.parse_args()

    # Check for multi-lambda subdirectories
    import re
    parent_dir = os.path.dirname(args.results_dir.rstrip('/'))
    base_name = os.path.basename(args.results_dir.rstrip('/'))
    
    # This regex will find the lambda value and the prefix before it.
    match = re.search(r'^(.*_lambda)\d+(\.\d+)?$', base_name)

    if match:
        # If results_dir is a lambda-specific directory, check for siblings
        prefix = match.group(1) # e.g., "ud_gum_dataset_qwen2_reg_lambda"
        siblings = [d for d in os.listdir(parent_dir) if d.startswith(prefix)]
        results_dirs = {}
        for sib in siblings:
            # Extract lambda value from directory name
            lambda_str = sib[len(prefix):]
            try:
                lambda_val = float(lambda_str)
                results_dirs[lambda_val] = os.path.join(parent_dir, sib)
            except ValueError:
                continue # Ignore directories that don't end in a valid number
        
        if len(results_dirs) > 1:
            # The output directory for multi-lambda plots should be the parent directory.
            plot_multi_lambda(results_dirs, parent_dir)

    summary_file = os.path.join(args.results_dir, "steering_summary.csv")
    results_file = os.path.join(args.results_dir, "steering_results.csv")
    if os.path.exists(summary_file):
        utils.log_info(f"Loading summary file: {summary_file}")
        results_df = pd.read_csv(summary_file)
    elif os.path.exists(results_file):
        utils.log_info(f"Loading detailed results file: {results_file}")
        results_df = pd.read_csv(results_file)
    else:
        print(f"Error: Neither steering_summary.csv nor steering_results.csv found in {args.results_dir}")
        return

    plot_results(results_df, args.results_dir)

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
