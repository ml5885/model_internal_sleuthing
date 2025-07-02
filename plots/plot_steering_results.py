import os
import pandas as pd

def find_and_print_steering_results(base_dir="../output/steering"):
    """
    Finds steering_results.csv files, processes them, and prints markdown tables.
    """
    models = ["qwen2"]
    probe_types = ["reg", "mlp"]
    dataset = "ud_gum_dataset"
    task = "inflection"

    print("# Steering Experiment Results\n")

    for model in models:
        for probe_type in probe_types:
            probe_dir_name = f"{dataset}_{model}_{task}_{probe_type}"
            results_path = os.path.join(base_dir, probe_dir_name, "steering_results.csv")

            print(f"## Model: `{model}`, Probe: `{probe_type.upper()}`\n")

            if os.path.exists(results_path):
                try:
                    df = pd.read_csv(results_path)
                    
                    if df.empty:
                        print("`steering_results.csv` is empty. No data to plot.\n")
                        continue

                    # Process flip rate results
                    if 'flip_rate' in df.columns:
                        print("### Flip Rate Results\n")
                        flip_pivot = df.pivot_table(index="layer", columns="lambda", values="flip_rate")
                        best_lambda_flip = flip_pivot.idxmax(axis=1)
                        best_val_flip = flip_pivot.max(axis=1)
                        flip_pivot['Best Lambda'] = best_lambda_flip
                        flip_pivot['Best Flip Rate'] = best_val_flip
                        print(flip_pivot.to_markdown(floatfmt=".3f"))
                        print("\n")
                    
                    # Process probability delta results
                    if 'prob_delta' in df.columns:
                        print("### Probability Delta Results\n")
                        prob_pivot = df.pivot_table(index="layer", columns="lambda", values="prob_delta")
                        best_lambda_prob = prob_pivot.idxmax(axis=1)
                        best_val_prob = prob_pivot.max(axis=1)
                        prob_pivot['Best Lambda'] = best_lambda_prob
                        prob_pivot['Best Prob Delta'] = best_val_prob
                        print(prob_pivot.to_markdown(floatfmt=".3f"))
                        print("\n")
                    
                    # Legacy support for old 'accuracy' column
                    if 'accuracy' in df.columns and 'flip_rate' not in df.columns:
                        print("### Legacy Accuracy Results\n")
                        pivot_df = df.pivot_table(index="layer", columns="lambda", values="accuracy")
                        best_lambda = pivot_df.idxmax(axis=1)
                        best_val = pivot_df.max(axis=1)
                        pivot_df['Best Lambda'] = best_lambda
                        pivot_df['Best Accuracy'] = best_val
                        print(pivot_df.to_markdown(floatfmt=".3f"))
                        print("\n")
                        
                except Exception as e:
                    print(f"Could not process file {results_path}: {e}\n")
            else:
                print(f"Results file not found at `{results_path}`\n")

if __name__ == "__main__":
    find_and_print_steering_results()
