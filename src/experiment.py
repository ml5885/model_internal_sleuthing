import os
import subprocess
import argparse
from src import config, utils

def run_activation_extraction(model_key, dataset):
    utils.log_info(f"Starting activation extraction for {model_key} on dataset {dataset}...")
    dataset_file = os.path.join("data", f"{dataset}.csv")
    output_file = os.path.join("output", f"{model_key}_{dataset}_reps.npz")
    
    if os.path.exists(output_file):
        utils.log_info(f"Using existing activations file: {output_file}")
        print(f"Using existing activations file: {output_file}")
    else:
        utils.log_info("Extracting new activations...")
        print("Extracting new activations...")
        subprocess.run([
            "python", "-m", "src.activation_extraction",
            "--data", dataset_file,
            "--output", output_file,
            "--model", model_key
        ], check=True)
    
    return output_file

def run_probe(exp_args):
    cmd = ["python", "-m", "src.train"] + exp_args
    subprocess.run(cmd, check=True)

def run_analysis(model_key, dataset):
    utils.log_info("Running analysis on activations...")
    dataset_file = os.path.join("output", f"{model_key}_{dataset}_reps.npz")
    subprocess.run([
        "python", "-m", "src.analysis",
        "--activations", dataset_file,
        "--labels", os.path.join("data", f"{dataset}.csv"),
        "--model", model_key,
        "--dataset", dataset,
    ], check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Run experiment pipeline for one-vs-rest inflection and lexeme probing tasks with control tasks."
    )
    parser.add_argument("--model", type=str, default="gpt2", help="Model key to use (e.g. 'gpt2').")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset label (e.g., controlled, wikitext, combined).")
    parser.add_argument("--experiment", type=str,
                        choices=["multiclass_inflection_dense", "lexeme_dense"],
                        help="Run only the specified experiment.")
    args = parser.parse_args()
    
    model_key = args.model
    dataset = args.dataset
    
    reps_file = run_activation_extraction(model_key, dataset)
    
    experiments = [
        {
            "name": "multiclass_inflection_dense",
            "args": [
                "--activations", reps_file,
                "--labels", os.path.join("data", f"{dataset}.csv"),
                "--task", "multiclass_inflection",
                "--lambda_reg", "1e-3",
                "--control_inflection",
                "--exp_label", f"{model_key}_multiclass_inflection_dense",
                "--dataset", dataset
            ]
        },
        {
            "name": "lexeme_dense",
            "args": [
                "--activations", reps_file,
                "--labels", os.path.join("data", f"{dataset}.csv"),
                "--task", "lexeme",
                "--lambda_reg", "1e-3",
                "--control_lexeme",
                "--exp_label", f"{model_key}_lexeme_dense",
                "--dataset", dataset
            ]
        }
    ]
    
    if args.experiment:
        experiments = [exp for exp in experiments if exp["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"Unknown experiment name: {args.experiment}")
    
    for exp in experiments:
        utils.log_info(f"Running experiment: {exp['name']}")
        print(f"Running experiment: {exp['name']}")
        run_probe(exp["args"])
    
    run_analysis(model_key, dataset)
    utils.log_info("All experiments and analysis completed.")

if __name__ == "__main__":
    main()
