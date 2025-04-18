import os
import subprocess
import argparse
from src import config, utils

def run_activation_extraction(model_key, dataset):
    utils.log_info(f"Starting activation extraction for {model_key} on dataset {dataset}...")
    dataset_file = os.path.join("data", f"{dataset}.csv")
    output_dir = os.path.join("output", f"{model_key}_{dataset}_reps")
    if os.path.isdir(output_dir) and any(f.startswith("activations_part") for f in os.listdir(output_dir)):
        utils.log_info(f"Using existing activation shards: {output_dir}")
        print(f"Using existing activation shards: {output_dir}")
    else:
        utils.log_info("Extracting new activations...")
        print("Extracting new activations...")
        subprocess.run([
            "python", "-m", "src.activation_extraction",
            "--data", dataset_file,
            "--output-dir", output_dir,
            "--model", model_key
        ], check=True)
    return output_dir

def run_probe(exp_args):
    cmd = ["python", "-m", "src.train"] + exp_args
    subprocess.run(cmd, check=True)

def run_analysis(model_key, dataset):
    utils.log_info("Running analysis on activations...")
    activations_dir = os.path.join("output", f"{model_key}_{dataset}_reps")
    subprocess.run([
        "python", "-m", "src.analysis",
        "--activations-dir", activations_dir,
        "--labels", os.path.join("data", f"{dataset}.csv"),
        "--model", model_key,
        "--dataset", dataset,
    ], check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Run experiment pipeline for probing tasks."
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model key (e.g. 'gpt2').")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset label (e.g. 'ud_gum_dataset').")
    parser.add_argument("--experiment", type=str,
                        choices=["multiclass_inflection", "lexeme"],
                        help="Specific experiment to run.")
    args = parser.parse_args()

    model_key = args.model
    dataset = args.dataset
    reps = run_activation_extraction(model_key, dataset)

    experiments = [
        {"name": "multiclass_inflection", "args": [
            "--activations", reps,
            "--labels", os.path.join("data", f"{dataset}.csv"),
            "--task", "multiclass_inflection",
            "--lambda_reg", "1e-3",
            "--exp_label", f"{model_key}_multiclass_inflection",
            "--dataset", dataset
        ]},
        {"name": "lexeme", "args": [
            "--activations", reps,
            "--labels", os.path.join("data", f"{dataset}.csv"),
            "--task", "lexeme",
            "--lambda_reg", "1e-3",
            "--exp_label", f"{model_key}_lexeme",
            "--dataset", dataset
        ]}
    ]

    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"Unknown experiment name: {args.experiment}")

    for exp in experiments:
        utils.log_info(f"Running experiment: {exp['name']}")
        print(f"Running experiment: {exp['name']}")
        run_probe(exp["args"])

    # run_analysis(model_key, dataset)
    utils.log_info("All experiments and analysis completed.")

if __name__ == "__main__":
    main()
