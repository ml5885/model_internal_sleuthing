import os
import subprocess
import argparse
from src import config, utils

def run_activation_extraction(model_key, dataset, output_dir=None):
    utils.log_info(f"Starting activation extraction for {model_key} on dataset {dataset}...")
    dataset_file = os.path.join("data", f"{dataset}.csv")
    
    base_output_dir = output_dir if output_dir else config.OUTPUT_DIR
    output_dir = os.path.join(base_output_dir, f"{model_key}_{dataset}_reps")
    npz_file = output_dir + ".npz"

    if os.path.isfile(npz_file):
        utils.log_info(f"Using existing activation file: {npz_file}")
        print(f"Using existing activation file: {npz_file}")
        return npz_file
    elif os.path.isdir(output_dir) and any(f.startswith("activations_part") for f in os.listdir(output_dir)):
        utils.log_info(f"Using existing activation shards: {output_dir}")
        print(f"Using existing activation shards: {output_dir}")
        return output_dir
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

def run_analysis(model_key, dataset, output_dir=None):
    utils.log_info("Running analysis on activations...")
    
    base_output_dir = output_dir if output_dir else config.OUTPUT_DIR
    activations_dir = os.path.join(base_output_dir, f"{model_key}_{dataset}_reps")
    
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
                        help="Dataset label (e.g._gum_dataset').")
    parser.add_argument("--experiment", type=str,
                        choices=["inflection", "lexeme"],
                        help="Specific experiment to run.")
    parser.add_argument("--lambda_reg", type=float, default=1e-3,
                        help="Regularization parameter for probes.")
    parser.add_argument("--probe_type", type=str, default="reg",
                        choices=["reg", "mlp", "nn"],
                        help="Type of probe to use (regression or neural network).")
    parser.add_argument("--pca_dim", type=int, default=0,
                        help="Dimensionality for PCA reduction.")
    parser.add_argument("--no_analysis", action="store_true",
                        help="Skip analysis after running experiments.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for results and activations")
    args = parser.parse_args()

    model_key = args.model
    dataset = args.dataset
    output_dir = args.output_dir
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        original_output_dir = config.OUTPUT_DIR
        config.OUTPUT_DIR = output_dir
    
    reps = run_activation_extraction(model_key, dataset, output_dir)

    experiments = [
        {"name": "inflection", "args": [
            "--activations", reps,
            "--labels", os.path.join("data", f"{dataset}.csv"),
            "--task", "inflection",
            "--lambda_reg", str(args.lambda_reg),
            "--exp_label", f"{model_key}_inflection",
            "--dataset", dataset,
            "--probe_type", args.probe_type,
            "--pca_dim", str(args.pca_dim)
        ]},
        {"name": "lexeme", "args": [
            "--activations", reps,
            "--labels", os.path.join("data", f"{dataset}.csv"),
            "--task", "lexeme",
            "--lambda_reg", str(args.lambda_reg),
            "--exp_label", f"{model_key}_lexeme",
            "--dataset", dataset,
            "--probe_type", args.probe_type,
            "--pca_dim", str(args.pca_dim)
        ]}
    ]

    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"Unknown experiment name: {args.experiment}")

    if output_dir:
        for exp in experiments:
            exp["args"].extend(["--output_dir", output_dir])

    for exp in experiments:
        utils.log_info(f"Running experiment: {exp['name']}")
        print(f"Running experiment: {exp['name']}")
        run_probe(exp["args"])

    if not args.experiment and not args.no_analysis:
        run_analysis(model_key, dataset, output_dir)
        
    utils.log_info("All experiments and analysis completed.")

if __name__ == "__main__":
    main()