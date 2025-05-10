import os
import subprocess
import argparse
from src import config, utils

def run_activation_extraction(model_key, dataset, output_dir=None):
    utils.log_info(f"Starting activation extraction for {model_key} on dataset {dataset}...")
    dataset_file = os.path.join("data", f"{dataset}.csv")

    checks = []
    if output_dir:
        custom = os.path.join(output_dir, f"{model_key}_{dataset}_reps")
        checks.append((custom, custom + ".npz"))
    default = os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset}_reps")
    checks.append((default, default + ".npz"))
    external = os.path.join("/data/user_data/ml6/probing_outputs", f"{model_key}_{dataset}_reps")
    checks.append((external, external + ".npz"))

    for d, f in checks:
        if os.path.isfile(f):
            utils.log_info(f"Using existing activation file: {f}")
            print(f"Using existing activation file: {f}")
            return f
        if os.path.isdir(d) and any(fn.startswith("activations_part") for fn in os.listdir(d)):
            utils.log_info(f"Using existing activation shards: {d}")
            print(f"Using existing activation shards: {d}")
            return d

    utils.log_info("No existing activations found. Extracting new activations...")
    print("No existing activations found. Extracting new activations...")
    save_dir = checks[0][0]
    os.makedirs(save_dir, exist_ok=True)

    subprocess.run([
        "python", "-m", "src.activation_extraction",
        "--data", dataset_file,
        "--output-dir", save_dir,
        "--model", model_key
    ], check=True)

    return save_dir

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
    parser = argparse.ArgumentParser(description="Run experiment pipeline for probing tasks.")
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
    probe_type = args.probe_type
    pca_dim = args.pca_dim
    pca = pca_dim > 0

    # this is the root under which each task folder will live
    base_probe_dir = args.output_dir if args.output_dir else config.OUTPUT_DIR

    # build exact final paths for each task
    probe_output_dirs = {
        task: utils.get_probe_output_dir(
            dataset, model_key, task, probe_type,
            pca=pca, pca_dim=pca_dim,
            base_dir=base_probe_dir
        )
        for task in ["inflection", "lexeme"]
    }

    reps = run_activation_extraction(model_key, dataset, args.output_dir)

    experiments = []
    for task in ["inflection", "lexeme"]:
        experiments.append({
            "name": task,
            "args": [
                "--activations", reps,
                "--labels", os.path.join("data", f"{dataset}.csv"),
                "--task", task,
                "--lambda_reg", str(args.lambda_reg),
                "--exp_label", f"{model_key}_{task}",
                "--dataset", dataset,
                "--probe_type", probe_type,
                "--pca_dim", str(pca_dim),
                "--output_dir", probe_output_dirs[task]
            ]
        })

    # filter by --experiment if given
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"Unknown experiment name: {args.experiment}")

    for exp in experiments:
        utils.log_info(f"Running experiment: {exp['name']}")
        subprocess.run(["python", "-m", "src.train"] + exp["args"], check=True)

    if not args.experiment and not args.no_analysis:
        run_analysis(model_key, dataset, args.output_dir)

    utils.log_info("All experiments and analysis completed.")

if __name__ == "__main__":
    main()
